package bench

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"
)

// metricStats is the reported distribution of one latency metric, all values in milliseconds.
type metricStats struct {
	Mean        float64            `json:"mean"`
	Median      float64            `json:"median"`
	Std         float64            `json:"std"`
	Percentiles map[string]float64 `json:"percentiles"`
}

// Report is the full benchmark result — printed as a table and (optionally) written as JSON for
// tracking across runs. Field names mirror vLLM's serving benchmark where they correspond.
type Report struct {
	Model          string  `json:"model"`
	NumPrompts     int     `json:"num_prompts"`
	InputLen       int     `json:"input_len"`
	OutputLen      int     `json:"output_len"`
	RequestRate    string  `json:"request_rate"`
	Burstiness     float64 `json:"burstiness"`
	MaxConcurrency int     `json:"max_concurrency"`

	Completed            int      `json:"completed"`
	Failed               int      `json:"failed"`
	DurationS            float64  `json:"duration_s"`
	TotalInputTokens     int      `json:"total_input_tokens"`
	TotalOutputTokens    int      `json:"total_output_tokens"`
	RequestThroughput    float64  `json:"request_throughput"`
	OutputThroughput     float64  `json:"output_token_throughput"`
	TotalTokenThroughput float64  `json:"total_token_throughput"`
	RequestGoodput       *float64 `json:"request_goodput,omitempty"`

	TTFT metricStats `json:"ttft_ms"`
	TPOT metricStats `json:"tpot_ms"`
	ITL  metricStats `json:"itl_ms"`
	E2EL metricStats `json:"e2el_ms"`
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

func pkey(p float64) string { return fmt.Sprintf("p%g", p) }

// percentile uses linear interpolation between closest ranks (numpy's default), on sorted input.
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	rank := p / 100 * float64(len(sorted)-1)
	lo := int(math.Floor(rank))
	hi := int(math.Ceil(rank))
	if lo == hi {
		return sorted[lo]
	}
	frac := rank - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

func summarize(vals, pctls []float64) metricStats {
	s := metricStats{Percentiles: map[string]float64{}}
	for _, p := range pctls {
		s.Percentiles[pkey(p)] = 0
	}
	if len(vals) == 0 {
		return s
	}
	sorted := append([]float64(nil), vals...)
	sort.Float64s(sorted)
	var sum float64
	for _, v := range sorted {
		sum += v
	}
	s.Mean = sum / float64(len(sorted))
	s.Median = percentile(sorted, 50)
	var sq float64
	for _, v := range sorted {
		sq += (v - s.Mean) * (v - s.Mean)
	}
	s.Std = math.Sqrt(sq / float64(len(sorted)))
	for _, p := range pctls {
		s.Percentiles[pkey(p)] = percentile(sorted, p)
	}
	return s
}

// meetsSLO reports whether one request satisfies every configured goodput SLO (threshold >= value),
// following the DistServe goodput definition. A tpot SLO can only be met by requests with >1 output
// token (the rest have no defined TPOT and so fail it).
func meetsSLO(slos map[string]float64, ttft, tpot, e2el float64, hasTPOT bool) bool {
	for k, thr := range slos {
		switch k {
		case "ttft":
			if ttft > thr {
				return false
			}
		case "tpot":
			if !hasTPOT || tpot > thr {
				return false
			}
		case "e2el":
			if e2el > thr {
				return false
			}
		}
	}
	return true
}

func aggregate(cfg Config, records []record, dur time.Duration) Report {
	var ttfts, tpots, e2els, itls []float64
	var totalIn, totalOut, completed, failed, good int
	useGoodput := len(cfg.Goodput) > 0

	for _, r := range records {
		if r.err != nil {
			failed++
			continue
		}
		completed++
		totalIn += r.promptTokens
		totalOut += r.outputTokens

		ttftMs, e2elMs := ms(r.ttft), ms(r.e2el)
		ttfts = append(ttfts, ttftMs)
		e2els = append(e2els, e2elMs)
		var tpotMs float64
		hasTPOT := r.outputTokens > 1
		if hasTPOT {
			tpotMs = (e2elMs - ttftMs) / float64(r.outputTokens-1)
			tpots = append(tpots, tpotMs)
		}
		for _, it := range r.itls {
			itls = append(itls, ms(it))
		}
		if useGoodput && meetsSLO(cfg.Goodput, ttftMs, tpotMs, e2elMs, hasTPOT) {
			good++
		}
	}

	durS := dur.Seconds()
	rep := Report{
		Model: cfg.Model, NumPrompts: cfg.NumPrompts, InputLen: cfg.InputLen, OutputLen: cfg.OutputLen,
		RequestRate: rateString(cfg.RequestRate), Burstiness: cfg.Burstiness, MaxConcurrency: cfg.MaxConc,
		Completed: completed, Failed: failed, DurationS: durS,
		TotalInputTokens: totalIn, TotalOutputTokens: totalOut,
		TTFT: summarize(ttfts, cfg.Percentiles), TPOT: summarize(tpots, cfg.Percentiles),
		ITL: summarize(itls, cfg.Percentiles), E2EL: summarize(e2els, cfg.Percentiles),
	}
	if durS > 0 {
		rep.RequestThroughput = float64(completed) / durS
		rep.OutputThroughput = float64(totalOut) / durS
		rep.TotalTokenThroughput = float64(totalIn+totalOut) / durS
		if useGoodput {
			g := float64(good) / durS
			rep.RequestGoodput = &g
		}
	}
	return rep
}

func rateString(rate float64) string {
	if math.IsInf(rate, 1) {
		return "inf"
	}
	return fmt.Sprintf("%g", rate)
}

// printReport writes the human-readable summary table.
func printReport(rep Report, cfg Config) {
	line := func(label, val string) { fmt.Printf("%-42s %s\n", label, val) }
	f2 := func(v float64) string { return fmt.Sprintf("%.2f", v) }

	fmt.Println()
	fmt.Println(banner(" FlashQwen Serving Benchmark ", '='))
	line("Successful requests:", fmt.Sprintf("%d", rep.Completed))
	if rep.Failed > 0 {
		line("Failed requests:", fmt.Sprintf("%d", rep.Failed))
	}
	line("Benchmark duration (s):", f2(rep.DurationS))
	if !math.IsInf(cfg.RequestRate, 1) {
		line("Request rate configured (req/s):", f2(cfg.RequestRate))
	}
	if cfg.MaxConc > 0 {
		line("Maximum request concurrency:", fmt.Sprintf("%d", cfg.MaxConc))
	}
	line("Total input tokens:", fmt.Sprintf("%d", rep.TotalInputTokens))
	line("Total generated tokens:", fmt.Sprintf("%d", rep.TotalOutputTokens))
	line("Request throughput (req/s):", f2(rep.RequestThroughput))
	if rep.RequestGoodput != nil {
		line("Request goodput (req/s):", f2(*rep.RequestGoodput))
	}
	line("Output token throughput (tok/s):", f2(rep.OutputThroughput))
	line("Total token throughput (tok/s):", f2(rep.TotalTokenThroughput))

	section := func(title, abbr string, m metricStats) {
		fmt.Println(banner(" "+title+" ", '-'))
		line("Mean "+abbr+" (ms):", f2(m.Mean))
		line("Median "+abbr+" (ms):", f2(m.Median))
		for _, p := range cfg.Percentiles {
			line(fmt.Sprintf("P%g %s (ms):", p, abbr), f2(m.Percentiles[pkey(p)]))
		}
	}
	section("Time to First Token", "TTFT", rep.TTFT)
	section("Time per Output Token (excl. 1st)", "TPOT", rep.TPOT)
	section("Inter-token Latency", "ITL", rep.ITL)
	section("End-to-end Latency", "E2EL", rep.E2EL)
	fmt.Println(strings.Repeat("=", bannerWidth))
}

const bannerWidth = 62

// banner centres s in a bannerWidth-wide line padded with fill.
func banner(s string, fill byte) string {
	if len(s) >= bannerWidth {
		return s
	}
	pad := bannerWidth - len(s)
	left := pad / 2
	return strings.Repeat(string(fill), left) + s + strings.Repeat(string(fill), pad-left)
}

func writeJSON(path string, rep Report) error {
	b, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}
