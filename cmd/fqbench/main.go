// fqbench — a self-contained serving benchmark client for the FlashQwen OpenAI endpoint, modelled on
// `vllm bench serve` (random dataset). It reproduces vLLM's request scheduling and metric math so the
// numbers are directly comparable to our vLLM baselines, with one deliberate improvement: prompts are
// sent as raw token-id arrays (the /v1/completions `prompt` array form), so input length is exact (no
// decode/re-encode drift) and a shared prefix is byte-for-byte identical across requests — which makes
// it a clean probe for prefix caching.
//
// Scheduling (matches vLLM):
//   --request-rate inf  : all requests are admissible immediately; --concurrency gates how many run at
//                         once (a worker pool / semaphore). This is our standard mode.
//   --request-rate R    : inter-arrival delays ~ Gamma(shape=burstiness, scale=1/(R*burstiness)),
//                         accumulated then normalised so the last arrival lands at N/R seconds; each
//                         launched request still passes through the --concurrency semaphore.
//
// Metrics (matches vLLM calculate_metrics): output throughput, total/req throughput, and
// mean/median/std/pNN of TTFT, TPOT (=(latency-ttft)/(out-1), only out>1), ITL, E2EL. Percentiles use
// numpy's linear interpolation; std is population std. Output token counts come from the stream's
// usage.completion_tokens (we force it via stream_options.include_usage).
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type config struct {
	base        string
	model       string
	endpoint    string
	inputLen    int
	outputLen   int
	prefixLen   int
	numPrompts  int
	concurrency int
	requestRate float64 // requests/s; +Inf = no pacing (gate by concurrency only)
	burstiness  float64
	rangeRatio  float64
	ignoreEOS   bool
	temperature float64
	seed        int64
	vocab       int // exclusive upper bound on sampled token ids (avoid special tokens)
	warmups     int
	percentiles []float64
	timeoutSec  int
}

// per-request measurement (mirrors vLLM RequestFuncOutput)
type result struct {
	success     bool
	ttft        float64 // s
	itl         []float64
	latency     float64 // s (start -> last token)
	outputToks  int
	promptLen   int
	startOffset float64 // s, relative to benchmark start
	err         string
}

func main() {
	cfg := parseFlags()
	rng := rand.New(rand.NewSource(cfg.seed))

	// Build the shared prefix once (identical token ids across every request -> a real shared prefix).
	prefix := make([]int, cfg.prefixLen)
	for i := range prefix {
		prefix[i] = rng.Intn(cfg.vocab)
	}
	// Per-request lengths + vocab offsets, sampled like vLLM's get_sampling_params.
	inLow, inHigh := rangeBounds(cfg.inputLen, cfg.rangeRatio)
	outLow, outHigh := rangeBounds(cfg.outputLen, cfg.rangeRatio)
	if outLow < 1 {
		outLow = 1
	}
	prompts := make([][]int, cfg.numPrompts)
	outLens := make([]int, cfg.numPrompts)
	for i := 0; i < cfg.numPrompts; i++ {
		inLen := inLow + rng.Intn(inHigh-inLow+1)
		outLens[i] = outLow + rng.Intn(outHigh-outLow+1)
		offset := rng.Intn(cfg.vocab)
		inner := make([]int, inLen) // vLLM: allowed[(offset+index+k) % vocab]
		for k := range inner {
			inner[k] = (offset + i + k) % cfg.vocab
		}
		prompts[i] = append(append([]int{}, prefix...), inner...)
	}

	client := newClient(cfg)
	apiURL := strings.TrimRight(cfg.base, "/") + "/v1/" + cfg.endpoint

	// Warmup (priming the shared prefix etc.) — not timed, same concurrency gate.
	if cfg.warmups > 0 {
		fmt.Printf("Warming up with %d requests...\n", cfg.warmups)
		runPool(client, apiURL, cfg, prompts[:min(cfg.warmups, cfg.numPrompts)],
			outLens, math.Inf(1), 1.0, rng)
		fmt.Println("Warmup done.")
	}

	fmt.Printf("Benchmarking: %d prompts, concurrency %d, in=%d out=%d prefix=%d, rate=%s, endpoint=%s\n",
		cfg.numPrompts, cfg.concurrency, cfg.inputLen, cfg.outputLen, cfg.prefixLen, rateStr(cfg.requestRate), cfg.endpoint)

	start := time.Now()
	results := runPool(client, apiURL, cfg, prompts, outLens, cfg.requestRate, cfg.burstiness, rng)
	dur := time.Since(start).Seconds()

	printReport(cfg, results, dur)
}

// runPool launches one goroutine per prompt; each waits for its scheduled arrival (0 when rate=inf),
// then passes through a `concurrency`-slot semaphore before issuing its streaming request. This mirrors
// vLLM's get_request (arrival pacing) + limited_request_func (semaphore) exactly.
func runPool(client *http.Client, apiURL string, cfg config, prompts [][]int, outLens []int,
	rate, burstiness float64, rng *rand.Rand) []result {

	n := len(prompts)
	delays := arrivalDelays(n, rate, burstiness, rng) // cumulative seconds from start
	sem := make(chan struct{}, cfg.concurrency)
	results := make([]result, n)
	var wg sync.WaitGroup
	base := time.Now()

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if d := delays[i] - time.Since(base).Seconds(); d > 0 {
				time.Sleep(time.Duration(d * float64(time.Second)))
			}
			sem <- struct{}{}
			defer func() { <-sem }()
			startOff := time.Since(base).Seconds()
			r := doRequest(client, apiURL, cfg, prompts[i], outLens[i])
			r.startOffset = startOff
			results[i] = r
		}(i)
	}
	wg.Wait()
	return results
}

// doRequest issues one streaming completion and records vLLM-equivalent timings.
func doRequest(client *http.Client, apiURL string, cfg config, prompt []int, outLen int) result {
	chat := cfg.endpoint == "chat/completions"
	payload := map[string]any{
		"model":          cfg.model,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
		"temperature":    cfg.temperature,
		"ignore_eos":     cfg.ignoreEOS,
	}
	if chat {
		// chat path needs text content; we send the token ids as a space-joined string is NOT valid,
		// so chat mode falls back to a trivial text prompt (use completions for exact-token control).
		payload["messages"] = []map[string]any{{"role": "user", "content": idsToText(prompt)}}
		payload["max_completion_tokens"] = outLen
	} else {
		payload["prompt"] = prompt
		payload["max_tokens"] = outLen
		payload["repetition_penalty"] = 1.0
	}
	body, _ := json.Marshal(payload)

	res := result{promptLen: len(prompt)}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(cfg.timeoutSec)*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	st := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		res.err = err.Error()
		return res
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		res.err = "HTTP " + resp.Status
		return res
	}

	firstSeen := false
	mostRecent := st
	reader := bufio.NewReaderSize(resp.Body, 1<<20)
	for {
		line, err := reader.ReadString('\n')
		if len(line) > 0 {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, ":") || line == "" {
				// SSE comment / blank separator
			} else if data, ok := strings.CutPrefix(line, "data:"); ok {
				data = strings.TrimSpace(data)
				if data == "[DONE]" {
					break
				}
				var ev streamEvent
				if json.Unmarshal([]byte(data), &ev) == nil {
					if ev.Error != nil {
						res.err = fmt.Sprintf("%v", ev.Error)
						return res
					}
					if len(ev.Choices) > 0 {
						// presence of a choice == a token event (matches vLLM; text may be empty,
						// e.g. the final finish_reason chunk — counted identically by both clients)
						now := time.Now()
						if !firstSeen {
							firstSeen = true
							res.ttft = time.Since(st).Seconds()
						} else {
							res.itl = append(res.itl, now.Sub(mostRecent).Seconds())
						}
						mostRecent = now // completions: advanced only on token chunks, not usage
					} else if ev.Usage != nil {
						res.outputToks = ev.Usage.CompletionTokens
						if ev.Usage.PromptTokens > 0 {
							res.promptLen = ev.Usage.PromptTokens
						}
					}
				}
			}
		}
		if err != nil {
			break
		}
	}
	res.latency = mostRecent.Sub(st).Seconds()
	res.success = firstSeen
	if !firstSeen && res.err == "" {
		res.err = "no token chunk received"
	}
	return res
}

type streamEvent struct {
	Choices []struct {
		Text  string `json:"text"`
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *struct {
		CompletionTokens int `json:"completion_tokens"`
		PromptTokens     int `json:"prompt_tokens"`
	} `json:"usage"`
	Error any `json:"error"`
}

// ---- metrics (mirrors vLLM calculate_metrics) ---------------------------------------------------

func printReport(cfg config, results []result, dur float64) {
	var ttfts, tpots, itls, e2els []float64
	totalIn, totalOut, completed, failed := 0, 0, 0, 0
	for _, r := range results {
		if !r.success {
			failed++
			if r.err != "" && failed <= 3 {
				fmt.Printf("  req error: %s\n", r.err)
			}
			continue
		}
		completed++
		outLen := r.outputToks
		if outLen == 0 {
			outLen = 1
		}
		totalIn += r.promptLen
		totalOut += outLen
		ttfts = append(ttfts, r.ttft)
		if outLen > 1 {
			tpots = append(tpots, (r.latency-r.ttft)/float64(outLen-1))
		}
		itls = append(itls, r.itl...)
		e2els = append(e2els, r.latency)
	}

	line := func(label string, val any) {
		switch v := val.(type) {
		case float64:
			fmt.Printf("%-40s %-10.2f\n", label, v)
		default:
			fmt.Printf("%-40s %-10v\n", label, v)
		}
	}
	fmt.Println(center(" Serving Benchmark Result ", 50, '='))
	line("Successful requests:", completed)
	line("Failed requests:", failed)
	line("Maximum request concurrency:", cfg.concurrency)
	if !math.IsInf(cfg.requestRate, 1) {
		line("Request rate configured (RPS):", cfg.requestRate)
	}
	line("Benchmark duration (s):", dur)
	line("Total input tokens:", totalIn)
	line("Total generated tokens:", totalOut)
	line("Request throughput (req/s):", float64(completed)/dur)
	line("Output token throughput (tok/s):", float64(totalOut)/dur)
	line("Total token throughput (tok/s):", float64(totalIn+totalOut)/dur)
	metricBlock(cfg, "Time to First Token", "TTFT", ttfts)
	metricBlock(cfg, "Time per Output Token (excl. 1st token)", "TPOT", tpots)
	metricBlock(cfg, "Inter-token Latency", "ITL", itls)
	metricBlock(cfg, "End-to-end Latency", "E2EL", e2els)
	fmt.Println(strings.Repeat("=", 50))
}

func metricBlock(cfg config, header, name string, xs []float64) {
	fmt.Println(center(header, 50, '-'))
	fmt.Printf("%-40s %-10.2f\n", "Mean "+name+" (ms):", mean(xs)*1000)
	fmt.Printf("%-40s %-10.2f\n", "Median "+name+" (ms):", percentile(xs, 50)*1000)
	for _, p := range cfg.percentiles {
		pw := strconv.FormatFloat(p, 'f', -1, 64)
		fmt.Printf("%-40s %-10.2f\n", fmt.Sprintf("P%s %s (ms):", pw, name), percentile(xs, p)*1000)
	}
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

// percentile replicates numpy.percentile default (linear interpolation, 'type 7').
func percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64{}, xs...)
	sort.Float64s(s)
	if len(s) == 1 {
		return s[0]
	}
	rank := p / 100 * float64(len(s)-1)
	lo := int(math.Floor(rank))
	frac := rank - float64(lo)
	if lo+1 >= len(s) {
		return s[len(s)-1]
	}
	return s[lo] + frac*(s[lo+1]-s[lo])
}

// ---- request scheduling (mirrors vLLM get_request) ----------------------------------------------

// arrivalDelays returns cumulative arrival times (s from start) for n requests. rate=inf -> all zero.
func arrivalDelays(n int, rate, burstiness float64, rng *rand.Rand) []float64 {
	d := make([]float64, n)
	if math.IsInf(rate, 1) || rate <= 0 {
		return d // all 0: admit immediately, gate by concurrency
	}
	theta := 1.0 / (rate * burstiness)
	for i := 0; i < n; i++ {
		d[i] = gamma(burstiness, theta, rng)
	}
	for i := 1; i < n; i++ {
		d[i] += d[i-1]
	}
	if d[n-1] != 0 { // normalise so the last arrival lands at n/rate (vLLM target_total_delay_s)
		factor := (float64(n) / rate) / d[n-1]
		for i := range d {
			d[i] *= factor
		}
	}
	return d
}

// gamma samples Gamma(shape, scale) (Marsaglia-Tsang). shape==1 -> exponential (vLLM burstiness==1).
func gamma(shape, scale float64, rng *rand.Rand) float64 {
	if shape < 1 {
		// boost: Gamma(shape) = Gamma(shape+1) * U^(1/shape)
		return gamma(shape+1, scale, rng) * math.Pow(rng.Float64(), 1.0/shape)
	}
	d := shape - 1.0/3.0
	c := 1.0 / math.Sqrt(9*d)
	for {
		x := rng.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := rng.Float64()
		if u < 1-0.0331*x*x*x*x || math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			return d * v * scale
		}
	}
}

func rangeBounds(mean int, ratio float64) (int, int) {
	low := int(math.Floor(float64(mean) * (1 - ratio)))
	high := int(math.Ceil(float64(mean) * (1 + ratio)))
	if low < 0 {
		low = 0
	}
	if high < low {
		high = low
	}
	return low, high
}

// ---- misc ---------------------------------------------------------------------------------------

func newClient(cfg config) *http.Client {
	tr := &http.Transport{
		Proxy:               nil, // ignore http_proxy/https_proxy (local engine)
		MaxIdleConns:        cfg.concurrency * 2,
		MaxIdleConnsPerHost: cfg.concurrency * 2,
		MaxConnsPerHost:     0,
		DisableCompression:  true,
	}
	return &http.Client{Transport: tr}
}

func idsToText(ids []int) string { // chat fallback only; not exact-token
	b := strings.Builder{}
	for i, id := range ids {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(strconv.Itoa(id))
	}
	return b.String()
}

func center(s string, n int, pad byte) string {
	if len(s) >= n {
		return s
	}
	total := n - len(s)
	left := total / 2
	return strings.Repeat(string(pad), left) + s + strings.Repeat(string(pad), total-left)
}

func rateStr(r float64) string {
	if math.IsInf(r, 1) {
		return "inf"
	}
	return strconv.FormatFloat(r, 'f', -1, 64)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func parseFlags() config {
	var cfg config
	var rate, pcts string
	flag.StringVar(&cfg.base, "base", "http://127.0.0.1:8000", "engine base URL (http://host:port)")
	flag.StringVar(&cfg.model, "model", "qwen3-8b", "model id")
	flag.StringVar(&cfg.endpoint, "endpoint", "completions", "completions | chat/completions")
	flag.IntVar(&cfg.inputLen, "input-len", 1024, "input tokens per request (random)")
	flag.IntVar(&cfg.outputLen, "output-len", 128, "output tokens per request")
	flag.IntVar(&cfg.prefixLen, "prefix-len", 0, "shared prefix tokens (identical across requests)")
	flag.IntVar(&cfg.numPrompts, "num-prompts", 320, "total requests")
	flag.IntVar(&cfg.concurrency, "concurrency", 32, "max concurrent requests (semaphore)")
	flag.StringVar(&rate, "request-rate", "inf", "requests/s, or 'inf' for no pacing (gate by concurrency)")
	flag.Float64Var(&cfg.burstiness, "burstiness", 1.0, "arrival burstiness (1=Poisson); only for finite rate")
	flag.Float64Var(&cfg.rangeRatio, "range-ratio", 0.0, "uniform +/- ratio around input/output lengths")
	flag.BoolVar(&cfg.ignoreEOS, "ignore-eos", true, "force exactly output-len tokens")
	flag.Float64Var(&cfg.temperature, "temperature", 0.0, "sampling temperature")
	flag.Int64Var(&cfg.seed, "seed", 1234, "RNG seed")
	flag.IntVar(&cfg.vocab, "vocab", 151643, "exclusive upper bound on sampled token ids (avoid specials)")
	flag.IntVar(&cfg.warmups, "warmup", 1, "warmup requests before the timed run")
	flag.StringVar(&pcts, "percentiles", "99", "comma-separated percentiles to report")
	flag.IntVar(&cfg.timeoutSec, "timeout", 600, "per-request timeout (s)")
	flag.Parse()

	if rate == "inf" {
		cfg.requestRate = math.Inf(1)
	} else {
		v, err := strconv.ParseFloat(rate, 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "bad -request-rate %q: %v\n", rate, err)
			os.Exit(2)
		}
		cfg.requestRate = v
	}
	for _, p := range strings.Split(pcts, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.ParseFloat(p, 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "bad percentile %q: %v\n", p, err)
			os.Exit(2)
		}
		cfg.percentiles = append(cfg.percentiles, v)
	}
	if cfg.vocab < 1 {
		cfg.vocab = 1
	}
	return cfg
}
