// Package bench is FlashQwen's end-to-end serving benchmark. It drives synthetic chat-completion
// load over HTTP against a running server — exercising the full tokenizer -> ChatML -> gRPC ->
// engine path, i.e. what a real client sees — and reports latency (TTFT / TPOT / ITL / E2EL) and
// throughput, modelled on vLLM's `vllm bench serve`.
package bench

import (
	"context"
	"fmt"
	"net/http"
	"os"

	"flashqwen/internal/tokenizer"
)

// Config is the benchmark configuration. BaseURL points at the server under test; the caller is
// responsible for starting it (or pointing at an external one). RequestRate is in req/s, with
// +Inf meaning "fire everything at once" (saturation). Goodput maps metric names (ttft/tpot/e2el)
// to SLO thresholds in milliseconds; nil disables goodput.
type Config struct {
	BaseURL     string
	Model       string
	NumPrompts  int
	InputLen    int
	OutputLen   int
	RangeRatio  float64
	PrefixLen   int
	RequestRate float64
	Burstiness  float64
	MaxConc     int
	Warmups     int
	Temperature float64
	TopP        float64
	Percentiles []float64
	Goodput     map[string]float64
	Seed        uint64
	OutputJSON  string
}

// Run synthesises the workload, warms up, drives the timed load, then prints (and optionally writes)
// the report. tok is used only to build prompts of a target token length.
func Run(ctx context.Context, cfg Config, tok *tokenizer.Tokenizer) error {
	idle := cfg.MaxConc
	if idle <= 0 {
		idle = 256
	}
	hc := &http.Client{Transport: &http.Transport{MaxIdleConns: 0, MaxIdleConnsPerHost: idle}}

	if cfg.Warmups > 0 {
		fmt.Fprintf(os.Stderr, "warming up (%d requests)...\n", cfg.Warmups)
		warm := randomWorkload(tok, cfg.Warmups, cfg.InputLen, cfg.OutputLen, cfg.PrefixLen, cfg.RangeRatio, cfg.Seed+1)
		for _, w := range warm {
			streamOnce(ctx, hc, cfg.BaseURL, cfg.Model, w, cfg.Temperature, cfg.TopP)
		}
	}

	specs := randomWorkload(tok, cfg.NumPrompts, cfg.InputLen, cfg.OutputLen, cfg.PrefixLen, cfg.RangeRatio, cfg.Seed)
	offsets := arrivalOffsets(cfg.NumPrompts, cfg.RequestRate, cfg.Burstiness, cfg.Seed)

	fmt.Fprintf(os.Stderr, "running %d requests (rate %s, concurrency %s)...\n",
		cfg.NumPrompts, rateString(cfg.RequestRate), concString(cfg.MaxConc))
	records, dur := runLoad(ctx, hc, cfg.BaseURL, cfg.Model, specs, offsets, cfg.MaxConc, cfg.Temperature, cfg.TopP)

	rep := aggregate(cfg, records, dur)
	printReport(rep, cfg)
	if cfg.OutputJSON != "" {
		if err := writeJSON(cfg.OutputJSON, rep); err != nil {
			return fmt.Errorf("write %s: %w", cfg.OutputJSON, err)
		}
		fmt.Fprintf(os.Stderr, "\nreport written to %s\n", cfg.OutputJSON)
	}
	if rep.Completed == 0 {
		return fmt.Errorf("all %d requests failed", rep.Failed)
	}
	return nil
}

func concString(maxConc int) string {
	if maxConc <= 0 {
		return "unbounded"
	}
	return fmt.Sprintf("%d", maxConc)
}
