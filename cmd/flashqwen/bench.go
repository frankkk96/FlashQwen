package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"flashqwen/internal/bench"
	"flashqwen/internal/server"
	"flashqwen/internal/tokenizer"

	"github.com/gin-gonic/gin"
)

// runBench is the end-to-end serving benchmark. It drives synthetic chat-completion load over HTTP
// — the full tokenizer -> ChatML -> gRPC -> engine path — and reports latency + throughput in the
// style of vLLM's `bench serve`. With no --base-url it spawns the embedded engine and an in-process
// HTTP server; with --base-url it benchmarks an already-running server.
func runBench(args []string) {
	fs := flag.NewFlagSet("benchmark", flag.ExitOnError)
	model := fs.String("model", "", "model directory (required: provides the tokenizer, and the engine when self-hosting)")
	baseURL := fs.String("base-url", "", "benchmark an already-running server (e.g. http://host:8000); empty => spawn engine + server in-process")
	slots := fs.Int("slots", 16, "max concurrent sequences (in-process server only)")
	maxCtx := fs.Int("max-ctx", 4096, "KV / context length (in-process server only)")
	maxQueue := fs.Int("max-queue", 0, "engine admission queue cap (in-process server only; 0 => 4*slots)")
	numPrompts := fs.Int("num-prompts", 200, "total requests in the timed run")
	inputLen := fs.Int("input-len", 512, "synthetic prompt length in tokens (approximate)")
	outputLen := fs.Int("output-len", 128, "max tokens to generate per request")
	rangeRatio := fs.Float64("range-ratio", 0.0, "uniform +/- spread on input length (0..1)")
	prefixLen := fs.Int("prefix-len", 0, "shared prefix length prepended to every prompt, in tokens (exercises prefix reuse)")
	rate := fs.String("request-rate", "inf", "request arrival rate in req/s, or 'inf' to fire all at once (saturation)")
	burstiness := fs.Float64("burstiness", 1.0, "arrival burstiness (Gamma shape): 1=Poisson, <1 burstier, >1 smoother")
	maxConc := fs.Int("max-concurrency", 0, "cap on in-flight requests (0 => unbounded)")
	warmups := fs.Int("warmups", 3, "warmup requests before the timed run (discarded)")
	temperature := fs.Float64("temperature", 0, "sampling temperature (0 => greedy)")
	topP := fs.Float64("top-p", 1.0, "nucleus top-p")
	percentiles := fs.String("percentiles", "50,90,95,99", "comma-separated latency percentiles to report")
	goodput := fs.String("goodput", "", "goodput SLOs, e.g. 'ttft:500,tpot:50,e2el:5000' (ms)")
	seed := fs.Uint64("seed", 1234, "PRNG seed for prompt synthesis and arrivals")
	outputJSON := fs.String("output-json", "", "also write the full report to this JSON file")
	fs.Parse(args)
	if *model == "" {
		log.Fatal("benchmark: --model is required")
	}

	reqRate, err := parseRate(*rate)
	if err != nil {
		log.Fatalf("benchmark: %v", err)
	}
	pctls, err := parsePercentiles(*percentiles)
	if err != nil {
		log.Fatalf("benchmark: %v", err)
	}
	slos, err := parseGoodput(*goodput)
	if err != nil {
		log.Fatalf("benchmark: %v", err)
	}

	// Resolve the target server and the tokenizer used to synthesise prompts.
	var tok *tokenizer.Tokenizer
	target := *baseURL
	modelName := "flashqwen"
	if target == "" {
		s, err := open(*model, *slots, *maxCtx, *maxQueue)
		if err != nil {
			log.Fatalf("benchmark: %v", err)
		}
		defer s.stop()
		addr, err := serveInProcess(s)
		if err != nil {
			log.Fatalf("benchmark: %v", err)
		}
		target, tok, modelName = "http://"+addr, s.tok, s.info.ID
	} else {
		tok, err = tokenizer.Load(*model)
		if err != nil {
			log.Fatalf("benchmark: load tokenizer: %v", err)
		}
		modelName = filepath.Base(strings.TrimRight(*model, "/"))
	}

	cfg := bench.Config{
		BaseURL: target, Model: modelName,
		NumPrompts: *numPrompts, InputLen: *inputLen, OutputLen: *outputLen,
		RangeRatio: *rangeRatio, PrefixLen: *prefixLen,
		RequestRate: reqRate, Burstiness: *burstiness, MaxConc: *maxConc,
		Warmups: *warmups, Temperature: *temperature, TopP: *topP,
		Percentiles: pctls, Goodput: slos, Seed: *seed, OutputJSON: *outputJSON,
	}
	if err := bench.Run(context.Background(), cfg, tok); err != nil {
		log.Fatalf("benchmark: %v", err)
	}
}

// serveInProcess starts the OpenAI server on a free loopback port in a goroutine and blocks until it
// answers /healthz, returning its address. gin's access log is silenced so it doesn't drown the
// benchmark output.
func serveInProcess(s *session) (string, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", err
	}
	addr := ln.Addr().String()
	ln.Close() // hand the port to gin (a small TOCTOU window, fine for a local benchmark)

	gin.DefaultWriter = io.Discard
	gin.DefaultErrorWriter = io.Discard
	go func() {
		if err := server.New(s.gen, s.info.ID).Run(addr); err != nil {
			log.Printf("benchmark: in-process server exited: %v", err)
		}
	}()

	hc := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(30 * time.Second)
	for {
		resp, err := hc.Get("http://" + addr + "/healthz")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return addr, nil
			}
		}
		if time.Now().After(deadline) {
			return "", fmt.Errorf("in-process server did not become ready within 30s")
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func parseRate(s string) (float64, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "inf", "+inf":
		return math.Inf(1), nil
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || v <= 0 {
		return 0, fmt.Errorf("invalid --request-rate %q (want a positive number or 'inf')", s)
	}
	return v, nil
}

func parsePercentiles(s string) ([]float64, error) {
	var out []float64
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		v, err := strconv.ParseFloat(part, 64)
		if err != nil || v < 0 || v > 100 {
			return nil, fmt.Errorf("invalid percentile %q (want 0..100)", part)
		}
		out = append(out, v)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("--percentiles is empty")
	}
	return out, nil
}

func parseGoodput(s string) (map[string]float64, error) {
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}
	out := map[string]float64{}
	for _, part := range strings.Split(s, ",") {
		k, v, ok := strings.Cut(strings.TrimSpace(part), ":")
		if !ok {
			return nil, fmt.Errorf("invalid --goodput entry %q (want metric:ms)", part)
		}
		k = strings.TrimSpace(k)
		if k != "ttft" && k != "tpot" && k != "e2el" {
			return nil, fmt.Errorf("invalid --goodput metric %q (want ttft, tpot, or e2el)", k)
		}
		thr, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		if err != nil || thr < 0 {
			return nil, fmt.Errorf("invalid --goodput threshold for %q: %q", k, v)
		}
		out[k] = thr
	}
	return out, nil
}
