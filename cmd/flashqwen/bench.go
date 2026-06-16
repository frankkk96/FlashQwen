package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"sync"
	"time"
)

// runBench is an end-to-end throughput driver: it fires synthetic requests (random token ids) at
// the embedded engine over gRPC and reports aggregate tok/s + mean time-to-first-token at a few
// concurrency levels. Unlike a pure-GPU microbenchmark it includes the gRPC + Go path, i.e. what a
// client actually sees.
func runBench(args []string) {
	fs := flag.NewFlagSet("benchmark", flag.ExitOnError)
	model := fs.String("model", "", "model directory (required)")
	slots := fs.Int("slots", 16, "max concurrent sequences")
	maxCtx := fs.Int("max-ctx", 2048, "KV / context length")
	input := fs.Int("input", 128, "synthetic prompt length (tokens)")
	output := fs.Int("output", 128, "tokens to generate per request")
	requests := fs.Int("requests", 32, "total requests per concurrency level")
	fs.Parse(args)
	if *model == "" {
		log.Fatal("benchmark: --model is required")
	}

	s, err := open(*model, *slots, *maxCtx)
	if err != nil {
		log.Fatalf("benchmark: %v", err)
	}
	defer s.stop()

	vocab := s.info.VocabSize
	rng := rand.New(rand.NewSource(1234))
	makePrompt := func() []int32 {
		ids := make([]int32, *input)
		for i := range ids {
			ids[i] = int32(100 + rng.Intn(vocab-1000)) // arbitrary valid, non-special ids
		}
		return ids
	}

	type result struct {
		ttft       time.Duration
		completion int
		err        error
	}

	runConc := func(conc int) {
		jobs := make(chan []int32, *requests)
		for i := 0; i < *requests; i++ {
			jobs <- makePrompt()
		}
		close(jobs)
		results := make(chan result, *requests)

		start := time.Now()
		var wg sync.WaitGroup
		for w := 0; w < conc; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for ids := range jobs {
					t0 := time.Now()
					var first time.Time
					got := false
					n := 0
					_, err := s.eng.Stream(context.Background(), ids, *output,
						0, 1.0, nil, // greedy, no nucleus, no stop ids: full length
						func(int32) {
							if !got {
								first = time.Now()
								got = true
							}
							n++
						})
					results <- result{ttft: first.Sub(t0), completion: n, err: err}
				}
			}()
		}
		wg.Wait()
		close(results)
		wall := time.Since(start)

		var totalTok, cnt, errs int
		var ttftSum time.Duration
		for r := range results {
			if r.err != nil {
				errs++
				continue
			}
			totalTok += r.completion
			ttftSum += r.ttft
			cnt++
		}
		meanTTFT := time.Duration(0)
		if cnt > 0 {
			meanTTFT = ttftSum / time.Duration(cnt)
		}
		errNote := ""
		if errs > 0 {
			errNote = fmt.Sprintf("  (%d errors)", errs)
		}
		fmt.Printf("  %5d  %7d  %9.2f  %12.1f  %10.1f%s\n",
			conc, *requests, wall.Seconds(), float64(totalTok)/wall.Seconds(),
			float64(meanTTFT.Microseconds())/1000.0, errNote)
	}

	fmt.Printf("\nend-to-end throughput (input %d, output %d, %d requests/level, model %s)\n\n",
		*input, *output, *requests, s.info.ID)
	fmt.Println("  conc  requests   wall (s)   aggregate t/s   TTFT (ms)")
	for _, c := range []int{1, 8, 16} {
		if c > *slots {
			continue
		}
		runConc(c)
	}
}
