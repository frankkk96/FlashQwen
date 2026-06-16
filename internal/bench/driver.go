package bench

import (
	"context"
	"math"
	"math/rand/v2"
	"net/http"
	"sync"
	"time"
)

// arrivalOffsets precomputes each request's start time relative to t0. At an infinite rate every
// request starts at t0 (a saturation test). Otherwise inter-arrival gaps are drawn from
// Gamma(shape=burstiness, scale=1/(rate·burstiness)) — mean gap 1/rate, with burstiness controlling
// variance: 1 => exponential (Poisson arrivals), <1 burstier, >1 smoother.
func arrivalOffsets(n int, rate, burstiness float64, seed uint64) []time.Duration {
	offsets := make([]time.Duration, n)
	if math.IsInf(rate, 1) || rate <= 0 {
		return offsets // all zero: fire everything at once
	}
	rng := rand.New(rand.NewPCG(seed^0xd1b54a32d192ed03, seed))
	scale := 1.0 / (rate * burstiness)
	var cum float64
	for i := range offsets {
		cum += sampleGamma(rng, burstiness) * scale
		offsets[i] = time.Duration(cum * float64(time.Second))
	}
	return offsets
}

// sampleGamma draws from Gamma(shape, 1) via the Marsaglia–Tsang method.
func sampleGamma(rng *rand.Rand, shape float64) float64 {
	if shape < 1 {
		return sampleGamma(rng, shape+1) * math.Pow(rng.Float64(), 1/shape)
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
			return d * v
		}
	}
}

// runLoad fires the workload and returns one record per request plus the wall-clock duration of the
// timed window. Each request is launched on its arrival offset (so the arrival process is preserved
// regardless of how the server keeps up); maxConc>0 caps in-flight requests via a semaphore.
func runLoad(ctx context.Context, hc *http.Client, baseURL, model string, specs []reqSpec,
	offsets []time.Duration, maxConc int, temp, topP float64) ([]record, time.Duration) {

	records := make([]record, len(specs))
	var sem chan struct{}
	if maxConc > 0 {
		sem = make(chan struct{}, maxConc)
	}
	var wg sync.WaitGroup
	start := time.Now()
	for i := range specs {
		if d := time.Until(start.Add(offsets[i])); d > 0 {
			time.Sleep(d)
		}
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if sem != nil {
				sem <- struct{}{}
				defer func() { <-sem }()
			}
			records[i] = streamOnce(ctx, hc, baseURL, model, specs[i], temp, topP)
		}(i)
	}
	wg.Wait()
	return records, time.Since(start)
}
