package bench

import (
	"math/rand/v2"

	"flashqwen/internal/tokenizer"
)

// reqSpec is a single synthetic request: a prompt and how many tokens to ask for.
type reqSpec struct {
	prompt    string
	maxTokens int
}

// randomWorkload synthesises n requests from random vocabulary tokens. A shared prefix of prefixLen
// tokens (decoded once) is prepended to every prompt — the lever for exercising prefix-cache reuse.
// Body length varies uniformly within ±rangeRatio of inputLen. Because byte-level BPE round-trips
// (id -> text -> re-tokenise) drift slightly, the realised prompt length is approximate; the bench
// reports the server's authoritative token counts, not these targets.
func randomWorkload(tok *tokenizer.Tokenizer, n, inputLen, outputLen, prefixLen int, rangeRatio float64, seed uint64) []reqSpec {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	vocab := tok.VocabSize()
	span := vocab - 1000
	if span < 1 {
		span = 1
	}
	randText := func(ntok int) string {
		ids := make([]int, ntok)
		for i := range ids {
			ids[i] = 100 + rng.IntN(span) // arbitrary valid, non-special ids
		}
		return tok.Decode(ids)
	}

	prefix := ""
	if prefixLen > 0 {
		prefix = randText(prefixLen)
	}
	specs := make([]reqSpec, n)
	for i := range specs {
		bodyLen := inputLen
		if rangeRatio > 0 {
			lo := float64(inputLen) * (1 - rangeRatio)
			hi := float64(inputLen) * (1 + rangeRatio)
			bodyLen = int(lo + rng.Float64()*(hi-lo))
		}
		if bodyLen < 1 {
			bodyLen = 1
		}
		specs[i] = reqSpec{prompt: prefix + randText(bodyLen), maxTokens: outputLen}
	}
	return specs
}
