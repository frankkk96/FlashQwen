// Token sampling: greedy (temp<=0), or temperature + top-p (nucleus). These are the two knobs
// the OpenAI API exposes; top-k is intentionally not supported.
#pragma once
#include <random>

struct SampleParams {
    float temp;    // <= 0 means greedy (argmax)
    float top_p;   // nucleus cutoff (1.0 = no truncation)
};

// Pick the next token id from a logits row (pointer + length, e.g. one row of a [B,vocab] buffer).
int sample(const float* logits, int V, const SampleParams& sp, std::mt19937& rng);
