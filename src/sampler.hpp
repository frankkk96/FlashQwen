// Token sampling: greedy (temp<=0), or temperature + top-k + top-p (nucleus).
#pragma once
#include <vector>
#include <random>

struct SampleParams {
    float temp;    // <= 0 means greedy (argmax)
    float top_p;
    int   top_k;
};

// Pick the next token id from a logits row (pointer + length, e.g. one row of a [B,vocab] buffer).
int sample(const float* logits, int V, const SampleParams& sp, std::mt19937& rng);

// Convenience overload for a whole-vector logits row.
int sample(const std::vector<float>& logits, const SampleParams& sp, std::mt19937& rng);
