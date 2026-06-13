// Token sampling: greedy (temp<=0), or temperature + top-k + top-p (nucleus).
#pragma once
#include <vector>
#include <random>

struct SampleParams {
    float temp;    // <= 0 means greedy (argmax)
    float top_p;
    int   top_k;
};

// Pick the next token id from a logits vector.
int sample(const std::vector<float>& logits, const SampleParams& sp, std::mt19937& rng);
