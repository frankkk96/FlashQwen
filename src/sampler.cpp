#include "sampler.hpp"
#include <algorithm>
#include <cmath>

static int argmax(const float* v, int V) {
    int best = 0; float bv = v[0];
    for (int i = 1; i < V; ++i) if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
}

int sample(const std::vector<float>& logits, const SampleParams& sp, std::mt19937& rng) {
    return sample(logits.data(), (int)logits.size(), sp, rng);
}

int sample(const float* logits, int V, const SampleParams& sp, std::mt19937& rng) {
    if (sp.temp <= 0.0f) return argmax(logits, V);

    // sort the whole vocab by logit, descending
    std::vector<int> idx(V);
    for (int i = 0; i < V; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return logits[a] > logits[b]; });

    // temperature softmax over the sorted logits
    float maxl = logits[idx[0]];
    std::vector<float> probs(V);
    float sum = 0.f;
    for (int i = 0; i < V; ++i) { probs[i] = std::exp((logits[idx[i]] - maxl) / sp.temp); sum += probs[i]; }
    for (auto& p : probs) p /= sum;

    // nucleus (top-p) truncation on the descending list
    float cum = 0.f; int keep = V;
    for (int i = 0; i < V; ++i) { cum += probs[i]; if (cum >= sp.top_p) { keep = i + 1; break; } }

    std::uniform_real_distribution<float> dist(0.f, cum);
    float r = dist(rng), acc = 0.f;
    for (int i = 0; i < keep; ++i) { acc += probs[i]; if (r <= acc) return idx[i]; }
    return idx[keep - 1];
}
