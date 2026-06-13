#include "sampler.hpp"
#include <algorithm>
#include <cmath>

static int argmax(const std::vector<float>& v) {
    int best = 0; float bv = v[0];
    for (int i = 1; i < (int)v.size(); ++i) if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
}

int sample(const std::vector<float>& logits, const SampleParams& sp, std::mt19937& rng) {
    if (sp.temp <= 0.0f) return argmax(logits);

    int V = (int)logits.size();
    std::vector<int> idx(V);
    for (int i = 0; i < V; ++i) idx[i] = i;

    int k = (sp.top_k <= 0 || sp.top_k > V) ? V : sp.top_k;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b){ return logits[a] > logits[b]; });
    idx.resize(k);

    float maxl = logits[idx[0]];
    std::vector<float> probs(k);
    float sum = 0.f;
    for (int i = 0; i < k; ++i) { probs[i] = std::exp((logits[idx[i]] - maxl) / sp.temp); sum += probs[i]; }
    for (auto& p : probs) p /= sum;

    // nucleus (top-p) truncation on the already-descending list
    float cum = 0.f; int keep = k;
    for (int i = 0; i < k; ++i) { cum += probs[i]; if (cum >= sp.top_p) { keep = i + 1; break; } }

    std::uniform_real_distribution<float> dist(0.f, cum);
    float r = dist(rng), acc = 0.f;
    for (int i = 0; i < keep; ++i) { acc += probs[i]; if (r <= acc) return idx[i]; }
    return idx[keep - 1];
}
