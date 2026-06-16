#include "sampler.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

static int argmax(const float* v, int V) {
    int best = 0; float bv = v[0];
    for (int i = 1; i < V; ++i) if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
}

int sample(const std::vector<float>& logits, const SampleParams& sp, std::mt19937& rng) {
    return sample(logits.data(), (int)logits.size(), sp, rng);
}

// Temperature + top-p sampling. The softmax denominator inherently scans the whole vocab (O(V),
// cheap), but we avoid the O(V log V) of sorting all of it: with no nucleus truncation we sample
// the full distribution in one pass, and with top_p<1 we pop a max-heap only until the nucleus
// reaches top_p — usually a few hundred tokens, not all ~150k.
int sample(const float* logits, int V, const SampleParams& sp, std::mt19937& rng) {
    if (sp.temp <= 0.0f) return argmax(logits, V);

    // Pass 1: max logit (numerical stability). Pass 2: exp-weights w[i] and their sum Z.
    float maxl = logits[0];
    for (int i = 1; i < V; ++i) if (logits[i] > maxl) maxl = logits[i];
    const float invT = 1.0f / sp.temp;
    std::vector<float> w(V);
    double Z = 0.0;
    for (int i = 0; i < V; ++i) { w[i] = std::exp((logits[i] - maxl) * invT); Z += w[i]; }

    std::uniform_real_distribution<float> dist(0.f, 1.f);

    // No nucleus truncation (top_p >= 1): sample the full distribution directly, O(V), no ordering.
    if (sp.top_p >= 1.0f) {
        float r = dist(rng) * (float)Z, acc = 0.f;
        for (int i = 0; i < V; ++i) { acc += w[i]; if (acc >= r) return i; }
        return V - 1;
    }

    // Nucleus (top_p < 1): visit tokens highest-logit first via a max-heap, accumulating true
    // probability until it reaches top_p. make_heap is O(V); only the nucleus is popped (O(log V)
    // each), so the full vocab is never sorted.
    const float invZ = (float)(1.0 / Z);
    std::vector<int> heap(V);
    for (int i = 0; i < V; ++i) heap[i] = i;
    auto by_logit = [&](int a, int b) { return logits[a] < logits[b]; };   // "less" => max-heap on logit
    std::make_heap(heap.begin(), heap.end(), by_logit);

    std::vector<int> kept;
    float cum = 0.f;
    for (int n = V; n > 0; ) {
        std::pop_heap(heap.begin(), heap.begin() + n, by_logit);
        int id = heap[--n];
        kept.push_back(id);
        cum += w[id] * invZ;
        if (cum >= sp.top_p) break;
    }

    // sample within the kept nucleus (probabilities normalised by Z; their sum cum >= top_p)
    float r = dist(rng) * cum, acc = 0.f;
    for (int id : kept) { acc += w[id] * invZ; if (acc >= r) return id; }
    return kept.back();
}
