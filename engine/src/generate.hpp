// Core generation loop shared by chat and benchmark.
//
// Prefills `chunk` at KV-cache position `past`, then decodes up to `max_gen` tokens.
// `past` is advanced to reflect the tokens written to the cache; because every emitted
// token is fed back through the model, the cache always holds the full prefix (which is
// what makes multi-turn chat work).
#pragma once
#include "model.hpp"
#include "tokenizer.hpp"
#include "sampler.hpp"
#include <vector>
#include <random>

struct GenStats {
    double ttft_ms = 0;          // prefill + first token
    std::vector<double> step_ms; // per-decode-step latency
    int n_out = 0;               // tokens emitted
};

GenStats generate(Model& model, const Tokenizer& tok, const std::vector<int>& chunk,
                  int& past, int max_gen, const SampleParams& sp, std::mt19937& rng,
                  bool stream, bool stop_on_eos);
