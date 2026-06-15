// Interactive multi-turn chat loop (default mode).
#pragma once
#include "model_runtime.hpp"
#include "tokenizer.hpp"
#include "kv_cache.hpp"
#include "sampler.hpp"
#include <random>

int run_chat(ModelRuntime& model, const KVCache& kv, const Tokenizer& tok, SampleParams sp,
             std::mt19937& rng, bool think, int max_ctx);
