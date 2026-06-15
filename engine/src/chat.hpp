// Interactive multi-turn chat loop (default mode).
#pragma once
#include "model/model.hpp"
#include "model/tokenizer.hpp"
#include "sampler.hpp"
#include <random>

int run_chat(Model& model, const Tokenizer& tok, SampleParams sp, std::mt19937& rng,
             bool think, int max_ctx);
