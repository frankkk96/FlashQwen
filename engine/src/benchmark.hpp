// Benchmark mode (single stream, batch 1).
//
// Runs a fixed, self-contained sweep: it builds synthetic prompts of several input lengths
// internally and decodes a fixed number of output tokens, printing a table of
// TTFT / TPOT / decode / output / peak throughput (median of a few runs, after a warmup).
// The sweep parameters are defined inside benchmark.cpp; there are no benchmark CLI flags.
#pragma once
#include "model/model.hpp"
#include "model/tokenizer.hpp"

int run_benchmark(Model& model, const Tokenizer& tok, int max_ctx);
