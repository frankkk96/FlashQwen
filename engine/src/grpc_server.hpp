// Token-level gRPC inference engine (the only mode). The Go app is the client: it sends prompt
// token ids + sampling + stop ids, and the engine streams back sampled token ids. The engine has
// no model-text knowledge — tokenisation, chat template, and tool-call detection all live in Go.
#pragma once
#include "model_runtime.hpp"
#include "kv_cache.hpp"
#include <string>
#include <random>

int run_grpc_server(ModelRuntime& model, const KVCache& kv, const std::string& address,
                    int n_slots, int max_queue, const std::string& model_id, std::mt19937& rng);
