// gRPC inference server (serve mode). The Go OpenAI gateway is the client. The engine owns all
// model-specific logic: it receives structured chat messages + tools, renders the Qwen3 template,
// tokenises, runs continuous-batched generation, detects tool calls at the token level, and
// streams typed events (text deltas / tool calls / done) back over a server-streaming RPC.
#pragma once
#include "model/model.hpp"
#include "model/tokenizer.hpp"
#include <string>
#include <random>

int run_grpc_server(Model& model, const Tokenizer& tok, const std::string& address,
                    int n_slots, const std::string& model_id, std::mt19937& rng);
