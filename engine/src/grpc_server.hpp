// Token-level gRPC inference engine (the only mode). The Go app is the client: it sends prompt
// token ids + sampling + stop ids, and the engine streams back sampled token ids. The engine has
// no model-text knowledge — tokenisation, chat template, and tool-call detection all live in Go.
//
// The server binds its port immediately and loads the model on a background thread, publishing
// progress through GetStatus; GetModel/Generate answer only once loading reaches READY.
#pragma once
#include "args.hpp"
#include <string>
#include <random>

int run_engine(const Args& a, const std::string& model_id, std::mt19937& rng);
