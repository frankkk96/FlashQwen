// Token-level gRPC inference engine (the only mode). The Go app is the client: it sends prompt
// token ids + sampling + stop ids, and the engine streams back sampled token ids. The engine has
// no model-text knowledge — tokenisation, chat template, and tool-call detection all live in Go.
//
// The engine loads the model first and only then binds its gRPC port, so a successful GetModel
// doubles as the readiness signal; load progress and failures are written to stderr.
#pragma once
#include "args.hpp"
#include <string>
#include <random>

int run_engine(const Args& a, const std::string& model_id, std::mt19937& rng);
