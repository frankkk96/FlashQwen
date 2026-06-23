// Token-level gRPC inference engine. The client sends prompt token ids +
// sampling + stop ids; the engine streams back sampled token ids. No model-text
// knowledge here — tokenisation/chat template/tool-call detection live in Go.
// The port binds only after the model loads, so a successful GetModel is the
// readiness signal; load progress/failures go to stderr.
#pragma once
#include <string>

#include "args.hpp"

int RunEngine(const Args& a, const std::string& model_id);
