#pragma once
#include <string>

#include "args.h"

// Load the model, then serve the gRPC Engine service (token ids in -> sampled
// token ids out). The port binds only after the model loads, so a successful
// GetModel is the readiness signal. Returns a process exit code.
int RunEngine(const Args& a, const std::string& model_id);
