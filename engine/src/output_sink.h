#pragma once
#include <string>

#include "errors.h"

// Per-request output channel. The scheduler streams tokens and the terminal
// Done/Error straight to the request's sink (gRPC implements the wire format),
// and polls Cancelled(), which the transport sets on client disconnect. Request
// co-owns the sink via shared_ptr, so it outlives the Request.
struct OutputSink {
  virtual ~OutputSink() = default;
  virtual void Token(int id) = 0;
  virtual void Done(const std::string& finish_reason, int prompt_tokens,
                    int completion_tokens) = 0;
  virtual void Error(EngineErrc code, const std::string& msg) = 0;
  virtual bool Cancelled() const = 0;
};
