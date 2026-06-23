// Per-request output channel. The scheduler writes results straight to the
// request's sink; Request co-owns it via shared_ptr so it outlives the Request,
// and the transport (gRPC) implements the wire format. Cancellation is a flag
// the transport sets and the scheduler polls — no request pointer crosses back.
#pragma once
#include <string>

#include "errors.hpp"

struct OutputSink {
  virtual ~OutputSink() = default;
  virtual void Token(int id) = 0;                      // one generated token id
  virtual void Done(const std::string& finish_reason,  // terminal: success
                    int prompt_tokens, int completion_tokens) = 0;
  virtual void Error(EngineErrc code,
                     const std::string& msg) = 0;  // terminal: failure
  virtual bool Cancelled() const = 0;  // set by transport on client disconnect
};
