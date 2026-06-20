// Per-request output channel — the engine's side of streaming results back to a caller.
//
// The scheduler writes a request's results straight to its sink (token / done / error) instead of
// firing callbacks or looking the request up in a central table: each Request co-owns its sink via
// shared_ptr, so the sink outlives the Request and the transport's read side keeps it alive until it
// has drained the stream. The transport (gRPC) implements this interface; the engine knows nothing
// about the wire format. Cancellation is a flag the transport sets and the scheduler polls — so no
// raw request pointer ever crosses back from the transport (no dangling, no central registry).
#pragma once
#include "errors.hpp"
#include <string>

struct OutputSink {
    virtual ~OutputSink() = default;
    virtual void token(int id) = 0;                       // one generated token id
    virtual void done(const std::string& finish_reason,   // terminal: success
                      int prompt_tokens, int completion_tokens) = 0;
    virtual void error(EngineErrc code, const std::string& msg) = 0;   // terminal: failure
    virtual bool cancelled() const = 0;                   // transport sets this on client disconnect
};
