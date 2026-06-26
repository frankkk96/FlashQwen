#pragma once

// Request-level serving errors (no gRPC/proto types); grpc_sink.h maps these to
// the proto wire code. Startup/load failures abort the process instead.
enum class EngineErrc {
  kOverCapacity,  // KV pool exhausted or queue full; retryable (Go -> 503)
  kInternal,      // unexpected engine failure (Go -> 502)
};
