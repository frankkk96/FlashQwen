// Request-level serving errors, free of gRPC/proto types so the scheduler can
// report them without the wire format. grpc_server.cpp maps these to proto
// ErrorCode. Startup/load failures abort the process instead.
#pragma once

enum class EngineErrc {
  kOverCapacity,  // KV pool exhausted or queue full; retryable (Go -> HTTP 503)
  kInternal,      // unexpected engine failure (Go -> HTTP 502)
};
