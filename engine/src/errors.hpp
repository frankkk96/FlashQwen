// Request-level error taxonomy for the engine's serving path. Deliberately free of gRPC/proto
// types so the scheduler (pure compute + scheduling) can report domain errors without depending on
// the wire format. The gRPC layer (grpc_server.cpp) is the single place that translates an
// EngineErrc into the proto ErrorCode put on the stream. Startup/load failures are NOT modelled
// here — those abort the process before any RPC is served (the Go supervisor sees the engine die).
#pragma once

enum class EngineErrc {
    OverCapacity,   // KV pool exhausted or request queue full — retryable (Go maps to HTTP 503)
    Internal,       // unexpected engine failure — (Go maps to HTTP 502)
};
