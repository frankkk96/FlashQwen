#pragma once

namespace fq {

// Request-level serving errors (no gRPC/proto types); grpc_sink.h maps these to
// the proto wire code. Startup/load failures abort the process instead.
enum class EngineErrc {
  kOverCapacity,
  kInternal,
};

}
