// flashqwen-engine: token-level C++/CUDA inference engine. Parses args, then
// serves the gRPC Engine service (token ids in -> sampled token ids out). Model
// loads before the port binds, so a successful GetModel is the readiness
// signal. Driven by the flashqwen Go app; not run directly.
//
//   flashqwen-engine --model DIR [--address host:port] [--slots N] [--max-ctx
//   N]
//
#include <string>

#include "args.hpp"
#include "grpc_service.hpp"

int main(int argc, char** argv) {
  Args a;
  if (int rc = ParseArgs(argc, argv, a); rc >= 0)
    return rc;  // --help / parse error

  std::string id = a.model_dir;  // model id = directory basename
  if (auto p = id.find_last_of('/'); p != std::string::npos)
    id = id.substr(p + 1);
  if (id.empty()) id = "flashqwen";

  return RunEngine(a, id);
}
