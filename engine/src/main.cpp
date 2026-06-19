// flashqwen-engine: token-level C++/CUDA inference engine. Parse args, then serve the gRPC Engine
// service (token ids in -> sampled token ids out): the server binds its port immediately and loads
// the model on a background thread, reporting progress through GetStatus. The flashqwen Go app is
// the client and owns all model-text concerns; this binary is not meant to be run directly.
//
//   flashqwen-engine --model DIR [--address host:port] [--slots N] [--max-ctx N]
//
#include "args.hpp"
#include "grpc_service.hpp"
#include <string>
#include <random>

int main(int argc, char** argv) {
    Args a;
    if (int rc = parse_args(argc, argv, a); rc >= 0) return rc;   // --help / parse error

    std::mt19937 rng(a.seed);
    std::string id = a.model_dir;                     // model id = directory basename
    if (auto p = id.find_last_of('/'); p != std::string::npos) id = id.substr(p + 1);
    if (id.empty()) id = "flashqwen";

    return run_engine(a, id, rng);
}
