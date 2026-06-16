// flashqwen-engine: token-level C++/CUDA inference engine. Parse args, load the model + KV pool,
// serve the gRPC Engine service (token ids in -> sampled token ids out). The flashqwen Go app is
// the client and owns all model-text concerns; this binary is not meant to be run directly.
//
//   flashqwen-engine --model DIR [--address host:port] [--slots N] [--max-ctx N]
//
#include "model_spec.hpp"
#include "model_runtime.hpp"
#include "kv_cache.hpp"
#include "args.hpp"
#include "grpc_server.hpp"
#include <cstdio>
#include <string>
#include <random>

int main(int argc, char** argv) {
    Args a;
    if (int rc = parse_args(argc, argv, a); rc >= 0) return rc;   // --help / parse error

    // Declarative spec first (cheap, no GPU); gate on whether it's a model we support.
    ModelSpec spec = ModelSpec::load(a.model_dir);
    if (!spec.supported()) {
        std::fprintf(stderr,
            "error: '%s' has no readable config.json / unsupported architecture '%s'.\n"
            "       --model must point at a dir with config.json + *.safetensors; the engine\n"
            "       supports dense Qwen3 (Qwen3ForCausalLM).\n",
            a.model_dir.c_str(), spec.arch.empty() ? "unknown" : spec.arch.c_str());
        return 1;
    }

    ModelRuntime model(spec, a.max_ctx);
    // The paged KV pool takes whatever VRAM is left under the cap, so build it after the model's
    // weights + activations are resident, then hand it to the model's attention kernels.
    KVCache kv(spec, a.max_ctx, a.gpu_mem_fraction);
    model.attach_kv(kv);

    std::mt19937 rng(a.seed);
    std::string id = a.model_dir;                     // model id = directory basename
    if (auto p = id.find_last_of('/'); p != std::string::npos) id = id.substr(p + 1);
    if (id.empty()) id = "flashqwen";

    return run_grpc_server(model, kv, a.address, a.slots, a.max_queue, id, rng);
}
