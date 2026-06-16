#include "args.hpp"
#include "CLI11.hpp"

int parse_args(int argc, char** argv, Args& out) {
    CLI::App app{"flashqwen-engine — token-level C++/CUDA inference engine for Qwen3 (dense). "
                 "Driven over gRPC by the flashqwen Go app; not meant to be run directly."};
    app.footer(
        "SUPPORTED MODELS\n"
        "  Any dense Qwen3 model (architecture Qwen3ForCausalLM): Qwen3-0.6B / 1.7B / 4B /\n"
        "  8B / 14B / 32B. Dims are read from config.json. (Tested: Qwen3-8B.)");

    app.add_option("--model", out.model_dir, "model directory: config.json + *.safetensors")
        ->required()->check(CLI::ExistingDirectory);
    app.add_option("--max-ctx", out.max_ctx, "KV / context length")->capture_default_str();
    app.add_option("--gpu-mem-fraction", out.gpu_mem_fraction,
                   "VRAM cap; the paged KV block pool gets whatever is left under it")
        ->capture_default_str()->check(CLI::Range(0.1, 1.0));
    app.add_option("--seed", out.seed, "RNG seed for sampling")->capture_default_str();
    app.add_option("--address", out.address, "gRPC listen address (host:port)")->capture_default_str();
    app.add_option("--slots", out.slots, "max concurrent sequences")->capture_default_str();
    app.add_option("--max-queue", out.max_queue,
                   "admission cap on waiting requests; over it new requests are rejected as "
                   "over-capacity (<=0 => 4*slots)")->capture_default_str();

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);   // prints help/error; exit code 0 for --help, nonzero on error
    }
    return -1;   // parsed OK; main continues (model-arch support is checked there via ModelSpec)
}
