#include "args.h"

#include "CLI11.hpp"

namespace fq {

int ParseArgs(int argc, char** argv, Args& out) {
  CLI::App app{
      "flashqwen-engine — token-level C++/CUDA inference engine for Qwen3-8B. "
      "Driven over gRPC by the flashqwen Go app; not meant to be run "
      "directly."};
  app.footer(
      "SUPPORTED MODELS\n"
      "  Qwen3-8B only (architecture Qwen3ForCausalLM). Dims are read from "
      "config.json.");

  app.add_option("--model", out.model_dir,
                 "model directory: config.json + *.safetensors")
      ->required()
      ->check(CLI::ExistingDirectory);
  app.add_option("--max-ctx", out.max_ctx, "KV / context length")
      ->capture_default_str();
  app.add_option(
         "--gpu-mem-fraction", out.gpu_mem_fraction,
         "VRAM cap; the paged KV block pool gets whatever is left under it")
      ->capture_default_str()
      ->check(CLI::Range(0.1, 1.0));
  app.add_option("--seed", out.seed, "RNG seed for sampling")
      ->capture_default_str();
  app.add_option("--address", out.address, "gRPC listen address (host:port)")
      ->capture_default_str();
  app.add_option("--slots", out.slots, "max concurrent sequences")
      ->capture_default_str();
  app.add_option("--max-waiting", out.max_waiting,
                 "admission cap on waiting requests; over it new requests are "
                 "rejected as "
                 "over-capacity (<=0 => 4*slots)")
      ->capture_default_str();
  app.add_option(
         "--token-budget", out.token_budget,
         "total tokens computed per scheduler step (max_num_batched_tokens)")
      ->capture_default_str();
  app.add_option("--prefill-chunk", out.prefill_chunk,
                 "per-request prefill chunk cap per step "
                 "(long_prefill_token_threshold)")
      ->capture_default_str();
  app.add_flag("--prefix-cache,!--no-prefix-cache", out.use_prefix_cache,
               "reuse KV of shared prompt prefixes across requests");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }
  return -1;
}

}
