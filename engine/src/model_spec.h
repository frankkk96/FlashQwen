#pragma once
#include <string>

namespace fq {

// Compile-time description of the one model this engine binary is built for
// (Qwen3-8B). The shape below is the single source of truth: kernels bake it
// in and buffer/loop sizing reads it; derived dims (q/kv/qkv widths, GQA
// group) and kernel tiling are computed at their point of use.
// Load() parses config.json only to VALIDATE a checkpoint against these
// constants (architecture and every dim) — it throws on any mismatch and never
// overwrites them. dir is the only per-load runtime field. ModelRuntime is the
// compute half.
struct ModelSpec {
  static constexpr const char* kArch = "Qwen3ForCausalLM";

  // Model shape (validated against config.json by Load).
  static constexpr int kHeadDim = 128;
  static constexpr int kNumHeads = 32;
  static constexpr int kNumKvHeads = 8;
  static constexpr int kHiddenSize = 4096;
  static constexpr int kNumLayers = 36;
  static constexpr int kIntermediate = 12288;
  static constexpr int kVocabSize = 151936;
  static constexpr float kRmsEps = 1e-6f;
  static constexpr float kRopeTheta = 1000000.0f;

  // Paged-KV page size, shared by KvStore / BlockAllocator / attn kernels.
  static constexpr int kKvBlock = 16;

  std::string dir;

  static ModelSpec Load(const std::string& dir);
};

}  // namespace fq
