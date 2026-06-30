#pragma once
#include <string>

namespace fq {

// Declarative model spec (dims + arch) parsed from config.json; no GPU/weights.
// Drives KV-pool sizing, scheduler limits, and CLI validation. Load() is
// lenient (a missing/invalid file or unknown arch leaves it unsupported, dims
// default); gate use with Supported(). ModelRuntime is the compute half.
struct ModelSpec {
  int hidden_size = 4096;
  int num_layers = 36;
  int num_heads = 32;
  int num_kv_heads = 8;
  int head_dim = 128;
  int intermediate = 12288;
  int vocab_size = 151936;
  float rms_eps = 1e-6f;
  float rope_theta = 1000000.0f;
  std::string arch;
  std::string dir;

  int QDim() const { return num_heads * head_dim; }
  int KvDim() const { return num_kv_heads * head_dim; }
  bool Supported() const { return arch == "Qwen3ForCausalLM"; }

  static ModelSpec Load(const std::string& dir);
};

}
