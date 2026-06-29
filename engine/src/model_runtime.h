#pragma once
#include <cublas_v2.h>

#include <random>
#include <string>
#include <vector>

#include "kernels.cuh"
#include "kv_store.h"
#include "model_spec.h"
#include "safetensors.h"
#include "sampler.h"

// Owning RAII handle for a cudaMalloc'd device buffer. Move-only; frees on
// destruction (so every buffer has exactly one owner — no central free list).
// The type marks the memory as device-resident and there is no operator*, so it
// can't be dereferenced on the host; pass Get() to kernels / cuda APIs.
template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    p_ = static_cast<T*>(p);
  }
  ~DeviceBuffer() { Free(); }
  DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      Free();
      p_ = o.p_;
      o.p_ = nullptr;
    }
    return *this;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* Get() const { return p_; }  // raw device pointer for kernels / cuda APIs

 private:
  void Free() {
    if (p_) cudaFree(p_);
    p_ = nullptr;
  }
  T* p_ = nullptr;
};

// One merged forward's inputs: a flattened mixed batch (prefill chunks +
// decodes together). tokens/positions/req_index are parallel per-row arrays;
// block_tables[r] is request r's KV blocks; logits_rows[s] / sample_params[s]
// pick the rows that feed sampling. Callers build it through the Add* methods.
struct ForwardInput {
  std::vector<int> tokens, positions, req_index, logits_rows;
  std::vector<std::vector<int>> block_tables;
  std::vector<SampleParams> sample_params;

  void Clear() {
    tokens.clear();
    positions.clear();
    req_index.clear();
    logits_rows.clear();
    block_tables.clear();
    sample_params.clear();
  }
  int NumRows() const { return static_cast<int>(tokens.size()); }

  int AddRequest(std::vector<int> block_table) {
    block_tables.push_back(std::move(block_table));
    return static_cast<int>(block_tables.size()) - 1;
  }
  void AddRow(int token, int position, int req) {
    tokens.push_back(token);
    positions.push_back(position);
    req_index.push_back(req);
  }
  void SampleLastRow(SampleParams sp) {
    logits_rows.push_back(static_cast<int>(tokens.size()) - 1);
    sample_params.push_back(sp);
  }
};

// Qwen3 dense decoder GPU executor (the compute half; dims/arch live in
// ModelSpec). Loads weights + activation scratch from spec.dir and borrows the
// paged KV store (non-owning) for the lifetime of the run. Forward() runs
// the whole flattened batch in one pass, then runs final norm + lm_head on the
// per-request last-token rows and samples them on the GPU ->
// out_tokens[logits_rows.size()].
class ModelRuntime {
 public:
  // Loads weights + activation scratch. Does NOT take the KvStore: the KV pool
  // is sized from the VRAM left AFTER this constructor runs, so it must be
  // built afterwards and wired in via AttachKvStore() before the first Forward.
  ModelRuntime(const ModelSpec& spec, int max_ctx, int slots, int token_budget,
               unsigned seed);
  ~ModelRuntime();

  // Attach the KV pool (built after this runtime so it gets the real leftover
  // VRAM). Must be called before Forward().
  void AttachKvStore(const KvStore& store) { store_ = &store; }

  void Forward(const ForwardInput& in, std::vector<int>& out_tokens);

 private:
  // Fused weights (BF16, HF layout [OUT, IN]): d_qkv = [q|k|v] and d_gateup =
  // [gate|up] stacked on OUT; d_o_proj / d_down stay separate.
  struct Layer {
    DeviceBuffer<bf16> d_qkv, d_o_proj, d_gateup, d_down;
    DeviceBuffer<float> d_in_norm, d_post_norm, d_q_norm, d_k_norm;
  };

  DeviceBuffer<bf16> UploadBf16(const std::string& name);
  DeviceBuffer<bf16> UploadBf16s(const std::vector<std::string>& names);
  DeviceBuffer<float> UploadFp32(const std::string& name);
  void PrecomputeRope();
  void RunLayers(const ForwardInput& in);
  void RunHeadAndSample(const ForwardInput& in, std::vector<int>& out_tokens);
  void UploadInputs(const ForwardInput& in);
  void GroupRequests(const ForwardInput& in);

  ModelSpec spec_;
  SafeTensors st_;
  int max_ctx_ = 0;
  int slots_ = 0;
  int max_rows_ = 0;
  int max_blocks_ = 0;
  int bt_stride_ = 0;
  const KvStore* store_ = nullptr;

  DeviceBuffer<bf16> d_embed_;
  DeviceBuffer<float> d_fnorm_;
  DeviceBuffer<bf16> d_lm_head_;
  DeviceBuffer<float> d_cos_tab_, d_sin_tab_;  // precomputed RoPE tables
  std::vector<Layer> layers_;

  // Activation scratch (BF16, max_rows_ rows); sampling buffers (slots_).
  DeviceBuffer<bf16> d_x_, d_xb_, d_xb2_, d_qkv_, d_attn_, d_gateup_, d_hmlp_;
  DeviceBuffer<bf16> d_xg_;
  DeviceBuffer<float> d_logits_;
  // FlashDecoding split partials (m, l, acc), preallocated once to the
  // worst-case decode batch; the decode attention launchers write/read them.
  DeviceBuffer<float> d_dec_pm_, d_dec_pl_, d_dec_pa_;
  DeviceBuffer<int> d_ids_, d_pos_, d_req_;
  DeviceBuffer<int> d_qstart_, d_qlen_;
  DeviceBuffer<int> d_decode_rids_, d_prefill_rids_;
  int n_decode_ = 0, n_prefill_ = 0, prefill_max_qlen_ = 1;
  DeviceBuffer<int> d_bt_;
  DeviceBuffer<int> d_lrows_, d_arg_;
  DeviceBuffer<float> d_invT_, d_topp_, d_u_;
  std::vector<int> h_bt_;
  std::vector<int> h_qstart_, h_qlen_;
  std::vector<int> h_decode_rids_, h_prefill_rids_;
  std::vector<float> h_invT_, h_topp_, h_u_;
  std::mt19937 rng_;

  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
};
