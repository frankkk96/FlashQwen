// Qwen3 dense decoder GPU executor: weight management + forward pass.
// Dims/arch live in ModelSpec (model_spec.hpp); this is the compute half.
#pragma once
#include <cublas_v2.h>

#include <random>
#include <string>
#include <vector>

#include "block_pool.hpp"
#include "kernels.cuh"
#include "model_spec.hpp"
#include "safetensors.hpp"
#include "sampler.hpp"

// One merged forward's inputs: a flattened mixed batch (prefill chunks +
// decodes together).
//   tokens[t]/positions[t]: query row t's token id and absolute position
//   req_index[t]:           request (0..R-1) that row t belongs to
//   block_tables[r]:        request r's physical KV block ids
//   logits_rows[s]:         flattened row whose logits feed sampling for req s
//   sample_params[s]:       sampling params for that row (parallel to
//   logits_rows)
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
  int NumRows() const { return (int)tokens.size(); }
  int NumRequests() const { return (int)block_tables.size(); }

  // Append primitives: callers build the batch through these, not the raw
  // vectors. Register a request's block table; returns its row index.
  int AddRequest(std::vector<int> block_table) {
    block_tables.push_back(std::move(block_table));
    return (int)block_tables.size() - 1;
  }
  void AddRow(int token, int position,
              int req) {  // one query row of request `req`
    tokens.push_back(token);
    positions.push_back(position);
    req_index.push_back(req);
  }
  // Flag the last appended row for sampling, with its sampling params.
  void SampleLastRow(SampleParams sp) {
    logits_rows.push_back((int)tokens.size() - 1);
    sample_params.push_back(sp);
  }
};

// ModelRuntime tuning knobs, resolved from CLI Args in RunEngine.
struct RuntimeConfig {
  int max_ctx;           // per-sequence KV / context length bound
  int max_batch_tokens;  // max query rows in one forward
  unsigned seed;         // sampling RNG seed
};

class ModelRuntime {
 public:
  // Load weights from spec.dir (safetensors); cfg bounds per-sequence KV
  // (max_ctx) and query rows per forward (max_batch_tokens), and seeds the RNG.
  // Allocates weights + activation scratch only; attach a BlockPool (the paged
  // KV pool, sized from leftover VRAM) before any forward call.
  ModelRuntime(const ModelSpec& spec, const RuntimeConfig& cfg);
  void AttachPool(const BlockPool& pool) {
    pool_ = &pool;
  }  // non-owning; KV storage the attention kernels read/write
  ~ModelRuntime();

  // One merged forward over the flattened mixed batch (see ForwardInput): whole
  // running set in one pass, then per-request "last token" rows run final norm
  // + lm_head and are sampled on the GPU -> out_tokens[S], S =
  // logits_rows.size().
  void Forward(const ForwardInput& in, std::vector<int>& out_tokens);

  const ModelSpec& Spec() const { return spec_; }
  int MaxCtx() const { return max_ctx_; }
  int MaxBatch() const { return MAX_DECODE_B; }  // concurrent request cap

 private:
  struct Layer {
    // Fused weights (BF16, HF layout [OUT, IN]): qkv = [q|k|v] stacked on OUT
    // (QDim+2*KvDim rows), gateup = [gate|up] (2*intermediate rows). o_proj /
    // down stay separate.
    bf16 *qkv, *o_proj, *gateup, *down;
    float *in_norm, *post_norm, *q_norm, *k_norm;
  };

  bf16* UploadBf16(const std::string& name);  // weight, kept BF16
  bf16* UploadConcat(
      const std::vector<std::string>& names);  // weights stacked on OUT (rows)
  float* UploadNorm(const std::string& name);  // bf16 -> fp32
  void PrecomputeRope();  // fill cos_tab_/sin_tab_ [max_ctx, head_dim/2]

  void RunLayers(int T, int R, int max_qlen);
  void UploadBlockTables(
      const std::vector<std::vector<int>>& bts);  // -> d_bt_, sets bt_stride_

  ModelSpec spec_;
  SafeTensors st_;
  int max_ctx_ = 4096;
  int max_rows_ =
      4096;  // max query rows in one forward = max(max_ctx, max_batch_tokens)
  int max_blocks_ = 0;  // ceil(max_ctx / BlockPool::BLOCK) = max block-table
                        // length per request
  int bt_stride_ = 0;   // current block-table row stride uploaded to d_bt_

  const BlockPool* pool_ = nullptr;  // paged KV storage the attention kernels
                                     // read/write (non-owning)

  bf16* embed_ = nullptr;
  float* fnorm_ = nullptr;
  bf16* lm_head_ = nullptr;
  float* cos_tab_ = nullptr;  // [max_ctx, head_dim/2] precomputed RoPE cos
                              // (layer-independent)
  float* sin_tab_ = nullptr;
  std::vector<Layer> layers_;

  // Activation scratch (BF16, max_rows_ rows). qkv_ = fused Q|K|V
  // [max_rows, QDim+2*KvDim]; gateup_ = fused gate|up [max_rows,
  // 2*intermediate].
  bf16 *x_, *xb_, *xb2_, *qkv_, *attn_, *gateup_, *hmlp_;
  bf16* xg_ = nullptr;  // gathered sampling rows [MAX_DECODE_B, H] before final
                        // norm + lm_head
  float* logits_ = nullptr;  // [S, vocab] FP32 (lm_head output -> sampling)
  int *d_ids_, *d_pos_,
      *d_req_;  // per-row: token id, absolute position, request index
  int *d_qstart_,
      *d_qlen_;  // per-request: flat row offset, row count (attention grouping)
  int *d_decode_rids_, *d_prefill_rids_;  // request ids split by type (qlen==1
                                          // decode vs qlen>1 prefill)
  int n_decode_ = 0, n_prefill_ = 0,
      prefill_max_qlen_ = 1;  // attention dispatch split (per step)
  int* d_bt_;                 // packed per-request block tables [R, bt_stride_]
  int *d_lrows_, *d_arg_;  // sampling-row indices [S]; sampled-token output [S]
  float *d_invT_, *d_topp_,
      *d_u_;  // per-sampling-row: 1/temp, nucleus cutoff, uniform draw [S]
  std::vector<int> host_bt_;  // scratch to pack padded block tables for upload
  std::vector<int> host_qstart_,
      host_qlen_;  // scratch for the per-request attention grouping
  std::vector<int> host_decode_rids_,
      host_prefill_rids_;  // scratch for the attention type split
  std::vector<float> host_invT_, host_topp_,
      host_u_;        // scratch for the per-row sampling inputs
  std::mt19937 rng_;  // sampling RNG (per-row uniforms generated on the host)

  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;

  std::vector<bf16*> all_bufs_;  // for cleanup (weights)
  std::vector<float*> all_fbufs_;
};
