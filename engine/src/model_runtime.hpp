// Qwen3 dense decoder — the GPU executor: weight management + forward pass. The declarative half
// (dims, arch) is ModelSpec (model_spec.hpp). Porting note: this is the compute half — the kernels
// and the unified forward are what you rewrite for a different model.
#pragma once
#include "model_spec.hpp"
#include "safetensors.hpp"
#include "kernels.cuh"
#include "block_pool.hpp"
#include "sampler.hpp"
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <random>

// One merged forward's inputs — a flattened mixed batch (prefill chunks + decodes together).
//   tokens[t], positions[t]  : the t-th query row's token id and its absolute logical position
//   req_index[t]             : which request that row belongs to (0..R-1)
//   block_tables[r]          : request r's physical KV block ids (row r of the uploaded table)
//   logits_rows[s]           : the flattened row index whose logits feed sampling for request s
//   sample_params[s]         : sampling params for that row (parallel to logits_rows)
struct ForwardInput {
    std::vector<int> tokens, positions, req_index, logits_rows;
    std::vector<std::vector<int>> block_tables;
    std::vector<SampleParams> sample_params;

    void clear() { tokens.clear(); positions.clear(); req_index.clear();
                   logits_rows.clear(); block_tables.clear(); sample_params.clear(); }
    int rows() const { return (int)tokens.size(); }
    int num_requests() const { return (int)block_tables.size(); }

    // --- append primitives (callers build the batch through these, not the raw vectors) ---
    // Register a request's block table; returns its row index (use it for that request's rows).
    int begin_request(std::vector<int> block_table) {
        block_tables.push_back(std::move(block_table));
        return (int)block_tables.size() - 1;
    }
    void add_row(int token, int position, int req) {   // append one query row of request `req`
        tokens.push_back(token); positions.push_back(position); req_index.push_back(req);
    }
    // flag the last appended row for sampling, with its per-request sampling params
    void mark_logits_row(SampleParams sp) {
        logits_rows.push_back((int)tokens.size() - 1);
        sample_params.push_back(sp);
    }
};

// ModelRuntime tuning knobs, resolved from CLI Args in run_engine.
struct RuntimeConfig {
    int max_ctx;          // per-sequence KV / context length bound
    int max_batch_tokens; // max query rows in one forward
    unsigned seed;        // sampling RNG seed
};

class ModelRuntime {
public:
    // Load the weights described by `spec` (safetensors, from spec.dir); cfg bounds the per-sequence
    // KV cache (max_ctx) and the query rows per forward (max_batch_tokens), and seeds the sampling RNG.
    // Allocates weights + activation scratch only — the paged KV pool lives in a separate BlockPool;
    // attach one (sized from the VRAM left after construction) before any forward call. Per-layer
    // upload progress is logged (LOG_INFO) as it goes.
    ModelRuntime(const ModelSpec& spec, const RuntimeConfig& cfg);
    void attach_pool(const BlockPool& pool) { pool_ = &pool; }   // non-owning; storage for the attention kernels
    ~ModelRuntime();

    // --- one merged forward over a flattened mixed batch (prefill chunks + decodes together) ----
    // The whole running set is computed in one pass (see ForwardInput); the per-request "last token"
    // rows then run final norm + lm_head and are sampled on the GPU (greedy or temperature, per each
    // row's in.sample_params) -> out_tokens[S]. S = in.logits_rows.size().
    void forward(const ForwardInput& in, std::vector<int>& out_tokens);

    const ModelSpec& spec() const { return spec_; }
    int max_ctx() const { return max_ctx_; }
    int max_batch() const { return MAX_DECODE_B; }                 // concurrent request cap

private:
    struct Layer {
        // Fused weights (BF16, HF layout [OUT, IN]): qkv = [q|k|v] proj stacked on OUT (q_dim+2*kv_dim
        // rows), gateup = [gate|up] stacked (2*intermediate rows). o_proj / down stay separate.
        bf16 *qkv, *o_proj, *gateup, *down;
        float *in_norm, *post_norm, *q_norm, *k_norm;
    };

    bf16*   upload_bf16(const std::string& name);   // weight, kept BF16
    bf16*   upload_concat(const std::vector<std::string>& names);  // weights stacked on OUT (rows)
    float*  upload_norm(const std::string& name);   // bf16 -> fp32
    void    precompute_rope();                       // fill cos_tab_/sin_tab_ [max_ctx, head_dim/2]

    // y[M,OUT] = x[M,IN] @ W[OUT,IN]^T via cuBLAS (BF16 in, FP32 accumulate). Ytype picks the output
    // element type: CUDA_R_16BF for activations, CUDA_R_32F for the lm_head logits.
    void gemm(const bf16* x, const bf16* W, void* y, int M, int IN, int OUT, cudaDataType_t Ytype);
    void run_layers(int T, int R, int max_qlen);
    void upload_block_tables(const std::vector<std::vector<int>>& bts);   // -> d_bt_, sets bt_stride_
    void forward_core(const ForwardInput& in);   // -> logits_ [S, vocab]

    ModelSpec spec_;
    SafeTensors st_;
    int max_ctx_      = 4096;
    int max_rows_     = 4096;   // max query rows in one forward = max(max_ctx, max_batch_tokens)
    int max_blocks_   = 0;      // ceil(max_ctx / BlockPool::BLOCK) = max block-table length per request
    int bt_stride_    = 0;      // current block-table row stride uploaded to d_bt_

    const BlockPool* pool_ = nullptr;   // paged KV storage the attention kernels read/write (non-owning)

    bf16*  embed_  = nullptr;
    float* fnorm_  = nullptr;
    bf16*  lm_head_ = nullptr;
    float* cos_tab_ = nullptr;   // [max_ctx, head_dim/2] precomputed RoPE cos (layer-independent)
    float* sin_tab_ = nullptr;
    std::vector<Layer> layers_;

    // activation scratch (BF16, sized to max_rows_ query rows); logits stay FP32 for sampling.
    // qkv_ holds the fused Q|K|V projection [max_rows, q_dim+2*kv_dim]; gateup_ the fused gate|up
    // [max_rows, 2*intermediate].
    bf16  *x_, *xb_, *xb2_, *qkv_, *attn_, *gateup_, *hmlp_;
    bf16  *xg_ = nullptr;    // gathered sampling rows [MAX_DECODE_B, H] before final norm + lm_head
    float *logits_ = nullptr;          // [S, vocab] FP32 (lm_head output -> sampling)
    int   *d_ids_, *d_pos_, *d_req_;   // per-row: token id, absolute position, request index
    int   *d_qstart_, *d_qlen_;        // per-request: flat row offset, row count (attention grouping)
    int   *d_bt_;                      // packed per-request block tables [R, bt_stride_]
    int   *d_lrows_, *d_arg_;          // sampling-row indices [S]; sampled-token output [S]
    float *d_invT_, *d_topp_, *d_u_;   // per-sampling-row: 1/temp, nucleus cutoff, uniform draw [S]
    std::vector<int>   host_bt_;       // scratch to pack padded block tables for upload
    std::vector<int>   host_qstart_, host_qlen_;          // scratch for the per-request attention grouping
    std::vector<float> host_invT_, host_topp_, host_u_;   // scratch for the per-row sampling inputs
    std::mt19937 rng_;                 // sampling RNG (per-row uniforms generated on the host)

    cudaStream_t  stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;

    std::vector<bf16*>  all_bufs_;    // for cleanup (weights)
    std::vector<float*> all_fbufs_;
};
