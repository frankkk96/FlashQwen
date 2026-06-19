// Qwen3 dense decoder — the GPU executor: weight management + forward pass. The declarative half
// (dims, arch) is ModelSpec (model_spec.hpp). Porting note: this is the compute half — the kernels
// and the unified forward are what you rewrite for a different model.
#pragma once
#include "model_spec.hpp"
#include "safetensors.hpp"
#include "kernels.cuh"
#include "block_pool.hpp"
#include <vector>
#include <string>
#include <functional>

// One merged forward's inputs — a flattened mixed batch (prefill chunks + decodes together).
//   tokens[t], positions[t]  : the t-th query row's token id and its absolute logical position
//   req_index[t]             : which request that row belongs to (0..R-1)
//   block_tables[r]          : request r's physical KV block ids (row r of the uploaded table)
//   logits_rows[s]           : the flattened row index whose logits feed sampling for request s
struct ForwardInput {
    std::vector<int> tokens, positions, req_index, logits_rows;
    std::vector<std::vector<int>> block_tables;

    void clear() { tokens.clear(); positions.clear(); req_index.clear();
                   logits_rows.clear(); block_tables.clear(); }
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
    void mark_logits_row() { logits_rows.push_back((int)tokens.size() - 1); }   // flag last row for sampling
};

class ModelRuntime {
public:
    // Load the weights described by `spec` (safetensors, from spec.dir). max_ctx bounds the
    // per-sequence KV cache; max_batch_tokens bounds the number of query rows in one forward.
    // Allocates weights + activation scratch only — the paged KV pool lives in a separate
    // BlockPool; attach one (sized from the VRAM left after construction) before any forward call.
    // on_progress (may be empty) is called as each transformer layer's weights upload, with
    // (layers_done, layers_total), for a startup progress bar.
    using ProgressFn = std::function<void(int done, int total)>;
    ModelRuntime(const ModelSpec& spec, int max_ctx, int max_batch_tokens, ProgressFn on_progress = {});
    void attach_pool(const BlockPool& pool) { pool_ = &pool; }   // non-owning; storage for the attention kernels
    ~ModelRuntime();

    // --- one merged forward over a flattened mixed batch (prefill chunks + decodes together) ----
    // The whole running set is computed in one pass (see ForwardInput). forward(): greedy argmax per
    // sampling row, on the GPU -> out_tokens[S]. forward_logits_host(): run the same forward and copy
    // the [S, vocab] logits to the host (row-major) for per-request sampling; the returned pointer is
    // valid until the next call. S = in.logits_rows.size().
    void forward(const ForwardInput& in, std::vector<int>& out_tokens);
    const float* forward_logits_host(const ForwardInput& in);

    const ModelSpec& spec() const { return spec_; }
    int max_ctx() const { return max_ctx_; }
    int max_batch() const { return MAX_DECODE_B; }                 // concurrent request cap

private:
    struct QWeight { int8_t* w = nullptr; float* scale = nullptr; };  // INT8 + per-row scale
    struct Layer {
        QWeight q_proj, k_proj, v_proj, o_proj, gate, up, down;
        float *in_norm, *post_norm, *q_norm, *k_norm;
    };

    bf16*   upload_bf16(const std::string& name);
    float*  upload_norm(const std::string& name);   // bf16 -> fp32
    QWeight upload_int8(const std::string& name);    // bf16 -> int8 + per-row scale

    // Pick the matmul kernel for the layer stack: pure-decode steps (1 token per request) use the
    // batched INT8 GEMV (reads weights once, amortised across the batch — the decode-throughput
    // win); steps that include any prefill rows use the WMMA path (dequantise to BF16, tensor cores).
    void mm(const float* x, const QWeight& w, float* y, int M, int IN, int OUT, bool pure_decode);
    void run_layers(int T, bool pure_decode);
    void upload_block_tables(const std::vector<std::vector<int>>& bts);   // -> d_bt_, sets bt_stride_
    void forward_core(const ForwardInput& in);   // -> logits_ [S, vocab]

    ModelSpec spec_;
    SafeTensors st_;
    int max_ctx_      = 4096;
    int max_rows_     = 4096;   // max query rows in one forward = max(max_ctx, max_batch_tokens)
    int max_blocks_   = 0;      // ceil(max_ctx / BlockPool::BLOCK) = max block-table length per request
    int bt_stride_    = 0;      // current block-table row stride uploaded to d_bt_

    const BlockPool* pool_ = nullptr;   // paged KV storage the attention kernels read/write (non-owning)

    bf16*   embed_  = nullptr;
    float*  fnorm_  = nullptr;
    QWeight lm_head_;
    std::vector<Layer> layers_;

    // activation scratch (sized to max_rows_ query rows)
    float *x_, *xb_, *xb2_, *q_, *k_, *v_, *attn_, *gate_, *up_, *hmlp_, *logits_;
    float *xg_ = nullptr;    // gathered sampling rows [MAX_DECODE_B, H] before final norm + lm_head
    bf16  *xbf_ = nullptr;   // BF16 activation scratch for the tensor-core matmul
    bf16  *w_dq_ = nullptr;  // BF16 dequantized-weight scratch for the tensor-core matmul
    int   *d_ids_, *d_pos_, *d_req_;   // per-row: token id, absolute position, request index
    int   *d_bt_;                      // packed per-request block tables [R, bt_stride_]
    int   *d_lrows_, *d_arg_;          // sampling-row indices [S]; greedy argmax output [S]
    std::vector<int>   host_bt_;       // scratch to pack padded block tables for upload
    std::vector<float> host_logits_;

    cudaStream_t stream_ = nullptr;

    std::vector<bf16*>   all_bufs_;   // for cleanup
    std::vector<float*>  all_fbufs_;
    std::vector<int8_t*> all_i8_;
};
