// Qwen3 dense decoder: weight management + forward pass on the GPU.
#pragma once
#include "config.hpp"
#include "safetensors.hpp"
#include "kernels.cuh"
#include <vector>
#include <string>

class Model {
public:
    // Loads weights from <dir> (config.json + safetensors). max_ctx bounds the per-sequence KV
    // cache and the prefill batch size. gpu_mem_fraction caps total VRAM use; whatever is left
    // after weights + activations becomes the KV pool, which fixes the number of sequence slots.
    void load(const std::string& dir, int max_ctx, float gpu_mem_fraction);
    ~Model();

    // --- single-sequence prefill ---------------------------------------------------------
    // Prefill `tokens` into the sequence whose paged KV is described by `block_table` (physical
    // block ids), at logical positions past_len..past_len+M-1 (past_len>0 extends an existing
    // sequence). The caller owns the block table and must have allocated enough blocks to cover
    // past_len+M tokens. Leaves the LAST token's logits in row 0 (use argmax_last / copy_logits).
    void prefill(const std::vector<int>& tokens, const std::vector<int>& block_table, int past_len = 0);

    int argmax_last();                          // argmax of the prefill logits (row 0), on GPU
    const std::vector<float>& copy_logits();    // prefill/decode logits row 0 -> host (sampling)

    // --- batched decode ------------------------------------------------------------------
    // One token per sequence for B sequences: sequence b feeds in_tokens[b], has past_len[b]
    // tokens cached in the paged blocks listed by block_tables[b]. decode() returns the greedy
    // next token per sequence (argmax on the GPU). decode_logits_host() instead runs the same
    // forward and copies the full [B, vocab] logits to the host (row-major) for per-sequence
    // sampling; the returned pointer is valid until the next decode/prefill call.
    void decode(const std::vector<int>& in_tokens, const std::vector<int>& past_len,
                const std::vector<std::vector<int>>& block_tables, std::vector<int>& out_tokens);
    const float* decode_logits_host(const std::vector<int>& in_tokens, const std::vector<int>& past_len,
                                    const std::vector<std::vector<int>>& block_tables);

    const ModelConfig& config() const { return cfg_; }
    int max_ctx() const { return max_ctx_; }
    int block_size() const { return block_; }                      // tokens per KV block (page)
    int num_blocks() const { return num_blocks_; }                 // physical blocks in the pool
    int max_blocks_per_seq() const { return max_blocks_; }         // ceil(max_ctx / block_size)
    int max_batch() const { return MAX_DECODE_B; }                 // concurrent decode cap

private:
    struct QWeight { int8_t* w = nullptr; float* scale = nullptr; };  // INT8 + per-row scale
    struct Layer {
        QWeight q_proj, k_proj, v_proj, o_proj, gate, up, down;
        float *in_norm, *post_norm, *q_norm, *k_norm;
    };

    bf16*   upload_bf16(const std::string& name);
    float*  upload_norm(const std::string& name);   // bf16 -> fp32
    QWeight upload_int8(const std::string& name);    // bf16 -> int8 + per-row scale

    void run_layers_prefill(int M);                  // single-seq prefill (block table in d_bt_ row 0)
    void run_layers_decode(int B);                   // batched single-token decode
    void upload_block_tables(const std::vector<std::vector<int>>& bts);   // -> d_bt_, returns stride in bt_stride_
    void decode_forward(const std::vector<int>& in_tokens, const std::vector<int>& past_len,
                        const std::vector<std::vector<int>>& block_tables);  // -> logits_ [B, vocab]

    ModelConfig cfg_;
    SafeTensors st_;
    int max_ctx_     = 4096;
    int block_       = 16;   // tokens per KV block (page)
    int num_blocks_  = 0;    // physical blocks in the paged KV pool
    int max_blocks_  = 0;    // ceil(max_ctx / block_) = max block-table length per sequence
    int bt_stride_   = 0;    // current block-table row stride uploaded to d_bt_

    bf16*   embed_  = nullptr;
    float*  fnorm_  = nullptr;
    QWeight lm_head_;
    std::vector<Layer> layers_;

    // Paged KV pool: per layer [num_blocks, block_, kv_dim], stored BF16
    std::vector<bf16*> cache_k_, cache_v_;

    // activation scratch (sized to max_ctx tokens, which also covers any decode batch)
    float *x_, *xb_, *xb2_, *q_, *k_, *v_, *attn_, *gate_, *up_, *hmlp_, *logits_;
    bf16  *xbf_ = nullptr;   // BF16 activation scratch for tensor-core prefill matmul
    bf16  *w_dq_ = nullptr;  // BF16 dequantized-weight scratch for prefill matmul
    float *part_m_, *part_l_, *part_acc_;   // flash-decoding split-K scratch (x MAX_DECODE_B)
    int   *d_ids_, *d_pos_, *d_arg_, *d_past_;
    int   *d_bt_, *d_iota_, *d_zero_;       // block tables; decode bt_row (0..B-1); prefill bt_row (all 0)
    std::vector<int>   host_bt_;            // scratch to pack padded block tables for upload
    std::vector<float> host_logits_;

    cudaStream_t stream_ = nullptr;

    std::vector<bf16*>   all_bufs_;   // for cleanup
    std::vector<float*>  all_fbufs_;
    std::vector<int8_t*> all_i8_;
};
