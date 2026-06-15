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
    // Prefill `tokens` into sequence slot `slot` at cache positions past_len..past_len+M-1
    // (past_len>0 extends an existing sequence, e.g. a new chat turn), leaving the LAST token's
    // logits in row 0 of the logits buffer (use argmax_last / copy_logits).
    void prefill(const std::vector<int>& tokens, int slot, int past_len = 0);

    int argmax_last();                          // argmax of the prefill logits (row 0), on GPU
    const std::vector<float>& copy_logits();    // prefill/decode logits row 0 -> host (sampling)

    // --- batched decode ------------------------------------------------------------------
    // One token per sequence for B sequences: sequence b feeds in_tokens[b], lives in slot
    // slots[b] with past_len[b] tokens already cached. Returns the greedy next token for each
    // sequence in out_tokens. For B==1 the row-0 logits are also left for optional sampling.
    void decode(const std::vector<int>& in_tokens, const std::vector<int>& past_len,
                const std::vector<int>& slots, std::vector<int>& out_tokens);

    const ModelConfig& config() const { return cfg_; }
    int max_ctx() const { return max_ctx_; }
    int num_slots() const { return b_max_; }                       // sequence slots in the KV pool
    int max_batch() const { return b_max_ < MAX_DECODE_B ? b_max_ : MAX_DECODE_B; }

private:
    static const int LM_CHUNK = 4;   // lm_head is run for at most this many sequences at a time
                                     // (bounds the [chunk, vocab] logits buffer)

    struct QWeight { int8_t* w = nullptr; float* scale = nullptr; };  // INT8 + per-row scale
    struct Layer {
        QWeight q_proj, k_proj, v_proj, o_proj, gate, up, down;
        float *in_norm, *post_norm, *q_norm, *k_norm;
    };

    bf16*   upload_bf16(const std::string& name);
    float*  upload_norm(const std::string& name);   // bf16 -> fp32
    QWeight upload_int8(const std::string& name);    // bf16 -> int8 + per-row scale

    void run_layers_prefill(int M, int slot);        // single-seq prefill into a KV slot
    void run_layers_decode(int B);                   // batched single-token decode

    ModelConfig cfg_;
    SafeTensors st_;
    int max_ctx_ = 4096;
    int b_max_   = 1;        // number of sequence slots backed by the KV pool

    bf16*   embed_  = nullptr;
    float*  fnorm_  = nullptr;
    QWeight lm_head_;
    std::vector<Layer> layers_;

    // KV pool: per layer [b_max, max_ctx, kv_dim], stored BF16
    std::vector<bf16*> cache_k_, cache_v_;

    // activation scratch (sized to max_ctx tokens, which also covers any decode batch)
    float *x_, *xb_, *xb2_, *q_, *k_, *v_, *attn_, *gate_, *up_, *hmlp_, *logits_;
    bf16  *xbf_ = nullptr;   // BF16 activation scratch for tensor-core prefill matmul
    bf16  *w_dq_ = nullptr;  // BF16 dequantized-weight scratch for prefill matmul
    float *part_m_, *part_l_, *part_acc_;   // flash-decoding split-K scratch (x MAX_DECODE_B)
    int   *d_ids_, *d_pos_, *d_arg_, *d_past_, *d_slot_;
    std::vector<float> host_logits_;

    cudaStream_t stream_ = nullptr;

    std::vector<bf16*>   all_bufs_;   // for cleanup
    std::vector<float*>  all_fbufs_;
    std::vector<int8_t*> all_i8_;
};
