// Qwen3 dense decoder: weight management + forward pass on the GPU.
#pragma once
#include "config.hpp"
#include "safetensors.hpp"
#include "kernels.cuh"
#include <vector>
#include <string>

class Model {
public:
    // Loads weights from <dir> (config.json + safetensors). max_ctx bounds the KV cache
    // and the prefill batch size.
    void load(const std::string& dir, int max_ctx);
    ~Model();

    // Run M tokens starting at absolute position past_len; KV cache is updated in place.
    // Returns the logits (size vocab) for the LAST token, copied to host.
    const std::vector<float>& forward(const std::vector<int>& tokens, int past_len);

    const ModelConfig& config() const { return cfg_; }
    int max_ctx() const { return max_ctx_; }

private:
    struct Layer {
        bf16 *q_proj, *k_proj, *v_proj, *o_proj;
        bf16 *gate, *up, *down;
        float *in_norm, *post_norm, *q_norm, *k_norm;
    };

    bf16*  upload_bf16(const std::string& name);
    float* upload_norm(const std::string& name);   // bf16 -> fp32

    ModelConfig cfg_;
    SafeTensors st_;
    int max_ctx_ = 4096;

    bf16*  embed_  = nullptr;
    float* fnorm_  = nullptr;
    bf16*  lm_head_= nullptr;
    std::vector<Layer> layers_;

    // KV cache: per layer [max_ctx, kv_dim], stored BF16
    std::vector<bf16*> cache_k_, cache_v_;

    // activation scratch (sized to max_ctx tokens)
    float *x_, *xb_, *xb2_, *q_, *k_, *v_, *attn_, *gate_, *up_, *hmlp_, *logits_;
    bf16  *xbf_ = nullptr;   // BF16 activation scratch for tensor-core prefill matmul
    int   *d_ids_, *d_pos_;
    std::vector<float> host_logits_;

    std::vector<bf16*> all_bufs_;   // for cleanup
    std::vector<float*> all_fbufs_;
};
