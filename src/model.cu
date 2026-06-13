#include "model.hpp"
#include <cmath>
#include <cstring>
#include <cstdio>

static inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f; std::memcpy(&f, &bits, 4);
    return f;
}

bf16* Model::upload_bf16(const std::string& name) {
    const TensorView& tv = st_.get(name);
    bf16* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, tv.nbytes));
    CUDA_CHECK(cudaMemcpy(d, tv.data, tv.nbytes, cudaMemcpyHostToDevice));
    all_bufs_.push_back(d);
    return d;
}

float* Model::upload_norm(const std::string& name) {
    const TensorView& tv = st_.get(name);
    int64_t n = tv.numel();
    std::vector<float> tmp(n);
    const uint16_t* src = (const uint16_t*)tv.data;   // BF16
    for (int64_t i = 0; i < n; ++i) tmp[i] = bf16_to_f32(src[i]);
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, tmp.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    all_fbufs_.push_back(d);
    return d;
}

void Model::load(const std::string& dir, int max_ctx) {
    cfg_ = ModelConfig::load(dir + "/config.json");
    max_ctx_ = ((max_ctx + 15) / 16) * 16;   // round up so WMMA tiles never read past buffers
    std::fprintf(stderr, "[model] loading weights from %s ...\n", dir.c_str());
    st_.load_dir(dir);

    embed_   = upload_bf16("model.embed_tokens.weight");
    fnorm_   = upload_norm("model.norm.weight");
    lm_head_ = upload_bf16("lm_head.weight");

    layers_.resize(cfg_.num_layers);
    for (int l = 0; l < cfg_.num_layers; ++l) {
        std::string p = "model.layers." + std::to_string(l) + ".";
        Layer& L = layers_[l];
        L.q_proj    = upload_bf16(p + "self_attn.q_proj.weight");
        L.k_proj    = upload_bf16(p + "self_attn.k_proj.weight");
        L.v_proj    = upload_bf16(p + "self_attn.v_proj.weight");
        L.o_proj    = upload_bf16(p + "self_attn.o_proj.weight");
        L.q_norm    = upload_norm(p + "self_attn.q_norm.weight");
        L.k_norm    = upload_norm(p + "self_attn.k_norm.weight");
        L.gate      = upload_bf16(p + "mlp.gate_proj.weight");
        L.up        = upload_bf16(p + "mlp.up_proj.weight");
        L.down      = upload_bf16(p + "mlp.down_proj.weight");
        L.in_norm   = upload_norm(p + "input_layernorm.weight");
        L.post_norm = upload_norm(p + "post_attention_layernorm.weight");
        if ((l + 1) % 8 == 0 || l + 1 == cfg_.num_layers)
            std::fprintf(stderr, "[model] uploaded layer %d/%d\n", l + 1, cfg_.num_layers);
    }

    // KV cache
    int kvd = cfg_.kv_dim();
    cache_k_.resize(cfg_.num_layers);
    cache_v_.resize(cfg_.num_layers);
    for (int l = 0; l < cfg_.num_layers; ++l) {
        CUDA_CHECK(cudaMalloc(&cache_k_[l], (size_t)max_ctx_ * kvd * sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&cache_v_[l], (size_t)max_ctx_ * kvd * sizeof(bf16)));
    }

    // activation scratch
    int H = cfg_.hidden_size, QD = cfg_.q_dim(), I = cfg_.intermediate, V = cfg_.vocab_size;
    auto fmalloc = [&](float** p, size_t n) { CUDA_CHECK(cudaMalloc(p, n * sizeof(float))); };
    fmalloc(&x_,    (size_t)max_ctx_ * H);
    fmalloc(&xb_,   (size_t)max_ctx_ * H);
    fmalloc(&xb2_,  (size_t)max_ctx_ * H);
    fmalloc(&q_,    (size_t)max_ctx_ * QD);
    fmalloc(&k_,    (size_t)max_ctx_ * kvd);
    fmalloc(&v_,    (size_t)max_ctx_ * kvd);
    fmalloc(&attn_, (size_t)max_ctx_ * QD);
    fmalloc(&gate_, (size_t)max_ctx_ * I);
    fmalloc(&up_,   (size_t)max_ctx_ * I);
    fmalloc(&hmlp_, (size_t)max_ctx_ * I);
    fmalloc(&logits_, (size_t)V);
    // BF16 activation scratch for the tensor-core prefill GEMM (largest IN is `I`).
    CUDA_CHECK(cudaMalloc(&xbf_, (size_t)max_ctx_ * I * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_ids_, max_ctx_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pos_, max_ctx_ * sizeof(int)));
    host_logits_.resize(V);

    size_t freeb, totalb;
    CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
    std::fprintf(stderr, "[model] ready. GPU mem: %.1f GB used / %.1f GB total\n",
                 (totalb - freeb) / 1e9, totalb / 1e9);
}

const std::vector<float>& Model::forward(const std::vector<int>& tokens, int past_len) {
    int M = (int)tokens.size();
    const ModelConfig& c = cfg_;
    int H = c.hidden_size, QD = c.q_dim(), KVD = c.kv_dim(), I = c.intermediate;
    int nH = c.num_heads, nKV = c.num_kv_heads, hd = c.head_dim;
    float eps = c.rms_eps, scale = 1.0f / std::sqrt((float)hd);
    cudaStream_t s = 0;

    // upload ids + positions
    CUDA_CHECK(cudaMemcpy(d_ids_, tokens.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    std::vector<int> pos(M);
    for (int i = 0; i < M; ++i) pos[i] = past_len + i;
    CUDA_CHECK(cudaMemcpy(d_pos_, pos.data(), M * sizeof(int), cudaMemcpyHostToDevice));

    launch_embed(d_ids_, embed_, x_, M, H, s);

    for (int l = 0; l < c.num_layers; ++l) {
        Layer& L = layers_[l];

        // --- attention block ---
        launch_rmsnorm(x_, L.in_norm, xb_, M, H, eps, s);
        launch_matmul(xb_, L.q_proj, q_, M, H, QD,  xbf_, s);
        launch_matmul(xb_, L.k_proj, k_, M, H, KVD, xbf_, s);
        launch_matmul(xb_, L.v_proj, v_, M, H, KVD, xbf_, s);

        // per-head QK RMSNorm (Qwen3), then RoPE
        launch_rmsnorm(q_, L.q_norm, q_, M * nH,  hd, eps, s);
        launch_rmsnorm(k_, L.k_norm, k_, M * nKV, hd, eps, s);
        launch_rope(q_, d_pos_, M, nH,  hd, c.rope_theta, s);
        launch_rope(k_, d_pos_, M, nKV, hd, c.rope_theta, s);

        // append K/V to cache (rows are contiguous [M, kv_dim]); FP32 -> BF16
        launch_to_bf16(k_, cache_k_[l] + (size_t)past_len * KVD, M * KVD, s);
        launch_to_bf16(v_, cache_v_[l] + (size_t)past_len * KVD, M * KVD, s);

        launch_attention(q_, cache_k_[l], cache_v_[l], attn_, M, nH, nKV, hd, past_len, scale, s);
        launch_matmul(attn_, L.o_proj, xb2_, M, QD, H, xbf_, s);
        launch_add(x_, xb2_, M * H, s);

        // --- MLP block (SwiGLU) ---
        launch_rmsnorm(x_, L.post_norm, xb_, M, H, eps, s);
        launch_matmul(xb_, L.gate, gate_, M, H, I, xbf_, s);
        launch_matmul(xb_, L.up,   up_,   M, H, I, xbf_, s);
        launch_silu_mul(gate_, up_, hmlp_, M * I, s);
        launch_matmul(hmlp_, L.down, xb2_, M, I, H, xbf_, s);
        launch_add(x_, xb2_, M * H, s);
    }

    // only the last token's logits are needed
    float* xlast = x_ + (size_t)(M - 1) * H;
    launch_rmsnorm(xlast, fnorm_, xb_, 1, H, eps, s);
    launch_matmul(xb_, lm_head_, logits_, 1, H, c.vocab_size, xbf_, s);

    CUDA_CHECK(cudaMemcpy(host_logits_.data(), logits_, c.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    return host_logits_;
}

Model::~Model() {
    for (auto p : all_bufs_)  if (p) cudaFree(p);
    for (auto p : all_fbufs_) if (p) cudaFree(p);
    for (auto p : cache_k_)   if (p) cudaFree(p);
    for (auto p : cache_v_)   if (p) cudaFree(p);
    cudaFree(x_); cudaFree(xb_); cudaFree(xb2_); cudaFree(q_); cudaFree(k_); cudaFree(v_);
    cudaFree(attn_); cudaFree(gate_); cudaFree(up_); cudaFree(hmlp_); cudaFree(logits_);
    cudaFree(xbf_);
    cudaFree(d_ids_); cudaFree(d_pos_);
}
