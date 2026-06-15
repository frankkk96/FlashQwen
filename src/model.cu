#include "model.hpp"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <thread>
#include <algorithm>

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

// Quantize a [OUT, IN] BF16 weight to INT8 with a symmetric per-row (per-output-channel)
// scale: scale[r] = max|W[r,:]| / 127. Done on the host, parallelized across rows.
Model::QWeight Model::upload_int8(const std::string& name) {
    const TensorView& tv = st_.get(name);
    int OUT = (int)tv.shape[0], IN = (int)tv.shape[1];
    long n = (long)OUT * IN;
    const uint16_t* src = (const uint16_t*)tv.data;
    std::vector<int8_t> q(n);
    std::vector<float> scale(OUT);

    auto worker = [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            const uint16_t* row = src + (long)r * IN;
            float maxabs = 0.f;
            for (int i = 0; i < IN; ++i) maxabs = std::max(maxabs, std::fabs(bf16_to_f32(row[i])));
            float sc = maxabs / 127.0f; if (sc == 0.f) sc = 1.f;
            float inv = 1.0f / sc;
            int8_t* qr = q.data() + (long)r * IN;
            for (int i = 0; i < IN; ++i) {
                int v = (int)lrintf(bf16_to_f32(row[i]) * inv);
                qr[i] = (int8_t)std::max(-127, std::min(127, v));
            }
            scale[r] = sc;
        }
    };
    int nt = std::max(1u, std::thread::hardware_concurrency());
    int chunk = (OUT + nt - 1) / nt;
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; ++t) {
        int a = t * chunk, b = std::min(OUT, a + chunk);
        if (a < b) ts.emplace_back(worker, a, b);
    }
    for (auto& th : ts) th.join();

    QWeight qw;
    CUDA_CHECK(cudaMalloc(&qw.w, n * sizeof(int8_t)));
    CUDA_CHECK(cudaMemcpy(qw.w, q.data(), n * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&qw.scale, OUT * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(qw.scale, scale.data(), OUT * sizeof(float), cudaMemcpyHostToDevice));
    all_i8_.push_back(qw.w);
    all_fbufs_.push_back(qw.scale);
    return qw;
}

void Model::load(const std::string& dir, int max_ctx, float gpu_mem_fraction) {
    cfg_ = ModelConfig::load(dir + "/config.json");
    max_ctx_ = ((max_ctx + 15) / 16) * 16;   // round up so WMMA tiles never read past buffers
    std::fprintf(stderr, "[model] loading weights from %s ...\n", dir.c_str());
    st_.load_dir(dir);

    embed_   = upload_bf16("model.embed_tokens.weight");   // gather, kept BF16
    fnorm_   = upload_norm("model.norm.weight");
    lm_head_ = upload_int8("lm_head.weight");

    layers_.resize(cfg_.num_layers);
    for (int l = 0; l < cfg_.num_layers; ++l) {
        std::string p = "model.layers." + std::to_string(l) + ".";
        Layer& L = layers_[l];
        L.q_proj    = upload_int8(p + "self_attn.q_proj.weight");
        L.k_proj    = upload_int8(p + "self_attn.k_proj.weight");
        L.v_proj    = upload_int8(p + "self_attn.v_proj.weight");
        L.o_proj    = upload_int8(p + "self_attn.o_proj.weight");
        L.q_norm    = upload_norm(p + "self_attn.q_norm.weight");
        L.k_norm    = upload_norm(p + "self_attn.k_norm.weight");
        L.gate      = upload_int8(p + "mlp.gate_proj.weight");
        L.up        = upload_int8(p + "mlp.up_proj.weight");
        L.down      = upload_int8(p + "mlp.down_proj.weight");
        L.in_norm   = upload_norm(p + "input_layernorm.weight");
        L.post_norm = upload_norm(p + "post_attention_layernorm.weight");
        if ((l + 1) % 8 == 0 || l + 1 == cfg_.num_layers)
            std::fprintf(stderr, "[model] uploaded layer %d/%d\n", l + 1, cfg_.num_layers);
    }

    // activation scratch (sized to max_ctx tokens; a decode batch is at most MAX_DECODE_B <= that)
    int H = cfg_.hidden_size, QD = cfg_.q_dim(), I = cfg_.intermediate, V = cfg_.vocab_size;
    int kvd = cfg_.kv_dim(), nh = cfg_.num_heads, hd = cfg_.head_dim;
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
    fmalloc(&logits_, (size_t)MAX_DECODE_B * V);   // [B, vocab] logits for the whole decode batch
    // BF16 activation scratch for the tensor-core prefill GEMM (largest IN is `I`).
    CUDA_CHECK(cudaMalloc(&xbf_, (size_t)max_ctx_ * I * sizeof(bf16)));
    // BF16 dequantized-weight scratch for prefill (largest matmul weight is I*H).
    CUDA_CHECK(cudaMalloc(&w_dq_, (size_t)I * H * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_ids_,  max_ctx_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pos_,  max_ctx_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_slot_, MAX_DECODE_B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_arg_,  MAX_DECODE_B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_past_, sizeof(int)));
    // flash-decoding split-K scratch, sized for a full decode batch
    CUDA_CHECK(cudaMalloc(&part_m_,   (size_t)MAX_DECODE_B * nh * ATTN_SPLITS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&part_l_,   (size_t)MAX_DECODE_B * nh * ATTN_SPLITS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&part_acc_, (size_t)MAX_DECODE_B * nh * ATTN_SPLITS * hd * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    host_logits_.resize((size_t)MAX_DECODE_B * V);

    // Everything except the KV pool is now allocated. Give the pool whatever VRAM is left under
    // the gpu_mem_fraction cap; that fixes how many sequence slots we can serve.
    size_t freeb, totalb;
    CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
    size_t used = totalb - freeb;
    size_t budget = (size_t)(totalb * gpu_mem_fraction);
    size_t kv_avail = budget > used ? budget - used : 0;
    size_t per_seq = (size_t)cfg_.num_layers * 2 * max_ctx_ * kvd * sizeof(bf16);
    b_max_ = (int)(kv_avail / per_seq);
    if (b_max_ < 1) {
        std::fprintf(stderr, "error: not enough VRAM for even one KV slot at max_ctx=%d "
                     "(need %.1f GB, have %.1f GB under the %.0f%% cap).\n",
                     max_ctx_, per_seq / 1e9, kv_avail / 1e9, gpu_mem_fraction * 100);
        std::exit(1);
    }
    // A single decode step handles at most MAX_DECODE_B sequences, and every running sequence
    // decodes each step, so more slots than that can never be used — don't allocate them.
    if (b_max_ > MAX_DECODE_B) b_max_ = MAX_DECODE_B;

    cache_k_.resize(cfg_.num_layers);
    cache_v_.resize(cfg_.num_layers);
    for (int l = 0; l < cfg_.num_layers; ++l) {
        CUDA_CHECK(cudaMalloc(&cache_k_[l], (size_t)b_max_ * max_ctx_ * kvd * sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&cache_v_[l], (size_t)b_max_ * max_ctx_ * kvd * sizeof(bf16)));
    }

    CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
    std::fprintf(stderr, "[model] ready. %d KV slots (max_ctx=%d). GPU mem: %.1f GB used / %.1f GB total\n",
                 b_max_, max_ctx_, (totalb - freeb) / 1e9, totalb / 1e9);
}

// --- single-sequence prefill: M tokens into KV slot `slot` at positions 0..M-1 -----------
// Reuses the single-sequence kernels; the only difference from a fresh decode is that the
// store/attention kernels point at this slot's region of the KV pool (d_past_ == 0).
void Model::run_layers_prefill(int M, int slot) {
    const ModelConfig& c = cfg_;
    int H = c.hidden_size, QD = c.q_dim(), KVD = c.kv_dim(), I = c.intermediate;
    int nH = c.num_heads, nKV = c.num_kv_heads, hd = c.head_dim;
    float eps = c.rms_eps, scale = 1.0f / std::sqrt((float)hd);
    cudaStream_t s = stream_;

    launch_embed(d_ids_, embed_, x_, M, H, s);

    for (int l = 0; l < c.num_layers; ++l) {
        Layer& L = layers_[l];
        bf16* ck = cache_k_[l] + (size_t)slot * max_ctx_ * KVD;   // this slot's KV region
        bf16* cv = cache_v_[l] + (size_t)slot * max_ctx_ * KVD;

        launch_rmsnorm(x_, L.in_norm, xb_, M, H, eps, s);
        launch_matmul(xb_, L.q_proj.w, L.q_proj.scale, q_, M, H, QD,  xbf_, w_dq_, s);
        launch_matmul(xb_, L.k_proj.w, L.k_proj.scale, k_, M, H, KVD, xbf_, w_dq_, s);
        launch_matmul(xb_, L.v_proj.w, L.v_proj.scale, v_, M, H, KVD, xbf_, w_dq_, s);

        launch_rmsnorm(q_, L.q_norm, q_, M * nH,  hd, eps, s);
        launch_rmsnorm(k_, L.k_norm, k_, M * nKV, hd, eps, s);
        launch_rope(q_, d_pos_, M, nH,  hd, c.rope_theta, s);
        launch_rope(k_, d_pos_, M, nKV, hd, c.rope_theta, s);

        launch_store_kv(k_, ck, d_past_, KVD, M, s);   // *d_past_ == 0
        launch_store_kv(v_, cv, d_past_, KVD, M, s);
        launch_attention(q_, ck, cv, attn_, M, nH, nKV, hd, d_past_, scale, s);

        launch_matmul(attn_, L.o_proj.w, L.o_proj.scale, xb2_, M, QD, H, xbf_, w_dq_, s);
        launch_add(x_, xb2_, M * H, s);

        launch_rmsnorm(x_, L.post_norm, xb_, M, H, eps, s);
        launch_matmul(xb_, L.gate.w, L.gate.scale, gate_, M, H, I, xbf_, w_dq_, s);
        launch_matmul(xb_, L.up.w,   L.up.scale,   up_,   M, H, I, xbf_, w_dq_, s);
        launch_silu_mul(gate_, up_, hmlp_, M * I, s);
        launch_matmul(hmlp_, L.down.w, L.down.scale, xb2_, M, I, H, xbf_, w_dq_, s);
        launch_add(x_, xb2_, M * H, s);
    }

    // only the last token's logits are needed (GEMV) -> logits_ row 0
    float* xlast = x_ + (size_t)(M - 1) * H;
    launch_rmsnorm(xlast, fnorm_, xb_, 1, H, eps, s);
    launch_matmul(xb_, lm_head_.w, lm_head_.scale, logits_, 1, H, c.vocab_size, xbf_, w_dq_, s);
}

void Model::prefill(const std::vector<int>& tokens, int slot, int past_len) {
    int M = (int)tokens.size();
    CUDA_CHECK(cudaMemcpy(d_ids_, tokens.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    std::vector<int> pos(M);
    for (int i = 0; i < M; ++i) pos[i] = past_len + i;
    CUDA_CHECK(cudaMemcpy(d_pos_, pos.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_past_, &past_len, sizeof(int), cudaMemcpyHostToDevice));
    run_layers_prefill(M, slot);
    CUDA_CHECK(cudaStreamSynchronize(stream_));   // logits_ row 0 ready
}

// --- batched single-token decode for B sequences -----------------------------------------
// Leaves the post-final-layer residual stream for all B sequences in x_ ([B, H]).
void Model::run_layers_decode(int B) {
    const ModelConfig& c = cfg_;
    int H = c.hidden_size, QD = c.q_dim(), KVD = c.kv_dim(), I = c.intermediate;
    int nH = c.num_heads, nKV = c.num_kv_heads, hd = c.head_dim;
    float eps = c.rms_eps, scale = 1.0f / std::sqrt((float)hd);
    cudaStream_t s = stream_;

    launch_embed(d_ids_, embed_, x_, B, H, s);

    for (int l = 0; l < c.num_layers; ++l) {
        Layer& L = layers_[l];

        launch_rmsnorm(x_, L.in_norm, xb_, B, H, eps, s);
        launch_matmul_decode(xb_, L.q_proj.w, L.q_proj.scale, q_, B, H, QD,  s);
        launch_matmul_decode(xb_, L.k_proj.w, L.k_proj.scale, k_, B, H, KVD, s);
        launch_matmul_decode(xb_, L.v_proj.w, L.v_proj.scale, v_, B, H, KVD, s);

        launch_rmsnorm(q_, L.q_norm, q_, B * nH,  hd, eps, s);
        launch_rmsnorm(k_, L.k_norm, k_, B * nKV, hd, eps, s);
        launch_rope(q_, d_pos_, B, nH,  hd, c.rope_theta, s);   // d_pos_[b] = past_len[b]
        launch_rope(k_, d_pos_, B, nKV, hd, c.rope_theta, s);

        // append each sequence's new K/V to its slot at row past_len[b] (= d_pos_)
        launch_store_kv_batch(k_, cache_k_[l], d_slot_, d_pos_, KVD, max_ctx_, B, s);
        launch_store_kv_batch(v_, cache_v_[l], d_slot_, d_pos_, KVD, max_ctx_, B, s);
        launch_attention_decode_batch(q_, cache_k_[l], cache_v_[l], attn_, B, nH, nKV, hd,
                                      d_slot_, d_pos_, max_ctx_, scale,
                                      part_m_, part_l_, part_acc_, s);

        launch_matmul_decode(attn_, L.o_proj.w, L.o_proj.scale, xb2_, B, QD, H, s);
        launch_add(x_, xb2_, B * H, s);

        launch_rmsnorm(x_, L.post_norm, xb_, B, H, eps, s);
        launch_matmul_decode(xb_, L.gate.w, L.gate.scale, gate_, B, H, I, s);
        launch_matmul_decode(xb_, L.up.w,   L.up.scale,   up_,   B, H, I, s);
        launch_silu_mul(gate_, up_, hmlp_, B * I, s);
        launch_matmul_decode(hmlp_, L.down.w, L.down.scale, xb2_, B, I, H, s);
        launch_add(x_, xb2_, B * H, s);
    }
}

// Shared forward for a decode batch: H2D inputs, the layer stack, final norm, and the lm_head
// GEMV — one pass reads the lm_head weight once for the whole batch and leaves the [B, vocab]
// logits in logits_ on the device. Async on stream_ (caller syncs).
void Model::decode_forward(const std::vector<int>& in_tokens, const std::vector<int>& past_len,
                           const std::vector<int>& slots) {
    int B = (int)in_tokens.size();
    int V = cfg_.vocab_size, H = cfg_.hidden_size;
    CUDA_CHECK(cudaMemcpy(d_ids_,  in_tokens.data(), B * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_,  past_len.data(),  B * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slot_, slots.data(),     B * sizeof(int), cudaMemcpyHostToDevice));

    run_layers_decode(B);
    launch_rmsnorm(x_, fnorm_, xb_, B, H, cfg_.rms_eps, stream_);          // final norm, all B rows
    launch_matmul_decode(xb_, lm_head_.w, lm_head_.scale, logits_, B, H, V, stream_);
}

void Model::decode(const std::vector<int>& in_tokens, const std::vector<int>& past_len,
                   const std::vector<int>& slots, std::vector<int>& out_tokens) {
    int B = (int)in_tokens.size();
    decode_forward(in_tokens, past_len, slots);
    launch_argmax_batch(logits_, B, cfg_.vocab_size, d_arg_, stream_);     // greedy, on the GPU
    out_tokens.resize(B);
    CUDA_CHECK(cudaMemcpyAsync(out_tokens.data(), d_arg_, B * sizeof(int),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

const float* Model::decode_logits_host(const std::vector<int>& in_tokens,
                                       const std::vector<int>& past_len, const std::vector<int>& slots) {
    int B = (int)in_tokens.size();
    decode_forward(in_tokens, past_len, slots);
    CUDA_CHECK(cudaMemcpyAsync(host_logits_.data(), logits_, (size_t)B * cfg_.vocab_size * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return host_logits_.data();
}

int Model::argmax_last() {
    launch_argmax(logits_, cfg_.vocab_size, d_arg_, stream_);
    int id = 0;
    CUDA_CHECK(cudaMemcpyAsync(&id, d_arg_, sizeof(int), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return id;
}

const std::vector<float>& Model::copy_logits() {
    CUDA_CHECK(cudaMemcpy(host_logits_.data(), logits_, cfg_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    return host_logits_;
}

Model::~Model() {
    for (auto p : all_bufs_)  if (p) cudaFree(p);
    for (auto p : all_fbufs_) if (p) cudaFree(p);
    for (auto p : all_i8_)    if (p) cudaFree(p);
    for (auto p : cache_k_)   if (p) cudaFree(p);
    for (auto p : cache_v_)   if (p) cudaFree(p);
    cudaFree(x_); cudaFree(xb_); cudaFree(xb2_); cudaFree(q_); cudaFree(k_); cudaFree(v_);
    cudaFree(attn_); cudaFree(gate_); cudaFree(up_); cudaFree(hmlp_); cudaFree(logits_);
    cudaFree(xbf_); cudaFree(w_dq_); cudaFree(part_m_); cudaFree(part_l_); cudaFree(part_acc_);
    cudaFree(d_ids_); cudaFree(d_pos_); cudaFree(d_arg_); cudaFree(d_past_); cudaFree(d_slot_);
    if (stream_) cudaStreamDestroy(stream_);
}
