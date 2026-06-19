#include "model_runtime.hpp"
#include "log.hpp"
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

bf16* ModelRuntime::upload_bf16(const std::string& name) {
    const TensorView& tv = st_.get(name);
    bf16* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, tv.nbytes));
    CUDA_CHECK(cudaMemcpy(d, tv.data, tv.nbytes, cudaMemcpyHostToDevice));
    all_bufs_.push_back(d);
    return d;
}

float* ModelRuntime::upload_norm(const std::string& name) {
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
ModelRuntime::QWeight ModelRuntime::upload_int8(const std::string& name) {
    const TensorView& tv = st_.get(name);
    int OUT = (int)tv.shape[0], IN = (int)tv.shape[1];
    int64_t n = (int64_t)OUT * IN;
    const uint16_t* src = (const uint16_t*)tv.data;
    std::vector<int8_t> q(n);
    std::vector<float> scale(OUT);

    auto worker = [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            const uint16_t* row = src + (int64_t)r * IN;
            float maxabs = 0.f;
            for (int i = 0; i < IN; ++i) maxabs = std::max(maxabs, std::fabs(bf16_to_f32(row[i])));
            float sc = maxabs / 127.0f; if (sc == 0.f) sc = 1.f;
            float inv = 1.0f / sc;
            int8_t* qr = q.data() + (int64_t)r * IN;
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

ModelRuntime::ModelRuntime(const ModelSpec& spec, const RuntimeConfig& cfg)
    : spec_(spec), rng_(cfg.seed) {
    max_ctx_  = ((cfg.max_ctx + 15) / 16) * 16;   // round up so WMMA tiles never read past buffers
    max_rows_ = std::max(max_ctx_, ((cfg.max_batch_tokens + 15) / 16) * 16);  // query rows per forward
    LOG_INFO("[model] loading weights from %s ...", spec_.dir.c_str());
    st_.load_dir(spec_.dir);

    embed_   = upload_bf16("model.embed_tokens.weight");   // gather, kept BF16
    fnorm_   = upload_norm("model.norm.weight");
    lm_head_ = upload_int8("lm_head.weight");

    layers_.resize(spec_.num_layers);
    for (int l = 0; l < spec_.num_layers; ++l) {
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
        if ((l + 1) % 8 == 0 || l + 1 == spec_.num_layers)
            LOG_INFO("[model] uploaded layer %d/%d", l + 1, spec_.num_layers);
    }

    // activation scratch (sized to max_rows_ query rows; the sampling buffers are sized to the
    // concurrent-request cap MAX_DECODE_B, since at most one logits row is produced per request).
    int H = spec_.hidden_size, QD = spec_.q_dim(), I = spec_.intermediate, V = spec_.vocab_size;
    int kvd = spec_.kv_dim();
    auto fmalloc = [&](float** p, size_t n) { CUDA_CHECK(cudaMalloc(p, n * sizeof(float))); };
    fmalloc(&x_,    (size_t)max_rows_ * H);
    fmalloc(&xb_,   (size_t)max_rows_ * H);
    fmalloc(&xb2_,  (size_t)max_rows_ * H);
    fmalloc(&q_,    (size_t)max_rows_ * QD);
    fmalloc(&k_,    (size_t)max_rows_ * kvd);
    fmalloc(&v_,    (size_t)max_rows_ * kvd);
    fmalloc(&attn_, (size_t)max_rows_ * QD);
    fmalloc(&gate_, (size_t)max_rows_ * I);
    fmalloc(&up_,   (size_t)max_rows_ * I);
    fmalloc(&hmlp_, (size_t)max_rows_ * I);
    fmalloc(&xg_,     (size_t)MAX_DECODE_B * H);   // gathered sampling rows
    fmalloc(&logits_, (size_t)MAX_DECODE_B * V);   // [S, vocab]
    // BF16 activation scratch for the tensor-core matmul (largest IN is `I`).
    CUDA_CHECK(cudaMalloc(&xbf_, (size_t)max_rows_ * I * sizeof(bf16)));
    // BF16 dequantized-weight scratch for the tensor-core matmul (largest matmul weight is I*H).
    CUDA_CHECK(cudaMalloc(&w_dq_, (size_t)I * H * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_ids_, max_rows_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pos_, max_rows_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_req_, max_rows_ * sizeof(int)));
    // Paged-KV block tables: up to MAX_DECODE_B requests, padded to the longest block table.
    max_blocks_ = (max_ctx_ + BlockPool::BLOCK - 1) / BlockPool::BLOCK;
    CUDA_CHECK(cudaMalloc(&d_bt_, (size_t)MAX_DECODE_B * max_blocks_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lrows_, MAX_DECODE_B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_arg_,   MAX_DECODE_B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_invT_,  MAX_DECODE_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topp_,  MAX_DECODE_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u_,     MAX_DECODE_B * sizeof(float)));
    host_bt_.resize((size_t)MAX_DECODE_B * max_blocks_);
    host_invT_.resize(MAX_DECODE_B); host_topp_.resize(MAX_DECODE_B); host_u_.resize(MAX_DECODE_B);
    CUDA_CHECK(cudaStreamCreate(&stream_));

    size_t freeb, totalb;
    CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
    LOG_INFO("[model] weights + activations ready (max_ctx=%d, max_rows=%d, up to %d concurrent). "
             "GPU mem: %.1f GB used / %.1f GB total. KV pool allocated separately.",
             max_ctx_, max_rows_, max_batch(), (totalb - freeb) / 1e9, totalb / 1e9);
}

void ModelRuntime::mm(const float* x, const QWeight& w, float* y, int M, int IN, int OUT, bool pure_decode) {
    if (pure_decode) launch_matmul_decode(x, w.w, w.scale, y, M, IN, OUT, stream_);
    else             launch_matmul(x, w.w, w.scale, y, M, IN, OUT, xbf_, w_dq_, stream_);
}

// The unified layer stack over T query rows. Every per-token op runs on the flattened batch; the
// attention kernel resolves each row's request/position/KV through d_req_/d_pos_/d_bt_.
void ModelRuntime::run_layers(int T, bool pure_decode) {
    const ModelSpec& c = spec_;
    int H = c.hidden_size, QD = c.q_dim(), KVD = c.kv_dim(), I = c.intermediate;
    int nH = c.num_heads, nKV = c.num_kv_heads, hd = c.head_dim;
    float eps = c.rms_eps, scale = 1.0f / std::sqrt((float)hd);
    cudaStream_t s = stream_;
    int blk = pool_->block_size();

    launch_embed(d_ids_, embed_, x_, T, H, s);

    for (int l = 0; l < c.num_layers; ++l) {
        Layer& L = layers_[l];

        launch_rmsnorm(x_, L.in_norm, xb_, T, H, eps, s);
        mm(xb_, L.q_proj, q_, T, H, QD,  pure_decode);
        mm(xb_, L.k_proj, k_, T, H, KVD, pure_decode);
        mm(xb_, L.v_proj, v_, T, H, KVD, pure_decode);

        launch_rmsnorm(q_, L.q_norm, q_, T * nH,  hd, eps, s);
        launch_rmsnorm(k_, L.k_norm, k_, T * nKV, hd, eps, s);
        launch_rope(q_, d_pos_, T, nH,  hd, c.rope_theta, s);   // d_pos_[m] = row m's absolute position
        launch_rope(k_, d_pos_, T, nKV, hd, c.rope_theta, s);

        // append each row's new K/V at its position (d_pos_), into its request's blocks (d_req_ row).
        launch_store_kv_paged(k_, pool_->k(l), d_bt_, bt_stride_, blk, KVD, d_req_, d_pos_, T, s);
        launch_store_kv_paged(v_, pool_->v(l), d_bt_, bt_stride_, blk, KVD, d_req_, d_pos_, T, s);
        launch_attention_varlen(q_, pool_->k(l), pool_->v(l), attn_, T, nH, nKV, hd,
                                d_pos_, d_req_, d_bt_, bt_stride_, blk, scale, s);

        mm(attn_, L.o_proj, xb2_, T, QD, H, pure_decode);
        launch_add(x_, xb2_, T * H, s);

        launch_rmsnorm(x_, L.post_norm, xb_, T, H, eps, s);
        mm(xb_, L.gate, gate_, T, H, I, pure_decode);
        mm(xb_, L.up,   up_,   T, H, I, pure_decode);
        launch_silu_mul(gate_, up_, hmlp_, T * I, s);
        mm(hmlp_, L.down, xb2_, T, I, H, pure_decode);
        launch_add(x_, xb2_, T * H, s);
    }
}

// Pack R block tables, padded to the longest, into host_bt_ and upload to d_bt_; bt_stride_ is the
// padded row length the kernels index with.
void ModelRuntime::upload_block_tables(const std::vector<std::vector<int>>& bts) {
    int R = (int)bts.size();
    int mb = 1;
    for (auto& t : bts) mb = std::max(mb, (int)t.size());
    bt_stride_ = mb;
    for (int r = 0; r < R; ++r) {
        int* row = host_bt_.data() + (size_t)r * mb;
        for (int g = 0; g < mb; ++g) row[g] = g < (int)bts[r].size() ? bts[r][g] : 0;
    }
    CUDA_CHECK(cudaMemcpy(d_bt_, host_bt_.data(), (size_t)R * mb * sizeof(int), cudaMemcpyHostToDevice));
}

// Shared forward: H2D inputs, the unified layer stack, then gather the per-request sampling rows and
// run final norm + lm_head only on those -> logits_ [S, vocab] on the device. Async on stream_.
void ModelRuntime::forward_core(const ForwardInput& in) {
    int T = in.rows(), R = (int)in.block_tables.size(), S = (int)in.logits_rows.size();
    int H = spec_.hidden_size, V = spec_.vocab_size;
    CUDA_CHECK(cudaMemcpy(d_ids_, in.tokens.data(),    T * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_, in.positions.data(), T * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_req_, in.req_index.data(), T * sizeof(int), cudaMemcpyHostToDevice));
    upload_block_tables(in.block_tables);

    bool pure_decode = (T == R);   // every request contributes exactly one row -> batched INT8 GEMV
    run_layers(T, pure_decode);

    if (S > 0) {
        CUDA_CHECK(cudaMemcpy(d_lrows_, in.logits_rows.data(), S * sizeof(int), cudaMemcpyHostToDevice));
        launch_gather_rows(x_, d_lrows_, xg_, S, H, stream_);
        launch_rmsnorm(xg_, fnorm_, xb_, S, H, spec_.rms_eps, stream_);
        launch_matmul_decode(xb_, lm_head_.w, lm_head_.scale, logits_, S, H, V, stream_);  // INT8 GEMV
    }
}

void ModelRuntime::forward(const ForwardInput& in, std::vector<int>& out_tokens) {
    int S = (int)in.logits_rows.size();
    forward_core(in);
    out_tokens.resize(S);
    if (S > 0) {
        // Per-row sampling inputs: 1/temp (<=0 => greedy), nucleus cutoff, and a uniform draw.
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < S; ++i) {
            const SampleParams& sp = in.sample_params[i];
            host_invT_[i] = sp.temp > 0.0f ? 1.0f / sp.temp : 0.0f;
            host_topp_[i] = sp.top_p;
            host_u_[i]    = dist(rng_);
        }
        CUDA_CHECK(cudaMemcpyAsync(d_invT_, host_invT_.data(), S * sizeof(float), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_topp_, host_topp_.data(), S * sizeof(float), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_u_,    host_u_.data(),    S * sizeof(float), cudaMemcpyHostToDevice, stream_));
        launch_sample_batch(logits_, S, spec_.vocab_size, d_invT_, d_topp_, d_u_, d_arg_, stream_);
        CUDA_CHECK(cudaMemcpyAsync(out_tokens.data(), d_arg_, S * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

ModelRuntime::~ModelRuntime() {
    for (auto p : all_bufs_)  if (p) cudaFree(p);
    for (auto p : all_fbufs_) if (p) cudaFree(p);
    for (auto p : all_i8_)    if (p) cudaFree(p);
    cudaFree(x_); cudaFree(xb_); cudaFree(xb2_); cudaFree(q_); cudaFree(k_); cudaFree(v_);
    cudaFree(attn_); cudaFree(gate_); cudaFree(up_); cudaFree(hmlp_); cudaFree(xg_); cudaFree(logits_);
    cudaFree(xbf_); cudaFree(w_dq_);
    cudaFree(d_ids_); cudaFree(d_pos_); cudaFree(d_req_);
    cudaFree(d_bt_); cudaFree(d_lrows_); cudaFree(d_arg_);
    cudaFree(d_invT_); cudaFree(d_topp_); cudaFree(d_u_);
    if (stream_) cudaStreamDestroy(stream_);
}
