#include "model_runtime.hpp"
#include "log.hpp"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>

#define CUBLAS_CHECK(call) do {                                                  \
    cublasStatus_t _s = (call);                                                  \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                           \
        std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, _s);\
        std::exit(1);                                                            \
    }                                                                            \
} while (0)

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

ModelRuntime::ModelRuntime(const ModelSpec& spec, const RuntimeConfig& cfg)
    : spec_(spec), rng_(cfg.seed) {
    max_ctx_  = cfg.max_ctx;
    max_rows_ = std::max(max_ctx_, cfg.max_batch_tokens);   // query rows per forward
    LOG_INFO("[model] loading weights from %s ...", spec_.dir.c_str());
    st_.load_dir(spec_.dir);

    embed_   = upload_bf16("model.embed_tokens.weight");   // gather, kept BF16
    fnorm_   = upload_norm("model.norm.weight");
    lm_head_ = upload_bf16("lm_head.weight");

    layers_.resize(spec_.num_layers);
    for (int l = 0; l < spec_.num_layers; ++l) {
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
        if ((l + 1) % 8 == 0 || l + 1 == spec_.num_layers)
            LOG_INFO("[model] uploaded layer %d/%d", l + 1, spec_.num_layers);
    }

    // activation scratch (BF16, sized to max_rows_ query rows); the sampling buffers are sized to the
    // concurrent-request cap MAX_DECODE_B, since at most one logits row is produced per request.
    int H = spec_.hidden_size, QD = spec_.q_dim(), I = spec_.intermediate, V = spec_.vocab_size;
    int kvd = spec_.kv_dim();
    auto bmalloc = [&](bf16** p, size_t n) { CUDA_CHECK(cudaMalloc(p, n * sizeof(bf16))); };
    bmalloc(&x_,    (size_t)max_rows_ * H);
    bmalloc(&xb_,   (size_t)max_rows_ * H);
    bmalloc(&xb2_,  (size_t)max_rows_ * H);
    bmalloc(&q_,    (size_t)max_rows_ * QD);
    bmalloc(&k_,    (size_t)max_rows_ * kvd);
    bmalloc(&v_,    (size_t)max_rows_ * kvd);
    bmalloc(&attn_, (size_t)max_rows_ * QD);
    bmalloc(&gate_, (size_t)max_rows_ * I);
    bmalloc(&up_,   (size_t)max_rows_ * I);
    bmalloc(&hmlp_, (size_t)max_rows_ * I);
    bmalloc(&xg_,   (size_t)MAX_DECODE_B * H);             // gathered sampling rows
    CUDA_CHECK(cudaMalloc(&logits_, (size_t)MAX_DECODE_B * V * sizeof(float)));   // [S, vocab] FP32
    CUDA_CHECK(cudaMalloc(&d_ids_, max_rows_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pos_, max_rows_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_req_, max_rows_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_qstart_, MAX_DECODE_B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_qlen_,   MAX_DECODE_B * sizeof(int)));
    // Paged-KV block tables: up to MAX_DECODE_B requests, padded to the longest block table.
    max_blocks_ = (max_ctx_ + BlockPool::BLOCK - 1) / BlockPool::BLOCK;
    CUDA_CHECK(cudaMalloc(&d_bt_, (size_t)MAX_DECODE_B * max_blocks_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lrows_, MAX_DECODE_B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_arg_,   MAX_DECODE_B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_invT_,  MAX_DECODE_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topp_,  MAX_DECODE_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u_,     MAX_DECODE_B * sizeof(float)));
    host_bt_.resize((size_t)MAX_DECODE_B * max_blocks_);
    host_qstart_.resize(MAX_DECODE_B); host_qlen_.resize(MAX_DECODE_B);
    host_invT_.resize(MAX_DECODE_B); host_topp_.resize(MAX_DECODE_B); host_u_.resize(MAX_DECODE_B);
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
    CUBLAS_CHECK(cublasSetMathMode(cublas_, CUBLAS_DEFAULT_MATH));   // BF16 inputs use tensor cores

    size_t freeb, totalb;
    CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
    LOG_INFO("[model] weights + activations ready (max_ctx=%d, max_rows=%d, up to %d concurrent). "
             "GPU mem: %.1f GB used / %.1f GB total. KV pool allocated separately.",
             max_ctx_, max_rows_, max_batch(), (totalb - freeb) / 1e9, totalb / 1e9);
}

// y[M,OUT] = x[M,IN] @ W[OUT,IN]^T. cuBLAS is column-major: a row-major [a,b] buffer is a column-major
// [b,a]. So the row-major W[OUT,IN] is column-major Wc[IN,OUT] and x[M,IN] is Xc[IN,M]; computing
// Z = Wc^T @ Xc (m=OUT, n=M, k=IN) yields column-major [OUT,M] = row-major y[M,OUT]. BF16 in, FP32
// accumulate; Ytype selects the output element type.
void ModelRuntime::gemm(const bf16* x, const bf16* W, void* y, int M, int IN, int OUT,
                        cudaDataType_t Ytype) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, OUT, M, IN,
                              &alpha, W, CUDA_R_16BF, IN, x, CUDA_R_16BF, IN,
                              &beta, y, Ytype, OUT, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// The unified layer stack over T query rows (prefill chunks + decodes together; no decode/prefill
// split). Every per-token op runs on the flattened batch; attention resolves each request's rows
// (qstart/qlen), positions (d_pos_) and KV (d_bt_) itself. R = number of requests, max_qlen = the
// longest per-request row count (sets the attention q-tile grid).
void ModelRuntime::run_layers(int T, int R, int max_qlen) {
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
        gemm(xb_, L.q_proj, q_, T, H, QD,  CUDA_R_16BF);
        gemm(xb_, L.k_proj, k_, T, H, KVD, CUDA_R_16BF);
        gemm(xb_, L.v_proj, v_, T, H, KVD, CUDA_R_16BF);

        launch_rmsnorm(q_, L.q_norm, q_, T * nH,  hd, eps, s);
        launch_rmsnorm(k_, L.k_norm, k_, T * nKV, hd, eps, s);
        launch_rope(q_, d_pos_, T, nH,  hd, c.rope_theta, s);   // d_pos_[m] = row m's absolute position
        launch_rope(k_, d_pos_, T, nKV, hd, c.rope_theta, s);

        // append each row's new K/V at its position (d_pos_), into its request's blocks (d_req_ row).
        launch_store_kv_paged(k_, pool_->k(l), d_bt_, bt_stride_, blk, KVD, d_req_, d_pos_, T, s);
        launch_store_kv_paged(v_, pool_->v(l), d_bt_, bt_stride_, blk, KVD, d_req_, d_pos_, T, s);
        launch_attention_flash(q_, pool_->k(l), pool_->v(l), attn_, nH, nKV, hd,
                               d_pos_, d_qstart_, d_qlen_, R, max_qlen, d_bt_, bt_stride_, blk, scale, s);

        gemm(attn_, L.o_proj, xb2_, T, QD, H, CUDA_R_16BF);
        launch_add(x_, xb2_, T * H, s);

        launch_rmsnorm(x_, L.post_norm, xb_, T, H, eps, s);
        gemm(xb_, L.gate, gate_, T, H, I, CUDA_R_16BF);
        gemm(xb_, L.up,   up_,   T, H, I, CUDA_R_16BF);
        launch_silu_mul(gate_, up_, hmlp_, T * I, s);
        gemm(hmlp_, L.down, xb2_, T, I, H, CUDA_R_16BF);
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

    // Per-request attention grouping: rows are contiguous and ordered by request (the batch is built
    // request by request), so qstart is a prefix sum of the per-request row counts.
    for (int r = 0; r < R; ++r) host_qlen_[r] = 0;
    for (int t = 0; t < T; ++t) host_qlen_[in.req_index[t]]++;
    int acc = 0, max_qlen = 1;
    for (int r = 0; r < R; ++r) { host_qstart_[r] = acc; acc += host_qlen_[r]; max_qlen = std::max(max_qlen, host_qlen_[r]); }
    CUDA_CHECK(cudaMemcpy(d_qstart_, host_qstart_.data(), R * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qlen_,   host_qlen_.data(),   R * sizeof(int), cudaMemcpyHostToDevice));

    run_layers(T, R, max_qlen);

    if (S > 0) {
        CUDA_CHECK(cudaMemcpy(d_lrows_, in.logits_rows.data(), S * sizeof(int), cudaMemcpyHostToDevice));
        launch_gather_rows(x_, d_lrows_, xg_, S, H, stream_);
        launch_rmsnorm(xg_, fnorm_, xb_, S, H, spec_.rms_eps, stream_);
        gemm(xb_, lm_head_, logits_, S, H, V, CUDA_R_32F);   // BF16 in -> FP32 logits
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
    cudaFree(x_); cudaFree(xb_); cudaFree(xb2_); cudaFree(q_); cudaFree(k_); cudaFree(v_);
    cudaFree(attn_); cudaFree(gate_); cudaFree(up_); cudaFree(hmlp_); cudaFree(xg_); cudaFree(logits_);
    cudaFree(d_ids_); cudaFree(d_pos_); cudaFree(d_req_);
    cudaFree(d_qstart_); cudaFree(d_qlen_);
    cudaFree(d_bt_); cudaFree(d_lrows_); cudaFree(d_arg_);
    cudaFree(d_invT_); cudaFree(d_topp_); cudaFree(d_u_);
    if (cublas_) cublasDestroy(cublas_);
    if (stream_) cudaStreamDestroy(stream_);
}
