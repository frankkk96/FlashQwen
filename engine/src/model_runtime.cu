#include "model_runtime.hpp"
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

ModelRuntime::ModelRuntime(const ModelSpec& spec, int max_ctx, ProgressFn on_progress)
    : spec_(spec) {
    max_ctx_ = ((max_ctx + 15) / 16) * 16;   // round up so WMMA tiles never read past buffers
    std::fprintf(stderr, "[model] loading weights from %s ...\n", spec_.dir.c_str());
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
            std::fprintf(stderr, "[model] uploaded layer %d/%d\n", l + 1, spec_.num_layers);
        if (on_progress) on_progress(l + 1, spec_.num_layers);
    }

    // activation scratch (sized to max_ctx tokens; a decode batch is at most MAX_DECODE_B <= that)
    int H = spec_.hidden_size, QD = spec_.q_dim(), I = spec_.intermediate, V = spec_.vocab_size;
    int kvd = spec_.kv_dim(), nh = spec_.num_heads, hd = spec_.head_dim;
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
    CUDA_CHECK(cudaMalloc(&d_arg_,  MAX_DECODE_B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_past_, sizeof(int)));
    // Paged-KV indexing buffers. d_bt_ holds up to MAX_DECODE_B padded block tables (one per
    // decode-batch row, or row 0 for prefill). d_iota_ = {0,1,..} is the decode bt_row map;
    // d_zero_ = {0,0,..} is the prefill bt_row map (all M tokens use block-table row 0).
    max_blocks_ = (max_ctx_ + KVCache::BLOCK - 1) / KVCache::BLOCK;
    CUDA_CHECK(cudaMalloc(&d_bt_,   (size_t)MAX_DECODE_B * max_blocks_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_iota_, MAX_DECODE_B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_zero_, max_ctx_ * sizeof(int)));
    host_bt_.resize((size_t)MAX_DECODE_B * max_blocks_);
    {
        std::vector<int> iota(MAX_DECODE_B);
        for (int i = 0; i < MAX_DECODE_B; ++i) iota[i] = i;
        CUDA_CHECK(cudaMemcpy(d_iota_, iota.data(), MAX_DECODE_B * sizeof(int), cudaMemcpyHostToDevice));
        std::vector<int> zero(max_ctx_, 0);
        CUDA_CHECK(cudaMemcpy(d_zero_, zero.data(), max_ctx_ * sizeof(int), cudaMemcpyHostToDevice));
    }
    // flash-decoding split-K scratch, sized for a full decode batch
    CUDA_CHECK(cudaMalloc(&part_m_,   (size_t)MAX_DECODE_B * nh * ATTN_SPLITS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&part_l_,   (size_t)MAX_DECODE_B * nh * ATTN_SPLITS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&part_acc_, (size_t)MAX_DECODE_B * nh * ATTN_SPLITS * hd * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    host_logits_.resize((size_t)MAX_DECODE_B * V);

    size_t freeb, totalb;
    CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
    std::fprintf(stderr, "[model] weights + activations ready (max_ctx=%d, up to %d concurrent). "
                 "GPU mem: %.1f GB used / %.1f GB total. KV pool allocated separately.\n",
                 max_ctx_, max_batch(), (totalb - freeb) / 1e9, totalb / 1e9);
}

// --- single-sequence prefill: M tokens into the paged KV described by d_bt_ row 0 ----------
// The store/attention kernels resolve every position through that block table; d_pos_ holds the
// logical positions (past_len..past_len+M-1) and d_past_ the base offset, so a chunked prefill
// (past_len>0) writes into the correct blocks. All M tokens use block-table row 0 (d_zero_).
void ModelRuntime::run_layers_prefill(int M) {
    const ModelSpec& c = spec_;
    int H = c.hidden_size, QD = c.q_dim(), KVD = c.kv_dim(), I = c.intermediate;
    int nH = c.num_heads, nKV = c.num_kv_heads, hd = c.head_dim;
    float eps = c.rms_eps, scale = 1.0f / std::sqrt((float)hd);
    cudaStream_t s = stream_;

    launch_embed(d_ids_, embed_, x_, M, H, s);

    for (int l = 0; l < c.num_layers; ++l) {
        Layer& L = layers_[l];

        launch_rmsnorm(x_, L.in_norm, xb_, M, H, eps, s);
        launch_matmul(xb_, L.q_proj.w, L.q_proj.scale, q_, M, H, QD,  xbf_, w_dq_, s);
        launch_matmul(xb_, L.k_proj.w, L.k_proj.scale, k_, M, H, KVD, xbf_, w_dq_, s);
        launch_matmul(xb_, L.v_proj.w, L.v_proj.scale, v_, M, H, KVD, xbf_, w_dq_, s);

        launch_rmsnorm(q_, L.q_norm, q_, M * nH,  hd, eps, s);
        launch_rmsnorm(k_, L.k_norm, k_, M * nKV, hd, eps, s);
        launch_rope(q_, d_pos_, M, nH,  hd, c.rope_theta, s);
        launch_rope(k_, d_pos_, M, nKV, hd, c.rope_theta, s);

        int blk = kv_->block_size();
        launch_store_kv_paged(k_, kv_->k(l), d_bt_, bt_stride_, blk, KVD, d_zero_, d_pos_, M, s);
        launch_store_kv_paged(v_, kv_->v(l), d_bt_, bt_stride_, blk, KVD, d_zero_, d_pos_, M, s);
        launch_attention_paged(q_, kv_->k(l), kv_->v(l), attn_, M, nH, nKV, hd, d_past_, scale,
                               d_bt_, bt_stride_, blk, s);

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

void ModelRuntime::prefill(const std::vector<int>& tokens, const std::vector<int>& block_table, int past_len) {
    int M = (int)tokens.size();
    CUDA_CHECK(cudaMemcpy(d_ids_, tokens.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    std::vector<int> pos(M);
    for (int i = 0; i < M; ++i) pos[i] = past_len + i;
    CUDA_CHECK(cudaMemcpy(d_pos_, pos.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_past_, &past_len, sizeof(int), cudaMemcpyHostToDevice));
    bt_stride_ = (int)block_table.size();          // block table goes into d_bt_ row 0
    CUDA_CHECK(cudaMemcpy(d_bt_, block_table.data(), bt_stride_ * sizeof(int), cudaMemcpyHostToDevice));
    run_layers_prefill(M);
    CUDA_CHECK(cudaStreamSynchronize(stream_));   // logits_ row 0 ready
}

// --- batched single-token decode for B sequences -----------------------------------------
// Leaves the post-final-layer residual stream for all B sequences in x_ ([B, H]).
void ModelRuntime::run_layers_decode(int B) {
    const ModelSpec& c = spec_;
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

        // append each sequence's new K/V at its logical position past_len[b] (= d_pos_); the
        // block table (d_bt_, row b via d_iota_) maps that to a physical block row.
        int blk = kv_->block_size();
        launch_store_kv_paged(k_, kv_->k(l), d_bt_, bt_stride_, blk, KVD, d_iota_, d_pos_, B, s);
        launch_store_kv_paged(v_, kv_->v(l), d_bt_, bt_stride_, blk, KVD, d_iota_, d_pos_, B, s);
        launch_attention_decode_paged(q_, kv_->k(l), kv_->v(l), attn_, B, nH, nKV, hd,
                                      d_pos_, d_bt_, bt_stride_, blk, scale,
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

// Pack B block tables, padded to the longest, into host_bt_ and upload to d_bt_; bt_stride_ is
// the padded row length the kernels index with.
void ModelRuntime::upload_block_tables(const std::vector<std::vector<int>>& bts) {
    int B = (int)bts.size();
    int mb = 1;
    for (auto& t : bts) mb = std::max(mb, (int)t.size());
    bt_stride_ = mb;
    for (int b = 0; b < B; ++b) {
        int* row = host_bt_.data() + (size_t)b * mb;
        for (int g = 0; g < mb; ++g) row[g] = g < (int)bts[b].size() ? bts[b][g] : 0;
    }
    CUDA_CHECK(cudaMemcpy(d_bt_, host_bt_.data(), (size_t)B * mb * sizeof(int), cudaMemcpyHostToDevice));
}

// Shared forward for a decode batch: H2D inputs, the layer stack, final norm, and the lm_head
// GEMV — one pass reads the lm_head weight once for the whole batch and leaves the [B, vocab]
// logits in logits_ on the device. Async on stream_ (caller syncs).
void ModelRuntime::decode_forward(const std::vector<int>& in_tokens, const std::vector<int>& past_len,
                           const std::vector<std::vector<int>>& block_tables) {
    int B = (int)in_tokens.size();
    int V = spec_.vocab_size, H = spec_.hidden_size;
    CUDA_CHECK(cudaMemcpy(d_ids_, in_tokens.data(), B * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_, past_len.data(),  B * sizeof(int), cudaMemcpyHostToDevice));
    upload_block_tables(block_tables);

    run_layers_decode(B);
    launch_rmsnorm(x_, fnorm_, xb_, B, H, spec_.rms_eps, stream_);          // final norm, all B rows
    launch_matmul_decode(xb_, lm_head_.w, lm_head_.scale, logits_, B, H, V, stream_);
}

void ModelRuntime::decode(const std::vector<int>& in_tokens, const std::vector<int>& past_len,
                   const std::vector<std::vector<int>>& block_tables, std::vector<int>& out_tokens) {
    int B = (int)in_tokens.size();
    decode_forward(in_tokens, past_len, block_tables);
    launch_argmax_batch(logits_, B, spec_.vocab_size, d_arg_, stream_);     // greedy, on the GPU
    out_tokens.resize(B);
    CUDA_CHECK(cudaMemcpyAsync(out_tokens.data(), d_arg_, B * sizeof(int),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

const float* ModelRuntime::decode_logits_host(const std::vector<int>& in_tokens,
                                       const std::vector<int>& past_len,
                                       const std::vector<std::vector<int>>& block_tables) {
    int B = (int)in_tokens.size();
    decode_forward(in_tokens, past_len, block_tables);
    CUDA_CHECK(cudaMemcpyAsync(host_logits_.data(), logits_, (size_t)B * spec_.vocab_size * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return host_logits_.data();
}

int ModelRuntime::argmax_last() {
    launch_argmax(logits_, spec_.vocab_size, d_arg_, stream_);
    int id = 0;
    CUDA_CHECK(cudaMemcpyAsync(&id, d_arg_, sizeof(int), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return id;
}

const std::vector<float>& ModelRuntime::copy_logits() {
    CUDA_CHECK(cudaMemcpy(host_logits_.data(), logits_, spec_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    return host_logits_;
}

ModelRuntime::~ModelRuntime() {
    for (auto p : all_bufs_)  if (p) cudaFree(p);
    for (auto p : all_fbufs_) if (p) cudaFree(p);
    for (auto p : all_i8_)    if (p) cudaFree(p);
    cudaFree(x_); cudaFree(xb_); cudaFree(xb2_); cudaFree(q_); cudaFree(k_); cudaFree(v_);
    cudaFree(attn_); cudaFree(gate_); cudaFree(up_); cudaFree(hmlp_); cudaFree(logits_);
    cudaFree(xbf_); cudaFree(w_dq_); cudaFree(part_m_); cudaFree(part_l_); cudaFree(part_acc_);
    cudaFree(d_ids_); cudaFree(d_pos_); cudaFree(d_arg_); cudaFree(d_past_);
    cudaFree(d_bt_); cudaFree(d_iota_); cudaFree(d_zero_);
    if (stream_) cudaStreamDestroy(stream_);
}
