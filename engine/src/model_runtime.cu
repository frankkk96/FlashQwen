#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "log.hpp"
#include "model_runtime.hpp"

static inline float Bf16ToF32(uint16_t h) {
  uint32_t bits = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

bf16* ModelRuntime::UploadBf16(const std::string& name) {
  const TensorView& tv = st_.Get(name);
  bf16* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, tv.nbytes));
  CUDA_CHECK(cudaMemcpy(d, tv.data, tv.nbytes, cudaMemcpyHostToDevice));
  all_bufs_.push_back(d);
  return d;
}

// Stack weights on their OUT (row) dim into one buffer. Each is HF row-major
// [OUT_i, IN] with shared IN, so contiguous rows give [sum(OUT_i), IN] —
// exactly the layout a fused projection (QKV, gate|up) needs.
bf16* ModelRuntime::UploadConcat(const std::vector<std::string>& names) {
  size_t total = 0;
  for (auto& n : names) total += st_.Get(n).nbytes;
  bf16* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, total));
  size_t off = 0;
  for (auto& n : names) {
    const TensorView& tv = st_.Get(n);
    CUDA_CHECK(
        cudaMemcpy((char*)d + off, tv.data, tv.nbytes, cudaMemcpyHostToDevice));
    off += tv.nbytes;
  }
  all_bufs_.push_back(d);
  return d;
}

float* ModelRuntime::UploadNorm(const std::string& name) {
  const TensorView& tv = st_.Get(name);
  int64_t n = tv.Numel();
  std::vector<float> tmp(n);
  const uint16_t* src = (const uint16_t*)tv.data;  // BF16
  for (int64_t i = 0; i < n; ++i) tmp[i] = Bf16ToF32(src[i]);
  float* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
  CUDA_CHECK(
      cudaMemcpy(d, tmp.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  all_fbufs_.push_back(d);
  return d;
}

// Precompute RoPE cos/sin once: angle(pos,i) = pos * theta^(-2i/head_dim),
// i in [0, head_dim/2). Layer-independent, so doing it here avoids the
// per-layer powf/cosf/sinf the old in-kernel RoPE paid. Tables [max_ctx,
// head_dim/2] FP32.
void ModelRuntime::PrecomputeRope() {
  int half = spec_.head_dim / 2, P = max_ctx_;
  std::vector<float> cs((size_t)P * half), sn((size_t)P * half);
  for (int p = 0; p < P; ++p)
    for (int i = 0; i < half; ++i) {
      float inv = std::pow(spec_.rope_theta, -2.0f * i / spec_.head_dim);
      float ang = p * inv;
      cs[(size_t)p * half + i] = std::cos(ang);
      sn[(size_t)p * half + i] = std::sin(ang);
    }
  CUDA_CHECK(cudaMalloc(&cos_tab_, cs.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&sin_tab_, sn.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(cos_tab_, cs.data(), cs.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(sin_tab_, sn.data(), sn.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}

ModelRuntime::ModelRuntime(const ModelSpec& spec, const RuntimeConfig& cfg)
    : spec_(spec), rng_(cfg.seed) {
  max_ctx_ = cfg.max_ctx;
  // A forward never exceeds max_batch_tokens rows (scheduler caps T, prefills
  // are chunked under it), so scratch is sized to that, NOT max_ctx — sizing to
  // max_ctx over-allocated ~4x and starved the KV pool. +16: the WMMA prefill
  // kernel reads a full 16-row Q tile, so the last non-aligned chunk can
  // over-read up to 15 rows past T; the margin keeps that (masked) read
  // in-bounds.
  max_rows_ = std::max(1, cfg.max_batch_tokens) + 16;
  LOG_INFO("[model] loading weights from %s ...", spec_.dir.c_str());
  st_.LoadDir(spec_.dir);

  embed_ = UploadBf16("model.embed_tokens.weight");  // gather, kept BF16
  fnorm_ = UploadNorm("model.norm.weight");
  lm_head_ = UploadBf16("lm_head.weight");
  PrecomputeRope();

  layers_.resize(spec_.num_layers);
  for (int l = 0; l < spec_.num_layers; ++l) {
    std::string p = "model.layers." + std::to_string(l) + ".";
    Layer& L = layers_[l];
    L.qkv = UploadConcat({p + "self_attn.q_proj.weight",
                          p + "self_attn.k_proj.weight",
                          p + "self_attn.v_proj.weight"});  // [QDim+2*KvDim, H]
    L.o_proj = UploadBf16(p + "self_attn.o_proj.weight");
    L.q_norm = UploadNorm(p + "self_attn.q_norm.weight");
    L.k_norm = UploadNorm(p + "self_attn.k_norm.weight");
    L.gateup = UploadConcat({p + "mlp.gate_proj.weight",
                             p + "mlp.up_proj.weight"});  // [2*intermediate, H]
    L.down = UploadBf16(p + "mlp.down_proj.weight");
    L.in_norm = UploadNorm(p + "input_layernorm.weight");
    L.post_norm = UploadNorm(p + "post_attention_layernorm.weight");
    if ((l + 1) % 8 == 0 || l + 1 == spec_.num_layers)
      LOG_INFO("[model] uploaded layer %d/%d", l + 1, spec_.num_layers);
  }

  // Activation scratch is sized to max_rows_; sampling buffers to kMaxDecodeB
  // (at most one logits row per request).
  int H = spec_.hidden_size, QD = spec_.QDim(), I = spec_.intermediate,
      V = spec_.vocab_size;
  int kvd = spec_.KvDim();
  int QKV = QD + 2 * kvd;  // fused Q|K|V projection width
  auto bmalloc = [&](bf16** p, size_t n) {
    CUDA_CHECK(cudaMalloc(p, n * sizeof(bf16)));
  };
  bmalloc(&x_, (size_t)max_rows_ * H);
  bmalloc(&xb_, (size_t)max_rows_ * H);
  bmalloc(&xb2_, (size_t)max_rows_ * H);
  bmalloc(&qkv_, (size_t)max_rows_ * QKV);
  bmalloc(&attn_, (size_t)max_rows_ * QD);
  bmalloc(&gateup_, (size_t)max_rows_ * 2 * I);
  bmalloc(&hmlp_, (size_t)max_rows_ * I);
  bmalloc(&xg_, (size_t)kMaxDecodeB * H);  // gathered sampling rows
  CUDA_CHECK(cudaMalloc(
      &logits_, (size_t)kMaxDecodeB * V * sizeof(float)));  // [S, vocab] FP32
  CUDA_CHECK(cudaMalloc(&d_ids_, max_rows_ * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_pos_, max_rows_ * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_req_, max_rows_ * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_qstart_, kMaxDecodeB * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_qlen_, kMaxDecodeB * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_decode_rids_, kMaxDecodeB * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_prefill_rids_, kMaxDecodeB * sizeof(int)));
  // Paged-KV block tables: up to kMaxDecodeB requests, padded to the longest.
  max_blocks_ = (max_ctx_ + BlockPool::kBlock - 1) / BlockPool::kBlock;
  CUDA_CHECK(
      cudaMalloc(&d_bt_, (size_t)kMaxDecodeB * max_blocks_ * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_lrows_, kMaxDecodeB * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_arg_, kMaxDecodeB * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_invT_, kMaxDecodeB * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_topp_, kMaxDecodeB * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_u_, kMaxDecodeB * sizeof(float)));
  host_bt_.resize((size_t)kMaxDecodeB * max_blocks_);
  host_qstart_.resize(kMaxDecodeB);
  host_qlen_.resize(kMaxDecodeB);
  host_decode_rids_.resize(kMaxDecodeB);
  host_prefill_rids_.resize(kMaxDecodeB);
  host_invT_.resize(kMaxDecodeB);
  host_topp_.resize(kMaxDecodeB);
  host_u_.resize(kMaxDecodeB);
  CUDA_CHECK(cudaStreamCreate(&stream_));
  CUBLAS_CHECK(cublasCreate(&cublas_));
  CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
  CUBLAS_CHECK(cublasSetMathMode(
      cublas_, CUBLAS_DEFAULT_MATH));  // BF16 inputs use tensor cores

  size_t freeb, totalb;
  CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
  LOG_INFO(
      "[model] weights + activations ready (max_ctx=%d, max_rows=%d, up to %d "
      "concurrent). "
      "GPU mem: %.1f GB used / %.1f GB total. KV pool allocated separately.",
      max_ctx_, max_rows_, MaxBatch(), (totalb - freeb) / 1e9, totalb / 1e9);
}

// Unified layer stack over T query rows (prefill chunks + decodes together, no
// split). Per-token ops run on the flattened batch; attention resolves each
// request's rows (qstart/qlen), positions (d_pos_) and KV (d_bt_) itself.
// R = request count, max_qlen = longest per-request row count (attention grid).
void ModelRuntime::RunLayers(int T, int R, int max_qlen) {
  const ModelSpec& c = spec_;
  int H = c.hidden_size, QD = c.QDim(), KVD = c.KvDim(), I = c.intermediate;
  int QKV = QD + 2 * KVD;
  int nH = c.num_heads, nKV = c.num_kv_heads, hd = c.head_dim;
  float eps = c.rms_eps, scale = 1.0f / std::sqrt((float)hd);
  cudaStream_t s = stream_;
  int blk = pool_->BlockSize();

  LaunchEmbed(d_ids_, embed_, x_, T, H, s);  // x_ = residual stream

  for (int l = 0; l < c.num_layers; ++l) {
    Layer& L = layers_[l];

    // Input norm fuses the previous layer's pending MLP residual (xb2_). Layer
    // 0 has none (x_ is the fresh embedding), so it's a plain rmsnorm.
    if (l == 0)
      LaunchRmsnorm(x_, L.in_norm, xb_, T, H, eps, s);
    else
      LaunchAddRmsnorm(x_, xb2_, L.in_norm, xb_, T, H, eps, s);

    // Fused Q|K|V projection -> qkv_ [T, QKV] (row = q(QD) | k(KVD) | v(KVD)).
    LaunchGemm(cublas_, xb_, L.qkv, qkv_, T, H, QKV, CUDA_R_16BF);
    // Per-head RMSNorm + RoPE on the q and k slices in place (v left raw).
    LaunchHeadNormRope(qkv_, L.q_norm, cos_tab_, sin_tab_, d_pos_, T, nH, hd,
                       QKV, eps, s);
    LaunchHeadNormRope(qkv_ + QD, L.k_norm, cos_tab_, sin_tab_, d_pos_, T, nKV,
                       hd, QKV, eps, s);

    // Append each row's new K/V at its position, read from the fused buffer.
    LaunchStoreKvPaged(qkv_, QD, QKV, pool_->K(l), d_bt_, bt_stride_, blk, KVD,
                       d_req_, d_pos_, T, s);
    LaunchStoreKvPaged(qkv_, QD + KVD, QKV, pool_->V(l), d_bt_, bt_stride_, blk,
                       KVD, d_req_, d_pos_, T, s);
    // Attention dispatched per request type; both write disjoint rows of attn_.
    LaunchAttnDecode(qkv_, QKV, pool_->K(l), pool_->V(l), attn_, nH, nKV, hd,
                     d_pos_, d_qstart_, d_decode_rids_, n_decode_, d_bt_,
                     bt_stride_, blk, scale, s);
    LaunchAttnPrefill(qkv_, QKV, pool_->K(l), pool_->V(l), attn_, nH, nKV, hd,
                      d_pos_, d_qstart_, d_qlen_, d_prefill_rids_, n_prefill_,
                      prefill_max_qlen_, d_bt_, bt_stride_, blk, scale, s);

    LaunchGemm(cublas_, attn_, L.o_proj, xb2_, T, QD, H, CUDA_R_16BF);
    // Post-attention norm fuses the attention residual (xb2_); x_ now carries
    // it.
    LaunchAddRmsnorm(x_, xb2_, L.post_norm, xb_, T, H, eps, s);

    // Fused gate|up -> gateup_ [T, 2I]; SiLU(gate)*up; down projection.
    LaunchGemm(cublas_, xb_, L.gateup, gateup_, T, H, 2 * I, CUDA_R_16BF);
    LaunchSiluMul(gateup_, hmlp_, T, I, s);
    LaunchGemm(cublas_, hmlp_, L.down, xb2_, T, I, H,
               CUDA_R_16BF);  // xb2_ = MLP residual, fused into next norm
  }

  LaunchAdd(x_, xb2_, T * H,
            s);  // commit the last layer's MLP residual (no norm follows)
}

// Pack R block tables, padded to the longest, into host_bt_ and upload to
// d_bt_; bt_stride_ is the padded row length the kernels index with.
void ModelRuntime::UploadBlockTables(const std::vector<std::vector<int>>& bts) {
  int R = (int)bts.size();
  int mb = 1;
  for (auto& t : bts) mb = std::max(mb, (int)t.size());
  bt_stride_ = mb;
  for (int r = 0; r < R; ++r) {
    int* row = host_bt_.data() + (size_t)r * mb;
    for (int g = 0; g < mb; ++g)
      row[g] = g < (int)bts[r].size() ? bts[r][g] : 0;
  }
  CUDA_CHECK(cudaMemcpy(d_bt_, host_bt_.data(), (size_t)R * mb * sizeof(int),
                        cudaMemcpyHostToDevice));
}

// Forward: H2D inputs, unified layer stack, then gather only the per-request
// sampling rows and run final norm + lm_head on those -> logits_ [S, vocab],
// then sample per row into out_tokens[S]. Async on stream_, synced once at end.
void ModelRuntime::Forward(const ForwardInput& in,
                           std::vector<int>& out_tokens) {
  int T = in.NumRows(), R = (int)in.block_tables.size(),
      S = (int)in.logits_rows.size();
  int H = spec_.hidden_size, V = spec_.vocab_size;
  CUDA_CHECK(cudaMemcpy(d_ids_, in.tokens.data(), T * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pos_, in.positions.data(), T * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_req_, in.req_index.data(), T * sizeof(int),
                        cudaMemcpyHostToDevice));
  UploadBlockTables(in.block_tables);

  // Per-request attention grouping: rows are contiguous and ordered by request,
  // so qstart is a prefix sum of the per-request row counts.
  for (int r = 0; r < R; ++r) host_qlen_[r] = 0;
  for (int t = 0; t < T; ++t) host_qlen_[in.req_index[t]]++;
  int acc = 0, max_qlen = 1;
  for (int r = 0; r < R; ++r) {
    host_qstart_[r] = acc;
    acc += host_qlen_[r];
    max_qlen = std::max(max_qlen, host_qlen_[r]);
  }
  CUDA_CHECK(cudaMemcpy(d_qstart_, host_qstart_.data(), R * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_qlen_, host_qlen_.data(), R * sizeof(int),
                        cudaMemcpyHostToDevice));

  // Split requests by type so attention runs the right kernel: decode
  // (q_len==1) uses the FlashDecoding kernel, prefill (q_len>1) the tiled
  // kernel.
  n_decode_ = n_prefill_ = 0;
  prefill_max_qlen_ = 1;
  for (int r = 0; r < R; ++r) {
    if (host_qlen_[r] == 1)
      host_decode_rids_[n_decode_++] = r;
    else {
      host_prefill_rids_[n_prefill_++] = r;
      prefill_max_qlen_ = std::max(prefill_max_qlen_, host_qlen_[r]);
    }
  }
  if (n_decode_)
    CUDA_CHECK(cudaMemcpy(d_decode_rids_, host_decode_rids_.data(),
                          n_decode_ * sizeof(int), cudaMemcpyHostToDevice));
  if (n_prefill_)
    CUDA_CHECK(cudaMemcpy(d_prefill_rids_, host_prefill_rids_.data(),
                          n_prefill_ * sizeof(int), cudaMemcpyHostToDevice));

  RunLayers(T, R, max_qlen);

  if (S > 0) {
    CUDA_CHECK(cudaMemcpy(d_lrows_, in.logits_rows.data(), S * sizeof(int),
                          cudaMemcpyHostToDevice));
    LaunchGatherRows(x_, d_lrows_, xg_, S, H, stream_);
    LaunchRmsnorm(xg_, fnorm_, xb_, S, H, spec_.rms_eps, stream_);
    LaunchGemm(cublas_, xb_, lm_head_, logits_, S, H, V,
               CUDA_R_32F);  // BF16 in -> FP32 logits
  }

  out_tokens.resize(S);
  if (S > 0) {
    // Per-row sampling inputs: 1/temp (<=0 => greedy), nucleus cutoff, uniform
    // draw.
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < S; ++i) {
      const SampleParams& sp = in.sample_params[i];
      host_invT_[i] = sp.temp > 0.0f ? 1.0f / sp.temp : 0.0f;
      host_topp_[i] = sp.top_p;
      host_u_[i] = dist(rng_);
    }
    CUDA_CHECK(cudaMemcpyAsync(d_invT_, host_invT_.data(), S * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_topp_, host_topp_.data(), S * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_u_, host_u_.data(), S * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    LaunchSampleBatch(logits_, S, spec_.vocab_size, d_invT_, d_topp_, d_u_,
                      d_arg_, stream_);
    CUDA_CHECK(cudaMemcpyAsync(out_tokens.data(), d_arg_, S * sizeof(int),
                               cudaMemcpyDeviceToHost, stream_));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

ModelRuntime::~ModelRuntime() {
  for (auto p : all_bufs_)
    if (p) cudaFree(p);
  for (auto p : all_fbufs_)
    if (p) cudaFree(p);
  cudaFree(cos_tab_);
  cudaFree(sin_tab_);
  cudaFree(x_);
  cudaFree(xb_);
  cudaFree(xb2_);
  cudaFree(qkv_);
  cudaFree(attn_);
  cudaFree(gateup_);
  cudaFree(hmlp_);
  cudaFree(xg_);
  cudaFree(logits_);
  cudaFree(d_ids_);
  cudaFree(d_pos_);
  cudaFree(d_req_);
  cudaFree(d_qstart_);
  cudaFree(d_qlen_);
  cudaFree(d_decode_rids_);
  cudaFree(d_prefill_rids_);
  cudaFree(d_bt_);
  cudaFree(d_lrows_);
  cudaFree(d_arg_);
  cudaFree(d_invT_);
  cudaFree(d_topp_);
  cudaFree(d_u_);
  if (cublas_) cublasDestroy(cublas_);
  if (stream_) cudaStreamDestroy(stream_);
}
