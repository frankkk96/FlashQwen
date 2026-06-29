#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "attn_cute.h"
#include "log.h"
#include "model_runtime.h"

// Upload a tensor to the device as BF16, unchanged.
DeviceBuffer<bf16> ModelRuntime::UploadBf16(const std::string& name) {
  const TensorView& tv = st_.Get(name);
  DeviceBuffer<bf16> d(tv.nbytes / sizeof(bf16));
  CUDA_CHECK(cudaMemcpy(d.Get(), tv.data, tv.nbytes, cudaMemcpyHostToDevice));
  return d;
}

// Upload several BF16 tensors stacked on their OUT (row) dim into one buffer —
// the contiguous [sum(OUT_i), IN] layout a fused projection (QKV, gate|up)
// needs.
DeviceBuffer<bf16> ModelRuntime::UploadBf16s(
    const std::vector<std::string>& names) {
  size_t total = 0;
  for (auto& n : names) total += st_.Get(n).nbytes;
  DeviceBuffer<bf16> d(total / sizeof(bf16));
  size_t off = 0;
  for (auto& n : names) {
    const TensorView& tv = st_.Get(n);
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<char*>(d.Get()) + off, tv.data,
                          tv.nbytes, cudaMemcpyHostToDevice));
    off += tv.nbytes;
  }
  return d;
}

// Upload a BF16 tensor to the device widened to FP32.
DeviceBuffer<float> ModelRuntime::UploadFp32(const std::string& name) {
  const TensorView& tv = st_.Get(name);
  int64_t n = tv.Numel();
  std::vector<float> tmp(n);
  const uint16_t* src = reinterpret_cast<const uint16_t*>(tv.data);
  for (int64_t i = 0; i < n; ++i) {
    uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
    std::memcpy(&tmp[i], &bits, 4);
  }
  DeviceBuffer<float> d(n);
  CUDA_CHECK(cudaMemcpy(d.Get(), tmp.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));
  return d;
}

// Precompute layer-independent RoPE cos/sin tables [max_ctx, head_dim/2] FP32:
// angle(pos,i) = pos * theta^(-2i/head_dim), i in [0, head_dim/2).
void ModelRuntime::PrecomputeRope() {
  int half = spec_.head_dim / 2, P = max_ctx_;
  std::vector<float> cs(static_cast<size_t>(P) * half),
      sn(static_cast<size_t>(P) * half);
  for (int p = 0; p < P; ++p)
    for (int i = 0; i < half; ++i) {
      float inv = std::pow(spec_.rope_theta, -2.0f * i / spec_.head_dim);
      float ang = p * inv;
      cs[p * half + i] = std::cos(ang);
      sn[p * half + i] = std::sin(ang);
    }
  d_cos_tab_ = DeviceBuffer<float>(cs.size());
  d_sin_tab_ = DeviceBuffer<float>(sn.size());
  CUDA_CHECK(cudaMemcpy(d_cos_tab_.Get(), cs.data(), cs.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sin_tab_.Get(), sn.data(), sn.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}

// Load weights from spec.dir (norms widened to FP32), precompute RoPE, and
// allocate activation + sampling scratch. Scratch rows are sized to
// token_budget (+16 slack for a prefill kernel's last-tile over-read), NOT
// max_ctx: sizing to max_ctx over-allocates ~4x and starves the KV pool.
ModelRuntime::ModelRuntime(const ModelSpec& spec, const KvStore& store,
                           int max_ctx, int slots, int token_budget,
                           unsigned seed)
    : spec_(spec), rng_(seed), store_(&store) {
  max_ctx_ = max_ctx;
  slots_ = slots;
  max_rows_ = std::max(1, token_budget) + 16;
  LOG_INFO("[model] loading weights from %s ...", spec_.dir.c_str());
  st_.LoadDir(spec_.dir);

  d_embed_ = UploadBf16("model.embed_tokens.weight");
  d_fnorm_ = UploadFp32("model.norm.weight");
  d_lm_head_ = UploadBf16("lm_head.weight");
  PrecomputeRope();

  layers_.resize(spec_.num_layers);
  for (int l = 0; l < spec_.num_layers; l++) {
    std::string p = "model.layers." + std::to_string(l) + ".";
    auto attn = [&](const char* n) { return p + "self_attn." + n; };
    auto mlp = [&](const char* n) { return p + "mlp." + n; };
    Layer& L = layers_[l];
    L.d_qkv = UploadBf16s(
        {attn("q_proj.weight"), attn("k_proj.weight"), attn("v_proj.weight")});
    L.d_o_proj = UploadBf16(attn("o_proj.weight"));
    L.d_q_norm = UploadFp32(attn("q_norm.weight"));
    L.d_k_norm = UploadFp32(attn("k_norm.weight"));
    L.d_gateup = UploadBf16s({mlp("gate_proj.weight"), mlp("up_proj.weight")});
    L.d_down = UploadBf16(mlp("down_proj.weight"));
    L.d_in_norm = UploadFp32(p + "input_layernorm.weight");
    L.d_post_norm = UploadFp32(p + "post_attention_layernorm.weight");
    if ((l + 1) % 8 == 0 || l + 1 == spec_.num_layers)
      LOG_INFO("[model] uploaded layer %d/%d", l + 1, spec_.num_layers);
  }

  int H = spec_.hidden_size, QD = spec_.QDim(), I = spec_.intermediate,
      V = spec_.vocab_size;
  int kvd = spec_.KvDim();
  int QKV = QD + 2 * kvd;
  size_t rows = static_cast<size_t>(max_rows_);
  d_x_ = DeviceBuffer<bf16>(rows * H);
  d_xb_ = DeviceBuffer<bf16>(rows * H);
  d_xb2_ = DeviceBuffer<bf16>(rows * H);
  d_qkv_ = DeviceBuffer<bf16>(rows * QKV);
  d_attn_ = DeviceBuffer<bf16>(rows * QD);
  d_gateup_ = DeviceBuffer<bf16>(rows * 2 * I);
  d_hmlp_ = DeviceBuffer<bf16>(rows * I);
  d_xg_ = DeviceBuffer<bf16>(static_cast<size_t>(slots_) * H);
  d_logits_ = DeviceBuffer<float>(static_cast<size_t>(slots_) * V);
  d_ids_ = DeviceBuffer<int>(max_rows_);
  d_pos_ = DeviceBuffer<int>(max_rows_);
  d_req_ = DeviceBuffer<int>(max_rows_);
  d_qstart_ = DeviceBuffer<int>(slots_);
  d_qlen_ = DeviceBuffer<int>(slots_);
  d_decode_rids_ = DeviceBuffer<int>(slots_);
  d_prefill_rids_ = DeviceBuffer<int>(slots_);
  max_blocks_ = (max_ctx_ + kKvBlock - 1) / kKvBlock;
  d_bt_ = DeviceBuffer<int>(static_cast<size_t>(slots_) * max_blocks_);
  d_lrows_ = DeviceBuffer<int>(slots_);
  d_arg_ = DeviceBuffer<int>(slots_);
  d_invT_ = DeviceBuffer<float>(slots_);
  d_topp_ = DeviceBuffer<float>(slots_);
  d_u_ = DeviceBuffer<float>(slots_);
  h_bt_.resize(static_cast<size_t>(slots_) * max_blocks_);
  h_qstart_.resize(slots_);
  h_qlen_.resize(slots_);
  h_decode_rids_.resize(slots_);
  h_prefill_rids_.resize(slots_);
  h_invT_.resize(slots_);
  h_topp_.resize(slots_);
  h_u_.resize(slots_);
  CUDA_CHECK(cudaStreamCreate(&stream_));
  CUBLAS_CHECK(cublasCreate(&cublas_));
  CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
  CUBLAS_CHECK(cublasSetMathMode(cublas_, CUBLAS_DEFAULT_MATH));

  size_t freeb, totalb;
  CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
  LOG_INFO(
      "[model] weights + activations ready (max_ctx=%d, max_rows=%d, up to %d "
      "concurrent). "
      "GPU mem: %.1f GB used / %.1f GB total. KV pool allocated separately.",
      max_ctx_, max_rows_, slots_, (totalb - freeb) / 1e9, totalb / 1e9);
}

// Unified layer stack over all query rows (prefill chunks + decodes together,
// no split). Per-token ops run on the flattened batch; attention resolves each
// request's rows (qstart/qlen), positions (d_pos_) and KV (d_bt_) itself.
void ModelRuntime::RunLayers(const ForwardInput& in) {
  int T = in.NumRows();
  const ModelSpec& c = spec_;
  int H = c.hidden_size, QD = c.QDim(), KVD = c.KvDim(), I = c.intermediate;
  int QKV = QD + 2 * KVD;
  int nH = c.num_heads, nKV = c.num_kv_heads, hd = c.head_dim;
  float eps = c.rms_eps, scale = 1.0f / std::sqrt(static_cast<float>(hd));
  cudaStream_t s = stream_;
  int blk = store_->BlockSize();

  LaunchEmbed(d_ids_.Get(), d_embed_.Get(), d_x_.Get(), T, H, s);

  for (int l = 0; l < c.num_layers; ++l) {
    Layer& L = layers_[l];

    if (l == 0)
      LaunchRmsnorm(d_x_.Get(), L.d_in_norm.Get(), d_xb_.Get(), T, H, eps, s);
    else
      LaunchAddRmsnorm(d_x_.Get(), d_xb2_.Get(), L.d_in_norm.Get(), d_xb_.Get(),
                       T, H, eps, s);

    LaunchGemm(cublas_, d_xb_.Get(), L.d_qkv.Get(), d_qkv_.Get(), T, H, QKV,
               CUDA_R_16BF);
    LaunchHeadNormRope(d_qkv_.Get(), L.d_q_norm.Get(), L.d_k_norm.Get(),
                       d_cos_tab_.Get(), d_sin_tab_.Get(), d_pos_.Get(), T, nH,
                       nKV, hd, QKV, eps, s);

    store_->StoreKV(l, d_qkv_.Get(), d_bt_.Get(), bt_stride_, d_req_.Get(),
                    d_pos_.Get(), T, s);
    // CuTe (CUTLASS) attention path, opt-in via FQ_ATTN_CUTE=1; default is the
    // hand-written kernels. Read once (env is process-constant).
    static const bool kCuteAttn = [] {
      const char* e = std::getenv("FQ_ATTN_CUTE");
      return e && e[0] == '1';
    }();
    auto decode = kCuteAttn ? LaunchAttnDecodeCute : LaunchAttnDecode;
    decode(d_qkv_.Get(), QKV, store_->KV(l), d_attn_.Get(), nH, nKV, hd,
           d_pos_.Get(), d_qstart_.Get(), d_decode_rids_.Get(), n_decode_,
           d_bt_.Get(), bt_stride_, blk, scale, slots_, s);
    auto prefill = kCuteAttn ? LaunchAttnPrefillCute : LaunchAttnPrefill;
    prefill(d_qkv_.Get(), QKV, store_->KV(l), d_attn_.Get(), nH, nKV, hd,
            d_pos_.Get(), d_qstart_.Get(), d_qlen_.Get(), d_prefill_rids_.Get(),
            n_prefill_, prefill_max_qlen_, d_bt_.Get(), bt_stride_, blk, scale,
            s);

    LaunchGemm(cublas_, d_attn_.Get(), L.d_o_proj.Get(), d_xb2_.Get(), T, QD, H,
               CUDA_R_16BF);
    LaunchAddRmsnorm(d_x_.Get(), d_xb2_.Get(), L.d_post_norm.Get(), d_xb_.Get(),
                     T, H, eps, s);

    LaunchGemm(cublas_, d_xb_.Get(), L.d_gateup.Get(), d_gateup_.Get(), T, H,
               2 * I, CUDA_R_16BF);
    LaunchSiluMul(d_gateup_.Get(), d_hmlp_.Get(), T, I, s);
    LaunchGemm(cublas_, d_hmlp_.Get(), L.d_down.Get(), d_xb2_.Get(), T, I, H,
               CUDA_R_16BF);
  }

  LaunchAdd(d_x_.Get(), d_xb2_.Get(), T * H, s);
}

// Upload this forward's per-row inputs (token ids, positions, req index) and
// the block tables, packed padded-to-the-longest into h_bt_ then copied to
// d_bt_; bt_stride_ is the padded row length the kernels index with.
void ModelRuntime::UploadInputs(const ForwardInput& in) {
  int T = in.NumRows();
  CUDA_CHECK(cudaMemcpy(d_ids_.Get(), in.tokens.data(), T * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pos_.Get(), in.positions.data(), T * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_req_.Get(), in.req_index.data(), T * sizeof(int),
                        cudaMemcpyHostToDevice));

  const auto& bts = in.block_tables;
  int R = static_cast<int>(bts.size());
  int mb = 1;
  for (auto& t : bts) mb = std::max(mb, static_cast<int>(t.size()));
  bt_stride_ = mb;
  for (int r = 0; r < R; ++r) {
    int* row = h_bt_.data() + r * mb;
    for (int g = 0; g < mb; ++g)
      row[g] = g < static_cast<int>(bts[r].size()) ? bts[r][g] : 0;
  }
  CUDA_CHECK(cudaMemcpy(d_bt_.Get(), h_bt_.data(),
                        static_cast<size_t>(R) * mb * sizeof(int),
                        cudaMemcpyHostToDevice));
}

// Per-request attention grouping over the batch. Rows are contiguous and
// ordered by request, so qstart is a prefix sum of the per-request row counts.
// Requests are split by type so attention runs the right kernel: decode
// (q_len==1) uses the FlashDecoding kernel, prefill (q_len>1) the tiled kernel.
void ModelRuntime::GroupRequests(const ForwardInput& in) {
  int T = in.NumRows(), R = static_cast<int>(in.block_tables.size());
  for (int r = 0; r < R; ++r) h_qlen_[r] = 0;
  for (int t = 0; t < T; ++t) h_qlen_[in.req_index[t]]++;
  int acc = 0;
  for (int r = 0; r < R; ++r) {
    h_qstart_[r] = acc;
    acc += h_qlen_[r];
  }
  CUDA_CHECK(cudaMemcpy(d_qstart_.Get(), h_qstart_.data(), R * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_qlen_.Get(), h_qlen_.data(), R * sizeof(int),
                        cudaMemcpyHostToDevice));

  n_decode_ = n_prefill_ = 0;
  prefill_max_qlen_ = 1;
  for (int r = 0; r < R; ++r) {
    if (h_qlen_[r] == 1)
      h_decode_rids_[n_decode_++] = r;
    else {
      h_prefill_rids_[n_prefill_++] = r;
      prefill_max_qlen_ = std::max(prefill_max_qlen_, h_qlen_[r]);
    }
  }
  if (n_decode_)
    CUDA_CHECK(cudaMemcpy(d_decode_rids_.Get(), h_decode_rids_.data(),
                          n_decode_ * sizeof(int), cudaMemcpyHostToDevice));
  if (n_prefill_)
    CUDA_CHECK(cudaMemcpy(d_prefill_rids_.Get(), h_prefill_rids_.data(),
                          n_prefill_ * sizeof(int), cudaMemcpyHostToDevice));
}

// Forward: H2D inputs, unified layer stack, then gather only the per-request
// sampling rows and run final norm + lm_head on those -> d_logits_ [S, vocab],
// then sample per row into out_tokens[S]. Async on stream_, synced once at end.
void ModelRuntime::Forward(const ForwardInput& in,
                           std::vector<int>& out_tokens) {
  UploadInputs(in);
  GroupRequests(in);
  RunLayers(in);
  RunHeadAndSample(in, out_tokens);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// Gather only the per-request sampling rows, run final norm + lm_head on those
// -> d_logits_ [S, vocab], then sample one token per row into out_tokens[S].
void ModelRuntime::RunHeadAndSample(const ForwardInput& in,
                                    std::vector<int>& out_tokens) {
  int S = static_cast<int>(in.logits_rows.size());
  int H = spec_.hidden_size, V = spec_.vocab_size;
  out_tokens.resize(S);
  if (S == 0) return;

  CUDA_CHECK(cudaMemcpy(d_lrows_.Get(), in.logits_rows.data(), S * sizeof(int),
                        cudaMemcpyHostToDevice));
  LaunchGatherRows(d_x_.Get(), d_lrows_.Get(), d_xg_.Get(), S, H, stream_);
  LaunchRmsnorm(d_xg_.Get(), d_fnorm_.Get(), d_xb_.Get(), S, H, spec_.rms_eps,
                stream_);
  LaunchGemm(cublas_, d_xb_.Get(), d_lm_head_.Get(), d_logits_.Get(), S, H, V,
             CUDA_R_32F);

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int i = 0; i < S; ++i) {
    const SampleParams& sp = in.sample_params[i];
    h_invT_[i] = sp.temp > 0.0f ? 1.0f / sp.temp : 0.0f;
    h_topp_[i] = sp.top_p;
    h_u_[i] = dist(rng_);
  }
  CUDA_CHECK(cudaMemcpyAsync(d_invT_.Get(), h_invT_.data(), S * sizeof(float),
                             cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_topp_.Get(), h_topp_.data(), S * sizeof(float),
                             cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_u_.Get(), h_u_.data(), S * sizeof(float),
                             cudaMemcpyHostToDevice, stream_));
  LaunchSampleBatch(d_logits_.Get(), S, spec_.vocab_size, d_invT_.Get(),
                    d_topp_.Get(), d_u_.Get(), d_arg_.Get(), stream_);
  CUDA_CHECK(cudaMemcpyAsync(out_tokens.data(), d_arg_.Get(), S * sizeof(int),
                             cudaMemcpyDeviceToHost, stream_));
}

// Each DeviceBuffer member frees its own device memory; only the cuBLAS/stream
// handles, which are not plain allocations, need explicit teardown.
ModelRuntime::~ModelRuntime() {
  if (cublas_) cublasDestroy(cublas_);
  if (stream_) cudaStreamDestroy(stream_);
}
