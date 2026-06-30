#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "attn_cute.h"
#include "log.h"
#include "model_runtime.h"

namespace fq {

namespace {
inline float Bf16ToFloat(uint16_t b) {
  uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

template <class T>
DeviceBuffer<T> UploadTensors(SafeTensors& st,
                              const std::vector<std::string>& names) {
  size_t total = 0;
  for (auto& name : names) total += st.Get(name).Numel();
  DeviceBuffer<T> d(total);
  size_t off = 0;
  for (auto& name : names) {
    const TensorView& tensor = st.Get(name);
    int64_t count = tensor.Numel();
    if constexpr (std::is_same_v<T, float>) {
      const uint16_t* src = reinterpret_cast<const uint16_t*>(tensor.data);
      std::vector<float> tmp(count);
      for (int64_t i = 0; i < count; ++i) tmp[i] = Bf16ToFloat(src[i]);
      d.Upload(tmp.data(), count, off);
    } else {
      d.Upload(reinterpret_cast<const T*>(tensor.data), count, off);
    }
    off += count;
  }
  return d;
}

constexpr int kRowSlack = 16;

void UploadRows(StepContext& ctx, const ForwardInput& in, cudaStream_t stream) {
  int n = ctx.n_rows;
  ctx.ids.Upload(in.tokens.data(), n, stream);
  ctx.pos.Upload(in.positions.data(), n, stream);
  ctx.req.Upload(in.req_index.data(), n, stream);
}

void PackBlockTables(StepContext& ctx, const ForwardInput& in,
                     cudaStream_t stream) {
  int n_req = static_cast<int>(in.block_tables.size());
  ctx.bt_stride = ctx.max_blocks;
  for (int r = 0; r < n_req; ++r) {
    int* row = ctx.bt.H() + r * ctx.bt_stride;
    const auto& blocks = in.block_tables[r];
    for (int g = 0; g < static_cast<int>(blocks.size()); ++g) row[g] = blocks[g];
  }
  ctx.bt.Flush(stream, static_cast<size_t>(n_req) * ctx.bt_stride);
}

void SplitDecodePrefill(StepContext& ctx, const ForwardInput& in,
                        cudaStream_t stream) {
  int n_req = static_cast<int>(in.block_tables.size());
  for (int r = 0; r < n_req; ++r) ctx.qlen[r] = 0;
  for (int t = 0; t < ctx.n_rows; ++t) ctx.qlen[in.req_index[t]]++;
  int acc = 0;
  for (int r = 0; r < n_req; ++r) {
    ctx.qstart[r] = acc;
    acc += ctx.qlen[r];
  }
  ctx.qstart.Flush(stream, n_req);
  ctx.qlen.Flush(stream, n_req);

  ctx.n_decode = ctx.n_prefill = 0;
  ctx.prefill_max_qlen = 1;
  for (int r = 0; r < n_req; ++r) {
    if (ctx.qlen[r] == 1)
      ctx.decode_rids[ctx.n_decode++] = r;
    else {
      ctx.prefill_rids[ctx.n_prefill++] = r;
      ctx.prefill_max_qlen = std::max(ctx.prefill_max_qlen, ctx.qlen[r]);
    }
  }
  if (ctx.n_decode) ctx.decode_rids.Flush(stream, ctx.n_decode);
  if (ctx.n_prefill) ctx.prefill_rids.Flush(stream, ctx.n_prefill);
}

void FillSampling(StepContext& ctx, const ForwardInput& in, std::mt19937& rng,
                  cudaStream_t stream) {
  ctx.n_sample = static_cast<int>(in.logits_rows.size());
  int n = ctx.n_sample;
  ctx.lrows.Upload(in.logits_rows.data(), n, stream);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int i = 0; i < n; ++i) {
    const SampleParams& sp = in.sample_params[i];
    ctx.invT[i] = sp.temp > 0.0f ? 1.0f / sp.temp : 0.0f;
    ctx.topp[i] = sp.top_p;
    ctx.u[i] = dist(rng);
  }
  ctx.invT.Flush(stream, n);
  ctx.topp.Flush(stream, n);
  ctx.u.Flush(stream, n);
}
}

void RopeTables::Build(const ModelSpec& spec, int max_ctx) {
  int half = spec.head_dim / 2, n_pos = max_ctx;
  std::vector<float> cos_tab(static_cast<size_t>(n_pos) * half),
      sin_tab(static_cast<size_t>(n_pos) * half);
  for (int p = 0; p < n_pos; ++p)
    for (int i = 0; i < half; ++i) {
      float inv = std::pow(spec.rope_theta, -2.0f * i / spec.head_dim);
      float ang = p * inv;
      cos_tab[p * half + i] = std::cos(ang);
      sin_tab[p * half + i] = std::sin(ang);
    }
  cos_ = DeviceBuffer<float>(cos_tab.size());
  sin_ = DeviceBuffer<float>(sin_tab.size());
  cos_.Upload(cos_tab.data(), cos_tab.size());
  sin_.Upload(sin_tab.data(), sin_tab.size());
}

ModelWeights::ModelWeights(const ModelSpec& spec, int max_ctx) {
  LOG_INFO("[model] loading weights from %s ...", spec.dir.c_str());
  SafeTensors st;
  st.LoadDir(spec.dir);

  embed = UploadTensors<bf16>(st, {"model.embed_tokens.weight"});
  fnorm = UploadTensors<float>(st, {"model.norm.weight"});
  lm_head = UploadTensors<bf16>(st, {"lm_head.weight"});
  rope.Build(spec, max_ctx);

  layers.resize(spec.num_layers);
  for (int l = 0; l < spec.num_layers; l++) {
    std::string p = "model.layers." + std::to_string(l) + ".";
    auto attn = [&](const char* n) { return p + "self_attn." + n; };
    auto mlp = [&](const char* n) { return p + "mlp." + n; };
    Layer& layer = layers[l];
    layer.qkv = UploadTensors<bf16>(
        st, {attn("q_proj.weight"), attn("k_proj.weight"), attn("v_proj.weight")});
    layer.o_proj = UploadTensors<bf16>(st, {attn("o_proj.weight")});
    layer.q_norm = UploadTensors<float>(st, {attn("q_norm.weight")});
    layer.k_norm = UploadTensors<float>(st, {attn("k_norm.weight")});
    layer.gateup =
        UploadTensors<bf16>(st, {mlp("gate_proj.weight"), mlp("up_proj.weight")});
    layer.down = UploadTensors<bf16>(st, {mlp("down_proj.weight")});
    layer.in_norm = UploadTensors<float>(st, {p + "input_layernorm.weight"});
    layer.post_norm =
        UploadTensors<float>(st, {p + "post_attention_layernorm.weight"});
    if ((l + 1) % 8 == 0 || l + 1 == spec.num_layers)
      LOG_INFO("[model] uploaded layer %d/%d", l + 1, spec.num_layers);
  }
}

StepContext::StepContext(int max_rows, int slots, int max_ctx) {
  ids = DeviceBuffer<int>(max_rows);
  pos = DeviceBuffer<int>(max_rows);
  req = DeviceBuffer<int>(max_rows);
  lrows = DeviceBuffer<int>(slots);
  qstart = StagedBuffer<int>(slots);
  qlen = StagedBuffer<int>(slots);
  decode_rids = StagedBuffer<int>(slots);
  prefill_rids = StagedBuffer<int>(slots);
  max_blocks = (max_ctx + kKvBlock - 1) / kKvBlock;
  bt = StagedBuffer<int>(static_cast<size_t>(slots) * max_blocks);
  invT = StagedBuffer<float>(slots);
  topp = StagedBuffer<float>(slots);
  u = StagedBuffer<float>(slots);
}

RuntimeBuffers::RuntimeBuffers(const ModelSpec& spec, int max_rows, int slots) {
  int hidden = spec.hidden_size, q_dim = spec.QDim(), inter = spec.intermediate,
      vocab = spec.vocab_size;
  int kv_dim = spec.KvDim();
  int qkv_dim = q_dim + 2 * kv_dim;
  size_t rows = static_cast<size_t>(max_rows);
  x = DeviceBuffer<bf16>(rows * hidden);
  xb = DeviceBuffer<bf16>(rows * hidden);
  xb2 = DeviceBuffer<bf16>(rows * hidden);
  qkv = DeviceBuffer<bf16>(rows * qkv_dim);
  attn = DeviceBuffer<bf16>(rows * q_dim);
  gateup = DeviceBuffer<bf16>(rows * 2 * inter);
  hmlp = DeviceBuffer<bf16>(rows * inter);
  xg = DeviceBuffer<bf16>(static_cast<size_t>(slots) * hidden);
  logits = DeviceBuffer<float>(static_cast<size_t>(slots) * vocab);
  size_t dec_floats = DecodePartialFloats(slots);
  dec_pm = DeviceBuffer<float>(dec_floats);
  dec_pl = DeviceBuffer<float>(dec_floats);
  dec_pa = DeviceBuffer<float>(dec_floats * spec.head_dim);
  sampled = DeviceBuffer<int>(slots);
}

void ModelRuntime::InitCudaResources() {
  CUDA_CHECK(cudaStreamCreate(&stream_));
  CUBLAS_CHECK(cublasCreate(&cublas_));
  CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
  CUBLAS_CHECK(cublasSetMathMode(cublas_, CUBLAS_DEFAULT_MATH));
  cublas_ws_ = DeviceBuffer<std::byte>(kCublasWsBytes);
  CUBLAS_CHECK(cublasSetWorkspace(cublas_, cublas_ws_.D(), kCublasWsBytes));
}

ModelRuntime::ModelRuntime(const ModelSpec& spec, int max_ctx, int slots,
                           int token_budget, unsigned seed)
    : spec_(spec), rng_(seed) {
  max_ctx_ = max_ctx;
  slots_ = slots;
  max_rows_ = std::max(1, token_budget) + kRowSlack;
  weights_ = ModelWeights(spec_, max_ctx_);
  InitCudaResources();
  buf_ = RuntimeBuffers(spec_, max_rows_, slots_);
  ctx_ = StepContext(max_rows_, slots_, max_ctx_);

  size_t freeb, totalb;
  CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
  LOG_INFO(
      "[model] weights + activations ready (max_ctx=%d, max_rows=%d, up to %d "
      "concurrent). "
      "GPU mem: %.1f GB used / %.1f GB total. KV pool allocated separately.",
      max_ctx_, max_rows_, slots_, (totalb - freeb) / 1e9, totalb / 1e9);
}

void ModelRuntime::RunLayers() {
  if (ctx_.n_prefill == 0 && ctx_.n_decode > 0)
    layers_graph_.Run(stream_, ctx_.n_rows, ctx_.n_decode,
                      [this] { RunLayersBody(); });
  else
    RunLayersBody();
}

void ModelRuntime::RunLayersBody() {
  int n_rows = ctx_.n_rows;
  const ModelSpec& spec = spec_;
  int hidden = spec.hidden_size, q_dim = spec.QDim(), kv_dim = spec.KvDim(),
      inter = spec.intermediate;
  int qkv_dim = q_dim + 2 * kv_dim;
  int n_heads = spec.num_heads, n_kv = spec.num_kv_heads,
      head_dim = spec.head_dim;
  float eps = spec.rms_eps,
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  cudaStream_t stream = stream_;
  int block_size = store_->BlockSize();

  LaunchEmbed(ctx_.ids.D(), weights_.embed.D(), buf_.x.D(), n_rows, hidden,
              stream);

  for (int l = 0; l < spec.num_layers; ++l) {
    Layer& layer = weights_.layers[l];

    if (l == 0)
      LaunchRmsnorm(buf_.x.D(), layer.in_norm.D(), buf_.xb.D(), n_rows, hidden,
                    eps, stream);
    else
      LaunchAddRmsnorm(buf_.x.D(), buf_.xb2.D(), layer.in_norm.D(), buf_.xb.D(),
                       n_rows, hidden, eps, stream);

    LaunchGemm(cublas_, buf_.xb.D(), layer.qkv.D(), buf_.qkv.D(), n_rows, hidden,
               qkv_dim, CUDA_R_16BF);
    LaunchHeadNormRope(buf_.qkv.D(), layer.q_norm.D(), layer.k_norm.D(),
                       weights_.rope.Cos(), weights_.rope.Sin(), ctx_.pos.D(),
                       n_rows, n_heads, n_kv, head_dim, qkv_dim, eps, stream);

    store_->StoreKV(l, buf_.qkv.D(), ctx_.bt.D(), ctx_.bt_stride, ctx_.req.D(),
                    ctx_.pos.D(), n_rows, stream);
    LaunchAttnDecodeCute(buf_.qkv.D(), qkv_dim, store_->KV(l), buf_.attn.D(),
                         n_heads, n_kv, head_dim, ctx_.pos.D(), ctx_.qstart.D(),
                         ctx_.decode_rids.D(), ctx_.n_decode, ctx_.bt.D(),
                         ctx_.bt_stride, block_size, scale, buf_.dec_pm.D(),
                         buf_.dec_pl.D(), buf_.dec_pa.D(), stream);
    LaunchAttnPrefillCute(buf_.qkv.D(), qkv_dim, store_->KV(l), buf_.attn.D(),
                          n_heads, n_kv, head_dim, ctx_.pos.D(), ctx_.qstart.D(),
                          ctx_.qlen.D(), ctx_.prefill_rids.D(), ctx_.n_prefill,
                          ctx_.prefill_max_qlen, ctx_.bt.D(), ctx_.bt_stride,
                          block_size, scale, stream);

    LaunchGemm(cublas_, buf_.attn.D(), layer.o_proj.D(), buf_.xb2.D(), n_rows,
               q_dim, hidden, CUDA_R_16BF);
    LaunchAddRmsnorm(buf_.x.D(), buf_.xb2.D(), layer.post_norm.D(), buf_.xb.D(),
                     n_rows, hidden, eps, stream);

    LaunchGemm(cublas_, buf_.xb.D(), layer.gateup.D(), buf_.gateup.D(), n_rows,
               hidden, 2 * inter, CUDA_R_16BF);
    LaunchSiluMul(buf_.gateup.D(), buf_.hmlp.D(), n_rows, inter, stream);
    LaunchGemm(cublas_, buf_.hmlp.D(), layer.down.D(), buf_.xb2.D(), n_rows,
               inter, hidden, CUDA_R_16BF);
  }

  LaunchAdd(buf_.x.D(), buf_.xb2.D(), n_rows * hidden, stream);
}

void StepContext::PrepareInputs(const ForwardInput& in, std::mt19937& rng,
                                cudaStream_t stream) {
  n_rows = in.NumRows();
  UploadRows(*this, in, stream);
  PackBlockTables(*this, in, stream);
  SplitDecodePrefill(*this, in, stream);
  FillSampling(*this, in, rng, stream);
}

void ModelRuntime::Forward(const ForwardInput& in,
                           std::vector<int>& out_tokens) {
  ctx_.PrepareInputs(in, rng_, stream_);
  RunLayers();
  RunHeadAndSample(out_tokens);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void ModelRuntime::RunHeadAndSample(std::vector<int>& out) {
  int n_sample = ctx_.n_sample;
  out.resize(n_sample);
  if (n_sample == 0) return;
  int hidden = spec_.hidden_size, vocab = spec_.vocab_size;

  LaunchGatherRows(buf_.x.D(), ctx_.lrows.D(), buf_.xg.D(), n_sample, hidden,
                   stream_);
  LaunchRmsnorm(buf_.xg.D(), weights_.fnorm.D(), buf_.xb.D(), n_sample, hidden,
                spec_.rms_eps, stream_);
  LaunchGemm(cublas_, buf_.xb.D(), weights_.lm_head.D(), buf_.logits.D(),
             n_sample, hidden, vocab, CUDA_R_32F);
  LaunchSampleBatch(buf_.logits.D(), n_sample, vocab, ctx_.invT.D(),
                    ctx_.topp.D(), ctx_.u.D(), buf_.sampled.D(), stream_);
  CUDA_CHECK(cudaMemcpyAsync(out.data(), buf_.sampled.D(),
                             n_sample * sizeof(int), cudaMemcpyDeviceToHost,
                             stream_));
}

ModelRuntime::~ModelRuntime() {
  if (cublas_) cublasDestroy(cublas_);
  if (stream_) cudaStreamDestroy(stream_);
}

}
