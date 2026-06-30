#pragma once
#include <cublas_v2.h>

#include <random>
#include <string>
#include <vector>

#include "cuda_helpers.h"
#include "kernels.cuh"
#include "kv_store.h"
#include "model_spec.h"
#include "safetensors.h"
#include "sampler.h"

namespace fq {

// Precomputed, layer-independent RoPE cos/sin tables [max_ctx, head_dim/2]
// (FP32, device-resident): angle(pos,i) = pos * theta^(-2i/head_dim). Owns just
// the tables; the rotation itself is applied by the fused head-norm+RoPE kernel
// that reads Cos()/Sin().
class RopeTables {
 public:
  void Build(const ModelSpec& spec, int max_ctx);
  const float* Cos() const { return cos_.D(); }
  const float* Sin() const { return sin_.D(); }

 private:
  DeviceBuffer<float> cos_, sin_;
};

// One merged forward's inputs: a flattened mixed batch (prefill chunks +
// decodes together). tokens/positions/req_index are parallel per-row arrays;
// block_tables[r] is request r's KV blocks; logits_rows[s] / sample_params[s]
// pick the rows that feed sampling. Callers build it through the Add* methods.
struct ForwardInput {
  std::vector<int> tokens, positions, req_index, logits_rows;
  std::vector<std::vector<int>> block_tables;
  std::vector<SampleParams> sample_params;

  void Clear() {
    tokens.clear();
    positions.clear();
    req_index.clear();
    logits_rows.clear();
    block_tables.clear();
    sample_params.clear();
  }
  int NumRows() const { return static_cast<int>(tokens.size()); }

  int AddRequest(std::vector<int> block_table) {
    block_tables.push_back(std::move(block_table));
    return static_cast<int>(block_tables.size()) - 1;
  }
  void AddRow(int token, int position, int req) {
    tokens.push_back(token);
    positions.push_back(position);
    req_index.push_back(req);
  }
  void SampleLastRow(SampleParams sp) {
    logits_rows.push_back(static_cast<int>(tokens.size()) - 1);
    sample_params.push_back(sp);
  }
};

// Fused weights for one decoder layer (BF16, HF layout [OUT, IN]): qkv =
// [q|k|v] and gateup = [gate|up] stacked on OUT; o_proj / down stay
// separate.
struct Layer {
  DeviceBuffer<bf16> qkv, o_proj, gateup, down;
  DeviceBuffer<float> in_norm, post_norm, q_norm, k_norm;
};

// All GPU-resident model weights: token embedding, final norm, LM head,
// precomputed RoPE tables, and the per-layer fused projections. The constructor
// reads them from spec.dir (st_) and uploads each to the GPU.
struct ModelWeights {
  DeviceBuffer<bf16> embed;
  DeviceBuffer<float> fnorm;
  DeviceBuffer<bf16> lm_head;
  RopeTables rope;
  std::vector<Layer> layers;

  ModelWeights() = default;
  ModelWeights(const ModelSpec& spec, int max_ctx);
};

// A device buffer paired with a host-side staging vector of the same length, so
// the two halves can't drift apart. Fill the host side (operator[] / H()), then
// Flush(stream, n) the first n elements to the device. Only StepContext uses it,
// passing its own stream; n is the step's live prefix (≤ capacity), so the stale
// tail isn't copied.
template <class T>
class StagedBuffer {
 public:
  StagedBuffer() = default;
  explicit StagedBuffer(size_t n) : d_(n), h_(n) {}
  T* H() { return h_.data(); }
  T* D() const { return d_.D(); }
  T& operator[](size_t i) { return h_[i]; }
  const T& operator[](size_t i) const { return h_[i]; }
  void Flush(cudaStream_t stream, size_t n) {
    CUDA_CHECK(cudaMemcpyAsync(d_.D(), h_.data(), n * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
  }

 private:
  DeviceBuffer<T> d_;
  std::vector<T> h_;
};

// The per-forward batch context: everything describing what this step runs,
// computed on the host and pushed to the GPU each forward. ids/pos/req and the
// sampling rows (lrows) come straight from the ForwardInput; qstart/qlen and the
// decode/prefill request-id lists and packed block tables (bt) are derived;
// invT/topp/u are the per-row sampling params. The methods fill each group from
// a ForwardInput and flush it to the device on the caller's stream. The ints describe
// the buffers (bt_stride = bt's row stride, n_decode = decode_rids' valid
// length, ...). The constructor sizes everything to the worst case once.
struct StepContext {
  DeviceBuffer<int> ids, pos, req;
  DeviceBuffer<int> lrows;
  StagedBuffer<int> qstart, qlen;
  StagedBuffer<int> decode_rids, prefill_rids;
  StagedBuffer<int> bt;
  StagedBuffer<float> invT, topp, u;

  int max_blocks = 0;
  int bt_stride = 0;
  int n_rows = 0, n_sample = 0;
  int n_decode = 0, n_prefill = 0, prefill_max_qlen = 1;

  StepContext() = default;
  StepContext(int max_rows, int slots, int max_ctx);
  void PrepareInputs(const ForwardInput& in, std::mt19937& rng,
                     cudaStream_t stream);
};

// Per-forward compute scratch: activation buffers + lm_head logits + the decode
// attention split partials + the sampler's device-side output (sampled). Pure
// kernel-written working memory, sized to the worst-case batch (max_rows query
// rows, slots concurrent requests) once.
struct RuntimeBuffers {
  DeviceBuffer<bf16> x, xb, xb2, qkv, attn, gateup, hmlp;
  DeviceBuffer<bf16> xg;
  DeviceBuffer<float> logits;
  DeviceBuffer<float> dec_pm, dec_pl, dec_pa;
  DeviceBuffer<int> sampled;

  RuntimeBuffers() = default;
  RuntimeBuffers(const ModelSpec& spec, int max_rows, int slots);
};

// A small LRU cache of capture-once / replay CUDA graphs, keyed on a (k0, k1)
// shape. Run() replays the cached graph for the current shape if present;
// otherwise it captures `body`'s kernel launches into a fresh graph and launches
// that. Keeping up to kMaxEntries graphs avoids re-capturing every step when the
// shape (the decode batch size) churns as requests finish/admit — a single
// cached graph would thrash there (measured: 376 recaptures vs 38 under a
// not-full load). Frees all executable graphs on destruction.
class GraphCache {
 public:
  GraphCache() = default;
  ~GraphCache() {
    for (auto& e : entries_) cudaGraphExecDestroy(e.exec);
  }
  GraphCache(const GraphCache&) = delete;
  GraphCache& operator=(const GraphCache&) = delete;

  template <class F>
  void Run(cudaStream_t stream, int k0, int k1, F&& body) {
    for (auto it = entries_.begin(); it != entries_.end(); ++it)
      if (it->k0 == k0 && it->k1 == k1) {
        Entry hit = *it;
        entries_.erase(it);
        entries_.insert(entries_.begin(), hit);
        CUDA_CHECK(cudaGraphLaunch(hit.exec, stream));
        return;
      }
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    body();
    cudaGraph_t g;
    CUDA_CHECK(cudaStreamEndCapture(stream, &g));
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec, g, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    while (entries_.size() >= kMaxEntries) {
      cudaGraphExecDestroy(entries_.back().exec);
      entries_.pop_back();
    }
    entries_.insert(entries_.begin(), Entry{k0, k1, exec});
    CUDA_CHECK(cudaGraphLaunch(exec, stream));
  }

 private:
  struct Entry {
    int k0, k1;
    cudaGraphExec_t exec;
  };
  static constexpr size_t kMaxEntries = 64;
  std::vector<Entry> entries_;
};

// Qwen3 dense decoder GPU executor (the compute half; dims/arch live in
// ModelSpec). Loads weights + activation scratch from spec.dir and borrows the
// paged KV store (non-owning) for the lifetime of the run. Forward() runs
// the whole flattened batch in one pass, then runs final norm + lm_head on the
// per-request last-token rows and samples them on the GPU ->
// out_tokens[logits_rows.size()].
class ModelRuntime {
 public:
  ModelRuntime(const ModelSpec& spec, int max_ctx, int slots, int token_budget,
               unsigned seed);
  ~ModelRuntime();

  void AttachKvStore(const KvStore& store) { store_ = &store; }

  void Forward(const ForwardInput& in, std::vector<int>& out_tokens);

 private:
  void RunLayers();
  void RunLayersBody();
  void RunHeadAndSample(std::vector<int>& out);
  void InitCudaResources();

  ModelSpec spec_;
  int max_ctx_ = 0;
  int slots_ = 0;
  int max_rows_ = 0;
  const KvStore* store_ = nullptr;

  ModelWeights weights_;
  RuntimeBuffers buf_;
  StepContext ctx_;
  std::mt19937 rng_;

  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  static constexpr size_t kCublasWsBytes = 32 << 20;
  DeviceBuffer<std::byte> cublas_ws_;
  GraphCache layers_graph_;
};

}
