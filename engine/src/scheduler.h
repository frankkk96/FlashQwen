#pragma once
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "block_allocator.h"
#include "errors.h"
#include "model_runtime.h"
#include "output_sink.h"
#include "sampler.h"

// A single generation request. Inputs are set once at construction and
// read-only after; generation state (KV residency, computed cursor, output)
// mutates only through these methods, so no external code can break an
// invariant. EmitChunk() writes this step's rows into the forward input;
// AcceptSampled() takes the sampled token, streams it to the sink, and reports
// finish.
class Request {
 public:
  Request(std::vector<int> prompt, int max_new, int max_ctx, SampleParams sp,
          std::vector<int> stop_ids, std::shared_ptr<OutputSink> sink)
      : prompt_(std::move(prompt)),
        max_new_(max_new),
        max_ctx_(max_ctx),
        sp_(sp),
        stop_ids_(std::move(stop_ids)),
        sink_(std::move(sink)) {}

  OutputSink* Sink() const { return sink_.get(); }
  std::vector<int>& Blocks() { return block_table_; }
  int Computed() const { return computed_; }
  int Remaining() const { return NumTokens() - computed_; }

  int NumTokens() const {
    return static_cast<int>(prompt_.size()) + static_cast<int>(output_.size());
  }
  int TokenAt(int i) const {
    int P = static_cast<int>(prompt_.size());
    return i < P ? prompt_[i] : output_[i - P];
  }
  int CachedBlocks() const { return cached_blocks_; }
  void SetCachedBlocks(int n) { cached_blocks_ = n; }
  uint64_t LastHash() const { return last_hash_; }
  void SetLastHash(uint64_t h) { last_hash_ = h; }

  // Adopt a cache-reused prefix: nb leading blocks (already spliced into the
  // block table) cover nb*bsz resident tokens, so skip recomputing them.
  void AdoptPrefix(int nb, int bsz, uint64_t last) {
    computed_ = nb * bsz;
    cached_blocks_ = nb;
    last_hash_ = last;
  }
  // Preempted: block table already freed, so recompute KV from scratch on
  // resume and drop the prefix-cache cursor.
  void ResetForRecompute() {
    computed_ = 0;
    cached_blocks_ = 0;
    last_hash_ = 0;
  }
  void Advance() { computed_ += num_new_; }

  void EmitChunk(ForwardInput& in, int n) {
    int ri = in.AddRequest(block_table_);
    for (int k = 0; k < n; ++k)
      in.AddRow(TokenAt(computed_ + k), computed_ + k, ri);
    if (computed_ + n == NumTokens()) in.SampleLastRow(sp_);
    num_new_ = n;
  }

  bool Done() const { return finished_; }

  bool AcceptSampled(int tok) {
    output_.push_back(tok);
    if (sink_) sink_->Token(tok);
    if (!IsFinished()) return false;
    finished_ = true;
    if (sink_) {
      bool stopped = HitStopToken();
      int comp = Generated() - (stopped ? 1 : 0);
      sink_->Done(stopped ? "stop" : "length", PromptLen(),
                  comp < 0 ? 0 : comp);
    }
    return true;
  }

 private:
  int Generated() const { return static_cast<int>(output_.size()); }
  int PromptLen() const { return static_cast<int>(prompt_.size()); }
  bool HitStopToken() const {
    return !output_.empty() && std::find(stop_ids_.begin(), stop_ids_.end(),
                                         output_.back()) != stop_ids_.end();
  }
  bool IsFinished() const {
    return Generated() >= max_new_ || HitStopToken() || NumTokens() >= max_ctx_;
  }

  std::vector<int> prompt_;
  int max_new_;
  int max_ctx_;
  SampleParams sp_;
  std::vector<int> stop_ids_;
  std::shared_ptr<OutputSink> sink_;
  std::vector<int> output_;
  std::vector<int> block_table_;
  int computed_ = 0;
  int num_new_ = 0;
  int cached_blocks_ = 0;
  uint64_t last_hash_ = 0;
  bool finished_ = false;
};

// One scheduling step's batch: the merged forward input plus pointers to the
// requests it runs (batch order). Per-request step state lives on each Request,
// so there are no parallel arrays. sampled[] parallels input.logits_rows.
struct CurrentBatch {
  ForwardInput input;
  std::vector<Request*> requests;
  std::vector<int> sampled;
  std::vector<Request*> finished;

  void Clear() {
    input.Clear();
    requests.clear();
    sampled.clear();
    finished.clear();
  }
  bool Empty() const { return requests.empty(); }

  void AddRequest(Request* r, int n) {
    r->EmitChunk(input, n);
    requests.push_back(r);
  }

  // Apply one forward's results: advance every cursor, then accept + stream the
  // sampled token for requests at their frontier, collecting finished ones.
  void Apply() {
    for (Request* r : requests) r->Advance();
    int s = 0;
    for (Request* r : requests)
      if (r->Remaining() == 0 && r->AcceptSampled(sampled[s++]))
        finished.push_back(r);
  }
};

// Scheduler tuning knobs, resolved from CLI Args in RunEngine.
struct SchedulerConfig {
  int n_slots;
  int max_waiting;
  int token_budget;
  int prefill_chunk;
  bool use_prefix_cache;
};

// Continuous-batching scheduler (vLLM-style unified token budget) over a paged
// KV cache. No prefill/decode split: each step packs running requests under the
// token budget into one merged forward (decodes outranking prefill chunks);
// blocks are handed out on demand, and when the pool is dry the youngest
// running request is preempted (KV freed, recomputed on resume). The scheduler
// OWNS its requests and IS the engine thread: Submit() (called by handler
// threads) hands requests in; Run() drains them and drives the batching loop
// forever.
class Scheduler {
 public:
  Scheduler(ModelRuntime& model, BlockAllocator& alloc,
            const SchedulerConfig& cfg);

  void Submit(std::unique_ptr<Request> r);
  void Run();

 private:
  bool Busy() const {
    return !waiting_.empty() || !running_.empty() || pending_valid_;
  }

  void Step();
  void AdmitAndDrop();
  bool BuildBatch(CurrentBatch& batch, int& step_prefill, int& step_decode);
  void ProcessPending();  // wait + accept-sample + retire the in-flight async batch
  int ChunkSize(const Request* r, int budget) const;
  bool Grow(Request* r, int upto_tokens);
  void AcquirePrefix(Request* r);
  void CacheFilled(Request* r);
  void Release(Request* r);
  void Retire(Request* r);
  void LogKvstat();

  // Cumulative instrumentation counters (single-threaded engine thread).
  int64_t stat_steps_ = 0;
  int64_t stat_fwd_rows_ = 0;
  int64_t stat_prefill_rows_ = 0;
  int64_t stat_decode_rows_ = 0;
  int64_t stat_preempt_ = 0;
  int64_t stat_recomp_tok_ = 0;
  int64_t stat_cache_hit_tok_ = 0;
  int64_t stat_prompt_tok_ = 0;
  int stat_peak_used_ = 0;

  ModelRuntime& model_;
  BlockAllocator& alloc_;
  const SchedulerConfig cfg_;

  // Cross-thread handoff: handler threads push to inbound_ via Submit(); Run()
  // drains it.
  std::mutex inbound_mu_;
  std::condition_variable inbound_cv_;
  std::deque<std::unique_ptr<Request>> inbound_;

  std::deque<std::unique_ptr<Request>> waiting_;
  std::vector<std::unique_ptr<Request>> running_;

  // Single-step async scheduling: one forward stays in flight (pending_batch_)
  // while the CPU processes the previous one. retiring_ keeps finished requests
  // alive until the in-flight batch that still references them is applied.
  bool pending_valid_ = false;
  int pending_ticket_ = 0;
  CurrentBatch pending_batch_;
  std::vector<std::unique_ptr<Request>> retiring_;
};
