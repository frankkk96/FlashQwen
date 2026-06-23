// Continuous-batching scheduler (vLLM-style unified token-budget) over a paged
// KV cache. No prefill/decode split: each step packs running requests under a
// token budget into one flattened ForwardInput and runs a single merged
// forward. Decodes outrank prefill chunks; blocks are handed out on demand, and
// when the pool is dry the youngest running request is preempted (KV freed,
// recomputed from prompt+output on resume).
//
// The scheduler OWNS its requests (unique_ptr) and IS the engine thread:
// Submit() (called by handler threads) hands Requests in; Run() drains them and
// drives the batching loop, each request streaming via its OutputSink.
// Cancellation is a sink flag polled once per step. Knobs: n_slots (max
// concurrent), max_batch_tokens (query rows/step), max_prefill (prefill chunk
// cap).
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

#include "errors.hpp"
#include "kv_cache.hpp"
#include "model_runtime.hpp"
#include "output_sink.hpp"
#include "sampler.hpp"

// A single generation request. Inputs are set once at construction, read-only
// after; generation state (KV residency, computed cursor, output) is private
// and mutates only through these methods, so no external code can break an
// invariant. EmitChunk() writes this step's rows into the forward input;
// AcceptSampled() takes the sampled token, streams it, and detects finish.
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

  // --- queried by the scheduler (admission / packing / preemption / reclaim)
  // ---
  OutputSink* Sink() const { return sink_.get(); }
  std::vector<int>& Blocks() {
    return block_table_;
  }  // mutable: KVCacheManager grows/frees it
  int Computed() const { return computed_; }
  int Remaining() const {
    return NumTokens() - computed_;
  }  // uncomputed (uncached) tokens

  // --- prefix caching: scheduler hashes blocks of (prompt ++ output), chained
  // ---
  int NumTokens() const { return (int)prompt_.size() + (int)output_.size(); }
  int TokenAt(int i) const {  // i-th committed token of (prompt ++ output)
    int P = (int)prompt_.size();
    return i < P ? prompt_[i] : output_[i - P];
  }
  int CachedBlocks() const {
    return cached_blocks_;
  }  // leading blocks already in the cache
  void SetCachedBlocks(int n) { cached_blocks_ = n; }
  uint64_t LastHash() const {
    return last_hash_;
  }  // content hash of block cached_blocks_-1
  void SetLastHash(uint64_t h) { last_hash_ = h; }

  // --- state transitions driven by scheduler / batch ---
  // Adopt a cache-reused prefix: nb leading blocks (already spliced into the
  // block table) cover nb*bsz resident tokens; skip recomputing them.
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
  void Advance() {
    computed_ += num_new_;
  }  // commit the chunk emitted this step

  // Emit this request's next n-token chunk (positions computed..computed+n-1)
  // into the forward input; flag the frontier row for sampling when the chunk
  // reaches the committed end; record n for Advance() to commit post-forward.
  void EmitChunk(ForwardInput& in, int n) {
    int ri = in.AddRequest(block_table_);
    for (int k = 0; k < n; ++k)
      in.AddRow(TokenAt(computed_ + k), computed_ + k, ri);
    if (computed_ + n == NumTokens()) in.SampleLastRow(sp_);
    num_new_ = n;
  }

  // Accept a sampled token: append, stream to sink, and on finish stream the
  // terminal done. Returns true when the request has finished.
  bool AcceptSampled(int tok) {
    output_.push_back(tok);
    if (sink_) sink_->Token(tok);
    if (!IsFinished()) return false;
    if (sink_) {
      bool stopped = HitStopToken();
      int comp =
          Generated() - (stopped ? 1 : 0);  // exclude the stop token itself
      sink_->Done(stopped ? "stop" : "length", PromptLen(),
                  comp < 0 ? 0 : comp);
    }
    return true;
  }

 private:
  int Generated() const { return (int)output_.size(); }
  int PromptLen() const { return (int)prompt_.size(); }
  bool HitStopToken()
      const {  // last sampled token is one of the caller's stop ids
    return !output_.empty() && std::find(stop_ids_.begin(), stop_ids_.end(),
                                         output_.back()) != stop_ids_.end();
  }
  bool IsFinished() const {
    return Generated() >= max_new_ || HitStopToken() || NumTokens() >= max_ctx_;
  }

  // inputs (set once)
  std::vector<int> prompt_;
  int max_new_;
  int max_ctx_;
  SampleParams sp_;
  std::vector<int> stop_ids_;
  std::shared_ptr<OutputSink> sink_;
  // generation state
  std::vector<int> output_;  // generated token ids
  std::vector<int>
      block_table_;   // physical KV block ids while resident (empty when not)
  int computed_ = 0;  // tokens whose KV is cached (vLLM num_computed_tokens)
  int num_new_ = 0;   // per-step scratch: tokens to advance this step
  // prefix-cache cursor: count of leading registered blocks, plus the last
  // registered block's hash (parent for hashing the next block as it fills).
  int cached_blocks_ = 0;
  uint64_t last_hash_ = 0;
};

// One scheduling step's batch: merged forward input plus the requests it runs.
// Per-request step state (num_new, whether it samples) lives on each Request,
// so this is just a flattened input plus a pointer list — no parallel arrays.
struct CurrentBatch {
  ForwardInput input;  // the flattened batch fed to ModelRuntime::Forward
  std::vector<Request*>
      requests;  // requests in this batch (batch order; the engine owns them)
  std::vector<int>
      sampled;  // GPU sampling output (parallel to input.logits_rows)
  std::vector<Request*> finished;  // requests that finished this step

  void Clear() {
    input.Clear();
    requests.clear();
    sampled.clear();
    finished.clear();
  }
  bool Empty() const { return requests.empty(); }

  // Add r to this batch, advancing it by n tokens (it writes its own rows).
  void AddRequest(Request* r, int n) {
    r->EmitChunk(input, n);
    requests.push_back(r);
  }

  // Apply one forward's results: advance every cursor, then for those at their
  // frontier (Remaining()==0) accept + stream the sampled token, collecting
  // finished ones. sampled[] parallels input.logits_rows (sampling reqs in
  // order).
  void Apply() {
    for (Request* r : requests) r->Advance();
    int s = 0;
    for (Request* r : requests)
      if (r->Remaining() == 0 && r->AcceptSampled(sampled[s++]))
        finished.push_back(r);
  }
};

// Scheduler tuning knobs, resolved from CLI Args in RunEngine (defaults
// applied).
struct SchedulerConfig {
  int n_slots;    // max concurrent sequences (clamped to model.MaxBatch())
  int max_queue;  // waiting-queue cap before rejecting as over-capacity
  int max_batch_tokens;  // total query rows computed per step
  int max_prefill;       // per-request prefill chunk cap
  bool prefix_cache;  // reuse cached KV for shared prompt prefixes (vLLM APC)
};

class Scheduler {
 public:
  Scheduler(ModelRuntime& model, KVCacheManager& kv,
            const SchedulerConfig& cfg);

  // Submit() hands a new request to the engine (thread-safe; called by gRPC
  // handler threads). Run() is the engine thread: drains submitted requests and
  // drives the batching loop forever.
  void Submit(std::unique_ptr<Request> r);
  void Run();

 private:
  // Work to do (something waiting or running) — Run() blocks when not Busy().
  bool Busy() const { return !waiting_.empty() || !running_.empty(); }

  void Step();  // one iteration: cancel, admit, pack, forward, apply, reclaim

  int ChunkSize(const Request* r, int budget)
      const;  // tokens to advance this step (budget + max_prefill capped)
  bool Grow(Request* r, int upto_tokens);  // grow r's KV (preempting younger
                                           // seqs); false => r failed + retired

  void AcquirePrefix(Request* r);  // on admission: reuse cached prefix blocks,
                                   // advancing computed_
  void CacheFilled(Request* r);  // after a forward: register newly-full blocks
                                 // into the content cache

  void Release(
      Request* r);  // return a request's KV blocks to the pool (it stays alive)
  void Retire(
      Request* r);  // request leaves for good: free KV + drop from running

  void LogKvstat();  // periodic cumulative KV/preemption instrumentation line

  // --- instrumentation (single-threaded engine thread; plain counters) ---
  long stat_steps_ = 0;  // non-empty scheduler steps
  long stat_fwd_rows_ =
      0;  // total query rows forwarded (prefill chunks + decode steps)
  long stat_prefill_rows_ = 0;  // of those, rows that were prefill (chunk n>1)
  long stat_decode_rows_ = 0;   // of those, rows that were decode (chunk n==1)
  long stat_preempt_ = 0;       // preemption events (victims chosen)
  long stat_recomp_tok_ =
      0;  // KV tokens discarded by preemption (must be recomputed on resume)
  long stat_cache_hit_tok_ = 0;  // prompt tokens skipped via prefix-cache hits
                                 // (KV reused, not recomputed)
  long stat_prompt_tok_ =
      0;  // total prompt tokens admitted (denominator for the cache hit rate)
  int stat_peak_used_ = 0;  // peak physical blocks in use across all steps

  ModelRuntime& model_;
  KVCacheManager& kv_;
  const SchedulerConfig cfg_;

  // cross-thread handoff: handler threads push to inbound_ via submit(); run()
  // drains it.
  std::mutex inbound_mu_;
  std::condition_variable inbound_cv_;
  std::deque<std::unique_ptr<Request>> inbound_;

  std::deque<std::unique_ptr<Request>> waiting_;  // not yet started / resumed
  std::vector<std::unique_ptr<Request>>
      running_;  // resident (prefilling or decoding)
};
