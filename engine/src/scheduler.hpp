// Continuous-batching scheduler (vLLM-style unified token-budget scheduling) over a paged KV cache.
//
// No prefill/decode split: each step packs running requests under a token budget into ONE flattened
// batch (a ForwardInput) and runs a single merged forward. Decodes are prioritised over prefill
// chunks; blocks are handed out on demand, and when the pool is exhausted the youngest running request
// is preempted (its KV freed and recomputed from prompt+output when later resumed).
//
// The scheduler OWNS its requests (unique_ptr) and IS the engine thread: handler threads hand Requests
// in via submit(); run() drains them and drives the batching loop, each request streaming its own
// results through its OutputSink. Cancellation is a sink flag, polled once per step. Three knobs:
// n_slots (max concurrent requests), max_batch_tokens (query rows per step), max_prefill (per-request
// prefill chunk cap).
#pragma once
#include "model_runtime.hpp"
#include "kv_cache.hpp"
#include "sampler.hpp"
#include "errors.hpp"
#include "output_sink.hpp"
#include <vector>
#include <deque>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <string>
#include <algorithm>
#include <utility>

// A single generation request. Inputs are set once at construction and read-only thereafter; the
// generation state (KV residency, computed cursor, output) is private and changes only through the
// methods below, so no external code can poke a field and break an invariant. Two behaviour methods
// keep the batch out of the request's internals: emit_chunk() writes this step's rows into the
// forward input, and accept_sampled() takes the sampled token, streams it, and detects finish.
class Request {
public:
    Request(std::vector<int> prompt, int max_new, int max_ctx, SampleParams sp,
            std::vector<int> stop_ids, std::shared_ptr<OutputSink> sink)
        : prompt_(std::move(prompt)), max_new_(max_new), max_ctx_(max_ctx), sp_(sp),
          stop_ids_(std::move(stop_ids)), sink_(std::move(sink)) {}

    // --- queried by the scheduler (admission / packing / preemption / reclaim) ---
    OutputSink*       sink()      const { return sink_.get(); }
    std::vector<int>& blocks()          { return block_table_; }   // mutable: KVCacheManager grows/frees it
    int               computed()  const { return computed_; }
    int               remaining() const { return num_tokens() - computed_; }   // uncomputed (uncached) tokens

    // --- state transitions driven by the scheduler / batch ---
    void reset_for_recompute() { computed_ = 0; }          // preempted: recompute KV on resume
    void advance()             { computed_ += num_new_; }  // commit the chunk emitted this step

    // Write this request's next n-token chunk (positions computed..computed+n-1) into the forward
    // input, flag the frontier row for sampling when the chunk reaches the end of the committed
    // tokens, and record n so advance() can commit it after the forward.
    void emit_chunk(ForwardInput& in, int n) {
        int ri = in.begin_request(block_table_);
        for (int k = 0; k < n; ++k)
            in.add_row(token_at(computed_ + k), computed_ + k, ri);
        if (computed_ + n == num_tokens()) in.mark_logits_row(sp_);
        num_new_ = n;
    }

    // Accept a sampled token: append it, stream it to the sink, and if this ends the request stream
    // the terminal done. Returns true when the request has finished.
    bool accept_sampled(int tok) {
        output_.push_back(tok);
        if (sink_) sink_->token(tok);
        if (!is_finished()) return false;
        if (sink_) {
            bool stopped = hit_stop_token();
            int  comp    = generated() - (stopped ? 1 : 0);   // exclude the stop token itself
            sink_->done(stopped ? "stop" : "length", prompt_len(), comp < 0 ? 0 : comp);
        }
        return true;
    }

private:
    int num_tokens() const { return (int)prompt_.size() + (int)output_.size(); }
    int token_at(int i) const {     // the i-th committed token of (prompt ++ output)
        int P = (int)prompt_.size();
        return i < P ? prompt_[i] : output_[i - P];
    }
    int  generated()  const { return (int)output_.size(); }
    int  prompt_len() const { return (int)prompt_.size(); }
    bool hit_stop_token() const {   // the last sampled token is one of the caller's stop ids
        return !output_.empty() &&
               std::find(stop_ids_.begin(), stop_ids_.end(), output_.back()) != stop_ids_.end();
    }
    bool is_finished() const {
        return generated() >= max_new_ || hit_stop_token() || num_tokens() >= max_ctx_;
    }

    // inputs (set once)
    std::vector<int> prompt_;
    int max_new_;
    int max_ctx_;
    SampleParams sp_;
    std::vector<int> stop_ids_;
    std::shared_ptr<OutputSink> sink_;
    // generation state
    std::vector<int> output_;       // generated token ids
    std::vector<int> block_table_;  // physical KV block ids while resident (empty when not)
    int computed_ = 0;              // tokens whose KV is cached (vLLM num_computed_tokens)
    int num_new_  = 0;              // per-step scratch: tokens to advance this step
};

// One scheduling step's batch: the merged forward input plus the requests it runs. The per-request
// step state (num_new, whether it samples) lives on each Request, so this is just a flattened input
// plus a pointer list — no parallel arrays to keep aligned.
struct CurrentBatch {
    ForwardInput          input;       // the flattened batch fed to ModelRuntime::forward
    std::vector<Request*> requests;    // requests in this batch (batch order; the engine owns them)
    std::vector<int>      sampled;     // GPU sampling output (parallel to input.logits_rows)
    std::vector<Request*> finished;    // requests that finished this step (for the scheduler to reclaim)

    void clear() { input.clear(); requests.clear(); sampled.clear(); finished.clear(); }
    bool empty() const { return requests.empty(); }

    // Add request r to this batch, advancing it by `n` tokens (it writes its own rows).
    void add_request(Request* r, int n) { r->emit_chunk(input, n); requests.push_back(r); }

    // Apply one forward's results: advance every request's cursor, then for those that reached their
    // frontier (remaining()==0) accept + stream the sampled token, collecting the finished ones for
    // the scheduler to reclaim. sampled[] is parallel to input.logits_rows (sampling reqs in order).
    void apply() {
        for (Request* r : requests) r->advance();
        int s = 0;
        for (Request* r : requests)
            if (r->remaining() == 0 && r->accept_sampled(sampled[s++]))
                finished.push_back(r);
    }
};

// Scheduler tuning knobs, resolved from CLI Args in run_engine (defaults already applied).
struct SchedulerConfig {
    int n_slots;          // max concurrent sequences (already clamped to model.max_batch())
    int max_queue;        // admission cap on waiting requests before rejecting as over-capacity
    int max_batch_tokens; // total query rows computed per step
    int max_prefill;      // per-request prefill chunk cap
};

class Scheduler {
public:
    Scheduler(ModelRuntime& model, KVCacheManager& kv, const SchedulerConfig& cfg);

    // submit() hands a new request to the engine (thread-safe; called by gRPC handler threads).
    // run() is the engine thread: it drains submitted requests and drives the batching loop forever.
    void submit(std::unique_ptr<Request> r);
    void run();

private:
    // busy(): there is work to do (something waiting or running) — run() blocks when not busy.
    bool busy() const { return !waiting_.empty() || !running_.empty(); }

    void step();   // one scheduling iteration (cancel, admit, pack, forward, apply, reclaim)

    int  chunk_size(const Request* r, int budget) const;   // tokens to advance this step (budget+max_prefill capped)
    bool grow(Request* r, int upto_tokens);  // grow r's KV (preempting younger seqs); false => r failed+retired

    void release(Request* r);               // return a request's KV blocks to the pool (it stays alive)
    void retire(Request* r);                // request leaves for good: free its KV + drop it from running

    ModelRuntime& model_;
    KVCacheManager& kv_;
    const SchedulerConfig cfg_;

    // cross-thread handoff: handler threads push to inbound_ via submit(); run() drains it.
    std::mutex inbound_mu_;
    std::condition_variable inbound_cv_;
    std::deque<std::unique_ptr<Request>> inbound_;

    std::deque<std::unique_ptr<Request>>  waiting_;   // not yet started / resumed
    std::vector<std::unique_ptr<Request>> running_;   // resident (prefilling or decoding)
};
