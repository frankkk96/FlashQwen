// Continuous-batching scheduler (vLLM-style unified token-budget scheduling) over a paged KV cache.
//
// There is no prefill/decode split: every request is "advance me by num_new tokens from my computed
// cursor". Each step fills a token budget — decode requests (num_new=1) first, then prefill chunks of
// waiting/continuing requests — packs them all into ONE flattened batch (a ForwardInput), and runs a
// single merged forward (ModelRuntime::forward). Blocks are handed out on demand as a request grows;
// when the pool is exhausted the youngest running request is preempted (its blocks freed, its KV
// recomputed from prompt+output when it is later resumed).
//
// The scheduler OWNS its requests (unique_ptr) and IS the engine thread: handler threads hand
// Requests in via submit(); run() drains them, drives the batching loop, and streams results out
// through each request's OutputSink (token / done / error) — no callbacks, no central registry.
// Cancellation is a flag the transport sets on the sink, polled at the top of each step. Three knobs:
// n_slots (max concurrent requests), max_batch_tokens (total query rows per step), max_prefill
// (per-request prefill chunk cap).
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
// generation state (KV residency, computed cursor, output, decode frontier) is private and only ever
// changes through the transition methods below — so the scheduler can't accidentally violate an
// invariant by poking a field. `num_new` is the one piece of per-step scratch (the chunk size chosen
// for the current step), set by schedule() during batch build and consumed by advance() at apply.
class Request {
public:
    Request(std::vector<int> prompt, int max_new, SampleParams sp,
            std::vector<int> stop_ids, std::shared_ptr<OutputSink> sink)
        : prompt_(std::move(prompt)), max_new_(max_new), sp_(sp),
          stop_ids_(std::move(stop_ids)), sink_(std::move(sink)) {}

    // --- read-only inputs ---
    const SampleParams& params() const { return sp_; }
    OutputSink* sink() const { return sink_.get(); }
    int prompt_len() const { return (int)prompt_.size(); }

    // --- committed sequence view (prompt ++ output) ---
    int num_tokens() const { return (int)prompt_.size() + (int)output_.size(); }
    int token_at(int i) const {     // the i-th committed token of (prompt ++ output)
        int P = (int)prompt_.size();
        return i < P ? prompt_[i] : output_[i - P];
    }
    int computed()   const { return computed_; }
    int remaining()  const { return num_tokens() - computed_; }   // uncomputed (uncached) tokens
    int last_token() const { return cur_; }                       // the decode frontier
    int generated()  const { return (int)output_.size(); }

    // --- terminal predicates ---
    bool hit_stop_token() const {   // sampled a caller-provided stop token
        return !output_.empty() &&
               std::find(stop_ids_.begin(), stop_ids_.end(), cur_) != stop_ids_.end();
    }
    bool is_finished(int max_ctx) const {
        return generated() >= max_new_ || hit_stop_token() || num_tokens() >= max_ctx;
    }

    // --- KV residency: the one mutable view we hand out, for KVCacheManager to grow/free ---
    std::vector<int>& blocks() { return block_table_; }

    // --- state transitions (the only way to mutate generation state) ---
    void schedule(int n)       { num_new_ = n; }                  // chunk size chosen for this step
    void advance()             { computed_ += num_new_; }         // n more tokens' KV now cached
    void accept_token(int tok) { output_.push_back(tok); cur_ = tok; }   // sampled a new token
    void reset_for_recompute() { computed_ = 0; }                 // preempted: recompute KV on resume

private:
    // inputs (set once)
    std::vector<int> prompt_;
    int max_new_;
    SampleParams sp_;
    std::vector<int> stop_ids_;
    std::shared_ptr<OutputSink> sink_;
    // generation state
    std::vector<int> output_;       // generated token ids
    std::vector<int> block_table_;  // physical KV block ids while resident (empty when not)
    int computed_ = 0;              // tokens whose KV is cached (vLLM num_computed_tokens)
    int cur_      = 0;              // last sampled token (the decode frontier)
    int num_new_  = 0;             // per-step scratch: tokens to advance this step
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

    // Append request r advancing it by `n` tokens (positions computed..computed+n-1); flags the
    // frontier row for sampling when the chunk reaches the end of r's committed tokens.
    void add_request(Request* r, int n) {
        int ri = input.begin_request(r->blocks());
        for (int k = 0; k < n; ++k)
            input.add_row(r->token_at(r->computed() + k), r->computed() + k, ri);
        if (r->computed() + n == r->num_tokens()) input.mark_logits_row(r->params());
        r->schedule(n);
        requests.push_back(r);
    }

    // Apply one forward's results to each request's state: advance every request's cursor, then for
    // the ones that reached their frontier (remaining()==0 after advancing) append + stream the
    // sampled token, and collect those that finished into finished[] for the scheduler to reclaim.
    // sampled[] is parallel to input.logits_rows, i.e. the sampling requests in batch order.
    void apply(int max_ctx) {
        for (Request* r : requests) r->advance();
        int s = 0;
        for (Request* r : requests) {
            if (r->remaining() != 0) continue;          // prefill chunk that didn't reach the end
            r->accept_token(sampled[s++]);
            if (r->sink()) r->sink()->token(r->last_token());
            if (r->is_finished(max_ctx)) {
                if (r->sink()) {
                    int comp = r->generated() - (r->hit_stop_token() ? 1 : 0);   // exclude stop token
                    r->sink()->done(r->hit_stop_token() ? "stop" : "length",
                                    r->prompt_len(), comp < 0 ? 0 : comp);
                }
                finished.push_back(r);
            }
        }
    }
};

class Scheduler {
public:
    Scheduler(ModelRuntime& model, KVCacheManager& kv, int n_slots, int max_queue,
              int max_batch_tokens, int max_prefill);

    // submit() hands a new request to the engine (thread-safe; called by gRPC handler threads).
    // run() is the engine thread: it drains submitted requests and drives the batching loop forever.
    void submit(std::unique_ptr<Request> r);
    void run();

private:
    // Admission: can_admit() is false once the waiting queue is at max_queue; run() then rejects the
    // request with OverCapacity instead of enqueuing it.
    bool can_admit() const { return max_queue_ <= 0 || (int)waiting_.size() < max_queue_; }
    int  queue_depth() const { return (int)waiting_.size(); }
    bool busy() const { return !waiting_.empty() || !running_.empty(); }

    void add(std::unique_ptr<Request> r);   // move a submitted request into the waiting set
    // Advance one scheduling iteration: drop cancelled requests, admit waiters, pack a token-budget
    // batch (preempting if the pool is exhausted), run one merged forward, apply the results to each
    // request, and reclaim the resources of any that finished. The batch is a step-local value.
    void step();

    int  chunk_size(const Request* r, int pass, int budget) const;   // tokens to advance this step
    bool grow_or_preempt(Request* r, int upto_tokens);  // false => r was failed + freed (rebuild)

    bool preempt_one(Request* protect);     // free youngest running seq != protect; requeue for recompute
    void release(Request* r);               // return a request's blocks to the pool
    void erase_running(Request* r);         // drop + free a request from the running set

    ModelRuntime& model_;
    KVCacheManager& kv_;
    int  n_slots_, max_queue_, max_ctx_, bsz_, max_batch_tokens_, max_prefill_;

    // cross-thread handoff: handler threads push to inbound_ via submit(); run() drains it.
    std::mutex inbound_mu_;
    std::condition_variable inbound_cv_;
    std::deque<std::unique_ptr<Request>> inbound_;

    std::deque<std::unique_ptr<Request>>  waiting_;   // not yet started / resumed
    std::vector<std::unique_ptr<Request>> running_;   // resident (prefilling or decoding)
};
