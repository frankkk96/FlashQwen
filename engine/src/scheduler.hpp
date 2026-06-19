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

struct Request {
    std::vector<int> prompt;        // input: prompt token ids
    int max_new = 0;                // input: number of tokens to generate
    SampleParams sp{0.0f, 1.0f};    // input: per-request sampling (temp<=0 => greedy)
    std::vector<int> stop_ids;      // input: stop as soon as one of these is sampled (empty => none)
    std::shared_ptr<OutputSink> sink;   // input: where this request's results stream out

    std::vector<int> output;        // result: generated token ids

    // scheduler-internal state
    std::vector<int> block_table;   // physical KV block ids while resident (empty when not)
    int computed = 0;               // tokens whose KV is cached (vLLM num_computed_tokens)
    int cur = 0;                    // last sampled token (the decode frontier)

    int num_tokens() const { return (int)prompt.size() + (int)output.size(); }
    int token_at(int i) const {     // the i-th committed token of (prompt ++ output)
        int P = (int)prompt.size();
        return i < P ? prompt[i] : output[i - P];
    }
};

// One scheduling step's batch: the merged forward input plus the bookkeeping to apply results back.
struct StepBatch {
    ForwardInput input;                      // the flattened batch fed to ModelRuntime::forward
    std::vector<Request*> scheduled;         // requests in this batch (batch order; engine owns them)
    std::vector<int>      num_new;           // tokens advanced per scheduled request
    std::vector<Request*> sampling;          // scheduled requests that sample (input.logits_rows order)
    void clear() { input.clear(); scheduled.clear(); num_new.clear(); sampling.clear(); }

    // Append request r advancing it by `n` tokens (positions computed..computed+n-1); flags the
    // frontier row for sampling when the chunk reaches the end of r's committed tokens.
    void add_request(Request* r, int n) {
        int ri = input.begin_request(r->block_table);
        for (int k = 0; k < n; ++k)
            input.add_row(r->token_at(r->computed + k), r->computed + k, ri);
        if (r->computed + n == r->num_tokens()) { input.mark_logits_row(r->sp); sampling.push_back(r); }
        scheduled.push_back(r);
        num_new.push_back(n);
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
    // Advance one scheduling iteration: drop cancelled requests, admit, build a token-budget batch
    // (preempting if the pool is exhausted), run one merged forward, and stream tokens / finishes /
    // errors out through each request's sink.
    void step();

    void drop_cancelled();                  // release + free any request whose sink was cancelled
    void admit();                           // waiting -> running while slots are free
    int  chunk_size(const Request* r, int pass, int budget) const;   // tokens to advance this step
    bool grow_or_preempt(Request* r, int upto_tokens);  // false => r was failed + freed (rebuild)
    void build_batch(StepBatch& b);
    void apply_results(const StepBatch& b, const std::vector<int>& sampled);

    bool finished(const Request* r) const;
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

    StepBatch        batch_;     // per-step scratch (reused to avoid reallocation)
    std::vector<int> sampled_;   // sampled tokens (parallel to batch_.sampling)
};
