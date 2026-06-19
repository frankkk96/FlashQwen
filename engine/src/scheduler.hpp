// Continuous-batching scheduler (vLLM-style unified token-budget scheduling) over a paged KV cache.
//
// There is no prefill/decode split: every request is "advance me by num_new tokens from my computed
// cursor". Each step fills a token budget — decode requests (num_new=1) first, then prefill chunks of
// waiting/continuing requests — packs them all into ONE flattened batch (a ForwardInput), and runs a
// single merged forward (ModelRuntime::forward). Blocks are handed out on demand as a request grows;
// when the pool is exhausted the youngest running request is preempted (its blocks freed, its KV
// recomputed from prompt+output when it is later resumed).
//
// The scheduler OWNS its requests (unique_ptr). Results stream out through each request's OutputSink
// (token / done / error) — no callbacks, no central registry. Cancellation is a flag the transport
// sets on the sink, polled at the top of each step. Three knobs: n_slots (max concurrent requests),
// max_batch_tokens (total query rows per step), max_prefill (per-request prefill chunk cap).
#pragma once
#include "model_runtime.hpp"
#include "kv_cache.hpp"
#include "sampler.hpp"
#include "errors.hpp"
#include "output_sink.hpp"
#include <vector>
#include <deque>
#include <memory>
#include <random>
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
};

class Scheduler {
public:
    Scheduler(ModelRuntime& model, KVCacheManager& kv, int n_slots, int max_queue,
              int max_batch_tokens, int max_prefill, std::mt19937& rng);

    // Admission: can_admit() reports whether another request may be queued (false once the waiting
    // queue is at max_queue); callers reject with OverCapacity instead of enqueuing.
    bool can_admit() const { return max_queue_ <= 0 || (int)waiting_.size() < max_queue_; }
    int  queue_depth() const { return (int)waiting_.size(); }
    int  max_queue() const { return max_queue_; }

    void add(std::unique_ptr<Request> r);   // enqueue a new request (engine takes ownership)
    bool busy() const { return !waiting_.empty() || !running_.empty(); }

    // Advance one scheduling iteration: drop cancelled requests, admit, build a token-budget batch
    // (preempting if the pool is exhausted), run one merged forward, and stream tokens / finishes /
    // errors out through each request's sink.
    void step();

private:
    void drop_cancelled();                  // release + free any request whose sink was cancelled
    void admit();                           // waiting -> running while slots are free
    int  chunk_size(const Request* r, int pass, int budget) const;   // tokens to advance this step
    bool grow_or_preempt(Request* r, int upto_tokens);  // false => r was failed + freed (rebuild)
    void append_request(StepBatch& b, Request* r, int n);
    void build_batch(StepBatch& b);
    void run_forward(const StepBatch& b, std::vector<int>& sampled);
    void apply_results(const StepBatch& b, const std::vector<int>& sampled);

    bool finished(const Request* r) const;
    bool preempt_one(Request* protect);     // free youngest running seq != protect; requeue for recompute
    void release(Request* r);               // return a request's blocks to the pool
    void erase_running(Request* r);         // drop + free a request from the running set

    ModelRuntime& model_;
    KVCacheManager& kv_;
    int  n_slots_, max_queue_, max_ctx_, V_, bsz_, max_batch_tokens_, max_prefill_;
    std::mt19937& rng_;

    std::deque<std::unique_ptr<Request>>  waiting_;   // not yet started / resumed
    std::vector<std::unique_ptr<Request>> running_;   // resident (prefilling or decoding)

    StepBatch        batch_;     // per-step scratch (reused to avoid reallocation)
    std::vector<int> sampled_;   // greedy sampled tokens (parallel to batch_.sampling)
};
