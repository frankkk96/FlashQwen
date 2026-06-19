// Continuous-batching scheduler (vLLM-style unified token-budget scheduling) over a paged KV cache.
//
// There is no prefill/decode split: every request is "advance me by num_new tokens from my computed
// cursor". Each step fills a token budget — decode requests (num_new=1) first, then prefill chunks of
// waiting/continuing requests — packs them all into ONE flattened batch, and runs a single merged
// forward (see ModelRuntime::forward). Blocks are handed out on demand as a request grows; when the
// pool is exhausted the youngest running request is preempted (its blocks freed, its KV recomputed
// from prompt+output when it is later resumed).
//
// Three knobs: n_slots (max concurrent requests), max_batch_tokens (total query rows per step), and
// max_prefill (per-request prefill chunk cap). The loop is driven through callbacks: add() enqueues,
// step() advances one iteration and reports tokens / finishes / errors. A request stops when it
// samples one of its own stop_ids or reaches max_new — the scheduler has no built-in notion of EOS.
#pragma once
#include "model_runtime.hpp"
#include "kv_cache.hpp"
#include "sampler.hpp"
#include "errors.hpp"
#include <vector>
#include <deque>
#include <random>
#include <functional>
#include <string>

struct Request {
    std::vector<int> prompt;        // input: prompt token ids
    int max_new = 0;                // input: number of tokens to generate
    SampleParams sp{0.0f, 1.0f};    // input: per-request sampling (temp<=0 => greedy)
    std::vector<int> stop_ids;      // input: stop as soon as one of these is sampled (empty => none)
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

class Scheduler {
public:
    Scheduler(ModelRuntime& model, KVCacheManager& kv, int n_slots, int max_queue,
              int max_batch_tokens, int max_prefill, std::mt19937& rng);

    // Admission: can_admit() reports whether another request may be queued (false once the waiting
    // queue is at max_queue); callers reject with OverCapacity instead of enqueuing.
    bool can_admit() const { return max_queue_ <= 0 || (int)waiting_.size() < max_queue_; }
    int  queue_depth() const { return (int)waiting_.size(); }
    int  max_queue() const { return max_queue_; }

    void add(Request* r);           // enqueue a new request (not yet prefilled)
    void remove(Request* r);        // drop a request (cancellation); frees its blocks. Engine-thread only.
    bool busy() const { return !waiting_.empty() || !running_.empty(); }

    using TokenFn  = std::function<void(Request*, int)>;   // fired per produced token id
    using FinishFn = std::function<void(Request*)>;        // fired once when a request completes
    using ErrorFn  = std::function<void(Request*, EngineErrc, std::string)>;

    // Advance one scheduling iteration: admit, build a token-budget batch (preempting if the pool is
    // exhausted), run one merged forward, emit tokens / finishes. No-ops to nullptr callbacks.
    void step(const TokenFn& on_token = nullptr, const FinishFn& on_finish = nullptr,
              const ErrorFn& on_error = nullptr);

private:
    bool finished(const Request* r) const;
    bool preempt_one(Request* protect);     // free youngest running seq != protect; requeue for recompute
    void release(Request* r);               // return a finished/preempted seq's blocks to the pool

    ModelRuntime& model_;
    KVCacheManager& kv_;
    int  n_slots_, max_queue_, max_ctx_, V_, bsz_, max_batch_tokens_, max_prefill_;
    std::mt19937& rng_;

    std::deque<Request*>  waiting_;          // not yet started / resumed
    std::vector<Request*> running_;          // resident (prefilling or decoding)

    // per-step scratch: the flattened batch + the bookkeeping to apply results back to requests.
    std::vector<int> in_tok_, pos_, reqidx_, lrows_, out_;
    std::vector<std::vector<int>> bts_;
    std::vector<Request*> sched_;            // requests scheduled this step (batch order)
    std::vector<int>      sched_new_;        // num_new tokens per scheduled request
    std::vector<Request*> sampling_;         // scheduled requests that sample this step (lrows_ order)
};
