// Continuous-batching scheduler (stage B) over a paged KV cache (stage C).
//
// The decode kernels accept an arbitrary set of running sequences, each described by its own
// block table (paged KV) and past_len, so serving many requests at once is a host-side
// scheduling problem: keep a pool of KV blocks busy, admit a waiting request the moment blocks
// are available, and decode every running sequence together in one step. Blocks are handed out
// on demand as a sequence grows, so memory tracks actual length instead of reserving max_ctx
// per sequence. When the pool is exhausted, the youngest running sequence is preempted (its
// blocks freed, its KV recomputed from prompt+output when it is later resumed).
//
// The loop is packaged as a class so the live API server (requests arriving over time, tokens
// streamed out) drives it through callbacks: add() enqueues, step() advances one iteration and
// reports tokens / finishes through callbacks. A request stops when it samples one of its own
// stop_ids or reaches max_new — the scheduler has no built-in notion of EOS (that's the caller's).
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
    std::vector<int> block_table;   // physical KV block ids while running (empty when not resident)
    int  past = 0;                  // tokens currently in this sequence's KV
    int  cur  = 0;                  // token to feed on the next decode step
};

class Scheduler {
public:
    Scheduler(ModelRuntime& model, const KVCache& kv, int n_slots, int max_queue, std::mt19937& rng);

    // Admission: can_admit() reports whether another request may be queued (false once the waiting
    // queue is at max_queue); callers reject with OverCapacity instead of enqueuing. queue_depth()
    // and max_queue() expose the numbers for a detailed rejection message.
    bool can_admit() const { return max_queue_ <= 0 || (int)waiting_.size() < max_queue_; }
    int  queue_depth() const { return (int)waiting_.size(); }
    int  max_queue() const { return max_queue_; }

    void add(Request* r);           // enqueue a new request (not yet prefilled)
    void remove(Request* r);        // drop a request (cancellation); frees its blocks. Engine-thread only.
    bool busy() const { return !waiting_.empty() || !running_.empty() || pf_ != nullptr; }

    using TokenFn  = std::function<void(Request*, int)>;   // fired per produced token id
    using FinishFn = std::function<void(Request*)>;        // fired once when a request completes
    // Fired once when a request fails (KV exhaustion); the request is dropped + its blocks freed
    // before this is called. msg is detailed, human-readable, and surfaced to the client as-is.
    using ErrorFn  = std::function<void(Request*, EngineErrc, std::string)>;

    // Advance one scheduling iteration: admit/prefill one chunk, grow blocks (preempting if the
    // pool is exhausted), then decode the whole running set. No-ops to nullptr callbacks.
    void step(const TokenFn& on_token = nullptr, const FinishFn& on_finish = nullptr,
              const ErrorFn& on_error = nullptr);

private:
    int  blocks_for(int n_tok) const { return (n_tok + bsz_ - 1) / bsz_; }
    bool finished(const Request* r) const;
    bool preempt_one(int protect);          // free youngest running seq != running_[protect]; requeue
    void release(Request* r);               // return a finished/preempted seq's blocks to the pool

    ModelRuntime& model_;
    const KVCache& kv_;
    int  n_slots_, max_queue_, max_ctx_, V_, bsz_;
    std::mt19937& rng_;

    std::vector<int>      free_blocks_;      // free KV block ids
    std::deque<Request*>  waiting_;          // not yet started / resumed
    std::vector<Request*> running_;          // currently decoding
    Request* pf_ = nullptr;                  // request being prefilled (one at a time)
    int pf_cursor_ = 0;                      // its prompt cursor
    std::vector<int> pf_tokens_;             // tokens to (re)cache for pf_ (prompt, or prompt+output)

    // per-step scratch
    std::vector<int> in_tok_, past_, out_;
    std::vector<std::vector<int>> bts_;
};
