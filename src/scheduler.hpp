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
// The loop is packaged as a class so both the offline benchmark (run_continuous, a fixed set of
// requests) and the live API server (requests arriving over time, tokens streamed out) drive the
// exact same logic: add() enqueues, step() advances one iteration and reports tokens / finishes
// through callbacks.
#pragma once
#include "model.hpp"
#include "sampler.hpp"
#include <vector>
#include <deque>
#include <random>
#include <functional>

struct Request {
    std::vector<int> prompt;        // input: prompt token ids
    int max_new = 0;                // input: number of tokens to generate
    SampleParams sp{0.0f, 1.0f};    // input: per-request sampling (temp<=0 => greedy)
    std::vector<int> output;        // result: generated token ids

    // scheduler-internal state
    std::vector<int> block_table;   // physical KV block ids while running (empty when not resident)
    int  past = 0;                  // tokens currently in this sequence's KV
    int  cur  = 0;                  // token to feed on the next decode step
};

class Scheduler {
public:
    Scheduler(Model& model, int n_slots, bool stop_on_eos, std::mt19937& rng);

    void add(Request* r);           // enqueue a new request (not yet prefilled)
    bool busy() const { return !waiting_.empty() || !running_.empty() || pf_ != nullptr; }

    using TokenFn  = std::function<void(Request*, int)>;   // fired per produced token id
    using FinishFn = std::function<void(Request*)>;        // fired once when a request completes

    // Advance one scheduling iteration: admit/prefill one chunk, grow blocks (preempting if the
    // pool is exhausted), then decode the whole running set. No-ops to nullptr callbacks.
    void step(const TokenFn& on_token = nullptr, const FinishFn& on_finish = nullptr);

private:
    int  blocks_for(int n_tok) const { return (n_tok + bsz_ - 1) / bsz_; }
    bool finished(const Request* r) const;
    bool preempt_one(int protect);          // free youngest running seq != running_[protect]; requeue
    void release(Request* r);               // return a finished/preempted seq's blocks to the pool

    Model& model_;
    int  n_slots_, max_ctx_, V_, bsz_;
    bool stop_on_eos_;
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

// Offline convenience: serve a fixed set of requests to completion using up to n_slots
// concurrent sequences. Each request samples with its own SampleParams (greedy when temp<=0).
void run_continuous(Model& model, std::vector<Request>& reqs, int n_slots, bool stop_on_eos,
                    std::mt19937& rng);
