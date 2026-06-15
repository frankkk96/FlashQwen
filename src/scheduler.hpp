// Continuous-batching scheduler (stage B) over a paged KV cache (stage C).
//
// The decode kernels accept an arbitrary set of running sequences, each described by its own
// block table (paged KV) and past_len, so serving many requests at once is a host-side
// scheduling problem: keep a pool of KV blocks busy, admit a waiting request the moment blocks
// are available, and decode every running sequence together in one step. Blocks are handed out
// on demand as a sequence grows, so memory tracks actual length instead of reserving max_ctx
// per sequence. When the pool is exhausted, the youngest running sequence is preempted (its
// blocks freed, its KV recomputed from prompt+output when it is later resumed).
#pragma once
#include "model.hpp"
#include "sampler.hpp"
#include <vector>
#include <random>

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

// Serve all requests to completion using up to n_slots concurrent sequences, admitting waiting
// requests as KV blocks allow. Each request samples with its own SampleParams (greedy when
// temp<=0). If stop_on_eos, a sequence also ends on an EOS token.
void run_continuous(Model& model, std::vector<Request>& reqs, int n_slots, bool stop_on_eos,
                    std::mt19937& rng);
