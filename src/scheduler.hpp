// Continuous-batching scheduler (stage B).
//
// The decode kernels already accept an arbitrary set of running sequences (per-sequence KV
// slot + past_len), so serving many requests at once is a host-side scheduling problem: keep
// a fixed number of KV slots busy, admit a waiting request the moment a slot frees, and decode
// every running sequence together in one step. This avoids the head-of-line blocking of static
// batching, where a whole batch is held until its longest sequence finishes.
#pragma once
#include "model.hpp"
#include <vector>

struct Request {
    std::vector<int> prompt;        // input: prompt token ids
    int max_new = 0;                // input: number of tokens to generate
    std::vector<int> output;        // result: generated token ids

    // scheduler-internal state
    int  slot = -1;                 // KV slot while running, -1 otherwise
    int  past = 0;                  // tokens currently in this sequence's KV
    int  cur  = 0;                  // token to feed on the next decode step
};

// Serve all requests to completion (greedy) using up to n_slots concurrent sequences, admitting
// waiting requests as slots free up. If stop_on_eos, a sequence also ends on an EOS token.
void run_continuous(Model& model, std::vector<Request>& reqs, int n_slots, bool stop_on_eos);
