#include "scheduler.hpp"
#include <algorithm>
#include <deque>
#include <cstdio>
#include <cstdlib>

static const int EOS1 = 151645, EOS2 = 151643;   // <|im_end|>, <|endoftext|>
static const int PREFILL_CHUNK = 256;            // tokens prefilled per scheduler iteration

static inline bool finished(const Request& r, bool stop_on_eos, int max_ctx) {
    if ((int)r.output.size() >= r.max_new) return true;
    if (stop_on_eos && !r.output.empty() && (r.cur == EOS1 || r.cur == EOS2)) return true;
    if (r.past >= max_ctx) return true;   // sequence full — no room for another token
    return false;
}

void run_continuous(Model& model, std::vector<Request>& reqs, int n_slots, bool stop_on_eos,
                    std::mt19937& rng) {
    n_slots = std::min(n_slots, model.max_batch());
    int max_ctx = model.max_ctx(), V = model.config().vocab_size, BSZ = model.block_size();
    auto blocks_for = [&](int n_tok) { return (n_tok + BSZ - 1) / BSZ; };

    // Block allocator: a free list of physical block ids.
    std::vector<int> free_blocks;
    free_blocks.reserve(model.num_blocks());
    for (int b = model.num_blocks() - 1; b >= 0; --b) free_blocks.push_back(b);

    std::deque<int> waiting;                       // request indices not yet started/resumed
    for (int i = 0; i < (int)reqs.size(); ++i) waiting.push_back(i);
    std::vector<int> running;                      // request indices currently decoding

    // Preempt the youngest running sequence other than running[protect] (or any if protect<0):
    // free its blocks, drop it from `running`, and requeue it so its KV is recomputed on resume.
    auto preempt_one = [&](int protect) -> bool {
        for (int k = (int)running.size() - 1; k >= 0; --k) {
            if (k == protect) continue;
            Request& v = reqs[running[k]];
            for (int b : v.block_table) free_blocks.push_back(b);
            v.block_table.clear();
            waiting.push_front(running[k]);        // resume soon (recompute from prompt+output)
            running.erase(running.begin() + k);
            return true;
        }
        return false;
    };

    // Admission state: prefill one waiting request at a time, in PREFILL_CHUNK pieces interleaved
    // with decode, so a long prompt does not stall the running sequences. pf_tokens is the token
    // sequence to (re)cache: the prompt for a fresh request, or prompt+output[:-1] on resume.
    int pf = -1, pf_cursor = 0;
    std::vector<int> pf_tokens;

    std::vector<int> in_tok, past, out;
    std::vector<std::vector<int>> bts;
    while (!waiting.empty() || !running.empty() || pf != -1) {
        // (a) start prefilling the next waiting request if there is room for another sequence.
        if (pf == -1 && !waiting.empty() && (int)running.size() < n_slots) {
            pf = waiting.front(); waiting.pop_front();
            Request& r = reqs[pf];
            r.block_table.clear();
            pf_tokens = r.prompt;
            if (!r.output.empty())                 // resume: re-cache everything but the last token
                pf_tokens.insert(pf_tokens.end(), r.output.begin(), r.output.end() - 1);
            if ((int)pf_tokens.size() > max_ctx) pf_tokens.resize(max_ctx);
            pf_cursor = 0;
        }
        // (b) advance the in-progress prefill by one chunk.
        if (pf != -1) {
            Request& r = reqs[pf];
            int chunk = std::min(PREFILL_CHUNK, (int)pf_tokens.size() - pf_cursor);
            // ensure enough blocks for positions [0, pf_cursor+chunk); preempt running if needed.
            int need = blocks_for(pf_cursor + chunk);
            while ((int)r.block_table.size() < need) {
                if (free_blocks.empty() && !preempt_one(-1)) {
                    std::fprintf(stderr, "scheduler: out of KV blocks during prefill\n"); std::exit(1);
                }
                if (free_blocks.empty()) continue;
                r.block_table.push_back(free_blocks.back()); free_blocks.pop_back();
            }
            std::vector<int> piece(pf_tokens.begin() + pf_cursor, pf_tokens.begin() + pf_cursor + chunk);
            model.prefill(piece, r.block_table, pf_cursor);
            pf_cursor += chunk;
            if (pf_cursor >= (int)pf_tokens.size()) {      // prompt fully cached
                if (r.output.empty()) {                    // fresh: take the first generated token
                    r.cur = (r.sp.temp <= 0.0f) ? model.argmax_last()
                                                : sample(model.copy_logits(), r.sp, rng);
                    r.output.push_back(r.cur);
                } else {                                   // resume: state continues from output.back()
                    r.cur = r.output.back();
                }
                r.past = (int)pf_tokens.size();
                if (finished(r, stop_on_eos, max_ctx)) {
                    for (int b : r.block_table) free_blocks.push_back(b);
                    r.block_table.clear();
                } else running.push_back(pf);
                pf = -1;
            }
        }
        if (running.empty()) continue;

        // (c) grow each running sequence's block table for the token it is about to write, then
        // decode every running sequence together in one step.
        for (bool rescan = true; rescan; ) {
            rescan = false;
            for (size_t k = 0; k < running.size(); ++k) {
                Request& r = reqs[running[k]];
                int need = blocks_for(r.past + 1);
                while ((int)r.block_table.size() < need) {
                    if (free_blocks.empty()) {             // preempt a younger sequence and rescan
                        if (!preempt_one((int)k)) {
                            std::fprintf(stderr, "scheduler: out of KV blocks during decode\n"); std::exit(1);
                        }
                        rescan = true; break;
                    }
                    r.block_table.push_back(free_blocks.back()); free_blocks.pop_back();
                }
                if (rescan) break;
            }
        }

        in_tok.clear(); past.clear(); bts.clear();
        bool any_sampling = false;
        for (int i : running) { in_tok.push_back(reqs[i].cur); past.push_back(reqs[i].past);
                                bts.push_back(reqs[i].block_table);
                                if (reqs[i].sp.temp > 0.0f) any_sampling = true; }
        if (any_sampling) {
            // At least one request samples: get the full [B, vocab] logits and pick per request
            // (sample() does greedy for temp<=0, so mixed greedy/sampling batches are fine).
            const float* L = model.decode_logits_host(in_tok, past, bts);
            out.resize(running.size());
            for (size_t k = 0; k < running.size(); ++k)
                out[k] = sample(L + (size_t)k * V, V, reqs[running[k]].sp, rng);
        } else {
            model.decode(in_tok, past, bts, out);   // all greedy: GPU argmax, no full-logits copy
        }

        std::vector<int> still;
        for (size_t k = 0; k < running.size(); ++k) {
            Request& r = reqs[running[k]];
            r.cur = out[k]; r.past += 1; r.output.push_back(r.cur);
            if (finished(r, stop_on_eos, max_ctx)) {            // blocks return to the pool
                for (int b : r.block_table) free_blocks.push_back(b);
                r.block_table.clear();
            } else still.push_back(running[k]);
        }
        running.swap(still);
    }
}
