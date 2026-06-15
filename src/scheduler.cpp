#include "scheduler.hpp"
#include <algorithm>
#include <deque>

static const int EOS1 = 151645, EOS2 = 151643;   // <|im_end|>, <|endoftext|>

static inline bool finished(const Request& r, bool stop_on_eos, int max_ctx) {
    if ((int)r.output.size() >= r.max_new) return true;
    if (stop_on_eos && !r.output.empty() && (r.cur == EOS1 || r.cur == EOS2)) return true;
    if (r.past >= max_ctx) return true;   // KV slot full — no room for another token
    return false;
}

void run_continuous(Model& model, std::vector<Request>& reqs, int n_slots, bool stop_on_eos,
                    std::mt19937& rng) {
    n_slots = std::min(n_slots, model.max_batch());
    int max_ctx = model.max_ctx(), V = model.config().vocab_size;

    std::vector<int> free_slots;                  // available KV slots (reused as sequences end)
    for (int s = n_slots - 1; s >= 0; --s) free_slots.push_back(s);
    std::deque<int> waiting;                       // request indices not yet started
    for (int i = 0; i < (int)reqs.size(); ++i) waiting.push_back(i);
    std::vector<int> running;                      // request indices currently decoding

    std::vector<int> in_tok, past, slots, out;
    while (!waiting.empty() || !running.empty()) {
        // Admit waiting requests into any free slots: prefill, take the first token, and either
        // finish immediately or join the running set.
        while (!waiting.empty() && !free_slots.empty()) {
            int i = waiting.front(); waiting.pop_front();
            Request& r = reqs[i];
            r.slot = free_slots.back(); free_slots.pop_back();
            // clamp an over-long prompt to the slot's capacity (last token predicts the next)
            if ((int)r.prompt.size() > max_ctx) r.prompt.resize(max_ctx);
            model.prefill(r.prompt, r.slot, 0);
            r.cur  = (r.sp.temp <= 0.0f) ? model.argmax_last()
                                         : sample(model.copy_logits(), r.sp, rng);
            r.past = (int)r.prompt.size();
            r.output.push_back(r.cur);
            if (finished(r, stop_on_eos, max_ctx)) free_slots.push_back(r.slot);
            else running.push_back(i);
        }
        if (running.empty()) continue;

        // Decode every running sequence together in one step.
        in_tok.clear(); past.clear(); slots.clear();
        bool any_sampling = false;
        for (int i : running) { in_tok.push_back(reqs[i].cur); past.push_back(reqs[i].past);
                                slots.push_back(reqs[i].slot);
                                if (reqs[i].sp.temp > 0.0f) any_sampling = true; }
        if (any_sampling) {
            // At least one request samples: get the full [B, vocab] logits and pick per request
            // (sample() does greedy for temp<=0, so mixed greedy/sampling batches are fine).
            const float* L = model.decode_logits_host(in_tok, past, slots);
            out.resize(running.size());
            for (size_t k = 0; k < running.size(); ++k)
                out[k] = sample(L + (size_t)k * V, V, reqs[running[k]].sp, rng);
        } else {
            model.decode(in_tok, past, slots, out);   // all greedy: GPU argmax, no full-logits copy
        }

        std::vector<int> still;
        for (size_t k = 0; k < running.size(); ++k) {
            Request& r = reqs[running[k]];
            r.cur = out[k]; r.past += 1; r.output.push_back(r.cur);
            if (finished(r, stop_on_eos, max_ctx)) free_slots.push_back(r.slot);   // slot returns to the pool
            else still.push_back(running[k]);
        }
        running.swap(still);
    }
}
