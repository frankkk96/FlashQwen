#include "scheduler.hpp"
#include <algorithm>
#include <deque>

static const int EOS1 = 151645, EOS2 = 151643;   // <|im_end|>, <|endoftext|>

static inline bool finished(const Request& r, bool stop_on_eos) {
    if ((int)r.output.size() >= r.max_new) return true;
    if (stop_on_eos && !r.output.empty() && (r.cur == EOS1 || r.cur == EOS2)) return true;
    return false;
}

void run_continuous(Model& model, std::vector<Request>& reqs, int n_slots, bool stop_on_eos) {
    n_slots = std::min(n_slots, model.max_batch());

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
            model.prefill(r.prompt, r.slot, 0);
            r.cur  = model.argmax_last();
            r.past = (int)r.prompt.size();
            r.output.push_back(r.cur);
            if (finished(r, stop_on_eos)) free_slots.push_back(r.slot);
            else running.push_back(i);
        }
        if (running.empty()) continue;

        // Decode every running sequence together in one step.
        in_tok.clear(); past.clear(); slots.clear();
        for (int i : running) { in_tok.push_back(reqs[i].cur); past.push_back(reqs[i].past);
                                slots.push_back(reqs[i].slot); }
        model.decode(in_tok, past, slots, out);

        std::vector<int> still;
        for (size_t k = 0; k < running.size(); ++k) {
            Request& r = reqs[running[k]];
            r.cur = out[k]; r.past += 1; r.output.push_back(r.cur);
            if (finished(r, stop_on_eos)) free_slots.push_back(r.slot);   // slot returns to the pool
            else still.push_back(running[k]);
        }
        running.swap(still);
    }
}

void run_static(Model& model, std::vector<Request>& reqs, int batch, bool stop_on_eos) {
    batch = std::min(batch, model.max_batch());

    for (int start = 0; start < (int)reqs.size(); start += batch) {
        int B = std::min(batch, (int)reqs.size() - start);
        std::vector<int> in_tok(B), past(B), slots(B), out;
        std::vector<char> done(B, 0);

        for (int b = 0; b < B; ++b) {
            Request& r = reqs[start + b];
            model.prefill(r.prompt, b, 0);
            r.cur = model.argmax_last(); r.past = (int)r.prompt.size(); r.slot = b;
            r.output.push_back(r.cur);
            in_tok[b] = r.cur; past[b] = r.past; slots[b] = b;
            done[b] = finished(r, stop_on_eos);
        }

        // The group keeps its full width B until every sequence in it has finished — sequences
        // that finish early still occupy a slot (head-of-line blocking). Slots are only reused
        // for the next group.
        auto any_active = [&]() { for (char d : done) if (!d) return true; return false; };
        while (any_active()) {
            model.decode(in_tok, past, slots, out);
            for (int b = 0; b < B; ++b) {
                Request& r = reqs[start + b];
                in_tok[b] = out[b]; past[b] += 1;
                if (!done[b]) {
                    r.cur = out[b]; r.output.push_back(r.cur);
                    done[b] = finished(r, stop_on_eos);
                }
            }
        }
    }
}
