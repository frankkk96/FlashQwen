#include "scheduler.hpp"
#include <algorithm>
#include <string>

Scheduler::Scheduler(ModelRuntime& model, KVCacheManager& kv, int n_slots, int max_queue,
                     int max_batch_tokens, int max_prefill, std::mt19937& rng)
    : model_(model), kv_(kv), rng_(rng) {
    n_slots_          = std::min(n_slots, model_.max_batch());
    max_queue_        = max_queue;
    max_batch_tokens_ = max_batch_tokens;
    max_prefill_      = max_prefill;
    max_ctx_ = model_.max_ctx();
    V_       = model_.spec().vocab_size;
    bsz_     = kv_.block_size();
}

void Scheduler::add(std::unique_ptr<Request> r) { waiting_.push_back(std::move(r)); }

bool Scheduler::finished(const Request* r) const {
    if ((int)r->output.size() >= r->max_new) return true;
    if (!r->output.empty() &&
        std::find(r->stop_ids.begin(), r->stop_ids.end(), r->cur) != r->stop_ids.end())
        return true;                            // sampled a caller-provided stop token
    if (r->num_tokens() >= max_ctx_) return true;   // sequence full — no room for another token
    return false;
}

void Scheduler::release(Request* r) {
    kv_.free(r->block_table);   // return blocks to the pool and clear the table
}

void Scheduler::erase_running(Request* r) {
    for (auto it = running_.begin(); it != running_.end(); ++it)
        if (it->get() == r) { running_.erase(it); return; }   // unique_ptr destroyed -> Request freed
}

// Free the youngest running sequence other than `protect`: return its blocks, reset its cursor, and
// requeue it so its KV is recomputed (from prompt+output) when it is later resumed.
bool Scheduler::preempt_one(Request* protect) {
    for (int k = (int)running_.size() - 1; k >= 0; --k) {
        if (running_[k].get() == protect) continue;
        Request* v = running_[k].get();
        release(v);
        v->computed = 0;                        // recompute from the start on resume
        waiting_.push_front(std::move(running_[k]));
        running_.erase(running_.begin() + k);
        return true;
    }
    return false;
}

// Drop any request the transport has cancelled (client disconnected). Polled once per step; freeing
// the unique_ptr releases the request's ref to its sink (the transport keeps the sink alive itself).
void Scheduler::drop_cancelled() {
    for (auto it = waiting_.begin(); it != waiting_.end(); ) {
        if ((*it)->sink && (*it)->sink->cancelled()) { release(it->get()); it = waiting_.erase(it); }
        else ++it;
    }
    for (auto it = running_.begin(); it != running_.end(); ) {
        if ((*it)->sink && (*it)->sink->cancelled()) { release(it->get()); it = running_.erase(it); }
        else ++it;
    }
}

void Scheduler::admit() {
    while ((int)running_.size() < n_slots_ && !waiting_.empty()) {
        running_.push_back(std::move(waiting_.front()));
        waiting_.pop_front();
    }
}

// How many tokens request r should advance this step. Pass 0 schedules decodes (1 token), pass 1
// schedules prefill chunks (>1 token); returns 0 to skip r in this pass / when the budget is spent.
int Scheduler::chunk_size(const Request* r, int pass, int budget) const {
    int remaining = r->num_tokens() - r->computed;
    if (remaining <= 0) return 0;
    if (pass == 0 && remaining != 1) return 0;   // decode pass: only 1-token requests
    if (pass == 1 && remaining <= 1) return 0;   // prefill pass: only multi-token requests
    int n = std::min(remaining, budget);
    if (remaining > 1) n = std::min(n, max_prefill_);   // chunk long prefills
    return n;
}

// Grow r's block table to cover `upto` tokens, preempting a younger sequence if the pool is dry.
// Returns false if the pool can't fit r even after preempting everyone else: r is failed (its sink
// gets an OverCapacity error) and freed, and the caller must rebuild the batch.
bool Scheduler::grow_or_preempt(Request* r, int upto) {
    while (!kv_.grow(r->block_table, upto)) {
        if (!preempt_one(r)) {
            std::string msg = "out of KV cache: pool has " + std::to_string(kv_.num_blocks()) +
                " blocks of " + std::to_string(bsz_) + " tokens, cannot grow a sequence to " +
                std::to_string(upto) + " tokens";
            release(r);
            if (r->sink) r->sink->error(EngineErrc::OverCapacity, std::move(msg));
            erase_running(r);
            return false;
        }
    }
    return true;
}

// Append request r's `n` query rows (positions computed..computed+n-1) to the batch, recording the
// frontier row for sampling when the chunk reaches the end of the request's committed tokens.
void Scheduler::append_request(StepBatch& b, Request* r, int n) {
    int ri = (int)b.input.block_tables.size();
    for (int k = 0; k < n; ++k) {
        b.input.tokens.push_back(r->token_at(r->computed + k));
        b.input.positions.push_back(r->computed + k);
        b.input.req_index.push_back(ri);
    }
    b.input.block_tables.push_back(r->block_table);
    if (r->computed + n == r->num_tokens()) {            // chunk reaches the frontier -> sample
        b.input.logits_rows.push_back((int)b.input.tokens.size() - 1);
        b.sampling.push_back(r);
    }
    b.scheduled.push_back(r);
    b.num_new.push_back(n);
}

// Build the flattened batch under the token budget: decode requests first (pass 0), then prefill
// chunks (pass 1). Growing a request may preempt a younger one, which mutates running_, so we rebuild
// the batch from scratch whenever that happens.
void Scheduler::build_batch(StepBatch& b) {
    for (bool restart = true; restart; ) {
        restart = false;
        b.clear();
        int budget = max_batch_tokens_;
        for (int pass = 0; pass < 2 && !restart; ++pass)
            for (size_t i = 0; i < running_.size() && !restart; ++i) {
                if (budget <= 0) break;
                Request* r = running_[i].get();
                int n = chunk_size(r, pass, budget);
                if (n == 0) continue;
                if (!grow_or_preempt(r, r->computed + n)) { restart = true; break; }
                append_request(b, r, n);
                budget -= n;
            }
    }
}

// Run the merged forward: GPU argmax for an all-greedy batch, host-side per-row sampling otherwise.
void Scheduler::run_forward(const StepBatch& b, std::vector<int>& sampled) {
    bool any_sampling = false;
    for (Request* r : b.sampling) if (r->sp.temp > 0.0f) any_sampling = true;
    sampled.resize(b.sampling.size());
    if (any_sampling) {
        const float* L = model_.forward_logits_host(b.input);
        for (size_t i = 0; i < b.sampling.size(); ++i)
            sampled[i] = sample(L + (size_t)i * V_, V_, b.sampling[i]->sp, rng_);
    } else {
        model_.forward(b.input, sampled);
    }
}

// Advance every scheduled request's cursor; for the sampling ones append + stream the token, and
// finish (stream done + free) those that hit a stop id / max_new / a full context.
void Scheduler::apply_results(const StepBatch& b, const std::vector<int>& sampled) {
    for (size_t i = 0; i < b.scheduled.size(); ++i) b.scheduled[i]->computed += b.num_new[i];
    for (size_t i = 0; i < b.sampling.size(); ++i) {
        Request* r = b.sampling[i];
        r->cur = sampled[i];
        r->output.push_back(r->cur);
        if (r->sink) r->sink->token(r->cur);
        if (finished(r)) {
            bool stopped = !r->stop_ids.empty() &&
                std::find(r->stop_ids.begin(), r->stop_ids.end(), r->cur) != r->stop_ids.end();
            int comp = (int)r->output.size() - (stopped ? 1 : 0);   // exclude the stop token itself
            release(r);
            if (r->sink) r->sink->done(stopped ? "stop" : "length", (int)r->prompt.size(),
                                       comp < 0 ? 0 : comp);
            erase_running(r);
        }
    }
}

void Scheduler::step() {
    drop_cancelled();
    admit();
    if (running_.empty()) return;
    build_batch(batch_);
    if (batch_.scheduled.empty()) return;
    run_forward(batch_, sampled_);
    apply_results(batch_, sampled_);
}
