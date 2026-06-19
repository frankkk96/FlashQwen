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

void Scheduler::add(Request* r) { waiting_.push_back(r); }

void Scheduler::remove(Request* r) {
    for (auto it = waiting_.begin(); it != waiting_.end(); ++it)
        if (*it == r) { waiting_.erase(it); return; }
    for (auto it = running_.begin(); it != running_.end(); ++it)
        if (*it == r) { release(r); running_.erase(it); return; }
}

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

// Free the youngest running sequence other than `protect`: return its blocks, reset its cursor, and
// requeue it so its KV is recomputed (from prompt+output) when it is later resumed.
bool Scheduler::preempt_one(Request* protect) {
    for (int k = (int)running_.size() - 1; k >= 0; --k) {
        if (running_[k] == protect) continue;
        Request* v = running_[k];
        release(v);
        v->computed = 0;                        // recompute from the start on resume
        waiting_.push_front(v);
        running_.erase(running_.begin() + k);
        return true;
    }
    return false;
}

void Scheduler::step(const TokenFn& on_token, const FinishFn& on_finish, const ErrorFn& on_error) {
    // (1) admit waiting -> running while there are free slots.
    while ((int)running_.size() < n_slots_ && !waiting_.empty()) {
        running_.push_back(waiting_.front());
        waiting_.pop_front();
    }
    if (running_.empty()) return;

    // (2) build the flattened batch under the token budget. Decode requests (num_new==1) are
    // scheduled first (pass 0), then prefill chunks (pass 1). Growing a request's blocks may preempt
    // a younger one; preemption mutates running_, so we rebuild the batch from scratch on any change.
    int budget = 0;
    auto reset_batch = [&] {
        in_tok_.clear(); pos_.clear(); reqidx_.clear(); bts_.clear(); lrows_.clear();
        sched_.clear(); sched_new_.clear(); sampling_.clear();
        budget = max_batch_tokens_;
    };

    bool restart = true;
    while (restart) {
        restart = false;
        reset_batch();

        auto try_schedule = [&](Request* r, int pass) -> bool {   // false => r removed; rebuild
            int remaining = r->num_tokens() - r->computed;
            if (remaining <= 0) return true;
            if (pass == 0 && remaining != 1) return true;   // decode pass: only 1-token requests
            if (pass == 1 && remaining <= 1) return true;   // prefill pass: only multi-token requests
            int num_new = std::min(remaining, budget);
            if (remaining > 1) num_new = std::min(num_new, max_prefill_);   // chunk long prefills
            if (num_new <= 0) return true;                  // budget exhausted; try a later step

            while (!kv_.grow(r->block_table, r->computed + num_new)) {
                if (!preempt_one(r)) {
                    // pool can't fit this request even after preempting everyone else: fail just it.
                    std::string msg = "out of KV cache: pool has " + std::to_string(kv_.num_blocks()) +
                        " blocks of " + std::to_string(bsz_) + " tokens, cannot grow a sequence to " +
                        std::to_string(r->computed + num_new) + " tokens";
                    release(r);
                    auto it = std::find(running_.begin(), running_.end(), r);
                    if (it != running_.end()) running_.erase(it);
                    if (on_error) on_error(r, EngineErrc::OverCapacity, std::move(msg));
                    return false;
                }
            }

            int ri = (int)bts_.size();
            for (int k = 0; k < num_new; ++k) {
                in_tok_.push_back(r->token_at(r->computed + k));
                pos_.push_back(r->computed + k);
                reqidx_.push_back(ri);
            }
            bts_.push_back(r->block_table);
            if (r->computed + num_new == r->num_tokens()) {   // chunk reaches the frontier -> sample
                lrows_.push_back((int)in_tok_.size() - 1);
                sampling_.push_back(r);
            }
            sched_.push_back(r);
            sched_new_.push_back(num_new);
            budget -= num_new;
            return true;
        };

        for (int pass = 0; pass < 2 && !restart; ++pass)
            for (size_t i = 0; i < running_.size() && !restart; ++i) {
                if (budget <= 0) break;
                if (!try_schedule(running_[i], pass)) restart = true;
            }
    }

    if (sched_.empty()) return;

    // (3) one merged forward over the whole batch.
    bool any_sampling = false;
    for (Request* r : sampling_) if (r->sp.temp > 0.0f) any_sampling = true;
    std::vector<int> sampled(sampling_.size());
    if (any_sampling) {
        const float* L = model_.forward_logits_host(in_tok_, pos_, reqidx_, bts_, lrows_);
        for (size_t i = 0; i < sampling_.size(); ++i)
            sampled[i] = sample(L + (size_t)i * V_, V_, sampling_[i]->sp, rng_);
    } else {
        model_.forward(in_tok_, pos_, reqidx_, bts_, lrows_, out_);
        sampled = out_;
    }

    // (4) advance every scheduled request's cursor; append + emit tokens for the sampling ones.
    for (size_t i = 0; i < sched_.size(); ++i) sched_[i]->computed += sched_new_[i];
    std::vector<Request*> done;
    for (size_t i = 0; i < sampling_.size(); ++i) {
        Request* r = sampling_[i];
        r->cur = sampled[i];
        r->output.push_back(r->cur);
        if (on_token) on_token(r, r->cur);
        if (finished(r)) done.push_back(r);
    }
    for (Request* r : done) {
        release(r);
        auto it = std::find(running_.begin(), running_.end(), r);
        if (it != running_.end()) running_.erase(it);
        if (on_finish) on_finish(r);
    }
}
