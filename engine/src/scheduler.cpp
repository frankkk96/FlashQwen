#include "scheduler.hpp"
#include <algorithm>
#include <string>

Scheduler::Scheduler(ModelRuntime& model, KVCacheManager& kv, int n_slots, int max_queue,
                     int max_batch_tokens, int max_prefill)
    : model_(model), kv_(kv) {
    n_slots_          = std::min(n_slots, model_.max_batch());
    max_queue_        = max_queue;
    max_batch_tokens_ = max_batch_tokens;
    max_prefill_      = max_prefill;
    max_ctx_ = model_.max_ctx();
    bsz_     = kv_.block_size();
}

void Scheduler::release(Request* r) {
    kv_.free(r->blocks());   // return blocks to the pool and clear the table
}

// A request has left the engine for good (finished or failed): return its KV blocks to the pool and
// drop it from the running set (destroying it).
void Scheduler::retire(Request* r) {
    release(r);
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
        v->reset_for_recompute();               // recompute from the start on resume
        waiting_.push_front(std::move(running_[k]));
        running_.erase(running_.begin() + k);
        return true;
    }
    return false;
}

// How many tokens request r should advance this step. Pass 0 schedules decodes (1 token), pass 1
// schedules prefill chunks (>1 token); returns 0 to skip r in this pass / when the budget is spent.
int Scheduler::chunk_size(const Request* r, int pass, int budget) const {
    int remaining = r->remaining();
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
    while (!kv_.grow(r->blocks(), upto)) {
        if (!preempt_one(r)) {
            std::string msg = "out of KV cache: pool has " + std::to_string(kv_.num_blocks()) +
                " blocks of " + std::to_string(bsz_) + " tokens, cannot grow a sequence to " +
                std::to_string(upto) + " tokens";
            if (r->sink()) r->sink()->error(EngineErrc::OverCapacity, std::move(msg));
            retire(r);
            return false;
        }
    }
    return true;
}

// One scheduling iteration, in four phases over a step-local batch:
//   1. drop any request the transport cancelled (client disconnected), freeing its KV. Polled once
//      per step here; freeing the unique_ptr releases its ref to the sink (the transport keeps the
//      sink alive itself).
//   2. admit waiters into the running set while slots are free.
//   3. pack the running set under the token budget: decode requests first (pass 0), then prefill
//      chunks (pass 1). Growing a request may preempt a younger one, which mutates running_, so we
//      rebuild the batch from scratch whenever that happens.
//   4. run one merged forward (+ GPU sampling), apply the results to each request, and reclaim the
//      resources of any that finished.
void Scheduler::step() {
    // 1. cancelled
    for (auto it = waiting_.begin(); it != waiting_.end(); ) {
        if ((*it)->sink() && (*it)->sink()->cancelled()) { release(it->get()); it = waiting_.erase(it); }
        else ++it;
    }
    for (auto it = running_.begin(); it != running_.end(); ) {
        if ((*it)->sink() && (*it)->sink()->cancelled()) { release(it->get()); it = running_.erase(it); }
        else ++it;
    }
    // 2. admit
    while ((int)running_.size() < n_slots_ && !waiting_.empty()) {
        running_.push_back(std::move(waiting_.front()));
        waiting_.pop_front();
    }
    // 3. pack
    CurrentBatch batch;
    for (bool restart = true; restart; ) {
        restart = false;
        batch.clear();
        int budget = max_batch_tokens_;
        for (int pass = 0; pass < 2 && !restart; ++pass)
            for (size_t i = 0; i < running_.size() && !restart; ++i) {
                if (budget <= 0) break;
                Request* r = running_[i].get();
                int n = chunk_size(r, pass, budget);
                if (n == 0) continue;
                if (!grow_or_preempt(r, r->computed() + n)) { restart = true; break; }
                batch.add_request(r, n);
                budget -= n;
            }
    }
    if (batch.empty()) return;
    // 4. forward + apply + reclaim
    model_.forward(batch.input, batch.sampled);   // merged forward + per-row GPU sampling
    batch.apply(max_ctx_);                          // advance + sample + stream -> batch.finished
    for (Request* r : batch.finished) retire(r);   // finished: free KV + drop from running
}

void Scheduler::submit(std::unique_ptr<Request> r) {
    { std::lock_guard<std::mutex> lk(inbound_mu_); inbound_.push_back(std::move(r)); }
    inbound_cv_.notify_one();
}

// The engine thread: drain submitted requests (rejecting any that overflow the queue), then advance
// one scheduling iteration. Blocks for the process lifetime.
void Scheduler::run() {
    std::deque<std::unique_ptr<Request>> incoming;
    while (true) {
        {
            std::unique_lock<std::mutex> lk(inbound_mu_);
            if (inbound_.empty() && !busy())
                inbound_cv_.wait(lk, [&] { return !inbound_.empty(); });
            incoming.swap(inbound_);
        }
        while (!incoming.empty()) {
            auto r = std::move(incoming.front()); incoming.pop_front();
            if (!can_admit()) {   // admission control: queue full -> reject as over-capacity
                if (r->sink()) r->sink()->error(EngineErrc::OverCapacity,
                    "request queue full: " + std::to_string(queue_depth()) +
                    " requests already waiting (limit " + std::to_string(max_queue_) +
                    "), engine running up to " + std::to_string(n_slots_) +
                    " concurrent sequences; retry shortly");
                // r is dropped here; the handler's sink ref still delivers the error event.
            } else {
                waiting_.push_back(std::move(r));   // admitted: into the scheduling queue
            }
        }
        if (busy()) step();
    }
}
