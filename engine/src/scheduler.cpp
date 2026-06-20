#include "scheduler.hpp"
#include <algorithm>
#include <numeric>
#include <string>

Scheduler::Scheduler(ModelRuntime& model, KVCacheManager& kv, const SchedulerConfig& cfg)
    : model_(model), kv_(kv), cfg_(cfg) {}

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

// How many tokens request r should advance this step: all its uncomputed tokens, capped by the
// remaining budget and (for multi-token prefill chunks) by max_prefill. 0 when none are left.
int Scheduler::chunk_size(const Request* r, int budget) const {
    int remaining = r->remaining();
    if (remaining <= 0) return 0;
    int n = std::min(remaining, budget);
    if (remaining > 1) n = std::min(n, cfg_.max_prefill);   // chunk long prefills
    return n;
}

// Grow r's KV block table to cover `upto` tokens, preempting younger sequences when the pool is dry:
// each preemption frees the youngest running sequence other than r — returns its blocks, resets its
// cursor, and requeues it (front) so its KV is recomputed from prompt+output when later resumed.
// Returns false if the pool can't fit r even after preempting everyone else: r is then failed (its
// sink gets an OverCapacity error) and retired, and the caller must rebuild the batch.
bool Scheduler::grow(Request* r, int upto) {
    while (!kv_.grow(r->blocks(), upto)) {
        // youngest running sequence other than r (scan from the end, skipping r itself)
        int k = (int)running_.size() - 1;
        while (k >= 0 && running_[k].get() == r) --k;
        if (k < 0) {   // nobody else to preempt: r alone can't fit the pool -> fail it
            if (r->sink()) r->sink()->error(EngineErrc::OverCapacity,
                "out of KV cache: pool has " + std::to_string(kv_.num_blocks()) +
                " blocks of " + std::to_string(kv_.block_size()) + " tokens, cannot grow a sequence to " +
                std::to_string(upto) + " tokens");
            retire(r);
            return false;
        }
        release(running_[k].get());
        running_[k]->reset_for_recompute();           // recompute from the start on resume
        waiting_.push_front(std::move(running_[k]));   // requeue at the front
        running_.erase(running_.begin() + k);
    }
    return true;
}

// One scheduling iteration over a step-local batch: cancel, admit, pack, forward, apply, reclaim.
void Scheduler::step() {
    // 1. drop cancelled requests (client disconnected), freeing their KV — polled once per step
    for (auto it = waiting_.begin(); it != waiting_.end(); ) {
        if ((*it)->sink() && (*it)->sink()->cancelled()) { release(it->get()); it = waiting_.erase(it); }
        else ++it;
    }
    for (auto it = running_.begin(); it != running_.end(); ) {
        if ((*it)->sink() && (*it)->sink()->cancelled()) { release(it->get()); it = running_.erase(it); }
        else ++it;
    }
    // 2. admit waiters into the running set while slots are free
    while ((int)running_.size() < cfg_.n_slots && !waiting_.empty()) {
        running_.push_back(std::move(waiting_.front()));
        waiting_.pop_front();
    }
    // 3. pack the running set under the token budget, scheduling by ascending remaining work — so
    //    decodes (remaining==1) come first (low TPOT), then the shortest prefills. We sort an index
    //    view, never running_ itself: its order is request age, which grow() uses to pick the youngest
    //    to preempt. A preempt mutates running_, so rebuild from scratch when it happens.
    CurrentBatch batch;
    std::vector<int> order;
    for (bool restart = true; restart; ) {
        restart = false;
        batch.clear();
        order.resize(running_.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
            [&](int a, int b) { return running_[a]->remaining() < running_[b]->remaining(); });
        int budget = cfg_.max_batch_tokens;
        for (int idx : order) {
            if (budget <= 0) break;
            Request* r = running_[idx].get();
            int n = chunk_size(r, budget);
            if (n == 0) continue;
            if (!grow(r, r->computed() + n)) { restart = true; break; }
            batch.add_request(r, n);
            budget -= n;
        }
    }
    if (batch.empty()) return;
    // 4. one merged forward (+ GPU sampling), then apply results + reclaim finished
    model_.forward(batch.input, batch.sampled);
    batch.apply();
    for (Request* r : batch.finished) retire(r);
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
            // admission control: reject when the waiting queue is full -> over-capacity
            if (cfg_.max_queue > 0 && (int)waiting_.size() >= cfg_.max_queue) {
                if (r->sink()) r->sink()->error(EngineErrc::OverCapacity,
                    "request queue full: " + std::to_string(waiting_.size()) +
                    " requests already waiting (limit " + std::to_string(cfg_.max_queue) +
                    "), engine running up to " + std::to_string(cfg_.n_slots) +
                    " concurrent sequences; retry shortly");
                // r is dropped here; the handler's sink ref still delivers the error event.
            } else {
                waiting_.push_back(std::move(r));   // admitted: into the scheduling queue
            }
        }
        if (busy()) step();
    }
}
