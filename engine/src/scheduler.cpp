#include "scheduler.hpp"
#include "log.hpp"
#include <algorithm>
#include <numeric>
#include <string>
#include <cstdint>

Scheduler::Scheduler(ModelRuntime& model, KVCacheManager& kv, const SchedulerConfig& cfg)
    : model_(model), kv_(kv), cfg_(cfg) {}

// 64-bit content hash of one block: hash(parent_hash, the block's token ids). Chaining the parent makes
// identical PREFIXES collide (not merely identical 16-token windows), so a cached block is reused only
// when the whole path of tokens leading to it matches. Pure hashing — collisions are accepted (the
// engine trades verification cost for speed, as vLLM does by default).
static uint64_t hash_block(uint64_t parent, const Request& r, int start, int n) {
    uint64_t h = parent ^ 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < n; ++i) {
        h ^= (uint64_t)(uint32_t)r.token_at(start + i);
        h *= 0x100000001B3ull;            // FNV-1a prime
        h = (h << 27) | (h >> 37);        // rotate for diffusion
    }
    return h ? h : 1;                      // reserve 0 as "no hash"
}

// On admission (block table empty): walk the request's full leading blocks, look each up in the content
// cache, and splice the hits onto its block table — advancing computed_ past them so the prefill never
// recomputes that KV. Stops at the first miss (a reusable prefix must be contiguous), and always leaves
// at least one token uncomputed (a forward must run to produce the next logits). A pure cache lookup —
// it references resident blocks and never allocates, so it cannot fail under pool pressure.
void Scheduler::acquire_prefix(Request* r) {
    stat_prompt_tok_ += r->num_tokens();
    if (!cfg_.prefix_cache) return;
    int bsz = kv_.block_size();
    int limit_blocks = (r->num_tokens() - 1) / bsz;   // full blocks reusable while leaving >=1 token
    uint64_t parent = 0;
    auto& bt = r->blocks();
    int nb = 0;
    for (; nb < limit_blocks; ++nb) {
        uint64_t h = hash_block(parent, *r, nb * bsz, bsz);
        int b = kv_.cache_lookup(h);      // refs the block on hit
        if (b < 0) break;                 // miss -> end of the reusable prefix
        bt.push_back(b);
        parent = h;
    }
    if (nb > 0) { r->adopt_prefix(nb, bsz, parent); stat_cache_hit_tok_ += nb * bsz; }
}

// After a forward advances a request: register any blocks that are now full AND fully computed into the
// content cache (their 16 token ids are immutable from here on), chaining each hash from the previous.
// A finished request's blocks stay cached after it retires (refcount 0 but mapping kept), so the next
// request sharing that prefix gets a hit.
void Scheduler::cache_filled(Request* r) {
    int bsz = kv_.block_size();
    int full = r->computed() / bsz;       // blocks whose every token is committed
    auto& bt = r->blocks();
    uint64_t parent = r->last_hash();
    for (int idx = r->cached_blocks(); idx < full; ++idx) {
        parent = hash_block(parent, *r, idx * bsz, bsz);
        kv_.cache_insert(bt[idx], parent);
    }
    r->set_cached_blocks(full);
    r->set_last_hash(parent);
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
        stat_preempt_++;                              // instrumentation: a victim is being preempted
        stat_recomp_tok_ += running_[k]->computed();  // its cached KV (computed_ tokens) is discarded
        if (stat_preempt_ == 1 || stat_preempt_ % 200 == 0) log_kvstat();   // catch preemption even in short runs
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
    // 2. admit waiters into the running set while slots are free, splicing in any cached prefix first
    //    (an empty block table marks a fresh or just-preempted request, both of which may reuse KV)
    while ((int)running_.size() < cfg_.n_slots && !waiting_.empty()) {
        if (waiting_.front()->blocks().empty()) acquire_prefix(waiting_.front().get());
        running_.push_back(std::move(waiting_.front()));
        waiting_.pop_front();
    }
    // 3. pack the running set under the token budget, scheduling by ascending remaining work — so
    //    decodes (remaining==1) come first (low TPOT), then the shortest prefills. We sort an index
    //    view, never running_ itself: its order is request age, which grow() uses to pick the youngest
    //    to preempt. A preempt mutates running_, so rebuild from scratch when it happens.
    CurrentBatch batch;
    std::vector<int> order;
    int step_prefill = 0, step_decode = 0;   // this step's prefill/decode row split (instrumentation)
    for (bool restart = true; restart; ) {
        restart = false;
        batch.clear();
        step_prefill = step_decode = 0;
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
            (n > 1 ? step_prefill : step_decode) += n;   // n>1 => prefill chunk, n==1 => decode step
            budget -= n;
        }
    }
    if (batch.empty()) return;
    // instrumentation: accumulate per-step counters + track peak pool occupancy
    stat_steps_++;
    stat_prefill_rows_ += step_prefill;
    stat_decode_rows_  += step_decode;
    stat_fwd_rows_     += step_prefill + step_decode;
    int used = kv_.num_blocks() - kv_.num_free();
    if (used > stat_peak_used_) stat_peak_used_ = used;
    if (stat_steps_ % 200 == 0) log_kvstat();
    // 4. one merged forward (+ GPU sampling), then apply results, cache newly-full blocks, reclaim
    model_.forward(batch.input, batch.sampled);
    batch.apply();
    if (cfg_.prefix_cache)
        for (Request* r : batch.requests) cache_filled(r);   // finished reqs cached too (before retire)
    for (Request* r : batch.finished) retire(r);
}

// Periodic cumulative instrumentation: pool occupancy + how much KV the preemption path recomputed.
// recomp% = discarded-KV tokens / total forwarded rows (the redo is included in the latter), i.e. the
// fraction of GPU token-work spent rebuilding KV that was thrown away by preemption.
void Scheduler::log_kvstat() {
    int nb = kv_.num_blocks();
    double peak_pct = nb > 0 ? 100.0 * stat_peak_used_ / nb : 0.0;
    double recomp_pct = stat_fwd_rows_ > 0 ? 100.0 * stat_recomp_tok_ / stat_fwd_rows_ : 0.0;
    double hit_pct = stat_prompt_tok_ > 0 ? 100.0 * stat_cache_hit_tok_ / stat_prompt_tok_ : 0.0;
    LOG_INFO("[kvstat] steps=%ld peak_used=%d/%d (%.1f%%) | rows: prefill=%ld decode=%ld | "
             "preempt=%ld recomp_tok=%ld (%.2f%% of fwd) | prefix_hit=%ld/%ld tok (%.1f%%)",
             stat_steps_, stat_peak_used_, nb, peak_pct,
             stat_prefill_rows_, stat_decode_rows_, stat_preempt_, stat_recomp_tok_, recomp_pct,
             stat_cache_hit_tok_, stat_prompt_tok_, hit_pct);
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
