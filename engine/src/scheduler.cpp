#include "scheduler.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>

#include "log.hpp"

Scheduler::Scheduler(ModelRuntime& model, KVCacheManager& kv,
                     const SchedulerConfig& cfg)
    : model_(model), kv_(kv), cfg_(cfg) {}

// 64-bit content hash of one block = hash(parent_hash, block's token ids).
// Chaining the parent makes whole PREFIXES collide (not just identical 16-token
// windows), so a block is reused only when the full path of tokens to it
// matches. Collisions are accepted (verification cost traded for speed, as vLLM
// does by default).
static uint64_t HashBlock(uint64_t parent, const Request& r, int start, int n) {
  uint64_t h = parent ^ 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < n; ++i) {
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(r.TokenAt(start + i)));
    h *= 0x100000001B3ull;      // FNV-1a prime
    h = (h << 27) | (h >> 37);  // rotate for diffusion
  }
  return h ? h : 1;  // reserve 0 as "no hash"
}

// On admission (block table empty): walk the request's full leading blocks,
// look each up in the content cache, and splice hits onto its block table,
// advancing computed_ so prefill skips that KV. Stops at the first miss (reused
// prefix must be contiguous) and always leaves >=1 token uncomputed (a forward
// must run to produce the next logits). Pure lookup: refs resident blocks,
// never allocates, so cannot fail under pool pressure.
void Scheduler::AcquirePrefix(Request* r) {
  stat_prompt_tok_ += r->NumTokens();
  if (!cfg_.prefix_cache) return;
  int bsz = kv_.BlockSize();
  int limit_blocks = (r->NumTokens() - 1) /
                     bsz;  // full blocks reusable while leaving >=1 token
  uint64_t parent = 0;
  auto& bt = r->Blocks();
  int nb = 0;
  for (; nb < limit_blocks; ++nb) {
    uint64_t h = HashBlock(parent, *r, nb * bsz, bsz);
    int b = kv_.CacheLookup(h);  // refs the block on hit
    if (b < 0) break;            // miss -> end of the reusable prefix
    bt.push_back(b);
    parent = h;
  }
  if (nb > 0) {
    r->AdoptPrefix(nb, bsz, parent);
    stat_cache_hit_tok_ += static_cast<int64_t>(nb) * bsz;
    if (stat_cache_hit_tok_ ==
        static_cast<int64_t>(nb) *
            bsz)  // first hit ever: positive proof in log
      LOG_INFO(
          "[prefix] first cache hit: reused %d blocks (%d tokens) of a "
          "%d-token prompt",
          nb, nb * bsz, r->NumTokens());
  }
}

// After a forward advances a request: register blocks now full AND fully
// computed into the content cache (their token ids are immutable from here),
// chaining each hash from the previous. A finished request's blocks stay cached
// after retire (refcount 0, mapping kept), so the next sharer gets a hit.
void Scheduler::CacheFilled(Request* r) {
  int bsz = kv_.BlockSize();
  int full = r->Computed() / bsz;  // blocks whose every token is committed
  auto& bt = r->Blocks();
  uint64_t parent = r->LastHash();
  for (int idx = r->CachedBlocks(); idx < full; ++idx) {
    parent = HashBlock(parent, *r, idx * bsz, bsz);
    kv_.CacheInsert(bt[idx], parent);
  }
  r->SetCachedBlocks(full);
  r->SetLastHash(parent);
}

void Scheduler::Release(Request* r) {
  kv_.Free(r->Blocks());  // return blocks to the pool and clear the table
}

// Request has left for good (finished or failed): return its KV to the pool and
// drop it from running (destroying it).
void Scheduler::Retire(Request* r) {
  Release(r);
  for (auto it = running_.begin(); it != running_.end(); ++it)
    if (it->get() == r) {
      running_.erase(it);
      return;
    }  // unique_ptr destroyed -> Request freed
}

// Tokens r should advance this step: all uncomputed tokens, capped by remaining
// budget and (for multi-token prefill chunks) by max_prefill. 0 when none left.
int Scheduler::ChunkSize(const Request* r, int budget) const {
  int remaining = r->Remaining();
  if (remaining <= 0) return 0;
  int n = std::min(remaining, budget);
  if (remaining > 1) n = std::min(n, cfg_.max_prefill);  // chunk long prefills
  return n;
}

// Grow r's KV to cover `upto` tokens, preempting younger sequences when the
// pool is dry: each preemption frees the youngest running seq other than r
// (returns its blocks, resets its cursor, requeues it at the front for
// recompute on resume). Returns false if r can't fit even after preempting
// everyone else: r is then failed (kOverCapacity to its sink) and retired, and
// the caller must rebuild the batch.
bool Scheduler::Grow(Request* r, int upto) {
  while (!kv_.Grow(r->Blocks(), upto)) {
    // youngest running seq other than r (scan from the end, skipping r)
    int k = static_cast<int>(running_.size()) - 1;
    while (k >= 0 && running_[k].get() == r) --k;
    if (k < 0) {  // nobody else to preempt: r alone can't fit -> fail it
      if (r->Sink())
        r->Sink()->Error(EngineErrc::kOverCapacity,
                         "out of KV cache: pool has " +
                             std::to_string(kv_.NumBlocks()) + " blocks of " +
                             std::to_string(kv_.BlockSize()) +
                             " tokens, cannot grow a sequence to " +
                             std::to_string(upto) + " tokens");
      Retire(r);
      return false;
    }
    stat_preempt_++;
    stat_recomp_tok_ += running_[k]->Computed();  // discarded cached KV
    if (stat_preempt_ == 1 || stat_preempt_ % 200 == 0)
      LogKvstat();  // catch preemption even in short runs
    Release(running_[k].get());
    running_[k]->ResetForRecompute();  // recompute from the start on resume
    waiting_.push_front(std::move(running_[k]));  // requeue at the front
    running_.erase(running_.begin() + k);
  }
  return true;
}

// One scheduling iteration over a step-local batch: cancel, admit, pack,
// forward, apply, reclaim.
void Scheduler::Step() {
  // 1. drop cancelled requests (client disconnected), freeing their KV
  for (auto it = waiting_.begin(); it != waiting_.end();) {
    if ((*it)->Sink() && (*it)->Sink()->Cancelled()) {
      Release(it->get());
      it = waiting_.erase(it);
    } else
      ++it;
  }
  for (auto it = running_.begin(); it != running_.end();) {
    if ((*it)->Sink() && (*it)->Sink()->Cancelled()) {
      Release(it->get());
      it = running_.erase(it);
    } else
      ++it;
  }
  // 2. admit waiters while slots are free, splicing in any cached prefix first
  //    (empty block table marks a fresh or just-preempted request — both may
  //    reuse KV)
  while (static_cast<int>(running_.size()) < cfg_.n_slots &&
         !waiting_.empty()) {
    if (waiting_.front()->Blocks().empty())
      AcquirePrefix(waiting_.front().get());
    running_.push_back(std::move(waiting_.front()));
    waiting_.pop_front();
  }
  // 3. pack the running set under the token budget, by ascending remaining
  // work:
  //    decodes (remaining==1) first (low TPOT), then shortest prefills. Sort an
  //    index view, never running_ itself — its order is request age, which
  //    Grow() uses to pick the youngest victim. A preempt mutates running_, so
  //    rebuild from scratch when it happens.
  CurrentBatch batch;
  std::vector<int> order;
  int step_prefill = 0, step_decode = 0;  // prefill/decode row split (instrum.)
  for (bool restart = true; restart;) {
    restart = false;
    batch.Clear();
    step_prefill = step_decode = 0;
    order.resize(running_.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return running_[a]->Remaining() < running_[b]->Remaining();
    });
    int budget = cfg_.max_batch_tokens;
    for (int idx : order) {
      if (budget <= 0) break;
      Request* r = running_[idx].get();
      int n = ChunkSize(r, budget);
      if (n == 0) continue;
      if (!Grow(r, r->Computed() + n)) {
        restart = true;
        break;
      }
      batch.AddRequest(r, n);
      (n > 1 ? step_prefill : step_decode) += n;  // n>1 prefill, n==1 decode
      budget -= n;
    }
  }
  if (batch.Empty()) return;
  // instrumentation: per-step counters + peak pool occupancy
  stat_steps_++;
  stat_prefill_rows_ += step_prefill;
  stat_decode_rows_ += step_decode;
  stat_fwd_rows_ += step_prefill + step_decode;
  int used = kv_.NumBlocks() - kv_.NumFree();
  if (used > stat_peak_used_) stat_peak_used_ = used;
  if (stat_steps_ % 200 == 0) LogKvstat();
  // 4. one merged forward (+ GPU sampling), apply, cache newly-full blocks,
  // reclaim
  model_.Forward(batch.input, batch.sampled);
  batch.Apply();
  if (cfg_.prefix_cache)
    for (Request* r : batch.requests)
      CacheFilled(r);  // finished reqs cached too (before retire)
  for (Request* r : batch.finished) Retire(r);
}

// Periodic cumulative instrumentation: pool occupancy + KV recomputed by
// preemption. recomp% = discarded-KV tokens / total forwarded rows (which
// includes the redo): fraction of GPU token-work spent rebuilding preempted KV.
void Scheduler::LogKvstat() {
  int nb = kv_.NumBlocks();
  double peak_pct = nb > 0 ? 100.0 * stat_peak_used_ / nb : 0.0;
  double recomp_pct = stat_fwd_rows_ > 0
                          ? 100.0 * static_cast<double>(stat_recomp_tok_) /
                                static_cast<double>(stat_fwd_rows_)
                          : 0.0;
  double hit_pct = stat_prompt_tok_ > 0
                       ? 100.0 * static_cast<double>(stat_cache_hit_tok_) /
                             static_cast<double>(stat_prompt_tok_)
                       : 0.0;
  LOG_INFO(
      "[kvstat] steps=%ld peak_used=%d/%d (%.1f%%) | rows: prefill=%ld "
      "decode=%ld | "
      "preempt=%ld recomp_tok=%ld (%.2f%% of fwd) | prefix_hit=%ld/%ld tok "
      "(%.1f%%)",
      stat_steps_, stat_peak_used_, nb, peak_pct, stat_prefill_rows_,
      stat_decode_rows_, stat_preempt_, stat_recomp_tok_, recomp_pct,
      stat_cache_hit_tok_, stat_prompt_tok_, hit_pct);
}

void Scheduler::Submit(std::unique_ptr<Request> r) {
  {
    std::lock_guard<std::mutex> lk(inbound_mu_);
    inbound_.push_back(std::move(r));
  }
  inbound_cv_.notify_one();
}

// The engine thread: drain submitted requests (rejecting queue overflows), then
// advance one scheduling iteration. Blocks for the process lifetime.
void Scheduler::Run() {
  std::deque<std::unique_ptr<Request>> incoming;
  while (true) {
    {
      std::unique_lock<std::mutex> lk(inbound_mu_);
      if (inbound_.empty() && !Busy())
        inbound_cv_.wait(lk, [&] { return !inbound_.empty(); });
      incoming.swap(inbound_);
    }
    while (!incoming.empty()) {
      auto r = std::move(incoming.front());
      incoming.pop_front();
      // admission control: reject when the waiting queue is full
      if (cfg_.max_queue > 0 &&
          static_cast<int>(waiting_.size()) >= cfg_.max_queue) {
        if (r->Sink())
          r->Sink()->Error(
              EngineErrc::kOverCapacity,
              "request queue full: " + std::to_string(waiting_.size()) +
                  " requests already waiting (limit " +
                  std::to_string(cfg_.max_queue) + "), engine running up to " +
                  std::to_string(cfg_.n_slots) +
                  " concurrent sequences; retry shortly");
        // r dropped here; the handler's sink ref still delivers the error
        // event.
      } else {
        waiting_.push_back(std::move(r));  // admitted into the scheduling queue
      }
    }
    if (Busy()) Step();
  }
}
