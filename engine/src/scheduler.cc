#include "scheduler.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>

#include "log.h"

Scheduler::Scheduler(ModelRuntime& model, BlockAllocator& alloc,
                     const SchedulerConfig& cfg)
    : model_(model), alloc_(alloc), cfg_(cfg) {}

namespace {
uint64_t HashBlock(uint64_t parent, const Request& r, int start, int n) {
  uint64_t h = parent ^ 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < n; ++i) {
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(r.TokenAt(start + i)));
    h *= 0x100000001B3ull;
    h = (h << 27) | (h >> 37);
  }
  return h ? h : 1;
}
}

void Scheduler::AcquirePrefix(Request* r) {
  stat_prompt_tok_ += r->NumTokens();
  if (!cfg_.use_prefix_cache) return;
  int bsz = alloc_.BlockSize();
  int limit_blocks = (r->NumTokens() - 1) / bsz;
  uint64_t parent = 0;
  auto& bt = r->Blocks();
  int nb = 0;
  for (; nb < limit_blocks; ++nb) {
    uint64_t h = HashBlock(parent, *r, nb * bsz, bsz);
    int b = alloc_.CacheLookup(h);
    if (b < 0) break;
    bt.push_back(b);
    parent = h;
  }
  if (nb > 0) {
    r->AdoptPrefix(nb, bsz, parent);
    stat_cache_hit_tok_ += static_cast<int64_t>(nb) * bsz;
    if (stat_cache_hit_tok_ ==
        static_cast<int64_t>(nb) *
            bsz)
      LOG_INFO(
          "[prefix] first cache hit: reused %d blocks (%d tokens) of a "
          "%d-token prompt",
          nb, nb * bsz, r->NumTokens());
  }
}

void Scheduler::CacheFilled(Request* r) {
  int bsz = alloc_.BlockSize();
  int full = r->Computed() / bsz;
  auto& bt = r->Blocks();
  uint64_t parent = r->LastHash();
  for (int idx = r->CachedBlocks(); idx < full; ++idx) {
    parent = HashBlock(parent, *r, idx * bsz, bsz);
    alloc_.CacheInsert(bt[idx], parent);
  }
  r->SetCachedBlocks(full);
  r->SetLastHash(parent);
}

void Scheduler::Release(Request* r) {
  alloc_.Free(r->Blocks());
}

void Scheduler::Retire(Request* r) {
  Release(r);
  for (auto it = running_.begin(); it != running_.end(); ++it)
    if (it->get() == r) {
      running_.erase(it);
      return;
    }
}

int Scheduler::ChunkSize(const Request* r, int budget) const {
  int remaining = r->Remaining();
  if (remaining <= 0) return 0;
  int n = std::min(remaining, budget);
  if (remaining > 1) n = std::min(n, cfg_.prefill_chunk);
  return n;
}

bool Scheduler::Grow(Request* r, int upto) {
  while (!alloc_.Grow(r->Blocks(), upto)) {
    int k = static_cast<int>(running_.size()) - 1;
    while (k >= 0 && running_[k].get() == r) --k;
    if (k < 0) {
      if (r->Sink())
        r->Sink()->Error(EngineErrc::kOverCapacity,
                         "out of KV cache: pool has " +
                             std::to_string(alloc_.NumBlocks()) + " blocks of " +
                             std::to_string(alloc_.BlockSize()) +
                             " tokens, cannot grow a sequence to " +
                             std::to_string(upto) + " tokens");
      Retire(r);
      return false;
    }
    stat_preempt_++;
    stat_recomp_tok_ += running_[k]->Computed();
    if (stat_preempt_ == 1 || stat_preempt_ % 200 == 0)
      LogKvstat();
    Release(running_[k].get());
    running_[k]->ResetForRecompute();
    waiting_.push_front(std::move(running_[k]));
    running_.erase(running_.begin() + k);
  }
  return true;
}

void Scheduler::Step() {
  auto drop_cancelled = [this](auto& queue) {
    for (auto it = queue.begin(); it != queue.end();) {
      if ((*it)->Sink() && (*it)->Sink()->Cancelled()) {
        Release(it->get());
        it = queue.erase(it);
      } else
        ++it;
    }
  };
  drop_cancelled(waiting_);
  drop_cancelled(running_);
  while (static_cast<int>(running_.size()) < cfg_.n_slots &&
         !waiting_.empty()) {
    if (waiting_.front()->Blocks().empty())
      AcquirePrefix(waiting_.front().get());
    running_.push_back(std::move(waiting_.front()));
    waiting_.pop_front();
  }
  CurrentBatch batch;
  std::vector<int> order;
  int step_prefill = 0, step_decode = 0;
  for (bool restart = true; restart;) {
    restart = false;
    batch.Clear();
    step_prefill = step_decode = 0;
    order.resize(running_.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return running_[a]->Remaining() < running_[b]->Remaining();
    });
    int budget = cfg_.token_budget;
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
      (n > 1 ? step_prefill : step_decode) += n;
      budget -= n;
    }
  }
  if (batch.Empty()) return;
  stat_steps_++;
  stat_prefill_rows_ += step_prefill;
  stat_decode_rows_ += step_decode;
  stat_fwd_rows_ += step_prefill + step_decode;
  int used = alloc_.NumBlocks() - alloc_.NumFree();
  if (used > stat_peak_used_) stat_peak_used_ = used;
  if (stat_steps_ % 200 == 0) LogKvstat();
  model_.Forward(batch.input, batch.sampled);
  batch.Apply();
  if (cfg_.use_prefix_cache)
    for (Request* r : batch.requests)
      CacheFilled(r);
  for (Request* r : batch.finished) Retire(r);
}

void Scheduler::LogKvstat() {
  int nb = alloc_.NumBlocks();
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
      if (cfg_.max_waiting > 0 &&
          static_cast<int>(waiting_.size()) >= cfg_.max_waiting) {
        if (r->Sink())
          r->Sink()->Error(
              EngineErrc::kOverCapacity,
              "request queue full: " + std::to_string(waiting_.size()) +
                  " requests already waiting (limit " +
                  std::to_string(cfg_.max_waiting) + "), engine running up to " +
                  std::to_string(cfg_.n_slots) +
                  " concurrent sequences; retry shortly");
      } else {
        waiting_.push_back(std::move(r));
      }
    }
    if (Busy()) Step();
  }
}
