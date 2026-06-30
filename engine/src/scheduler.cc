#include "scheduler.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <string>

namespace fq {

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
  if (!cfg_.use_prefix_cache) return;
  int block_size = alloc_.BlockSize();
  int limit_blocks = (r->NumTokens() - 1) / block_size;
  uint64_t parent = 0;
  auto& bt = r->Blocks();
  int n_blocks = 0;
  for (; n_blocks < limit_blocks; ++n_blocks) {
    uint64_t h = HashBlock(parent, *r, n_blocks * block_size, block_size);
    int block = alloc_.CacheLookup(h);
    if (block < 0) break;
    bt.push_back(block);
    parent = h;
  }
  if (n_blocks > 0) {
    r->AdoptPrefix(n_blocks, block_size, parent);
  }
}

void Scheduler::CacheBlocks(Request* r) {
  int block_size = alloc_.BlockSize();
  int full = r->Computed() / block_size;
  auto& bt = r->Blocks();
  uint64_t parent = r->LastHash();
  for (int idx = r->CachedBlocks(); idx < full; ++idx) {
    parent = HashBlock(parent, *r, idx * block_size, block_size);
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
    Release(running_[k].get());
    running_[k]->ResetForRecompute();
    waiting_.push_front(std::move(running_[k]));
    running_.erase(running_.begin() + k);
  }
  return true;
}

void Scheduler::Refill() {
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
}

bool Scheduler::BuildBatch(CurrentBatch& batch) {
  std::vector<int> order;
  for (bool restart = true; restart;) {
    restart = false;
    batch.Clear();
    order.resize(running_.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return running_[a]->Remaining() < running_[b]->Remaining();
    });
    int budget = cfg_.token_budget;
    for (int idx : order) {
      if (budget <= 0) break;
      Request* r = running_[idx].get();
      int remaining = r->Remaining();
      if (remaining <= 0) continue;
      int n = std::min(remaining, budget);
      if (remaining > 1) n = std::min(n, cfg_.prefill_chunk);
      if (!Grow(r, r->Computed() + n)) {
        restart = true;
        break;
      }
      batch.AddRequest(r, n);
      budget -= n;
    }
  }
  return !batch.Empty();
}

void Scheduler::Step() {
  Refill();
  CurrentBatch batch;
  if (!BuildBatch(batch)) return;
  model_.Forward(batch.input, batch.sampled);
  batch.Apply();
  if (cfg_.use_prefix_cache)
    for (Request* r : batch.requests) CacheBlocks(r);
  for (Request* r : batch.finished) Retire(r);
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

}
