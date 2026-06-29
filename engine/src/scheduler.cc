#include "scheduler.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <string>

#include "log.h"

Scheduler::Scheduler(ModelRuntime& model, BlockAllocator& alloc,
                     const SchedulerConfig& cfg)
    : model_(model), alloc_(alloc), cfg_(cfg) {
  const char* e = std::getenv("FQ_ASYNC_SCHED");
  async_on_ = e && e[0] == '1';
}

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
  if (async_on_)
    StepAsync();
  else
    StepSync();
}

// Drop cancelled requests, then admit waiting -> running up to the slot cap.
void Scheduler::AdmitAndDrop() {
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

// Pack running requests under the token budget into one merged forward (decodes
// outrank prefill chunks). Returns false (and leaves batch empty) if nothing
// runs. Grows block tables on demand; a failed Grow preempts and restarts.
bool Scheduler::BuildBatch(CurrentBatch& batch, int& step_prefill,
                           int& step_decode) {
  std::vector<int> order;
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
  return !batch.Empty();
}

void Scheduler::StepSync() {
  AdmitAndDrop();
  CurrentBatch batch;
  int step_prefill = 0, step_decode = 0;
  if (!BuildBatch(batch, step_prefill, step_decode)) return;
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
    for (Request* r : batch.requests) CacheFilled(r);
  for (Request* r : batch.finished) Retire(r);
}

// Wait on the in-flight async batch, accept its sampled tokens (skipping
// requests an earlier step already finished — their token is wasted), and move
// newly-finished requests to retiring_ (kept alive + KV held until the next
// drain, since the just-launched following batch may still read their KV).
void Scheduler::ProcessPending() {
  model_.WaitForward(pending_ticket_, pending_batch_.sampled);
  CurrentBatch& b = pending_batch_;
  int s = 0;
  for (Request* r : b.requests) {
    if (!r->Done()) {
      if (r->Remaining() == 0 && r->AcceptSampled(b.sampled[s]))
        b.finished.push_back(r);
      if (cfg_.use_prefix_cache) CacheFilled(r);
    }
    ++s;
  }
  for (Request* r : b.finished)
    for (auto it = running_.begin(); it != running_.end(); ++it)
      if (it->get() == r) {
        retiring_.push_back(std::move(*it));
        running_.erase(it);
        break;
      }
}

// Single-step async: keep one forward in flight (pending_batch_) and process the
// previous one's output while it runs. The pipeline only continues across
// pure-decode steps whose request set is unchanged (so the previous step's
// GPU-resident sampled tokens map 1:1 to this step's rows and are fed back via
// embed feedback). Any set change (finish / admit / a prefill step) drains the
// pipeline and falls back to a synchronous step.
void Scheduler::StepAsync() {
  if (!pending_valid_) AdmitAndDrop();
  CurrentBatch cand;
  int pf = 0, dec = 0;
  bool built = BuildBatch(cand, pf, dec);

  if (pending_valid_ && built && pf == 0 &&
      cand.requests == pending_batch_.requests) {
    int t = model_.ForwardAsync(cand.input, /*feedback=*/true);
    for (Request* r : cand.requests) r->Advance();  // positions for next build
    ProcessPending();
    stat_steps_++;
    stat_decode_rows_ += dec;
    stat_fwd_rows_ += dec;
    pending_batch_ = std::move(cand);
    pending_ticket_ = t;
    pending_valid_ = true;
    return;
  }

  // Drain the in-flight batch, then free the requests retired by it (their
  // KV-reading batch has now completed).
  if (pending_valid_) {
    ProcessPending();
    pending_valid_ = false;
    for (auto& up : retiring_) Release(up.get());
    retiring_.clear();
  }
  AdmitAndDrop();
  CurrentBatch b;
  if (!BuildBatch(b, pf, dec)) return;
  stat_steps_++;
  stat_prefill_rows_ += pf;
  stat_decode_rows_ += dec;
  stat_fwd_rows_ += pf + dec;
  int used = alloc_.NumBlocks() - alloc_.NumFree();
  if (used > stat_peak_used_) stat_peak_used_ = used;
  if (stat_steps_ % 200 == 0) LogKvstat();
  if (pf == 0) {
    // Prime the pipeline: a pure-decode step whose input tokens are still
    // CPU-known (output_ is current after the drain), so no feedback yet.
    int t = model_.ForwardAsync(b.input, /*feedback=*/false);
    for (Request* r : b.requests) r->Advance();
    pending_batch_ = std::move(b);
    pending_ticket_ = t;
    pending_valid_ = true;
    return;
  }
  model_.Forward(b.input, b.sampled);
  b.Apply();
  if (cfg_.use_prefix_cache)
    for (Request* r : b.requests) CacheFilled(r);
  for (Request* r : b.finished) Retire(r);
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
