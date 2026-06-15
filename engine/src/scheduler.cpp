#include "scheduler.hpp"
#include "special_tokens.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

static const int PREFILL_CHUNK = 256;            // tokens prefilled per scheduler iteration

Scheduler::Scheduler(Model& model, int n_slots, bool stop_on_eos, std::mt19937& rng)
    : model_(model), stop_on_eos_(stop_on_eos), rng_(rng) {
    n_slots_ = std::min(n_slots, model_.max_batch());
    max_ctx_ = model_.max_ctx();
    V_       = model_.config().vocab_size;
    bsz_     = model_.block_size();
    free_blocks_.reserve(model_.num_blocks());
    for (int b = model_.num_blocks() - 1; b >= 0; --b) free_blocks_.push_back(b);
}

void Scheduler::add(Request* r) { waiting_.push_back(r); }

void Scheduler::remove(Request* r) {
    for (auto it = waiting_.begin(); it != waiting_.end(); ++it)
        if (*it == r) { waiting_.erase(it); return; }
    if (pf_ == r) { release(r); pf_ = nullptr; return; }
    for (auto it = running_.begin(); it != running_.end(); ++it)
        if (*it == r) { release(r); running_.erase(it); return; }
}

bool Scheduler::finished(const Request* r) const {
    if ((int)r->output.size() >= r->max_new) return true;
    if (stop_on_eos_ && !r->output.empty() && special::is_eos(r->cur)) return true;
    if (r->past >= max_ctx_) return true;   // sequence full — no room for another token
    return false;
}

void Scheduler::release(Request* r) {
    for (int b : r->block_table) free_blocks_.push_back(b);
    r->block_table.clear();
}

// Free the youngest running sequence other than running_[protect] (or any if protect<0): return
// its blocks and requeue it so its KV is recomputed (from prompt+output) when later resumed.
bool Scheduler::preempt_one(int protect) {
    for (int k = (int)running_.size() - 1; k >= 0; --k) {
        if (k == protect) continue;
        release(running_[k]);
        waiting_.push_front(running_[k]);
        running_.erase(running_.begin() + k);
        return true;
    }
    return false;
}

void Scheduler::step(const TokenFn& on_token, const FinishFn& on_finish) {
    // (a) start prefilling the next waiting request if there is room for another sequence.
    if (pf_ == nullptr && !waiting_.empty() && (int)running_.size() < n_slots_) {
        pf_ = waiting_.front(); waiting_.pop_front();
        pf_->block_table.clear();
        pf_tokens_ = pf_->prompt;
        if (!pf_->output.empty())                  // resume: re-cache everything but the last token
            pf_tokens_.insert(pf_tokens_.end(), pf_->output.begin(), pf_->output.end() - 1);
        if ((int)pf_tokens_.size() > max_ctx_) pf_tokens_.resize(max_ctx_);
        pf_cursor_ = 0;
    }
    // (b) advance the in-progress prefill by one chunk.
    if (pf_ != nullptr) {
        Request* r = pf_;
        int chunk = std::min(PREFILL_CHUNK, (int)pf_tokens_.size() - pf_cursor_);
        int need = blocks_for(pf_cursor_ + chunk);
        while ((int)r->block_table.size() < need) {
            if (free_blocks_.empty() && !preempt_one(-1)) {
                std::fprintf(stderr, "scheduler: out of KV blocks during prefill\n"); std::exit(1);
            }
            if (free_blocks_.empty()) continue;
            r->block_table.push_back(free_blocks_.back()); free_blocks_.pop_back();
        }
        std::vector<int> piece(pf_tokens_.begin() + pf_cursor_, pf_tokens_.begin() + pf_cursor_ + chunk);
        model_.prefill(piece, r->block_table, pf_cursor_);
        pf_cursor_ += chunk;
        if (pf_cursor_ >= (int)pf_tokens_.size()) {        // prompt fully cached
            if (r->output.empty()) {                       // fresh: take the first generated token
                r->cur = (r->sp.temp <= 0.0f) ? model_.argmax_last()
                                              : sample(model_.copy_logits(), r->sp, rng_);
                r->output.push_back(r->cur);
                if (on_token) on_token(r, r->cur);
            } else {                                        // resume: continue from output.back()
                r->cur = r->output.back();
            }
            r->past = (int)pf_tokens_.size();
            if (finished(r)) { release(r); if (on_finish) on_finish(r); }
            else running_.push_back(r);
            pf_ = nullptr;
        }
    }
    if (running_.empty()) return;

    // (c) grow each running sequence's block table for the token it is about to write, preempting
    // a younger sequence (and rescanning) if the pool is exhausted.
    for (bool rescan = true; rescan; ) {
        rescan = false;
        for (size_t k = 0; k < running_.size(); ++k) {
            Request* r = running_[k];
            int need = blocks_for(r->past + 1);
            while ((int)r->block_table.size() < need) {
                if (free_blocks_.empty()) {
                    if (!preempt_one((int)k)) {
                        std::fprintf(stderr, "scheduler: out of KV blocks during decode\n"); std::exit(1);
                    }
                    rescan = true; break;
                }
                r->block_table.push_back(free_blocks_.back()); free_blocks_.pop_back();
            }
            if (rescan) break;
        }
    }

    // decode the whole running set in one step.
    in_tok_.clear(); past_.clear(); bts_.clear();
    bool any_sampling = false;
    for (Request* r : running_) { in_tok_.push_back(r->cur); past_.push_back(r->past);
                                  bts_.push_back(r->block_table);
                                  if (r->sp.temp > 0.0f) any_sampling = true; }
    if (any_sampling) {
        const float* L = model_.decode_logits_host(in_tok_, past_, bts_);
        out_.resize(running_.size());
        for (size_t k = 0; k < running_.size(); ++k)
            out_[k] = sample(L + (size_t)k * V_, V_, running_[k]->sp, rng_);
    } else {
        model_.decode(in_tok_, past_, bts_, out_);   // all greedy: GPU argmax, no full-logits copy
    }

    std::vector<Request*> still;
    for (size_t k = 0; k < running_.size(); ++k) {
        Request* r = running_[k];
        r->cur = out_[k]; r->past += 1; r->output.push_back(r->cur);
        if (on_token) on_token(r, r->cur);
        if (finished(r)) { release(r); if (on_finish) on_finish(r); }
        else still.push_back(r);
    }
    running_.swap(still);
}

void run_continuous(Model& model, std::vector<Request>& reqs, int n_slots, bool stop_on_eos,
                    std::mt19937& rng) {
    Scheduler sched(model, n_slots, stop_on_eos, rng);
    for (auto& r : reqs) sched.add(&r);
    while (sched.busy()) sched.step();
}
