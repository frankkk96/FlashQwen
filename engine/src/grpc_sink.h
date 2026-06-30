#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>

#include "engine.grpc.pb.h"
#include "errors.h"
#include "output_sink.h"

namespace fq {

inline flashqwen::ErrorCode ToProto(EngineErrc e) {
  switch (e) {
    case EngineErrc::kOverCapacity:
      return flashqwen::ERROR_CODE_OVER_CAPACITY;
    case EngineErrc::kInternal:
      return flashqwen::ERROR_CODE_INTERNAL;
  }
  return flashqwen::ERROR_CODE_UNSPECIFIED;
}

// Thread-safe event handoff: the scheduler thread pushes, the gRPC handler
// pops. Pop() waits up to timeout_ms and returns false on timeout or close.
class EventQueue {
 public:
  void Push(flashqwen::GenerateEvent e) {
    {
      std::lock_guard<std::mutex> lk(m_);
      q_.push_back(std::move(e));
    }
    cv_.notify_one();
  }
  void Close() {
    {
      std::lock_guard<std::mutex> lk(m_);
      closed_ = true;
    }
    cv_.notify_all();
  }
  bool Pop(flashqwen::GenerateEvent& out, int timeout_ms) {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                 [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
  }
  bool IsClosed() {
    std::lock_guard<std::mutex> lk(m_);
    return closed_ && q_.empty();
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<flashqwen::GenerateEvent> q_;
  bool closed_ = false;
};

// The OutputSink for one Generate stream: the scheduler pushes tokens / the
// terminal event into `queue`, the gRPC handler drains it onto the wire and
// sets `cancel` on client disconnect.
struct GrpcSink : OutputSink {
  EventQueue queue;
  std::atomic<bool> cancel{false};

  void Token(int id) override {
    flashqwen::GenerateEvent ev;
    ev.set_token_id(id);
    queue.Push(std::move(ev));
  }
  void Done(const std::string& finish_reason, int prompt_tokens,
            int completion_tokens) override {
    flashqwen::GenerateEvent ev;
    auto* d = ev.mutable_done();
    d->set_finish_reason(finish_reason);
    d->set_prompt_tokens(prompt_tokens);
    d->set_completion_tokens(completion_tokens);
    queue.Push(std::move(ev));
    queue.Close();
  }
  void Error(EngineErrc code, const std::string& msg) override {
    flashqwen::GenerateEvent ev;
    auto* e = ev.mutable_error();
    e->set_code(ToProto(code));
    e->set_message(msg);
    queue.Push(std::move(ev));
    queue.Close();
  }
  bool Cancelled() const override {
    return cancel.load(std::memory_order_relaxed);
  }
};

}
