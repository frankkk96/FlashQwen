// Wire-format output for one Generate stream. This is the only place that maps the engine's domain
// types onto the proto wire (`to_proto`) and frames results as GenerateEvents — so the proto headers
// stay confined to the transport layer (grpc_sink + grpc_service); the engine loop and scheduler
// never see them.
//
// GrpcSink is the engine's OutputSink for one stream: the scheduler writes tokens / the terminal
// event into the EventQueue, the gRPC handler thread drains it onto the wire. The Request (owned by
// the scheduler) and the handler each hold a shared_ptr to it, so it lives until both are done;
// cancellation is a flag the handler sets and the scheduler polls.
#pragma once
#include "errors.hpp"
#include "output_sink.hpp"
#include "engine.grpc.pb.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>

// Engine domain error -> proto wire code (the single mapping spot).
inline flashqwen::ErrorCode to_proto(EngineErrc e) {
    switch (e) {
        case EngineErrc::OverCapacity: return flashqwen::ERROR_CODE_OVER_CAPACITY;
        case EngineErrc::Internal:     return flashqwen::ERROR_CODE_INTERNAL;
    }
    return flashqwen::ERROR_CODE_UNSPECIFIED;
}

// Thread-safe queue of stream events: the engine thread pushes, the handler thread pops.
class EventQueue {
public:
    void push(flashqwen::GenerateEvent e) {
        { std::lock_guard<std::mutex> lk(m_); q_.push_back(std::move(e)); }
        cv_.notify_one();
    }
    void close() {
        { std::lock_guard<std::mutex> lk(m_); closed_ = true; }
        cv_.notify_all();
    }
    // Wait up to timeout_ms for an event. Returns true with `out` set, or false on timeout/closed.
    bool pop(flashqwen::GenerateEvent& out, int timeout_ms) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        out = std::move(q_.front()); q_.pop_front();
        return true;
    }
    bool is_closed() { std::lock_guard<std::mutex> lk(m_); return closed_ && q_.empty(); }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<flashqwen::GenerateEvent> q_;
    bool closed_ = false;
};

struct GrpcSink : OutputSink {
    EventQueue queue;
    std::atomic<bool> cancel{false};

    void token(int id) override {
        flashqwen::GenerateEvent ev; ev.set_token_id(id); queue.push(std::move(ev));
    }
    void done(const std::string& finish_reason, int prompt_tokens, int completion_tokens) override {
        flashqwen::GenerateEvent ev; auto* d = ev.mutable_done();
        d->set_finish_reason(finish_reason);
        d->set_prompt_tokens(prompt_tokens);
        d->set_completion_tokens(completion_tokens);
        queue.push(std::move(ev)); queue.close();
    }
    void error(EngineErrc code, const std::string& msg) override {
        flashqwen::GenerateEvent ev; auto* e = ev.mutable_error();
        e->set_code(to_proto(code)); e->set_message(msg);
        queue.push(std::move(ev)); queue.close();
    }
    bool cancelled() const override { return cancel.load(std::memory_order_relaxed); }
};
