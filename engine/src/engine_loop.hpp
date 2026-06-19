// The engine: the single thread that touches the model / GPU. Handler threads hand Requests in over
// `inbound`; the engine thread owns them from there on (via the Scheduler) and streams results
// straight to each request's sink. It is deliberately transport-agnostic — it knows Scheduler /
// Request / OutputSink, but nothing about gRPC or the proto wire (no callbacks, no per-request
// registry, no cancellation pointers crossing back).
#pragma once
#include "scheduler.hpp"
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <random>

struct EngineLoop {
    ModelRuntime& model; KVCacheManager& kv; std::mt19937& rng;
    int n_slots; int max_queue; int max_batch_tokens; int max_prefill;
    std::mutex mu; std::condition_variable cv;
    std::deque<std::unique_ptr<Request>> inbound;

    void submit(std::unique_ptr<Request> r) {
        { std::lock_guard<std::mutex> lk(mu); inbound.push_back(std::move(r)); }
        cv.notify_one();
    }

    void run();   // becomes the engine thread; drives the Scheduler, blocks for the process lifetime
};
