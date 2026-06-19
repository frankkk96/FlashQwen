#include "engine_loop.hpp"
#include "errors.hpp"
#include <string>

void EngineLoop::run() {
    Scheduler sched(model, kv, n_slots, max_queue, max_batch_tokens, max_prefill, rng);
    std::deque<std::unique_ptr<Request>> incoming;
    while (true) {
        {
            std::unique_lock<std::mutex> lk(mu);
            if (inbound.empty() && !sched.busy())
                cv.wait(lk, [&] { return !inbound.empty(); });
            incoming.swap(inbound);
        }
        while (!incoming.empty()) {
            auto r = std::move(incoming.front()); incoming.pop_front();
            if (!sched.can_admit()) {   // admission control: queue full -> reject as over-capacity
                if (r->sink) r->sink->error(EngineErrc::OverCapacity,
                    "request queue full: " + std::to_string(sched.queue_depth()) +
                    " requests already waiting (limit " + std::to_string(sched.max_queue()) +
                    "), engine running up to " + std::to_string(n_slots) +
                    " concurrent sequences; retry shortly");
                // r is dropped here; the handler's sink ref still delivers the error event.
            } else {
                sched.add(std::move(r));
            }
        }
        if (sched.busy()) sched.step();
    }
}
