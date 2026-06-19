#include "grpc_server.hpp"
#include "model_spec.hpp"
#include "model_runtime.hpp"
#include "block_pool.hpp"
#include "kv_cache.hpp"
#include "scheduler.hpp"
#include "errors.hpp"
#include "output_sink.hpp"
#include "startup.hpp"
#include "engine.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <cstdio>
#include <exception>

using flashqwen::Engine;
using flashqwen::GenerateRequest;
using flashqwen::GenerateEvent;
using flashqwen::ModelRequest;
using flashqwen::ModelInfo;
using flashqwen::StatusRequest;
using flashqwen::StatusResponse;

// The single place that maps the engine's domain error taxonomy to the proto wire codes.
static flashqwen::ErrorCode to_proto(EngineErrc e) {
    switch (e) {
        case EngineErrc::OverCapacity: return flashqwen::ERROR_CODE_OVER_CAPACITY;
        case EngineErrc::Internal:     return flashqwen::ERROR_CODE_INTERNAL;
    }
    return flashqwen::ERROR_CODE_UNSPECIFIED;
}

// Maps the startup lifecycle to the proto state enum.
static flashqwen::EngineState to_proto(LoadState s) {
    switch (s) {
        case LoadState::Loading: return flashqwen::ENGINE_STATE_LOADING;
        case LoadState::Ready:   return flashqwen::ENGINE_STATE_READY;
        case LoadState::Failed:  return flashqwen::ENGINE_STATE_FAILED;
    }
    return flashqwen::ENGINE_STATE_UNSPECIFIED;
}

// ---- per-request output sink ------------------------------------------------------------
// GrpcSink is the engine's OutputSink for one Generate stream: the scheduler writes tokens / the
// terminal event into the EventQueue, the handler thread drains it onto the wire. The Request (owned
// by the scheduler) and the handler each hold a shared_ptr to it, so it lives until both are done;
// cancellation is a flag the handler sets and the scheduler polls.
struct EventQueue {
    std::mutex m; std::condition_variable cv;
    std::deque<GenerateEvent> q; bool closed = false;
    void push(GenerateEvent e) { { std::lock_guard<std::mutex> lk(m); q.push_back(std::move(e)); } cv.notify_one(); }
    void close() { { std::lock_guard<std::mutex> lk(m); closed = true; } cv.notify_all(); }
    // Wait up to timeout_ms for an event. Returns true with `out` set, or false on timeout/closed.
    bool pop(GenerateEvent& out, int timeout_ms) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return !q.empty() || closed; });
        if (q.empty()) return false;
        out = std::move(q.front()); q.pop_front(); return true;
    }
    bool is_closed() { std::lock_guard<std::mutex> lk(m); return closed && q.empty(); }
};

struct GrpcSink : OutputSink {
    EventQueue queue;
    std::atomic<bool> cancel{false};

    void token(int id) override {
        GenerateEvent ev; ev.set_token_id(id); queue.push(std::move(ev));
    }
    void done(const std::string& finish_reason, int prompt_tokens, int completion_tokens) override {
        GenerateEvent ev; auto* d = ev.mutable_done();
        d->set_finish_reason(finish_reason);
        d->set_prompt_tokens(prompt_tokens);
        d->set_completion_tokens(completion_tokens);
        queue.push(std::move(ev)); queue.close();
    }
    void error(EngineErrc code, const std::string& msg) override {
        GenerateEvent ev; auto* e = ev.mutable_error();
        e->set_code(to_proto(code)); e->set_message(msg);
        queue.push(std::move(ev)); queue.close();
    }
    bool cancelled() const override { return cancel.load(std::memory_order_relaxed); }
};

// ---- the engine: the single thread that touches the model / GPU -------------------------
// The handler threads hand Requests in over `inbound`; the engine thread owns them from there on
// (via the scheduler) and streams results straight to each request's sink — no callbacks, no
// per-request registry, no cancellation pointers crossing back.
struct EngineLoop {
    ModelRuntime& model; KVCacheManager& kv; std::mt19937& rng;
    int n_slots; int max_queue; int max_batch_tokens; int max_prefill;
    std::mutex mu; std::condition_variable cv;
    std::deque<std::unique_ptr<Request>> inbound;

    void submit(std::unique_ptr<Request> r) {
        { std::lock_guard<std::mutex> lk(mu); inbound.push_back(std::move(r)); }
        cv.notify_one();
    }

    void run() {
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
};

// ---- gRPC service -----------------------------------------------------------------------
// GetStatus answers from the moment the port binds (driven by the loader thread through
// StartupStatus). GetModel/Generate are gated on readiness: until the loader calls set_ready they
// return UNAVAILABLE, so a client that connects mid-load gets a clear, retryable signal.
class ServiceImpl final : public Engine::Service {
public:
    explicit ServiceImpl(StartupStatus& status) : status_(status) {}

    // Published once by the loader thread when the model is resident and the engine loop is running.
    void set_ready(EngineLoop* eng, std::string model_id, int max_ctx, int vocab) {
        std::lock_guard<std::mutex> lk(rmu_);
        eng_ = eng; model_id_ = std::move(model_id); max_ctx_ = max_ctx; vocab_ = vocab;
    }

    grpc::Status GetStatus(grpc::ServerContext*, const StatusRequest*, StatusResponse* out) override {
        StartupStatus::Snapshot s = status_.snapshot();
        out->set_state(to_proto(s.state));
        out->set_phase(s.phase);
        out->set_done(s.done);
        out->set_total(s.total);
        out->set_message(s.message);
        return grpc::Status::OK;
    }

    grpc::Status GetModel(grpc::ServerContext*, const ModelRequest*, ModelInfo* out) override {
        std::lock_guard<std::mutex> lk(rmu_);
        if (!eng_) return grpc::Status(grpc::StatusCode::UNAVAILABLE, "engine is still loading the model");
        out->set_id(model_id_);
        out->set_max_ctx(max_ctx_);
        out->set_vocab_size(vocab_);
        return grpc::Status::OK;
    }

    grpc::Status Generate(grpc::ServerContext* ctx, const GenerateRequest* req,
                          grpc::ServerWriter<GenerateEvent>* writer) override {
        EngineLoop* eng; int max_ctx;
        { std::lock_guard<std::mutex> lk(rmu_); eng = eng_; max_ctx = max_ctx_; }
        if (!eng) return grpc::Status(grpc::StatusCode::UNAVAILABLE, "engine is still loading the model");

        auto sink = std::make_shared<GrpcSink>();
        auto r = std::make_unique<Request>();
        r->prompt.assign(req->input_ids().begin(), req->input_ids().end());
        r->stop_ids.assign(req->stop_token_ids().begin(), req->stop_token_ids().end());
        int max_new = req->max_tokens() > 0 ? req->max_tokens() : 512;
        int room = max_ctx - (int)r->prompt.size(); if (room < 1) room = 1;
        r->max_new = std::min(max_new, room);
        r->sp = SampleParams{req->temperature(), req->top_p() > 0.f ? req->top_p() : 1.0f};
        r->sink = sink;
        eng->submit(std::move(r));   // hand the request to the engine; we keep `sink` for the stream

        while (true) {
            GenerateEvent ev;
            if (sink->queue.pop(ev, 100)) {
                if (!writer->Write(ev)) { sink->cancel.store(true); break; }   // client write failed
                // Done and Error are both terminal; the client decodes Error into a typed failure.
                if (ev.event_case() == GenerateEvent::kDone ||
                    ev.event_case() == GenerateEvent::kError) break;
            } else {
                if (ctx->IsCancelled()) { sink->cancel.store(true); break; }   // client went away
                if (sink->queue.is_closed()) break;                           // closed by the engine
            }
        }
        return grpc::Status::OK;
    }

private:
    StartupStatus& status_;
    std::mutex rmu_;                 // guards the fields published by set_ready
    EngineLoop* eng_ = nullptr;
    std::string model_id_;
    int max_ctx_ = 0, vocab_ = 0;
};

// Bind the port first, then load the model on a background thread reporting progress through
// StartupStatus, so the Go client can poll GetStatus to drive a progress bar + stall watchdog.
// Catchable load failures become an ENGINE_STATE_FAILED status (the client reads the cause and tears
// the process down); hard failures (CUDA_CHECK -> exit, bad_alloc that escapes) kill the process and
// are surfaced by the supervisor's process-death detection instead.
int run_engine(const Args& a, const std::string& model_id, std::mt19937& rng) {
    StartupStatus status;
    ServiceImpl service(status);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(a.address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (!server) { std::fprintf(stderr, "[engine] failed to bind %s\n", a.address.c_str()); return 1; }
    std::fprintf(stderr, "[engine] gRPC listening on %s; loading model %s ...\n",
                 a.address.c_str(), model_id.c_str());

    std::thread loader([&] {
        try {
            ModelSpec spec = ModelSpec::load(a.model_dir);
            if (!spec.supported()) {
                status.mark_failed(
                    "unsupported model at '" + a.model_dir + "': architecture '" +
                    (spec.arch.empty() ? "unknown" : spec.arch) +
                    "'. --model must point at a dir with config.json + *.safetensors; the engine "
                    "supports dense Qwen3 (Qwen3ForCausalLM).");
                return;
            }
            status.set_phase("loading weights", spec.num_layers);
            ModelRuntime model(spec, a.max_ctx, a.max_batch_tokens,
                               [&](int done, int total) { status.advance(done, total); });

            status.set_phase("allocating kv pool");
            BlockPool pool(spec, a.max_ctx, a.gpu_mem_fraction);
            model.attach_pool(pool);
            KVCacheManager kv(pool);

            int n_slots = a.slots;
            if (n_slots > model.max_batch()) n_slots = model.max_batch();
            int max_queue = a.max_queue <= 0 ? 4 * n_slots : a.max_queue;
            int max_prefill = a.max_prefill_tokens > 0 ? a.max_prefill_tokens : a.max_batch_tokens;
            EngineLoop engine{model, kv, rng, n_slots, max_queue, a.max_batch_tokens, max_prefill};
            service.set_ready(&engine, model_id, model.max_ctx(), spec.vocab_size);
            status.mark_ready();
            std::fprintf(stderr, "[engine] ready: %d slots, max_queue %d, max_batch_tokens %d, "
                         "max_prefill %d, model %s\n",
                         n_slots, max_queue, a.max_batch_tokens, max_prefill, model_id.c_str());
            engine.run();   // becomes the engine thread; blocks for the process lifetime
        } catch (const std::exception& e) {
            status.mark_failed(std::string("model load failed: ") + e.what());
        }
    });
    loader.detach();

    server->Wait();
    return 0;
}
