#include "grpc_server.hpp"
#include "model_spec.hpp"
#include "model_runtime.hpp"
#include "kv_cache.hpp"
#include "scheduler.hpp"
#include "errors.hpp"
#include "startup.hpp"
#include "engine.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <algorithm>
#include <deque>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>
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

// ---- per-request server-side state ------------------------------------------------------
// Owned by a shared_ptr held jointly by the gRPC handler thread (writes the stream) and the
// engine thread (produces events); whichever finishes last frees it. The scheduler holds
// &RequestCtx::req, which stays valid as long as any shared_ptr does.
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

struct RequestCtx {
    Request req;
    EventQueue queue;
    int prompt_tokens = 0;
};

// ---- the engine: the single thread that touches the model / GPU -------------------------
struct EngineLoop {
    ModelRuntime& model; const KVCache& kv; std::mt19937& rng; int n_slots; int max_queue;
    std::mutex mu; std::condition_variable cv;
    std::deque<std::shared_ptr<RequestCtx>> inbound;
    std::deque<Request*> cancel_q;

    void submit(std::shared_ptr<RequestCtx> c) {
        { std::lock_guard<std::mutex> lk(mu); inbound.push_back(std::move(c)); }
        cv.notify_one();
    }
    void request_cancel(Request* r) {
        { std::lock_guard<std::mutex> lk(mu); cancel_q.push_back(r); }
        cv.notify_one();
    }

    void run() {
        Scheduler sched(model, kv, n_slots, max_queue, rng);
        std::unordered_map<Request*, std::shared_ptr<RequestCtx>> conns;

        // Terminate a request's stream with a structured error event (its blocks are already freed
        // by the scheduler). The detailed message rides through to the client unchanged.
        auto fail = [&](RequestCtx* c, EngineErrc code, const std::string& msg) {
            GenerateEvent ev;
            auto* e = ev.mutable_error();
            e->set_code(to_proto(code));
            e->set_message(msg);
            c->queue.push(std::move(ev));
            c->queue.close();
        };

        auto on_token = [&](Request* r, int t) {
            auto it = conns.find(r); if (it == conns.end()) return;
            GenerateEvent ev; ev.set_token_id(t);
            it->second->queue.push(std::move(ev));
        };
        auto on_error = [&](Request* r, EngineErrc code, std::string msg) {
            auto it = conns.find(r); if (it == conns.end()) return;
            fail(it->second.get(), code, msg);
            conns.erase(it);
        };
        auto on_finish = [&](Request* r) {
            auto it = conns.find(r); if (it == conns.end()) return;
            RequestCtx* c = it->second.get();
            bool stopped = !r->stop_ids.empty() &&
                std::find(r->stop_ids.begin(), r->stop_ids.end(), r->cur) != r->stop_ids.end();
            int comp = (int)r->output.size() - (stopped ? 1 : 0);  // exclude the stop token itself
            GenerateEvent ev;
            auto* d = ev.mutable_done();
            d->set_finish_reason(stopped ? "stop" : "length");
            d->set_prompt_tokens(c->prompt_tokens);
            d->set_completion_tokens(comp < 0 ? 0 : comp);
            c->queue.push(std::move(ev));
            c->queue.close();
            conns.erase(it);
        };

        while (true) {
            {
                std::unique_lock<std::mutex> lk(mu);
                if (inbound.empty() && cancel_q.empty() && !sched.busy())
                    cv.wait(lk, [&] { return !inbound.empty() || !cancel_q.empty(); });
                while (!inbound.empty()) {
                    auto c = std::move(inbound.front()); inbound.pop_front();
                    if (!sched.can_admit()) {   // admission control: queue full -> reject as over-capacity
                        fail(c.get(), EngineErrc::OverCapacity,
                             "request queue full: " + std::to_string(sched.queue_depth()) +
                             " requests already waiting (limit " + std::to_string(sched.max_queue()) +
                             "), engine running up to " + std::to_string(n_slots) +
                             " concurrent sequences; retry shortly");
                        continue;
                    }
                    Request* rp = &c->req; conns[rp] = c; sched.add(rp);
                }
                while (!cancel_q.empty()) {
                    Request* rp = cancel_q.front(); cancel_q.pop_front();
                    auto it = conns.find(rp);
                    if (it != conns.end()) { sched.remove(rp); it->second->queue.close(); conns.erase(it); }
                }
            }
            if (sched.busy()) sched.step(on_token, on_finish, on_error);
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

        auto c = std::make_shared<RequestCtx>();
        c->req.prompt.assign(req->input_ids().begin(), req->input_ids().end());
        c->req.stop_ids.assign(req->stop_token_ids().begin(), req->stop_token_ids().end());
        c->prompt_tokens = (int)c->req.prompt.size();
        int max_new = req->max_tokens() > 0 ? req->max_tokens() : 512;
        int room = max_ctx - c->prompt_tokens; if (room < 1) room = 1;
        c->req.max_new = std::min(max_new, room);
        c->req.sp = SampleParams{req->temperature(), req->top_p() > 0.f ? req->top_p() : 1.0f};

        Request* rp = &c->req;
        eng->submit(c);   // hands a shared ref to the engine; we keep `c` for the stream

        while (true) {
            GenerateEvent ev;
            if (c->queue.pop(ev, 100)) {
                if (!writer->Write(ev)) { eng->request_cancel(rp); break; }   // client write failed
                // Done and Error are both terminal; the client decodes Error into a typed failure.
                if (ev.event_case() == GenerateEvent::kDone ||
                    ev.event_case() == GenerateEvent::kError) break;
            } else {
                if (ctx->IsCancelled()) { eng->request_cancel(rp); break; }   // client went away
                if (c->queue.is_closed()) break;                             // cancelled by engine
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
            ModelRuntime model(spec, a.max_ctx, [&](int done, int total) { status.advance(done, total); });

            status.set_phase("allocating kv pool");
            KVCache kv(spec, a.max_ctx, a.gpu_mem_fraction);
            model.attach_kv(kv);

            int n_slots = a.slots;
            if (n_slots > model.max_batch()) n_slots = model.max_batch();
            int max_queue = a.max_queue <= 0 ? 4 * n_slots : a.max_queue;
            EngineLoop engine{model, kv, rng, n_slots, max_queue};
            service.set_ready(&engine, model_id, model.max_ctx(), spec.vocab_size);
            status.mark_ready();
            std::fprintf(stderr, "[engine] ready: %d slots, max_queue %d, model %s\n",
                         n_slots, max_queue, model_id.c_str());
            engine.run();   // becomes the engine thread; blocks for the process lifetime
        } catch (const std::exception& e) {
            status.mark_failed(std::string("model load failed: ") + e.what());
        }
    });
    loader.detach();

    server->Wait();
    return 0;
}
