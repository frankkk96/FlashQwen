#include "grpc_server.hpp"
#include "scheduler.hpp"
#include "errors.hpp"
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

using flashqwen::Engine;
using flashqwen::GenerateRequest;
using flashqwen::GenerateEvent;
using flashqwen::ModelRequest;
using flashqwen::ModelInfo;

// The single place that maps the engine's domain error taxonomy to the proto wire codes.
static flashqwen::ErrorCode to_proto(EngineErrc e) {
    switch (e) {
        case EngineErrc::OverCapacity: return flashqwen::ERROR_CODE_OVER_CAPACITY;
        case EngineErrc::Internal:     return flashqwen::ERROR_CODE_INTERNAL;
    }
    return flashqwen::ERROR_CODE_UNSPECIFIED;
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
class ServiceImpl final : public Engine::Service {
public:
    ServiceImpl(EngineLoop& eng, std::string model_id, int max_ctx, int vocab)
        : eng_(eng), model_id_(std::move(model_id)), max_ctx_(max_ctx), vocab_(vocab) {}

    grpc::Status GetModel(grpc::ServerContext*, const ModelRequest*, ModelInfo* out) override {
        out->set_id(model_id_);
        out->set_max_ctx(max_ctx_);
        out->set_vocab_size(vocab_);
        return grpc::Status::OK;
    }

    grpc::Status Generate(grpc::ServerContext* ctx, const GenerateRequest* req,
                          grpc::ServerWriter<GenerateEvent>* writer) override {
        auto c = std::make_shared<RequestCtx>();
        c->req.prompt.assign(req->input_ids().begin(), req->input_ids().end());
        c->req.stop_ids.assign(req->stop_token_ids().begin(), req->stop_token_ids().end());
        c->prompt_tokens = (int)c->req.prompt.size();
        int max_new = req->max_tokens() > 0 ? req->max_tokens() : 512;
        int room = max_ctx_ - c->prompt_tokens; if (room < 1) room = 1;
        c->req.max_new = std::min(max_new, room);
        c->req.sp = SampleParams{req->temperature(), req->top_p() > 0.f ? req->top_p() : 1.0f};

        Request* rp = &c->req;
        eng_.submit(c);   // hands a shared ref to the engine; we keep `c` for the stream

        while (true) {
            GenerateEvent ev;
            if (c->queue.pop(ev, 100)) {
                if (!writer->Write(ev)) { eng_.request_cancel(rp); break; }   // client write failed
                // Done and Error are both terminal; the client decodes Error into a typed failure.
                if (ev.event_case() == GenerateEvent::kDone ||
                    ev.event_case() == GenerateEvent::kError) break;
            } else {
                if (ctx->IsCancelled()) { eng_.request_cancel(rp); break; }   // client went away
                if (c->queue.is_closed()) break;                             // cancelled by engine
            }
        }
        return grpc::Status::OK;
    }

private:
    EngineLoop& eng_;
    std::string model_id_;
    int max_ctx_, vocab_;
};

int run_grpc_server(ModelRuntime& model, const KVCache& kv, const std::string& address,
                    int n_slots, int max_queue, const std::string& model_id, std::mt19937& rng) {
    if (n_slots > model.max_batch()) n_slots = model.max_batch();
    if (max_queue <= 0) max_queue = 4 * n_slots;   // default admission depth
    EngineLoop engine{model, kv, rng, n_slots, max_queue};
    std::thread engine_thread([&] { engine.run(); });
    engine_thread.detach();

    ServiceImpl service(engine, model_id, model.max_ctx(), model.spec().vocab_size);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (!server) { std::fprintf(stderr, "[engine] failed to bind %s\n", address.c_str()); return 1; }
    std::fprintf(stderr, "[engine] gRPC listening on %s (%d slots, max_queue %d, model %s)\n",
                 address.c_str(), n_slots, max_queue, model_id.c_str());
    server->Wait();
    return 0;
}
