#include "grpc_service.hpp"
#include "grpc_sink.hpp"
#include "model_spec.hpp"
#include "model_runtime.hpp"
#include "block_pool.hpp"
#include "kv_cache.hpp"
#include "scheduler.hpp"
#include "startup.hpp"
#include "log.hpp"
#include "engine.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <algorithm>
#include <memory>
#include <mutex>
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

// Generate-stream tuning.
constexpr int kDefaultMaxTokens = 512;   // used when a request omits max_tokens
constexpr int kStreamPollMs     = 100;   // how often the handler re-checks for client cancellation

// ---- gRPC service -----------------------------------------------------------------------
// GetStatus answers from the moment the port binds (driven by the loader thread through
// StartupStatus). GetModel/Generate are gated on readiness: until the loader calls set_ready they
// return UNAVAILABLE, so a client that connects mid-load gets a clear, retryable signal.
class ServiceImpl final : public Engine::Service {
public:
    explicit ServiceImpl(StartupStatus& status) : status_(status) {}

    // Published once by the loader thread when the model is resident and the scheduler is running.
    void set_ready(Scheduler* sched, std::string model_id, int max_ctx, int vocab) {
        std::lock_guard<std::mutex> lk(rmu_);
        sched_ = sched; model_id_ = std::move(model_id); max_ctx_ = max_ctx; vocab_ = vocab;
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
        if (!sched_) return grpc::Status(grpc::StatusCode::UNAVAILABLE, "engine is still loading the model");
        out->set_id(model_id_);
        out->set_max_ctx(max_ctx_);
        out->set_vocab_size(vocab_);
        return grpc::Status::OK;
    }

    grpc::Status Generate(grpc::ServerContext* ctx, const GenerateRequest* req,
                          grpc::ServerWriter<GenerateEvent>* writer) override {
        Scheduler* sched; int max_ctx;
        { std::lock_guard<std::mutex> lk(rmu_); sched = sched_; max_ctx = max_ctx_; }
        if (!sched) return grpc::Status(grpc::StatusCode::UNAVAILABLE, "engine is still loading the model");

        auto sink = std::make_shared<GrpcSink>();
        auto r = std::make_unique<Request>();
        r->prompt.assign(req->input_ids().begin(), req->input_ids().end());
        r->stop_ids.assign(req->stop_token_ids().begin(), req->stop_token_ids().end());
        int max_new = req->max_tokens() > 0 ? req->max_tokens() : kDefaultMaxTokens;
        int room = max_ctx - (int)r->prompt.size(); if (room < 1) room = 1;
        r->max_new = std::min(max_new, room);
        r->sp = SampleParams{req->temperature(), req->top_p() > 0.f ? req->top_p() : 1.0f};
        r->sink = sink;
        sched->submit(std::move(r));   // hand the request to the engine; we keep `sink` for the stream

        while (true) {
            GenerateEvent ev;
            if (sink->queue.pop(ev, kStreamPollMs)) {
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
    Scheduler* sched_ = nullptr;
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
    if (!server) { LOG_ERROR("[engine] failed to bind %s", a.address.c_str()); return 1; }
    LOG_INFO("[engine] gRPC listening on %s; loading model %s ...", a.address.c_str(), model_id.c_str());

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
            Scheduler sched(model, kv, n_slots, max_queue, a.max_batch_tokens, max_prefill, rng);
            service.set_ready(&sched, model_id, model.max_ctx(), spec.vocab_size);
            status.mark_ready();
            LOG_INFO("[engine] ready: %d slots, max_queue %d, max_batch_tokens %d, max_prefill %d, "
                     "model %s", n_slots, max_queue, a.max_batch_tokens, max_prefill, model_id.c_str());
            sched.run();   // becomes the engine thread; blocks for the process lifetime
        } catch (const std::exception& e) {
            status.mark_failed(std::string("model load failed: ") + e.what());
        }
    });
    loader.detach();

    server->Wait();
    return 0;
}
