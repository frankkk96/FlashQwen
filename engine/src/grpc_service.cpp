#include "grpc_service.hpp"
#include "grpc_sink.hpp"
#include "model_spec.hpp"
#include "model_runtime.hpp"
#include "block_pool.hpp"
#include "kv_cache.hpp"
#include "scheduler.hpp"
#include "log.hpp"
#include "engine.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <algorithm>
#include <memory>
#include <thread>
#include <exception>
#include <cstdlib>
#include <string>

using flashqwen::Engine;
using flashqwen::GenerateRequest;
using flashqwen::GenerateEvent;
using flashqwen::ModelRequest;
using flashqwen::ModelInfo;

// Generate-stream tuning.
constexpr int kDefaultMaxTokens = 512;   // used when a request omits max_tokens
constexpr int kStreamPollMs     = 100;   // how often the handler re-checks for client cancellation

// ---- gRPC service -----------------------------------------------------------------------
// The engine loads the model before this service is constructed, so it is always ready: GetModel
// answers immediately (which is how the client detects readiness) and Generate submits straight to
// the scheduler.
class ServiceImpl final : public Engine::Service {
public:
    ServiceImpl(Scheduler& sched, std::string model_id, int max_ctx, int vocab)
        : sched_(sched), model_id_(std::move(model_id)), max_ctx_(max_ctx), vocab_(vocab) {}

    grpc::Status GetModel(grpc::ServerContext*, const ModelRequest*, ModelInfo* out) override {
        out->set_id(model_id_);
        out->set_max_ctx(max_ctx_);
        out->set_vocab_size(vocab_);
        return grpc::Status::OK;
    }

    grpc::Status Generate(grpc::ServerContext* ctx, const GenerateRequest* req,
                          grpc::ServerWriter<GenerateEvent>* writer) override {
        auto sink = std::make_shared<GrpcSink>();
        std::vector<int> prompt(req->input_ids().begin(), req->input_ids().end());
        std::vector<int> stop_ids(req->stop_token_ids().begin(), req->stop_token_ids().end());
        int max_new = req->max_tokens() > 0 ? req->max_tokens() : kDefaultMaxTokens;
        int room = max_ctx_ - (int)prompt.size(); if (room < 1) room = 1;
        SampleParams sp{req->temperature(), req->top_p() > 0.f ? req->top_p() : 1.0f};
        auto r = std::make_unique<Request>(std::move(prompt), std::min(max_new, room), max_ctx_, sp,
                                           std::move(stop_ids), sink);
        sched_.submit(std::move(r));   // hand the request to the engine; we keep `sink` for the stream

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
    Scheduler& sched_;
    std::string model_id_;
    int max_ctx_, vocab_;
};

// Load the model, then bind + serve. Loading happens before the port opens, so the client simply
// retries GetModel until it answers (up to its own startup timeout); load progress + failures go to
// stderr, and a fatal load failure returns non-zero so the supervisor detects the process death.
int run_engine(const Args& a, const std::string& model_id) {
    try {
        ModelSpec spec = ModelSpec::load(a.model_dir);
        if (!spec.supported()) {
            LOG_ERROR("[engine] unsupported model at '%s': architecture '%s'. --model must point at a "
                      "dir with config.json + *.safetensors; the engine supports dense Qwen3 "
                      "(Qwen3ForCausalLM).",
                      a.model_dir.c_str(), spec.arch.empty() ? "unknown" : spec.arch.c_str());
            return 1;
        }
        LOG_INFO("[engine] loading model %s ...", model_id.c_str());
        ModelRuntime model(spec, RuntimeConfig{a.max_ctx, a.max_batch_tokens, a.seed});
        BlockPool pool(spec, a.max_ctx, a.gpu_mem_fraction);
        model.attach_pool(pool);
        KVCacheManager kv(pool);

        int n_slots = std::min(a.slots, model.max_batch());   // clamp to the runtime cap
        // Automatic prefix caching is on by default; FQ_PREFIX_CACHE=0 (or =false) disables it (A/B).
        const char* pc = std::getenv("FQ_PREFIX_CACHE");
        bool prefix_cache = !(pc && (std::string(pc) == "0" || std::string(pc) == "false"));
        SchedulerConfig scfg{
            n_slots,
            a.max_queue <= 0 ? 4 * n_slots : a.max_queue,
            a.max_batch_tokens,
            a.max_prefill_tokens > 0 ? a.max_prefill_tokens : a.max_batch_tokens,
            prefix_cache,
        };
        Scheduler sched(model, kv, scfg);

        ServiceImpl service(sched, model_id, model.max_ctx(), spec.vocab_size);
        grpc::ServerBuilder builder;
        builder.AddListeningPort(a.address, grpc::InsecureServerCredentials());
        builder.RegisterService(&service);
        builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
        std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
        if (!server) { LOG_ERROR("[engine] failed to bind %s", a.address.c_str()); return 1; }
        LOG_INFO("[engine] ready: %d slots, max_queue %d, max_batch_tokens %d, max_prefill %d, "
                 "prefix_cache %s, gRPC on %s", scfg.n_slots, scfg.max_queue, scfg.max_batch_tokens,
                 scfg.max_prefill, scfg.prefix_cache ? "on" : "off", a.address.c_str());

        std::thread engine([&] { sched.run(); });   // the engine thread; runs for the process lifetime
        engine.detach();
        server->Wait();
        return 0;
    } catch (const std::exception& e) {
        LOG_ERROR("[engine] model load failed: %s", e.what());
        return 1;
    }
}
