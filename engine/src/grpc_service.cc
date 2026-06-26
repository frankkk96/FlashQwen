#include "grpc_service.h"

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <string>
#include <thread>

#include "block_allocator.h"
#include "engine.grpc.pb.h"
#include "grpc_sink.h"
#include "kv_store.h"
#include "log.h"
#include "model_runtime.h"
#include "model_spec.h"
#include "scheduler.h"

using flashqwen::Engine;
using flashqwen::GenerateEvent;
using flashqwen::GenerateRequest;
using flashqwen::ModelInfo;
using flashqwen::ModelRequest;

constexpr int kDefaultMaxTokens = 512;
constexpr int kStreamPollMs =
    100;

class ServiceImpl final : public Engine::Service {
 public:
  ServiceImpl(Scheduler& sched, std::string model_id, int max_ctx, int vocab)
      : sched_(sched),
        model_id_(std::move(model_id)),
        max_ctx_(max_ctx),
        vocab_(vocab) {}

  grpc::Status GetModel(grpc::ServerContext*, const ModelRequest*,
                        ModelInfo* out) override {
    out->set_id(model_id_);
    out->set_max_ctx(max_ctx_);
    out->set_vocab_size(vocab_);
    return grpc::Status::OK;
  }

  grpc::Status Generate(grpc::ServerContext* ctx, const GenerateRequest* req,
                        grpc::ServerWriter<GenerateEvent>* writer) override {
    auto sink = std::make_shared<GrpcSink>();
    std::vector<int> prompt(req->input_ids().begin(), req->input_ids().end());
    std::vector<int> stop_ids(req->stop_token_ids().begin(),
                              req->stop_token_ids().end());
    int max_new = req->max_tokens() > 0 ? req->max_tokens() : kDefaultMaxTokens;
    int room = max_ctx_ - static_cast<int>(prompt.size());
    if (room < 1) room = 1;
    SampleParams sp{req->temperature(),
                    req->top_p() > 0.f ? req->top_p() : 1.0f};
    auto r =
        std::make_unique<Request>(std::move(prompt), std::min(max_new, room),
                                  max_ctx_, sp, std::move(stop_ids), sink);
    sched_.Submit(std::move(r));

    while (true) {
      GenerateEvent ev;
      if (sink->queue.Pop(ev, kStreamPollMs)) {
        if (!writer->Write(ev)) {
          sink->cancel.store(true);
          break;
        }
        if (ev.event_case() == GenerateEvent::kDone ||
            ev.event_case() == GenerateEvent::kError)
          break;
      } else {
        if (ctx->IsCancelled()) {
          sink->cancel.store(true);
          break;
        }
        if (sink->queue.IsClosed()) break;
      }
    }
    return grpc::Status::OK;
  }

 private:
  Scheduler& sched_;
  std::string model_id_;
  int max_ctx_, vocab_;
};

int RunEngine(const Args& a, const std::string& model_id) {
  try {
    ModelSpec spec = ModelSpec::Load(a.model_dir);
    if (!spec.Supported()) {
      LOG_ERROR(
          "[engine] unsupported model at '%s': architecture '%s'. --model must "
          "point at a "
          "dir with config.json + *.safetensors; the engine supports "
          "Qwen3-8B "
          "(Qwen3ForCausalLM).",
          a.model_dir.c_str(),
          spec.arch.empty() ? "unknown" : spec.arch.c_str());
      return 1;
    }
    LOG_INFO("[engine] loading model %s ...", model_id.c_str());
    KvStore store(spec, a.max_ctx, a.gpu_mem_fraction);
    ModelRuntime model(spec, store, a.max_ctx, a.slots, a.token_budget, a.seed);

    SchedulerConfig scfg{
        a.slots,
        a.max_waiting > 0 ? a.max_waiting : 4 * a.slots,
        a.token_budget,
        a.prefill_chunk > 0 ? a.prefill_chunk : a.token_budget,
        a.use_prefix_cache,
    };

    BlockAllocator alloc(store.NumBlocks());
    Scheduler sched(model, alloc, scfg);

    ServiceImpl service(sched, model_id, a.max_ctx, spec.vocab_size);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(a.address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (!server) {
      LOG_ERROR("[engine] failed to bind %s", a.address.c_str());
      return 1;
    }
    LOG_INFO(
        "[engine] ready: %d slots, max_waiting %d, token_budget %d, "
        "prefill_chunk %d, "
        "use_prefix_cache %s, gRPC on %s",
        scfg.n_slots, scfg.max_waiting, scfg.token_budget, scfg.prefill_chunk,
        scfg.use_prefix_cache ? "on" : "off", a.address.c_str());

    std::thread engine(
        [&] { sched.Run(); });
    engine.detach();
    server->Wait();
    return 0;
  } catch (const std::exception& e) {
    LOG_ERROR("[engine] model load failed: %s", e.what());
    return 1;
  }
}
