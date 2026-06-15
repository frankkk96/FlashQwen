#include "grpc_server.hpp"
#include "scheduler.hpp"
#include "prompt.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "inference.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <deque>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>
#include <atomic>
#include <cstdio>

using flashqwen::Inference;
using flashqwen::GenerateRequest;
using flashqwen::GenerateEvent;
using flashqwen::ModelRequest;
using flashqwen::ModelInfo;

static const int EOS1 = 151645, EOS2 = 151643;        // <|im_end|>, <|endoftext|>
static const int TOOLCALL_OPEN = 151657, TOOLCALL_CLOSE = 151658;  // <tool_call>, </tool_call>

// ---- per-request server-side state ------------------------------------------------------
// Owned by a shared_ptr held jointly by the gRPC handler thread (writes to the stream) and the
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
    Tokenizer::Stream detok;
    int  prompt_tokens = 0;
    bool in_tool = false;        // currently inside a <tool_call> block
    std::string tool_buf;        // accumulated tool-call JSON text
    int  n_tools = 0;            // tool calls emitted so far
};

// ---- the engine: the single thread that touches the model / GPU -------------------------
struct Engine {
    Model& model; const Tokenizer& tok; std::mt19937& rng; int n_slots;
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

    void emit_tool(RequestCtx* c) {
        // tool_buf holds the JSON between <tool_call> and </tool_call>; parse {name, arguments}.
        rapidjson::Document d;
        d.Parse(c->tool_buf.c_str(), c->tool_buf.size());
        if (d.HasParseError() || !d.IsObject() || !d.HasMember("name") || !d["name"].IsString())
            return;
        std::string args = "{}";
        if (d.HasMember("arguments")) {
            rapidjson::StringBuffer sb; rapidjson::Writer<rapidjson::StringBuffer> w(sb);
            d["arguments"].Accept(w); args.assign(sb.GetString(), sb.GetSize());
        }
        GenerateEvent ev;
        auto* tc = ev.mutable_tool_call();
        tc->set_id("call_" + std::to_string(c->n_tools));
        tc->set_name(d["name"].GetString());
        tc->set_arguments_json(args);
        c->queue.push(std::move(ev));
        c->n_tools++;
    }

    void run() {
        Scheduler sched(model, n_slots, /*stop_on_eos=*/true, rng);
        std::unordered_map<Request*, std::shared_ptr<RequestCtx>> conns;

        auto on_token = [&](Request* r, int t) {
            auto it = conns.find(r); if (it == conns.end()) return;
            RequestCtx* c = it->second.get();
            if (t == EOS1 || t == EOS2) return;
            if (t == TOOLCALL_OPEN)  { c->in_tool = true; c->tool_buf.clear(); return; }
            if (t == TOOLCALL_CLOSE) { emit_tool(c); c->in_tool = false; return; }
            std::string piece = tok.is_special(t) ? tok.token_bytes(t) : tok.stream_decode(c->detok, t);
            if (c->in_tool) c->tool_buf += piece;
            else if (!piece.empty()) { GenerateEvent ev; ev.set_text_delta(piece); c->queue.push(std::move(ev)); }
        };
        auto on_finish = [&](Request* r) {
            auto it = conns.find(r); if (it == conns.end()) return;
            RequestCtx* c = it->second.get();
            bool eos = (r->cur == EOS1 || r->cur == EOS2);
            const char* reason = c->n_tools > 0 ? "tool_calls" : (eos ? "stop" : "length");
            int comp = (int)r->output.size() - (eos ? 1 : 0);
            GenerateEvent ev;
            auto* d = ev.mutable_done();
            d->set_finish_reason(reason);
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
                    Request* rp = &c->req; conns[rp] = c; sched.add(rp);
                }
                while (!cancel_q.empty()) {
                    Request* rp = cancel_q.front(); cancel_q.pop_front();
                    auto it = conns.find(rp);
                    if (it != conns.end()) { sched.remove(rp); it->second->queue.close(); conns.erase(it); }
                }
            }
            if (sched.busy()) sched.step(on_token, on_finish);
        }
    }
};

// ---- gRPC service -----------------------------------------------------------------------
class ServiceImpl final : public Inference::Service {
public:
    ServiceImpl(Engine& eng, std::string model_id) : eng_(eng), model_id_(std::move(model_id)) {}

    grpc::Status GetModel(grpc::ServerContext*, const ModelRequest*, ModelInfo* out) override {
        out->set_id(model_id_);
        return grpc::Status::OK;
    }

    grpc::Status Generate(grpc::ServerContext* ctx, const GenerateRequest* req,
                          grpc::ServerWriter<GenerateEvent>* writer) override {
        // Render the Qwen3 prompt and tokenise here (off the engine thread).
        std::vector<ChatMessage> msgs;
        for (const auto& m : req->messages()) {
            ChatMessage cm; cm.role = m.role(); cm.content = m.content(); cm.tool_call_id = m.tool_call_id();
            for (const auto& tc : m.tool_calls())
                cm.tool_calls.push_back({tc.id(), tc.name(), tc.arguments_json()});
            msgs.push_back(std::move(cm));
        }
        std::vector<ChatToolDef> tools;
        for (const auto& t : req->tools())
            tools.push_back({t.name(), t.description(), t.parameters_json()});

        std::string prompt = render_prompt(msgs, tools, req->enable_thinking());

        auto c = std::make_shared<RequestCtx>();
        c->req.prompt = eng_.tok.encode(prompt);
        c->prompt_tokens = (int)c->req.prompt.size();
        int max_new = req->max_tokens() > 0 ? req->max_tokens() : 512;
        int room = eng_.model.max_ctx() - c->prompt_tokens; if (room < 1) room = 1;
        c->req.max_new = std::min(max_new, room);
        c->req.sp = SampleParams{req->temperature(), req->top_p() > 0.f ? req->top_p() : 1.0f};

        Request* rp = &c->req;
        eng_.submit(c);   // hands a shared ref to the engine; we keep `c` for the stream

        while (true) {
            GenerateEvent ev;
            if (c->queue.pop(ev, 100)) {
                if (!writer->Write(ev)) { eng_.request_cancel(rp); break; }   // client write failed
                if (ev.event_case() == GenerateEvent::kDone) break;
            } else {
                if (ctx->IsCancelled()) { eng_.request_cancel(rp); break; }   // client went away
                if (c->queue.is_closed()) break;                             // cancelled by engine
            }
        }
        return grpc::Status::OK;
    }

private:
    Engine& eng_;
    std::string model_id_;
};

int run_grpc_server(Model& model, const Tokenizer& tok, const std::string& address,
                    int n_slots, const std::string& model_id, std::mt19937& rng) {
    if (n_slots > model.max_batch()) n_slots = model.max_batch();
    Engine engine{model, tok, rng, n_slots};
    std::thread engine_thread([&] { engine.run(); });
    engine_thread.detach();

    ServiceImpl service(engine, model_id);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (!server) { std::fprintf(stderr, "[server] failed to bind %s\n", address.c_str()); return 1; }
    std::fprintf(stderr, "[server] gRPC listening on %s (%d slots, model %s)\n",
                 address.c_str(), n_slots, model_id.c_str());
    server->Wait();
    return 0;
}
