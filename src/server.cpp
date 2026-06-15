#include "server.hpp"
#include "scheduler.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <deque>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

static const int EOS1 = 151645, EOS2 = 151643;   // <|im_end|>, <|endoftext|>

// One client connection = one in-flight request.
struct Conn {
    int fd = -1;
    Request req;                  // scheduler request (prompt tokens + sampling)
    Tokenizer::Stream detok;      // streaming detokenizer state
    int  prompt_tokens = 0;
    bool alive = true;            // false once a socket write fails (client gone)
};

// Read one '\n'-terminated line from fd into `line` (without the newline). False on EOF/error.
static bool read_line(int fd, std::string& line) {
    line.clear();
    char buf[4096];
    while (true) {
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n <= 0) return false;
        for (ssize_t i = 0; i < n; ++i) {
            if (buf[i] == '\n') { line.append(buf, i); return true; }
        }
        line.append(buf, n);
        if (line.size() > (size_t)64 << 20) return false;   // 64 MB guard
    }
}

// Write all bytes; returns false if the client has gone away.
static bool write_all(int fd, const char* p, size_t n) {
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) return false;
        p += w; n -= (size_t)w;
    }
    return true;
}

// Serialize {"delta": <text>, "done": false}\n with proper JSON string escaping.
static std::string delta_line(const std::string& text) {
    rapidjson::StringBuffer sb;
    rapidjson::Writer<rapidjson::StringBuffer> w(sb);
    w.StartObject();
    w.Key("delta"); w.String(text.c_str(), (rapidjson::SizeType)text.size());
    w.Key("done");  w.Bool(false);
    w.EndObject();
    std::string s(sb.GetString(), sb.GetSize());
    s.push_back('\n');
    return s;
}

static std::string done_line(const char* reason, int prompt_tokens, int completion_tokens) {
    rapidjson::StringBuffer sb;
    rapidjson::Writer<rapidjson::StringBuffer> w(sb);
    w.StartObject();
    w.Key("done");              w.Bool(true);
    w.Key("finish_reason");     w.String(reason);
    w.Key("prompt_tokens");     w.Int(prompt_tokens);
    w.Key("completion_tokens"); w.Int(completion_tokens);
    w.EndObject();
    std::string s(sb.GetString(), sb.GetSize());
    s.push_back('\n');
    return s;
}

// Parse a request line into a Conn (prompt tokenised here, off the engine thread). Null on error.
static std::unique_ptr<Conn> parse_request(int fd, const std::string& line, const Tokenizer& tok,
                                           int max_ctx) {
    rapidjson::Document d;
    d.Parse(line.c_str(), line.size());
    if (d.HasParseError() || !d.IsObject() || !d.HasMember("prompt") || !d["prompt"].IsString())
        return nullptr;
    auto c = std::make_unique<Conn>();
    c->fd = fd;
    c->req.prompt = tok.encode(d["prompt"].GetString());
    int max_new = d.HasMember("max_tokens") && d["max_tokens"].IsInt() ? d["max_tokens"].GetInt() : 256;
    if (max_new < 1) max_new = 1;
    // leave at least one slot of context for the prompt; cap generation to what fits
    int room = max_ctx - (int)c->req.prompt.size();
    if (room < 1) room = 1;
    c->req.max_new = std::min(max_new, room);
    float temp  = d.HasMember("temperature") && d["temperature"].IsNumber() ? (float)d["temperature"].GetDouble() : 0.0f;
    float top_p = d.HasMember("top_p")       && d["top_p"].IsNumber()       ? (float)d["top_p"].GetDouble()       : 1.0f;
    c->req.sp = SampleParams{temp, top_p};
    c->prompt_tokens = (int)c->req.prompt.size();
    return c;
}

int run_server(Model& model, const Tokenizer& tok, const std::string& socket_path,
               int n_slots, std::mt19937& rng) {
    std::signal(SIGPIPE, SIG_IGN);   // a client vanishing must not kill the engine

    // --- listening Unix-domain socket -----------------------------------------------------
    int lfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (lfd < 0) { std::perror("socket"); return 1; }
    unlink(socket_path.c_str());
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (socket_path.size() >= sizeof(addr.sun_path)) {
        std::fprintf(stderr, "socket path too long: %s\n", socket_path.c_str()); return 1;
    }
    std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
    if (bind(lfd, (sockaddr*)&addr, sizeof(addr)) < 0) { std::perror("bind"); return 1; }
    if (listen(lfd, 64) < 0) { std::perror("listen"); return 1; }
    std::fprintf(stderr, "[server] listening on %s (%d slots)\n", socket_path.c_str(), n_slots);

    // --- inbound queue shared with the accept thread --------------------------------------
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::unique_ptr<Conn>> inbound;
    int max_ctx = model.max_ctx();

    std::thread acceptor([&] {
        while (true) {
            int cfd = accept(lfd, nullptr, nullptr);
            if (cfd < 0) { if (errno == EINTR) continue; break; }
            std::string line;
            if (!read_line(cfd, line)) { close(cfd); continue; }
            auto c = parse_request(cfd, line, tok, max_ctx);
            if (!c) { close(cfd); continue; }
            { std::lock_guard<std::mutex> lk(mu); inbound.push_back(std::move(c)); }
            cv.notify_one();
        }
    });
    acceptor.detach();

    // --- engine thread: the only thread that touches the model / GPU ----------------------
    Scheduler sched(model, n_slots, /*stop_on_eos=*/true, rng);
    std::unordered_map<Request*, std::unique_ptr<Conn>> conns;

    auto on_token = [&](Request* r, int t) {
        auto it = conns.find(r);
        if (it == conns.end()) return;
        Conn* c = it->second.get();
        if (!c->alive || t == EOS1 || t == EOS2) return;     // never stream the stop token
        // Emit other special tokens (e.g. <tool_call>, </tool_call>, <think>) as their literal
        // text so the gateway can see them — stream_decode drops specials, which would hide the
        // tool-call markers. Ordinary tokens go through the UTF-8-buffering streaming detokenizer.
        std::string piece = tok.is_special(t) ? tok.token_bytes(t) : tok.stream_decode(c->detok, t);
        if (piece.empty()) return;
        std::string line = delta_line(piece);
        if (!write_all(c->fd, line.data(), line.size())) c->alive = false;
    };
    auto on_finish = [&](Request* r) {
        auto it = conns.find(r);
        if (it == conns.end()) return;
        Conn* c = it->second.get();
        bool eos = (r->cur == EOS1 || r->cur == EOS2);
        int completion = (int)r->output.size() - (eos ? 1 : 0);
        if (c->alive) {
            std::string line = done_line(eos ? "stop" : "length", c->prompt_tokens, completion);
            write_all(c->fd, line.data(), line.size());
        }
        close(c->fd);
        conns.erase(it);
    };

    while (true) {
        {
            std::unique_lock<std::mutex> lk(mu);
            if (inbound.empty() && !sched.busy())
                cv.wait(lk, [&] { return !inbound.empty(); });
            while (!inbound.empty()) {
                auto c = std::move(inbound.front()); inbound.pop_front();
                Request* rp = &c->req;
                conns[rp] = std::move(c);
                sched.add(rp);
            }
        }
        if (sched.busy()) sched.step(on_token, on_finish);
    }
    return 0;
}
