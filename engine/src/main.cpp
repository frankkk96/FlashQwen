// FlashQwen entry point: parse arguments (args.*), load the model + KV pool, dispatch to a mode.
//
//   ./flashqwen --model DIR                 # interactive chat (default)
//   ./flashqwen benchmark --model DIR       # benchmark (TTFT / TPOT / tok/s)
//   ./flashqwen serve --model DIR           # gRPC inference server (for the Go OpenAI gateway)
//   ./flashqwen --help                      # usage + supported models
//
#include "model.hpp"
#include "tokenizer.hpp"
#include "kv_cache.hpp"
#include "sampler.hpp"
#include "args.hpp"
#include "chat.hpp"
#include "benchmark.hpp"
#include "grpc_server.hpp"
#include <string>
#include <random>

int main(int argc, char** argv) {
    Args a;
    if (int rc = parse_args(argc, argv, a); rc >= 0) return rc;   // --help / parse error / bad model

    Tokenizer tok;
    tok.load(a.model_dir);
    Model model;
    model.load(a.model_dir, a.max_ctx);
    // The paged KV pool takes whatever VRAM is left under the cap, so build it after the model's
    // weights + activations are resident, then hand it to the model's attention kernels.
    KVCache kv(model.config(), a.max_ctx, a.gpu_mem_fraction);
    model.attach_kv(kv);

    std::mt19937 rng(a.seed);

    switch (a.mode) {
        case Args::Mode::Benchmark:
            return run_benchmark(model, kv, tok, a.max_ctx);
        case Args::Mode::Serve: {
            std::string id = a.model_dir;                     // model id = directory basename
            if (auto p = id.find_last_of('/'); p != std::string::npos) id = id.substr(p + 1);
            if (id.empty()) id = "flashqwen";
            return run_grpc_server(model, kv, tok, a.address, a.slots, id, rng);
        }
        case Args::Mode::Chat:
            return run_chat(model, kv, tok, SampleParams{a.temp, a.top_p}, rng, a.think, a.max_ctx);
    }
    return 0;
}
