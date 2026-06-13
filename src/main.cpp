// FlashQwen entry point: parse arguments, load the model, dispatch to chat or benchmark.
//
//   ./flashqwen --model DIR                 # interactive chat (default)
//   ./flashqwen benchmark --model DIR       # benchmark (TTFT / TPOT / tok/s)
//   ./flashqwen --help                      # usage + supported models
//
#include "model.hpp"
#include "tokenizer.hpp"
#include "sampler.hpp"
#include "cli.hpp"
#include "chat.hpp"
#include "benchmark.hpp"
#include <cstdio>
#include <string>
#include <random>

int main(int argc, char** argv) {
    enum Mode { CHAT, BENCHMARK } mode = CHAT;
    std::string model_dir;
    int   max_ctx   = 4096;
    float temp      = 0.0f, top_p = 0.95f;
    int   top_k     = 20;
    unsigned seed   = 1234;
    bool  think     = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--help" || a == "-h") { print_help(); return 0; }
        else if (a == "chat")          mode = CHAT;
        else if (a == "benchmark" || a == "bench") mode = BENCHMARK;
        else if (a == "--model")       model_dir = next();
        else if (a == "--max-ctx")     max_ctx = std::stoi(next());
        else if (a == "--temperature") temp = std::stof(next());
        else if (a == "--top-p")       top_p = std::stof(next());
        else if (a == "--top-k")       top_k = std::stoi(next());
        else if (a == "--seed")        seed = (unsigned)std::stoul(next());
        else if (a == "--think")       think = true;
        else { std::fprintf(stderr, "unknown argument: %s  (try --help)\n", a.c_str()); return 1; }
    }

    if (model_dir.empty()) {
        std::fprintf(stderr, "error: --model is required.\n\n");
        print_help();
        return 1;
    }

    std::string arch = read_arch(model_dir);
    if (!arch_supported(arch)) {
        std::fprintf(stderr,
            "error: '%s' has no readable config.json / unsupported architecture '%s'.\n"
            "       --model must point at a plain dir with config.json + *.safetensors +\n"
            "       vocab.json + merges.txt; FlashQwen supports dense Qwen3 (Qwen3ForCausalLM).\n",
            model_dir.c_str(), arch.empty() ? "unknown" : arch.c_str());
        return 1;
    }

    Tokenizer tok;
    tok.load(model_dir);
    Model model;
    model.load(model_dir, max_ctx);

    SampleParams sp{temp, top_p, top_k};
    std::mt19937 rng(seed);

    if (mode == BENCHMARK)
        return run_benchmark(model, tok, max_ctx);
    return run_chat(model, tok, sp, rng, think, max_ctx);
}
