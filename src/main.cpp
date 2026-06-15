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
#include "grpc_server.hpp"
#include "CLI11.hpp"
#include <cstdio>
#include <string>
#include <random>

int main(int argc, char** argv) {
    std::string model_dir;
    int   max_ctx   = 4096;
    float temp      = 0.0f, top_p = 0.95f;
    unsigned seed   = 1234;
    bool  think     = false;
    float gpu_mem_fraction = 0.9f;   // VRAM cap; the KV pool gets whatever is left under it
    std::string address = "127.0.0.1:50051";   // serve mode: gRPC listen address
    int   slots     = 16;            // serve mode: max concurrent sequences

    CLI::App app{"FlashQwen — minimal from-scratch C++/CUDA inference engine for Qwen3 (dense)"};
    app.footer(
        "SUPPORTED MODELS\n"
        "  Any dense Qwen3 model (architecture Qwen3ForCausalLM): Qwen3-0.6B / 1.7B / 4B /\n"
        "  8B / 14B / 32B. They share GQA + RoPE + RMSNorm + SwiGLU + QK-Norm; dims are read\n"
        "  from config.json. (Tested: Qwen3-8B.) NOT supported: Qwen3 MoE variants and\n"
        "  non-Qwen architectures.\n"
        "  In-chat commands: /exit  /quit  /reset  /think on|off");

    // Shared options (live on the root app; subcommands fall through to them).
    app.add_option("--model", model_dir,
                   "model directory: config.json + *.safetensors + index + vocab.json + merges.txt")
        ->required()->check(CLI::ExistingDirectory);
    app.add_option("--max-ctx", max_ctx, "KV / context length")->capture_default_str();
    app.add_option("--gpu-mem-fraction", gpu_mem_fraction,
                   "VRAM cap; the paged KV block pool gets whatever is left under it")
        ->capture_default_str()->check(CLI::Range(0.1, 1.0));
    app.add_option("--temperature", temp, "0 = greedy (chat only)")->capture_default_str();
    app.add_option("--top-p", top_p, "nucleus sampling cutoff (chat only)")->capture_default_str();
    app.add_option("--seed", seed, "RNG seed (chat only)")->capture_default_str();
    app.add_flag("--think", think, "enable Qwen3 thinking mode (chat only)");

    // Mode is selected by an optional subcommand; chat is the default.
    auto* chat  = app.add_subcommand("chat", "interactive chat (default)");
    auto* bench = app.add_subcommand("benchmark", "measure TTFT / TPOT / tok/s");
    auto* serve = app.add_subcommand("serve", "run the gRPC inference server (for the Go OpenAI gateway)");
    bench->alias("bench");
    serve->add_option("--address", address, "gRPC listen address (host:port)")->capture_default_str();
    serve->add_option("--slots", slots, "max concurrent sequences")->capture_default_str();
    chat->fallthrough();
    bench->fallthrough();
    serve->fallthrough();
    app.require_subcommand(0, 1);

    CLI11_PARSE(app, argc, argv);

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
    model.load(model_dir, max_ctx, gpu_mem_fraction);

    SampleParams sp{temp, top_p};
    std::mt19937 rng(seed);

    if (bench->parsed())
        return run_benchmark(model, tok, max_ctx);
    if (serve->parsed()) {
        std::string id = model_dir;                       // model id = directory basename
        if (auto p = id.find_last_of('/'); p != std::string::npos) id = id.substr(p + 1);
        if (id.empty()) id = "flashqwen";
        return run_grpc_server(model, tok, address, slots, id, rng);
    }
    return run_chat(model, tok, sp, rng, think, max_ctx);
}
