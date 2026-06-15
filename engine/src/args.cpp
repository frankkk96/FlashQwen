#include "args.hpp"
#include "CLI11.hpp"
#include "rapidjson/document.h"
#include <cstdio>
#include <fstream>
#include <sstream>

// Read architectures[0] from <dir>/config.json (empty string if unavailable).
static std::string read_arch(const std::string& dir) {
    std::ifstream f(dir + "/config.json");
    if (!f) return "";
    std::stringstream ss; ss << f.rdbuf();
    std::string text = ss.str();
    rapidjson::Document root;
    root.Parse(text.c_str());
    if (!root.HasParseError() && root.HasMember("architectures") &&
        root["architectures"].IsArray() && !root["architectures"].Empty() &&
        root["architectures"][0].IsString())
        return root["architectures"][0].GetString();
    return "";
}

// Whether FlashQwen supports this architecture (dense Qwen3 only).
static bool arch_supported(const std::string& arch) { return arch == "Qwen3ForCausalLM"; }

int parse_args(int argc, char** argv, Args& out) {
    CLI::App app{"FlashQwen — minimal from-scratch C++/CUDA inference engine for Qwen3 (dense)"};
    app.footer(
        "SUPPORTED MODELS\n"
        "  Any dense Qwen3 model (architecture Qwen3ForCausalLM): Qwen3-0.6B / 1.7B / 4B /\n"
        "  8B / 14B / 32B. They share GQA + RoPE + RMSNorm + SwiGLU + QK-Norm; dims are read\n"
        "  from config.json. (Tested: Qwen3-8B.) NOT supported: Qwen3 MoE variants and\n"
        "  non-Qwen architectures.\n"
        "  In-chat commands: /exit  /quit  /reset  /think on|off");

    // Shared options (live on the root app; subcommands fall through to them).
    app.add_option("--model", out.model_dir,
                   "model directory: config.json + *.safetensors + index + vocab.json + merges.txt")
        ->required()->check(CLI::ExistingDirectory);
    app.add_option("--max-ctx", out.max_ctx, "KV / context length")->capture_default_str();
    app.add_option("--gpu-mem-fraction", out.gpu_mem_fraction,
                   "VRAM cap; the paged KV block pool gets whatever is left under it")
        ->capture_default_str()->check(CLI::Range(0.1, 1.0));
    app.add_option("--temperature", out.temp, "0 = greedy (chat only)")->capture_default_str();
    app.add_option("--top-p", out.top_p, "nucleus sampling cutoff (chat only)")->capture_default_str();
    app.add_option("--seed", out.seed, "RNG seed (chat only)")->capture_default_str();
    app.add_flag("--think", out.think, "enable Qwen3 thinking mode (chat only)");

    // Mode is selected by an optional subcommand; chat is the default.
    auto* chat  = app.add_subcommand("chat", "interactive chat (default)");
    auto* bench = app.add_subcommand("benchmark", "measure TTFT / TPOT / tok/s");
    auto* serve = app.add_subcommand("serve", "run the gRPC inference server (for the Go OpenAI gateway)");
    bench->alias("bench");
    serve->add_option("--address", out.address, "gRPC listen address (host:port)")->capture_default_str();
    serve->add_option("--slots", out.slots, "max concurrent sequences")->capture_default_str();
    chat->fallthrough();
    bench->fallthrough();
    serve->fallthrough();
    app.require_subcommand(0, 1);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);   // prints help/error; exit code 0 for --help, nonzero on error
    }

    out.mode = bench->parsed() ? Args::Mode::Benchmark
             : serve->parsed() ? Args::Mode::Serve
                               : Args::Mode::Chat;

    std::string arch = read_arch(out.model_dir);
    if (!arch_supported(arch)) {
        std::fprintf(stderr,
            "error: '%s' has no readable config.json / unsupported architecture '%s'.\n"
            "       --model must point at a plain dir with config.json + *.safetensors +\n"
            "       vocab.json + merges.txt; FlashQwen supports dense Qwen3 (Qwen3ForCausalLM).\n",
            out.model_dir.c_str(), arch.empty() ? "unknown" : arch.c_str());
        return 1;
    }
    return -1;   // parsed and validated; main continues
}
