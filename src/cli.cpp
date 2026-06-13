#include "cli.hpp"
#include "json.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <dirent.h>

static const char* DEFAULT_HUB = "/autodl-fs/data/huggingface/hub";

static bool file_exists(const std::string& p) {
    FILE* f = fopen(p.c_str(), "r");
    if (f) { fclose(f); return true; }
    return false;
}

std::string resolve_model_dir(const std::string& in) {
    if (file_exists(in + "/config.json")) return in;
    std::string snap = in + "/snapshots";
    DIR* d = opendir(snap.c_str());
    if (d) {
        struct dirent* e;
        while ((e = readdir(d)) != nullptr) {
            std::string name = e->d_name;
            if (name == "." || name == "..") continue;
            std::string cand = snap + "/" + name;
            if (file_exists(cand + "/config.json")) { closedir(d); return cand; }
        }
        closedir(d);
    }
    return in;
}

std::string read_arch(const std::string& dir) {
    std::string cfg = resolve_model_dir(dir) + "/config.json";
    std::ifstream f(cfg);
    if (!f) return "";
    std::stringstream ss; ss << f.rdbuf();
    try {
        auto root = minijson::parse(ss.str());
        if (root->contains("architectures") && !(*root)["architectures"].arr.empty())
            return (*root)["architectures"][0].as_str();
    } catch (...) {}
    return "";
}

bool arch_supported(const std::string& arch) { return arch == "Qwen3ForCausalLM"; }

// Scan the HF hub cache and print each model with a supported/unsupported tag.
static void list_local_models(const std::string& hub) {
    DIR* d = opendir(hub.c_str());
    if (!d) { std::printf("  (no hub cache found at %s)\n", hub.c_str()); return; }
    struct dirent* e;
    bool any = false;
    while ((e = readdir(d)) != nullptr) {
        std::string n = e->d_name;
        if (n.rfind("models--", 0) != 0) continue;
        std::string path = hub + "/" + n;
        std::string pretty = n.substr(8);
        for (size_t i = 0; (i = pretty.find("--", i)) != std::string::npos; ) pretty.replace(i, 2, "/");
        std::string arch = read_arch(path);
        std::printf("    %-22s  arch=%-32s  %s\n", pretty.c_str(), arch.c_str(),
                    arch_supported(arch) ? "[OK]" : "[unsupported]");
        std::printf("        --model %s\n", path.c_str());
        any = true;
    }
    closedir(d);
    if (!any) std::printf("  (no models found in %s)\n", hub.c_str());
}

void print_help() {
    std::printf(
"FlashQwen - minimal from-scratch C++/CUDA inference engine for Qwen3 (dense)\n\n"
"USAGE\n"
"  flashqwen [chat]      --model DIR [options]     interactive chat (default)\n"
"  flashqwen benchmark   --model DIR               measure TTFT / TPOT / tok/s\n"
"  flashqwen --help\n\n"
"REQUIRED\n"
"  --model DIR           path to a model (an HF snapshot dir with config.json, or an\n"
"                        HF hub cache dir like .../models--Qwen--Qwen3-8B)\n\n"
"OPTIONS\n"
"  --max-ctx N           KV-cache / context length          (default 4096)\n"
"  --temperature T       0 = greedy            (chat only)   (default 0.0)\n"
"  --top-p P             nucleus sampling      (chat only)   (default 0.95)\n"
"  --top-k K             top-k sampling        (chat only)   (default 20)\n"
"  --seed S              RNG seed              (chat only)   (default 1234)\n"
"  --think               enable Qwen3 thinking (chat only)   (default off)\n\n"
"CHAT\n"
"  in-chat commands: /exit  /quit  /reset  /think on|off\n\n"
"BENCHMARK\n"
"  Runs a fixed self-contained sweep (synthetic prompts of several input lengths,\n"
"  median of a few runs after a warmup). No benchmark-specific options.\n\n"
"SUPPORTED MODELS\n"
"  Any dense Qwen3 model (architecture Qwen3ForCausalLM): Qwen3-0.6B / 1.7B / 4B /\n"
"  8B / 14B / 32B. They share GQA + RoPE + RMSNorm + SwiGLU + QK-Norm; dims are read\n"
"  from config.json. (Tested: Qwen3-8B.)\n"
"  NOT supported: Qwen3.5 (hybrid linear-attention + multimodal), Qwen3 MoE\n"
"  variants (e.g. Qwen3-30B-A3B), and non-Qwen architectures.\n\n"
"MODELS DETECTED LOCALLY (%s)\n", DEFAULT_HUB);
    list_local_models(DEFAULT_HUB);
    std::printf("\n");
}
