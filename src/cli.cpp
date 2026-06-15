#include "cli.hpp"
#include "json.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>

// Read architectures[0] from <dir>/config.json (empty string if unavailable).
std::string read_arch(const std::string& dir) {
    std::ifstream f(dir + "/config.json");
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

void print_help() {
    std::printf(
"FlashQwen - minimal from-scratch C++/CUDA inference engine for Qwen3 (dense)\n\n"
"USAGE\n"
"  flashqwen [chat]      --model DIR [options]     interactive chat (default)\n"
"  flashqwen benchmark   --model DIR               measure TTFT / TPOT / tok/s\n"
"  flashqwen --help\n\n"
"REQUIRED\n"
"  --model DIR           a plain directory holding config.json, the *.safetensors\n"
"                        shards, model.safetensors.index.json, vocab.json and\n"
"                        merges.txt (e.g. a `git clone` of the HF model repo)\n\n"
"OPTIONS\n"
"  --max-ctx N           per-sequence KV / context length   (default 4096)\n"
"  --gpu-mem-fraction F  VRAM cap; the paged KV block pool gets whatever is left\n"
"                        under it (more blocks => more concurrency)  (default 0.9)\n"
"  --temperature T       0 = greedy            (chat only)   (default 0.0)\n"
"  --top-p P             nucleus sampling      (chat only)   (default 0.95)\n"
"  --seed S              RNG seed              (chat only)   (default 1234)\n"
"  --think               enable Qwen3 thinking (chat only)   (default off)\n\n"
"CHAT\n"
"  in-chat commands: /exit  /quit  /reset  /think on|off\n\n"
"BENCHMARK\n"
"  Runs a fixed self-contained sweep: (1) single-sequence latency over several input\n"
"  lengths, then (2) static-batch decode throughput over input length x batch size.\n"
"  Median of a few runs after a warmup. No benchmark-specific options.\n\n"
"SUPPORTED MODELS\n"
"  Any dense Qwen3 model (architecture Qwen3ForCausalLM): Qwen3-0.6B / 1.7B / 4B /\n"
"  8B / 14B / 32B. They share GQA + RoPE + RMSNorm + SwiGLU + QK-Norm; dims are read\n"
"  from config.json. (Tested: Qwen3-8B.)\n"
"  NOT supported: Qwen3.5 (hybrid linear-attention + multimodal), Qwen3 MoE\n"
"  variants (e.g. Qwen3-30B-A3B), and non-Qwen architectures.\n\n");
}
