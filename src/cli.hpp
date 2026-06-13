// Command-line helpers: model-dir resolution, architecture check, and --help.
#pragma once
#include <string>

// Accept either an HF snapshot dir (containing config.json) or an HF hub cache dir
// (containing snapshots/<hash>/config.json); returns a dir with config.json.
std::string resolve_model_dir(const std::string& in);

// Read architectures[0] from config.json (empty string if unavailable).
std::string read_arch(const std::string& dir);

// Whether FlashQwen supports this architecture (dense Qwen3 only).
bool arch_supported(const std::string& arch);

// Print usage, supported models, and models detected in the local HF hub cache.
void print_help();
