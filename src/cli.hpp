// Command-line helpers: architecture check and --help.
#pragma once
#include <string>

// Read architectures[0] from <dir>/config.json (empty string if unavailable).
std::string read_arch(const std::string& dir);

// Whether FlashQwen supports this architecture (dense Qwen3 only).
bool arch_supported(const std::string& arch);

// Print usage, supported models, and models detected in the local HF hub cache.
void print_help();
