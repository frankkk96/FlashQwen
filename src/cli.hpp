// Command-line helpers: model architecture detection. (Usage/--help is handled by CLI11 in main.)
#pragma once
#include <string>

// Read architectures[0] from <dir>/config.json (empty string if unavailable).
std::string read_arch(const std::string& dir);

// Whether FlashQwen supports this architecture (dense Qwen3 only).
bool arch_supported(const std::string& arch);
