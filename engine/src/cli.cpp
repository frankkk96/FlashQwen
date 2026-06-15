#include "cli.hpp"
#include "rapidjson/document.h"
#include <fstream>
#include <sstream>

// Read architectures[0] from <dir>/config.json (empty string if unavailable).
std::string read_arch(const std::string& dir) {
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

bool arch_supported(const std::string& arch) { return arch == "Qwen3ForCausalLM"; }
