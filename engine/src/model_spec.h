#pragma once
#include <fstream>
#include <sstream>
#include <string>

#include "rapidjson/document.h"

// Declarative model spec (dims + arch) parsed from config.json; no GPU/weights.
// Drives KV-pool sizing, scheduler limits, and CLI validation. Load() is
// lenient (a missing/invalid file or unknown arch leaves it unsupported, dims
// default); gate use with Supported(). ModelRuntime is the compute half.
struct ModelSpec {
  int hidden_size = 4096;
  int num_layers = 36;
  int num_heads = 32;
  int num_kv_heads = 8;
  int head_dim = 128;
  int intermediate = 12288;
  int vocab_size = 151936;
  float rms_eps = 1e-6f;
  float rope_theta = 1000000.0f;
  std::string arch;
  std::string dir;

  int QDim() const { return num_heads * head_dim; }
  int KvDim() const { return num_kv_heads * head_dim; }
  bool Supported() const { return arch == "Qwen3ForCausalLM"; }

  static ModelSpec Load(const std::string& dir) {
    ModelSpec c;
    c.dir = dir;
    std::ifstream f(dir + "/config.json");
    if (!f) return c;
    std::stringstream ss;
    ss << f.rdbuf();
    rapidjson::Document o;
    o.Parse(ss.str().c_str());
    if (o.HasParseError() || !o.IsObject()) return c;
    if (o.HasMember("architectures") && o["architectures"].IsArray() &&
        !o["architectures"].Empty() && o["architectures"][0].IsString())
      c.arch = o["architectures"][0].GetString();
    if (!c.Supported()) return c;

    c.hidden_size = o["hidden_size"].GetInt();
    c.num_layers = o["num_hidden_layers"].GetInt();
    c.num_heads = o["num_attention_heads"].GetInt();
    c.num_kv_heads = o["num_key_value_heads"].GetInt();
    c.head_dim = o.HasMember("head_dim") ? o["head_dim"].GetInt()
                                         : c.hidden_size / c.num_heads;
    c.intermediate = o["intermediate_size"].GetInt();
    c.vocab_size = o["vocab_size"].GetInt();
    c.rms_eps = static_cast<float>(o["rms_norm_eps"].GetDouble());
    c.rope_theta = static_cast<float>(o["rope_theta"].GetDouble());
    return c;
  }
};
