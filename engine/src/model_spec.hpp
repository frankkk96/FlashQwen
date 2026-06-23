// Declarative model spec (dims + arch) parsed from config.json. No GPU/weights;
// cheap to load. Used for KV-pool sizing, scheduler limits, and CLI validation.
// ModelRuntime is the compute half; tokenizer/chat-template live in the Go app.
#pragma once
#include <fstream>
#include <sstream>
#include <string>

#include "rapidjson/document.h"

struct ModelSpec {
  int hidden_size = 4096;
  int num_layers = 36;
  int num_heads = 32;    // query heads
  int num_kv_heads = 8;  // key/value heads (GQA)
  int head_dim = 128;
  int intermediate = 12288;
  int vocab_size = 151936;
  float rms_eps = 1e-6f;
  float rope_theta = 1000000.0f;
  std::string arch;  // architectures[0] from config.json ("" if unreadable)
  std::string dir;   // directory it was loaded from (config.json + weights +
                     // vocab live here)

  int QDim() const { return num_heads * head_dim; }
  int KvDim() const { return num_kv_heads * head_dim; }
  // Engine assumes dense Qwen3 layout (head_dim==128, GQA); reject anything
  // else.
  bool Supported() const { return arch == "Qwen3ForCausalLM"; }

  // Parse <dir>/config.json. Lenient: missing/invalid file or unknown arch
  // leaves `arch` empty (gate with Supported()); dims read only for a supported
  // arch, where a real Qwen3 config always provides them.
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
    if (!c.Supported())
      return c;  // unknown layout: skip dim reads (fields may be absent)

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
