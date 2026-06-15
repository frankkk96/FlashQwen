// The model's declarative spec: dimensions + architecture, parsed from config.json. No GPU, no
// weights — cheap to load and inspect. Shared by the KV cache (pool sizing), the scheduler
// (vocab / context limits), CLI validation, and the runtime. The GPU executor that actually runs
// the forward pass is ModelRuntime (model_runtime.*).
//
// Porting note: this is the declarative half of the model — declare your model's shape, the
// special-token ids (special_tokens.hpp), and how config.json maps to these fields. The compute
// half (kernels + forward) is ModelRuntime.
#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include "rapidjson/document.h"

struct ModelSpec {
    int hidden_size  = 4096;
    int num_layers   = 36;
    int num_heads    = 32;   // query heads
    int num_kv_heads = 8;    // key/value heads (GQA)
    int head_dim     = 128;
    int intermediate = 12288;
    int vocab_size   = 151936;
    float rms_eps    = 1e-6f;
    float rope_theta = 1000000.0f;
    int max_pos      = 40960;
    std::string arch;        // architectures[0] from config.json ("" if unreadable)
    std::string dir;         // directory it was loaded from (config.json + weights + vocab live here)

    int  q_dim()  const { return num_heads * head_dim; }
    int  kv_dim() const { return num_kv_heads * head_dim; }
    bool supported() const { return arch == "Qwen3ForCausalLM"; }   // dense Qwen3 only

    // Parse <dir>/config.json. Lenient about presence: a missing/invalid file or unsupported arch
    // leaves `arch` empty / unset (gate with supported()); dims are only read for a supported arch
    // (a real Qwen3 config always has them).
    static ModelSpec load(const std::string& dir) {
        ModelSpec c;
        c.dir = dir;
        std::ifstream f(dir + "/config.json");
        if (!f) return c;
        std::stringstream ss; ss << f.rdbuf();
        rapidjson::Document o;
        o.Parse(ss.str().c_str());
        if (o.HasParseError() || !o.IsObject()) return c;
        if (o.HasMember("architectures") && o["architectures"].IsArray() &&
            !o["architectures"].Empty() && o["architectures"][0].IsString())
            c.arch = o["architectures"][0].GetString();
        if (!c.supported()) return c;   // don't read dims for an unknown layout (fields may be absent)

        c.hidden_size  = o["hidden_size"].GetInt();
        c.num_layers   = o["num_hidden_layers"].GetInt();
        c.num_heads    = o["num_attention_heads"].GetInt();
        c.num_kv_heads = o["num_key_value_heads"].GetInt();
        c.head_dim     = o.HasMember("head_dim") ? o["head_dim"].GetInt()
                                                 : c.hidden_size / c.num_heads;
        c.intermediate = o["intermediate_size"].GetInt();
        c.vocab_size   = o["vocab_size"].GetInt();
        c.rms_eps      = (float)o["rms_norm_eps"].GetDouble();
        c.rope_theta   = (float)o["rope_theta"].GetDouble();
        if (o.HasMember("max_position_embeddings"))
            c.max_pos  = o["max_position_embeddings"].GetInt();
        return c;
    }
};
