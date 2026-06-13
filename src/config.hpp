// Model hyper-parameters, parsed from config.json.
#pragma once
#include <string>
#include "json.hpp"
#include <fstream>
#include <sstream>

struct ModelConfig {
    int hidden_size      = 4096;
    int num_layers       = 36;
    int num_heads        = 32;   // query heads
    int num_kv_heads     = 8;    // key/value heads (GQA)
    int head_dim         = 128;
    int intermediate     = 12288;
    int vocab_size       = 151936;
    float rms_eps        = 1e-6f;
    float rope_theta     = 1000000.0f;
    int max_pos          = 40960;

    // Derived
    int q_dim()  const { return num_heads * head_dim; }
    int kv_dim() const { return num_kv_heads * head_dim; }

    static ModelConfig load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot open config.json: " + path);
        std::stringstream ss; ss << f.rdbuf();
        auto root = minijson::parse(ss.str());
        const auto& o = *root;
        ModelConfig c;
        c.hidden_size  = o["hidden_size"].as_int();
        c.num_layers   = o["num_hidden_layers"].as_int();
        c.num_heads    = o["num_attention_heads"].as_int();
        c.num_kv_heads = o["num_key_value_heads"].as_int();
        c.head_dim     = o.contains("head_dim") ? o["head_dim"].as_int()
                                                 : c.hidden_size / c.num_heads;
        c.intermediate = o["intermediate_size"].as_int();
        c.vocab_size   = o["vocab_size"].as_int();
        c.rms_eps      = (float)o["rms_norm_eps"].as_double();
        c.rope_theta   = (float)o["rope_theta"].as_double();
        if (o.contains("max_position_embeddings"))
            c.max_pos  = o["max_position_embeddings"].as_int();
        return c;
    }
};
