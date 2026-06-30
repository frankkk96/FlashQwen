#include "model_spec.h"

#include <fstream>
#include <sstream>

#include "rapidjson/document.h"

namespace fq {

ModelSpec ModelSpec::Load(const std::string& dir) {
  ModelSpec spec;
  spec.dir = dir;
  std::ifstream f(dir + "/config.json");
  if (!f) return spec;
  std::stringstream ss;
  ss << f.rdbuf();
  rapidjson::Document doc;
  doc.Parse(ss.str().c_str());
  if (doc.HasParseError() || !doc.IsObject()) return spec;
  if (doc.HasMember("architectures") && doc["architectures"].IsArray() &&
      !doc["architectures"].Empty() && doc["architectures"][0].IsString())
    spec.arch = doc["architectures"][0].GetString();
  if (!spec.Supported()) return spec;

  spec.hidden_size = doc["hidden_size"].GetInt();
  spec.num_layers = doc["num_hidden_layers"].GetInt();
  spec.num_heads = doc["num_attention_heads"].GetInt();
  spec.num_kv_heads = doc["num_key_value_heads"].GetInt();
  spec.head_dim = doc.HasMember("head_dim") ? doc["head_dim"].GetInt()
                                            : spec.hidden_size / spec.num_heads;
  spec.intermediate = doc["intermediate_size"].GetInt();
  spec.vocab_size = doc["vocab_size"].GetInt();
  spec.rms_eps = static_cast<float>(doc["rms_norm_eps"].GetDouble());
  spec.rope_theta = static_cast<float>(doc["rope_theta"].GetDouble());
  return spec;
}

}
