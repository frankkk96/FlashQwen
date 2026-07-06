#include "model_spec.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "rapidjson/document.h"

namespace fq {

ModelSpec ModelSpec::Load(const std::string& dir) {
  ModelSpec spec;
  spec.dir = dir;
  std::ifstream f(dir + "/config.json");
  if (!f)
    throw std::runtime_error(
        "no config.json at '" + dir +
        "'; --model must point at a model directory (config.json + "
        "*.safetensors).");
  std::stringstream ss;
  ss << f.rdbuf();
  rapidjson::Document doc;
  doc.Parse(ss.str().c_str());
  if (doc.HasParseError() || !doc.IsObject())
    throw std::runtime_error("invalid config.json at '" + dir + "'.");

  std::string arch;
  if (doc.HasMember("architectures") && doc["architectures"].IsArray() &&
      !doc["architectures"].Empty() && doc["architectures"][0].IsString())
    arch = doc["architectures"][0].GetString();
  if (arch != kArch)
    throw std::runtime_error(
        "unsupported architecture '" +
        (arch.empty() ? std::string("unknown") : arch) + "' at '" + dir +
        "'; the engine supports Qwen3-8B (" + kArch + ").");

  std::string m;
  auto chk_i = [&](const char* key, int got, int want) {
    if (got != want)
      m += (m.empty() ? "" : ", ") + std::string(key) + "=" +
           std::to_string(got) + " (built for " + std::to_string(want) + ")";
  };
  auto chk_f = [&](const char* key, double got, double want) {
    if (std::abs(got - want) > 1e-4 * std::abs(want))
      m += (m.empty() ? "" : ", ") + std::string(key) + "=" +
           std::to_string(got) + " (built for " + std::to_string(want) + ")";
  };
  int cfg_head_dim = doc.HasMember("head_dim")
                         ? doc["head_dim"].GetInt()
                         : doc["hidden_size"].GetInt() /
                               doc["num_attention_heads"].GetInt();
  chk_i("head_dim", cfg_head_dim, kHeadDim);
  chk_i("num_attention_heads", doc["num_attention_heads"].GetInt(), kNumHeads);
  chk_i("num_key_value_heads", doc["num_key_value_heads"].GetInt(),
        kNumKvHeads);
  chk_i("hidden_size", doc["hidden_size"].GetInt(), kHiddenSize);
  chk_i("num_hidden_layers", doc["num_hidden_layers"].GetInt(), kNumLayers);
  chk_i("intermediate_size", doc["intermediate_size"].GetInt(), kIntermediate);
  chk_i("vocab_size", doc["vocab_size"].GetInt(), kVocabSize);
  chk_f("rms_norm_eps", doc["rms_norm_eps"].GetDouble(), kRmsEps);
  chk_f("rope_theta", doc["rope_theta"].GetDouble(), kRopeTheta);
  if (!m.empty())
    throw std::runtime_error("model at '" + dir +
                             "' does not match this build (" + m +
                             "); the engine is compiled for Qwen3-8B.");
  return spec;
}

}  // namespace fq
