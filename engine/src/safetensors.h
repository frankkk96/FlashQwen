#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace fq {

// A view into an mmapped safetensors shard (points straight into the mapping).
struct TensorView {
  const uint8_t* data = nullptr;
  size_t nbytes = 0;
  std::vector<int64_t> shape;
  std::string dtype;
  int64_t Numel() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
};

// Loads HuggingFace .safetensors shards (mmapped, lazy-paged) and serves
// tensors by name. Shard layout: [8-byte LE header len N][N-byte JSON
// header][data].
class SafeTensors {
 public:
  void LoadDir(const std::string& dir);
  const TensorView& Get(const std::string& name) const;

 private:
  void MapShard(const std::string& path);

  std::unordered_map<std::string, TensorView> tensors_;
  std::vector<std::pair<void*, size_t>> mappings_;
};

}
