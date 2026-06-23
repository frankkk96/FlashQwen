// Loads HuggingFace .safetensors shards.
//
// Shard layout: [8-byte LE header length N][N-byte JSON header][tensor bytes].
// Header maps name -> {dtype, shape, data_offsets:[begin,end]}, offsets
// relative to the data section start (byte 8+N). Shards are mmapped (lazy
// paging); a TensorView points straight into the mapping. Qwen3 weights are
// BF16.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct TensorView {
  const uint8_t* data = nullptr;  // pointer into the mmapped shard
  size_t nbytes = 0;
  std::vector<int64_t> shape;
  std::string dtype;  // e.g. "BF16", "F32"
  int64_t Numel() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
};

class SafeTensors {
 public:
  // Loads shards from model.safetensors.index.json, or a lone
  // model.safetensors.
  void LoadDir(const std::string& dir);

  const TensorView& Get(const std::string& name) const;

 private:
  void MapShard(const std::string& path);

  std::unordered_map<std::string, TensorView> tensors_;
  std::vector<std::pair<void*, size_t>> mappings_;  // (addr, len) for unmap
};
