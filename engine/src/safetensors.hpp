// Loader for HuggingFace .safetensors shards.
//
// File layout:  [8-byte little-endian header length N][N bytes of JSON header][raw tensor bytes]
// The JSON header maps tensor name -> {dtype, shape, data_offsets:[begin,end]} where the
// offsets are relative to the start of the data section (i.e. byte 8+N).
//
// We mmap every shard so the OS pages weights in on demand; get_raw() returns a pointer
// straight into the mapping plus the byte size. All Qwen3 weights are BF16.
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

struct TensorView {
    const uint8_t* data = nullptr;  // pointer into the mmapped shard
    size_t nbytes = 0;
    std::vector<int64_t> shape;
    std::string dtype;              // e.g. "BF16", "F32"
    int64_t numel() const { int64_t n=1; for (auto s: shape) n*=s; return n; }
};

class SafeTensors {
public:
    // Loads all shards referenced by <dir>/model.safetensors.index.json, or a single
    // model.safetensors if no index is present.
    void load_dir(const std::string& dir);

    const TensorView& get(const std::string& name) const;

private:
    void map_shard(const std::string& path);

    std::unordered_map<std::string, TensorView> tensors_;
    std::vector<std::pair<void*, size_t>> mappings_;  // (addr, length) for cleanup
};
