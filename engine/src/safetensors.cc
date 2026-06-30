#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "rapidjson/document.h"

namespace fq {

const TensorView& SafeTensors::Get(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end())
    throw std::runtime_error("tensor not found: " + name);
  return it->second;
}

void SafeTensors::MapShard(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open shard: " + path);
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    throw std::runtime_error("fstat failed: " + path);
  }
  size_t fsize = static_cast<size_t>(st.st_size);

  void* base = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (base == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
  mappings_.emplace_back(base, fsize);

  const uint8_t* bytes = static_cast<const uint8_t*>(base);
  uint64_t header_len;
  std::memcpy(&header_len, bytes,
              8);
  const char* json_begin = reinterpret_cast<const char*>(bytes + 8);
  const uint8_t* data_begin = bytes + 8 + header_len;

  rapidjson::Document root;
  root.Parse(json_begin, static_cast<size_t>(header_len));
  if (root.HasParseError() || !root.IsObject())
    throw std::runtime_error("invalid safetensors header: " + path);
  for (auto& kv : root.GetObject()) {
    std::string name = kv.name.GetString();
    if (name == "__metadata__") continue;
    const auto& meta = kv.value;
    TensorView tensor;
    tensor.dtype = meta["dtype"].GetString();
    for (auto& d : meta["shape"].GetArray()) tensor.shape.push_back(d.GetInt64());
    int64_t begin = meta["data_offsets"][0].GetInt64();
    int64_t end = meta["data_offsets"][1].GetInt64();
    tensor.data = data_begin + begin;
    tensor.nbytes = static_cast<size_t>(end - begin);
    tensors_[name] = tensor;
  }
}

void SafeTensors::LoadDir(const std::string& dir) {
  std::string index = dir + "/model.safetensors.index.json";
  std::ifstream f(index);
  std::set<std::string> shards;
  if (f) {
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();
    rapidjson::Document root;
    root.Parse(text.c_str());
    if (root.HasParseError() || !root.HasMember("weight_map"))
      throw std::runtime_error("invalid safetensors index: " + index);
    for (auto& kv : root["weight_map"].GetObject())
      shards.insert(kv.value.GetString());
  } else {
    shards.insert("model.safetensors");
  }
  for (auto& shard : shards) {
    std::string path = dir;
    path += '/';
    path += shard;
    MapShard(path);
  }
}

}
