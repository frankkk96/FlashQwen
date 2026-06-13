#include "safetensors.hpp"
#include "json.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <set>
#include <stdexcept>
#include <cstring>

const TensorView& SafeTensors::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
    return it->second;
}

void SafeTensors::map_shard(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open shard: " + path);
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); throw std::runtime_error("fstat failed: " + path); }
    size_t fsize = (size_t)st.st_size;

    void* base = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
    mappings_.emplace_back(base, fsize);

    const uint8_t* bytes = (const uint8_t*)base;
    uint64_t header_len;
    std::memcpy(&header_len, bytes, 8);                 // little-endian host assumed (x86)
    const char* json_begin = (const char*)(bytes + 8);
    const uint8_t* data_begin = bytes + 8 + header_len;

    auto root = minijson::parse(json_begin, header_len);
    for (auto& kv : root->obj) {
        const std::string& name = kv.first;
        if (name == "__metadata__") continue;
        const auto& meta = *kv.second;
        TensorView tv;
        tv.dtype = meta["dtype"].as_str();
        for (auto& s : meta["shape"].arr) tv.shape.push_back((int64_t)s->number);
        int64_t begin = (int64_t)meta["data_offsets"][0].number;
        int64_t end   = (int64_t)meta["data_offsets"][1].number;
        tv.data = data_begin + begin;
        tv.nbytes = (size_t)(end - begin);
        tensors_[name] = tv;
    }
}

void SafeTensors::load_dir(const std::string& dir) {
    std::string index = dir + "/model.safetensors.index.json";
    std::ifstream f(index);
    std::set<std::string> shards;
    if (f) {
        std::stringstream ss; ss << f.rdbuf();
        auto root = minijson::parse(ss.str());
        const auto& wm = (*root)["weight_map"];
        for (auto& kv : wm.obj) shards.insert(kv.second->as_str());
    } else {
        shards.insert("model.safetensors");
    }
    for (auto& shard : shards) map_shard(dir + "/" + shard);
}
