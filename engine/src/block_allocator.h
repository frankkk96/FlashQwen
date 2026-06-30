#pragma once
#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

#include "kv_layout.h"

namespace fq {

// Bookkeeping half of the paged KV cache: a refcounted, vLLM-style free list
// with prefix-cache support over block ids [0, NumBlocks). Owns NO device
// memory — it hands out integer ids that index KvStore's tensors, and is the
// only thing the scheduler touches for KV (it never sees the free list). A
// refcount-0 block stays in the LRU queue with its content-hash mapping so an
// identical prefix can resurrect it (cache hit); allocation pops the LRU front
// and evicts that mapping. Grow() grows a request's block table to cover
// num_tokens, returning false the moment the pool runs dry (leaving partial
// growth) so the caller can preempt a sequence and retry. CUDA-free, so it
// builds and unit-tests without a GPU.
class BlockAllocator {
 public:
  explicit BlockAllocator(int num_blocks) : num_blocks_(num_blocks) {
    ref_cnt_.assign(num_blocks_, 0);
    blk_hash_.assign(num_blocks_, 0);
    pos_.assign(num_blocks_, free_lru_.end());
    for (int b = 0; b < num_blocks_; ++b) {
      free_lru_.push_back(b);
      pos_[b] = std::prev(free_lru_.end());
    }
  }

  int BlockSize() const { return kKvBlock; }
  int NumBlocks() const { return num_blocks_; }
  int NumFree() const { return static_cast<int>(free_lru_.size()); }
  int BlocksFor(int n_tok) const { return (n_tok + kKvBlock - 1) / kKvBlock; }

  bool AllocOne(int& out) {
    if (free_lru_.empty()) return false;
    int b = free_lru_.front();
    free_lru_.pop_front();
    pos_[b] = free_lru_.end();
    if (blk_hash_[b]) {
      auto it = cached_.find(blk_hash_[b]);
      if (it != cached_.end() && it->second == b) cached_.erase(it);
      blk_hash_[b] = 0;
    }
    ref_cnt_[b] = 1;
    out = b;
    return true;
  }
  void FreeOne(int b) {
    if (--ref_cnt_[b] == 0) {
      free_lru_.push_back(b);
      pos_[b] = std::prev(free_lru_.end());
    }
  }
  void FreeMany(const std::vector<int>& blocks) {
    for (int b : blocks) FreeOne(b);
  }

  bool Grow(std::vector<int>& block_table, int num_tokens) {
    int need = BlocksFor(num_tokens);
    while (static_cast<int>(block_table.size()) < need) {
      int b;
      if (!AllocOne(b)) return false;
      block_table.push_back(b);
    }
    return true;
  }
  void Free(std::vector<int>& block_table) {
    FreeMany(block_table);
    block_table.clear();
  }

  int CacheLookup(uint64_t hash) {
    auto it = cached_.find(hash);
    if (it == cached_.end()) return -1;
    int b = it->second;
    if (ref_cnt_[b] == 0) {
      free_lru_.erase(pos_[b]);
      pos_[b] = free_lru_.end();
    }
    ++ref_cnt_[b];
    return b;
  }
  void CacheInsert(int b, uint64_t hash) {
    if (blk_hash_[b]) return;
    if (cached_.count(hash)) return;
    blk_hash_[b] = hash;
    cached_[hash] = b;
  }

 private:
  std::vector<int> ref_cnt_;
  std::vector<uint64_t> blk_hash_;
  std::unordered_map<uint64_t, int> cached_;
  std::list<int> free_lru_;
  std::vector<std::list<int>::iterator> pos_;
  int num_blocks_ = 0;
};

}
