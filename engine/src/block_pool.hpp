// Paged KV block pool — lowest layer of the KV stack.
//   STORAGE: flat per-layer pool [NumBlocks, BLOCK, kv_dim] BF16, shared by all
//            sequences; attention kernels read/write it directly via K(l)/V(l).
//   BOOKKEEPING: free list of physical block ids handed out / returned on
//   demand.
// KVCacheManager wraps this with a request-level API; the two meet only through
// block tables and the raw K/V pointers. Sized from leftover VRAM, so construct
// AFTER weights + activation scratch are uploaded.
#pragma once
#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

#include "kernels.cuh"  // bf16
#include "model_spec.hpp"

class BlockPool {
 public:
  static constexpr int BLOCK = 16;  // tokens per block (page)

  // Carve the pool from VRAM left under gpu_mem_fraction; seed the free list
  // with every block id. Throws if there isn't room for one full max_ctx seq.
  BlockPool(const ModelSpec& spec, int max_ctx, float gpu_mem_fraction);
  ~BlockPool();
  BlockPool(const BlockPool&) = delete;
  BlockPool& operator=(const BlockPool&) = delete;

  // Storage: per-layer base pointers, each [NumBlocks, BLOCK, kv_dim] BF16.
  bf16* K(int layer) const { return k_[layer]; }
  bf16* V(int layer) const { return v_[layer]; }

  int BlockSize() const { return BLOCK; }
  int NumBlocks() const { return num_blocks_; }

  // Block bookkeeping: refcounted + prefix-cache-aware (vLLM-style). Each block
  // has a refcount and optionally the content hash of the KV it holds. A
  // refcount-0 block is reclaimable but not erased: it waits in the LRU free
  // queue keeping its hash->id mapping, so an identical prefix can resurrect it
  // (cache hit). Allocation pops the LRU front, evicting that mapping if set.

  // Reclaimable (refcount-0) blocks; scheduler preempts only when this hits 0.
  int NumFree() const { return (int)free_lru_.size(); }

  // Fresh block (refcount 1); false (out untouched) if none free.
  bool AllocOne(int& out) {
    if (free_lru_.empty()) return false;
    int b = free_lru_.front();
    free_lru_.pop_front();
    pos_[b] = free_lru_.end();
    if (blk_hash_[b]) {  // was cached-but-free: drop its content mapping
      auto it = cached_.find(blk_hash_[b]);
      if (it != cached_.end() && it->second == b) cached_.erase(it);
      blk_hash_[b] = 0;
    }
    ref_cnt_[b] = 1;
    out = b;
    return true;
  }
  void FreeOne(int b) {  // refcount-1; at 0 returns to LRU tail (mapping kept)
    if (--ref_cnt_[b] == 0) {
      free_lru_.push_back(b);
      pos_[b] = std::prev(free_lru_.end());
    }
  }
  void FreeMany(const std::vector<int>& blocks) {
    for (int b : blocks) FreeOne(b);
  }

  // Block id holding `hash` (resurrecting + reffing it), else -1. Pure lookup
  // over resident blocks — never allocates, never fails on pool pressure.
  int CacheLookup(uint64_t hash) {
    auto it = cached_.find(hash);
    if (it == cached_.end()) return -1;
    int b = it->second;
    if (ref_cnt_[b] == 0) {
      free_lru_.erase(pos_[b]);
      pos_[b] = free_lru_.end();
    }  // resurrect
    ++ref_cnt_[b];
    return b;
  }
  // Register held block `b` (now full + fully computed) as the entry for
  // `hash`. Idempotent; no-ops on collision so the hash<->id mapping stays 1:1.
  void CacheInsert(int b, uint64_t hash) {
    if (blk_hash_[b]) return;
    if (cached_.count(hash)) return;
    blk_hash_[b] = hash;
    cached_[hash] = b;
  }

 private:
  int num_blocks_ = 0;
  int max_blocks_ = 0;
  std::vector<bf16*> k_, v_;

  std::vector<int> ref_cnt_;  // per-block refcount (0 => reclaimable)
  std::vector<uint64_t>
      blk_hash_;  // content hash held by each block (0 => none)
  std::unordered_map<uint64_t, int> cached_;  // hash -> block id (1:1)
  std::list<int> free_lru_;  // reclaimable ids, LRU (front = evict first)
  std::vector<std::list<int>::iterator>
      pos_;  // each block's node (end() if out)
};

// Pool write side: scatter M freshly-projected K (or V) rows into `cache` (one
// layer's K(l)/V(l)). Token m's source row is src + m*src_stride + src_offset
// (reads straight from a fused QKV buffer); destination is block-table row
// bt_row[m] of `bt` (row stride `max_blocks`) at logical position pos[m].
void LaunchStoreKvPaged(const bf16* src, int src_offset, int src_stride,
                        bf16* cache, const int* bt, int max_blocks,
                        int block_size, int kv_dim, const int* bt_row,
                        const int* pos, int M, cudaStream_t s);
