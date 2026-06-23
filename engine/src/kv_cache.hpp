// KV cache manager — request-level layer between Scheduler and BlockPool. The
// scheduler thinks in requests, the pool in physical block ids; this grows a
// request's block table out of the pool and returns it on release, so the
// scheduler never touches the free list. Also the prefix-caching seam
// (CacheLookup / CacheInsert), forwarding to the pool's registry.
#pragma once
#include <vector>

#include "block_pool.hpp"

class KVCacheManager {
 public:
  explicit KVCacheManager(BlockPool& pool)
      : pool_(pool), bsz_(pool.BlockSize()) {}

  // Sizing / capacity (scheduler budgeting + error messages).
  int BlockSize() const { return bsz_; }
  int NumBlocks() const { return pool_.NumBlocks(); }
  int NumFree() const { return pool_.NumFree(); }
  int BlocksFor(int n_tok) const { return (n_tok + bsz_ - 1) / bsz_; }

  // Grow `block_table` to cover `num_tokens`, pulling from the pool. Returns
  // false the moment the pool runs dry (partial growth left in place) so the
  // caller can preempt a sequence and retry.
  bool Grow(std::vector<int>& block_table, int num_tokens) {
    int need = BlocksFor(num_tokens);
    while ((int)block_table.size() < need) {
      int b;
      if (!pool_.AllocOne(b)) return false;
      block_table.push_back(b);
    }
    return true;
  }

  // Return a request's blocks to the pool and clear its table.
  void Free(std::vector<int>& block_table) {
    pool_.FreeMany(block_table);
    block_table.clear();
  }

  // Prefix-caching seam: scheduler owns token ids + block-hash chaining; this
  // just forwards to the pool's registry.
  int CacheLookup(uint64_t hash) {
    return pool_.CacheLookup(hash);
  }  // hit refs the block; -1 = miss
  void CacheInsert(int b, uint64_t hash) { pool_.CacheInsert(b, hash); }

 private:
  BlockPool& pool_;
  int bsz_;
};
