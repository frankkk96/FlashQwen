// KV cache manager — the request-level layer between the Scheduler and the BlockPool.
//   * the SCHEDULER thinks in requests ("this sequence needs room for one more token");
//   * the BLOCKPOOL thinks in physical block ids;
//   * this manager translates between them — it grows a request's block table out of the pool and
//     returns it on release, so the scheduler never touches the pool's free list directly.
// It is also the seam where prefix caching will later live (get_computed_blocks / cache_blocks);
// for now it is a thin, behavior-preserving wrapper around BlockPool.
#pragma once
#include "block_pool.hpp"
#include <vector>

class KVCacheManager {
public:
    explicit KVCacheManager(BlockPool& pool) : pool_(pool), bsz_(pool.block_size()) {}

    // --- sizing / capacity (used by the scheduler for budgeting + error messages) ---
    int block_size() const { return bsz_; }
    int num_blocks() const { return pool_.num_blocks(); }
    int num_free()   const { return pool_.num_free(); }
    int blocks_for(int n_tok) const { return (n_tok + bsz_ - 1) / bsz_; }

    // Grow `block_table` until it covers `num_tokens` logical tokens, pulling blocks from the pool.
    // Returns true once covered; returns false the moment the pool runs dry (leaving the partial
    // growth in place) so the caller can preempt a sequence and call again to continue.
    bool grow(std::vector<int>& block_table, int num_tokens) {
        int need = blocks_for(num_tokens);
        while ((int)block_table.size() < need) {
            int b;
            if (!pool_.alloc_one(b)) return false;
            block_table.push_back(b);
        }
        return true;
    }

    // Return a request's blocks to the pool and clear its table (finished / preempted / cancelled).
    void free(std::vector<int>& block_table) {
        pool_.free_many(block_table);
        block_table.clear();
    }

    // --- prefix caching: the get_computed_blocks / cache_blocks seam (driven by the Scheduler, which
    //     owns the token ids + block-hash chaining; this layer just forwards to the pool's registry) ---
    int  cache_lookup(uint64_t hash)        { return pool_.cache_lookup(hash); }  // hit refs the block; -1 = miss
    void cache_insert(int b, uint64_t hash) { pool_.cache_insert(b, hash); }      // register a full block's content
    void incref(int b)                      { pool_.incref(b); }

private:
    BlockPool& pool_;
    int bsz_;
};
