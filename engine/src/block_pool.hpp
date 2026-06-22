// Paged KV block pool — the lowest layer of the KV stack: storage + block bookkeeping.
//   * STORAGE: a flat per-layer pool [num_blocks, BLOCK, kv_dim] BF16, shared by all sequences.
//   * BOOKKEEPING: a free list of physical block ids, handed out / returned on demand.
// On top of it sits KVCacheManager (kv_cache.hpp), which exposes a request-level API to the
// Scheduler; the model's attention kernels read/write the storage directly through k(l)/v(l). The
// three meet only through block tables (vectors of physical block ids) and the raw k/v base
// pointers. Sizing is "whatever VRAM is left", so a BlockPool must be constructed AFTER the model
// has uploaded its weights + activation scratch (see grpc_server.cpp).
//
// Porting note: a different transformer that still uses a per-layer paged KV cache can reuse this
// class unchanged — only the model's attention kernels that index k(l)/v(l) need rewriting.
#pragma once
#include "model_spec.hpp"
#include "kernels.cuh"   // bf16
#include <vector>
#include <list>
#include <unordered_map>
#include <cstdint>

class BlockPool {
public:
    static constexpr int BLOCK = 16;   // tokens per block (page)

    // Carve the pool out of the VRAM left under gpu_mem_fraction, then seed the free list with every
    // physical block id. Needs room for at least one full-length (max_ctx) sequence; prints an error
    // and exits the process if not.
    BlockPool(const ModelSpec& spec, int max_ctx, float gpu_mem_fraction);
    ~BlockPool();
    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;

    // --- storage: raw per-layer base pointers for the attention/store kernels ---
    bf16* k(int layer) const { return k_[layer]; }   // [num_blocks, BLOCK, kv_dim] for `layer`
    bf16* v(int layer) const { return v_[layer]; }

    int block_size() const { return BLOCK; }
    int num_blocks() const { return num_blocks_; }          // physical blocks in the pool
    int max_blocks_per_seq() const { return max_blocks_; }  // ceil(max_ctx / BLOCK)

    // --- block bookkeeping: reference-counted, prefix-cache-aware (vLLM-style) -------------------
    // Each block carries a refcount and (optionally) the content hash of the KV it holds. A block at
    // refcount 0 is *reclaimable* but not erased: it sits in an LRU free queue keeping its hash->id
    // mapping, so an identical prefix can resurrect it (cache hit). A fresh allocation pops the LRU
    // front (least-recently-freed); if that block was still cached, its mapping is evicted then.
    //   alloc_one : a brand-new block (refcount 1), evicting an LRU cached block if needed
    //   incref    : share a block already held (e.g. a reused prefix block) -> refcount+1
    //   free_one  : refcount-1; at 0 the block returns to the LRU tail (mapping kept = still cacheable)
    //   cache_lookup : block id for `hash` if cached (refs it on hit), else -1 -- NEVER allocates
    //   cache_insert : register a now-full, fully-computed block's content hash (idempotent; no-ops on
    //                  a hash collision so the hash<->id invariant stays 1:1)

    // reclaimable (refcount-0) blocks; the scheduler preempts a running sequence only when this hits 0
    int  num_free() const { return (int)free_lru_.size(); }

    bool alloc_one(int& out) {            // a fresh block (refcount 1); false (out untouched) if none free
        if (free_lru_.empty()) return false;
        int b = free_lru_.front(); free_lru_.pop_front();
        pos_[b] = free_lru_.end();        // no longer in the free queue
        if (blk_hash_[b]) {               // it was a cached-but-free block: evict its content mapping
            auto it = cached_.find(blk_hash_[b]);
            if (it != cached_.end() && it->second == b) cached_.erase(it);
            blk_hash_[b] = 0;
        }
        ref_cnt_[b] = 1;
        out = b;
        return true;
    }
    void incref(int b) { ++ref_cnt_[b]; }

    void free_one(int b) {                // return one block to the pool (refcount-1)
        if (--ref_cnt_[b] == 0) { free_lru_.push_back(b); pos_[b] = std::prev(free_lru_.end()); }
    }
    void free_many(const std::vector<int>& blocks) { for (int b : blocks) free_one(b); }

    // Block id holding `hash`'s content if present (resurrecting it from the free queue and adding a
    // reference), else -1. Pure lookup over already-resident blocks -> can never fail on pool pressure.
    int cache_lookup(uint64_t hash) {
        auto it = cached_.find(hash);
        if (it == cached_.end()) return -1;
        int b = it->second;
        if (ref_cnt_[b] == 0) { free_lru_.erase(pos_[b]); pos_[b] = free_lru_.end(); }  // resurrect
        ++ref_cnt_[b];
        return b;
    }
    // Register block `b` (currently held, now full + fully computed) as the cache entry for `hash`.
    void cache_insert(int b, uint64_t hash) {
        if (blk_hash_[b]) return;          // already cached
        if (cached_.count(hash)) return;   // identical content cached elsewhere: keep hash<->id 1:1
        blk_hash_[b] = hash;
        cached_[hash] = b;
    }

private:
    int num_blocks_ = 0;
    int max_blocks_ = 0;
    std::vector<bf16*> k_, v_;

    std::vector<int>      ref_cnt_;    // per-block reference count (0 => reclaimable)
    std::vector<uint64_t> blk_hash_;   // content hash currently held by each block (0 => none)
    std::unordered_map<uint64_t, int> cached_;   // content hash -> block id (1:1 with blk_hash_)
    std::list<int>        free_lru_;   // reclaimable block ids, LRU order (front = evict first)
    std::vector<std::list<int>::iterator> pos_;  // each block's node in free_lru_ (== end() if not in)
};

// Write side of the pool: scatter M freshly-projected K (or V) rows into `cache` (one layer's
// k(l)/v(l)). Token m's source row starts at src + m*src_stride + src_offset (so K/V can be read
// straight from a fused QKV buffer); it goes to block-table row bt_row[m] of `bt` (row stride
// `max_blocks`), at logical position pos[m]. Defined in block_pool.cu; read side = attention kernels.
void launch_store_kv_paged(const bf16* src, int src_offset, int src_stride, bf16* cache,
                           const int* bt, int max_blocks, int block_size, int kv_dim,
                           const int* bt_row, const int* pos, int M, cudaStream_t s);
