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

    // --- block bookkeeping: hand out / return physical block ids ---
    int  num_free() const { return (int)free_blocks_.size(); }
    bool alloc_one(int& out) {            // pop one free block; false (out untouched) if none left
        if (free_blocks_.empty()) return false;
        out = free_blocks_.back(); free_blocks_.pop_back();
        return true;
    }
    void free_many(const std::vector<int>& blocks) {   // return blocks to the pool
        for (int b : blocks) free_blocks_.push_back(b);
    }

private:
    int num_blocks_ = 0;
    int max_blocks_ = 0;
    std::vector<bf16*> k_, v_;
    std::vector<int>   free_blocks_;   // free physical block ids (used as a stack)
};

// Write side of the pool: scatter M freshly-projected K (or V) rows into `cache` (one layer's
// k(l)/v(l)). Token m goes to block-table row bt_row[m] of `bt` (row stride `max_blocks`), at
// logical position pos[m]. Defined in block_pool.cu; the read side is the attention kernels.
void launch_store_kv_paged(const float* src, bf16* cache, const int* bt, int max_blocks,
                           int block_size, int kv_dim, const int* bt_row, const int* pos,
                           int M, cudaStream_t s);
