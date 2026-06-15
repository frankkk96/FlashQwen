// Paged KV block pool — the storage layer for PagedAttention, kept separate from both the model
// and the scheduler:
//   * the MODEL only computes — its attention kernels read/write KV through k(l)/v(l);
//   * the SCHEDULER only allocates — it hands out block ids from a free list (see scheduler.cpp);
//   * this class only stores — a flat set of fixed-size blocks shared by all sequences, laid out
//     per layer as [num_blocks, BLOCK, kv_dim] in BF16.
// The three meet only through block tables (vectors of physical block ids) and the raw k/v base
// pointers. Sizing is "whatever VRAM is left", so a KVCache must be constructed AFTER the model
// has uploaded its weights + activation scratch (see main.cpp).
//
// Porting note: a different transformer that still uses a per-layer paged KV cache can reuse this
// class unchanged — only the model's attention kernels that index k(l)/v(l) need rewriting.
#pragma once
#include "model_spec.hpp"
#include "kernels.cuh"   // bf16
#include <vector>

class KVCache {
public:
    static constexpr int BLOCK = 16;   // tokens per block (page)

    // Carve the pool out of the VRAM left under gpu_mem_fraction. Needs room for at least one
    // full-length (max_ctx) sequence; prints an error and exits the process if not.
    KVCache(const ModelSpec& spec, int max_ctx, float gpu_mem_fraction);
    ~KVCache();
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    bf16* k(int layer) const { return k_[layer]; }   // [num_blocks, BLOCK, kv_dim] for `layer`
    bf16* v(int layer) const { return v_[layer]; }

    int block_size() const { return BLOCK; }
    int num_blocks() const { return num_blocks_; }          // physical blocks in the pool
    int max_blocks_per_seq() const { return max_blocks_; }  // ceil(max_ctx / BLOCK)

private:
    int num_blocks_ = 0;
    int max_blocks_ = 0;
    std::vector<bf16*> k_, v_;
};

// Write side of the pool: scatter M freshly-projected K (or V) rows into `cache` (one layer's
// k(l)/v(l)). Token m goes to block-table row bt_row[m] of `bt` (row stride `max_blocks`), at
// logical position pos[m]. Defined in kv_cache.cu; the read side is the attention kernels.
void launch_store_kv_paged(const float* src, bf16* cache, const int* bt, int max_blocks,
                           int block_size, int kv_dim, const int* bt_row, const int* pos,
                           int M, cudaStream_t s);
