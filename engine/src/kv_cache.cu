#include "kv_cache.hpp"
#include "kv_cache.cuh"   // kv_phys_row: shared addressing contract
#include <cstdio>
#include <cstdlib>

KVCache::KVCache(const ModelSpec& spec, int max_ctx, float gpu_mem_fraction) {
    int kvd = spec.kv_dim();
    max_blocks_ = (max_ctx + BLOCK - 1) / BLOCK;   // block-table length for a full-length sequence

    // The pool gets whatever VRAM is left under the gpu_mem_fraction cap, carved into fixed-size
    // blocks of BLOCK tokens. Unlike a per-sequence reservation, blocks are handed out on demand by
    // the scheduler, so the number of concurrent sequences is decoupled from max_ctx.
    size_t freeb, totalb;
    CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
    size_t used      = totalb - freeb;
    size_t budget    = (size_t)(totalb * gpu_mem_fraction);
    size_t kv_avail  = budget > used ? budget - used : 0;
    size_t per_block = (size_t)spec.num_layers * 2 * BLOCK * kvd * sizeof(bf16);
    num_blocks_ = (int)(kv_avail / per_block);
    if (num_blocks_ < max_blocks_) {
        std::fprintf(stderr, "error: not enough VRAM for even one full-length sequence at "
                     "max_ctx=%d (need %d blocks = %.1f GB, have %d blocks = %.1f GB under the "
                     "%.0f%% cap).\n", max_ctx, max_blocks_, max_blocks_ * per_block / 1e9,
                     num_blocks_, kv_avail / 1e9, gpu_mem_fraction * 100);
        std::exit(1);
    }

    k_.resize(spec.num_layers);
    v_.resize(spec.num_layers);
    for (int l = 0; l < spec.num_layers; ++l) {
        CUDA_CHECK(cudaMalloc(&k_[l], (size_t)num_blocks_ * BLOCK * kvd * sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&v_[l], (size_t)num_blocks_ * BLOCK * kvd * sizeof(bf16)));
    }

    CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
    std::fprintf(stderr, "[kv] %d blocks x %d tok = %d-token pool (max_ctx=%d). "
                 "GPU mem: %.1f GB used / %.1f GB total\n",
                 num_blocks_, BLOCK, num_blocks_ * BLOCK, max_ctx,
                 (totalb - freeb) / 1e9, totalb / 1e9);
}

KVCache::~KVCache() {
    for (auto p : k_) if (p) cudaFree(p);
    for (auto p : v_) if (p) cudaFree(p);
}

// ---- write side: scatter freshly-projected K/V rows into the pool -----------------------
// The cache's only mutating operation (the attention kernels read it; see kernels.cu). Token m
// belongs to the sequence whose block table is row bt_row[m] of `bt`, at logical position pos[m].
__global__ void store_kv_paged_kernel(const float* __restrict__ src, bf16* __restrict__ cache,
                                      const int* __restrict__ bt, int max_blocks, int block_size,
                                      int kv_dim, const int* __restrict__ bt_row,
                                      const int* __restrict__ pos, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * kv_dim) return;
    int m = idx / kv_dim, i = idx % kv_dim;
    size_t phys = kv_phys_row(bt + (size_t)bt_row[m] * max_blocks, pos[m], block_size);
    cache[phys * kv_dim + i] = __float2bfloat16(src[idx]);
}

void launch_store_kv_paged(const float* src, bf16* cache, const int* bt, int max_blocks,
                           int block_size, int kv_dim, const int* bt_row, const int* pos,
                           int M, cudaStream_t s) {
    int n = M * kv_dim, blk = 256;
    store_kv_paged_kernel<<<(n + blk - 1) / blk, blk, 0, s>>>(
        src, cache, bt, max_blocks, block_size, kv_dim, bt_row, pos, M);
}
