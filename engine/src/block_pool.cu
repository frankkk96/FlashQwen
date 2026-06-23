#include <cstdio>
#include <stdexcept>

#include "block_pool.hpp"
#include "kv_cache.cuh"  // KvPhysRow: shared addressing contract
#include "log.hpp"

BlockPool::BlockPool(const ModelSpec& spec, int max_ctx,
                     float gpu_mem_fraction) {
  int kvd = spec.KvDim();
  max_blocks_ =
      (max_ctx + kBlock - 1) / kBlock;  // block-table len for a full seq

  // Pool = VRAM left under the gpu_mem_fraction cap, carved into kBlock-token
  // blocks handed out on demand, so concurrency is decoupled from max_ctx.
  size_t freeb, totalb;
  CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
  size_t used = totalb - freeb;
  size_t budget = (size_t)(totalb * gpu_mem_fraction);
  size_t kv_avail = budget > used ? budget - used : 0;
  size_t per_block = (size_t)spec.num_layers * 2 * kBlock * kvd * sizeof(bf16);
  num_blocks_ = (int)(kv_avail / per_block);
  if (num_blocks_ < max_blocks_) {
    char msg[256];
    std::snprintf(
        msg, sizeof msg,
        "not enough VRAM for even one full-length sequence at max_ctx=%d "
        "(need %d blocks = %.1f GB, have %d blocks = %.1f GB under the %.0f%% "
        "cap)",
        max_ctx, max_blocks_, max_blocks_ * per_block / 1e9, num_blocks_,
        kv_avail / 1e9, gpu_mem_fraction * 100);
    throw std::runtime_error(msg);
  }

  k_.resize(spec.num_layers);
  v_.resize(spec.num_layers);
  for (int l = 0; l < spec.num_layers; ++l) {
    CUDA_CHECK(
        cudaMalloc(&k_[l], (size_t)num_blocks_ * kBlock * kvd * sizeof(bf16)));
    CUDA_CHECK(
        cudaMalloc(&v_[l], (size_t)num_blocks_ * kBlock * kvd * sizeof(bf16)));
  }

  // All blocks start reclaimable (refcount 0, uncached); seed the LRU queue in
  // id order (front = block 0, handed out first).
  ref_cnt_.assign(num_blocks_, 0);
  blk_hash_.assign(num_blocks_, 0);
  pos_.assign(num_blocks_, free_lru_.end());
  for (int b = 0; b < num_blocks_; ++b) {
    free_lru_.push_back(b);
    pos_[b] = std::prev(free_lru_.end());
  }

  CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
  LOG_INFO(
      "[kv] %d blocks x %d tok = %d-token pool (max_ctx=%d). GPU mem: %.1f GB "
      "used / %.1f GB total",
      num_blocks_, kBlock, num_blocks_ * kBlock, max_ctx,
      (totalb - freeb) / 1e9, totalb / 1e9);
}

BlockPool::~BlockPool() {
  for (auto p : k_)
    if (p) cudaFree(p);
  for (auto p : v_)
    if (p) cudaFree(p);
}

// Write side: scatter freshly-projected K/V rows into the pool (attention
// kernels read it; see kernels.cu). Token m -> block-table row bt_row[m] of
// `bt`, logical position pos[m].
__global__ void StoreKvPagedKernel(const bf16* __restrict__ src, int src_offset,
                                   int src_stride, bf16* __restrict__ cache,
                                   const int* __restrict__ bt, int max_blocks,
                                   int block_size, int kv_dim,
                                   const int* __restrict__ bt_row,
                                   const int* __restrict__ pos, int M) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * kv_dim) return;
  int m = idx / kv_dim, i = idx % kv_dim;
  size_t phys =
      KvPhysRow(bt + (size_t)bt_row[m] * max_blocks, pos[m], block_size);
  cache[phys * kv_dim + i] = src[(size_t)m * src_stride + src_offset + i];
}

void LaunchStoreKvPaged(const bf16* src, int src_offset, int src_stride,
                        bf16* cache, const int* bt, int max_blocks,
                        int block_size, int kv_dim, const int* bt_row,
                        const int* pos, int M, cudaStream_t s) {
  int n = M * kv_dim, blk = 256;
  StoreKvPagedKernel<<<(n + blk - 1) / blk, blk, 0, s>>>(
      src, src_offset, src_stride, cache, bt, max_blocks, block_size, kv_dim,
      bt_row, pos, M);
}
