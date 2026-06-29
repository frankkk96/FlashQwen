#include <cstdio>
#include <stdexcept>

#include "kernels.cuh"  // CUDA_CHECK
#include "kv_store.h"
#include "log.h"

KvStore::KvStore(const ModelSpec& spec, int max_ctx, float gpu_mem_fraction)
    : spec_(spec) {
  int kvd = spec.KvDim();
  int max_blocks = (max_ctx + kKvBlock - 1) / kKvBlock;  // full-seq table len

  // Pool = VRAM left under the gpu_mem_fraction cap, carved into kKvBlock-token
  // blocks handed out on demand, so concurrency is decoupled from max_ctx.
  size_t freeb, totalb;
  CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
  size_t used = totalb - freeb;
  size_t budget = static_cast<size_t>(totalb * gpu_mem_fraction);
  size_t kv_avail = budget > used ? budget - used : 0;
  size_t per_block =
      static_cast<size_t>(spec.num_layers) * 2 * kKvBlock * kvd * sizeof(bf16);
  num_blocks_ = static_cast<int>(kv_avail / per_block);
  if (num_blocks_ < max_blocks) {
    char msg[256];
    std::snprintf(
        msg, sizeof msg,
        "not enough VRAM for even one full-length sequence at max_ctx=%d "
        "(need %d blocks = %.1f GB, have %d blocks = %.1f GB under the %.0f%% "
        "cap)",
        max_ctx, max_blocks, max_blocks * per_block / 1e9, num_blocks_,
        kv_avail / 1e9, gpu_mem_fraction * 100);
    throw std::runtime_error(msg);
  }

  d_kv_.resize(spec.num_layers);
  for (int l = 0; l < spec.num_layers; ++l) {
    CUDA_CHECK(cudaMalloc(&d_kv_[l], static_cast<size_t>(num_blocks_) *
                                         kKvPlanes * kKvBlock * kvd *
                                         sizeof(bf16)));
  }

  CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
  LOG_INFO(
      "[kv] %d blocks x %d tok = %d-token pool (max_ctx=%d). GPU mem: %.1f GB "
      "used / %.1f GB total",
      num_blocks_, kKvBlock, num_blocks_ * kKvBlock, max_ctx,
      (totalb - freeb) / 1e9, totalb / 1e9);
}

KvStore::~KvStore() {
  for (auto p : d_kv_)
    if (p) cudaFree(p);
}

// Write side: scatter freshly-projected K AND V rows into the combined pool in
// one pass (attention kernels read it; see kernels.cu). Token m -> block-table
// row bt_row[m] of `bt`, logical position pos[m]. The K destination is computed
// once; V lives one plane stride (kKvBlock*kv_dim) further in the same tensor
// (FlashInfer NHD; see kv_layout.h). The K/V slices live in the fused qkv row
// at k_off / v_off (stride src_stride).
__global__ void StoreKvKernel(const bf16* __restrict__ qkv, int src_stride,
                              int k_off, int v_off, bf16* __restrict__ kv_cache,
                              const int* __restrict__ bt, int bt_stride,
                              int kv_dim, const int* __restrict__ bt_row,
                              const int* __restrict__ pos, int M) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * kv_dim) return;
  int m = idx / kv_dim, i = idx % kv_dim;
  // Logical position pos[m] -> physical K slot: block_table[pos/bs] selects the
  // block, *kKvPlanes makes room for the K and V planes, +pos%bs is the token.
  // The attention read side mirrors this addressing inline (see kernels.cu).
  const int* btr = bt + bt_row[m] * bt_stride;
  int p = pos[m];
  int64_t plane = static_cast<int64_t>(kKvBlock) * kv_dim;
  int64_t k_dst = static_cast<int64_t>(btr[p / kKvBlock]) * kKvPlanes * plane +
                  static_cast<int64_t>(p % kKvBlock) * kv_dim + i;
  int base = m * src_stride;
  kv_cache[k_dst] = qkv[base + k_off + i];
  kv_cache[k_dst + plane] = qkv[base + v_off + i];
}

void KvStore::StoreKV(int layer, const bf16* qkv, const int* bt, int bt_stride,
                      const int* bt_row, const int* pos, int M,
                      cudaStream_t s) const {
  int qd = spec_.QDim(), kvd = spec_.KvDim();
  int src_stride = qd + 2 * kvd, k_off = qd, v_off = qd + kvd;
  int n = M * kvd, blk = 256;
  StoreKvKernel<<<(n + blk - 1) / blk, blk, 0, s>>>(
      qkv, src_stride, k_off, v_off, d_kv_[layer], bt, bt_stride, kvd, bt_row,
      pos, M);
}
