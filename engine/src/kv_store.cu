#include <cstdio>
#include <stdexcept>

#include "cuda_helpers.h"
#include "kv_store.h"
#include "log.h"

namespace fq {

KvStore::KvStore(const ModelSpec& spec, int max_ctx, float gpu_mem_fraction)
    : spec_(spec) {
  int kv_dim = spec.KvDim();
  int max_blocks = (max_ctx + kKvBlock - 1) / kKvBlock;

  size_t freeb, totalb;
  CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
  size_t used = totalb - freeb;
  size_t budget = static_cast<size_t>(totalb * gpu_mem_fraction);
  size_t kv_avail = budget > used ? budget - used : 0;
  size_t per_block =
      static_cast<size_t>(spec.num_layers) * 2 * kKvBlock * kv_dim * sizeof(bf16);
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

  kv_.resize(spec.num_layers);
  for (int l = 0; l < spec.num_layers; ++l)
    kv_[l] = DeviceBuffer<bf16>(static_cast<size_t>(num_blocks_) * kKvPlanes *
                                kKvBlock * kv_dim);

  CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
  LOG_INFO(
      "[kv] %d blocks x %d tok = %d-token pool (max_ctx=%d). GPU mem: %.1f GB "
      "used / %.1f GB total",
      num_blocks_, kKvBlock, num_blocks_ * kKvBlock, max_ctx,
      (totalb - freeb) / 1e9, totalb / 1e9);
}

__global__ void StoreKvKernel(const bf16* __restrict__ qkv, int src_stride,
                              int k_off, int v_off, bf16* __restrict__ kv_cache,
                              const int* __restrict__ bt, int bt_stride,
                              int kv_dim, const int* __restrict__ bt_row,
                              const int* __restrict__ pos, int M) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * kv_dim) return;
  int m = idx / kv_dim, i = idx % kv_dim;
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
                      cudaStream_t stream) const {
  int q_dim = spec_.QDim(), kv_dim = spec_.KvDim();
  int src_stride = q_dim + 2 * kv_dim, k_off = q_dim, v_off = q_dim + kv_dim;
  int n = M * kv_dim, threads = 256;
  StoreKvKernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
      qkv, src_stride, k_off, v_off, kv_[layer].D(), bt, bt_stride, kv_dim,
      bt_row, pos, M);
}

}
