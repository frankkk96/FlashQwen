#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

#include "cuda_helpers.h"
#include "model_spec.h"

namespace fq {

// Model-aware paged KV cache for this engine's single attention layout. Owns
// one combined per-layer device tensor [NumBlocks, kKvPlanes, kKvBlock,
// kv_dim] BF16 (FlashInfer NHD) that all sequences share — attention kernels
// read it via KV(l), StoreKV() scatters into it. The qkv slice offsets,
// kv_dim and block size are derived from the compile-time ModelSpec
// constants. Sized from VRAM left under the gpu_mem_fraction cap, so
// construct AFTER weights + activation scratch upload. The block ids that index
// this tensor are managed by BlockAllocator; the two meet only at the integer
// block id (same seam vLLM draws between runner tensors and the block pool).
class KvStore {
 public:
  static constexpr int kKvPlanes = 2;  // K,V planes per block

  KvStore(int max_ctx, float gpu_mem_fraction);
  KvStore(const KvStore&) = delete;
  KvStore& operator=(const KvStore&) = delete;

  void StoreKV(int layer, const bf16* qkv, const int* bt, int bt_stride,
               const int* bt_row, const int* pos, int M,
               cudaStream_t stream) const;

  bf16* KV(int layer) const { return kv_[layer].D(); }
  int BlockSize() const { return ModelSpec::kKvBlock; }
  int NumBlocks() const { return num_blocks_; }

 private:
  int num_blocks_ = 0;
  std::vector<DeviceBuffer<bf16>> kv_;
};

}
