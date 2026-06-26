#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

#include "kv_layout.h"  // kKvBlock
#include "model_spec.h"

using bf16 = __nv_bfloat16;

// Model-aware paged KV cache for this engine's single attention layout. Owns
// the flat per-layer device tensors [NumBlocks, kKvBlock, kv_dim] BF16 that all
// sequences share — attention kernels read them via K(l)/V(l), StoreKV()
// scatters into them. Retains ModelSpec so the qkv slice offsets (QD/KVD),
// kv_dim and block size are derived internally rather than threaded through the
// call site. Sized from VRAM left under the gpu_mem_fraction cap, so construct
// AFTER weights + activation scratch upload. The block ids that index these
// tensors are managed by BlockAllocator; the two meet only at the integer
// block id (same seam vLLM draws between runner tensors and the block pool).
class KvStore {
 public:
  KvStore(const ModelSpec& spec, int max_ctx, float gpu_mem_fraction);
  ~KvStore();
  KvStore(const KvStore&) = delete;
  KvStore& operator=(const KvStore&) = delete;

  // Append token m's freshly-projected K and V (read from the fused qkv row at
  // the model's QD/KVD offsets) into `layer`'s pool, at block-table row
  // bt_row[m] (stride bt_stride), logical position pos[m]. M tokens total.
  void StoreKV(int layer, const bf16* qkv, const int* bt, int bt_stride,
               const int* bt_row, const int* pos, int M, cudaStream_t s) const;

  bf16* K(int layer) const { return d_k_[layer]; }
  bf16* V(int layer) const { return d_v_[layer]; }
  int BlockSize() const { return kKvBlock; }
  int NumBlocks() const { return num_blocks_; }

 private:
  ModelSpec spec_;
  int num_blocks_ = 0;
  std::vector<bf16*> d_k_, d_v_;
};
