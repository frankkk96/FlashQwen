#pragma once

// Page size of the paged KV cache: tokens per block. Single source of truth
// shared by the storage (KvStore), the allocator (BlockAllocator), and the
// inline logical->physical addressing in the store/attention kernels.
// Host-clean so the CUDA-free allocator can include it.
constexpr int kKvBlock = 16;

// Combined paged KV layout, FlashInfer NHD:
//   KV[layer] : [num_blocks, 2, kKvBlock, num_kv_heads, head_dim]  bf16
// The size-2 axis selects K (0) vs V (1). K and V share one allocation per
// layer; the store/attention kernels address the K plane and reach V by adding
// one plane stride. Strides, with kv_dim = num_kv_heads * head_dim:
//   plane stride = kKvBlock * kv_dim   (one full page of tokens; K plane -> V)
//   block stride = 2 * kKvBlock * kv_dim
//   K_off(block, tok, h, d) = block*block_stride + tok*kv_dim + h*head_dim + d
//   V_off                   = K_off + plane_stride
constexpr int kKvPlanes = 2;
