#pragma once

// Page size of the paged KV cache: tokens per block. Single source of truth
// shared by the storage (KvStore), the allocator (BlockAllocator), and the
// inline logical->physical addressing in the store/attention kernels.
// Host-clean so the CUDA-free allocator can include it.
constexpr int kKvBlock = 16;
