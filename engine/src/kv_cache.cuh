// Device-side layout contract for the paged KV pool: per-layer pool is
// [NumBlocks, kBlock, kv_dim] BF16, a sequence's KV is its block table (list of
// physical block ids). SINGLE definition of "logical position -> physical row",
// shared by the store kernel (block_pool.cu) and attention kernels
// (kernels.cu).
#pragma once
#include <cstddef>

// Physical row for logical position p of the sequence whose block table starts
// at `bt_row`. Address as cache + row*kv_dim.
__device__ __forceinline__ size_t KvPhysRow(const int* bt_row, int p,
                                            int block_size) {
  return (size_t)bt_row[p / block_size] * block_size + (p % block_size);
}
