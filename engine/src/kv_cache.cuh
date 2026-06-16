// Device-side layout contract for the paged KV pool. The pool for a layer is
// [num_blocks, BLOCK, kv_dim] BF16; a sequence's KV is a list of physical block ids (its block
// table). This is the SINGLE definition of "logical position -> physical row" — the store kernel
// (kv_cache.cu) and the attention kernels (kernels.cu) both resolve addresses through it, so the
// layout lives in exactly one place.
#pragma once
#include <cstddef>

// Physical row in the pool for logical position p of the sequence whose block table starts at
// `bt_row` (one row of the packed block-table buffer). Address as cache + row*kv_dim.
__device__ __forceinline__ size_t kv_phys_row(const int* bt_row, int p, int block_size) {
    return (size_t)bt_row[p / block_size] * block_size + (p % block_size);
}
