#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// CuTe (CUTLASS) implementation of the prefill attention path. Drop-in for
// LaunchAttnPrefill (see kernels.cuh): same paged combined-NHD KV cache, GQA,
// causal masking, head_dim==128, block_size==16. Lives in its own translation
// unit so the heavy CUTLASS template instantiation does not slow the rest of
// the engine's build. Selected at runtime via the FQ_ATTN_CUTE env flag.
void LaunchAttnPrefillCute(const __nv_bfloat16* q, int q_stride,
                           const __nv_bfloat16* cache_kv, __nv_bfloat16* out,
                           int n_heads, int n_kv, int head_dim, const int* pos,
                           const int* qstart, const int* qlen, const int* rids,
                           int R, int max_qlen, const int* bt, int max_blocks,
                           int block_size, float scale, cudaStream_t s);

// CuTe decode path; drop-in for LaunchAttnDecode (see kernels.cuh). pm/pl/pa
// are caller-owned split-partials scratch (DecodePartialFloats()).
void LaunchAttnDecodeCute(const __nv_bfloat16* q, int q_stride,
                          const __nv_bfloat16* cache_kv, __nv_bfloat16* out,
                          int n_heads, int n_kv, int head_dim, const int* pos,
                          const int* qstart, const int* decode_rids, int n_decode,
                          const int* bt, int max_blocks, int block_size,
                          float scale, float* pm, float* pl, float* pa,
                          cudaStream_t s);
