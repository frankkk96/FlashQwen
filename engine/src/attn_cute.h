#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace fq {

void LaunchAttnPrefillCute(const __nv_bfloat16* q,
                           const __nv_bfloat16* cache_kv, __nv_bfloat16* out,
                           const int* pos, const int* qstart, const int* qlen,
                           const int* rids, int R, int max_qlen, const int* bt,
                           int max_blocks, cudaStream_t s);

void LaunchAttnDecodeCute(const __nv_bfloat16* q,
                          const __nv_bfloat16* cache_kv, __nv_bfloat16* out,
                          const int* pos, const int* qstart,
                          const int* decode_rids, int n_decode, const int* bt,
                          int max_blocks, float* pm, float* pl,
                          float* pa, cudaStream_t s);

}  // namespace fq
