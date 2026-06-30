#pragma once
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "kv_layout.h"
#include "cuda_helpers.h"
#include "log.h"

namespace fq {

void LaunchGemm(cublasHandle_t handle, const bf16* x, const bf16* W, void* y,
                int M, int IN, int OUT, cudaDataType_t y_type);

void LaunchRmsnorm(const bf16* x, const float* w, bf16* out, int M, int H,
                   float eps, cudaStream_t s);

void LaunchAddRmsnorm(bf16* x, const bf16* res, const float* w, bf16* out,
                      int M, int H, float eps, cudaStream_t s);

void LaunchEmbed(const int* ids, const bf16* embed, bf16* out, int M, int H,
                 cudaStream_t s);

void LaunchHeadNormRope(bf16* buf, const float* wq, const float* wk,
                        const float* cos_tab, const float* sin_tab,
                        const int* pos, int M, int n_q, int n_kv, int head_dim,
                        int stride, float eps, cudaStream_t s);

constexpr int kMaxKsplit = 16;
constexpr int kDecodeMaxHeads = 64;
inline size_t DecodePartialFloats(int max_decode) {
  return static_cast<size_t>(max_decode) * kDecodeMaxHeads * kMaxKsplit;
}

void LaunchGatherRows(const bf16* x, const int* rows, bf16* out, int S, int H,
                      cudaStream_t s);

void LaunchAdd(bf16* out, const bf16* in, int N, cudaStream_t s);

void LaunchSiluMul(const bf16* gateup, bf16* h, int M, int I, cudaStream_t s);

void LaunchSampleBatch(const float* logits, int B, int N, const float* invT,
                       const float* topp, const float* u, int* out,
                       cudaStream_t s);

}
