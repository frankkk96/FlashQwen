// CUDA kernels for the Qwen3 forward pass.
//
// Weights and activations are BF16 (no quantization). Matmuls use cuBLAS
// (cublasGemmEx, BF16 in / FP32 accumulate); elementwise/attention kernels are
// BF16 in/out with FP32 internal math. Linear layers use the HF layout
// y = x @ W^T with W shaped [out_features, in_features].
#pragma once
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "kv_layout.h"  // kKvBlock, kKvPlanes (combined KV addressing)
#include "log.h"

#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t _e = (call);                                \
    if (_e != cudaSuccess) {                                \
      LOG_ERROR("CUDA error %s:%d: %s", __FILE__, __LINE__, \
                cudaGetErrorString(_e));                    \
      std::exit(1);                                         \
    }                                                       \
  } while (0)

#define CUBLAS_CHECK(call)                                         \
  do {                                                             \
    cublasStatus_t _s = (call);                                    \
    if (_s != CUBLAS_STATUS_SUCCESS) {                             \
      LOG_ERROR("cuBLAS error %s:%d: %d", __FILE__, __LINE__, _s); \
      std::exit(1);                                                \
    }                                                              \
  } while (0)

using bf16 = __nv_bfloat16;

// y[M,OUT] = x[M,IN] @ W[OUT,IN]^T via cuBLAS (BF16 in / FP32 accumulate).
// cuBLAS is column-major, so row-major W/x map to column-major Wc[IN,OUT] /
// Xc[IN,M] and Z = Wc^T @ Xc (m=OUT,n=M,k=IN) gives row-major y[M,OUT]. Ytype
// picks the output element type (CUDA_R_16BF activations, CUDA_R_32F lm_head
// logits). Uses whatever stream is set on `handle`.
void LaunchGemm(cublasHandle_t handle, const bf16* x, const bf16* W, void* y,
                int M, int IN, int OUT, cudaDataType_t Ytype);

// out[M,H] = rmsnorm(x[M,H]) * w[H]   (x/out BF16, weight FP32, FP32 reduction)
void LaunchRmsnorm(const bf16* x, const float* w, bf16* out, int M, int H,
                   float eps, cudaStream_t s);

// Fused residual-add + rmsnorm: x[m] += res[m] (written back to carry the
// residual forward), then out[m] = rmsnorm(x[m]) * w. Saves a full H
// read/write of x and a kernel launch vs separate add + rmsnorm.
void LaunchAddRmsnorm(bf16* x, const bf16* res, const float* w, bf16* out,
                      int M, int H, float eps, cudaStream_t s);

// Gather embedding rows: out[m,:] = embed[ids[m], :]   (embed + out BF16)
void LaunchEmbed(const int* ids, const bf16* embed, bf16* out, int M, int H,
                 cudaStream_t s);

// Fused per-head RMSNorm + RoPE over the q and k slices of a fused QKV row of
// width `stride` in one launch. The q heads (weight wq, n_q of them) and k
// heads (weight wk, n_kv) are contiguous: head g of token m lives at buf +
// m*stride + g*head_dim, RMSNorm'd (FP32) then rotated via precomputed cos/sin
// tables ([max_pos, head_dim/2]) indexed by pos[m]. v is left raw. In place;
// blockDim must be head_dim.
void LaunchHeadNormRope(bf16* buf, const float* wq, const float* wk,
                        const float* cos_tab, const float* sin_tab,
                        const int* pos, int M, int n_q, int n_kv, int head_dim,
                        int stride, float eps, cudaStream_t s);

// ---- Paged-KV attention ----
// A layer's KV pool is one combined [num_blocks, 2, kKvBlock, kv_dim] BF16
// tensor (FlashInfer NHD; see kv_layout.h), the size-2 axis selecting K/V; a
// sequence's KV is a list of physical block ids (its block table).
// Logical->physical addressing (block_table[pos/bs]*kKvPlanes selects the
// block, +pos%bs the token; V is one plane stride past K) is computed inline
// here and in the write side (kv_store.cu). `bt` holds one block table per
// request (stride `max_blocks`).
//
// The attention kernels themselves (prefill q_len>1, decode q_len==1) live in
// attn_cute.cu (CuTe/CUTLASS); this header only owns the shared decode-partials
// sizing they share with ModelRuntime.

// Max grid.z KV-splits for the decode path. The split partials pm/pl/pa are
// owned/preallocated by the caller (ModelRuntime), sized DecodePartialFloats():
// pm/pl one float per (decode-slot, head, split); pa adds head_dim.
constexpr int kMaxKsplit = 16;
inline size_t DecodePartialFloats(int max_decode) {
  return static_cast<size_t>(max_decode) * 64 /*heads guard*/ * kMaxKsplit;
}

// Gather S rows from x[*, H] into out[S, H]: out[i] = x[rows[i]] (BF16). Pulls
// the per-request last-token rows out of the flattened batch before the final
// norm + lm_head.
void LaunchGatherRows(const bf16* x, const int* rows, bf16* out, int S, int H,
                      cudaStream_t s);

// out[i] += in[i] for N elements (BF16, FP32 accumulate)
void LaunchAdd(bf16* out, const bf16* in, int N, cudaStream_t s);

// h[m,i] = silu(gateup[m,i]) * gateup[m, I+i] over M rows of a fused gate|up
// buffer (row width 2I).
void LaunchSiluMul(const bf16* gateup, bf16* h, int M, int I, cudaStream_t s);

// Batched sampling: logits is [B, N] FP32; out[b] = sampled token id for row b.
// One block per row.
//   invT[b]: 1/temperature, or <= 0 for greedy argmax (ignores u/topp)
//   topp[b]: nucleus cutoff (>= 1 disables truncation)
//   u[b]:    uniform(0,1) draw for the stochastic paths
void LaunchSampleBatch(const float* logits, int B, int N, const float* invT,
                       const float* topp, const float* u, int* out,
                       cudaStream_t s);
