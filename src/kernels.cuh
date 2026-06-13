// CUDA kernels for the Qwen3 forward pass.
//
// Convention: weights are stored on-device as BF16 (exactly as in the safetensors file);
// activations are FP32. Matmul reads BF16 weights, FP32 activations, accumulates in FP32.
// Linear layers follow the HF layout y = x @ W^T with W shaped [out_features, in_features].
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                     cudaGetErrorString(_e));                                   \
        std::exit(1);                                                           \
    }                                                                           \
} while (0)

using bf16 = __nv_bfloat16;

// y[M,OUT] = x[M,IN] @ W[OUT,IN]^T            (W is BF16, x/y are FP32, no bias)
// Decode (M==1) uses a memory-bound GEMV. Prefill (M>1) converts x to BF16 into the
// `x_bf16` scratch (>= ceil(M/16)*16 * IN elements) and uses tensor cores (WMMA).
// IN and OUT must be multiples of 16 (true for all Qwen3 layer dims).
void launch_matmul(const float* x, const bf16* W, float* y, int M, int IN, int OUT,
                   bf16* x_bf16, cudaStream_t s);

// out[M,H] = rmsnorm(x[M,H]) * w[H]           (w is FP32)
void launch_rmsnorm(const float* x, const float* w, float* out, int M, int H, float eps, cudaStream_t s);

// Gather embedding rows: out[m,:] = embed[ids[m], :]   (embed BF16 -> out FP32)
void launch_embed(const int* ids, const bf16* embed, float* out, int M, int H, cudaStream_t s);

// In-place rotary position embedding on x[M, n_heads, head_dim], positions pos[M].
void launch_rope(float* x, const int* pos, int M, int n_heads, int head_dim, float theta, cudaStream_t s);

// Grouped-query attention with a KV cache. q[M, n_heads, hd]; cache_k/v are
// [max_seq, n_kv, hd]; out[M, n_heads, hd]. Query token m attends keys [0, past_len+m].
void launch_attention(const float* q, const float* cache_k, const float* cache_v,
                      float* out, int M, int n_heads, int n_kv, int head_dim,
                      int past_len, float scale, cudaStream_t s);

// out[i] += in[i]   for N elements
void launch_add(float* out, const float* in, int N, cudaStream_t s);

// h[i] = silu(gate[i]) * up[i]   for N elements
void launch_silu_mul(const float* gate, const float* up, float* h, int N, cudaStream_t s);
