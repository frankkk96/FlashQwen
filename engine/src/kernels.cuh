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
#include <cstdint>

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                     cudaGetErrorString(_e));                                   \
        std::exit(1);                                                           \
    }                                                                           \
} while (0)

using bf16 = __nv_bfloat16;

// y[M,OUT] = x[M,IN] @ W[OUT,IN]^T   — W is INT8 with a per-row `scale[OUT]` (no bias).
// Decode (M==1): memory-bound INT8 GEMV, dequantized in-kernel. Prefill (M>1): dequantize
// W to BF16 in `w_dq` (>= OUT*IN elements), convert x to BF16 in `x_bf16`, then WMMA.
// IN and OUT must be multiples of 16 (true for all Qwen3 layer dims).
void launch_matmul(const float* x, const int8_t* W, const float* scale, float* y,
                   int M, int IN, int OUT, bf16* x_bf16, bf16* w_dq, cudaStream_t s);

// Largest decode batch a single GEMV pass supports (register-array bound). Decode batches are
// capped at this; KV slots (B_max) may exceed it but never the per-step token count.
#define MAX_DECODE_B 32

// Batched decode GEMV: y[B,OUT] = x[B,IN] @ W[OUT,IN]^T, INT8 W + per-row scale. Each warp
// reads one weight row ONCE and computes all B dot products (weight traffic amortized across
// the batch — the throughput win of decode batching). B must be <= MAX_DECODE_B; the kernel
// over-computes up to the next template size in {1,2,4,8,16,32}, so y must hold that many rows.
void launch_matmul_decode(const float* x, const int8_t* W, const float* scale, float* y,
                          int B, int IN, int OUT, cudaStream_t s);

// out[M,H] = rmsnorm(x[M,H]) * w[H]           (w is FP32)
void launch_rmsnorm(const float* x, const float* w, float* out, int M, int H, float eps, cudaStream_t s);

// Gather embedding rows: out[m,:] = embed[ids[m], :]   (embed BF16 -> out FP32)
void launch_embed(const int* ids, const bf16* embed, float* out, int M, int H, cudaStream_t s);

// In-place rotary position embedding on x[M, n_heads, head_dim], positions pos[M].
void launch_rope(float* x, const int* pos, int M, int n_heads, int head_dim, float theta, cudaStream_t s);

// ---- Paged KV cache attention -----------------------------------------------------------
// The KV pool for a layer is [num_blocks, BLOCK, kv_dim] BF16. A sequence's KV is a list of
// physical block ids — its "block table"; the logical->physical addressing lives in kv_cache.cuh
// (kv_phys_row), and the write side (store) in kv_cache.* . `bt` holds one block table per row
// (stride `max_blocks`); a sequence is identified by its row.

// Prefill attention (M query rows of ONE sequence, block table = row 0 of `bt`) over the paged
// pool. q[M, n_heads, hd] FP32; out[M, n_heads, hd]. Query token m attends keys [0, *d_past + m].
void launch_attention_paged(const float* q, const bf16* cache_k, const bf16* cache_v, float* out,
                            int M, int n_heads, int n_kv, int head_dim, const int* d_past,
                            float scale, const int* bt, int max_blocks, int block_size,
                            cudaStream_t s);

// Batched flash-decoding split-K over the paged pool. Sequence b uses block-table row b of `bt`,
// attending keys [0, past_len[b]]. q/out are [B, n_heads, head_dim]. part_* are scratch sized for
// B*n_heads*ATTN_SPLITS (m/l) and *head_dim (acc).
#define ATTN_SPLITS 16
void launch_attention_decode_paged(const float* q, const bf16* cache_k, const bf16* cache_v,
                                   float* out, int B, int n_heads, int n_kv, int head_dim,
                                   const int* past_len, const int* bt, int max_blocks, int block_size,
                                   float scale, float* part_m, float* part_l, float* part_acc,
                                   cudaStream_t s);

// out[i] += in[i]   for N elements
void launch_add(float* out, const float* in, int N, cudaStream_t s);

// h[i] = silu(gate[i]) * up[i]   for N elements
void launch_silu_mul(const float* gate, const float* up, float* h, int N, cudaStream_t s);

// argmax over logits[N] -> single index in d_out (greedy decode; avoids copying all logits).
void launch_argmax(const float* logits, int N, int* d_out, cudaStream_t s);

// Batched argmax: logits is [B, N]; d_out[b] = argmax over row b. One block per row.
void launch_argmax_batch(const float* logits, int B, int N, int* d_out, cudaStream_t s);
