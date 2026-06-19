// CUDA kernels for the Qwen3 forward pass.
//
// Convention: weights AND activations are BF16 (no quantization). Matmuls go through cuBLAS
// (cublasGemmEx, BF16 in / FP32 accumulate); the elementwise / attention kernels below are BF16
// in / out with FP32 math internally. Linear layers follow the HF layout y = x @ W^T with W shaped
// [out_features, in_features].
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

// Max concurrent sequences (sizes the sampling / block-table buffers). One logits row per request.
constexpr int MAX_DECODE_B = 32;

// out[M,H] = rmsnorm(x[M,H]) * w[H]    (x/out BF16, weight w FP32, reduction in FP32)
void launch_rmsnorm(const bf16* x, const float* w, bf16* out, int M, int H, float eps, cudaStream_t s);

// Gather embedding rows: out[m,:] = embed[ids[m], :]   (embed + out BF16)
void launch_embed(const int* ids, const bf16* embed, bf16* out, int M, int H, cudaStream_t s);

// In-place rotary position embedding on x[M, n_heads, head_dim] (BF16), positions pos[M].
void launch_rope(bf16* x, const int* pos, int M, int n_heads, int head_dim, float theta, cudaStream_t s);

// ---- Paged KV cache attention -----------------------------------------------------------
// The KV pool for a layer is [num_blocks, BLOCK, kv_dim] BF16. A sequence's KV is a list of physical
// block ids — its "block table"; the logical->physical addressing lives in kv_cache.cuh
// (kv_phys_row), and the write side (store) in block_pool.* . `bt` holds one block table per request
// (stride `max_blocks`).
//
// FlashAttention-2 style paged varlen attention over the whole mixed batch (prefill chunks + decodes
// together). q/out are [T, n_heads, head_dim] BF16 (T = total query rows). The batch is grouped by
// request: request r owns the contiguous rows [qstart[r], qstart[r]+qlen[r]); each of its rows m is
// at absolute position pos[m] and attends that request's keys [0, pos[m]] (causal). One thread block
// handles (query-tile, head, request): BM query rows (one warp each) stream the request's K/V in
// BLOCK-sized tiles through shared memory — reused across all BM rows — with online softmax in
// registers and deferred normalization. max_qlen = max over requests of qlen (sets the q-tile grid).
void launch_attention_flash(const bf16* q, const bf16* cache_k, const bf16* cache_v, bf16* out,
                            int n_heads, int n_kv, int head_dim,
                            const int* pos, const int* qstart, const int* qlen,
                            int R, int max_qlen,
                            const int* bt, int max_blocks, int block_size, float scale,
                            cudaStream_t s);

// Gather S rows from x[*, H] into out[S, H]: out[i] = x[rows[i]] (BF16). Used to pull the per-request
// "last token" rows out of the flattened batch before the final norm + lm_head.
void launch_gather_rows(const bf16* x, const int* rows, bf16* out, int S, int H, cudaStream_t s);

// out[i] += in[i]   for N elements (BF16, accumulated in FP32)
void launch_add(bf16* out, const bf16* in, int N, cudaStream_t s);

// h[i] = silu(gate[i]) * up[i]   for N elements (BF16)
void launch_silu_mul(const bf16* gate, const bf16* up, bf16* h, int N, cudaStream_t s);

// Batched sampling: logits is [B, N] FP32; out[b] = a sampled token id over row b. One block per row.
//   invT[b] : 1/temperature for row b, or <= 0 to take the greedy argmax (ignores u/topp)
//   topp[b] : nucleus cutoff (>= 1 means no truncation; < 1 restricts to the top-p nucleus)
//   u[b]    : a uniform(0,1) draw for row b (used by the stochastic paths)
void launch_sample_batch(const float* logits, int B, int N,
                         const float* invT, const float* topp, const float* u,
                         int* out, cudaStream_t s);
