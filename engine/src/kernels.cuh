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

// Fused residual-add + rmsnorm: x[m] += res[m] (written back, carries the residual forward), then
// out[m] = rmsnorm(x[m]) * w. One launch + one pass over x replaces a separate add then rmsnorm
// (saves a full H read/write of x and a kernel launch per residual).
void launch_add_rmsnorm(bf16* x, const bf16* res, const float* w, bf16* out,
                        int M, int H, float eps, cudaStream_t s);

// Gather embedding rows: out[m,:] = embed[ids[m], :]   (embed + out BF16)
void launch_embed(const int* ids, const bf16* embed, bf16* out, int M, int H, cudaStream_t s);

// Fused per-head RMSNorm + RoPE over one sub-tensor of a fused QKV buffer. `buf` points at the q (or
// k) slice of a row of width `stride`; head (m,h) lives at buf + m*stride + h*head_dim. Each head row
// is RMSNorm'd (weight w, FP32) then rotated using the precomputed cos/sin tables (cos/sin laid out
// [max_pos, head_dim/2]) indexed by pos[m]. In place. blockDim must be head_dim.
void launch_head_norm_rope(bf16* buf, const float* w, const float* cos_tab, const float* sin_tab,
                           const int* pos, int M, int n_heads, int head_dim, int stride, float eps,
                           cudaStream_t s);

// ---- Paged-KV attention -----------------------------------------------------------------
// The KV pool for a layer is [num_blocks, BLOCK, kv_dim] BF16; a sequence's KV is a list of physical
// block ids (its "block table") — logical->physical addressing lives in kv_cache.cuh (kv_phys_row),
// the write side in block_pool.*. `bt` holds one block table per request (stride `max_blocks`).
//
// Attention is split by request type (run_layers dispatches each on its subset): prefill rows
// (q_len>1) and decode rows (q_len==1). Shared contract for both: q is read from the fused-QKV buffer
// (row stride `q_stride`), out is [T, n_heads, head_dim]; request r owns rows [qstart[r], +qlen[r]) at
// absolute positions pos[], attending keys [0, pos] (causal); `rids[grid_slot]` -> actual request id
// so each kernel runs on its request subset (R / n_decode = number of entries).

// Prefill (q_len>1): tensor-core WMMA FlashAttention-2 (see kernels.cu). Requires head_dim==128 and
// block_size==16 (the engine's only layout: Qwen3 + BlockPool::BLOCK). max_qlen sets the q-tile grid.
void launch_attn_prefill(const bf16* q, int q_stride, const bf16* cache_k, const bf16* cache_v,
                         bf16* out, int n_heads, int n_kv, int head_dim,
                         const int* pos, const int* qstart, const int* qlen,
                         const int* rids, int R, int max_qlen,
                         const int* bt, int max_blocks, int block_size, float scale,
                         cudaStream_t s);

// Decode (q_len==1): FlashDecoding — one block per (head, decode-request), warps split the request's
// KV range and online-softmax their slices reading K/V straight from the cache (no staging: a single
// query has no reuse), then an in-block combine merges the per-warp partials.
void launch_attn_decode(const bf16* q, int q_stride, const bf16* cache_k, const bf16* cache_v,
                        bf16* out, int n_heads, int n_kv, int head_dim,
                        const int* pos, const int* qstart, const int* decode_rids, int n_decode,
                        const int* bt, int max_blocks, int block_size, float scale, cudaStream_t s);

// Gather S rows from x[*, H] into out[S, H]: out[i] = x[rows[i]] (BF16). Used to pull the per-request
// "last token" rows out of the flattened batch before the final norm + lm_head.
void launch_gather_rows(const bf16* x, const int* rows, bf16* out, int S, int H, cudaStream_t s);

// out[i] += in[i]   for N elements (BF16, accumulated in FP32)
void launch_add(bf16* out, const bf16* in, int N, cudaStream_t s);

// h[m,i] = silu(gateup[m, i]) * gateup[m, I + i]  over M rows of a fused gate|up buffer (row width 2I).
void launch_silu_mul(const bf16* gateup, bf16* h, int M, int I, cudaStream_t s);

// Batched sampling: logits is [B, N] FP32; out[b] = a sampled token id over row b. One block per row.
//   invT[b] : 1/temperature for row b, or <= 0 to take the greedy argmax (ignores u/topp)
//   topp[b] : nucleus cutoff (>= 1 means no truncation; < 1 restricts to the top-p nucleus)
//   u[b]    : a uniform(0,1) draw for row b (used by the stochastic paths)
void launch_sample_batch(const float* logits, int B, int N,
                         const float* invT, const float* topp, const float* u,
                         int* out, cudaStream_t s);
