#include "kernels.cuh"
#include <mma.h>
using namespace nvcuda;

// ---------------------------------------------------------------------------------------
// GEMV (decode, M==1): one warp per output row. Weights are INT8 with a per-row scale; the
// row is dequantized as it's read (acc * scale at the end). Reading 1-byte weights instead
// of 2-byte halves the decode memory traffic. Each lane reads 16 INT8 per pass (16-byte
// load), striding by 32*16. Requires IN % 16 == 0.
// ---------------------------------------------------------------------------------------
__global__ void gemv_kernel(const float* __restrict__ x, const int8_t* __restrict__ W,
                            const float* __restrict__ scale, float* __restrict__ y, int IN, int OUT) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;   // output feature
    if (row >= OUT) return;

    const int8_t* wr = W + (size_t)row * IN;
    float acc = 0.f;
    for (int i = lane * 16; i + 16 <= IN; i += 512) {
        int4 wpack = *reinterpret_cast<const int4*>(wr + i);          // 16 INT8 weights
        const int8_t* wb = reinterpret_cast<const int8_t*>(&wpack);
        const float4* xv = reinterpret_cast<const float4*>(x + i);
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            float4 xx = xv[q];
            acc += xx.x * (float)wb[q*4+0] + xx.y * (float)wb[q*4+1]
                 + xx.z * (float)wb[q*4+2] + xx.w * (float)wb[q*4+3];
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) y[row] = acc * scale[row];
}

// Dequantize an INT8 weight matrix (+ per-row scale) to BF16, for the prefill WMMA path.
__global__ void dequant_kernel(const int8_t* __restrict__ W, const float* __restrict__ scale,
                               bf16* __restrict__ out, int IN, long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __float2bfloat16((float)W[idx] * scale[idx / IN]);
}

// ---------------------------------------------------------------------------------------
// FP32 -> BF16 convert with zero-padding of the tail rows (so the last WMMA tile is clean).
// ---------------------------------------------------------------------------------------
__global__ void f32_to_bf16_kernel(const float* __restrict__ in, bf16* __restrict__ out,
                                   int valid, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = __float2bfloat16(i < valid ? in[i] : 0.0f);
}

// ---------------------------------------------------------------------------------------
// Tensor-core GEMM (prefill, M>1): y[M,OUT] = x[M,IN] @ W[OUT,IN]^T, BF16 in / FP32 acc.
// One warp computes a 16x16 output tile; 8 warps per block. W (row-major [OUT,IN]) is fed
// as the matrix_b operand in col_major with leading dim IN, which is exactly W^T.
// ---------------------------------------------------------------------------------------
__global__ void wmma_kernel(const bf16* __restrict__ x, const bf16* __restrict__ W,
                            float* __restrict__ y, int IN, int OUT) {
    int warp = threadIdx.x >> 5;
    int tile_n = blockIdx.x * 8 + warp;        // output-feature tile (16 wide)
    int tile_m = blockIdx.y;                   // token tile (16 tall)
    if (tile_n * 16 >= OUT) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);

    int m0 = tile_m * 16, n0 = tile_n * 16;
    for (int k0 = 0; k0 < IN; k0 += 16) {
        wmma::load_matrix_sync(a, x + (size_t)m0 * IN + k0, IN);
        wmma::load_matrix_sync(b, W + (size_t)n0 * IN + k0, IN);   // col_major load == W^T
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(y + (size_t)m0 * OUT + n0, c, OUT, wmma::mem_row_major);
}

void launch_matmul(const float* x, const int8_t* W, const float* scale, float* y,
                   int M, int IN, int OUT, bf16* x_bf16, bf16* w_dq, cudaStream_t s) {
    if (M <= 1) {                              // decode: memory-bound INT8 GEMV
        const int warps = 8;
        gemv_kernel<<<(OUT + warps - 1) / warps, warps * 32, 0, s>>>(x, W, scale, y, IN, OUT);
        return;
    }
    // prefill: dequantize the INT8 weights to BF16, convert activations to BF16, then run the
    // tensor-core GEMM. (Dequant cost is amortized — prefill runs once per request.)
    long wn = (long)OUT * IN; int blk = 256;
    dequant_kernel<<<(wn + blk - 1) / blk, blk, 0, s>>>(W, scale, w_dq, IN, wn);

    int Mp = ((M + 15) / 16) * 16;
    int total = Mp * IN, valid = M * IN;
    f32_to_bf16_kernel<<<(total + blk - 1) / blk, blk, 0, s>>>(x, x_bf16, valid, total);

    dim3 block(256);                           // 8 warps
    dim3 grid((OUT / 16 + 7) / 8, Mp / 16);
    wmma_kernel<<<grid, block, 0, s>>>(x_bf16, w_dq, y, IN, OUT);
}

// ---------------------------------------------------------------------------------------
// Batched decode GEMV: y[B,OUT] = x[B,IN] @ W[OUT,IN]^T. One warp per output row reads the
// INT8 weight row ONCE (16-byte loads) and computes all B dot products, so weight traffic is
// amortized across the batch — this is what makes batched decode faster per token than B
// separate single-token GEMVs. B is a compile-time template so the acc[] array stays in
// registers (a runtime-bounded array would spill to local memory). Requires IN % 16 == 0.
// ---------------------------------------------------------------------------------------
template<int B>
__global__ void gemv_batch_kernel(const float* __restrict__ x, const int8_t* __restrict__ W,
                                  const float* __restrict__ scale, float* __restrict__ y,
                                  int IN, int OUT) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= OUT) return;

    const int8_t* wr = W + (size_t)row * IN;
    float acc[B];
    #pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.f;

    for (int i = lane * 16; i + 16 <= IN; i += 512) {
        int4 wpack = *reinterpret_cast<const int4*>(wr + i);     // 16 INT8 weights, read once
        const int8_t* wb = reinterpret_cast<const int8_t*>(&wpack);
        #pragma unroll
        for (int b = 0; b < B; ++b) {
            const float4* xv = reinterpret_cast<const float4*>(x + (size_t)b * IN + i);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                float4 xx = xv[q];
                acc[b] += xx.x * (float)wb[q*4+0] + xx.y * (float)wb[q*4+1]
                        + xx.z * (float)wb[q*4+2] + xx.w * (float)wb[q*4+3];
            }
        }
    }
    #pragma unroll
    for (int b = 0; b < B; ++b) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[b] += __shfl_down_sync(0xffffffff, acc[b], o);
        if (lane == 0) y[(size_t)b * OUT + row] = acc[b] * scale[row];
    }
}

void launch_matmul_decode(const float* x, const int8_t* W, const float* scale, float* y,
                          int B, int IN, int OUT, cudaStream_t s) {
    const int warps = 8;
    int blocks = (OUT + warps - 1) / warps, th = warps * 32;
    // Round B up to the nearest instantiated template size; extra rows compute from (unused)
    // activation rows and their outputs are ignored by the caller.
    #define LAUNCH_GEMV_B(T) gemv_batch_kernel<T><<<blocks, th, 0, s>>>(x, W, scale, y, IN, OUT)
    if      (B <= 1)  LAUNCH_GEMV_B(1);
    else if (B <= 2)  LAUNCH_GEMV_B(2);
    else if (B <= 4)  LAUNCH_GEMV_B(4);
    else if (B <= 8)  LAUNCH_GEMV_B(8);
    else if (B <= 16) LAUNCH_GEMV_B(16);
    else              LAUNCH_GEMV_B(32);
    #undef LAUNCH_GEMV_B
}

// ---------------------------------------------------------------------------------------
// rmsnorm: one block per row, block-reduction over H.
// ---------------------------------------------------------------------------------------
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                               float* __restrict__ out, int M, int H, float eps) {
    int m = blockIdx.x;
    const float* xr = x + (size_t)m * H;
    float* outr = out + (size_t)m * H;

    extern __shared__ float red[];
    float local = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) local += xr[i] * xr[i];
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        outr[i] = xr[i] * inv * w[i];
}

void launch_rmsnorm(const float* x, const float* w, float* out, int M, int H, float eps, cudaStream_t s) {
    int block = 256;
    rmsnorm_kernel<<<M, block, block * sizeof(float), s>>>(x, w, out, M, H, eps);
}

// ---------------------------------------------------------------------------------------
// embedding gather
// ---------------------------------------------------------------------------------------
__global__ void embed_kernel(const int* __restrict__ ids, const bf16* __restrict__ embed,
                             float* __restrict__ out, int M, int H) {
    int m = blockIdx.x;
    int id = ids[m];
    const bf16* src = embed + (size_t)id * H;
    float* dst = out + (size_t)m * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        dst[i] = __bfloat162float(src[i]);
}

void launch_embed(const int* ids, const bf16* embed, float* out, int M, int H, cudaStream_t s) {
    embed_kernel<<<M, 256, 0, s>>>(ids, embed, out, M, H);
}

// ---------------------------------------------------------------------------------------
// RoPE (rotate-half convention, matching HF Qwen)
// ---------------------------------------------------------------------------------------
__global__ void rope_kernel(float* __restrict__ x, const int* __restrict__ pos,
                            int M, int n_heads, int head_dim, float theta) {
    int idx = blockIdx.x;            // over M * n_heads
    int m = idx / n_heads;
    int i = threadIdx.x;             // over head_dim/2
    int half = head_dim >> 1;
    if (i >= half) return;

    float* v = x + (size_t)idx * head_dim;
    float inv = powf(theta, -2.0f * i / head_dim);
    float ang = pos[m] * inv;
    float cs = cosf(ang), sn = sinf(ang);
    float x1 = v[i], x2 = v[i + half];
    v[i]        = x1 * cs - x2 * sn;
    v[i + half] = x2 * cs + x1 * sn;
}

void launch_rope(float* x, const int* pos, int M, int n_heads, int head_dim, float theta, cudaStream_t s) {
    rope_kernel<<<M * n_heads, head_dim / 2, 0, s>>>(x, pos, M, n_heads, head_dim, theta);
}

// ---------------------------------------------------------------------------------------
// Paged KV cache: a layer's pool is [num_blocks, BLOCK, kv_dim]; a sequence's KV is a list of
// block ids (its block table). Logical position p -> physical row bt[p/BLOCK]*BLOCK + p%BLOCK,
// addressed as cache + row*kv_dim. The math below is identical to the contiguous version; only
// the per-token / per-key row lookup now goes through the block table.
// ---------------------------------------------------------------------------------------

// Store M new K (or V) rows into the paged pool. Token m -> block-table row bt_row[m], logical
// position pos[m].
__global__ void store_kv_paged_kernel(const float* __restrict__ src, bf16* __restrict__ cache,
                                      const int* __restrict__ bt, int max_blocks, int block_size,
                                      int kv_dim, const int* __restrict__ bt_row,
                                      const int* __restrict__ pos, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * kv_dim) return;
    int m = idx / kv_dim, i = idx % kv_dim;
    int p = pos[m], g = p / block_size, off = p - g * block_size;
    size_t phys = (size_t)bt[(size_t)bt_row[m] * max_blocks + g] * block_size + off;
    cache[phys * kv_dim + i] = __float2bfloat16(src[idx]);
}

void launch_store_kv_paged(const float* src, bf16* cache, const int* bt, int max_blocks,
                           int block_size, int kv_dim, const int* bt_row, const int* pos,
                           int M, cudaStream_t s) {
    int n = M * kv_dim, blk = 256;
    store_kv_paged_kernel<<<(n + blk - 1) / blk, blk, 0, s>>>(
        src, cache, bt, max_blocks, block_size, kv_dim, bt_row, pos, M);
}

// Prefill attention — one WARP per (head, query token), online softmax over the paged KV. Each
// lane owns head_dim/32 dims; the per-key q.k dot is a warp-shuffle reduction (no __syncthreads).
__global__ void attention_paged_kernel(const float* __restrict__ q, const bf16* __restrict__ cache_k,
                                       const bf16* __restrict__ cache_v, float* __restrict__ out,
                                       int M, int n_heads, int n_kv, int head_dim,
                                       const int* __restrict__ d_past, float scale,
                                       const int* __restrict__ bt, int max_blocks, int block_size) {
    constexpr int DPL = 4;
    int h = blockIdx.x;              // query head
    int m = blockIdx.y;              // query token
    int lane = threadIdx.x;          // 0..31
    int kvh = h / (n_heads / n_kv);
    int kv_dim = n_kv * head_dim;
    int qpos = *d_past + m;          // attend keys [0, qpos]

    const float* qv = q + ((size_t)m * n_heads + h) * head_dim;
    float qreg[DPL], acc[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) { qreg[i] = qv[lane + (i << 5)]; acc[i] = 0.f; }
    float m_run = -1e30f, l_run = 0.f;

    for (int j = 0; j <= qpos; ++j) {
        int g = j / block_size, off = j - g * block_size;
        size_t phys = (size_t)bt[g] * block_size + off;          // block table row 0
        const bf16* kj = cache_k + phys * kv_dim + kvh * head_dim;
        float partial = 0.f;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) partial += qreg[i] * __bfloat162float(kj[lane + (i << 5)]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, o);
        float score = partial * scale;          // identical on all lanes

        float m_new = fmaxf(m_run, score);
        float corr  = __expf(m_run - m_new);
        float p     = __expf(score - m_new);
        l_run = l_run * corr + p;
        const bf16* vj = cache_v + phys * kv_dim + kvh * head_dim;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) acc[i] = acc[i] * corr + p * __bfloat162float(vj[lane + (i << 5)]);
        m_run = m_new;
    }
    float inv = 1.0f / l_run;
    float* o = out + ((size_t)m * n_heads + h) * head_dim;
    #pragma unroll
    for (int i = 0; i < DPL; ++i) o[lane + (i << 5)] = acc[i] * inv;
}

void launch_attention_paged(const float* q, const bf16* cache_k, const bf16* cache_v, float* out,
                            int M, int n_heads, int n_kv, int head_dim, const int* d_past,
                            float scale, const int* bt, int max_blocks, int block_size,
                            cudaStream_t s) {
    attention_paged_kernel<<<dim3(n_heads, M), 32, 0, s>>>(
        q, cache_k, cache_v, out, M, n_heads, n_kv, head_dim, d_past, scale, bt, max_blocks, block_size);
}

// Batched flash-decoding phase 1: block (h, split, b) does a partial online-softmax over its key
// chunk of sequence b (block-table row b), attending keys [0, past_len[b]].
__global__ void attn_decode_split_paged_kernel(
        const float* __restrict__ q, const bf16* __restrict__ cache_k, const bf16* __restrict__ cache_v,
        float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
        int n_heads, int n_kv, int head_dim, const int* __restrict__ past_len,
        const int* __restrict__ bt, int max_blocks, int block_size, float scale) {
    constexpr int DPL = 4;
    int h = blockIdx.x, sp = blockIdx.y, b = blockIdx.z, lane = threadIdx.x;
    int kvh = h / (n_heads / n_kv);
    int kv_dim = n_kv * head_dim;
    int L = past_len[b] + 1;                        // keys [0, past_len[b]]
    int per = (L + ATTN_SPLITS - 1) / ATTN_SPLITS;
    int kb = sp * per, ke = min(kb + per, L);

    const int* btb = bt + (size_t)b * max_blocks;   // this sequence's block table
    const float* qv = q + ((size_t)b * n_heads + h) * head_dim;
    float qreg[DPL], acc[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) { qreg[i] = qv[lane + (i << 5)]; acc[i] = 0.f; }
    float m_run = -1e30f, l_run = 0.f;

    for (int j = kb; j < ke; ++j) {
        int g = j / block_size, off = j - g * block_size;
        size_t phys = (size_t)btb[g] * block_size + off;
        const bf16* kj = cache_k + phys * kv_dim + kvh * head_dim;
        float p = 0.f;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) p += qreg[i] * __bfloat162float(kj[lane + (i << 5)]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffff, p, o);
        float score = p * scale;
        float m_new = fmaxf(m_run, score);
        float corr = __expf(m_run - m_new), pp = __expf(score - m_new);
        l_run = l_run * corr + pp;
        const bf16* vj = cache_v + phys * kv_dim + kvh * head_dim;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) acc[i] = acc[i] * corr + pp * __bfloat162float(vj[lane + (i << 5)]);
        m_run = m_new;
    }
    int slot_idx = ((size_t)b * n_heads + h) * ATTN_SPLITS + sp;
    if (lane == 0) { part_m[slot_idx] = m_run; part_l[slot_idx] = l_run; }
    #pragma unroll
    for (int i = 0; i < DPL; ++i) part_acc[(size_t)slot_idx * head_dim + lane + (i << 5)] = acc[i];
}

// Batched flash-decoding phase 2: one warp per (b, head) merges the ATTN_SPLITS partials.
__global__ void attn_decode_combine_paged_kernel(
        const float* __restrict__ part_m, const float* __restrict__ part_l,
        const float* __restrict__ part_acc, float* __restrict__ out, int n_heads, int head_dim) {
    constexpr int DPL = 4;
    int h = blockIdx.x, b = blockIdx.y, lane = threadIdx.x;
    int base = ((size_t)b * n_heads + h) * ATTN_SPLITS;
    float gm = -1e30f;
    #pragma unroll
    for (int sp = 0; sp < ATTN_SPLITS; ++sp) gm = fmaxf(gm, part_m[base + sp]);
    float gl = 0.f, gacc[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) gacc[i] = 0.f;
    #pragma unroll
    for (int sp = 0; sp < ATTN_SPLITS; ++sp) {
        int slot_idx = base + sp;
        float w = __expf(part_m[slot_idx] - gm);
        gl += part_l[slot_idx] * w;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) gacc[i] += part_acc[(size_t)slot_idx * head_dim + lane + (i << 5)] * w;
    }
    float inv = 1.0f / gl;
    float* o = out + ((size_t)b * n_heads + h) * head_dim;
    #pragma unroll
    for (int i = 0; i < DPL; ++i) o[lane + (i << 5)] = gacc[i] * inv;
}

void launch_attention_decode_paged(const float* q, const bf16* cache_k, const bf16* cache_v,
                                   float* out, int B, int n_heads, int n_kv, int head_dim,
                                   const int* past_len, const int* bt, int max_blocks, int block_size,
                                   float scale, float* part_m, float* part_l, float* part_acc,
                                   cudaStream_t s) {
    attn_decode_split_paged_kernel<<<dim3(n_heads, ATTN_SPLITS, B), 32, 0, s>>>(
        q, cache_k, cache_v, part_m, part_l, part_acc, n_heads, n_kv, head_dim,
        past_len, bt, max_blocks, block_size, scale);
    attn_decode_combine_paged_kernel<<<dim3(n_heads, B), 32, 0, s>>>(
        part_m, part_l, part_acc, out, n_heads, head_dim);
}

// ---------------------------------------------------------------------------------------
// elementwise helpers
// ---------------------------------------------------------------------------------------
__global__ void add_kernel(float* out, const float* in, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] += in[i];
}
void launch_add(float* out, const float* in, int N, cudaStream_t s) {
    int block = 256;
    add_kernel<<<(N + block - 1) / block, block, 0, s>>>(out, in, N);
}

__global__ void silu_mul_kernel(const float* gate, const float* up, float* h, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float g = gate[i];
        h[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}
void launch_silu_mul(const float* gate, const float* up, float* h, int N, cudaStream_t s) {
    int block = 256;
    silu_mul_kernel<<<(N + block - 1) / block, block, 0, s>>>(gate, up, h, N);
}

// argmax over N logits in a single block (256 threads), result index in *out.
__global__ void argmax_kernel(const float* __restrict__ logits, int N, int* __restrict__ out) {
    __shared__ float sval[256];
    __shared__ int   sidx[256];
    int tid = threadIdx.x;
    float best = -1e30f; int bi = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = logits[i];
        if (v > best) { best = v; bi = i; }
    }
    sval[tid] = best; sidx[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && sval[tid + s] > sval[tid]) { sval[tid] = sval[tid + s]; sidx[tid] = sidx[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) *out = sidx[0];
}

void launch_argmax(const float* logits, int N, int* d_out, cudaStream_t s) {
    argmax_kernel<<<1, 256, 0, s>>>(logits, N, d_out);
}

// Batched argmax: one block per row of logits[B, N]; d_out[b] = argmax over row b.
__global__ void argmax_batch_kernel(const float* __restrict__ logits, int N, int* __restrict__ out) {
    __shared__ float sval[256];
    __shared__ int   sidx[256];
    int b = blockIdx.x, tid = threadIdx.x;
    const float* lg = logits + (size_t)b * N;
    float best = -1e30f; int bi = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = lg[i];
        if (v > best) { best = v; bi = i; }
    }
    sval[tid] = best; sidx[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && sval[tid + s] > sval[tid]) { sval[tid] = sval[tid + s]; sidx[tid] = sidx[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) out[b] = sidx[0];
}

void launch_argmax_batch(const float* logits, int B, int N, int* d_out, cudaStream_t s) {
    argmax_batch_kernel<<<B, 256, 0, s>>>(logits, N, d_out);
}

