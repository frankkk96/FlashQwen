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
// attention with online softmax — one WARP per (head, query token). Each lane owns
// head_dim/32 dims; the per-key q.k dot product is a warp-shuffle reduction (no
// __syncthreads, so no serialized per-key barrier). This keeps the per-key cost tiny, so
// TPOT no longer balloons as the KV cache grows. blockDim = 32, head_dim a multiple of 32.
// ---------------------------------------------------------------------------------------
__global__ void attention_kernel(const float* __restrict__ q, const bf16* __restrict__ cache_k,
                                 const bf16* __restrict__ cache_v, float* __restrict__ out,
                                 int M, int n_heads, int n_kv, int head_dim,
                                 const int* __restrict__ d_past, float scale) {
    // DPL = head_dim/32, fixed at compile time (Qwen3 head_dim is 128) so qreg/acc stay in
    // registers instead of spilling to local memory.
    constexpr int DPL = 4;
    int h = blockIdx.x;              // query head
    int m = blockIdx.y;              // query token
    int lane = threadIdx.x;          // 0..31
    int kvh = h / (n_heads / n_kv);
    int qpos = *d_past + m;          // attend keys [0, qpos]

    const float* qv = q + ((size_t)m * n_heads + h) * head_dim;
    float qreg[DPL], acc[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) { qreg[i] = qv[lane + (i << 5)]; acc[i] = 0.f; }
    float m_run = -1e30f, l_run = 0.f;

    for (int j = 0; j <= qpos; ++j) {
        const bf16* kj = cache_k + ((size_t)j * n_kv + kvh) * head_dim;
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
        const bf16* vj = cache_v + ((size_t)j * n_kv + kvh) * head_dim;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) acc[i] = acc[i] * corr + p * __bfloat162float(vj[lane + (i << 5)]);
        m_run = m_new;
    }
    float inv = 1.0f / l_run;
    float* o = out + ((size_t)m * n_heads + h) * head_dim;
    #pragma unroll
    for (int i = 0; i < DPL; ++i) o[lane + (i << 5)] = acc[i] * inv;
}

void launch_attention(const float* q, const bf16* cache_k, const bf16* cache_v,
                      float* out, int M, int n_heads, int n_kv, int head_dim,
                      const int* d_past, float scale, cudaStream_t s) {
    dim3 grid(n_heads, M);
    attention_kernel<<<grid, 32, 0, s>>>(
        q, cache_k, cache_v, out, M, n_heads, n_kv, head_dim, d_past, scale);
}

// Append K/V rows to the BF16 cache at device-resident offset *d_past (FP32 -> BF16).
__global__ void store_kv_kernel(const float* __restrict__ src, bf16* __restrict__ cache,
                                const int* __restrict__ d_past, int kv_dim, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * kv_dim) return;
    int m = idx / kv_dim, i = idx % kv_dim;
    cache[((size_t)(*d_past + m)) * kv_dim + i] = __float2bfloat16(src[idx]);
}

void launch_store_kv(const float* src, bf16* cache, const int* d_past, int kv_dim, int M, cudaStream_t s) {
    int n = M * kv_dim, blk = 256;
    store_kv_kernel<<<(n + blk - 1) / blk, blk, 0, s>>>(src, cache, d_past, kv_dim, M);
}

// Batched store into the [B_max, max_ctx, kv_dim] pool: token m -> slot[m], row rowpos[m].
__global__ void store_kv_batch_kernel(const float* __restrict__ src, bf16* __restrict__ cache,
                                      const int* __restrict__ slot, const int* __restrict__ rowpos,
                                      int kv_dim, int max_ctx, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * kv_dim) return;
    int m = idx / kv_dim, i = idx % kv_dim;
    size_t row = (size_t)slot[m] * max_ctx + rowpos[m];
    cache[row * kv_dim + i] = __float2bfloat16(src[idx]);
}

void launch_store_kv_batch(const float* src, bf16* cache, const int* slot, const int* rowpos,
                           int kv_dim, int max_ctx, int M, cudaStream_t s) {
    int n = M * kv_dim, blk = 256;
    store_kv_batch_kernel<<<(n + blk - 1) / blk, blk, 0, s>>>(src, cache, slot, rowpos, kv_dim, max_ctx, M);
}

// Flash-decoding phase 1: warp (h, split) does a partial online-softmax over its key chunk.
// Stores UN-normalized acc plus (m, l) so the combine pass can merge chunks.
__global__ void attn_decode_split_kernel(const float* __restrict__ q, const bf16* __restrict__ cache_k,
                                         const bf16* __restrict__ cache_v, float* __restrict__ part_m,
                                         float* __restrict__ part_l, float* __restrict__ part_acc,
                                         int n_heads, int n_kv, int head_dim, const int* __restrict__ d_past,
                                         float scale) {
    constexpr int DPL = 4;
    int h = blockIdx.x, sp = blockIdx.y, lane = threadIdx.x;
    int kvh = h / (n_heads / n_kv);
    int L = *d_past + 1;                          // keys [0, *d_past]
    int per = (L + ATTN_SPLITS - 1) / ATTN_SPLITS;
    int kb = sp * per, ke = min(kb + per, L);

    const float* qv = q + (size_t)h * head_dim;   // M=1
    float qreg[DPL], acc[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) { qreg[i] = qv[lane + (i << 5)]; acc[i] = 0.f; }
    float m_run = -1e30f, l_run = 0.f;

    for (int j = kb; j < ke; ++j) {
        const bf16* kj = cache_k + ((size_t)j * n_kv + kvh) * head_dim;
        float p = 0.f;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) p += qreg[i] * __bfloat162float(kj[lane + (i << 5)]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffff, p, o);
        float score = p * scale;
        float m_new = fmaxf(m_run, score);
        float corr = __expf(m_run - m_new), pp = __expf(score - m_new);
        l_run = l_run * corr + pp;
        const bf16* vj = cache_v + ((size_t)j * n_kv + kvh) * head_dim;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) acc[i] = acc[i] * corr + pp * __bfloat162float(vj[lane + (i << 5)]);
        m_run = m_new;
    }
    int slot = h * ATTN_SPLITS + sp;
    if (lane == 0) { part_m[slot] = m_run; part_l[slot] = l_run; }
    #pragma unroll
    for (int i = 0; i < DPL; ++i) part_acc[(size_t)slot * head_dim + lane + (i << 5)] = acc[i];
}

// Flash-decoding phase 2: one warp per head merges the ATTN_SPLITS partials.
__global__ void attn_decode_combine_kernel(const float* __restrict__ part_m, const float* __restrict__ part_l,
                                           const float* __restrict__ part_acc, float* __restrict__ out,
                                           int head_dim) {
    constexpr int DPL = 4;
    int h = blockIdx.x, lane = threadIdx.x;
    float gm = -1e30f;
    #pragma unroll
    for (int sp = 0; sp < ATTN_SPLITS; ++sp) gm = fmaxf(gm, part_m[h * ATTN_SPLITS + sp]);
    float gl = 0.f, gacc[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) gacc[i] = 0.f;
    #pragma unroll
    for (int sp = 0; sp < ATTN_SPLITS; ++sp) {
        int slot = h * ATTN_SPLITS + sp;
        float w = __expf(part_m[slot] - gm);
        gl += part_l[slot] * w;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) gacc[i] += part_acc[(size_t)slot * head_dim + lane + (i << 5)] * w;
    }
    float inv = 1.0f / gl;
    float* o = out + (size_t)h * head_dim;
    #pragma unroll
    for (int i = 0; i < DPL; ++i) o[lane + (i << 5)] = gacc[i] * inv;
}

void launch_attention_decode(const float* q, const bf16* cache_k, const bf16* cache_v,
                             float* out, int n_heads, int n_kv, int head_dim, const int* d_past,
                             float scale, float* part_m, float* part_l, float* part_acc,
                             cudaStream_t s) {
    attn_decode_split_kernel<<<dim3(n_heads, ATTN_SPLITS), 32, 0, s>>>(
        q, cache_k, cache_v, part_m, part_l, part_acc, n_heads, n_kv, head_dim, d_past, scale);
    attn_decode_combine_kernel<<<n_heads, 32, 0, s>>>(part_m, part_l, part_acc, out, head_dim);
}

// Batched flash-decoding phase 1: block (h, split, b) does a partial online-softmax over its
// key chunk of sequence b (slot[b]), attending keys [0, past_len[b]].
__global__ void attn_decode_split_batch_kernel(
        const float* __restrict__ q, const bf16* __restrict__ cache_k, const bf16* __restrict__ cache_v,
        float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
        int n_heads, int n_kv, int head_dim, const int* __restrict__ slot,
        const int* __restrict__ past_len, int max_ctx, float scale) {
    constexpr int DPL = 4;
    int h = blockIdx.x, sp = blockIdx.y, b = blockIdx.z, lane = threadIdx.x;
    int kvh = h / (n_heads / n_kv);
    int kv_dim = n_kv * head_dim;
    int L = past_len[b] + 1;                       // keys [0, past_len[b]]
    int per = (L + ATTN_SPLITS - 1) / ATTN_SPLITS;
    int kb = sp * per, ke = min(kb + per, L);

    const bf16* ck = cache_k + (size_t)slot[b] * max_ctx * kv_dim;
    const bf16* cv = cache_v + (size_t)slot[b] * max_ctx * kv_dim;
    const float* qv = q + ((size_t)b * n_heads + h) * head_dim;
    float qreg[DPL], acc[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) { qreg[i] = qv[lane + (i << 5)]; acc[i] = 0.f; }
    float m_run = -1e30f, l_run = 0.f;

    for (int j = kb; j < ke; ++j) {
        const bf16* kj = ck + ((size_t)j * n_kv + kvh) * head_dim;
        float p = 0.f;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) p += qreg[i] * __bfloat162float(kj[lane + (i << 5)]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffff, p, o);
        float score = p * scale;
        float m_new = fmaxf(m_run, score);
        float corr = __expf(m_run - m_new), pp = __expf(score - m_new);
        l_run = l_run * corr + pp;
        const bf16* vj = cv + ((size_t)j * n_kv + kvh) * head_dim;
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
__global__ void attn_decode_combine_batch_kernel(
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

void launch_attention_decode_batch(const float* q, const bf16* cache_k, const bf16* cache_v,
                                   float* out, int B, int n_heads, int n_kv, int head_dim,
                                   const int* slot, const int* past_len, int max_ctx, float scale,
                                   float* part_m, float* part_l, float* part_acc, cudaStream_t s) {
    attn_decode_split_batch_kernel<<<dim3(n_heads, ATTN_SPLITS, B), 32, 0, s>>>(
        q, cache_k, cache_v, part_m, part_l, part_acc, n_heads, n_kv, head_dim, slot, past_len, max_ctx, scale);
    attn_decode_combine_batch_kernel<<<dim3(n_heads, B), 32, 0, s>>>(
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

// ---------------------------------------------------------------------------------------
// Fused lm_head + batched argmax. Phase 1: like the batched GEMV (each warp reads a weight
// row once, computes all B dot products), but instead of writing logits it keeps a per-(b)
// running argmax over the rows this block handled, and writes one partial (val,idx) per b per
// block. Phase 2 reduces the LM_ARGMAX_BLOCKS partials per b to the final argmax. Weight is
// read exactly once for the whole batch; the [B,OUT] logits are never materialized.
// ---------------------------------------------------------------------------------------
template<int B>
__global__ void lm_head_argmax_partial_kernel(
        const float* __restrict__ x, const int8_t* __restrict__ W, const float* __restrict__ scale,
        int IN, int OUT, float* __restrict__ part_val, int* __restrict__ part_idx, int nblocks) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NW = 8;                                   // blockDim 256 -> 8 warps
    float bv[B]; int bi[B];
    #pragma unroll
    for (int b = 0; b < B; ++b) { bv[b] = -1e30f; bi[b] = 0; }

    for (int row = blockIdx.x * NW + warp; row < OUT; row += gridDim.x * NW) {
        const int8_t* wr = W + (size_t)row * IN;
        float acc[B];
        #pragma unroll
        for (int b = 0; b < B; ++b) acc[b] = 0.f;
        for (int i = lane * 16; i + 16 <= IN; i += 512) {
            int4 wpack = *reinterpret_cast<const int4*>(wr + i);
            const int8_t* wb = reinterpret_cast<const int8_t*>(&wpack);
            #pragma unroll
            for (int b = 0; b < B; ++b) {
                const float4* xv = reinterpret_cast<const float4*>(x + (size_t)b * IN + i);
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    float4 xx = xv[q];
                    acc[b] += xx.x*(float)wb[q*4+0] + xx.y*(float)wb[q*4+1]
                            + xx.z*(float)wb[q*4+2] + xx.w*(float)wb[q*4+3];
                }
            }
        }
        #pragma unroll
        for (int b = 0; b < B; ++b) {
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[b] += __shfl_down_sync(0xffffffff, acc[b], o);
        }
        if (lane == 0) {
            float sc = scale[row];
            #pragma unroll
            for (int b = 0; b < B; ++b) { float v = acc[b] * sc; if (v > bv[b]) { bv[b] = v; bi[b] = row; } }
        }
    }

    __shared__ float sv[NW * B];
    __shared__ int   si[NW * B];
    if (lane == 0) {
        #pragma unroll
        for (int b = 0; b < B; ++b) { sv[warp * B + b] = bv[b]; si[warp * B + b] = bi[b]; }
    }
    __syncthreads();
    if (threadIdx.x < B) {                              // one thread per sequence reduces 8 warps
        int b = threadIdx.x;
        float best = sv[b]; int idx = si[b];
        #pragma unroll
        for (int w = 1; w < NW; ++w) { int o = w * B + b; if (sv[o] > best) { best = sv[o]; idx = si[o]; } }
        part_val[(size_t)b * nblocks + blockIdx.x] = best;
        part_idx[(size_t)b * nblocks + blockIdx.x] = idx;
    }
}

__global__ void lm_head_argmax_combine_kernel(const float* __restrict__ part_val,
        const int* __restrict__ part_idx, int nblocks, int* __restrict__ out) {
    __shared__ float sv[256];
    __shared__ int   si[256];
    int b = blockIdx.x, tid = threadIdx.x;
    const float* pv = part_val + (size_t)b * nblocks;
    const int*   pi = part_idx + (size_t)b * nblocks;
    float best = -1e30f; int idx = 0;
    for (int i = tid; i < nblocks; i += blockDim.x) if (pv[i] > best) { best = pv[i]; idx = pi[i]; }
    sv[tid] = best; si[tid] = idx;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) out[b] = si[0];
}

void launch_lm_head_argmax(const float* x, const int8_t* W, const float* scale, int B, int IN, int OUT,
                           float* part_val, int* part_idx, int* out, cudaStream_t s) {
    int nblocks = LM_ARGMAX_BLOCKS;
    #define LAUNCH_LMA(T) lm_head_argmax_partial_kernel<T><<<nblocks, 256, 0, s>>>( \
        x, W, scale, IN, OUT, part_val, part_idx, nblocks)
    if      (B <= 1)  LAUNCH_LMA(1);
    else if (B <= 2)  LAUNCH_LMA(2);
    else if (B <= 4)  LAUNCH_LMA(4);
    else if (B <= 8)  LAUNCH_LMA(8);
    else if (B <= 16) LAUNCH_LMA(16);
    else              LAUNCH_LMA(32);
    #undef LAUNCH_LMA
    lm_head_argmax_combine_kernel<<<B, 256, 0, s>>>(part_val, part_idx, nblocks, out);
}
