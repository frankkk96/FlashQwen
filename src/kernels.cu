#include "kernels.cuh"
#include <mma.h>
using namespace nvcuda;

// ---------------------------------------------------------------------------------------
// GEMV (decode, M==1): one warp computes one output element (dot product over IN).
// Memory-bound — reading the BF16 weights dominates, so tensor cores would not help here.
// ---------------------------------------------------------------------------------------
__global__ void gemv_kernel(const float* __restrict__ x, const bf16* __restrict__ W,
                            float* __restrict__ y, int IN, int OUT) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;   // output feature
    if (row >= OUT) return;

    const bf16* wr = W + (size_t)row * IN;
    float acc = 0.f;
    for (int i = lane; i < IN; i += 32)
        acc += x[i] * __bfloat162float(wr[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) y[row] = acc;
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

void launch_matmul(const float* x, const bf16* W, float* y, int M, int IN, int OUT,
                   bf16* x_bf16, cudaStream_t s) {
    if (M <= 1) {                              // decode: memory-bound GEMV
        const int warps = 8;
        gemv_kernel<<<(OUT + warps - 1) / warps, warps * 32, 0, s>>>(x, W, y, IN, OUT);
        return;
    }
    // prefill: convert activations to BF16 (zero-padding tail rows to a 16-row multiple),
    // then run the tensor-core GEMM.
    int Mp = ((M + 15) / 16) * 16;
    int total = Mp * IN, valid = M * IN, blk = 256;
    f32_to_bf16_kernel<<<(total + blk - 1) / blk, blk, 0, s>>>(x, x_bf16, valid, total);

    dim3 block(256);                           // 8 warps
    dim3 grid((OUT / 16 + 7) / 8, Mp / 16);
    wmma_kernel<<<grid, block, 0, s>>>(x_bf16, W, y, IN, OUT);
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
                                 int past_len, float scale) {
    // DPL = head_dim/32, fixed at compile time (Qwen3 head_dim is 128) so qreg/acc stay in
    // registers instead of spilling to local memory.
    constexpr int DPL = 4;
    int h = blockIdx.x;              // query head
    int m = blockIdx.y;              // query token
    int lane = threadIdx.x;          // 0..31
    int kvh = h / (n_heads / n_kv);
    int qpos = past_len + m;         // attend keys [0, qpos]

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
                      int past_len, float scale, cudaStream_t s) {
    dim3 grid(n_heads, M);
    attention_kernel<<<grid, 32, 0, s>>>(
        q, cache_k, cache_v, out, M, n_heads, n_kv, head_dim, past_len, scale);
}

// FP32 -> BF16 (no padding); used to write the KV cache.
void launch_to_bf16(const float* in, bf16* out, int n, cudaStream_t s) {
    int blk = 256;
    f32_to_bf16_kernel<<<(n + blk - 1) / blk, blk, 0, s>>>(in, out, n, n);
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
