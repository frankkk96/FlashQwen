#include "kernels.cuh"

// ---------------------------------------------------------------------------------------
// matmul (scalar baseline): one warp computes one output element (dot product over IN).
// grid.x tiles OUT in groups of 8 warps; grid.y = M (tokens). Used for both prefill and
// decode. No tensor cores. (`x_bf16` scratch is unused in this baseline.)
// ---------------------------------------------------------------------------------------
__global__ void matmul_kernel(const float* __restrict__ x, const bf16* __restrict__ W,
                              float* __restrict__ y, int M, int IN, int OUT) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;   // output feature
    int m   = blockIdx.y;                              // token
    if (row >= OUT) return;

    const float* xr = x + (size_t)m * IN;
    const bf16*  wr = W + (size_t)row * IN;
    float acc = 0.f;
    for (int i = lane; i < IN; i += 32)
        acc += xr[i] * __bfloat162float(wr[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) y[(size_t)m * OUT + row] = acc;
}

void launch_matmul(const float* x, const bf16* W, float* y, int M, int IN, int OUT,
                   bf16* x_bf16, cudaStream_t s) {
    (void)x_bf16;
    const int warps = 8;
    dim3 block(warps * 32);
    dim3 grid((OUT + warps - 1) / warps, M);
    matmul_kernel<<<grid, block, 0, s>>>(x, W, y, M, IN, OUT);
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
// attention with online (flash-style) softmax: one block per (head, query token).
// blockDim = head_dim.
// ---------------------------------------------------------------------------------------
__global__ void attention_kernel(const float* __restrict__ q, const float* __restrict__ cache_k,
                                 const float* __restrict__ cache_v, float* __restrict__ out,
                                 int M, int n_heads, int n_kv, int head_dim,
                                 int past_len, float scale) {
    int h = blockIdx.x;              // query head
    int m = blockIdx.y;              // query token
    int t = threadIdx.x;             // dim within head
    int group = n_heads / n_kv;
    int kvh = h / group;
    int qpos = past_len + m;         // attend keys [0, qpos]

    const float* qv = q + ((size_t)m * n_heads + h) * head_dim;
    float qd = qv[t];

    extern __shared__ float red[];   // size head_dim
    float m_run = -1e30f, l_run = 0.f, acc = 0.f;

    for (int j = 0; j <= qpos; ++j) {
        const float* kv = cache_k + ((size_t)j * n_kv + kvh) * head_dim;
        red[t] = qd * kv[t];
        __syncthreads();
        for (int s = head_dim >> 1; s > 0; s >>= 1) {
            if (t < s) red[t] += red[t + s];
            __syncthreads();
        }
        float score = red[0] * scale;
        __syncthreads();

        float m_new = fmaxf(m_run, score);
        float corr  = expf(m_run - m_new);
        float p     = expf(score - m_new);
        const float* vv = cache_v + ((size_t)j * n_kv + kvh) * head_dim;
        l_run = l_run * corr + p;
        acc   = acc   * corr + p * vv[t];
        m_run = m_new;
    }
    out[((size_t)m * n_heads + h) * head_dim + t] = acc / l_run;
}

void launch_attention(const float* q, const float* cache_k, const float* cache_v,
                      float* out, int M, int n_heads, int n_kv, int head_dim,
                      int past_len, float scale, cudaStream_t s) {
    dim3 grid(n_heads, M);
    attention_kernel<<<grid, head_dim, head_dim * sizeof(float), s>>>(
        q, cache_k, cache_v, out, M, n_heads, n_kv, head_dim, past_len, scale);
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
