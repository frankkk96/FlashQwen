#include <cuda_pipeline.h>  // cp.async (prefill K/V staging)

#include <cassert>

#include "kernels.cuh"

// gemm: linear layers via cuBLAS (BF16 in / FP32 accumulate). Row-major <->
// column-major derivation is in kernels.cuh.
void LaunchGemm(cublasHandle_t handle, const bf16* x, const bf16* W, void* y,
                int M, int IN, int OUT, cudaDataType_t Ytype) {
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, OUT, M, IN,
                            &alpha, W, CUDA_R_16BF, IN, x, CUDA_R_16BF, IN,
                            &beta, y, Ytype, OUT, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}

// rmsnorm: one block per row, block-reduction over H. FP32 reduction + scale.
__global__ void RmsnormKernel(const bf16* __restrict__ x,
                              const float* __restrict__ w,
                              bf16* __restrict__ out, int M, int H, float eps) {
  int m = blockIdx.x;
  const bf16* xr = x + static_cast<int64_t>(m) * H;
  bf16* outr = out + static_cast<int64_t>(m) * H;

  extern __shared__ float red[];
  float local = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float v = __bfloat162float(xr[i]);
    local += v * v;
  }
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / H + eps);
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    outr[i] = __float2bfloat16(__bfloat162float(xr[i]) * inv * w[i]);
}

void LaunchRmsnorm(const bf16* x, const float* w, bf16* out, int M, int H,
                   float eps, cudaStream_t s) {
  int block = 256;
  RmsnormKernel<<<M, block, block * sizeof(float), s>>>(x, w, out, M, H, eps);
}

// Fused residual-add + rmsnorm: x += res (stored back to carry the residual
// forward), out = rmsnorm(x).
__global__ void AddRmsnormKernel(bf16* __restrict__ x,
                                 const bf16* __restrict__ res,
                                 const float* __restrict__ w,
                                 bf16* __restrict__ out, int M, int H,
                                 float eps) {
  int m = blockIdx.x;
  bf16* xr = x + static_cast<int64_t>(m) * H;
  const bf16* rr = res + static_cast<int64_t>(m) * H;
  bf16* outr = out + static_cast<int64_t>(m) * H;

  extern __shared__ float red[];
  float local = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float v = __bfloat162float(xr[i]) + __bfloat162float(rr[i]);
    xr[i] = __float2bfloat16(v);  // updated residual stays in x for next layer
    local += v * v;
  }
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / H + eps);
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    outr[i] = __float2bfloat16(__bfloat162float(xr[i]) * inv * w[i]);
}

void LaunchAddRmsnorm(bf16* x, const bf16* res, const float* w, bf16* out,
                      int M, int H, float eps, cudaStream_t s) {
  int block = 256;
  AddRmsnormKernel<<<M, block, block * sizeof(float), s>>>(x, res, w, out, M, H,
                                                           eps);
}

// embedding gather (BF16 -> BF16, straight copy)
__global__ void EmbedKernel(const int* __restrict__ ids,
                            const bf16* __restrict__ embed,
                            bf16* __restrict__ out, int M, int H) {
  int m = blockIdx.x;
  const bf16* src = embed + static_cast<int64_t>(ids[m]) * H;
  bf16* dst = out + static_cast<int64_t>(m) * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x) dst[i] = src[i];
}

void LaunchEmbed(const int* ids, const bf16* embed, bf16* out, int M, int H,
                 cudaStream_t s) {
  EmbedKernel<<<M, 256, 0, s>>>(ids, embed, out, M, H);
}

// Fused per-head RMSNorm + RoPE (rotate-half, matching HF Qwen) over the q AND
// k slices of a fused QKV row of width `stride` in one launch: the q heads
// (weight wq) and k heads (weight wk) are contiguous, so head g of token m lives
// at buf + m*stride + g*head_dim for g in [0, n_q+n_kv); g < n_q is a q head,
// else a k head. v is left untouched. Angles come from precomputed cos/sin
// tables ([max_pos, head_dim/2]) indexed by pos[m] (no per-call transcendental;
// tables are identical across all layers and built once). One block per
// (token, head); blockDim == head_dim. FP32 math.
__global__ void HeadNormRopeKernel(bf16* __restrict__ buf,
                                   const float* __restrict__ wq,
                                   const float* __restrict__ wk,
                                   const float* __restrict__ cos_tab,
                                   const float* __restrict__ sin_tab,
                                   const int* __restrict__ pos, int n_q,
                                   int n_kv, int head_dim, int stride,
                                   float eps) {
  int hpr = n_q + n_kv;  // q + k heads per row
  int row = blockIdx.x;  // over M * hpr
  int m = row / hpr, g = row % hpr;
  const float* w = g < n_q ? wq : wk;
  int t = threadIdx.x;  // 0..head_dim-1
  int half = head_dim >> 1;
  bf16* v = buf + static_cast<int64_t>(m) * stride +
            static_cast<int64_t>(g) * head_dim;

  extern __shared__ float sh[];  // [0,head_dim): reduction then normed values
  float x = __bfloat162float(v[t]);
  sh[t] = x * x;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (t < s) sh[t] += sh[t + s];
    __syncthreads();
  }
  float inv = rsqrtf(sh[0] / head_dim + eps);
  __syncthreads();
  sh[t] = x * inv * w[t];  // normed value
  __syncthreads();

  if (t < half) {
    const float* cb = cos_tab + static_cast<int64_t>(pos[m]) * half;
    const float* sb = sin_tab + static_cast<int64_t>(pos[m]) * half;
    float cs = cb[t], sn = sb[t];
    float x1 = sh[t], x2 = sh[t + half];
    v[t] = __float2bfloat16(x1 * cs - x2 * sn);
    v[t + half] = __float2bfloat16(x2 * cs + x1 * sn);
  }
}

void LaunchHeadNormRope(bf16* buf, const float* wq, const float* wk,
                        const float* cos_tab, const float* sin_tab,
                        const int* pos, int M, int n_q, int n_kv, int head_dim,
                        int stride, float eps, cudaStream_t s) {
  HeadNormRopeKernel<<<M * (n_q + n_kv), head_dim, head_dim * sizeof(float),
                       s>>>(buf, wq, wk, cos_tab, sin_tab, pos, n_q, n_kv,
                            head_dim, stride, eps);
}

// ==== Paged-KV attention ====
// Split by request type (dispatched from RunLayers): prefill (q_len>1) ->
// AttnPrefill (tensor cores), decode (q_len==1) -> AttnDecode (FlashDecoding).
// Both read the paged KV pool [num_blocks, kBlock, kv_dim] via per-request
// block tables `bt` (stride max_blocks); q comes from the fused-QKV buffer at
// row stride q_stride. `rids[grid_slot]` -> request id, so each runs on its
// subset.

// GQA-shared FlashDecoding (decode path). One block owns a (kv-head, request)
// and computes all G = n_heads/n_kv query heads, loading each K/V element ONCE
// into registers and reusing it across the G heads (vs. re-reading each
// kv-head's K/V once per query head in the group, ~2.3x more work). Since n_kv
// gives 4x fewer blocks than n_heads, the KV range is split across grid.z
// (ksplit) FlashDecoding-style to keep the GPU saturated; a tiny second kernel
// combines the per-split partials. (S4: grouping by kv-head with only n_kv
// blocks starved the GPU; the KV split is what makes it a win.)
//
// Phase 1: each block (kv-head, request, split) -> NW warps stream their slice
// of the split's keys, online-softmax per head, in-block combine across the NW
// warps, write G UNNORMALIZED partials (m, l, acc[head_dim]) into pm/pl/pa
// indexed [(di*n_heads + h)*ksplit + sp].
// Single-config for qwen3-8B: head_dim=128 => DPL = head_dim/32 = 4;
// n_heads/n_kv=4 => G = 4; NW = 8 warps/block. LaunchAttnDecode asserts these.
constexpr int NW = 8, DPL = 4, G = 4;
__global__ void AttnDecodeGqaKernel(
    const bf16* __restrict__ q, int q_stride, const bf16* __restrict__ cache_k,
    const bf16* __restrict__ cache_v, float* __restrict__ pm,
    float* __restrict__ pl, float* __restrict__ pa, int n_heads, int n_kv,
    int head_dim, const int* __restrict__ pos, const int* __restrict__ qstart,
    const int* __restrict__ decode_rids, const int* __restrict__ bt,
    int max_blocks, int block_size, float scale, int ksplit) {
  int kvh = blockIdx.x;  // kv head (owns this GQA group)
  int di = blockIdx.y;   // decode-request slot
  int sp = blockIdx.z;   // KV split
  int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int r = decode_rids[di], flat = qstart[r], qpos = pos[flat];
  int kv_dim = n_kv * head_dim;
  const int* btr = bt + static_cast<int64_t>(r) * max_blocks;

  float qreg[G][DPL];  // the G query heads of this group, in registers
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const bf16* qv = q + static_cast<int64_t>(flat) * q_stride +
                     static_cast<int64_t>(kvh * G + g) * head_dim;
#pragma unroll
    for (int i = 0; i < DPL; ++i)
      qreg[g][i] = __bfloat162float(qv[lane + (i << 5)]);
  }
  float m_run[G], l_run[G], acc[G][DPL];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    m_run[g] = -1e30f;
    l_run[g] = 0.f;
#pragma unroll
    for (int i = 0; i < DPL; ++i) acc[g][i] = 0.f;
  }

  // warp w of split sp streams keys sp*NW+w, +NW*ksplit, ... (each key hits
  // exactly one (sp,w)).
  for (int kpos = sp * NW + w; kpos <= qpos; kpos += NW * ksplit) {
    int64_t base =
        static_cast<int64_t>(btr[kpos / block_size]) * block_size * kv_dim +
        static_cast<int64_t>(kpos % block_size) * kv_dim +
        static_cast<int64_t>(kvh) * head_dim;
    const bf16* kc = cache_k + base;
    const bf16* vc = cache_v + base;
    float kreg[DPL], vreg[DPL];  // K/V loaded ONCE, reused across the G heads
#pragma unroll
    for (int i = 0; i < DPL; ++i) {
      kreg[i] = __bfloat162float(kc[lane + (i << 5)]);
      vreg[i] = __bfloat162float(vc[lane + (i << 5)]);
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float partial = 0.f;
#pragma unroll
      for (int i = 0; i < DPL; ++i) partial += qreg[g][i] * kreg[i];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, o);
      float score = partial * scale;
      float m_new = fmaxf(m_run[g], score), corr = __expf(m_run[g] - m_new),
            p = __expf(score - m_new);
      l_run[g] = l_run[g] * corr + p;
#pragma unroll
      for (int i = 0; i < DPL; ++i) acc[g][i] = acc[g][i] * corr + p * vreg[i];
      m_run[g] = m_new;
    }
  }

  __shared__ float sm[G][NW], sl[G][NW], sa[G][NW][DPL * 32];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    sm[g][w] = m_run[g];
    sl[g][w] = l_run[g];
#pragma unroll
    for (int i = 0; i < DPL; ++i) sa[g][w][lane + (i << 5)] = acc[g][i];
  }
  __syncthreads();

  if (w == 0) {  // merge the NW warps -> one (m,l,acc) per head for this split
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float gm = -1e30f;
#pragma unroll
      for (int t = 0; t < NW; ++t) gm = fmaxf(gm, sm[g][t]);
      float gl = 0.f, gacc[DPL];
#pragma unroll
      for (int i = 0; i < DPL; ++i) gacc[i] = 0.f;
      for (int t = 0; t < NW; ++t) {
        float c = __expf(sm[g][t] - gm);
        gl += sl[g][t] * c;
#pragma unroll
        for (int i = 0; i < DPL; ++i) gacc[i] += sa[g][t][lane + (i << 5)] * c;
      }
      int h = kvh * G + g;
      int64_t idx = (static_cast<int64_t>(di) * n_heads + h) * ksplit + sp;
      if (lane == 0) {
        pm[idx] = gm;
        pl[idx] = gl;
      }
#pragma unroll
      for (int i = 0; i < DPL; ++i)
        pa[idx * head_dim + lane + (i << 5)] = gacc[i];
    }
  }
}

// Phase 2: combine the ksplit partials of each (head, request) -> the final
// normalized output row. One block per (head, decode-request); head_dim
// threads, each owns one output element.
__global__ void AttnDecodeCombineKernel(
    const float* __restrict__ pm, const float* __restrict__ pl,
    const float* __restrict__ pa, bf16* __restrict__ out, int n_heads,
    int head_dim, const int* __restrict__ qstart,
    const int* __restrict__ decode_rids, int ksplit) {
  int h = blockIdx.x, di = blockIdx.y, d = threadIdx.x;
  int r = decode_rids[di], flat = qstart[r];
  int64_t base = (static_cast<int64_t>(di) * n_heads + h) * ksplit;
  float gm = -1e30f;
  for (int s = 0; s < ksplit; ++s) gm = fmaxf(gm, pm[base + s]);
  float gl = 0.f, gacc = 0.f;
  for (int s = 0; s < ksplit; ++s) {
    float c = __expf(pm[base + s] - gm);
    gl += pl[base + s] *
          c;  // ksplit is small; recomputing gl per thread is cheap
    gacc += pa[(base + s) * head_dim + d] * c;
  }
  float inv = gl > 0.f ? 1.0f / gl : 0.f;
  out[(static_cast<int64_t>(flat) * n_heads + h) * head_dim + d] =
      __float2bfloat16(gacc * inv);
}

void LaunchAttnDecode(const bf16* q, int q_stride, const bf16* cache_k,
                      const bf16* cache_v, bf16* out, int n_heads, int n_kv,
                      int head_dim, const int* pos, const int* qstart,
                      const int* decode_rids, int n_decode, const int* bt,
                      int max_blocks, int block_size, float scale,
                      int max_decode, cudaStream_t s) {
  if (n_decode <= 0) return;
  assert(head_dim == 128 && n_heads / n_kv == 4);  // Qwen3-8B: DPL=4, G=4

  // Split the KV range over grid.z so n_kv (< n_heads) blocks still saturate
  // the GPU; a second kernel combines the per-split partials.
  constexpr int MAX_KSPLIT = 16;
  int ksplit = 128 / (n_decode > 0 ? n_decode : 1);
  if (ksplit < 1) ksplit = 1;
  if (ksplit > MAX_KSPLIT) ksplit = MAX_KSPLIT;

  // Persistent partials, reused across layers/steps; sized on first call to the
  // worst case (max_decode is fixed for the process, so this never re-grows).
  static float *pm = nullptr, *pl = nullptr, *pa = nullptr;
  if (!pm) {
    size_t nent =
        static_cast<size_t>(max_decode) * 64 /*heads guard*/ * MAX_KSPLIT;
    cudaMalloc(&pm, nent * sizeof(float));
    cudaMalloc(&pl, nent * sizeof(float));
    cudaMalloc(&pa, nent * 128 * sizeof(float));
  }
  dim3 g1(n_kv, n_decode, ksplit);
  AttnDecodeGqaKernel<<<g1, NW * 32, 0, s>>>(
      q, q_stride, cache_k, cache_v, pm, pl, pa, n_heads, n_kv, head_dim, pos,
      qstart, decode_rids, bt, max_blocks, block_size, scale, ksplit);
  dim3 g2(n_heads, n_decode);
  AttnDecodeCombineKernel<<<g2, head_dim, 0, s>>>(
      pm, pl, pa, out, n_heads, head_dim, qstart, decode_rids, ksplit);
}

// prefill: mma.sync (m16n8k16 bf16) FlashAttention-style. 4
// warps per block, 64-query-row tile; block-shared K/V staged ONCE per 16-key
// tile and reused by all 4 warps (~4x less KV traffic); O kept in REGISTERS
// (mma C-fragment) and rescaled per online-softmax step (no shared O
// round-trip); online softmax in registers via quad shuffles. V is staged
// transposed so the P@V B-operand is natural; P passes through a tiny shared
// buffer to skip the C->A fragment repack. ~2.5x a WMMA implementation
// (microbench, hd=128). HD==128, block_size==16 only.
static __device__ __forceinline__ unsigned FqaPack(__nv_bfloat16 a,
                                                   __nv_bfloat16 b) {
  __nv_bfloat162 t = __halves2bfloat162(a, b);
  unsigned x;
  memcpy(&x, &t, 4);
  return x;
}
static __device__ __forceinline__ void FqaMma(float d[4], const unsigned a[4],
                                              const unsigned b[2],
                                              const float c[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__global__ void AttnPrefillMmaKernel(
    const bf16* __restrict__ q, int q_stride, const bf16* __restrict__ cache_k,
    const bf16* __restrict__ cache_v, bf16* __restrict__ out, int n_heads,
    int n_kv, const int* __restrict__ pos, const int* __restrict__ qstart,
    const int* __restrict__ qlen, const int* __restrict__ rids,
    const int* __restrict__ bt, int max_blocks, float scale) {
  constexpr int HDc = 128, NT = HDc / 8;
  int r = rids[blockIdx.z], h = blockIdx.y;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, grp = lane >> 2,
      tg = lane & 3;
  int ql = qlen[r], qs = qstart[r];
  int qbase = blockIdx.x * 64 + warp * 16;
  int kvh = h / (n_heads / n_kv), kv_dim = n_kv * HDc;
  const int* btr = bt + static_cast<int64_t>(r) * max_blocks;
  if (blockIdx.x * 64 >= ql) return;  // whole block past this request's rows

  __shared__ bf16 Ksh[16 * HDc];  // cp.async-staged K tile (natural layout)
  __shared__ bf16 Vsh[16 * HDc];  // cp.async-staged V tile (natural layout)
  __shared__ bf16 Psh[4][16 * 16];
  bf16* Ps = Psh[warp];

  unsigned qa[8][4];
  for (int kk = 0; kk < 8; ++kk) {
    const bf16* qp = q + static_cast<int64_t>(qs + qbase) * q_stride +
                     static_cast<int64_t>(h) * HDc + kk * 16;
#define FQ_QG(row, col)                                                    \
  ((qbase + (row)) < ql ? qp[static_cast<int64_t>(row) * q_stride + (col)] \
                        : __float2bfloat16(0.f))
    qa[kk][0] = FqaPack(FQ_QG(grp, tg * 2), FQ_QG(grp, tg * 2 + 1));
    qa[kk][1] = FqaPack(FQ_QG(grp + 8, tg * 2), FQ_QG(grp + 8, tg * 2 + 1));
    qa[kk][2] = FqaPack(FQ_QG(grp, tg * 2 + 8), FQ_QG(grp, tg * 2 + 9));
    qa[kk][3] = FqaPack(FQ_QG(grp + 8, tg * 2 + 8), FQ_QG(grp + 8, tg * 2 + 9));
#undef FQ_QG
  }
  float O[NT][4];
  for (int t = 0; t < NT; t++) {
    O[t][0] = O[t][1] = O[t][2] = O[t][3] = 0.f;
  }
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;

  int blast = min(blockIdx.x * 64 + 63, ql - 1);
  int ntiles = pos[qs + blast] / 16 + 1;

  // cp.async-copy tile kt's K&V (natural layout) into shared. Single-buffered:
  // a double-buffered prefetch had a correctness hazard in the engine's
  // back-to-back multi-layer launch context (only on cache-hit / chunked-tail
  // prefills — few query rows + many K/V tiles), and cp.async overlap was worth
  // only ~0.9% e2e, so single-buffer staging keeps the register-O + mma +
  // K/V-reuse win. See prefix-cache-large-hit-bug.
  auto stage = [&](int kt) {
    int64_t kvbase = static_cast<int64_t>(btr[kt]) * 16 * kv_dim +
                     static_cast<int64_t>(kvh) * HDc;
    for (int c = threadIdx.x; c < 16 * 16; c += 128) {
      int key = c >> 4, ck = c & 15;  // 16B (8 bf16) chunks
      __pipeline_memcpy_async(
          &Ksh[key * HDc + ck * 8],
          &cache_k[kvbase + static_cast<int64_t>(key) * kv_dim + ck * 8], 16);
      __pipeline_memcpy_async(
          &Vsh[key * HDc + ck * 8],
          &cache_v[kvbase + static_cast<int64_t>(key) * kv_dim + ck * 8], 16);
    }
    __pipeline_commit();
  };
  for (int kt = 0; kt < ntiles; ++kt) {
    stage(kt);
    __pipeline_wait_prior(0);
    __syncthreads();
    bf16* Ks = Ksh;
    bf16* Vs = Vsh;

    float S[2][4];
    for (int nh = 0; nh < 2; ++nh) {
      float acc[4] = {0, 0, 0, 0};
      for (int kk = 0; kk < 8; ++kk) {
        unsigned b[2];
        int kb = kk * 16;
        b[0] = FqaPack(Ks[(nh * 8 + grp) * HDc + kb + tg * 2],
                       Ks[(nh * 8 + grp) * HDc + kb + tg * 2 + 1]);
        b[1] = FqaPack(Ks[(nh * 8 + grp) * HDc + kb + tg * 2 + 8],
                       Ks[(nh * 8 + grp) * HDc + kb + tg * 2 + 9]);
        float c[4] = {acc[0], acc[1], acc[2], acc[3]};
        FqaMma(acc, qa[kk], b, c);
      }
      S[nh][0] = acc[0];
      S[nh][1] = acc[1];
      S[nh][2] = acc[2];
      S[nh][3] = acc[3];
    }
    float rmx0 = -1e30f, rmx1 = -1e30f;
    for (int nh = 0; nh < 2; nh++) {
      int kpos0 = kt * 16 + nh * 8 + tg * 2, kpos1 = kpos0 + 1;
      int qp0 = (qbase + grp < ql) ? pos[qs + qbase + grp] : -1;
      int qp1 = (qbase + grp + 8 < ql) ? pos[qs + qbase + grp + 8] : -1;
      S[nh][0] = (kpos0 <= qp0) ? S[nh][0] * scale : -1e30f;
      S[nh][1] = (kpos1 <= qp0) ? S[nh][1] * scale : -1e30f;
      S[nh][2] = (kpos0 <= qp1) ? S[nh][2] * scale : -1e30f;
      S[nh][3] = (kpos1 <= qp1) ? S[nh][3] * scale : -1e30f;
      rmx0 = fmaxf(rmx0, fmaxf(S[nh][0], S[nh][1]));
      rmx1 = fmaxf(rmx1, fmaxf(S[nh][2], S[nh][3]));
    }
    rmx0 = fmaxf(rmx0, __shfl_xor_sync(0xffffffff, rmx0, 1));
    rmx0 = fmaxf(rmx0, __shfl_xor_sync(0xffffffff, rmx0, 2));
    rmx1 = fmaxf(rmx1, __shfl_xor_sync(0xffffffff, rmx1, 1));
    rmx1 = fmaxf(rmx1, __shfl_xor_sync(0xffffffff, rmx1, 2));
    float nm0 = fmaxf(m0, rmx0), nm1 = fmaxf(m1, rmx1);
    float cc0 = __expf(m0 - nm0), cc1 = __expf(m1 - nm1);
    float ps0 = 0, ps1 = 0;
    for (int nh = 0; nh < 2; nh++) {
      float p0 = __expf(S[nh][0] - nm0), p1 = __expf(S[nh][1] - nm0);
      float p2 = __expf(S[nh][2] - nm1), p3 = __expf(S[nh][3] - nm1);
      ps0 += p0 + p1;
      ps1 += p2 + p3;
      Ps[(grp)*16 + nh * 8 + tg * 2] = __float2bfloat16(p0);
      Ps[(grp)*16 + nh * 8 + tg * 2 + 1] = __float2bfloat16(p1);
      Ps[(grp + 8) * 16 + nh * 8 + tg * 2] = __float2bfloat16(p2);
      Ps[(grp + 8) * 16 + nh * 8 + tg * 2 + 1] = __float2bfloat16(p3);
    }
    ps0 += __shfl_xor_sync(0xffffffff, ps0, 1);
    ps0 += __shfl_xor_sync(0xffffffff, ps0, 2);
    ps1 += __shfl_xor_sync(0xffffffff, ps1, 1);
    ps1 += __shfl_xor_sync(0xffffffff, ps1, 2);
    l0 = l0 * cc0 + ps0;
    l1 = l1 * cc1 + ps1;
    m0 = nm0;
    m1 = nm1;
    for (int t = 0; t < NT; t++) {
      O[t][0] *= cc0;
      O[t][1] *= cc0;
      O[t][2] *= cc1;
      O[t][3] *= cc1;
    }
    __syncwarp();
    unsigned pa[4];
    pa[0] = FqaPack(Ps[grp * 16 + tg * 2], Ps[grp * 16 + tg * 2 + 1]);
    pa[1] =
        FqaPack(Ps[(grp + 8) * 16 + tg * 2], Ps[(grp + 8) * 16 + tg * 2 + 1]);
    pa[2] = FqaPack(Ps[grp * 16 + tg * 2 + 8], Ps[grp * 16 + tg * 2 + 9]);
    pa[3] = FqaPack(Ps[(grp + 8) * 16 + tg * 2 + 8],
                    Ps[(grp + 8) * 16 + tg * 2 + 9]);
    for (int nt = 0; nt < NT; nt++) {
      unsigned b[2];
      int hd = nt * 8 + grp;  // V natural, transposed read: Vs[key*HD + hd]
      b[0] = FqaPack(Vs[(tg * 2) * HDc + hd], Vs[(tg * 2 + 1) * HDc + hd]);
      b[1] = FqaPack(Vs[(tg * 2 + 8) * HDc + hd], Vs[(tg * 2 + 9) * HDc + hd]);
      float c[4] = {O[nt][0], O[nt][1], O[nt][2], O[nt][3]};
      FqaMma(O[nt], pa, b, c);
    }
    __syncthreads();
  }
  float inv0 = l0 > 0 ? 1.f / l0 : 0.f, inv1 = l1 > 0 ? 1.f / l1 : 0.f;
  for (int nt = 0; nt < NT; nt++) {
    int rr0 = qbase + grp, rr1 = qbase + grp + 8, hd = nt * 8 + tg * 2;
    if (rr0 < ql) {
      out[(static_cast<int64_t>(qs + rr0) * n_heads + h) * HDc + hd] =
          __float2bfloat16(O[nt][0] * inv0);
      out[(static_cast<int64_t>(qs + rr0) * n_heads + h) * HDc + hd + 1] =
          __float2bfloat16(O[nt][1] * inv0);
    }
    if (rr1 < ql) {
      out[(static_cast<int64_t>(qs + rr1) * n_heads + h) * HDc + hd] =
          __float2bfloat16(O[nt][2] * inv1);
      out[(static_cast<int64_t>(qs + rr1) * n_heads + h) * HDc + hd + 1] =
          __float2bfloat16(O[nt][3] * inv1);
    }
  }
}

void LaunchAttnPrefill(const bf16* q, int q_stride, const bf16* cache_k,
                       const bf16* cache_v, bf16* out, int n_heads, int n_kv,
                       int head_dim, const int* pos, const int* qstart,
                       const int* qlen, const int* rids, int R, int max_qlen,
                       const int* bt, int max_blocks, int block_size,
                       float scale, cudaStream_t s) {
  if (R <= 0) return;
  assert(head_dim == 128 &&
         block_size == 16);  // Qwen3: 16 n8-tiles, 1 tile/blk
  dim3 grid((max_qlen + 63) / 64, n_heads, R);
  AttnPrefillMmaKernel<<<grid, 128, 0, s>>>(q, q_stride, cache_k, cache_v, out,
                                            n_heads, n_kv, pos, qstart, qlen,
                                            rids, bt, max_blocks, scale);
}

// Gather S rows: out[i, :] = x[rows[i], :]   (one block per gathered row, BF16)
__global__ void GatherRowsKernel(const bf16* __restrict__ x,
                                 const int* __restrict__ rows,
                                 bf16* __restrict__ out, int H) {
  int i = blockIdx.x;
  const bf16* src = x + static_cast<int64_t>(rows[i]) * H;
  bf16* dst = out + static_cast<int64_t>(i) * H;
  for (int j = threadIdx.x; j < H; j += blockDim.x) dst[j] = src[j];
}

void LaunchGatherRows(const bf16* x, const int* rows, bf16* out, int S, int H,
                      cudaStream_t s) {
  if (S > 0) GatherRowsKernel<<<S, 256, 0, s>>>(x, rows, out, H);
}

// elementwise helpers (BF16 in/out, FP32 math)
__global__ void AddKernel(bf16* out, const bf16* in, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    out[i] =
        __float2bfloat16(__bfloat162float(out[i]) + __bfloat162float(in[i]));
}
void LaunchAdd(bf16* out, const bf16* in, int N, cudaStream_t s) {
  int block = 256;
  AddKernel<<<(N + block - 1) / block, block, 0, s>>>(out, in, N);
}

// h[m,i] = silu(gateup[m,i]) * gateup[m, I+i] — gate and up are the two halves
// of a fused row (width 2I).
__global__ void SiluMulKernel(const bf16* __restrict__ gateup,
                              bf16* __restrict__ h, int M, int I) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(M) * I) return;
  int m = idx / I, i = idx % I;
  const bf16* row = gateup + static_cast<int64_t>(m) * 2 * I;
  float g = __bfloat162float(row[i]);
  h[idx] =
      __float2bfloat16((g / (1.0f + expf(-g))) * __bfloat162float(row[I + i]));
}
void LaunchSiluMul(const bf16* gateup, bf16* h, int M, int I, cudaStream_t s) {
  int block = 256;
  long N = static_cast<long>(M) * I;
  SiluMulKernel<<<(N + block - 1) / block, block, 0, s>>>(gateup, h, M, I);
}

static constexpr int kSampleThreads = 256;

// Greedy: argmax over row b of logits[B, N] via a strided sweep + block reduce.
// Rows with invT[b] > 0 are sampled instead (handled by SampleKernel), so this
// block returns early for them.
__global__ void ArgmaxKernel(const float* __restrict__ logits, int N,
                             const float* __restrict__ invT,
                             int* __restrict__ out) {
  int b = blockIdx.x;
  if (invT[b] > 0.0f) return;
  int tid = threadIdx.x, nt = blockDim.x;
  const float* lg = logits + static_cast<int64_t>(b) * N;

  __shared__ float sval[kSampleThreads];
  __shared__ int sidx[kSampleThreads];
  float best = -1e30f;
  int bi = 0;
  for (int i = tid; i < N; i += nt) {
    float v = lg[i];
    if (v > best) {
      best = v;
      bi = i;
    }
  }
  sval[tid] = best;
  sidx[tid] = bi;
  __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) {
    if (tid < s && sval[tid + s] > sval[tid]) {
      sval[tid] = sval[tid + s];
      sidx[tid] = sidx[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) out[b] = sidx[0];
}

// Temperature categorical: one block per row of logits[B, N]. Softmax (with
// max-subtraction for stability) + inverse-CDF sampling, restricted to the
// nucleus when top_p[b] < 1. Each thread owns a CONTIGUOUS index range so the
// cumulative scan runs in token-id order (any consistent order gives the same
// distribution; contiguous makes each thread's prefix a true prefix over token
// ids). Nucleus = smallest set of highest-prob tokens with cumulative prob >=
// top_p, found by binary-searching a weight threshold wt with
// sum_{w_i >= wt} w_i >= top_p*Z. No global sort — all work stays in one block.
// Rows with invT[b] <= 0 are greedy (handled by ArgmaxKernel), returned early.
__global__ void SampleKernel(const float* __restrict__ logits, int N,
                             const float* __restrict__ invT,
                             const float* __restrict__ topp,
                             const float* __restrict__ u,
                             int* __restrict__ out) {
  int b = blockIdx.x;
  float it = invT[b];
  if (it <= 0.0f) return;
  int tid = threadIdx.x, nt = blockDim.x;
  const float* lg = logits + static_cast<int64_t>(b) * N;

  __shared__ float sval[kSampleThreads];  // reduction / partial-sum scratch
  int chunk = (N + nt - 1) / nt;
  int lo = min(tid * chunk, N), hi = min(lo + chunk, N);

  // pass 1: max logit (numerical stability), block max-reduce
  float lmax = -1e30f;
  for (int i = lo; i < hi; ++i) lmax = fmaxf(lmax, lg[i]);
  sval[tid] = lmax;
  __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) {
    if (tid < s) sval[tid] = fmaxf(sval[tid], sval[tid + s]);
    __syncthreads();
  }
  float m = sval[0];
  __syncthreads();

  // pass 2: per-thread exp-weight sum over its chunk; prefix -> base[], total
  // Z. w_i in (0,1] since the max logit gives w = exp(0) = 1.
  __shared__ float base[kSampleThreads];
  __shared__ float total;  // Z first, then the nucleus mass
  float lsum = 0.0f;
  for (int i = lo; i < hi; ++i) lsum += expf((lg[i] - m) * it);
  sval[tid] = lsum;
  __syncthreads();
  if (tid == 0) {
    float acc = 0.0f;
    for (int t = 0; t < nt; ++t) {
      base[t] = acc;
      acc += sval[t];
    }
    total = acc;
  }
  __syncthreads();
  float Z = total;

  // nucleus: binary-search the largest weight threshold wt whose kept mass
  // still covers top_p*Z (largest wt => smallest set), then rebuild
  // base[]/total over the kept tokens. wt stays 0 (keep all) when top_p >= 1.
  float wt = 0.0f;
  if (topp[b] < 1.0f) {
    float need = topp[b] * Z, wlo = 0.0f, whi = 1.0f;
    for (int it2 = 0; it2 < 32; ++it2) {
      float mid = 0.5f * (wlo + whi);
      float mmass = 0.0f;
      for (int i = lo; i < hi; ++i) {
        float w = expf((lg[i] - m) * it);
        if (w >= mid) mmass += w;
      }
      sval[tid] = mmass;
      __syncthreads();
      for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) sval[tid] += sval[tid + s];
        __syncthreads();
      }
      float mass = sval[0];
      __syncthreads();
      if (mass >= need)
        wlo = mid;
      else
        whi = mid;
    }
    wt = wlo;
    float km = 0.0f;
    for (int i = lo; i < hi; ++i) {
      float w = expf((lg[i] - m) * it);
      if (w >= wt) km += w;
    }
    sval[tid] = km;
    __syncthreads();
    if (tid == 0) {
      float acc = 0.0f;
      for (int t = 0; t < nt; ++t) {
        base[t] = acc;
        acc += sval[t];
      }
      total = acc;
    }
    __syncthreads();
  }

  // inverse-CDF within the kept set (all tokens when wt == 0): the one thread
  // whose [base, base+sum) interval straddles the target re-scans its chunk in
  // token-id order.
  __shared__ int result;
  if (tid == 0) result = N - 1;  // defensive fallback (target ~ total)
  __syncthreads();
  float target = u[b] * total;
  if (target >= base[tid] && target < base[tid] + sval[tid]) {
    float acc = base[tid];
    for (int i = lo; i < hi; ++i) {
      float w = expf((lg[i] - m) * it);
      if (w >= wt) {
        acc += w;
        if (acc >= target) {
          result = i;
          break;
        }
      }
    }
  }
  __syncthreads();
  if (tid == 0) out[b] = result;
}

// Greedy and sampled rows are disjoint (split by invT[b] sign), each kernel
// returns early for rows it does not own, so the two launches over B blocks
// together fill out[0..B).
void LaunchSampleBatch(const float* logits, int B, int N, const float* invT,
                       const float* topp, const float* u, int* out,
                       cudaStream_t s) {
  ArgmaxKernel<<<B, kSampleThreads, 0, s>>>(logits, N, invT, out);
  SampleKernel<<<B, kSampleThreads, 0, s>>>(logits, N, invT, topp, u, out);
}
