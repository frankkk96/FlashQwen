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
  const bf16* xr = x + m * H;
  bf16* outr = out + m * H;

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
  bf16* xr = x + m * H;
  const bf16* rr = res + m * H;
  bf16* outr = out + m * H;

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
  const bf16* src = embed + ids[m] * H;
  bf16* dst = out + m * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x) dst[i] = src[i];
}

void LaunchEmbed(const int* ids, const bf16* embed, bf16* out, int M, int H,
                 cudaStream_t s) {
  EmbedKernel<<<M, 256, 0, s>>>(ids, embed, out, M, H);
}

// Fused per-head RMSNorm + RoPE (rotate-half, matching HF Qwen) over the q AND
// k slices of a fused QKV row of width `stride` in one launch: the q heads
// (weight wq) and k heads (weight wk) are contiguous, so head g of token m
// lives at buf + m*stride + g*head_dim for g in [0, n_q+n_kv); g < n_q is a q
// head, else a k head. v is left untouched. Angles come from precomputed
// cos/sin tables ([max_pos, head_dim/2]) indexed by pos[m] (no per-call
// transcendental; tables are identical across all layers and built once). One
// block per (token, head); blockDim == head_dim. FP32 math.
__global__ void HeadNormRopeKernel(
    bf16* __restrict__ buf, const float* __restrict__ wq,
    const float* __restrict__ wk, const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab, const int* __restrict__ pos, int n_q,
    int n_kv, int head_dim, int stride, float eps) {
  int hpr = n_q + n_kv;  // q + k heads per row
  int row = blockIdx.x;  // over M * hpr
  int m = row / hpr, g = row % hpr;
  const float* w = g < n_q ? wq : wk;
  int t = threadIdx.x;  // 0..head_dim-1
  int half = head_dim >> 1;
  bf16* v = buf + m * stride + g * head_dim;

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
    const float* cb = cos_tab + pos[m] * half;
    const float* sb = sin_tab + pos[m] * half;
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
  HeadNormRopeKernel<<<M*(n_q + n_kv), head_dim, head_dim * sizeof(float), s>>>(
      buf, wq, wk, cos_tab, sin_tab, pos, n_q, n_kv, head_dim, stride, eps);
}

// Paged-KV attention (prefill + decode) lives in attn_cute.cu (CuTe/CUTLASS).

// Gather S rows: out[i, :] = x[rows[i], :]   (one block per gathered row, BF16)
__global__ void GatherRowsKernel(const bf16* __restrict__ x,
                                 const int* __restrict__ rows,
                                 bf16* __restrict__ out, int H) {
  int i = blockIdx.x;
  const bf16* src = x + rows[i] * H;
  bf16* dst = out + i * H;
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
  if (idx >= M * I) return;
  int m = idx / I, i = idx % I;
  const bf16* row = gateup + m * 2 * I;
  float g = __bfloat162float(row[i]);
  h[idx] =
      __float2bfloat16((g / (1.0f + expf(-g))) * __bfloat162float(row[I + i]));
}
void LaunchSiluMul(const bf16* gateup, bf16* h, int M, int I, cudaStream_t s) {
  int block = 256;
  int N = M * I;
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
  const float* lg = logits + b * N;

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
  const float* lg = logits + b * N;

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
