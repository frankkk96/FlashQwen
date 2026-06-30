#include <cuda_pipeline.h>

#include <cassert>

#include "kernels.cuh"

namespace fq {

void LaunchGemm(cublasHandle_t handle, const bf16* x, const bf16* W, void* y,
                int M, int IN, int OUT, cudaDataType_t y_type) {
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, OUT, M, IN,
                            &alpha, W, CUDA_R_16BF, IN, x, CUDA_R_16BF, IN,
                            &beta, y, y_type, OUT, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}

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
    xr[i] = __float2bfloat16(v);
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

__global__ void HeadNormRopeKernel(
    bf16* __restrict__ buf, const float* __restrict__ wq,
    const float* __restrict__ wk, const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab, const int* __restrict__ pos, int n_q,
    int n_kv, int head_dim, int stride, float eps) {
  int hpr = n_q + n_kv;
  int row = blockIdx.x;
  int m = row / hpr, g = row % hpr;
  const float* w = g < n_q ? wq : wk;
  int t = threadIdx.x;
  int half = head_dim >> 1;
  bf16* v = buf + m * stride + g * head_dim;

  extern __shared__ float sh[];
  float x = __bfloat162float(v[t]);
  sh[t] = x * x;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (t < s) sh[t] += sh[t + s];
    __syncthreads();
  }
  float inv = rsqrtf(sh[0] / head_dim + eps);
  __syncthreads();
  sh[t] = x * inv * w[t];
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

  __shared__ float sval[kSampleThreads];
  int chunk = (N + nt - 1) / nt;
  int lo = min(tid * chunk, N), hi = min(lo + chunk, N);

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

  __shared__ float base[kSampleThreads];
  __shared__ float total;
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

  __shared__ int result;
  if (tid == 0) result = N - 1;
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

void LaunchSampleBatch(const float* logits, int B, int N, const float* invT,
                       const float* topp, const float* u, int* out,
                       cudaStream_t s) {
  ArgmaxKernel<<<B, kSampleThreads, 0, s>>>(logits, N, invT, out);
  SampleKernel<<<B, kSampleThreads, 0, s>>>(logits, N, invT, topp, u, out);
}

}
