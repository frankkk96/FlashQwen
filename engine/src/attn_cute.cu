#include <cassert>

#include "attn_cute.h"
#include "cute/tensor.hpp"
#include "kernels.cuh"
#include "kv_layout.h"

namespace fq {

using namespace cute;
using cbf16 = cutlass::bfloat16_t;

namespace {
constexpr int HD = 128;
constexpr int TQ = 64;
constexpr int TK = 16;
constexpr int G = 4;
constexpr int DPL = 4;
constexpr int NW = 4;

template <typename Layout>
__device__ __forceinline__ auto AccRowcol(Layout l) {
  using namespace cute;
  auto d = logical_divide(l, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(d), get<1>(d)),
                     make_layout(get<0, 0>(d), get<2>(d)));
}
}

static __global__ void __launch_bounds__(128, 4)
CutePrefillKernel(const cbf16* __restrict__ q, int q_stride,
                                  const cbf16* __restrict__ cache_kv,
                                  cbf16* __restrict__ out, int n_heads, int n_kv,
                                  const int* __restrict__ pos,
                                  const int* __restrict__ qstart,
                                  const int* __restrict__ qlen,
                                  const int* __restrict__ rids,
                                  const int* __restrict__ bt, int max_blocks,
                                  float scale) {
  int r = rids[blockIdx.z], h = blockIdx.y, qtile = blockIdx.x;
  int ql = qlen[r], qs = qstart[r];
  int kv_dim = n_kv * HD;
  int64_t plane = static_cast<int64_t>(TK) * kv_dim;
  int kvh = h / (n_heads / n_kv);
  const int* btr = bt + r * max_blocks;
  int q0 = qtile * TQ;
  if (q0 >= ql) return;
  int tid = threadIdx.x;

  extern __shared__ char smem[];
  cbf16* sK = reinterpret_cast<cbf16*>(smem);
  cbf16* sV = sK + TK * HD;
  cbf16* sP = sV + TK * HD;
  cbf16* sQ = reinterpret_cast<cbf16*>(smem);

  const int4 z4 = {0, 0, 0, 0};
  for (int c = tid; c < TQ * HD / 8; c += blockDim.x) {
    int row = c / (HD / 8), hd8 = (c % (HD / 8)) * 8, grow = q0 + row;
    *reinterpret_cast<int4*>(&sQ[row * HD + hd8]) =
        grow < ql ? *reinterpret_cast<const int4*>(&q[(qs + grow) * q_stride + h * HD + hd8]) : z4;
  }
  __syncthreads();

  TiledMMA mmaQK = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_4, _1, _1>>{});
  TiledMMA mmaPV = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_4, _1, _1>>{});
  auto thrQK = mmaQK.get_thread_slice(tid);
  auto thrPV = mmaPV.get_thread_slice(tid);

  Tensor sQt = make_tensor(make_smem_ptr(sQ), make_layout(Shape<Int<TQ>, Int<HD>>{}, LayoutRight{}));
  Tensor sKt = make_tensor(make_smem_ptr(sK), make_layout(Shape<Int<TK>, Int<HD>>{}, LayoutRight{}));
  Tensor sPt = make_tensor(make_smem_ptr(sP), make_layout(Shape<Int<TQ>, Int<TK>>{}, LayoutRight{}));
  Tensor sVt = make_tensor(make_smem_ptr(sV), make_layout(Shape<Int<HD>, Int<TK>>{}, Stride<_1, Int<HD>>{}));

  Tensor tSrQ = thrQK.partition_fragment_A(sQt);
  copy(thrQK.partition_A(sQt), tSrQ);
  __syncthreads();

  Tensor tOrO = partition_fragment_C(mmaPV, Shape<Int<TQ>, Int<HD>>{});
  clear(tOrO);
  Tensor tOcO = thrPV.partition_C(make_identity_tensor(Shape<Int<TQ>, Int<HD>>{}));
  Tensor O_rc = make_tensor(tOrO.data(), AccRowcol(tOrO.layout()));
  Tensor cO_rc = make_tensor(tOcO.data(), AccRowcol(tOcO.layout()));
  Tensor tScS = thrQK.partition_C(make_identity_tensor(Shape<Int<TQ>, Int<TK>>{}));
  Tensor cS_rc = make_tensor(tScS.data(), AccRowcol(tScS.layout()));
  constexpr int NROW = 2, NSC = decltype(size<1>(cS_rc))::value, NOC = decltype(size<1>(O_rc))::value;
  float rm[NROW], rl[NROW];
  for (int r = 0; r < NROW; ++r) { rm[r] = -1e30f; rl[r] = 0.f; }

  int qlast = min(q0 + TQ - 1, ql - 1);
  int ntiles = pos[qs + qlast] / TK + 1;

  for (int kt = 0; kt < ntiles; ++kt) {
    int64_t kvbase = static_cast<int64_t>(btr[kt]) * kKvPlanes * plane +
                     static_cast<int64_t>(kvh) * HD;
    for (int c = tid; c < TK * HD / 8; c += blockDim.x) {
      int key = c / (HD / 8), hd8 = (c % (HD / 8)) * 8;
      *reinterpret_cast<int4*>(&sK[key * HD + hd8]) = *reinterpret_cast<const int4*>(
          &cache_kv[kvbase + static_cast<int64_t>(key) * kv_dim + hd8]);
      *reinterpret_cast<int4*>(&sV[key * HD + hd8]) = *reinterpret_cast<const int4*>(
          &cache_kv[kvbase + plane + static_cast<int64_t>(key) * kv_dim + hd8]);
    }
    __syncthreads();

    Tensor tSrK = thrQK.partition_fragment_B(sKt);
    Tensor tSrS = partition_fragment_C(mmaQK, Shape<Int<TQ>, Int<TK>>{});
    clear(tSrS);
    copy(thrQK.partition_B(sKt), tSrK);
    gemm(mmaQK, tSrQ, tSrK, tSrS);

    Tensor S_rc = make_tensor(tSrS.data(), AccRowcol(tSrS.layout()));
    for (int r = 0; r < NROW; ++r) {
      int row = get<0>(cS_rc(r, 0)), grow = q0 + row;
      int qp = grow < ql ? pos[qs + grow] : -1;
      float rmax = -1e30f;
      for (int c = 0; c < NSC; ++c) {
        int kpos = kt * TK + get<1>(cS_rc(r, c));
        float v = (kpos <= qp) ? S_rc(r, c) * scale : -1e30f;
        S_rc(r, c) = v; rmax = fmaxf(rmax, v);
      }
      rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffff, rmax, 1));
      rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffff, rmax, 2));
      float nm = fmaxf(rm[r], rmax), corr = __expf(rm[r] - nm), rsum = 0.f;
      for (int c = 0; c < NSC; ++c) { float p = __expf(S_rc(r, c) - nm); S_rc(r, c) = p; rsum += p; }
      rsum += __shfl_xor_sync(0xffffffff, rsum, 1);
      rsum += __shfl_xor_sync(0xffffffff, rsum, 2);
      rl[r] = rl[r] * corr + rsum; rm[r] = nm;
      for (int c = 0; c < NOC; ++c) O_rc(r, c) *= corr;
      for (int c = 0; c < NSC; ++c) sP[row * TK + get<1>(cS_rc(r, c))] = cbf16(S_rc(r, c));
    }
    __syncthreads();

    Tensor tOrP = thrPV.partition_fragment_A(sPt);
    Tensor tOrV = thrPV.partition_fragment_B(sVt);
    copy(thrPV.partition_A(sPt), tOrP);
    copy(thrPV.partition_B(sVt), tOrV);
    gemm(mmaPV, tOrP, tOrV, tOrO);
    __syncthreads();
  }

  for (int r = 0; r < NROW; ++r) {
    float inv = rl[r] > 0 ? 1.f / rl[r] : 0.f;
    for (int c = 0; c < NOC; ++c) {
      int row = get<0>(cO_rc(r, c)), col = get<1>(cO_rc(r, c)), grow = q0 + row;
      if (grow < ql) out[((qs + grow) * n_heads + h) * HD + col] = cbf16(O_rc(r, c) * inv);
    }
  }
}

void LaunchAttnPrefillCute(const __nv_bfloat16* q, int q_stride,
                           const __nv_bfloat16* cache_kv, __nv_bfloat16* out,
                           int n_heads, int n_kv, int head_dim, const int* pos,
                           const int* qstart, const int* qlen, const int* rids,
                           int R, int max_qlen, const int* bt, int max_blocks,
                           int block_size, float scale, cudaStream_t s) {
  if (R <= 0) return;
  assert(head_dim == HD && block_size == TK);
  dim3 grid((max_qlen + TQ - 1) / TQ, n_heads, R);
  int smem = TQ * HD * sizeof(cbf16);
  CutePrefillKernel<<<grid, 128, smem, s>>>(
      reinterpret_cast<const cbf16*>(q), q_stride,
      reinterpret_cast<const cbf16*>(cache_kv), reinterpret_cast<cbf16*>(out),
      n_heads, n_kv, pos, qstart, qlen, rids, bt, max_blocks, scale);
}

static __global__ void CuteDecodeSplitKernel(
    const cbf16* __restrict__ q, int q_stride, const cbf16* __restrict__ cache_kv,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pa,
    int n_heads, int n_kv, const int* __restrict__ pos,
    const int* __restrict__ qstart, const int* __restrict__ decode_rids,
    const int* __restrict__ bt, int max_blocks, float scale, int ksplit) {
  int kvh = blockIdx.x, di = blockIdx.y, sp = blockIdx.z;
  int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int r = decode_rids[di], flat = qstart[r], qpos = pos[flat];
  int kv_dim = n_kv * HD;
  int64_t plane = static_cast<int64_t>(TK) * kv_dim;
  const int* btr = bt + r * max_blocks;

  float qreg[G][DPL];
  for (int g = 0; g < G; ++g) {
    const cbf16* qv = q + static_cast<int64_t>(flat) * q_stride +
                      static_cast<int64_t>(kvh * G + g) * HD;
    for (int i = 0; i < DPL; ++i) qreg[g][i] = float(qv[lane + i * 32]);
  }
  float m[G], l[G], acc[G][DPL];
  for (int g = 0; g < G; ++g) { m[g] = -1e30f; l[g] = 0.f;
    for (int i = 0; i < DPL; ++i) acc[g][i] = 0.f; }

  for (int kpos = sp * NW + w; kpos <= qpos; kpos += NW * ksplit) {
    int64_t base = static_cast<int64_t>(btr[kpos / TK]) * kKvPlanes * plane +
                   static_cast<int64_t>(kpos % TK) * kv_dim +
                   static_cast<int64_t>(kvh) * HD;
    Tensor K = make_tensor(make_gmem_ptr(cache_kv + base), make_layout(Shape<Int<HD>>{}));
    Tensor V = make_tensor(make_gmem_ptr(cache_kv + base + plane), make_layout(Shape<Int<HD>>{}));
    float kreg[DPL], vreg[DPL];
    for (int i = 0; i < DPL; ++i) { kreg[i] = float(K(lane + i * 32)); vreg[i] = float(V(lane + i * 32)); }
    for (int g = 0; g < G; ++g) {
      float p = 0; for (int i = 0; i < DPL; ++i) p += qreg[g][i] * kreg[i];
      for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffff, p, o);
      float score = p * scale;
      float nm = fmaxf(m[g], score), corr = __expf(m[g] - nm), pp = __expf(score - nm);
      l[g] = l[g] * corr + pp;
      for (int i = 0; i < DPL; ++i) acc[g][i] = acc[g][i] * corr + pp * vreg[i];
      m[g] = nm;
    }
  }

  __shared__ float sm[G][NW], sl[G][NW], sa[G][NW][HD];
  for (int g = 0; g < G; ++g) { sm[g][w] = m[g]; sl[g][w] = l[g];
    for (int i = 0; i < DPL; ++i) sa[g][w][lane + i * 32] = acc[g][i]; }
  __syncthreads();
  if (w == 0) {
    for (int g = 0; g < G; ++g) {
      float gm = -1e30f; for (int t = 0; t < NW; ++t) gm = fmaxf(gm, sm[g][t]);
      float gl = 0.f, gacc[DPL]; for (int i = 0; i < DPL; ++i) gacc[i] = 0.f;
      for (int t = 0; t < NW; ++t) { float c = __expf(sm[g][t] - gm); gl += sl[g][t] * c;
        for (int i = 0; i < DPL; ++i) gacc[i] += sa[g][t][lane + i * 32] * c; }
      int h = kvh * G + g;
      int64_t idx = (static_cast<int64_t>(di) * n_heads + h) * ksplit + sp;
      if (lane == 0) { pm[idx] = gm; pl[idx] = gl; }
      for (int i = 0; i < DPL; ++i) pa[idx * HD + lane + i * 32] = gacc[i];
    }
  }
}

static __global__ void CuteDecodeCombineKernel(
    const float* __restrict__ pm, const float* __restrict__ pl,
    const float* __restrict__ pa, cbf16* __restrict__ out, int n_heads,
    const int* __restrict__ qstart, const int* __restrict__ decode_rids, int ksplit) {
  int h = blockIdx.x, di = blockIdx.y, d = threadIdx.x;
  int r = decode_rids[di], flat = qstart[r];
  int64_t base = (static_cast<int64_t>(di) * n_heads + h) * ksplit;
  float gm = -1e30f; for (int s = 0; s < ksplit; ++s) gm = fmaxf(gm, pm[base + s]);
  float gl = 0.f, gacc = 0.f;
  for (int s = 0; s < ksplit; ++s) { float c = __expf(pm[base + s] - gm);
    gl += pl[base + s] * c; gacc += pa[(base + s) * HD + d] * c; }
  float inv = gl > 0 ? 1.f / gl : 0.f;
  out[(static_cast<int64_t>(flat) * n_heads + h) * HD + d] = cbf16(gacc * inv);
}

void LaunchAttnDecodeCute(const __nv_bfloat16* q, int q_stride,
                          const __nv_bfloat16* cache_kv, __nv_bfloat16* out,
                          int n_heads, int n_kv, int head_dim, const int* pos,
                          const int* qstart, const int* decode_rids, int n_decode,
                          const int* bt, int max_blocks, int block_size,
                          float scale, float* pm, float* pl, float* pa,
                          cudaStream_t s) {
  if (n_decode <= 0) return;
  assert(head_dim == HD && n_heads / n_kv == G && block_size == TK);
  int ksplit = 128 / n_decode;
  if (ksplit < 1) ksplit = 1;
  if (ksplit > kMaxKsplit) ksplit = kMaxKsplit;

  dim3 g1(n_kv, n_decode, ksplit);
  CuteDecodeSplitKernel<<<g1, NW * 32, 0, s>>>(
      reinterpret_cast<const cbf16*>(q), q_stride,
      reinterpret_cast<const cbf16*>(cache_kv), pm, pl, pa, n_heads, n_kv, pos,
      qstart, decode_rids, bt, max_blocks, scale, ksplit);
  dim3 g2(n_heads, n_decode);
  CuteDecodeCombineKernel<<<g2, HD, 0, s>>>(
      pm, pl, pa, reinterpret_cast<cbf16*>(out), n_heads, qstart, decode_rids, ksplit);
}

}
