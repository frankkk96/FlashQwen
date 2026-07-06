#include "attn_cute.h"
#include "cute/tensor.hpp"
#include "kernels.cuh"
#include "kv_store.h"
#include "model_spec.h"

namespace fq {

using namespace cute;
using cbf16 = cutlass::bfloat16_t;

namespace {
using S = ModelSpec;
constexpr int kGqaGroup = S::kNumHeads / S::kNumKvHeads;
constexpr int kDimPerLane = S::kHeadDim / 32;
constexpr int kQTile = 64;
constexpr int kWarps = 4;

template <typename Layout>
__device__ __forceinline__ auto AccRowcol(Layout l) {
  using namespace cute;
  auto d = logical_divide(l, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(d), get<1>(d)),
                     make_layout(get<0, 0>(d), get<2>(d)));
}
}

static __global__ void __launch_bounds__(128, 4)
CutePrefillKernel(const cbf16* __restrict__ q,
                                  const cbf16* __restrict__ cache_kv,
                                  cbf16* __restrict__ out,
                                  const int* __restrict__ pos,
                                  const int* __restrict__ qstart,
                                  const int* __restrict__ qlen,
                                  const int* __restrict__ rids,
                                  const int* __restrict__ bt, int max_blocks) {
  const float scale = rsqrtf(static_cast<float>(S::kHeadDim));
  int r = rids[blockIdx.z], h = blockIdx.y, qtile = blockIdx.x;
  int ql = qlen[r], qs = qstart[r];
  int kv_dim = S::kNumKvHeads * S::kHeadDim;
  int qkv_dim = S::kNumHeads * S::kHeadDim + 2 * kv_dim;
  int64_t plane = static_cast<int64_t>(S::kKvBlock) * kv_dim;
  int kvh = h / kGqaGroup;
  const int* btr = bt + r * max_blocks;
  int q0 = qtile * kQTile;
  if (q0 >= ql) return;
  int tid = threadIdx.x;

  extern __shared__ char smem[];
  cbf16* sK = reinterpret_cast<cbf16*>(smem);
  cbf16* sV = sK + S::kKvBlock * S::kHeadDim;
  cbf16* sP = sV + S::kKvBlock * S::kHeadDim;
  cbf16* sQ = reinterpret_cast<cbf16*>(smem);

  const int4 z4 = {0, 0, 0, 0};
  for (int c = tid; c < kQTile * S::kHeadDim / 8; c += blockDim.x) {
    int row = c / (S::kHeadDim / 8), hd8 = (c % (S::kHeadDim / 8)) * 8, grow = q0 + row;
    *reinterpret_cast<int4*>(&sQ[row * S::kHeadDim + hd8]) =
        grow < ql ? *reinterpret_cast<const int4*>(&q[(qs + grow) * qkv_dim + h * S::kHeadDim + hd8]) : z4;
  }
  __syncthreads();

  TiledMMA mmaQK = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_4, _1, _1>>{});
  TiledMMA mmaPV = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_4, _1, _1>>{});
  auto thrQK = mmaQK.get_thread_slice(tid);
  auto thrPV = mmaPV.get_thread_slice(tid);

  Tensor sQt = make_tensor(make_smem_ptr(sQ), make_layout(Shape<Int<kQTile>, Int<S::kHeadDim>>{}, LayoutRight{}));
  Tensor sKt = make_tensor(make_smem_ptr(sK), make_layout(Shape<Int<S::kKvBlock>, Int<S::kHeadDim>>{}, LayoutRight{}));
  Tensor sPt = make_tensor(make_smem_ptr(sP), make_layout(Shape<Int<kQTile>, Int<S::kKvBlock>>{}, LayoutRight{}));
  Tensor sVt = make_tensor(make_smem_ptr(sV), make_layout(Shape<Int<S::kHeadDim>, Int<S::kKvBlock>>{}, Stride<_1, Int<S::kHeadDim>>{}));

  Tensor tSrQ = thrQK.partition_fragment_A(sQt);
  copy(thrQK.partition_A(sQt), tSrQ);
  __syncthreads();

  Tensor tOrO = partition_fragment_C(mmaPV, Shape<Int<kQTile>, Int<S::kHeadDim>>{});
  clear(tOrO);
  Tensor tOcO = thrPV.partition_C(make_identity_tensor(Shape<Int<kQTile>, Int<S::kHeadDim>>{}));
  Tensor O_rc = make_tensor(tOrO.data(), AccRowcol(tOrO.layout()));
  Tensor cO_rc = make_tensor(tOcO.data(), AccRowcol(tOcO.layout()));
  Tensor tScS = thrQK.partition_C(make_identity_tensor(Shape<Int<kQTile>, Int<S::kKvBlock>>{}));
  Tensor cS_rc = make_tensor(tScS.data(), AccRowcol(tScS.layout()));
  constexpr int NROW = 2, NSC = decltype(size<1>(cS_rc))::value, NOC = decltype(size<1>(O_rc))::value;
  float rm[NROW], rl[NROW];
  for (int r = 0; r < NROW; ++r) { rm[r] = -1e30f; rl[r] = 0.f; }

  int qlast = min(q0 + kQTile - 1, ql - 1);
  int ntiles = pos[qs + qlast] / S::kKvBlock + 1;

  for (int kt = 0; kt < ntiles; ++kt) {
    int64_t kvbase = static_cast<int64_t>(btr[kt]) * KvStore::kKvPlanes * plane +
                     static_cast<int64_t>(kvh) * S::kHeadDim;
    for (int c = tid; c < S::kKvBlock * S::kHeadDim / 8; c += blockDim.x) {
      int key = c / (S::kHeadDim / 8), hd8 = (c % (S::kHeadDim / 8)) * 8;
      *reinterpret_cast<int4*>(&sK[key * S::kHeadDim + hd8]) = *reinterpret_cast<const int4*>(
          &cache_kv[kvbase + static_cast<int64_t>(key) * kv_dim + hd8]);
      *reinterpret_cast<int4*>(&sV[key * S::kHeadDim + hd8]) = *reinterpret_cast<const int4*>(
          &cache_kv[kvbase + plane + static_cast<int64_t>(key) * kv_dim + hd8]);
    }
    __syncthreads();

    Tensor tSrK = thrQK.partition_fragment_B(sKt);
    Tensor tSrS = partition_fragment_C(mmaQK, Shape<Int<kQTile>, Int<S::kKvBlock>>{});
    clear(tSrS);
    copy(thrQK.partition_B(sKt), tSrK);
    gemm(mmaQK, tSrQ, tSrK, tSrS);

    Tensor S_rc = make_tensor(tSrS.data(), AccRowcol(tSrS.layout()));
    for (int r = 0; r < NROW; ++r) {
      int row = get<0>(cS_rc(r, 0)), grow = q0 + row;
      int qp = grow < ql ? pos[qs + grow] : -1;
      float rmax = -1e30f;
      for (int c = 0; c < NSC; ++c) {
        int kpos = kt * S::kKvBlock + get<1>(cS_rc(r, c));
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
      for (int c = 0; c < NSC; ++c) sP[row * S::kKvBlock + get<1>(cS_rc(r, c))] = cbf16(S_rc(r, c));
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
      if (grow < ql) out[((qs + grow) * S::kNumHeads + h) * S::kHeadDim + col] = cbf16(O_rc(r, c) * inv);
    }
  }
}

void LaunchAttnPrefillCute(const __nv_bfloat16* q,
                           const __nv_bfloat16* cache_kv, __nv_bfloat16* out,
                           const int* pos, const int* qstart, const int* qlen,
                           const int* rids, int R, int max_qlen, const int* bt,
                           int max_blocks, cudaStream_t s) {
  if (R <= 0) return;
  dim3 grid((max_qlen + kQTile - 1) / kQTile, S::kNumHeads, R);
  int smem = kQTile * S::kHeadDim * sizeof(cbf16);
  CutePrefillKernel<<<grid, 128, smem, s>>>(
      reinterpret_cast<const cbf16*>(q),
      reinterpret_cast<const cbf16*>(cache_kv), reinterpret_cast<cbf16*>(out),
      pos, qstart, qlen, rids, bt, max_blocks);
}

static __global__ void CuteDecodeSplitKernel(
    const cbf16* __restrict__ q, const cbf16* __restrict__ cache_kv,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pa,
    const int* __restrict__ pos, const int* __restrict__ qstart,
    const int* __restrict__ decode_rids, const int* __restrict__ bt,
    int max_blocks, int ksplit) {
  const float scale = rsqrtf(static_cast<float>(S::kHeadDim));
  int kvh = blockIdx.x, di = blockIdx.y, sp = blockIdx.z;
  int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int r = decode_rids[di], flat = qstart[r], qpos = pos[flat];
  int kv_dim = S::kNumKvHeads * S::kHeadDim;
  int qkv_dim = S::kNumHeads * S::kHeadDim + 2 * kv_dim;
  int64_t plane = static_cast<int64_t>(S::kKvBlock) * kv_dim;
  const int* btr = bt + r * max_blocks;

  float qreg[kGqaGroup][kDimPerLane];
  for (int g = 0; g < kGqaGroup; ++g) {
    const cbf16* qv = q + static_cast<int64_t>(flat) * qkv_dim +
                      static_cast<int64_t>(kvh * kGqaGroup + g) * S::kHeadDim;
    for (int i = 0; i < kDimPerLane; ++i) qreg[g][i] = float(qv[lane + i * 32]);
  }
  float m[kGqaGroup], l[kGqaGroup], acc[kGqaGroup][kDimPerLane];
  for (int g = 0; g < kGqaGroup; ++g) { m[g] = -1e30f; l[g] = 0.f;
    for (int i = 0; i < kDimPerLane; ++i) acc[g][i] = 0.f; }

  for (int kpos = sp * kWarps + w; kpos <= qpos; kpos += kWarps * ksplit) {
    int64_t base = static_cast<int64_t>(btr[kpos / S::kKvBlock]) * KvStore::kKvPlanes * plane +
                   static_cast<int64_t>(kpos % S::kKvBlock) * kv_dim +
                   static_cast<int64_t>(kvh) * S::kHeadDim;
    Tensor K = make_tensor(make_gmem_ptr(cache_kv + base), make_layout(Shape<Int<S::kHeadDim>>{}));
    Tensor V = make_tensor(make_gmem_ptr(cache_kv + base + plane), make_layout(Shape<Int<S::kHeadDim>>{}));
    float kreg[kDimPerLane], vreg[kDimPerLane];
    for (int i = 0; i < kDimPerLane; ++i) { kreg[i] = float(K(lane + i * 32)); vreg[i] = float(V(lane + i * 32)); }
    for (int g = 0; g < kGqaGroup; ++g) {
      float p = 0; for (int i = 0; i < kDimPerLane; ++i) p += qreg[g][i] * kreg[i];
      for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffff, p, o);
      float score = p * scale;
      float nm = fmaxf(m[g], score), corr = __expf(m[g] - nm), pp = __expf(score - nm);
      l[g] = l[g] * corr + pp;
      for (int i = 0; i < kDimPerLane; ++i) acc[g][i] = acc[g][i] * corr + pp * vreg[i];
      m[g] = nm;
    }
  }

  __shared__ float sm[kGqaGroup][kWarps], sl[kGqaGroup][kWarps], sa[kGqaGroup][kWarps][S::kHeadDim];
  for (int g = 0; g < kGqaGroup; ++g) { sm[g][w] = m[g]; sl[g][w] = l[g];
    for (int i = 0; i < kDimPerLane; ++i) sa[g][w][lane + i * 32] = acc[g][i]; }
  __syncthreads();
  if (w == 0) {
    for (int g = 0; g < kGqaGroup; ++g) {
      float gm = -1e30f; for (int t = 0; t < kWarps; ++t) gm = fmaxf(gm, sm[g][t]);
      float gl = 0.f, gacc[kDimPerLane]; for (int i = 0; i < kDimPerLane; ++i) gacc[i] = 0.f;
      for (int t = 0; t < kWarps; ++t) { float c = __expf(sm[g][t] - gm); gl += sl[g][t] * c;
        for (int i = 0; i < kDimPerLane; ++i) gacc[i] += sa[g][t][lane + i * 32] * c; }
      int h = kvh * kGqaGroup + g;
      int64_t idx = (static_cast<int64_t>(di) * S::kNumHeads + h) * ksplit + sp;
      if (lane == 0) { pm[idx] = gm; pl[idx] = gl; }
      for (int i = 0; i < kDimPerLane; ++i) pa[idx * S::kHeadDim + lane + i * 32] = gacc[i];
    }
  }
}

static __global__ void CuteDecodeCombineKernel(
    const float* __restrict__ pm, const float* __restrict__ pl,
    const float* __restrict__ pa, cbf16* __restrict__ out,
    const int* __restrict__ qstart, const int* __restrict__ decode_rids, int ksplit) {
  int h = blockIdx.x, di = blockIdx.y, d = threadIdx.x;
  int r = decode_rids[di], flat = qstart[r];
  int64_t base = (static_cast<int64_t>(di) * S::kNumHeads + h) * ksplit;
  float gm = -1e30f; for (int s = 0; s < ksplit; ++s) gm = fmaxf(gm, pm[base + s]);
  float gl = 0.f, gacc = 0.f;
  for (int s = 0; s < ksplit; ++s) { float c = __expf(pm[base + s] - gm);
    gl += pl[base + s] * c; gacc += pa[(base + s) * S::kHeadDim + d] * c; }
  float inv = gl > 0 ? 1.f / gl : 0.f;
  out[(static_cast<int64_t>(flat) * S::kNumHeads + h) * S::kHeadDim + d] = cbf16(gacc * inv);
}

void LaunchAttnDecodeCute(const __nv_bfloat16* q,
                          const __nv_bfloat16* cache_kv, __nv_bfloat16* out,
                          const int* pos, const int* qstart,
                          const int* decode_rids, int n_decode, const int* bt,
                          int max_blocks, float* pm, float* pl,
                          float* pa, cudaStream_t s) {
  if (n_decode <= 0) return;
  int ksplit = 128 / n_decode;
  if (ksplit < 1) ksplit = 1;
  if (ksplit > kMaxKsplit) ksplit = kMaxKsplit;

  dim3 g1(S::kNumKvHeads, n_decode, ksplit);
  CuteDecodeSplitKernel<<<g1, kWarps * 32, 0, s>>>(
      reinterpret_cast<const cbf16*>(q),
      reinterpret_cast<const cbf16*>(cache_kv), pm, pl, pa, pos, qstart,
      decode_rids, bt, max_blocks, ksplit);
  dim3 g2(S::kNumHeads, n_decode);
  CuteDecodeCombineKernel<<<g2, S::kHeadDim, 0, s>>>(
      pm, pl, pa, reinterpret_cast<cbf16*>(out), qstart, decode_rids, ksplit);
}

}
