#include "attn.h"
#include "cute/tensor.hpp"
#include "kernels.cuh"
#include "kv_store.h"
#include "model_spec.h"

namespace fq {

using namespace cute;
using cbf16 = cutlass::bfloat16_t;

namespace {
using S = ModelSpec;

constexpr int kWarpSize = 32;
constexpr int kQTile = 64;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kTile = S::kKvBlock * S::kHeadDim;
constexpr int kDimPerLane = S::kHeadDim / kWarpSize;
constexpr int kGqaGroup = S::kNumHeads / S::kNumKvHeads;
constexpr int kKvDim = S::kNumKvHeads * S::kHeadDim;
constexpr int kQkvDim = S::kNumHeads * S::kHeadDim + 2 * kKvDim;
constexpr int64_t kPlane = static_cast<int64_t>(S::kKvBlock) * kKvDim;

template <typename Layout>
__device__ __forceinline__ auto AccRowcol(Layout l) {
  auto d = logical_divide(l, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(d), get<1>(d)),
                     make_layout(get<0, 0>(d), get<2>(d)));
}

template <typename Layout>
__device__ __forceinline__ auto AccToAregs(Layout acc) {
  auto l = logical_divide(acc, Shape<Underscore, Underscore, _2>{});
  return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l),
                     get<2, 1>(l));
}

static __global__ void __launch_bounds__(128, 4)
    PrefillKernel(const cbf16* __restrict__ qkv,
                  const cbf16* __restrict__ cache_kv, cbf16* __restrict__ out,
                  const int* __restrict__ pos, const int* __restrict__ qstart,
                  const int* __restrict__ qlen, const int* __restrict__ rids,
                  const int* __restrict__ bt, int max_blocks) {
  int req = rids[blockIdx.z], h = blockIdx.y, qtile = blockIdx.x;
  int ql = qlen[req], qs = qstart[req];
  int q0 = qtile * kQTile;
  if (q0 >= ql) return;
  int kvh = h / kGqaGroup;
  const int* btr = bt + req * max_blocks;
  const float scale = rsqrtf(static_cast<float>(S::kHeadDim));

  extern __shared__ cbf16 smem[];
  const int tid = threadIdx.x;

  cbf16* sQ = smem;

  cbf16* sKring = smem;
  cbf16* sVring = sKring + 2 * kTile;

  using QCopyAtom =
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, cbf16>;
  auto q_copy =
      make_tiled_copy(QCopyAtom{}, Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                      Layout<Shape<_1, _16>>{});
  auto q_thr = q_copy.get_slice(tid);
  Tensor gQ =
      make_tensor(make_gmem_ptr(qkv + (static_cast<int64_t>(qs + q0) * kQkvDim +
                                       h * S::kHeadDim)),
                  make_layout(Shape<Int<kQTile>, Int<S::kHeadDim>>{},
                              make_stride(kQkvDim, Int<1>{})));
  Tensor sQd = make_tensor(
      make_smem_ptr(sQ),
      make_layout(Shape<Int<kQTile>, Int<S::kHeadDim>>{}, LayoutRight{}));
  Tensor cQ = make_identity_tensor(Shape<Int<kQTile>, Int<S::kHeadDim>>{});
  Tensor tQg = q_thr.partition_S(gQ);
  Tensor tQs = q_thr.partition_D(sQd);
  Tensor tQc = q_thr.partition_S(cQ);
  Tensor tQp = make_tensor<bool>(shape(tQs));
  CUTE_UNROLL
  for (int i = 0; i < size(tQp); ++i) tQp(i) = (q0 + get<0>(tQc(i))) < ql;
  copy_if(q_copy, tQp, tQg, tQs);
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  TiledMMA mmaQK = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                  Layout<Shape<Int<kWarps>, _1, _1>>{});
  auto thrQK = mmaQK.get_thread_slice(tid);
  Tensor sQt = make_tensor(
      make_smem_ptr(sQ),
      make_layout(Shape<Int<kQTile>, Int<S::kHeadDim>>{}, LayoutRight{}));
  Tensor tSrQ = thrQK.partition_fragment_A(sQt);
  copy(thrQK.partition_A(sQt), tSrQ);
  __syncthreads();
  Tensor tScS = thrQK.partition_C(
      make_identity_tensor(Shape<Int<kQTile>, Int<S::kKvBlock>>{}));
  Tensor cS_rc = make_tensor(tScS.data(), AccRowcol(tScS.layout()));

  TiledMMA mmaPV = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                  Layout<Shape<Int<kWarps>, _1, _1>>{});
  auto thrPV = mmaPV.get_thread_slice(tid);
  Tensor tOrO =
      partition_fragment_C(mmaPV, Shape<Int<kQTile>, Int<S::kHeadDim>>{});
  clear(tOrO);
  Tensor tOcO = thrPV.partition_C(
      make_identity_tensor(Shape<Int<kQTile>, Int<S::kHeadDim>>{}));
  Tensor O_rc = make_tensor(tOrO.data(), AccRowcol(tOrO.layout()));
  Tensor cO_rc = make_tensor(tOcO.data(), AccRowcol(tOcO.layout()));

  constexpr int NROW = decltype(size<0>(O_rc))::value;
  constexpr int NSC = decltype(size<1>(cS_rc))::value;
  constexpr int NOC = decltype(size<1>(O_rc))::value;
  float rm[NROW], rl[NROW];
  for (int i = 0; i < NROW; ++i) {
    rm[i] = kNegInf;
    rl[i] = 0.f;
  }

  int qlast = min(q0 + kQTile - 1, ql - 1);
  int n_blocks = pos[qs + qlast] / S::kKvBlock + 1;

  using GCopyAtom =
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cbf16>;
  auto gmem_copy = make_tiled_copy(
      GCopyAtom{}, Layout<Shape<Int<S::kKvBlock>, _8>, Stride<_8, _1>>{},
      Layout<Shape<_1, _16>>{});
  auto thr_copy = gmem_copy.get_slice(tid);
  auto tile_layout =
      make_layout(Shape<Int<S::kKvBlock>, Int<S::kHeadDim>>{}, LayoutRight{});
  auto paged_layout = make_layout(Shape<Int<S::kKvBlock>, Int<S::kHeadDim>>{},
                                  make_stride(kKvDim, Int<1>{}));

  auto load_block = [&](int kb, int st) {
    int64_t kvbase =
        static_cast<int64_t>(btr[kb]) * KvStore::kKvPlanes * kPlane +
        static_cast<int64_t>(kvh) * S::kHeadDim;
    Tensor gK = make_tensor(make_gmem_ptr(cache_kv + kvbase), paged_layout);
    Tensor gV =
        make_tensor(make_gmem_ptr(cache_kv + kvbase + kPlane), paged_layout);
    Tensor sK = make_tensor(make_smem_ptr(sKring + st * kTile), tile_layout);
    Tensor sV = make_tensor(make_smem_ptr(sVring + st * kTile), tile_layout);
    copy(gmem_copy, thr_copy.partition_S(gK), thr_copy.partition_D(sK));
    copy(gmem_copy, thr_copy.partition_S(gV), thr_copy.partition_D(sV));
  };

  if (n_blocks > 0) load_block(0, 0);
  cp_async_fence();

  int cur = 0;
  for (int kb = 0; kb < n_blocks; ++kb) {
    cp_async_wait<0>();
    __syncthreads();
    if (kb + 1 < n_blocks) load_block(kb + 1, cur ^ 1);
    cp_async_fence();

    Tensor sKt =
        make_tensor(make_smem_ptr(sKring + cur * kTile),
                    make_layout(Shape<Int<S::kKvBlock>, Int<S::kHeadDim>>{},
                                LayoutRight{}));
    Tensor sVt =
        make_tensor(make_smem_ptr(sVring + cur * kTile),
                    make_layout(Shape<Int<S::kHeadDim>, Int<S::kKvBlock>>{},
                                Stride<_1, Int<S::kHeadDim>>{}));

    Tensor tSrK = thrQK.partition_fragment_B(sKt);
    Tensor tSrS =
        partition_fragment_C(mmaQK, Shape<Int<kQTile>, Int<S::kKvBlock>>{});
    clear(tSrS);
    copy(thrQK.partition_B(sKt), tSrK);
    gemm(mmaQK, tSrQ, tSrK, tSrS);

    Tensor S_rc = make_tensor(tSrS.data(), AccRowcol(tSrS.layout()));
    for (int r = 0; r < NROW; ++r) {
      int row = get<0>(cS_rc(r, 0)), grow = q0 + row;
      int qpos = grow < ql ? pos[qs + grow] : -1;
      float rmax = kNegInf;
      for (int c = 0; c < NSC; ++c) {
        int kpos = kb * S::kKvBlock + get<1>(cS_rc(r, c));
        float v = (kpos <= qpos) ? S_rc(r, c) * scale : kNegInf;
        S_rc(r, c) = v;
        rmax = fmaxf(rmax, v);
      }
      rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffff, rmax, 1));
      rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffff, rmax, 2));
      float nm = fmaxf(rm[r], rmax), corr = __expf(rm[r] - nm), rsum = 0.f;
      for (int c = 0; c < NSC; ++c) {
        float p = __expf(S_rc(r, c) - nm);
        S_rc(r, c) = p;
        rsum += p;
      }
      rsum += __shfl_xor_sync(0xffffffff, rsum, 1);
      rsum += __shfl_xor_sync(0xffffffff, rsum, 2);
      rl[r] = rl[r] * corr + rsum;
      rm[r] = nm;
      for (int c = 0; c < NOC; ++c) O_rc(r, c) *= corr;
    }

    Tensor rP = make_tensor<cbf16>(tSrS.layout());
    CUTE_UNROLL
    for (int i = 0; i < size(rP); ++i) rP(i) = static_cast<cbf16>(tSrS(i));
    Tensor tOrP = make_tensor(rP.data(), AccToAregs(rP.layout()));
    Tensor tOrV = thrPV.partition_fragment_B(sVt);
    copy(thrPV.partition_B(sVt), tOrV);
    gemm(mmaPV, tOrP, tOrV, tOrO);
    __syncthreads();
    cur ^= 1;
  }

  for (int r = 0; r < NROW; ++r) {
    float inv = rl[r] > 0 ? 1.f / rl[r] : 0.f;
    for (int c = 0; c < NOC; ++c) {
      int row = get<0>(cO_rc(r, c)), col = get<1>(cO_rc(r, c)), grow = q0 + row;
      if (grow < ql)
        out[((qs + grow) * S::kNumHeads + h) * S::kHeadDim + col] =
            cbf16(O_rc(r, c) * inv);
    }
  }
}

static __global__ void DecodeSplitKernel(
    const cbf16* __restrict__ qkv, const cbf16* __restrict__ cache_kv,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pa,
    const int* __restrict__ pos, const int* __restrict__ qstart,
    const int* __restrict__ decode_rids, const int* __restrict__ bt,
    int max_blocks, int ksplit) {
  int kvh = blockIdx.x, di = blockIdx.y, sp = blockIdx.z;
  const float scale = rsqrtf(static_cast<float>(S::kHeadDim));

  int w = threadIdx.x / kWarpSize, lane = threadIdx.x % kWarpSize;

  int r = decode_rids[di], flat = qstart[r], qpos = pos[flat];
  const int* btr = bt + r * max_blocks;

  float qreg[kGqaGroup][kDimPerLane];
  for (int g = 0; g < kGqaGroup; ++g) {
    const cbf16* qv = qkv + static_cast<int64_t>(flat) * kQkvDim +
                      static_cast<int64_t>(kvh * kGqaGroup + g) * S::kHeadDim;
    for (int i = 0; i < kDimPerLane; ++i)
      qreg[g][i] = float(qv[lane + i * kWarpSize]);
  }
  float m[kGqaGroup], l[kGqaGroup], acc[kGqaGroup][kDimPerLane];
  for (int g = 0; g < kGqaGroup; ++g) {
    m[g] = kNegInf;
    l[g] = 0.f;
    for (int i = 0; i < kDimPerLane; ++i) acc[g][i] = 0.f;
  }

  for (int kpos = sp * kWarps + w; kpos <= qpos; kpos += kWarps * ksplit) {
    int64_t base = static_cast<int64_t>(btr[kpos / S::kKvBlock]) *
                       KvStore::kKvPlanes * kPlane +
                   static_cast<int64_t>(kpos % S::kKvBlock) * kKvDim +
                   static_cast<int64_t>(kvh) * S::kHeadDim;
    Tensor K = make_tensor(make_gmem_ptr(cache_kv + base),
                           make_layout(Shape<Int<S::kHeadDim>>{}));
    Tensor V = make_tensor(make_gmem_ptr(cache_kv + base + kPlane),
                           make_layout(Shape<Int<S::kHeadDim>>{}));
    float kreg[kDimPerLane], vreg[kDimPerLane];
    for (int i = 0; i < kDimPerLane; ++i) {
      kreg[i] = float(K(lane + i * kWarpSize));
      vreg[i] = float(V(lane + i * kWarpSize));
    }

    for (int g = 0; g < kGqaGroup; ++g) {
      float p = 0;
      for (int i = 0; i < kDimPerLane; ++i) p += qreg[g][i] * kreg[i];
      for (int o = kWarpSize / 2; o > 0; o >>= 1)
        p += __shfl_xor_sync(0xffffffff, p, o);
      float score = p * scale;
      float nm = fmaxf(m[g], score), corr = __expf(m[g] - nm),
            pp = __expf(score - nm);
      l[g] = l[g] * corr + pp;
      for (int i = 0; i < kDimPerLane; ++i)
        acc[g][i] = acc[g][i] * corr + pp * vreg[i];
      m[g] = nm;
    }
  }

  __shared__ float sm[kGqaGroup][kWarps], sl[kGqaGroup][kWarps],
      sa[kGqaGroup][kWarps][S::kHeadDim];
  for (int g = 0; g < kGqaGroup; ++g) {
    sm[g][w] = m[g];
    sl[g][w] = l[g];
    for (int i = 0; i < kDimPerLane; ++i)
      sa[g][w][lane + i * kWarpSize] = acc[g][i];
  }
  __syncthreads();

  if (w == 0) {
    for (int g = 0; g < kGqaGroup; ++g) {
      float gm = kNegInf;
      for (int t = 0; t < kWarps; ++t) gm = fmaxf(gm, sm[g][t]);
      float gl = 0.f, gacc[kDimPerLane];
      for (int i = 0; i < kDimPerLane; ++i) gacc[i] = 0.f;
      for (int t = 0; t < kWarps; ++t) {
        float c = __expf(sm[g][t] - gm);
        gl += sl[g][t] * c;
        for (int i = 0; i < kDimPerLane; ++i)
          gacc[i] += sa[g][t][lane + i * kWarpSize] * c;
      }
      int h = kvh * kGqaGroup + g;
      int64_t idx = (static_cast<int64_t>(di) * S::kNumHeads + h) * ksplit + sp;
      if (lane == 0) {
        pm[idx] = gm;
        pl[idx] = gl;
      }
      for (int i = 0; i < kDimPerLane; ++i)
        pa[idx * S::kHeadDim + lane + i * kWarpSize] = gacc[i];
    }
  }
}

static __global__ void DecodeCombineKernel(const float* __restrict__ pm,
                                           const float* __restrict__ pl,
                                           const float* __restrict__ pa,
                                           cbf16* __restrict__ out,
                                           const int* __restrict__ qstart,
                                           const int* __restrict__ decode_rids,
                                           int ksplit) {
  int h = blockIdx.x, di = blockIdx.y, d = threadIdx.x;
  int r = decode_rids[di], flat = qstart[r];
  int64_t base = (static_cast<int64_t>(di) * S::kNumHeads + h) * ksplit;
  float gm = kNegInf;
  for (int s = 0; s < ksplit; ++s) gm = fmaxf(gm, pm[base + s]);
  float gl = 0.f, gacc = 0.f;
  for (int s = 0; s < ksplit; ++s) {
    float c = __expf(pm[base + s] - gm);
    gl += pl[base + s] * c;
    gacc += pa[(base + s) * S::kHeadDim + d] * c;
  }
  float inv = gl > 0 ? 1.f / gl : 0.f;
  out[(static_cast<int64_t>(flat) * S::kNumHeads + h) * S::kHeadDim + d] =
      cbf16(gacc * inv);
}
}  // namespace

void LaunchAttnPrefill(const __nv_bfloat16* qkv, const __nv_bfloat16* cache_kv,
                       __nv_bfloat16* out, const int* pos, const int* qstart,
                       const int* qlen, const int* rids, int R, int max_qlen,
                       const int* bt, int max_blocks, cudaStream_t s) {
  if (R <= 0) return;
  dim3 grid((max_qlen + kQTile - 1) / kQTile, S::kNumHeads, R);
  int smem = (2 * 2 * S::kKvBlock * S::kHeadDim) * sizeof(cbf16);
  PrefillKernel<<<grid, kThreads, smem, s>>>(
      reinterpret_cast<const cbf16*>(qkv),
      reinterpret_cast<const cbf16*>(cache_kv), reinterpret_cast<cbf16*>(out),
      pos, qstart, qlen, rids, bt, max_blocks);
}

void LaunchAttnDecode(const __nv_bfloat16* qkv, const __nv_bfloat16* cache_kv,
                      __nv_bfloat16* out, const int* pos, const int* qstart,
                      const int* decode_rids, int n_decode, const int* bt,
                      int max_blocks, float* pm, float* pl, float* pa,
                      cudaStream_t s) {
  if (n_decode <= 0) return;
  int ksplit = 128 / n_decode;
  if (ksplit < 1) ksplit = 1;
  if (ksplit > kMaxKsplit) ksplit = kMaxKsplit;

  dim3 g1(S::kNumKvHeads, n_decode, ksplit);
  DecodeSplitKernel<<<g1, kThreads, 0, s>>>(
      reinterpret_cast<const cbf16*>(qkv),
      reinterpret_cast<const cbf16*>(cache_kv), pm, pl, pa, pos, qstart,
      decode_rids, bt, max_blocks, ksplit);
  dim3 g2(S::kNumHeads, n_decode);
  DecodeCombineKernel<<<g2, S::kHeadDim, 0, s>>>(
      pm, pl, pa, reinterpret_cast<cbf16*>(out), qstart, decode_rids, ksplit);
}

}  // namespace fq
