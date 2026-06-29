// CuTe (CUTLASS) prefill attention. The QK and PV matmuls use CuTe TiledMMA
// (Ampere m16n8k16 bf16 atoms, valid on sm_89); the online softmax runs in
// shared memory per-row, with the S scores and P weights round-tripping through
// smem so we avoid distributed-accumulator softmax reductions. Mirrors the
// hand-written AttnPrefillMmaKernel's contract exactly (paged combined-NHD KV,
// GQA, causal, head_dim==128, page==16) and is numerically equivalent.
#include <cassert>

#include "attn_cute.h"
#include "cute/tensor.hpp"
#include "kv_layout.h"  // kKvPlanes

using namespace cute;
using cbf16 = cutlass::bfloat16_t;

namespace {
constexpr int HD = 128;  // head_dim
constexpr int TQ = 64;   // query rows per CTA (4 warps x 16)
constexpr int TK = 16;   // key tile == page size
}  // namespace

// One CTA per (q-tile of TQ rows, head h, request r). 128 threads = 4 warps.
static __global__ void CutePrefillKernel(const cbf16* __restrict__ q, int q_stride,
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
  int64_t plane = static_cast<int64_t>(TK) * kv_dim;  // K plane -> V plane
  int kvh = h / (n_heads / n_kv);
  const int* btr = bt + r * max_blocks;
  int q0 = qtile * TQ;
  if (q0 >= ql) return;
  int tid = threadIdx.x;

  // Dynamic smem, aliased: Q (16KB) is loaded, copied to registers, then its
  // memory is recycled for the K/V/S/P tiles (disjoint in the loop). Keeps
  // per-CTA smem ~16KB (vs 31KB if Q stayed resident) for occupancy parity
  // with the hand-written kernel.
  extern __shared__ char smem[];
  cbf16* sK = reinterpret_cast<cbf16*>(smem);          // [TK*HD]  [0,4K)
  cbf16* sV = sK + TK * HD;                             // [TK*HD]  [4K,8K)
  float* sS = reinterpret_cast<float*>(sV + TK * HD);  // [TQ*TK]  [8K,12K)
  cbf16* sP = reinterpret_cast<cbf16*>(sS + TQ * TK);  // [TQ*TK]  [12K,14K)
  cbf16* sQ = reinterpret_cast<cbf16*>(smem);          // [TQ*HD] aliases sK..sP
  __shared__ float sM[TQ], sL[TQ], sCorr[TQ];

  for (int i = tid; i < TQ * HD; i += blockDim.x) {
    int row = i / HD, col = i % HD, grow = q0 + row;
    sQ[i] = grow < ql ? q[(qs + grow) * q_stride + h * HD + col] : cbf16(0.f);
  }
  for (int row = tid; row < TQ; row += blockDim.x) { sM[row] = -1e30f; sL[row] = 0.f; }
  __syncthreads();

  TiledMMA mmaQK = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_4, _1, _1>>{});
  TiledMMA mmaPV = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_4, _1, _1>>{});
  auto thrQK = mmaQK.get_thread_slice(tid);
  auto thrPV = mmaPV.get_thread_slice(tid);

  Tensor sQt = make_tensor(make_smem_ptr(sQ), make_layout(Shape<Int<TQ>, Int<HD>>{}, LayoutRight{}));
  Tensor sKt = make_tensor(make_smem_ptr(sK), make_layout(Shape<Int<TK>, Int<HD>>{}, LayoutRight{}));
  Tensor sSt = make_tensor(make_smem_ptr(sS), make_layout(Shape<Int<TQ>, Int<TK>>{}, LayoutRight{}));
  Tensor sPt = make_tensor(make_smem_ptr(sP), make_layout(Shape<Int<TQ>, Int<TK>>{}, LayoutRight{}));
  // V^T view [HD, TK]: element (hd,key) = sV[key*HD + hd] -> stride (1, HD).
  Tensor sVt = make_tensor(make_smem_ptr(sV), make_layout(Shape<Int<HD>, Int<TK>>{}, Stride<_1, Int<HD>>{}));

  // Q -> registers ONCE (A-operand), reused across all key tiles; then recycle
  // the smem that held Q for the K/V/S/P tiles.
  Tensor tSrQ = thrQK.partition_fragment_A(sQt);
  copy(thrQK.partition_A(sQt), tSrQ);
  __syncthreads();

  Tensor tOrO = partition_fragment_C(mmaPV, Shape<Int<TQ>, Int<HD>>{});
  clear(tOrO);
  Tensor tOcO = thrPV.partition_C(make_identity_tensor(Shape<Int<TQ>, Int<HD>>{}));

  int qlast = min(q0 + TQ - 1, ql - 1);
  int ntiles = pos[qs + qlast] / TK + 1;

  for (int kt = 0; kt < ntiles; ++kt) {
    int64_t kvbase = static_cast<int64_t>(btr[kt]) * kKvPlanes * plane +
                     static_cast<int64_t>(kvh) * HD;
    for (int i = tid; i < TK * HD; i += blockDim.x) {
      int key = i / HD, col = i % HD;
      sK[i] = cache_kv[kvbase + static_cast<int64_t>(key) * kv_dim + col];
      sV[i] = cache_kv[kvbase + plane + static_cast<int64_t>(key) * kv_dim + col];
    }
    __syncthreads();

    Tensor tSrK = thrQK.partition_fragment_B(sKt);
    Tensor tSrS = partition_fragment_C(mmaQK, Shape<Int<TQ>, Int<TK>>{});
    clear(tSrS);
    copy(thrQK.partition_B(sKt), tSrK);
    gemm(mmaQK, tSrQ, tSrK, tSrS);
    copy(tSrS, thrQK.partition_C(sSt));
    __syncthreads();

    for (int row = tid; row < TQ; row += blockDim.x) {
      int grow = q0 + row;
      int qp = grow < ql ? pos[qs + grow] : -1;
      float tmax = -1e30f;
      for (int c = 0; c < TK; ++c) {
        int kpos = kt * TK + c;
        float v = (kpos <= qp) ? sS[row * TK + c] * scale : -1e30f;
        sS[row * TK + c] = v;
        tmax = fmaxf(tmax, v);
      }
      float nm = fmaxf(sM[row], tmax);
      float corr = __expf(sM[row] - nm);
      float tsum = 0.f;
      for (int c = 0; c < TK; ++c) {
        float p = __expf(sS[row * TK + c] - nm);
        sP[row * TK + c] = cbf16(p);
        tsum += p;
      }
      sL[row] = sL[row] * corr + tsum;
      sM[row] = nm;
      sCorr[row] = corr;
    }
    __syncthreads();

    for (int i = 0; i < size(tOrO); ++i) tOrO(i) *= sCorr[get<0>(tOcO(i))];

    Tensor tOrP = thrPV.partition_fragment_A(sPt);
    Tensor tOrV = thrPV.partition_fragment_B(sVt);
    copy(thrPV.partition_A(sPt), tOrP);
    copy(thrPV.partition_B(sVt), tOrV);
    gemm(mmaPV, tOrP, tOrV, tOrO);
    __syncthreads();
  }

  for (int i = 0; i < size(tOrO); ++i) {
    int row = get<0>(tOcO(i)), col = get<1>(tOcO(i)), grow = q0 + row;
    if (grow < ql) {
      float l = sL[row];
      out[((qs + grow) * n_heads + h) * HD + col] =
          cbf16(tOrO(i) * (l > 0.f ? 1.f / l : 0.f));
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
  int smem = TQ * HD * sizeof(cbf16);  // 16KB: Q footprint (recycled for K/V/S/P)
  CutePrefillKernel<<<grid, 128, smem, s>>>(
      reinterpret_cast<const cbf16*>(q), q_stride,
      reinterpret_cast<const cbf16*>(cache_kv), reinterpret_cast<cbf16*>(out),
      n_heads, n_kv, pos, qstart, qlen, rids, bt, max_blocks, scale);
}
