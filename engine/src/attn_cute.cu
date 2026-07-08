#include <cuda_pipeline.h>  // cp.async 内建函数（__pipeline_memcpy_async 等）

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
// GQA 分组数：多少个 Q head 共享一个 KV head（kNumHeads / kNumKvHeads）。
constexpr int kGqaGroup = S::kNumHeads / S::kNumKvHeads;
// decode kernel 里每个 lane 负责 head_dim 的多少个元素（一个 warp 32 lane 分摊 head_dim）。
constexpr int kDimPerLane = S::kHeadDim / 32;
// prefill 每个 block 一次处理的 query 行数（一个 Q-tile 的高度）。
constexpr int kQTile = 64;
// 每个 block 的 warp 数（128 线程 = 4 warp）。
constexpr int kWarps = 4;

// 把 Tensor Core 累加器的碎片 layout ((2,2), MMA_M, MMA_N)
// 重排成逻辑上的二维 (行, 列) 视图 ((行2,MMA_M),(列2,MMA_N))，
// 便于按行做 online softmax 的归约。只改 layout、不动数据。
// 细节：logical_divide 把 mode-0 拆成 (列2, 行2)——列在内层是本 atom 的硬件规定；
//       随后行=get<0,1>+MMA_M、列=get<0,0>+MMA_N 各自拼成一个轴。
template <typename Layout>
__device__ __forceinline__ auto AccRowcol(Layout l) {
  using namespace cute;
  auto d = logical_divide(l, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(d), get<1>(d)),   // 行轴 = (atom内行, MMA_M)
                     make_layout(get<0, 0>(d), get<2>(d)));   // 列轴 = (atom内列, MMA_N)
}
}  // namespace

static __global__ void __launch_bounds__(128, 4)
CutePrefillKernel(const cbf16* __restrict__ qkv,
                                  const cbf16* __restrict__ cache_kv,
                                  cbf16* __restrict__ out,
                                  const int* __restrict__ pos,
                                  const int* __restrict__ qstart,
                                  const int* __restrict__ qlen,
                                  const int* __restrict__ rids,
                                  const int* __restrict__ bt, int max_blocks) {
  // ========== 阶段 1：定位与寻址 ==========
  // grid 三维含义：z=第几个请求, y=第几个 head, x=第几个 Q-tile。
  // 一个 block 负责「某请求 / 某 head / 某 Q 分块」的完整 attention。
  const float scale = rsqrtf(static_cast<float>(S::kHeadDim));  // 注意力缩放 1/sqrt(head_dim)
  int r = rids[blockIdx.z], h = blockIdx.y, qtile = blockIdx.x;
  int ql = qlen[r], qs = qstart[r];                    // 本请求的序列长度、query 起始偏移
  int kv_dim = S::kNumKvHeads * S::kHeadDim;            // 一个 token 的 K（或 V）总宽度
  int qkv_dim = S::kNumHeads * S::kHeadDim + 2 * kv_dim;  // QKV 打包在一起时每 token 的总宽度
  int64_t plane = static_cast<int64_t>(S::kKvBlock) * kv_dim; // 一个 page 的大小（有多少个数字）
  int kvh = h / kGqaGroup;                             // GQA：该 Q head 对应的 KV head 编号
  const int* btr = bt + r * max_blocks;                // 本请求的 block table（分页 KV 的页号数组）
  int q0 = qtile * kQTile;                             // 本 tile 的全局 query 起始行
  if (q0 >= ql) return;                                // 整个 tile 超出序列长度：直接退出
  int tid = threadIdx.x;

  // ========== 阶段 2：把 Q 搬进 shared memory ==========
  // smem 复用：Q 用完后同一块空间被 K/V/P 覆盖（sQ 和 sKb[0] 都从 smem 起点开始）。
  // K/V 各开两份（双缓冲）：一份供当前 tile 计算，另一份被 cp.async 后台预取下一 tile。
  extern __shared__ char smem[];
  constexpr int kKvTile = S::kKvBlock * S::kHeadDim;   // 一份 K（或 V）的元素数
  cbf16* sKb[2] = {reinterpret_cast<cbf16*>(smem), reinterpret_cast<cbf16*>(smem) + kKvTile};
  cbf16* sVb[2] = {sKb[1] + kKvTile, sKb[1] + 2 * kKvTile};
  cbf16* sP = sVb[1] + kKvTile;                        // P 排在两份 K/V 之后
  cbf16* sQ = reinterpret_cast<cbf16*>(smem);          // Q 只在 KV 循环前用，与 K/V buffer 复用

  // 128 线程协作，用 int4（一次 8 个 bf16）向量化把 Q-tile 拷进 smem；越界行填 0。
  const int4 kZero4 = {0, 0, 0, 0};
  constexpr int kVecPerRow = S::kHeadDim / 8;   // 每行多少个 int4（128/8 = 16）
  constexpr int kNumVec = kQTile * kVecPerRow;  // 整个 tile 的 int4 总数（64×16 = 1024）
  for (int c = tid; c < kNumVec; c += blockDim.x) {
    int row = c / kVecPerRow;         // tile 内第几行（0..kQTile-1）
    int col8 = (c % kVecPerRow) * 8;  // 行内起始列（bf16 偏移：0,8,...,120）
    int grow = q0 + row;              // 对应的全局 query 行号

    int4* dst = reinterpret_cast<int4*>(&sQ[row * S::kHeadDim + col8]);
    if (grow < ql) {  // 该行在序列内：从全局 Q 搬 8 个 bf16
      *dst = *reinterpret_cast<const int4*>(
          &qkv[(qs + grow) * qkv_dim + h * S::kHeadDim + col8]);
    } else {  // 超出序列长度的尾行：填 0，直接返回会死锁（有__syncthreads的barrier），而且MMA指令的计算会拿到旧数据导致不可预期的结果
      *dst = kZero4;
    }
  }
  __syncthreads();

  // ========== 阶段 3：建立 MMA 与张量分区 ==========
  // 用 m16n8k16 atom（bf16 输入 / f32 累加），Layout<_4,_1,_1> 表示 4 个 warp 在 M 方向叠 → 一步覆盖 64 行。
  // mmaQK 算 QKᵀ，mmaPV 算 P·V；get_thread_slice 取得「当前线程」的碎片视角。
  TiledMMA mmaQK = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_4, _1, _1>>{});
  TiledMMA mmaPV = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_4, _1, _1>>{});
  auto thrQK = mmaQK.get_thread_slice(tid);
  auto thrPV = mmaPV.get_thread_slice(tid);

  // 给 smem 缓冲区套上逻辑 (shape, stride)，CuTe 才知道每个 (i,j) 落在哪。
  // 注意 sVt 用列主序 Stride<_1, kHeadDim>：V 需要转置布局才能当 PV 的 B 矩阵。
  Tensor sQt = make_tensor(make_smem_ptr(sQ), make_layout(Shape<Int<kQTile>, Int<S::kHeadDim>>{}, LayoutRight{}));
  Tensor sPt = make_tensor(make_smem_ptr(sP), make_layout(Shape<Int<kQTile>, Int<S::kKvBlock>>{}, LayoutRight{}));
  // sKt/sVt 依赖「当前 buffer」，在 KV 循环内按 sKb[cur]/sVb[cur] 构建。

  // Q 在整个 KV 循环里不变，只需一次性加载进寄存器碎片 tSrQ。
  Tensor tSrQ = thrQK.partition_fragment_A(sQt);   // 开出「本线程」的 Q 寄存器碎片
  copy(thrQK.partition_A(sQt), tSrQ);              // 从 smem 搬入寄存器
  __syncthreads();

  // ========== 阶段 4：初始化累加器与 softmax 状态 ==========
  Tensor tOrO = partition_fragment_C(mmaPV, Shape<Int<kQTile>, Int<S::kHeadDim>>{});  // 输出 O 的寄存器累加器
  clear(tOrO);                                                                         // 清零
  // 坐标张量（identity）：用来反查「本线程的 (r,c) 对应全局第几行 / 第几列 / 第几个 key」。
  Tensor tOcO = thrPV.partition_C(make_identity_tensor(Shape<Int<kQTile>, Int<S::kHeadDim>>{}));
  Tensor O_rc = make_tensor(tOrO.data(), AccRowcol(tOrO.layout()));   // O 的 (行,列) 视图
  Tensor cO_rc = make_tensor(tOcO.data(), AccRowcol(tOcO.layout()));  // O 坐标的 (行,列) 视图
  Tensor tScS = thrQK.partition_C(make_identity_tensor(Shape<Int<kQTile>, Int<S::kKvBlock>>{}));
  Tensor cS_rc = make_tensor(tScS.data(), AccRowcol(tScS.layout()));  // S 坐标的 (行,列) 视图
  // NROW=每线程负责的行数(2), NSC=S 每行的列数, NOC=O 每行的列数。
  constexpr int NROW = 2, NSC = decltype(size<1>(cS_rc))::value, NOC = decltype(size<1>(O_rc))::value;
  float rm[NROW], rl[NROW];                            // online softmax 每行的运行最大值 / 运行分母
  for (int r = 0; r < NROW; ++r) { rm[r] = -1e30f; rl[r] = 0.f; }

  // 因果掩码下，本 tile 最后一个 query 的位置决定最多要看多少个 KV 块（之后的块全被 mask）。
  int qlast = min(q0 + kQTile - 1, ql - 1);
  int ntiles = pos[qs + qlast] / S::kKvBlock + 1;

  // 用 int4（16B）cp.async 把逻辑第 kt 个 KV 块异步预取进 buffer b（直接 global→smem，不经寄存器）。
  // 块表把「逻辑第 kt 块」翻译成物理 page 地址；同一 page 内 K 在前、V 在 +plane 处。
  auto prefetch_kv = [&](int kt, int b) {
    int64_t kvbase = static_cast<int64_t>(btr[kt]) * KvStore::kKvPlanes * plane +
                     static_cast<int64_t>(kvh) * S::kHeadDim;
    for (int c = tid; c < S::kKvBlock * S::kHeadDim / 8; c += blockDim.x) {
      int key = c / (S::kHeadDim / 8), hd8 = (c % (S::kHeadDim / 8)) * 8;
      __pipeline_memcpy_async(&sKb[b][key * S::kHeadDim + hd8],
                              &cache_kv[kvbase + static_cast<int64_t>(key) * kv_dim + hd8], 16);
      __pipeline_memcpy_async(&sVb[b][key * S::kHeadDim + hd8],
                              &cache_kv[kvbase + plane + static_cast<int64_t>(key) * kv_dim + hd8], 16);
    }
    __pipeline_commit();
  };

  // ========== 阶段 5：逐个 KV tile 做 flash attention（cp.async 双缓冲流水）==========
  // prologue：先发起第 0 块预取；之后每轮在计算当前块的同时后台预取下一块，藏住访存延迟。
  if (ntiles > 0) prefetch_kv(0, 0);
  for (int kt = 0; kt < ntiles; ++kt) {
    int cur = kt & 1;
    // --- 5a. 后台预取下一块 + 等待当前块到齐 ---
    if (kt + 1 < ntiles) {
      prefetch_kv(kt + 1, (kt + 1) & 1);  // 下一块的拷贝在后台飞行，与本轮计算重叠
      __pipeline_wait_prior(1);           // 仍留 1 组（下一块）在飞 → 当前块 kt 必已到齐
    } else {
      __pipeline_wait_prior(0);           // 最后一块：等全部 cp.async 完成
    }
    __syncthreads();

    // 当前 buffer 的 K/V 逻辑视图（sVt 用列主序，转置后当 PV 的 B 矩阵）。
    Tensor sKt = make_tensor(make_smem_ptr(sKb[cur]),
        make_layout(Shape<Int<S::kKvBlock>, Int<S::kHeadDim>>{}, LayoutRight{}));
    Tensor sVt = make_tensor(make_smem_ptr(sVb[cur]),
        make_layout(Shape<Int<S::kHeadDim>, Int<S::kKvBlock>>{}, Stride<_1, Int<S::kHeadDim>>{}));

    // --- 5b. QKᵀ：S = Q · Kᵀ，结果在寄存器累加器 tSrS ---
    Tensor tSrK = thrQK.partition_fragment_B(sKt);
    Tensor tSrS = partition_fragment_C(mmaQK, Shape<Int<kQTile>, Int<S::kKvBlock>>{});
    clear(tSrS);
    copy(thrQK.partition_B(sKt), tSrK);
    gemm(mmaQK, tSrQ, tSrK, tSrS);

    // --- 5c. Online softmax（本 kernel 的核心）：逐行处理 ---
    Tensor S_rc = make_tensor(tSrS.data(), AccRowcol(tSrS.layout()));  // S 的 (行,列) 视图
    for (int r = 0; r < NROW; ++r) {
      int row = get<0>(cS_rc(r, 0)), grow = q0 + row;   // 本行的局部/全局行号
      int qp = grow < ql ? pos[qs + grow] : -1;         // 该 query 的位置（用于因果掩码）
      // ① 因果掩码 + 缩放：key 位置 > query 位置的置 -inf，其余乘 scale；顺带求本块行最大值。
      float rmax = -1e30f;
      for (int c = 0; c < NSC; ++c) {
        int kpos = kt * S::kKvBlock + get<1>(cS_rc(r, c));  // 该列对应的 key 全局位置
        float v = (kpos <= qp) ? S_rc(r, c) * scale : -1e30f;
        S_rc(r, c) = v; rmax = fmaxf(rmax, v);
      }
      // ② warp 内跨 lane 归约行最大：同一逻辑行的列分散在相邻 4 个 lane 上，xor 1、xor 2 合并它们。
      rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffff, rmax, 1));
      rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffff, rmax, 2));
      // ③ 更新运行最大 nm，算修正因子 corr = exp(旧max - nm)。
      float nm = fmaxf(rm[r], rmax), corr = __expf(rm[r] - nm), rsum = 0.f;
      // ④ exp 得到概率 P，并累加本块行和 rsum（同样 shuffle 归约）。
      for (int c = 0; c < NSC; ++c) { float p = __expf(S_rc(r, c) - nm); S_rc(r, c) = p; rsum += p; }
      rsum += __shfl_xor_sync(0xffffffff, rsum, 1);
      rsum += __shfl_xor_sync(0xffffffff, rsum, 2);
      // ⑤ 更新运行分母；已累加的 O 也要乘 corr（flash attention 的重缩放）。
      rl[r] = rl[r] * corr + rsum; rm[r] = nm;
      for (int c = 0; c < NOC; ++c) O_rc(r, c) *= corr;
      // ⑥ 把概率 P 写进 smem，供下一步 PV gemm 当 A 矩阵。
      for (int c = 0; c < NSC; ++c) sP[row * S::kKvBlock + get<1>(cS_rc(r, c))] = cbf16(S_rc(r, c));
    }
    __syncthreads();

    // --- 5d. P·V 累加进 O：O += P · V ---
    Tensor tOrP = thrPV.partition_fragment_A(sPt);
    Tensor tOrV = thrPV.partition_fragment_B(sVt);
    copy(thrPV.partition_A(sPt), tOrP);
    copy(thrPV.partition_B(sVt), tOrV);
    gemm(mmaPV, tOrP, tOrV, tOrO);
    __syncthreads();
  }

  // ========== 阶段 6：归一化并写回 ==========
  // 循环结束时 tOrO = 未归一化的 Σ P·V，rl[r] = 最终 softmax 分母；此处除以分母、转 bf16 写回。
  for (int r = 0; r < NROW; ++r) {
    float inv = rl[r] > 0 ? 1.f / rl[r] : 0.f;
    for (int c = 0; c < NOC; ++c) {
      int row = get<0>(cO_rc(r, c)), col = get<1>(cO_rc(r, c)), grow = q0 + row;
      if (grow < ql) out[((qs + grow) * S::kNumHeads + h) * S::kHeadDim + col] = cbf16(O_rc(r, c) * inv);
    }
  }
}

// 启动 prefill kernel。grid = (Q-tile 数, head 数, 请求数)，每 block 128 线程。
void LaunchAttnPrefillCute(const __nv_bfloat16* q,
                           const __nv_bfloat16* cache_kv, __nv_bfloat16* out,
                           const int* pos, const int* qstart, const int* qlen,
                           const int* rids, int R, int max_qlen, const int* bt,
                           int max_blocks, cudaStream_t s) {
  if (R <= 0) return;
  dim3 grid((max_qlen + kQTile - 1) / kQTile, S::kNumHeads, R);
  // smem 布局：K/V 各两份（双缓冲）+ P。共 4*kKvBlock*kHeadDim + kQTile*kKvBlock 个 bf16，
  // 已 ≥ Q-tile（kQTile*kHeadDim），Q 与前两份 buffer 复用同一块空间。
  int smem = (4 * S::kKvBlock * S::kHeadDim + kQTile * S::kKvBlock) * sizeof(cbf16);
  CutePrefillKernel<<<grid, 128, smem, s>>>(
      reinterpret_cast<const cbf16*>(q),
      reinterpret_cast<const cbf16*>(cache_kv), reinterpret_cast<cbf16*>(out),
      pos, qstart, qlen, rids, bt, max_blocks);
}

// ============================================================================
// Decode（自回归单步）注意力，采用 Flash-Decoding 的「split-KV + 两阶段归约」：
//   - Split kernel：把 KV 序列沿位置切成 ksplit 份，每份各自算局部 softmax，
//     产出局部的 (max, sum, 加权和) 三元组，写入 pm/pl/pa。
//   - Combine kernel：把同一 (请求, head) 的 ksplit 份局部结果合并成最终输出。
// decode 每步只有 1 个 query，用不满 Tensor Core，所以走「每 lane 管 head_dim 的一段」
// 的手写点积路线，不用 MMA。
// ============================================================================
static __global__ void CuteDecodeSplitKernel(
    const cbf16* __restrict__ q, const cbf16* __restrict__ cache_kv,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pa,
    const int* __restrict__ pos, const int* __restrict__ qstart,
    const int* __restrict__ decode_rids, const int* __restrict__ bt,
    int max_blocks, int ksplit) {
  const float scale = rsqrtf(static_cast<float>(S::kHeadDim));
  // grid: x=KV head, y=第几个 decode 请求, z=第几个 split。
  int kvh = blockIdx.x, di = blockIdx.y, sp = blockIdx.z;
  int w = threadIdx.x >> 5, lane = threadIdx.x & 31;   // warp 号、lane 号
  int r = decode_rids[di], flat = qstart[r], qpos = pos[flat];  // 请求、query 偏移、query 位置
  int kv_dim = S::kNumKvHeads * S::kHeadDim;
  int qkv_dim = S::kNumHeads * S::kHeadDim + 2 * kv_dim;
  int64_t plane = static_cast<int64_t>(S::kKvBlock) * kv_dim;
  const int* btr = bt + r * max_blocks;

  // 载入 Q：该 KV head 对应的 kGqaGroup 个 Q head，每个 head 的 head_dim 由 32 个 lane 分摊。
  // qreg[g][i]：第 g 个 Q head、本 lane 负责的第 i 个元素。
  float qreg[kGqaGroup][kDimPerLane];
  for (int g = 0; g < kGqaGroup; ++g) {
    const cbf16* qv = q + static_cast<int64_t>(flat) * qkv_dim +
                      static_cast<int64_t>(kvh * kGqaGroup + g) * S::kHeadDim;
    for (int i = 0; i < kDimPerLane; ++i) qreg[g][i] = float(qv[lane + i * 32]);
  }
  // 每个 Q head 的局部 online-softmax 状态：运行最大 m、运行分母 l、加权和 acc。
  float m[kGqaGroup], l[kGqaGroup], acc[kGqaGroup][kDimPerLane];
  for (int g = 0; g < kGqaGroup; ++g) { m[g] = -1e30f; l[g] = 0.f;
    for (int i = 0; i < kDimPerLane; ++i) acc[g][i] = 0.f; }

  // 遍历本 split 分到的 KV 位置：步长 kWarps*ksplit，让 (split, warp) 交错覆盖整个序列。
  for (int kpos = sp * kWarps + w; kpos <= qpos; kpos += kWarps * ksplit) {
    // 分页寻址：第 kpos 个 token 落在哪个 page 的哪一行。
    int64_t base = static_cast<int64_t>(btr[kpos / S::kKvBlock]) * KvStore::kKvPlanes * plane +
                   static_cast<int64_t>(kpos % S::kKvBlock) * kv_dim +
                   static_cast<int64_t>(kvh) * S::kHeadDim;
    Tensor K = make_tensor(make_gmem_ptr(cache_kv + base), make_layout(Shape<Int<S::kHeadDim>>{}));
    Tensor V = make_tensor(make_gmem_ptr(cache_kv + base + plane), make_layout(Shape<Int<S::kHeadDim>>{}));
    // 本 lane 负责的那几段 K、V。
    float kreg[kDimPerLane], vreg[kDimPerLane];
    for (int i = 0; i < kDimPerLane; ++i) { kreg[i] = float(K(lane + i * 32)); vreg[i] = float(V(lane + i * 32)); }
    for (int g = 0; g < kGqaGroup; ++g) {
      // 点积 Q·K：先各 lane 算局部和，再用蝶形 shuffle 在整个 warp（32 lane）内求全和。
      float p = 0; for (int i = 0; i < kDimPerLane; ++i) p += qreg[g][i] * kreg[i];
      for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffff, p, o);
      float score = p * scale;
      // online softmax 单步更新（同 prefill 的重缩放逻辑）。
      float nm = fmaxf(m[g], score), corr = __expf(m[g] - nm), pp = __expf(score - nm);
      l[g] = l[g] * corr + pp;
      for (int i = 0; i < kDimPerLane; ++i) acc[g][i] = acc[g][i] * corr + pp * vreg[i];
      m[g] = nm;
    }
  }

  // block 内跨 warp 归约：本 block 的 kWarps 个 warp 各处理了序列的一部分，先在 smem 汇总。
  __shared__ float sm[kGqaGroup][kWarps], sl[kGqaGroup][kWarps], sa[kGqaGroup][kWarps][S::kHeadDim];
  for (int g = 0; g < kGqaGroup; ++g) { sm[g][w] = m[g]; sl[g][w] = l[g];
    for (int i = 0; i < kDimPerLane; ++i) sa[g][w][lane + i * 32] = acc[g][i]; }
  __syncthreads();
  // 由 warp 0 把 kWarps 个 warp 的局部结果合并成「本 split 的局部结果」，写入 pm/pl/pa。
  if (w == 0) {
    for (int g = 0; g < kGqaGroup; ++g) {
      float gm = -1e30f; for (int t = 0; t < kWarps; ++t) gm = fmaxf(gm, sm[g][t]);   // 全局最大
      float gl = 0.f, gacc[kDimPerLane]; for (int i = 0; i < kDimPerLane; ++i) gacc[i] = 0.f;
      // 用统一最大 gm 重缩放各 warp 的 sum 和加权和，再相加。
      for (int t = 0; t < kWarps; ++t) { float c = __expf(sm[g][t] - gm); gl += sl[g][t] * c;
        for (int i = 0; i < kDimPerLane; ++i) gacc[i] += sa[g][t][lane + i * 32] * c; }
      int h = kvh * kGqaGroup + g;
      // 输出到中间缓冲 pm/pl/pa，索引按 (请求, head, split) 排布，供 combine kernel 二次归约。
      int64_t idx = (static_cast<int64_t>(di) * S::kNumHeads + h) * ksplit + sp;
      if (lane == 0) { pm[idx] = gm; pl[idx] = gl; }
      for (int i = 0; i < kDimPerLane; ++i) pa[idx * S::kHeadDim + lane + i * 32] = gacc[i];
    }
  }
}

// Combine 阶段：把同一 (请求, head) 的 ksplit 份局部结果合并成最终注意力输出。
// grid: x=head, y=第几个 decode 请求；每 block head_dim 个线程，thread d 负责第 d 维。
static __global__ void CuteDecodeCombineKernel(
    const float* __restrict__ pm, const float* __restrict__ pl,
    const float* __restrict__ pa, cbf16* __restrict__ out,
    const int* __restrict__ qstart, const int* __restrict__ decode_rids, int ksplit) {
  int h = blockIdx.x, di = blockIdx.y, d = threadIdx.x;
  int r = decode_rids[di], flat = qstart[r];
  int64_t base = (static_cast<int64_t>(di) * S::kNumHeads + h) * ksplit;
  // ① 先求跨所有 split 的全局最大。
  float gm = -1e30f; for (int s = 0; s < ksplit; ++s) gm = fmaxf(gm, pm[base + s]);
  // ② 用 gm 重缩放各 split 的分母和加权和，再累加（标准的 log-sum-exp 合并）。
  float gl = 0.f, gacc = 0.f;
  for (int s = 0; s < ksplit; ++s) { float c = __expf(pm[base + s] - gm);
    gl += pl[base + s] * c; gacc += pa[(base + s) * S::kHeadDim + d] * c; }
  // ③ 除以最终分母得到本维输出，转 bf16 写回。
  float inv = gl > 0 ? 1.f / gl : 0.f;
  out[(static_cast<int64_t>(flat) * S::kNumHeads + h) * S::kHeadDim + d] = cbf16(gacc * inv);
}

// 启动 decode：两个 kernel 串联。ksplit 按 decode 请求数自适应（请求少 → 切更多份填满 GPU）。
void LaunchAttnDecodeCute(const __nv_bfloat16* q,
                          const __nv_bfloat16* cache_kv, __nv_bfloat16* out,
                          const int* pos, const int* qstart,
                          const int* decode_rids, int n_decode, const int* bt,
                          int max_blocks, float* pm, float* pl,
                          float* pa, cudaStream_t s) {
  if (n_decode <= 0) return;
  int ksplit = 128 / n_decode;          // 请求越少，切越多份以填满 SM
  if (ksplit < 1) ksplit = 1;
  if (ksplit > kMaxKsplit) ksplit = kMaxKsplit;

  // 阶段一：split-KV，产出局部 (max,sum,加权和) 到 pm/pl/pa。
  dim3 g1(S::kNumKvHeads, n_decode, ksplit);
  CuteDecodeSplitKernel<<<g1, kWarps * 32, 0, s>>>(
      reinterpret_cast<const cbf16*>(q),
      reinterpret_cast<const cbf16*>(cache_kv), pm, pl, pa, pos, qstart,
      decode_rids, bt, max_blocks, ksplit);
  // 阶段二：combine，跨 split 二次归约得最终输出。
  dim3 g2(S::kNumHeads, n_decode);
  CuteDecodeCombineKernel<<<g2, S::kHeadDim, 0, s>>>(
      pm, pl, pa, reinterpret_cast<cbf16*>(out), qstart, decode_rids, ksplit);
}

}  // namespace fq
