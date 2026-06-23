# S4 — tensor-core + GQA 分组 attention(尝试,已回退)

## 背景/动机

S3 之后,conc=32 剩余的 ~2× 差距被归因于 GEMM + launch 开销,而非 attention。S4 是一次相反方向的验证尝试:用 tensor core 重写 attention,并用 GQA 分组复用 K/V,看 attention 是否还是 conc=32 的瓶颈。

## 改动

- 把 attention kernel 重写为使用 WMMA(16×16×16 tensor cores)来做 S=Q·Kᵀ 和 O=P·V。
- 分组方式:one block per (q-tile, **KV head**, request),使得一个 GQA group 的全部 4 个 q-head 复用同一份 staged K/V tile(K/V traffic 减少 4×)。
- O accumulator 保留在 shared(fp32),由一个 thread loop 做 rescale,P·V 通过 load/store_matrix_sync 累加——不依赖任何 WMMA fragment-layout 假设。
- Greedy 输出与 S3 bit-identical(正确)。

## 实测结果

- **vs S3**:conc=32 350.0 → 355.5(持平,噪声范围);conc=1 **38.2 → 25.1(−34%)**。净负面。

## 经验

为什么没收益:

1. 在 S3 的 BM=16 K/V tiling 之后,**attention 已经不是 conc=32 的瓶颈**——GEMM/launch 开销才是主导,所以即便 attention 快 2–3× 也几乎不动总吞吐。
2. 在 conc=1,decode rows 的 q_len=1,但 WMMA 强制一个完整的 16-row tile(浪费 16× 的计算量);而 per-KV-head 分组只留下 8 个 grid blocks(对比原来的 32)→ GPU 被饿着,attention latency 增长,拖累单流 TPOT(26→40 ms)。

**决策**:回退到 S3(commit 7650654)。保留此条目作为记录:attention 这个 lever 已经用尽——下一批真正的 lever 是 launch-overhead(CUDA graphs / kernel fusion)以及 vLLM 拥有的吞吐特性(persistent batch、prefix caching)。

> 后续补记:S4 的 tensor-core 想法本身是对的,只是用错了地方(用在了 decode 上)。S7 把 WMMA 用在 prefill 路径(q_len 最高 512、整 tile),S4 的失败模式就消失了;S12 用 KV-split 让 GQA 分组在 decode 上变成净赢。
