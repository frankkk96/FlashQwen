# S11 尝试 — 纯 decode 步骤的 CUDA Graphs(尝试,已回退)

## 背景/动机

到 S10 为止,conc=32/1024 已经达到 589 tok/s(84% of vLLM,5.5× 于 `main` baseline)。在排查剩余差距时,CUDA graphs 是一个经典的“消除 per-step launch 开销”的候选手段:把纯 decode 的整段前向(run_layers + lm_head + sample)按 batch size B 捕获成一张 CUDA graph 然后回放,理论上能省下每步约 470 次 kernel launch 的 CPU 端开销。

尤其是 conc=1 单流场景,直觉上认为它“latency-bound”,似乎应该最受益于 launch elision。S11 就是用来检验这个假设的。

## 改动

把纯 decode 前向(run_layers + lm_head + sample)按 batch size B 逐一捕获成一张 CUDA graph 并回放:

- 将前向拆分为 eager 的 `upload_inputs` 阶段 + 可捕获(captureable)的 compute 阶段;
- pin 一块 cuBLAS workspace,并固定 block-table 的 stride,使捕获后的 launch 仍然合法;
- 捕获失败时优雅回退(graceful fallback)。

捕获**成功**,输出保持 coherent。但结果是**净负收益**,因此回退。

## 实测结果

| 指标 | S10 | S11(CUDA graphs) | Δ |
|---|---|---|---|
| conc=1 | 45.2 | 45.7 | 基本持平 |
| conc=32 | 589 | 520 | **−12%** |

- **为什么 conc=1 没有收益**:conc=1 的 decode 是 **weight-bandwidth-bound**,而不是 launch-bound——每个 token 都要读完全部 17.3 GB 的权重(~17.3 ms 下限;我们的 TPOT 22 ms ≈ HBM 的 78%)。那 ~470 次 launch 是异步发射的,会与这 17 ms 的 GPU 计算 overlap,所以把它们消除掉省下的 wall-clock ≈ 0。(vLLM 的 conc=1 17.9 ms ≈ 那 17.3 ms 下限;它的优势来自更精简的 *eager* per-step 工作——H2D + sync——而这部分是 graph 覆盖不到的。)
- **为什么 conc=32 反而退化**:GPU 本来就 ~97% busy(没有 launch gap 可隐藏),而且 B 在整个运行过程中从 32 逐渐 drain 到 1,因此每出现一个新的 batch size 都要付一次性的 eager-warmup + graph-instantiate 成本 → 净开销。

## 经验

FlashQwen 在任何并发下都**不是 launch-bound**(conc=32 已经饱和,conc=1 是 bandwidth-bound),所以 CUDA graphs——它只能移除 launch 开销——在这里没有任何东西可以回收。

这条结论关上了这一路的搜索:到 vLLM 的剩余差距(589→698)是 HBM bandwidth + vLLM 更精简的 per-step eager path,而不是一个我们能在不上量化的前提下修掉的 kernel/launch 问题。
