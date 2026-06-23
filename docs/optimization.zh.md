# FlashQwen 优化日志 — 追赶 vLLM

> English version: [optimization.md](optimization.md) · 每个实验的**详细记录**在 [`exps/`](exps/)(全中文,一实验一文档)

这是 FlashQwen 吞吐优化全过程的**总览**:每个落地步骤一行、头条结果、以及结论。每个实验的完整细节
(改了什么、为什么、所有实测数据、经验教训)放在 [`docs/exps/`](exps/),下表每行都有链接。

## 概要

FlashQwen 是一个从零实现的 Qwen3-8B 推理引擎(Go 前端 + C++/CUDA 后端,经 gRPC 通信)。本日志记录在单张
RTX 4090 上、以 **bf16 精度严格对齐(无量化)** 缩小与 **vLLM** 服务吞吐差距的过程。从 `main`(INT8、按
序列调度器)出发,我们重写了调度器、GEMM 路径以及 attention kernel。

**最终结果**(conc=32,`vllm bench serve` random,output 128,对比功能对齐的 vLLM):

| input | FlashQwen | vLLM | **% of vLLM** |
|---|---|---|---|
| 128(decode 密集) | 1318 | 1393 | **94.6%** |
| 512 | 927 | 948 | **97.7%** |
| 1024(prefill 密集) | 640 | 652 | **98.1%** |
| ShareGPT(1000 条,真实数据) | 1286 | 1328 | **96.8%** |

- 相对 `main` 基线在 conc=32/1024 上提升 **~6×**(107 → 640 tok/s)。
- 到 S14 为止,差距**随输入长度增大**(残差 = prefill 侧算力)。**S16**(mma.sync prefill-attention 重写)
  正是针对这一点,把 1024 从 92.8% 拉到 **98.1%**——曲线现在几乎拉平,最后一个结构性差距(长上下文 prefill)
  基本填平。
- 在真实对话数据(1000 条 ShareGPT,等输出预算)上 FlashQwen 是 vLLM 的 **96.8%**;经 TTFT 口径修复
  (commit `32d05ea`)后,FQ 的 TTFT 真实地低于 vLLM(conc=32 下 94.6 vs 138.8 ms——vLLM 把排队的
  chunked-prefill 前置),稳态 TPOT 高约 4%。Prefix caching 在 ShareGPT 上对两边都中性(跨请求重叠≈0)。
- decode GEMM、KV 容量/驱逐、CUDA graphs、CPU/调度 overlap 都被逐一实测并**排除**为 conc=32 瓶颈(见结论)。

## 追踪的指标

`vllm bench serve --dataset-name random`,**1024 input / 128 output**,greedy(temp 0),关闭 thinking,
并发 **32**(对齐 vLLM `--max-num-seqs 32`);单流视角另报 conc=1。诚实指标:**output tok/s**。除非特别说明,
所有对比都是对功能对齐的 vLLM(`--no-enable-prefix-caching`,bf16,0.9 mem)。

## 优化历程(每步一行)

口径数字来自 2026-06-20 的同机复盘;标准测试 = 1024/128,`% vLLM` 对比无缓存 vLLM(conc=32 = 652)。
回退的尝试(S4、S11)也作为记录保留——负结果同样重要。

| 步骤 | 改动 | conc=32 @1024 | % vLLM | 详情 |
|---|---|---|---|---|
| B0/R1/R2 | 基线 = main(INT8、按序列调度)→ 统一 token 预算调度器 + GPU 采样 | 107 → 86 | 16→13% | [00](exps/00-baseline-and-scheduler.md) |
| S3 | bf16 权重 + cuBLAS GEMM + FlashAttention-2 — *引擎本体* | **348** | 53% | [s03](exps/s03-bf16-cublas-fa2.md) |
| S4 | tensor-core + GQA 分组 attention — *尝试,已回退(−34%)* | — | — | [s04](exps/s04-tensorcore-gqa-reverted.md) |
| S5 | 算子融合(QKV/gate-up GEMM、add+rmsnorm、RoPE 表) | 356 | 55% | [s05](exps/s05-kernel-fusion.md) |
| S6 | FlashDecoding decode-attention kernel(按请求类型拆分) | **454** | 70% | [s06](exps/s06-flashdecoding.md) |
| S7 | WMMA tensor-core prefill attention | **501** | 77% | [s07](exps/s07-wmma-prefill.md) |
| S8 | prefill-attn 占用率(压缩 shared memory) | **531** | 81% | [s08](exps/s08-prefill-occupancy.md) |
| S10 | 调度器:默认 max-batch-tokens 2048 → 1024(KV cliff) | **581** | 89% | [s10](exps/s10-max-batch-tokens.md) |
| S11 | 纯 decode 步骤的 CUDA graphs — *尝试,已回退(−12%)* | — | — | [s11](exps/s11-cuda-graphs-reverted.md) |
| S12 | GQA-shared FlashDecoding(每组只读一次 K/V + KV-split) | **604** | 93% | [s12](exps/s12-gqa-flashdecoding.md) |
| S14 | activation scratch 按真实每步上界裁剪(+ 隐藏的 WMMA OOB 修复) | 605 | 93% | [s14](exps/s14-activation-scratch.md) |
| S15 | 自动 prefix caching(内容哈希 KV 复用) | 621* | — | [s15](exps/s15-prefix-caching.md) |
| **S16** | **prefill attention 重写:WMMA → mma.sync(FlashAttention 风格)** | **640** | **98.1%** | [s16](exps/s16-prefill-mma-rewrite.md) |
| S17 | 全负载/全数据综合对比 + prefix-cache 正确性修复 | — | — | [s17](exps/s17-comprehensive-comparison.md) |
| — | **vLLM(无 prefix cache),参照** | **652** | 100% | — |

*S15 的收益只在共享前缀 / cache-on 负载上体现(512-token 共享前缀 +36%);纯随机输入下 ≈ S14。

## 跨输入长度的最终结果

conc=32 output tok/s(括号为占功能对齐 vLLM 的百分比),每个落地步骤都重新干净编译并在 128/512/1024 输入上压测:

| 步骤 | in=128 | in=512 | in=1024 | 这步换来了什么 |
|---|---|---|---|---|
| **S3**  | 1094 (79.5%) | 628 (66.5%) | 348 (53.4%) | bf16 + cuBLAS + FA2 — 全面 ~4× |
| **S6**  | 1317 (95.7%) | 824 (87.3%) | 454 (69.6%) | FlashDecoding — 短输入提升最大 |
| S7/S8 | ~1328 (96.5%) | ~878 (93%) | 531 (81.5%) | prefill attn — 提升长输入 |
| S10 | 1334 (97.0%) | 882 (93.4%) | 581 (89.1%) | KV-cliff 调度修复 — 仅 @1024 |
| S12/S14 | 1341 (97.5%) | 908 (96.2%) | 605 (92.8%) | GQA-shared decode + pool 修复 |
| **S16** | 1318 (94.6%) | **927 (97.7%)** | **640 (98.1%)** | mma.sync prefill — 填平长输入差距 |
| vLLM | 1376/1393 | 944/948 | 652 | 功能对齐参照 |

两个清晰事实:(1)每个优化作用在它针对的区间且互不重叠(S6 提升 in=128;S7/S8 提升 in=1024;S10 *只*动
in=1024)。(2)到 S14 为止差距随输入长度单调(97.5% → 92.8%)——残差集中在长输入 = prefill 侧算力。S16 正是
攻这一点,所以曲线现在几乎拉平。详情见 [s16](exps/s16-prefill-mma-rewrite.md)。

## 与 vLLM 的综合对比(S17,bf16 parity)

宽口径正面对比(FQ 并发上限 = 32,`MAX_DECODE_B`)。完整表格与方法见 [s17](exps/s17-comprehensive-comparison.md)。

- **输入长度**(out=128, c=32;2048@c16):97.1% / 98.3% / 97.8% / 91.9%(128/512/1024/2048)。
- **并发**(in=512):96.4% / 95.0% / 96.6% / 98.3%(c=1/8/16/32)。
- **输出长度**(in=512, c=32):out=128 98.3%、out=512 96.1%、**out=1024 107.4%**(FQ 更快)。
- **ShareGPT 真实数据**(1000 条,c=32,等输出预算):**96.8%**(1286 vs 1328);修复后的真实 TTFT FQ 更低
  (94.6 vs 138.8 ms),TPOT vLLM 低约 4%;prefix cache 中性。详见 [sharegpt-1000-comparison](exps/sharegpt-1000-comparison.md)。

S17 还发现并修复了一个 prefix-cache 正确性 bug(S16 的 cp.async 双缓冲在大前缀命中时损坏 → 空输出;改为
单缓冲 staging 修复,commit `aaf4e0d`)。一度怀疑的高并发挂起经查是压测脚本的 GPU 清理假象,并非引擎 bug。
详情见 [s17](exps/s17-comprehensive-comparison.md)。

## 结论 — 什么已榨干、残差在哪

在 4090 的 bf16 parity 下,conc=32 瓶颈已追到底。**用数据排除**的杠杆:decode GEMM(~84% HBM 峰值——物理
地板,只有量化能突破)、CPU/调度 overlap(GPU 98.7% 忙、host bubble ~0.2%)、CUDA graphs(−12%,S11)、
KV 容量/驱逐(把 pool 扩到超过峰值需求,吞吐变化为 0,S14)、prefill-attn 占用率(S8 已吃下;GQA-shared
prefill 反而慢 0.73×)、prefill chunk size / gpu-mem-fraction(无变化)。

S16 之后长输入残差基本填平(1024 = 98.1% 无缓存 / 96.2% cache-on,原 92.8%)。剩下的很小且局部;两条继续往前
的路都**脱离 bf16 parity**:

1. **Prefix caching**(已落地,S15)——内容哈希 KV 复用,默认开。随机输入近中性,真实共享前缀负载大幅获益
   (512-token 共享前缀 +36%,与 vLLM 的 +37% 相当)。
2. **量化**(fp8/int 权重或 fp8 KV)——直接削减主导的权重带宽 / prefill-GEMM 成本,但破坏严格 bf16 对齐。

**结论:** 在同精度 + 同功能下,FlashQwen 跨输入长度是 vLLM 的 **~95–98%**,真实对话数据(1000 条 ShareGPT)上 **~97%**,
相对起点 **~6×**——剩余差距已局部化、已实测、已解释清楚。
