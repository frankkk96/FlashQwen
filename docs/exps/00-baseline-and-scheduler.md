# 起点:基线 (main) 与调度器重构 (B0/R1/R2)

## 背景/动机

FlashQwen 是一个从零实现的 Qwen3-8B 推理引擎(Go 前端 + C++/CUDA 后端,通过 gRPC 通信)。本优化日志追踪的目标是:在单张 RTX 4090 上、在 **bf16 等精度(不做量化)** 的前提下,缩小与 **vLLM** 的 serving 吞吐差距。

跟踪的核心指标(standard test)是 `vllm bench serve --dataset-name random`,**1024 input / 128 output**,greedy(temp 0),thinking 关闭;并发 **32**(对齐 vLLM 的 `--max-num-seqs 32`);同时也报告 conc=1 用于单流 / 单步视角。诚实指标是 **output tok/s**。

vLLM 参考值(conc=32 output tok/s,feature-matched 即 `--no-enable-prefix-caching`):**128 → 1376,512 → 944,1024 → 652**。

本文把三个起点条目(B0、R1、R2)合并为一篇:B0 是出发点的 `main`,R1/R2 是调度器的奠基性重构。

## 改动

### B0 — baseline = main (08392df)

- **状态**:per-sequence 调度器(单个 prefill chunk)、INT8 weight-quant matmul、原始的 paged-attention kernel。测量是在一个 `main` worktree 上进行的,cherry-pick 了一个 Go-only 的 `/v1/completions`+`ignore_eos` shim(`650fdaa`),以便能跑相同的 bench——没有任何计算改动。

### R1 — unified token-budget scheduler + GPU sampling(回归 REGRESSION,5065b2e)

- **改动**:把调度重写为 vLLM 风格的 unified token budget(chunked prefill + decode 合并到一次 merged forward),三层 paged-KV stack,GPU sampling;把 attention 替换为一个 naive 的 paged varlen kernel(one warp per (head, query-row),online softmax,无 split-K)。

### R2 — scheduler/kernel refactor(perf-neutral,291cb74)

- **改动**:纯结构性改动——精简 `step()`,合并 `grow`/`retire`,把 batch 逻辑下推到 `Request`,把各类旋钮归拢到 `SchedulerConfig`/`RuntimeConfig`,删除死掉的 kernel 代码(`gemv_kernel`;`launch_matmul`→`launch_matmul_prefill`)。没有 hot-path 或 numerics 改动。

## 实测结果

Progress 表中的相关行(canonical = 2026-06-20 journey replay,standard test = 1024 input / 128 output;`% vLLM` 对比 feature-matched no-cache vLLM,conc=32 = 652):

| Step | change | conc=1 | conc=32 | % vLLM | commit |
|---|---|---|---|---|---|
| B0 | baseline = main (per-seq scheduler, INT8, old attn kernel) | 56.7 | 106.9 | 16% | 08392df |
| R1 | unified token-budget scheduler + GPU sampling (still INT8) | 24.2 | 85.9 | 13% | 5065b2e |
| R2 | scheduler/kernel refactor (perf-neutral, Δ0) | 24.1 | 85.9 | 13% | 291cb74 |

各条目内的同期(contemporaneous)测量:

- **B0 aligned test**:conc=1 56.7 tok/s(TPOT 11.5 ms)= vLLM 的 102%;conc=32 106.9(TPOT 251 ms)= 15.3%。
- **R1 result vs B0**:**更慢**——conc=1 56.7→25.0(−56%,TPOT 11.5→33.2 ms),conc=32 106.9→87.2(−18%)。
- **R2 result vs R1**:不变——conc=1 24.9,conc=32 87.4(Δ0;两次独立测量,5065b2e ↔ HEAD)。

> 注:B0 的 conc=32 = 106.9 tok/s 是 no-cache vLLM 652 的 15%。B0 是 INT8 + 不同分支,所以它是从原始记录里引用的,而不是在 bf16 replay 里重新跑的。INT8 让它的单流(conc=1)decode 看起来不错(weight traffic 更轻),但它**在并发下崩塌**——这个崩塌正是整个优化旅程要修的。

## 经验

- **B0 read**:单流的 FlashQwen 已经能击败 vLLM(INT8 = 更少的 weight traffic,decode 是 memory-bound)。整个差距都在并发上——vLLM 的 batching 强得多。
- **R1 lesson**:这次重写换来了正确性 + 并发下的鲁棒性 + chunked prefill,但 no-split-K 的 varlen attention 比旧 kernel 慢得多,所以净吞吐回归了。调度器没问题;**attention kernel 才是欠下的债。**
- **R2 lesson**:确认差距在 kernel,而不在 scheduling/bookkeeping。为接下来的 attention 重写(split-K)清场。
