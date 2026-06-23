# FlashQwen 优化日志 — 追赶 vLLM

> English version: [optimization.md](optimization.md)

这是吞吐优化全过程的唯一、自包含记录。每个落地步骤精选一条记录：**改了什么、针对的瓶颈、实测效果、得到的经验。** 所有
数字和表格都内联呈现；细粒度的原始资产（每次运行的 CSV、profiler dump、bench 脚本）有意不纳入仓库。

## Executive summary

FlashQwen 是一个从零实现的 Qwen3-8B 推理引擎（Go 前端 + C++/CUDA 后端，经由 gRPC 通信）。
本日志记录在单张 RTX 4090 上、以 **bf16 精度对齐（无量化）** 缩小与 **vLLM** 服务吞吐差距的过程。从 `main`（INT8、按序列调度器）出发，我们重写了
调度器、GEMM 路径以及 attention kernel。

**最终结果** — 在同一台机器上重新跑完每个落地步骤，conc=32，输出 128，对比一个
**功能对齐** 的 vLLM（`--no-enable-prefix-caching`，bf16，0.9 mem）：

| input | FlashQwen (S14) | vLLM (no prefix cache) | **% of vLLM** |
|---|---|---|---|
| 128 (decode-heavy) | 1341 | 1376 | **97.5%** |
| 512 | 908 | 944 | **96.2%** |
| 1024 (prefill-heavy) | 605 | 652 | **92.8%** |

conc=1 单流：56.4 / 54.6 / 51.3 ≈ 在相同输入下达到 vLLM 的 92–97%。

**核心要点**
- 在 conc=32/1024 下相比 `main` 基线达到 **5.5×**（107 → 605 tok/s）。
- 与 vLLM 的差距 **随输入长度增大**（97.5% → 92.8%）：残余完全来自 **prefill 侧
  计算**（prefill GEMM 已到 cuBLAS/HBM 极限 + 更长上下文的 attention）。decode-heavy 的服务
  基本已与 vLLM 持平。
- decode、KV-cache 容量/驱逐，以及 CPU/调度开销都各自被测量并 **排除** 在 conc=32 瓶颈之外（见 Conclusions）。
- 这里的所有对比都是针对 **功能对齐** 的 vLLM（无前缀缓存）。vLLM 的默认配置
  （前缀缓存开启，~698/1031）是一个早期不对等的基线——它缓存了 bench 共享的
  chat-template 前缀，而 FlashQwen 当时没有对应能力——已不再用作参考。

## Goal & constraints

- 在 **Qwen3-8B**、**bf16，无量化**（公平、等精度）的前提下，于 RTX 4090（24 GB，sm_89，~1 TB/s）上缩小与 **vLLM** 的服务吞吐差距。36 层，hidden 4096，32 q-head / 8 kv-head GQA，
  head_dim 128，intermediate 12288，QK-RMSNorm，SwiGLU，无 bias。

## Standard test (the one number we track)

- 负载：`vllm bench serve --dataset-name random`，**1024 input / 128 output**，greedy（temp 0），
  关闭 thinking。并发 **32**（与 vLLM 的 `--max-num-seqs 32` 对齐）；也报告 conc=1 用于
  单流 / 单步视角。
- Harness：`bash /root/bench-compare/std.sh <label>` → 追加到 `results/std.csv`（启动 server，
  跑 conc=1 与 conc=32，打印 output tok/s）。先用 `make` 构建。
- 诚实指标：**output tok/s**。（TTFT 是个假象——FlashQwen 在 prefill 之前就发出 SSE role chunk，因此 prefill 时间被计入了 TPOT。）
- 每一步的正确性门禁：`bash /root/bench-compare/validate.sh`（数值）+ `gencheck.sh`（文本）。

## Baselines & reference (canonical: the 2026-06-20 journey replay)

本报告中的所有数字都来自 **唯一的权威数据集**——同机器上的全过程重放
（2026-06-20）：每个落地步骤都干净重建并在 128/512/1024
input 下对比一个 **功能对齐的 vLLM**（`--no-enable-prefix-caching`，bf16，0.9 mem）进行 bench。对齐的
`vllm bench serve --dataset-name random`，output 128，temp 0。

vLLM 参考（conc=32 output tok/s）：**128 → 1376, 512 → 944, 1024 → 652**。

起点 — `main`（B0）：按序列调度器，INT8 weight-quant。在 conc=32/1024 下记录为
**106.9** tok/s（为无缓存 vLLM 652 的 15%）。B0 是 INT8 + 另一个分支，因此引自原始记录，而非在 bf16 重放中重新跑。INT8 让其单流（conc=1）decode 看起来
不错（权重流量更轻），但它 **在并发下崩溃**——这次崩溃正是这趟历程要修的问题。

> 本日志的早期版本是对比 **前缀缓存开启** 的 vLLM（其默认 → ~698 /
> 1031），这并不功能对齐（FlashQwen 没有前缀共享）。该对比已废弃，此处不再使用。

## Progress

来自 2026-06-20 全过程重放的权威数字，**standard test = 1024 input / 128 output**
（`% vLLM` 是相对功能对齐的无缓存 vLLM，conc=32 = 652）。跨输入（128/512）结果见
下文 Final results。

| Step | change | conc=1 | conc=32 | % vLLM | commit |
|---|---|---|---|---|---|
| B0 | baseline = main (per-seq scheduler, INT8, old attn kernel) | 56.7 | 106.9 | 16% | 08392df |
| R1 | unified token-budget scheduler + GPU sampling (still INT8) | 24.2 | 85.9 | 13% | 5065b2e |
| R2 | scheduler/kernel refactor (perf-neutral, Δ0) | 24.1 | 85.9 | 13% | 291cb74 |
| S3 | bf16 weights + cuBLAS GEMM + FlashAttention-2 | 37.9 | **348** | **53%** | 7650654 |
| S5 | kernel fusion: fused QKV/gate-up GEMM, add+rmsnorm, RoPE table | 38.8 | 356 | 55% | c2f98a0 |
| S6 | FlashDecoding decode-attention kernel (split by request type) | 44.5 | **454** | **70%** | f0b1499 |
| S7 | WMMA tensor-core prefill attention | 44.8 | **501** | **77%** | 0dd4010 |
| S8 | attn_prefill occupancy (shrink shared, drop PVt) | 45.0 | **531** | **81%** | 392cda5 |
| S10 | scheduler: default max-batch-tokens 2048→1024 | 45.0 | **581** | **89%** | 642cc28 |
| S12 | GQA-shared FlashDecoding (read K/V once per group + KV-split) | **51.2** | **604** | **93%** | 240aaa1 |
| S14 | activation scratch → per-step bound (pool↑, KV-cliff gone) + latent WMMA OOB fix | 51.3 | 605 | **93%** | e5a99c8 |
| — | **vLLM (no prefix cache), reference** | 55.5 | **652** | 100% | — |

B0 是原始 INT8 记录（不同分支/精度）；R1→S14 是全新的 bf16 重放。S14 在本指标上
吞吐中性（攒下一个内存/鲁棒性/正确性修复）。被回退/未落地的尝试 S4、S9、S11、S13 记录在步骤条目中，不在表里。

## Step entries

> 注：下文各条目 *内部* 的绝对 tok/s 与「→ N」差距目标都是 **当时同期** 的——许多
> 是用旧 harness 测的、针对当时的（前缀缓存）vLLM 基线。它们是开发叙事；**权威数字是 Progress 和 Final-results 两张表**
> （2026-06-20 重放，对比无缓存 vLLM）。相对增量和经验仍然成立。

### B0 — baseline = main (08392df)
- **State**：按序列调度器（单个 prefill chunk），INT8 weight-quant matmul，原始的
  paged-attention kernel。在一个 `main` worktree 上测量，cherry-pick 了一个 Go-only 的 `/v1/completions`+`ignore_eos`
  shim（`650fdaa`）以便跑相同的 bench——无计算改动。
- **Aligned test**：conc=1 56.7 tok/s（TPOT 11.5 ms）= vLLM 的 102%；conc=32 106.9（TPOT 251 ms）= 15.3%。
- **Read**：单流 FlashQwen 已经胜过 vLLM（INT8 = 更少权重流量，decode 是
  memory-bound）。整个差距都在并发上——vLLM 的 batch 做得好得多。

### R1 — unified token-budget scheduler + GPU sampling (REGRESSION)
- **Change**：把调度重写为 vLLM 风格的统一 token budget（chunked prefill + decode 合并到一次
  forward），三层 paged-KV 栈，GPU sampling；用一个朴素的 paged varlen kernel 替换 attention（每个 (head, query-row) 一个 warp，online softmax，无 split-K）。
- **Result vs B0**：**更慢**——conc=1 56.7→25.0（−56%，TPOT 11.5→33.2 ms），conc=32 106.9→87.2（−18%）。
- **Lesson**：重写带来了正确性 + 并发下的鲁棒性 + chunked prefill，但
  无 split-K 的 varlen attention 远慢于旧 kernel，所以净吞吐回退了。
  调度器没问题；**attention kernel 才是欠的债。**

### R2 — scheduler/kernel refactor (perf-neutral)
- **Change**：纯结构性——精简 `step()`，合并 `grow`/`retire`，把 batch 逻辑下放进
  `Request`，把各种旋钮归拢进 `SchedulerConfig`/`RuntimeConfig`，移除死掉的 kernel 代码
  （`gemv_kernel`；`launch_matmul`→`launch_matmul_prefill`）。无热路径或数值改动。
- **Result vs R1**：不变——conc=1 24.9，conc=32 87.4（Δ0；两次独立测量，
  5065b2e ↔ HEAD）。
- **Lesson**：确认差距在 kernel，而非调度/簿记。为 attention 重写（split-K）清场。

### S3 — bf16 + cuBLAS GEMM + FlashAttention-2 (7650654)
- **Change**：三处协同的 kernel 重写，解决 R1/R2 指出的债。
  1. **去掉 INT8 weight-quant → 全面 bf16。** 权重以 bf16 加载（无 dequant），整个
     激活流水线为 bf16（norm/rope/residual/silu 内部用 fp32 累加）。现在
     已是 **与 vLLM 公平的 bf16 对齐**——早先的精度警告消失了。
  2. **所有 matmul 用 cuBLAS。** 一条 `cublasGemmEx` 路径（bf16 输入，fp32 累加，tensor core）
     替换手写的 WMMA prefill GEMM + 每次调用的全权重 dequant + batched INT8
     decode GEMV。decode/prefill matmul 的分裂被删除——合并 batch 走一条路径。
  3. **FlashAttention-2 paged kernel。** 替换那个每 (head,row) 的 varlen kernel——它把所有
     K/V 从 global 重读且零复用（O(T²)）。新 kernel：每 (q-tile, head, request) 一个 block；BM=16
     query 行（每行一个 warp）共享暂存在 shared memory 的 BLOCK 大小 K/V tile（复用系数 BM），
     带延迟归一化的 online softmax（只在最后除以 running sum），causal 整 tile 跳过。Host 按 request（qstart/qlen）对 batch 分组以形成 q-tile grid。
- **Result vs R2**：conc=32 **87.4 → 350.0 tok/s（4.0×）**——**12.5% → 50.1% of vLLM**，并且是
  `main` 基线（106.9）的 3.3×。conc=1 24.9 → 38.2（仍低于 main 的 INT8 56.7：bf16 权重读取的字节数是 INT8 的 2×，
  而单流 decode 是 weight-bandwidth-bound——这是预期中的去量化代价；追踪指标是 conc=32）。greedy 生成验证连贯。
- **Lesson**：R1/R2 是对的——差距在 kernel，而非调度。统一调度器
  的回退不仅被收复，还被远远甩开。conc=32 下与 vLLM 剩下的约 ~2× 不是 attention
  （见下文 S4）——是 GEMM（cuBLAS，近最优）+ 每步 launch/调度开销（~720
  kernel launch/step，无 CUDA graph、无 persistent batch、无前缀缓存）。单流（conc=1）
  若我们日后在意，则是另一个独立的 bandwidth-bound 议题（bf16 权重流量）。

### S4 — tensor-core + GQA-grouped attention (TRIED, REVERTED)
- **Change**：把 attention kernel 重写为用 WMMA（16×16×16 tensor core）算 S=Q·Kᵀ 与 O=P·V，
  并按 (q-tile, **KV head**, request) 每个一个 block，使一个 GQA 组的全部 4 个 q-head 复用一份
  暂存的 K/V tile（K/V 流量减少 4×）。O 累加器保留在 shared（fp32），由 thread
  循环 rescale，P·V 经 load/store_matrix_sync 累加——不依赖 WMMA fragment 布局假设。greedy 输出与 S3 逐位相同（正确）。
- **Result vs S3**：conc=32 350.0 → 355.5（持平，噪声）；conc=1 **38.2 → 25.1（−34%）**。净负。
- **Why it didn't pay off**：(1) 在 S3 的 BM=16 K/V tiling 之后，**attention 已不再是 conc=32
  瓶颈**——GEMM/launch 开销主导，所以快 2–3× 的 attention 几乎不动总吞吐。(2) conc=1 时，decode 行
  q_len=1 但 WMMA 强制一个完整 16 行 tile（浪费 16× 的计算），且 per-KV-head 分组只剩 8 个 grid block（vs 32）→ GPU 被饿着，
  attention 延迟增长，拖累单流 TPOT（26→40 ms）。
- **Decision**：回退到 S3（commit 7650654）。保留在此作为 attention 这根杠杆已耗尽的记录
  ——下一根真正的杠杆是 launch 开销（CUDA graph / kernel fusion）和 vLLM 拥有的
  吞吐特性（persistent batch、前缀缓存）。

### S5 — kernel fusion: fused QKV/gate-up GEMM, add+rmsnorm, RoPE table (landed)
- **Target**：S3 点名的每步 launch + 冗余流量开销，作为剩余 conc=32 杠杆
  （~720 kernel launch/step）。三处融合，全是 bf16 等价数学：
  1. **融合 QKV 与 gate|up GEMM。** q/k/v proj 权重在加载时按 OUT 维拼成一个
     `wqkv`（[q_dim+2·kv_dim, H]），同理 gate+up 拼成 `wgateup`（[2·I, H]）；各用一个
     `cublasGemmEx` 替换 3 个和 2 个。下游 kernel 经 offset/stride 读取交错的融合缓冲：一个新的融合 **per-head RMSNorm+RoPE**
     kernel 原位归一化+旋转 q 与 k 切片，`store_kv` 直接从 QKV 缓冲读 k/v，attention 接受一个 `q_stride`，
     而 `silu_mul` 把 gate/up 当作一行的两半读取。
  2. **融合 residual-add + RMSNorm。** `add_rmsnorm` 做 `x += residual`（写回，把
     residual 向前携带），然后一次 pass 内做 `rmsnorm(x)`——替换单独的 `add` + `rmsnorm`，省下一次 x 的整 H 读写
     和每个 residual 一次 launch。run_layers 被重构为除第一个外每个 norm 都消费待处理的 residual；一个尾随的 `add` 提交最后一层 MLP 的 residual。
  3. **预计算 RoPE cos/sin 表。** 角度只依赖 (pos, i) 且在所有
     36 层都相同；旧 kernel 每层每元素重算 `powf/cosf/sinf`（36× 浪费）。现在一个
     `[max_ctx, head_dim/2]` 表在启动时构建一次并查表（折进融合的 norm+rope kernel）。
  - 净效果：每层约 ~19 → 12 次 kernel launch（每步少约 ~252 次，~720 → ~470）。
- **Result vs S3**：conc=1 38.2 → **39.1（+2.3%）**；conc=32 350.0 → **356.1（+1.7%，在噪声内）**。
  greedy 输出验证连贯。
- **Lesson**：移除每步 ~250 次 launch + 冗余 elementwise 流量几乎不动 conc=32——所以在
  conc=32 下 launch/elementwise 开销只是 **很小一部分**；瘦长的 decode **GEMM 主导**（每步从 HBM 流式读取全部 17 GB 的 bf16 权重）。这重新校准了路线图：
  「~720 次 launch」这个数字高估了杠杆。**CUDA graph 对 conc=1（latency-bound）的帮助大于
  conc=32**；要闭合 conc=32 剩下的 2× 需要 GEMM 侧的胜利（cuBLASLt autotuning / 更好的
  decode-GEMM 形状）或 vLLM 拥有的吞吐特性（更大有效 batch、前缀缓存），而不只是 launch 消除。无论如何这些融合都保留：正确、更干净，并预先铺好 CUDA-graph 捕获将需要的 persistent-buffer 布局。

### S6 — FlashDecoding decode-attention kernel (f0b1499)
- **Target**：S5 profiling 留下的 conc=32 天花板。profile 用的是短上下文；追踪
  指标是 1024-ctx，那里 decode attention 每步读取约 ~1152-tok 的 KV。统一 attention
  kernel 处理 q_len==1（decode）很糟糕：16 个 warp 里只有 1 个活跃（~6% 占用率），而那个
  warp 串行扫过全部 ~72 个 KV tile。GEMM 当时已到 84% HBM BW（S5 profile），所以 attention 是剩下的杠杆。
- **Change**：在统一 batch *之内* 按 request 类型拆分 attention——GEMM/norm 仍合并
  （attention 本就是 per-request，所以这与统一 batching 正交，**不是** 回到
  prefill/decode-split forward）。decode 行（q_len==1）→ 新的 `decode_attn_kernel`；prefill 行
  （q_len>1）→ 现有的 tiled flash kernel。两者都接受一个 `rids[]` grid→request 间接索引，使各自
  在其 request 子集上运行，写入 `attn_` 的不相交行。在稳态 conc=32 下大多数步是纯
  decode，所以 decode kernel 承载了常见情形。
  - `decode_attn_kernel`：每 (head, decode-request) 一个 block；NW=8 个 warp 把该 request 的 KV
    [0,qpos] 切成 strided 切片；每个 warp 在寄存器中对其切片做 online-softmax，直接从 paged cache
    读取 K/V（一个 query 行没有跨行复用 → shared-memory 暂存纯属开销）；然后一次 in-block combine
    合并 NW 个 partial（online-softmax）。相比旧的 1-warp 串行扫描，达成满 warp 占用率 + NW 路 KV 并行。
  - GQA K/V 仍按 q-head 读取（一个组内 4×），接受：KV 字节约为 GEMM 的 ~4%，且一个组的
    4 个 q-head 命中相同 cache line（L2 吸收大部分）。
- **Result vs S5**：conc=32 **357 → 455 tok/s（+27.5%，TPOT 84.9 → 65.3 ms）**；conc=1 39.1 → 44.7
  （+14%，TPOT 25.8 → 22.5 ms）。**51% → 65% of vLLM**，是 `main` 基线的 4.26×。greedy 输出连贯。
- **Lesson**：S5 的 profile（短上下文）低估了 attention；在追踪的 1024-ctx 下手写的
  decode attention 确实曾是 conc=32 瓶颈——并且不同于已耗尽的 S4 WMMA 尝试（没修任何实质问题），
  FlashDecoding 风格的 **KV 并行 + 占用率** 才是正确的修法。这印证了 profiling sweep 得出的规则：GEMM/elementwise/调度都是死路；唯一一个还有余量的手写 kernel 交付了成果。下一段与 vLLM 的差距（455→698）很可能是 prefill 侧
  attention + GQA 4× 冗余 KV 读取（一个 GQA-shared decode kernel）以及 conc=1 的 CUDA graph。

### S7 — WMMA tensor-core prefill attention (0dd4010)
- **Target**：S6 把 decode 拆出去之后，仅 prefill 的 attention kernel 对于一个 compute-bound 的 512×L×128 matmul 仍在用 CUDA-core FMA 点积
  （per-key warp-shuffle 归约）——没有 tensor core。S4 的 WMMA 尝试失败是因为它被用在 *decode*（q_len=1 浪费 16 行 WMMA tile 的 15/16 + per-kv-head 分组饿死 grid）；如今 decode 已有自己的 kernel，
  把 WMMA 用在 prefill 路径（q_len 高达 512，满 tile）才是正确的应用，S4 的失效模式消失了。
- **Change**：`attention_prefill_wmma_kernel`——带 WMMA 16×16×16 bf16（fp32
  累加）的 FlashAttention-2，每 (16-query-tile, head, request) 一个 warp。Q 直接从融合 QKV 缓冲读，
  K/V 直接从 paged cache 读（一个 16-key tile == 一个 page，因为 block_size==16），全部用 WMMA
  load——不物化 S。online softmax + 延迟归一化；O 保留在 shared fp32 并
  按 tile rescale（可移植——不依赖 WMMA fragment 布局假设）。在 prefill 子集上经
  `launch_attention_prefill` 派发；当 block_size!=16 或 head_dim!=128 时回退到 FMA。
- **Result vs S6**：conc=32 **455 → 493 tok/s（+8.5%，TPOT 65.3 → 59.8 ms）**；conc=1 持平（44.7 → 44.9
  ——单流是 decode-bound，prefill 是一次性代价，摊到 128 个输出上）。**65% →
  71% of vLLM**，是 `main` 基线的 4.6×。输出验证连贯。
- **Lesson**：prefill attention 确曾是 conc=32 中有意义的一块（+8.5% 即证明），而 S4
  的想法（tensor core）一直是对的——只是错用在了 decode 上。按 request 类型拆分 attention（S6）是解锁它的关键：每个 regime 现在都得到它想要的 kernel（decode 用 FlashDecoding，prefill 用
  WMMA-FA）。与 vLLM 剩下的差距（493→698）：GQA-shared decode（4× 冗余 KV 读取——
  判为边际收益，KV 约为字节的 ~4% 且被 L2 缓存）、conc=1 的 CUDA graph，以及这个 WMMA kernel 的 1-warp/block
  占用率（更大的 query tile / 每 block 更多 warp 也许能再推一把）。

### S8 — attn_prefill occupancy: shrink shared memory (392cda5)
- **Target**：S7 的 WMMA prefill kernel 是 1 warp/block，约 ~18 KB shared → 每 SM 只有约 ~5 个 block
  常驻（~8% warp 占用率）。warp 这么少，tensor core 在 per-tile softmax（CUDA-core 工作）期间会 stall——没有别的可以重叠。
- **Change**：去掉 `[16,128]` 的 `PVt` 临时缓冲（8 KB）。不再「先 rescale 整个 O，再加完整
  P·V」，而是经一个小的 `[16,16]` 临时缓冲按每个 16×16 d-tile 折叠两者：`O[:,d] = O[:,d]*corr + (P·V)[:,d]`。
  每个 O 元素仍只 rescale 一次、只得到一次它的 P·V——逐位相同的数学。shared ~18 KB →
  ~11 KB → 每 SM ~9 个 block（常驻 warp 约 2×）。
- **Result vs S7**：conc=32 **493 → 528 tok/s（+7.1%，TPOT 59.8 → 55.5 ms）**；conc=1 持平（44.9 → 45.2）。
  **71% → 76% of vLLM**，是 `main` 基线的 4.94×。输出连贯。
- **Lesson**：一个 1 warp/block 的手写 WMMA kernel 即便 tensor-core 吞吐看起来没问题也是占用率被饿着的
  ——释放 shared memory 让常驻 warp 翻倍，收回了 7%。很可能还有余量（multi-warp block / KV-split 会进一步推高占用率），但那是更大的重写、带着惯常的 WMMA 正确性
  风险；廉价的 shared 削减已攒下大部分容易拿的收益。

### S10 — scheduler: default max-batch-tokens 2048 → 1024 (642cc28)
- **Target**：扫描调度器旋钮（`sched_sweep.sh`）暴露了 `max-batch-tokens`
  （max_num_batched_tokens）是一根大杠杆，之前的 max-prefill-tokens sweep 漏掉了。
- **Root cause found**：在 conc=32 / 1024-in 下，KV 池（~36.7k tokens）正好坐在需求处
  （32×1152 = 36864）。默认 mbt=2048 让一步能接纳约 ~4 个并发 prefill chunk（4×512），
  把峰值 KV 需求顶过池子 → recompute-preemption 抖动 → 527 tok/s。mbt=1024 把
  prefill 串行化到足以保持在池子之下 → 589（+11%），并有更好的 TTFT/TPOT，在 **与
  vLLM 默认相同的 0.9 VRAM** 下（公平）。
- **Verified the mechanism**：用一次 `--gpu-mem-fraction` sweep（穿透到 Go）：把池子
  扩到 43.5k tokens 修复了 mbt=2048（527 → 588），但 **没有** 超过 mbt=1024 的 589。所以 ~588 是
  no-preemption 天花板——两条路径（限并发，或扩池）到达同一处，
  与 vLLM（698）剩下的差距是 **计算/框架，而非 KV/抢占**。
- **Result vs S8**：conc=32 528 → **589（+11%）**；conc=1 不变（~45，单流与
  mbt 无关）。**76% → 84% of vLLM**，是 `main` 基线的 5.5×。（gpu-mem-fraction 默认保持
  0.9；该 flag 现在对用其他 VRAM/模型的用户开放。）
- **Lesson**：与输入长度耦合（用户的判断）——mbt 甜点存在是因为 std test
  坐在 KV cliff 上；在 512-in（KV 装得下）时 mbt 无关（~890 tok/s），在 2048-in（超 2×）时它
  崩溃。公平、免费的胜利是经 mbt 限制接纳；追过 588 意味着计算，而非内存。

#### inlen × mbt sweep — throughput is dominated by KV capacity vs demand (2026-06-20)
`mbt_inlen.sh`，conc=32，按 (input length, max-batch-tokens) 的 otps：

| inlen | KV needed (32×(in+128)) | best otps | mbt effect |
|---|---|---|---|
| 512  | 20480 (57% of pool) | ~890 | flat (mbt irrelevant — KV fits) |
| 1024 | 36864 (≈pool, the cliff) | ~589 | small mbt wins (eases preemption) |
| 2048 | 69632 (190% of pool) | 78 → 7 | collapses; bigger mbt = worse (thrash) |

被追踪的 1024/128 指标正好坐在 KV-capacity cliff 上——这就是为什么这里的一切都如此
敏感。吞吐首先由 KV 池容量 vs 需求支配，其次才是 kernel。

### S11 attempt — CUDA graphs for pure-decode steps (TRIED, REVERTED, 2026-06-20)
把纯 decode forward（run_layers + lm_head + sample）按每个 batch size B 捕获进一个 CUDA graph 并
回放（把 forward 拆成 eager upload_inputs + 可捕获的 compute；固定一块 cuBLAS workspace 并
固定 block-table stride 以保证 launch 有效；捕获失败时优雅回退）。捕获
**成功** 且输出保持连贯——但 **净负**：conc=1 持平（45.2 → 45.7），conc=32
**589 → 520（−12%）**。已回退。
- **Why no conc=1 win**：conc=1 decode 是 **weight-bandwidth-bound**，而非 launch-bound——每个 token 读取
  全部 17.3 GB 权重（~17.3 ms 下限；我们的 TPOT 22 ms ≈ HBM 的 78%）。那 ~470 次 launch 异步运行并
  与那 17 ms 的 GPU 工作重叠，所以消除它们几乎省不下 wall-clock。（vLLM 的 conc=1 17.9 ms ≈ 那个
  17.3 ms 下限；它的优势在于更精简的 *eager* 每步工作——H2D + sync——这是 graph 覆盖不到的。）
- **Why conc=32 regressed**：GPU 已经 ~97% 忙（没有 launch 间隙可藏），且 B 在一轮中从 32 漏到 1，所以每个新 batch size 都付一次性 eager-warmup + graph-instantiate → 净开销。
- **Takeaway**：FlashQwen 在任何并发下都不是 launch-bound（32 时饱和，1 时 bandwidth-bound），
  所以 CUDA graph——它只移除 launch 开销——在这里无可收复。这关闭了
  搜索：与 vLLM 剩下的差距（589→698）是 HBM 带宽 + vLLM 更精简的每步 eager 路径，而非
  一个我们能在不量化的情况下修的 kernel/launch 问题。

#### S9 attempt — attn_prefill head-dim split (TRIED, REVERTED, 2026-06-20)
S8 之后，尝试 2 warps/block 拆分 head 维（warp 0 处理 dims [0,64)，warp 1 处理 [64,128)；各
为部分 S 收缩自己那半，warp 0 求和 + softmax，各自为自己的输出半做 P·V）。Os
保持 8 KB（分区，不复制），所以占用率从 9 → ~14 warps/SM 上升。输出连贯，但
conc=32 528 → 534（**+1.1%，在噪声内**）。部分-S 合并的额外 `__syncthreads`（耦合
两个 warp）大致抵消了占用率收益——而且 S8 已攒下真正的占用率胜利，所以
attn_prefill 已不再 occupancy-bound。已回退（保留更简单的 1-warp S8）。Lesson：占用率
有一次廉价胜利（S8 的 shared 削减）；过此以后，更多 warp 在这里不划算。

#### S5 ablation — which of the three fusions actually paid (2026-06-19)
每项改动在 S3 base 上单独隔离（1024/128，temp 0）；只有引擎不同。

| variant | conc=1 | conc=32 | vs S3 (conc=32) |
|---|---|---|---|
| S3 base | 38.2 | 350.0 | — |
| A: #5 RoPE cos/sin table only | 38.2 | 350.2 | ~0 (noise) |
| B: #7 add+rmsnorm fusion only | 38.3 | 350.0 | ~0 (noise) |
| C: #6 fused QKV/gate-up GEMM only | 38.9 | 355.8 | **+1.7%** |
| S5: all three | 39.1 | 356.1 | +1.7% |

**Finding：没有一项是回退，但 #5 与 #7 是净零——整个 S5 收益来自 #6（融合 GEMM）。**
为什么 #5/#7 在这里是安慰剂：RoPE 与 add/rmsnorm 这些 elementwise kernel 与 GEMM 相比微不足道，
而在 conc=32 下这一步是 GEMM/HBM-bound，所以消除它们的 launch + 冗余流量并不动
wall clock。#6 有帮助是因为把 3→1 和 2→1 个 GEMM 折叠给了 cuBLAS 更宽的 N（在瘦长的 M=32 decode 形状上更好的 tensor-core 利用率）外加更少 launch。在 #6 之上叠加 #5+#7
（C→S5）什么都没变（355.8→356.1）。无论如何三项都保留：正确、不为负、为最终的 CUDA-graph
捕获减少 launch——但经验是 **GEMM 形状的胜利才是杠杆，elementwise/launch
融合不是**，在这个 batch size 下。

### S12 — GQA-shared FlashDecoding decode attention (2026-06-20)
- **What**：S6 的 decode kernel 每 (q-head, request) 跑一个 block；以 Qwen3 的 4:1 GQA，每个
  kv-head 的 K/V 都被其组内每个 query head 重读一次（4×）。新的两阶段 kernel：phase 1 = 每
  (kv-head, request, **KV-split**) 一个 block 计算该组全部 G=4 个 q-head，把每个 K/V
  元素 **一次** 加载进寄存器并在 4 个 head 间复用；phase 2 合并 per-split 的
  FlashDecoding partial。`grid.z` 的 KV-split（ksplit = clamp(128/n_decode, 1, 16)）即使在 n_kv 的 block 数比 n_heads 少 4× 时也让 4090 保持饱和（conc=32：8·32·4 = 1024 个 block × 8 warp =
  8192 warp = 满）。旧的 per-head kernel 保留为回退（head_dim≠128 或不支持的 group）。
- **Why it works now (vs the spent S4 GQA attempt)**：S4 按 kv-head 分组但只用了 n_kv 个 block
  → GPU 饿着 → 看起来持平。KV-split 才是让分组成为净胜利的关键。L2 已吸收 4× 重读中大部分冗余的 *HBM* 流量，但冗余的 L2 带宽 + load
  指令 + score 归约仍要花 ~2.3×（microbench：0.246→0.105 ms/layer）。
- **Result**：conc=32 **589 → 608.8（+3.4%，84.4%→87.2% of vLLM）**，TPOT 54→52.1 ms；conc=1
  **45.7 → 51.3（+12.2%）**，TPOT 22→19.6 ms。greedy 输出验证逐位连贯（Rayleigh 答案 ok）；
  对比参考 kernel 的独立数值检查：对 KVLEN ∈ {1..2000}，max|Δ| ≤ 1e-4（bf16 噪声）。
- **Why conc=1 gains more than conc=32**：单请求 KV（~4.7 MB/layer）装进 L2，所以 4×
  重读纯属浪费的 L2 带宽——被干净移除。conc=32 时 KV（151 MB/layer）溢出
  L2（更 HBM-bound），且 1024/128 负载是 prefill-heavy（attn_decode 仅占 GPU 时间的 ~13%），所以端到端的占比更小。
- **Lesson**：一个 kernel 隔离的加速（2.3×）在端到端被稀释到 +3.4%，因为 decode attention 在 prefill-heavy 负载上只占 GPU 时间的少数。剩下的手写热 kernel 是
  **attn_prefill**（~11%），它仍按 GQA 组重读 K/V 4×——下一根 attention 杠杆。

#### S13 attempt — GQA-shared PREFILL attention (TRIED via microbench, REJECTED, not landed)
S12 的 decode 胜利之后，在 `attn_prefill` 上尝试同样的 GQA-sharing：每 (q-tile, kv-head,
request) 一个 block，G=4 个 warp（组内每个 q-head 一个）共享一份暂存在 shared 的 K/V page，使每个
K/V tile 从 global 只读一次而非 4×（每个 q-head 一次）。Microbench（`prefill_probe.cu`，
R=4 reqs × qlen 512）裁定：**0.73× — 慢 37%**（0.684 vs 0.499 ms/call），而且那还是
*有利* 的构建（Os 缩到 bf16；自然的 fp32-Os 版本需要 51968 B shared > 48 KB
静态上限，根本编不过）。为什么它输了，不像 decode：
- **prefill attention 是 compute(WMMA)/occupancy-bound，而非 K/V-load-bound。** 共享 K/V 的 *load*
  省的是本不构成瓶颈的东西，而 4× 的常驻 `Os[16,128]` 累加器 + 跨 warp 的
  `__syncthreads` 屏障花掉了占用率与 stall。（decode 没有大 O 累加器——
  只有 DPL=4 寄存器/head——并且有收益是因为它确实 bandwidth/L2-bound。）
- 与 S8（prefill occupancy-bound；胜利在于 *缩小* shared）和 S9（经 head-dim
  split 更多 warp → 持平）一致。prefill attention 的占用率已被榨干；GQA-sharing 只会增加 shared。
**Lesson：GQA K/V-sharing 对 bandwidth-bound 的 decode 路径划算（S12），但对
compute/occupancy-bound 的 prefill 路径净负。在此架构上不要在 prefill 上重试。**

<!-- Append one ### entry per landed step: What / Why / Change / Result (vs prev) / Lesson -->

### S14 — activation scratch sized to the real per-step bound (2026-06-20)
- **What**：`max_rows_`（所有激活缓冲都按其行数定大小）原为 `max(max_ctx=4096,
  max_batch_tokens=1024)` = 4096，但一次 forward 永远不会超过 `max_batch_tokens` 行（调度器的
  每步预算 cap 住 T；prefill 在其之下分块）。改为按 `max_batch_tokens + 16` 定大小。
- **Why the +16**：暴露了一个潜在 OOB——WMMA prefill-attention 的 `load_matrix_sync` 读取一个完整
  16 行 Q tile，所以一个非 16 对齐的末尾 chunk 会越界读取至多 15 行（无害，被
  `grow<ql` 掩盖）。旧的 4096 定大小吸收了它；按正好 1024 定大小使 T=1024 越界读出
  缓冲 → illegal memory access（model_runtime.cu:272）。+16 给出一个 Q-tile 的余量。
- **Result**：weights+activations 17.3→17.0 GB；KV 池 **36,656 → 39,040 tokens**（2291→2440 blocks），
  现在 **高于** conc=32/1024 的峰值需求（36,864）→ KV-capacity cliff 被消除。吞吐：
  conc=32 512 912.8→915.0，1024 605.6→606.0 — **不变（噪声）**。连贯。
- **Lesson（坐实 KV 假设）**：把池子扩过峰值需求——保证零
  抢占——使吞吐变动为 0。**KV cache 大小 / 驱逐不是 conc=32 瓶颈**
  （在 S12 之后重新确认 S10，现在是公平的内存修复而非不公平的 gpu-mem 抬升）。无论如何保留：
  修了潜在 OOB，收回 0.36 GB（池子现在 ≈ vLLM 的 40,816），并为更高并发 / 更长上下文下的鲁棒性移除了 cliff。

### S15 — automatic prefix caching (branch `feat/prefix-caching`, fad12b7, 2026-06-22)
- **What**：vLLM 风格的 content-hashed KV 复用——相同的 prompt 前缀只 prefill 一次，缓存的
  KV block 被拼接到后续请求上。这落地了 Conclusions 中曾标记为「越过差距的 off-table 路线」（route #1）的特性，并把长期存在的警告
  （「我们的头条差距是对比一个 *带* 前缀缓存的 vLLM，而我们当时没有」）变成了
  功能对齐、双开的对比。
- **Change**：三层结构镜像 vLLM。**BlockPool** 增加 per-block refcount + LRU free queue +
  一个 content-hash→block 注册表；refcount-0 的 block 可回收但被保留（保留 hash 映射），以便
  相同前缀能复活它，而一次新分配从 LRU 队首弹出并驱逐该 block 的映射。**Scheduler** 在 (prompt ++ output) 上串联 64-bit block hash；在接纳时
  `acquire_prefix()` 把缓存的前缀 block 拼接到 block table 上并把 `computed_` 推过
  它们（始终至少留 ≥1 token），而每次 forward 之后 `cache_filled()` 注册新填满的 block
  （一个完成请求的 block 为下一个相同 prompt 保持缓存；一个被抢占的请求在恢复时
  重新获取它自己仍被缓存的前缀）。正确性建立在绝对位置 KV 寻址 + 16 对齐的整 block 复用之上，所以 chunked-prefill 的 forward 路径不变。默认开启；**`FQ_PREFIX_CACHE=0` 关闭它** 用于 A/B，并且 `[kvstat]` 现在报告前缀命中率。
- **A/B（2026-06-22，与全过程重放同口径：`vllm bench serve --dataset-name random`，
  out=128，temp 0，chat endpoint，enable_thinking=false；harness `/root/bench-compare/prefix_test.sh`）。
  conc=32 output tok/s，每个引擎前缀缓存 OFF → ON：**

  | workload (conc=32) | FQ off → on | ΔFQ | vLLM off → on | ΔvLLM | FQ-on / vLLM-on |
  |---|---|---|---|---|---|
  | 128 random | 1350 → 1359 | +0.6% | 1406 → 1407 | ~0 | 96.6% |
  | 512 random | 920 → 953 | +3.6% | 962 → 982 | +2.0% | 97.1% |
  | 1024 random | 615 → 630 | +2.4% | 668 → 681 | +1.9% | 92.4% |
  | **512 shared + 512 random** | **621 → 844** | **+35.9%** | **666 → 915** | **+37.4%** | **92.2%** |

  conc=1（单流）两者都持平——只有 template 前缀是共享的：FQ 56/55/51 ≈ vLLM 58/57/56 的 97/96/92%。（FQ-off prefix_hit 确认为 0%；FQ-on 在纯随机上 ~8%，在共享前缀探测下
  攀升到 ~22%——vLLM 报告的命中率与之吻合：~6% → ~19%。）
- **Result / Lesson**：在 *纯随机* prompt 上收益不大（+0.6–3.6%），因为唯一共享的
  前缀是 Qwen3 chat-template 前导（~8% 的 token）——而在 conc=32 下瓶颈是
  prefill/decode 计算，而非省下的那点小 prefill。该特性的真正价值出现在 prompt
  **真正共享前缀** 时（system prompt、RAG context、few-shot、多轮）：一个 512-token 共享
  前缀带来 **+35.9%（FQ）≈ +37.4%（vLLM）**——FlashQwen 的前缀缓存 **与
  vLLM 一样有效**。关键是，现在两个引擎都功能对齐 *包括* 前缀缓存，FlashQwen 在各输入下保持
  **92–97% of vLLM**——与下文无缓存全过程重放 **相同的 parity band**。这
  关闭了前缀缓存这条线：该特性不再是缺失的能力，残余差距仍是 Conclusions 中指出的 prefill 侧计算，而非缺失的特性。

### S16 — prefill attention rewrite: WMMA → mma.sync FlashAttention-style (branch `feat/prefix-caching`, 2026-06-23)

**What.** 对被追踪的 1024/conc32 步（前缀缓存开启）做 nsys 重新 profile，把 GPU 时间拆成
**GEMM ~72% / prefill-attn ~11% / decode-attn ~12% / elementwise ~5%**——GEMM 是与 vLLM *完全相同* 的 cuBLAS
路径（相同的 `ampere_*_s1688gemm_*` kernel，相同的 ~1.12 ms/call），所以唯一还有余量的手写热点
是 attention。对 vLLM 的直接 trace 显示它跑 **一个** 融合的 `flash_fwd_splitkv`
（FlashAttention），约 10%，而 FlashQwen 是两个手写 kernel。

**先尝试并被否决——prefill split-K**（`FQ_PREFILL_KSPLIT`）：把 decode 的 KV-split 镜像到
WMMA prefill kernel 上。**0 e2e 收益**——prefill 即便在 conc=1 下也已有数千个独立 block
（qtiles×heads×reqs），所以从不会 block-starved；瓶颈是 *per-block 占用率*
（8 KB shared O 累加器 + 1 warp/block），split-K 改变不了它。已回退。

**Change（真正的修复）.** 把 `attn_prefill_kernel`（WMMA，1 warp/block，O 在 shared）重写为
`attn_prefill_mma_kernel`，为 sm_89 移植 FlashAttention 的技术：
1. **O 累加器放在寄存器** 经裸 `mma.sync.m16n8k16.f32.bf16.bf16.f32`（不是 `nvcuda::wmma`）——
   释放那曾是占用率瓶颈的 8 KB shared O，并把 S 保留在寄存器（无 shared 往返）。
2. **64 行 M tile = 4 warps/block**（每 warp 16 行）。
3. **K/V 每 16-key tile 暂存到 block-shared 一次，被全部 4 个 warp 复用**（KV 流量减少 ~4×——
   单根最大的杠杆）；P 经一个很小的 shared 缓冲传递以跳过 C→A fragment 重打包；online
   softmax 在寄存器中经 quad `__shfl_xor` 完成。
4. **cp.async double-buffer** K/V 流（在计算 tile n 时预取 tile n+1）。
在独立 microbench 中对一个 fp32 参考验证（数值上与 WMMA 相同，max|Δ|~7e-4；
**不带 cp.async 2.5×，带 cp.async 2.8×**）。开关 `FQ_PREFILL_V2`（默认 1）；否则回退到 WMMA。

**Result — 同会话 A/B（`FQ_PREFILL_V2` 0=WMMA / 1=mma），conc=32 output tok/s，两种缓存状态。**
无缓存的 S14/WMMA 行复现了历史全过程重放（606 vs 605 @1024；vLLM 652 相同），
确认两者可直接对比。

*No prefix cache（功能对齐，扩展全过程重放表）：*

| in | S14 (WMMA) | **S16 (mma)** | ΔFQ | vLLM | S16 % of vLLM |
|---|---|---|---|---|---|
| 128  | 1315 | 1318 | +0.2% | 1393 | 94.6% |
| 512  | 913  | 927  | +1.5% | 948  | 97.7% |
| **1024** | **606** | **640** | **+5.6%** | **652** | **93.0% → 98.1%** |

*Prefix cache ON（分支默认 / 部署模式）：*

| in | S15 (WMMA+cache) | **S16 (mma+cache)** | ΔFQ | vLLM+cache | S16 % of vLLM |
|---|---|---|---|---|---|
| 128  | 1351 | 1352 | ~0    | 1407 | 96.1% |
| 512  | 940  | 955  | +1.7% | 982  | 97.3% |
| **1024** | **621** | **655** | **+5.5%** | **681** | **92.4% → 96.2%** |
| 512 shared + 512 random | 831 | 870 | +4.7% | — | — |

in=128 和所有 conc=1 都持平（prefill-attn 在那里占比可忽略）——符合预期。该重写恰好抬升了它瞄准的长上下文、prefill-heavy 区间：**在 1024/conc32 下与 vLLM 的差距从
~7% 收窄到 ~2%（无缓存）/ ~4%（缓存开启）**，这是 bf16 对齐下有史以来最小的。

**Lesson.** microbench 的 2.8× 恰如 profile 预测的那样稀释为 +5.5% e2e（prefill-attn ~11% of
GPU → 减半 → ~5% 实现）；cp.async 只加了最后 ~0.9%（一旦头一个 2.5× 缩小了 attention 占比后便递减）。**decode 保持不变，并刻意 NOT 统一进一个 mma kernel：** FA 为库的可维护性把它的 mainloop 复用于 decode（splitkv），但在 q_len=1 时 mma 浪费 M tile 的 15/16
（S4 教训），而 FlashQwen 的 SIMT decode kernel 有一个 FA 缺乏的 GQA 寄存器复用优势——两个
专用 kernel（S6 的 split-by-request-type）在这里胜过一个统一 mma kernel。完整的 FA-vs-FlashQwen
对比：[`attention-vs-flashattention.md`](attention-vs-flashattention.md)。

### S17 — 与 vLLM 的全负载/全数据综合对比 + 一个 prefix-cache 正确性修复 (2026-06-23)

在严格 bf16 parity 下(同 tokenizer、max-model-len 4096、max-num-seqs 32、gpu-mem-util 0.9、seed 1234、
temp 0、warmup 丢弃)对最终引擎与 vLLM 做一次宽口径正面对比。FlashQwen 的并发上限是 **32**
(`MAX_DECODE_B`,decode kernel 的编译期常量),所以并发扫描封顶在 32。任何会产生大 prefix-cache 命中的
负载都在**两边都 cache-off** 下跑以保证公平(见下方修复说明)。所有数字均为 `vllm bench serve` 的
Output token throughput (tok/s)。

**吞吐 vs 输入长度**(out=128, conc=32;in=2048 用 conc=16——32×2176 token 会超出 39k 的 KV pool):

| input | FlashQwen | vLLM | **% of vLLM** |
|---|---|---|---|
| 128  | 1345.7 | 1386.0 | **97.1%** |
| 512  | 944.7  | 961.3  | **98.3%** |
| 1024 | 638.3  | 652.6  | **97.8%** |
| 2048 (c16) | 304.9 | 331.8 | **91.9%** ← 最弱(长上下文 prefill) |

**并发扫描**(in=512, out=128)与**输出长度**(in=512, conc=32):

| conc | FQ | vLLM | % |  | output | FQ | vLLM | % |
|---|---|---|---|---|---|---|---|---|
| 1  | 54.9  | 56.9  | 96.4% |  | 128  | 944.7  | 961.3  | 98.3% |
| 8  | 358.9 | 377.8 | 95.0% |  | 512  | 1200.8 | 1250.0 | 96.1% |
| 16 | 604.7 | 626.0 | 96.6% |  | 1024 | 1132.9 | 1055.0 | **107.4%**(FQ 更快) |
| 32 | 944.7 | 961.3 | 98.3% |  | | | | |

**真实数据 — ShareGPT**(500 prompts, conc=32, 两边 cache-off, 固定 out=128):

| | FlashQwen | vLLM | **% of vLLM** |
|---|---|---|---|
| Output tok/s | **1170.2** | 1159.1 | **100.9%** |
| Mean TTFT | **5.4 ms** | 118.5 ms | FQ 约低 22× |
| Mean TPOT | 27.4 ms | 23.9 ms | vLLM 低 13% |

**三点观察。**(1)**TTFT 与 TPOT 互补:** 饱和时 FlashQwen 的 TTFT 极低(3–22 ms),vLLM 为 355–1166 ms
(vLLM 把排队的 chunked-prefill 前置),而 vLLM 的稳态 TPOT 略低。(2)**长输出(out=1024)FQ 反超 7%**——
decode 密集,FQ 的低 TTFT 在长生成里累积,盖过了 TPOT 的小劣势。(3)**in=2048 是最弱点(91.9%)**——
即便经过 S16 的 mma 重写,长上下文 prefill 仍是 FQ 的相对短板。

**Prefix-cache 收益。** 在强制共享前缀的负载上(1024 共享 + 128 独有, conc=32),vLLM 的缓存把吞吐从
**308 提升到 513 tok/s(+66%)**,并让该负载能放进 KV pool。在 ShareGPT(跨请求重叠很少)上,两边缓存
基本中性(FQ cache-on 1168 ≈ cache-off 1170 ≈ vLLM 的 100.9%)。

**发现并修复的 bug(commit `aaf4e0d`)。** 这次矩阵暴露了一个正确性缺陷:`FQ_PREFIX_CACHE=1` 时,带**大**
prefix-cache 命中的请求返回空补全(或 502)。根因:S16 的 prefill kernel 里 **cp.async 双缓冲预取**有个时序
hazard,只在引擎逐层背靠背启动的真实上下文里、且只在 cache-hit/chunked-tail 形状下(少 query 行 + 多 K/V
tile——全新 prefill 永远不会产生这个形状)触发。诊断:`FQ_PREFILL_V2=0`(WMMA)正常、kernel 间插 stream
sync 正常、隔离单 kernel 在 `compute-sanitizer --tool racecheck` 下干净——即**跨 kernel 异步 hazard**,不是
kernel 内竞争。修复:**单缓冲同步 cp.async staging**(保留 register-O + mma.sync + block-shared K/V 复用;
overlap 本来只值 ~0.9% e2e,且 1024/conc32 cache-off 吞吐不变,仍 639)。修复后,cache-on 在 conc=32 真实
ShareGPT 上正确稳定(500/500,0 失败),命中到 ~88 block 的前缀也输出正确。(另外曾怀疑的"conc=32 cache-on
挂起"经查是压测脚本的 harness 假象——频繁重启 server 导致下一个引擎在启动时 OOM——并非引擎 bug。)

## Final results — journey replay across input lengths (2026-06-20)

每个落地步骤都 `git checkout`、干净重建、在 128/512/1024 input（output 128）下于同一
机器/同一天 bench，对比一个功能对齐的 vLLM（`--no-enable-prefix-caching`）。

**conc=32 output tok/s（及 % of feature-matched vLLM）：**

| step | in=128 | in=512 | in=1024 | what this step bought |
|---|---|---|---|---|
| R1  | 248 (18.0%) | 139 (14.7%) | 86 (13.2%)  | INT8 unified scheduler — correct but kernel-bound |
| R2  | 247 (18.0%) | 139 (14.8%) | 86 (13.2%)  | refactor, perf-neutral |
| **S3**  | 1094 (79.5%) | 628 (66.5%) | 348 (53.4%) | **bf16 + cuBLAS + FA2 — the engine; ~4× everywhere** |
| S5  | 1141 (82.9%) | 648 (68.6%) | 356 (54.6%) | fused QKV/gate-up GEMM (the only fusion that paid) |
| **S6**  | 1317 (95.7%) | 824 (87.3%) | 454 (69.6%) | **FlashDecoding — biggest lift at short input (decode)** |
| S7  | 1326 (96.4%) | 860 (91.1%) | 501 (76.8%) | WMMA prefill attn — lifts long input (+7pt @1024) |
| S8  | 1328 (96.5%) | 878 (93.0%) | 531 (81.5%) | prefill occupancy — lifts long input (+4.7pt @1024) |
| S10 | 1334 (97.0%) | 882 (93.4%) | 581 (89.1%) | max-batch-tokens=1024 — **only @1024** (+7.6pt, the cliff) |
| S12 | 1338 (97.2%) | 906 (96.0%) | 604 (92.7%) | GQA-shared decode attn — mid/long input |
| **S14** | **1341 (97.5%)** | **908 (96.2%)** | **605 (92.8%)** | pool fix (throughput-neutral) |
| **S16** | **1318 (94.6%)** | **927 (97.7%)** | **640 (98.1%)** | **prefill attn WMMA→mma.sync — lifts long input most (+5.6pt→98% @1024)** |
| vLLM | 1376/1393 (100%) | 944/948 (100%) | 652 (100%) | feature-matched reference |

（S16 行 + 其 vLLM `%` 是 2026-06-23 重测的同会话 A/B——S16-off/WMMA 对照复现了
历史 S14 行：606 vs 605 @1024，vLLM 652 相同——所以该行可直接对比。S15
（前缀缓存）在此省略，因为前缀缓存 OFF 时它与 S14 相同；其效果见上文 S15 条目的缓存开启 A/B。vLLM 128/512 单元：历史 / S16 会话。）

**Reading the chart.** 两个干净的事实：(1) 每项优化作用于它瞄准的 regime 且彼此
不重叠——S6（decode attn）抬升 in=128 最多；S7/S8（prefill attn）抬升 in=1024 最多；S10（KV-cliff 调度器修复）*只* 移动 in=1024。(2) 直到 S14，与 vLLM 的差距是输入长度的单调函数
（97.5% → 96.2% → 92.8%）——残余集中在长输入，即 prefill 侧计算。**S16 的
mma.sync prefill-attention 重写随即恰好攻击了那个长输入残余**，把 1024 从
92.8% 抬到 vLLM 的 **98.1%**（512 抬到 97.7%）——所以曲线现在跨输入长度近乎平直，而
长上下文 prefill 差距，这最后一个结构性差距，在 bf16 对齐下已基本闭合。

## Conclusions — what's exhausted and where the residual lives

> **Update (S16, 2026-06-23)：** 下文的 prefill-attention 重写闭合了本节归因为「prefill 侧计算」的
> 大部分长输入差距——1024/conc32 现在是 **98.1% of vLLM no-cache /
> 96.2% cache-on**（曾为 92.8%）。下文分析对 *为什么* 存在差距以及哪些杠杆
> 是死路仍然成立；它标记的那一根活的 attention 杠杆现在已被拿下。


在 4090 上的 bf16 对齐下，conc=32 瓶颈已被追到底。尝试过并 **排除** 的杠杆
（每个都有数据，而非推理）：

- **Decode GEMM** — 已在 HBM peak 的 ~84%；两个引擎都撞到同一物理下限（只有量化
  能打破它，off-table）。cuBLAS `cublasGemmEx(DEFAULT)` 已按每个 shape/M 选取 aspect-ratio 最优的 kernel
  （split-K 自动调优：large-K 的 `down` 总是 split，large-N 的 `gateup`/`lm_head` 从不；
  cuBLASLt autotune 在 M≥8 时打平，只在 M=1/conc-1 时胜出）。GEMM 对 conc=32 已关闭。
- **CPU/调度重叠（async scheduler）** — 稳态 GPU **98.7% 忙**；每步 host
  气泡约 ~0.2%。我们 50 ms 的 GPU 步远超 ~0.12 ms 的 host 准备，所以 vLLM/SGLang 风格的「提前跑一步」在这里几乎买不到什么。
- **CUDA graphs**（S11）— 净负（−12% @conc=32）：不是 launch-bound。
- **KV cache 大小 / 驱逐**（S14）— 把池子扩到 *超过* 峰值需求（零抢占可能）
  使吞吐变动 **0**。不是瓶颈；边际 cliff 被完成错峰吸收。
- **prefill attention 占用率**（S8 攒下；S9 head-split 持平；S13 GQA-shared prefill 0.73×
  *更慢*——compute/occupancy-bound，付不起 4× O 累加器）。已榨干。
- **prefill chunk 大小 / gpu-mem-fraction** — 持平 / 不公平。已了结。

**残余差距（in=1024 ~7%）是 prefill 侧计算**——prefill GEMM 是 large-M 且在 cuBLAS 极限上 compute-bound，而我们手写的 WMMA prefill attention 虽好，但在那里略
落后于 vLLM 的 FlashAttention。它随输入变短趋向 0（decode 主导）。

**越过它的路线只有两条，都在 bf16 对齐表之外：**
1. **前缀缓存** — **已落地（S15，`feat/prefix-caching`）。** 自动 content-hashed KV 复用，
   默认开启。在纯随机输入上它只收回共享的 chat-template 前缀（conc=32 下 ~2–4%，与 vLLM 默认所得相当）；在有真实共享前缀的负载上它是一个大
   胜利（512-token 共享前缀下 **+36%**，与 vLLM 的 +37% 持平）。现在两个引擎都
   功能对齐包括前缀缓存，FlashQwen 保持同样的 92–97%-of-vLLM band。这里更进一步是为共享前缀上的 *attention* 做 cascade / shared-prefix attention（FlashInfer 分解），而不只是 KV 复用。
2. **量化**（fp8/int 权重或 fp8 KV）— 直接削减主导的 weight-traffic / prefill-
   GEMM 代价，但打破严格的 bf16 对齐。

净：在等精度 + 等特性下，FlashQwen 视输入长度是 **92.8–97.5% of vLLM**，相比起点 **5.5×**，剩余差距已定位、已测量、已解释。
