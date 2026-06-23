# S17 — 与 vLLM 全负载/全数据综合对比 + prefix-cache 正确性修复

> branch `feat/prefix-caching`,2026-06-23。在严格 bf16 parity 下对最终引擎做了一次与 vLLM 的宽口径正面对比(输入长度 / 并发 / 输出长度 / ShareGPT 真实数据),并在这张矩阵中发现并修复了一个 prefix-cache 正确性缺陷;另外澄清了一个被误判为 "conc=32 hang" 的 bench-harness artifact。

## 背景/动机

S16 之后,引擎在各个区间已经接近 vLLM。需要一张宽口径的 head-to-head 矩阵来:(1) 确认 parity 的全貌而不只是 tracked 的 1024/128 这一个点;(2) 用真实数据(ShareGPT)而不只是合成 random 来检验;(3) 在这个过程里压力测试新落地的 prefix cache。

对比口径(严格 bf16 parity):同 tokenizer、max-model-len 4096、max-num-seqs 32、gpu-mem-util 0.9、seed 1234、temp 0、warmup 丢弃。FlashQwen 的并发上限是 **32**(`MAX_DECODE_B`,一个 decode-kernel 的 compile-time 常量),所以并发 sweep 到此为止。所有 prefix-cache 命中会很大的负载,都在 **两边都 cache-off** 的前提下跑(原因见下面的 bug),以保持 apples-to-apples。所有数字都是 `vllm bench serve` 的 Output token throughput(tok/s)。

## 对比方法

四组对比:输入长度 sweep、并发 sweep、输出长度 sweep、ShareGPT 真实数据。Bench 脚本在 `/root/prof`(`full_matrix.sh`、`sharegpt_fixed.sh`);raw 在 `/root/prof/full/`。

## 实测结果

**Throughput vs input length**(out=128,conc=32;in=2048 在 conc=16 跑 —— 32×2176 token 会超过 39k-token 的 KV pool):

| input | FlashQwen | vLLM | **% of vLLM** |
|---|---|---|---|
| 128  | 1345.7 | 1386.0 | **97.1%** |
| 512  | 944.7  | 961.3  | **98.3%** |
| 1024 | 638.3  | 652.6  | **97.8%** |
| 2048 (c16) | 304.9 | 331.8 | **91.9%** ← weakest (long-context prefill) |

**Concurrency sweep**(in=512,out=128)和 **output length**(in=512,conc=32):

| conc | FQ | vLLM | % |  | output | FQ | vLLM | % |
|---|---|---|---|---|---|---|---|---|
| 1  | 54.9  | 56.9  | 96.4% |  | 128  | 944.7  | 961.3  | 98.3% |
| 8  | 358.9 | 377.8 | 95.0% |  | 512  | 1200.8 | 1250.0 | 96.1% |
| 16 | 604.7 | 626.0 | 96.6% |  | 1024 | 1132.9 | 1055.0 | **107.4%** (FQ faster) |
| 32 | 944.7 | 961.3 | 98.3% |  | | | | |

**Real data — ShareGPT**(500 prompts,conc=32,两边 cache-off,固定 out=128):

| | FlashQwen | vLLM | **% of vLLM** |
|---|---|---|---|
| Output tok/s | **1170.2** | 1159.1 | **100.9%** |
| Mean TTFT | **5.4 ms** | 118.5 ms | FQ ~22× lower |
| Mean TPOT | 27.4 ms | 23.9 ms | vLLM 13% lower |

### 三点观察

1. **TTFT 与 TPOT 是互补的:** 在饱和状态下 FlashQwen 的 TTFT 极小(3–22 ms),而 vLLM 是 355–1166 ms(vLLM 把排队的 chunked-prefill 前置堆积),但 vLLM 的 steady-state TPOT 略低。
2. **长输出(out=1024)让 FQ 反超 7%** —— decode-heavy,且 FQ 的低 TTFT 在长 generation 上累积放大,盖过了那点小小的 TPOT 劣势。
3. **in=2048 是最弱点(91.9%)** —— 即便有了 S16 的 mma 重写,long-context prefill 仍是 FQ 相对的软肋。

### Prefix-cache 有效性

在强制共享前缀的负载(1024 shared + 128 unique,conc=32)上,vLLM 的 cache 把 throughput 从 **308 → 513 tok/s(+66%)**,并让该负载装进了 KV pool。在 ShareGPT(跨请求重叠很低)上,缓存在两边都 ~中性(FQ cache-on 1168 ≈ cache-off 1170 ≈ vLLM 的 100.9%)。

## 发现的 bug 与修复

### BUG#1 —— 已修复(commit `aaf4e0d`)

这张矩阵暴露出一个正确性缺陷:`FQ_PREFIX_CACHE=1` 时,prefix-cache **命中很大** 的请求返回空 completion(或偶发 502)。

**症状:** `FQ_PREFIX_CACHE=1` 下,prompt 有 LARGE prefix-cache 命中的请求返回空 completion —— vllm-bench 里显示 HTTP 200、`Total generated tokens: 0`、`Failed requests: 0`,Go 前端偶发 `502`。cache-OFF 路径正常。

**Repro / 边界(vllm bench serve,openai-chat endpoint):**
- 同一请求发两次,24-token prompt → 第 2 个命中 1 block(16 tok)→ **正确**(与 miss bit-identical)。小命中没问题。
- b2 random in=512:命中 8 blocks / 128 tok → **正确**(所以 S-curve B1/B2/B3 的 cache-on 数字是有效的;random data + prefix-len 0 本来就几乎不命中)。
- b4 random `--random-prefix-len 1024`(~71-block 共享前缀命中)→ **0 output**。
- b5 ShareGPT(真实多轮对话)cache-ON → 在全部 200–500 个 prompt 上 **0 output**。

即:小命中 OK;大的多 block 复用 / 真实对话负载 → 空 / 报错。

**根因:** S16 prefill kernel 里的 **cp.async double-buffered prefetch** 有一个时序 hazard,只在引擎背靠背的多层 kernel launch 上下文里、且只在 cache-hit / chunked-tail 的 prefill 形状(query row 很少 + K/V tile 很多 —— 一个 fresh prefill 永远不会产生的形状)下才暴露。

**定位过程(钉死它的诊断):**
- (a) `FQ_PREFILL_V2=0`(WMMA)→ 干净;
- (b) 在 kernel 之间加一个 stream sync(`FQ_SYNC_DEBUG`)→ 干净;
- (c) 把单个 kernel 孤立出来用 `compute-sanitizer --tool racecheck` 跑 → CLEAN。

所以这是一个 **inter-kernel 的 async hazard,不是 intra-kernel 的 smem race**。

**修复:single-buffer 同步 cp.async staging**(`stage(kt); __pipeline_wait_prior(0); __syncthreads();`)。保留了 register-O + mma.sync + block-shared K/V reuse;cp.async overlap 只值约 0.9% e2e,所以这个代价可忽略。

**验证:** 连续相同请求,命中多达 88 blocks 仍正确;1024/conc32 cache-off 性能不变(639)。Repro `/root/prof/repro.sh`(同一长 coherent prompt 发两次 → 第 2 次命中);standalone racecheck harness `/root/prof/race/race_test.cu`。修复后,cache-on 在真实 ShareGPT 上 conc=32 正确且稳定(500/500,0 failure)。

### "false alarm" —— 被怀疑的 conc=32 hang 实为测试 harness 的 GPU teardown OOM artifact

修复 BUG#1 后,曾一度怀疑还有一个 **BUG#2**:cache-on 在真实 ShareGPT 高并发下仍失败 —— conc=1 和 conc=4 正确,conc=32 → 所有请求失败(引擎 HANG:kvstat 在中途停住,无 CUDA error、无 502、client timeout)。当时排除了容量问题(peak pool 16.8%、preempt=0、prefix_hit ~70%);同样数据 cache-OFF 在 conc=32 正常。

最终查明:**这个 "conc=32 cache-on hang" 是 benchmark-harness 的 artifact,不是引擎 bug** —— 快速反复重启 server 会让下一个 engine 在启动时 OOM(上一个进程的 GPU teardown 还没释放完显存)。在一个健康的 server 上重测,conc=32 cache-on 在真实 ShareGPT 上 **4/4 干净通过**。

(这也说明:BUG#1 修复后 prefill kernel 本身已经正确 —— conc=4 用相同 shape 是工作的;之前 conc=32 的失败是被启动 OOM 这层 artifact 掩盖的。)

## 经验

- **宽口径才看得清全貌。** 单一 tracked 指标(1024/128)无法暴露 in=2048 的软肋、out=1024 的反超、以及 prefix-cache 的大命中 bug —— 这些都是 full matrix 才逼出来的。
- **TTFT 与 TPOT 不可只看一个。** FQ 在饱和下 TTFT 远低(3–22 ms vs vLLM 355–1166 ms),vLLM steady-state TPOT 略低;两者此消彼长,长输出场景 FQ 因低 TTFT 累积而反超。
- **小命中通过 ≠ 正确。** 早期乐观的 "+36% shared-prefix" 结论只测了 small/synthetic 命中,从未走到 large-hit 路径;在打开真实数据 / 大命中前,prefix-cache 对比必须 **两边都 cache-OFF** 才有效。
- **bug 与 harness artifact 要分清。** 一个看起来像引擎 hang 的现象,实际是测试脚本快速重启导致下一个 engine 启动 OOM;不在健康 server 上复现就别归因到引擎。诊断的层次法(WMMA fallback → stream sync → racecheck)能把 inter-kernel async hazard 和 intra-kernel race 干净地区分开。
- bf16 parity、且 cache-off(大命中处)的有效全矩阵结论:FlashQwen 在 input-len {128,512,1024}@c32 与并发 sweep {1,8,16,32} 上是 vLLM 的 **96–98%**;out=1024 **107%**(长 decode,FQ 极低 TTFT 取胜);in=2048 **91.9%**(long-context prefill,FQ 最弱);ShareGPT 真实数据 **~100%**(1170 vs 1159 tok/s)。
