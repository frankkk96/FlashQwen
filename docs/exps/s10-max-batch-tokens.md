# S10 — 调度器：默认 max-batch-tokens 2048 → 1024

## 背景/动机

对调度器旋钮（scheduler knobs）做 sweep（`sched_sweep.sh`）时，浮现出 `max-batch-tokens`（即 `max_num_batched_tokens`）是一个之前 max-prefill-tokens sweep 漏掉的大 lever。

**找到的根因（root cause）：** 在 conc=32 / 1024-in 下，KV pool（约 36.7k tokens）正好卡在需求线上（32×1152 = 36864）。默认的 mbt=2048 让一个 step 能 admit（准入）约 4 个并发 prefill chunk（4×512），把峰值 KV 需求顶过了 pool → recompute-preemption thrash（重算抢占抖动）→ 527 tok/s。mbt=1024 把 prefill 串行化到足以保持在 pool 之下 → 589（+11%），并带来更好的 TTFT/TPOT，而且这是在与 vLLM 默认 **相同的 0.9 VRAM** 下达成（公平）。

**用 `--gpu-mem-fraction` sweep 验证了机制**（已贯穿 Go 打通）：把 pool 扩大到 43.5k tokens 可以修复 mbt=2048（527 → 588），但 **并不能** 超过 mbt=1024 的 589。所以约 588 就是 no-preemption（无抢占）的天花板——两条路径（限制并发，或扩大 pool）到达同一个地方，而剩余到 vLLM（698）的差距是 **compute/framework，而非 KV/preemption**。

## 改动

把调度器默认的 max-batch-tokens 从 2048 改为 1024。gpu-mem-fraction 默认仍保持 0.9；该 flag 现已暴露出来，供拥有其他 VRAM/模型的用户使用。

## 实测结果

相对 S8：
- conc=32 528 → **589（+11%）**；
- conc=1 不变（约 45，single-stream 与 mbt 无关）。
- **76% → 84% of vLLM**，达到 `main` 基线的 5.5×。

commit: 642cc28。

在跨输入长度的 journey replay 中（conc=32 output tok/s，括号内为相对 feature-matched vLLM 的百分比）：

| step | in=128 | in=512 | in=1024 | 这一步带来了什么 |
|---|---|---|---|---|
| S10 | 1334 (97.0%) | 882 (93.4%) | 581 (89.1%) | max-batch-tokens=1024——**仅 @1024**（+7.6pt，the cliff） |

Progress 表中的对应行：

| Step | change | conc=1 | conc=32 | % vLLM | commit |
|---|---|---|---|---|---|
| S10 | scheduler: default max-batch-tokens 2048→1024 | 45.0 | **581** | **89%** | 642cc28 |

### inlen × mbt sweep —— throughput 由 KV 容量 vs 需求主导（2026-06-20）

`mbt_inlen.sh`，conc=32，按 (input length, max-batch-tokens) 的 otps：

| inlen | KV needed (32×(in+128)) | best otps | mbt effect |
|---|---|---|---|
| 512  | 20480 (57% of pool) | ~890 | flat (mbt irrelevant — KV fits) |
| 1024 | 36864 (≈pool, the cliff) | ~589 | small mbt wins (eases preemption) |
| 2048 | 69632 (190% of pool) | 78 → 7 | collapses; bigger mbt = worse (thrash) |

追踪的 1024/128 指标恰好坐落在 KV-capacity cliff（KV 容量悬崖）上——这正是这里一切都如此敏感的原因。throughput 首先由 KV-pool 容量 vs 需求决定，其次才是 kernel。

## 经验

与 input-length 强耦合（这是用户场景的判断）——mbt 的甜点（sweet spot）之所以存在，是因为标准测试正好坐在 KV cliff 上；在 512-in 时（KV 装得下）mbt 无关紧要（约 890 tok/s），在 2048-in 时（2× 超出）它崩溃（collapse）。

公平、免费的赢法是通过 mbt 限制 admission（准入）；想超过 588 就意味着要拼 compute，而不是 memory。
