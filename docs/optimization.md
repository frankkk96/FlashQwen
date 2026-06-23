# FlashQwen optimization log — catching up to vLLM

> 中文版见 [optimization.zh.md](optimization.zh.md) · 每个实验的**详细记录**在 [`exps/`](exps/)（全中文，一实验一文档）

This is the **overview** of FlashQwen's throughput-optimization journey: one line per landed step, the
headline results, and the conclusions. Full per-experiment detail (what changed, why, every measurement,
the lessons) lives in [`docs/exps/`](exps/) — linked from each row below.

## Executive summary

FlashQwen is a from-scratch Qwen3-8B inference engine (Go front-end + C++/CUDA backend over gRPC). This
log tracks closing the serving-throughput gap to **vLLM** on a single RTX 4090, at **bf16 parity (no
quantization)**. Starting from `main` (INT8, a per-sequence scheduler) we rebuilt the scheduler, the GEMM
path, and the attention kernels.

**Where it ended up** (conc=32, `vllm bench serve` random, output 128, vs a feature-matched vLLM):

| input | FlashQwen | vLLM | **% of vLLM** |
|---|---|---|---|
| 128 (decode-heavy) | 1318 | 1393 | **94.6%** |
| 512 | 927 | 948 | **97.7%** |
| 1024 (prefill-heavy) | 640 | 652 | **98.1%** |
| ShareGPT (real data) | 1170 | 1159 | **100.9%** |

- **~6×** over the `main` baseline at conc=32/1024 (107 → 640 tok/s).
- Through S14 the gap **grew with input length** (residual = prefill-side compute). **S16** (the
  mma.sync prefill-attention rewrite) attacked exactly that, lifting 1024 from 92.8% → **98.1%** — the
  curve is now nearly flat and the last structural gap (long-context prefill) is largely closed.
- On real conversational data (ShareGPT) FlashQwen is **at parity** (100.9%), with far lower TTFT
  (3–22 ms vs vLLM's 355–1166 ms at saturation) and slightly higher steady-state TPOT.
- Decode GEMM, KV capacity/eviction, CUDA graphs, and CPU/scheduling overlap were each measured and
  **ruled out** as the conc=32 bottleneck (see Conclusions).

## What we track

`vllm bench serve --dataset-name random`, **1024 input / 128 output**, greedy (temp 0), thinking off,
concurrency **32** (matches vLLM `--max-num-seqs 32`); conc=1 reported for the single-stream view.
Honest metric: **output tok/s**. All comparisons are vs a **feature-matched** vLLM
(`--no-enable-prefix-caching`, bf16, 0.9 mem) unless stated.

## The journey (one line per step)

Canonical numbers from the 2026-06-20 same-machine replay; standard test = 1024/128, `% vLLM` vs the
no-cache vLLM (conc=32 = 652). Reverted attempts (S4, S11) are kept as records — negative results matter.

| Step | change | conc=32 @1024 | % vLLM | detail |
|---|---|---|---|---|
| B0/R1/R2 | baseline = main (INT8, per-seq sched) → unified token-budget scheduler + GPU sampling | 107 → 86 | 16→13% | [00](exps/00-baseline-and-scheduler.md) |
| S3 | bf16 weights + cuBLAS GEMM + FlashAttention-2 — *the engine* | **348** | 53% | [s03](exps/s03-bf16-cublas-fa2.md) |
| S4 | tensor-core + GQA-grouped attention — *tried, reverted (−34%)* | — | — | [s04](exps/s04-tensorcore-gqa-reverted.md) |
| S5 | kernel fusion (fused QKV/gate-up GEMM, add+rmsnorm, RoPE table) | 356 | 55% | [s05](exps/s05-kernel-fusion.md) |
| S6 | FlashDecoding decode-attention kernel (split by request type) | **454** | 70% | [s06](exps/s06-flashdecoding.md) |
| S7 | WMMA tensor-core prefill attention | **501** | 77% | [s07](exps/s07-wmma-prefill.md) |
| S8 | prefill-attn occupancy (shrink shared memory) | **531** | 81% | [s08](exps/s08-prefill-occupancy.md) |
| S10 | scheduler: default max-batch-tokens 2048 → 1024 (the KV cliff) | **581** | 89% | [s10](exps/s10-max-batch-tokens.md) |
| S11 | CUDA graphs for pure-decode steps — *tried, reverted (−12%)* | — | — | [s11](exps/s11-cuda-graphs-reverted.md) |
| S12 | GQA-shared FlashDecoding (read K/V once per group + KV-split) | **604** | 93% | [s12](exps/s12-gqa-flashdecoding.md) |
| S14 | activation scratch → real per-step bound (+ latent WMMA OOB fix) | 605 | 93% | [s14](exps/s14-activation-scratch.md) |
| S15 | automatic prefix caching (content-hashed KV reuse) | 621* | — | [s15](exps/s15-prefix-caching.md) |
| **S16** | **prefill attention rewrite: WMMA → mma.sync (FlashAttention-style)** | **640** | **98.1%** | [s16](exps/s16-prefill-mma-rewrite.md) |
| S17 | comprehensive load/data comparison + prefix-cache correctness fix | — | — | [s17](exps/s17-comprehensive-comparison.md) |
| — | **vLLM (no prefix cache), reference** | **652** | 100% | — |

*S15's effect shows only on shared-prefix / cache-on workloads (+36% with a 512-token shared prefix); on
pure-random input it is ≈ S14.

## Results across input length (final)

conc=32 output tok/s (% of feature-matched vLLM), every landed step rebuilt clean and benched at
128/512/1024 input:

| step | in=128 | in=512 | in=1024 | what it bought |
|---|---|---|---|---|
| **S3**  | 1094 (79.5%) | 628 (66.5%) | 348 (53.4%) | bf16 + cuBLAS + FA2 — ~4× everywhere |
| **S6**  | 1317 (95.7%) | 824 (87.3%) | 454 (69.6%) | FlashDecoding — biggest lift at short input |
| S7/S8 | ~1328 (96.5%) | ~878 (93%) | 531 (81.5%) | prefill attn — lifts long input |
| S10 | 1334 (97.0%) | 882 (93.4%) | 581 (89.1%) | KV-cliff scheduler fix — only @1024 |
| S12/S14 | 1341 (97.5%) | 908 (96.2%) | 605 (92.8%) | GQA-shared decode + pool fix |
| **S16** | 1318 (94.6%) | **927 (97.7%)** | **640 (98.1%)** | mma.sync prefill — closes the long-input gap |
| vLLM | 1376/1393 | 944/948 | 652 | feature-matched reference |

Two clean facts: (1) each optimization acts on the regime it targets and they don't overlap (S6 lifts
in=128; S7/S8 lift in=1024; S10 moves *only* in=1024). (2) Through S14 the gap was monotone in input
length (97.5% → 92.8%) — residual at long input = prefill-side compute. S16 attacked exactly that, so the
curve is now nearly flat. Detail: [s16](exps/s16-prefill-mma-rewrite.md).

## Comprehensive comparison vs vLLM (S17, bf16 parity)

Wide head-to-head (FQ concurrency ceiling = 32, `MAX_DECODE_B`). Full tables + method in
[s17](exps/s17-comprehensive-comparison.md).

- **Input length** (out=128, c=32; 2048@c16): 97.1% / 98.3% / 97.8% / 91.9% (128/512/1024/2048).
- **Concurrency** (in=512): 96.4% / 95.0% / 96.6% / 98.3% (c=1/8/16/32).
- **Output length** (in=512, c=32): out=128 98.3%, out=512 96.1%, **out=1024 107.4%** (FQ faster).
- **ShareGPT real data** (c=32): **100.9%** (1170 vs 1159); FQ TTFT ~22× lower, vLLM TPOT 13% lower.

S17 also found & fixed a prefix-cache correctness bug (the S16 cp.async double-buffer corrupted large
prefix hits → empty output; fixed by single-buffer staging, commit `aaf4e0d`). A suspected high-concurrency
hang turned out to be a benchmark-harness GPU-teardown artifact, not an engine bug. Detail:
[s17](exps/s17-comprehensive-comparison.md).

## Conclusions — what's exhausted, where the residual lives

At bf16 parity on a 4090, the conc=32 bottleneck was chased to ground. Levers **ruled out with data**:
decode GEMM (~84% of HBM peak — physical floor, only quantization beats it), CPU/scheduling overlap (GPU
98.7% busy, host bubble ~0.2%), CUDA graphs (−12%, S11), KV size/eviction (growing the pool past peak
demand moved throughput by 0, S14), prefill-attn occupancy (S8 banked it; GQA-shared prefill was 0.73×
*slower*), prefill chunk size / gpu-mem-fraction (flat).

After S16 the long-input residual is largely closed (1024 = 98.1% no-cache / 96.2% cache-on, was 92.8%).
What remains is small and localized; the two routes past it are both **off the bf16-parity table**:

1. **Prefix caching** (landed, S15) — content-hashed KV reuse, on by default. ≈neutral on random input,
   a large win on real shared-prefix workloads (+36% @512-shared, on par with vLLM's +37%).
2. **Quantization** (fp8/int weights or fp8 KV) — cuts the dominant weight-traffic / prefill-GEMM cost,
   but breaks strict bf16 parity.

**Net:** at equal precision + equal features, FlashQwen is **~95–98% of vLLM** across input lengths and
**at parity (~100%) on real conversational data**, **~6×** over where it started — with the remaining gap
localized, measured, and explained.
