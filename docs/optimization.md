# FlashQwen optimization log — catching up to vLLM

The single main-line record of the throughput-optimization journey. One curated entry per landed
step: **what changed, the bottleneck it targeted, the measured effect, the lesson.** Throwaway
experiment notes (tuning sweeps, dead ends, profiler dumps) live under `docs/exps/` so this file
stays readable.

## Goal & constraints

- Close the serving-throughput gap to **vLLM** on **Qwen3-8B**, **bf16, no quantization** (fair, equal
  precision). RTX 4090 (24 GB, sm_89, ~1 TB/s). 36 layers, hidden 4096, 32 q-head / 8 kv-head GQA,
  head_dim 128, intermediate 12288, QK-RMSNorm, SwiGLU, no bias.

## Standard test (the one number we track)

- Workload: `vllm bench serve --dataset-name random`, **1024 input / 128 output**, greedy (temp 0),
  thinking disabled. Concurrency **32** (matches vLLM's `--max-num-seqs 32`); conc=1 reported too for
  the single-stream / per-step view.
- Harness: `bash /root/bench-compare/std.sh <label>` → appends to `results/std.csv` (starts the server,
  runs conc=1 and conc=32, prints output tok/s). Build first with `make`.
- Honest metric: **output tok/s**. (TTFT is an artifact — FlashQwen emits the SSE role chunk before
  prefill, so prefill time lands in TPOT.)
- Correctness gates each step: `bash /root/bench-compare/validate.sh` (numerical) + `gencheck.sh` (text).

## Baselines (this exact standard: 1024/128)

| | conc=1 | **conc=32** | % of vLLM |
|---|---|---|---|
| **P0 (start, this branch's main)** | 53 | **92** | 14% |
| **vLLM (reference)** | 56 | **678** | 100% |

P0 = hand-written INT8 GEMM + FP32 activations, single prefill chunk that blocks decode, no KV-aware
admission. It **collapses under concurrency** (conc=32 tpot ~349 ms): that's the first thing to fix.

## Progress

| Step | change | conc=1 | conc=32 | % vLLM | commit |
|---|---|---|---|---|---|
| P0 | baseline | 53 | 92 | 14% | (main) |
| | | | | | |

## Roadmap (intended order — fill results as landed)

Each step: diagnose first (profile / reason about the bottleneck), change, measure with the standard
test, gate on validate+gencheck, then record an entry below.

- [ ] **Chunked prefill** — stop a long prefill from blocking the whole decode batch (the P0 collapse).
- [ ] **cuBLAS bf16 GEMM** — replace hand-written matmul (drop INT8; fair bf16).
- [ ] **KV-aware admission** — reserve each sequence's lifetime KV up front; stop over-admission thrash.
- [ ] **bf16 activations** — free VRAM for a bigger KV pool / more concurrency.
- [ ] **Fuse QKV + gate/up GEMMs** — fewer launches (expect ~neutral at saturation; confirm).
- [ ] **FlashAttention-2 prefill** — tensor-core paged attention via raw `mma.sync` (O in registers).
- [ ] **Multi-sequence prefill** — let several sequences prefill in one step; cut prefill drag on decode.
- [ ] **Decode-attention occupancy** — multi-warp blocks (1-warp blocks cap occupancy ~37%).
- [ ] **Fuse QKV post-process** — split + q/k-norm + RoPE + KV-store in one kernel (collapse serial chain).

> Reference (first pass, S2 = 1024/**256** workload, so absolute numbers differ): branch
> `perf/vllm-catchup`, see `git show perf/vllm-catchup:docs/benchmarks/优化历程.md`. Relative effects
> and lessons carry over; re-measure everything on this branch's 1024/128 standard.

---

## Step entries

### P0 — baseline
- **State**: hand-written INT8 GEMM, FP32 activations, single 256-token prefill chunk (blocks decode),
  no KV-aware admission.
- **Standard test**: conc=1 = 53 tok/s, conc=32 = 92 tok/s (collapses, tpot ~349 ms) = 14% of vLLM.
- **Bottlenecks to attack**: (1) prefill blocks decode → concurrency collapse; (2) hand-written GEMM far
  below cuBLAS; (3) no KV-aware admission.

<!-- Append one ### entry per landed step: What / Why / Change / Result (vs prev) / Lesson -->
