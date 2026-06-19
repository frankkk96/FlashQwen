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

## Baselines (this exact standard: 1024/128, temp 0, ignore_eos)

Aligned `vllm bench serve --dataset-name random` — identical config on both servers, only the engine
differs. (main has no `/v1/completions`+`ignore_eos`, so it was measured on a throwaway `main` worktree
with that **Go-only** shim cherry-picked from `650fdaa` — no engine/compute change.)

| | conc=1 | **conc=32** | % of vLLM (conc=32) |
|---|---|---|---|
| **main (B0)** | 56.7 (TPOT 11.5 ms) | **106.9** (TPOT 251 ms) | **15.3%** |
| **vLLM (reference)** | 55.7 (TPOT 17.1 ms) | **698.2** (TPOT 37.2 ms) | 100% |

main is at parity with vLLM **single-stream** (even ~2% ahead — INT8 weights are memory-lighter than
vLLM's bf16, and decode is memory-bound), but **collapses under concurrency**: 15% at conc=32, where
vLLM's batched GEMM dominates. The number to catch up = **conc=32**.
(Precision caveat: FlashQwen is INT8 weight-quant vs vLLM bf16 — not equal precision yet; this flatters
FlashQwen's single-stream decode.)

## Progress

| Step | change | conc=1 | conc=32 | % vLLM | commit |
|---|---|---|---|---|---|
| B0 | baseline = main (per-seq scheduler, old attn kernel) | 56.7 | 106.9 | 15.3% | 08392df |
| R1 | unified token-budget scheduler + GPU sampling | 25.0 | 87.2 | 12.5% | 5065b2e |
| R2 | scheduler/kernel refactor (perf-neutral, Δ0) | 24.9 | 87.4 | 12.5% | 291cb74 |

## Step entries

### B0 — baseline = main (08392df)
- **State**: per-sequence scheduler (single prefill chunk), INT8 weight-quant matmul, the original
  paged-attention kernel. Measured on a `main` worktree with a Go-only `/v1/completions`+`ignore_eos`
  shim cherry-picked (`650fdaa`) so the identical bench could run — no compute change.
- **Aligned test**: conc=1 56.7 tok/s (TPOT 11.5 ms) = 102% of vLLM; conc=32 106.9 (TPOT 251 ms) = 15.3%.
- **Read**: single-stream FlashQwen already beats vLLM (INT8 = less weight traffic, decode is
  memory-bound). The whole gap is at concurrency — vLLM batches far better.

### R1 — unified token-budget scheduler + GPU sampling (REGRESSION)
- **Change**: rewrote scheduling to a vLLM-style unified token budget (chunked prefill + decode in one
  merged forward), three-layer paged-KV stack, GPU sampling; replaced attention with a naive paged
  varlen kernel (one warp per (head, query-row), online softmax, no split-K).
- **Result vs B0**: **slower** — conc=1 56.7→25.0 (−56%, TPOT 11.5→33.2 ms), conc=32 106.9→87.2 (−18%).
- **Lesson**: the rewrite bought correctness + robustness under concurrency + chunked prefill, but the
  no-split-K varlen attention is far slower than the old kernel, so net throughput regressed. The
  scheduler is fine; **the attention kernel is the debt.**

### R2 — scheduler/kernel refactor (perf-neutral)
- **Change**: structural only — slimmed `step()`, merged `grow`/`retire`, pushed batch logic into
  `Request`, bundled knobs into `SchedulerConfig`/`RuntimeConfig`, removed dead kernel code
  (`gemv_kernel`; `launch_matmul`→`launch_matmul_prefill`). No hot-path or numerics change.
- **Result vs R1**: unchanged — conc=1 24.9, conc=32 87.4 (Δ0; two independent measurements,
  5065b2e ↔ HEAD).
- **Lesson**: confirms the gap is the kernels, not scheduling/bookkeeping. Clears the deck for the
  attention rewrite (split-K).

<!-- Append one ### entry per landed step: What / Why / Change / Result (vs prev) / Lesson -->
