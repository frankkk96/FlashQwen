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
| S3 | bf16 weights + cuBLAS GEMM + FlashAttention-2 | 38.2 | **350.0** | **50.1%** | 7650654 |
| S5 | kernel fusion: fused QKV/gate-up GEMM, add+rmsnorm, RoPE table | 39.1 | 356.1 | 51.0% | c2f98a0 |

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

### S3 — bf16 + cuBLAS GEMM + FlashAttention-2 (7650654)
- **Change**: three coordinated kernel rewrites, addressing the debt R1/R2 identified.
  1. **Drop INT8 weight-quant → bf16 everywhere.** Weights load as bf16 (no dequant), and the whole
     activation pipeline is bf16 (norm/rope/residual/silu accumulate in fp32 internally). This is now
     **fair bf16-parity with vLLM** — the earlier precision caveat is gone.
  2. **cuBLAS for all matmuls.** One `cublasGemmEx` path (bf16 in, fp32 accumulate, tensor cores)
     replaces the hand-rolled WMMA prefill GEMM + per-call full-weight dequant + the batched INT8
     decode GEMV. The decode/prefill matmul split is deleted — one path for the merged batch.
  3. **FlashAttention-2 paged kernel.** Replaces the per-(head,row) varlen kernel that re-read all
     K/V from global with zero reuse (O(T²)). New kernel: one block per (q-tile, head, request); BM=16
     query rows (one warp each) share BLOCK-sized K/V tiles staged in shared memory (reuse factor BM),
     online softmax with deferred normalization (divide by the running sum only at the end), causal
     whole-tile skip. Host groups the batch by request (qstart/qlen) for the q-tile grid.
- **Result vs R2**: conc=32 **87.4 → 350.0 tok/s (4.0×)** — **12.5% → 50.1% of vLLM**, and 3.3× the
  `main` baseline (106.9). conc=1 24.9 → 38.2 (still below main's INT8 56.7: bf16 weights read 2× the
  bytes of INT8, and single-stream decode is weight-bandwidth-bound — the expected de-quant cost; the
  tracked metric is conc=32). Greedy generation verified coherent.
- **Lesson**: R1/R2 were right — the gap was the kernels, not scheduling. The unified-scheduler
  regression is not just recovered but blown past. The remaining ~2× to vLLM at conc=32 is NOT attention
  (see S4 below) — it's the GEMMs (cuBLAS, near-optimal) + per-step launch/scheduling overhead (~720
  kernel launches/step, no CUDA graphs, no persistent batch, no prefix caching). Single-stream (conc=1)
  is a separate, bandwidth-bound concern (bf16 weight traffic) if we ever care about it.

### S4 — tensor-core + GQA-grouped attention (TRIED, REVERTED)
- **Change**: rewrote the attention kernel to use WMMA (16×16×16 tensor cores) for S=Q·Kᵀ and O=P·V,
  and grouped one block per (q-tile, **KV head**, request) so all 4 q-heads of a GQA group reuse one
  staged K/V tile (4× less K/V traffic). O accumulator kept in shared (fp32), rescaled by a thread
  loop, P·V added via load/store_matrix_sync — no WMMA fragment-layout assumptions. Greedy output was
  bit-identical to S3 (correct).
- **Result vs S3**: conc=32 350.0 → 355.5 (flat, noise); conc=1 **38.2 → 25.1 (−34%)**. Net negative.
- **Why it didn't pay off**: (1) after S3's BM=16 K/V tiling, **attention is no longer the conc=32
  bottleneck** — the GEMMs/launch overhead dominate, so a 2–3× faster attention barely moves total
  throughput. (2) At conc=1, decode rows have q_len=1 but WMMA forces a full 16-row tile (16× wasted
  compute), and the per-KV-head grouping leaves only 8 grid blocks (vs 32) → the GPU is starved and
  attention latency grows, dragging single-stream TPOT (26→40 ms).
- **Decision**: reverted to S3 (commit 7650654). Kept here as the record that the attention lever is
  spent — the next real levers are launch-overhead (CUDA graphs / kernel fusion) and the throughput
  features vLLM has (persistent batch, prefix caching).

### S5 — kernel fusion: fused QKV/gate-up GEMM, add+rmsnorm, RoPE table (landed)
- **Target**: the per-step launch + redundant-traffic overhead S3 named as the remaining conc=32 lever
  (~720 kernel launches/step). Three fusions, all bf16-identical math:
  1. **Fused QKV and gate|up GEMMs.** q/k/v proj weights are concatenated on the OUT dim into one
     `wqkv` ([q_dim+2·kv_dim, H]) at load, likewise gate+up into `wgateup` ([2·I, H]); one
     `cublasGemmEx` each replaces 3 and 2. Downstream kernels read the interleaved fused buffers via
     offset/stride: a new fused **per-head RMSNorm+RoPE** kernel normalizes+rotates the q and k slices
     in place, `store_kv` reads k/v straight from the QKV buffer, attention takes a `q_stride`, and
     `silu_mul` reads gate/up as the two halves of one row.
  2. **Fused residual-add + RMSNorm.** `add_rmsnorm` does `x += residual` (written back, carrying the
     residual forward) then `rmsnorm(x)` in one pass — replaces a separate `add` + `rmsnorm`, saving a
     full H read/write of x and a launch per residual. run_layers restructured so every norm but the
     first consumes the pending residual; one trailing `add` commits the last layer's MLP residual.
  3. **Precomputed RoPE cos/sin table.** Angles depend only on (pos, i) and are identical across all
     36 layers; the old kernel recomputed `powf/cosf/sinf` per element per layer (36× waste). Now a
     `[max_ctx, head_dim/2]` table is built once at startup and looked up (folded into the fused
     norm+rope kernel).
  - Net: ~19 → 12 kernel launches per layer (~252 fewer/step, ~720 → ~470).
- **Result vs S3**: conc=1 38.2 → **39.1 (+2.3%)**; conc=32 350.0 → **356.1 (+1.7%, within noise)**.
  Greedy output verified coherent.
- **Lesson**: removing ~250 launches/step + redundant elementwise traffic barely moved conc=32 — so at
  conc=32 the launch/elementwise overhead is a **small fraction**; the skinny decode **GEMMs dominate**
  (each step streams all 17 GB of bf16 weights from HBM). This recalibrates the roadmap: the
  "~720 launches" figure overstated the lever. **CUDA graphs will help conc=1 (latency-bound) more than
  conc=32**; closing the remaining 2× at conc=32 needs GEMM-side wins (cuBLASLt autotuning / better
  decode-GEMM shapes) or the throughput features vLLM has (larger effective batch, prefix caching), not
  just launch elision. The fusions are kept regardless — correct, cleaner, and they pre-stage the
  persistent-buffer layout CUDA-graph capture will want.

#### S5 ablation — which of the three fusions actually paid (2026-06-19)
Each change isolated on the S3 base (1024/128, temp 0); only the engine differs.

| variant | conc=1 | conc=32 | vs S3 (conc=32) |
|---|---|---|---|
| S3 base | 38.2 | 350.0 | — |
| A: #5 RoPE cos/sin table only | 38.2 | 350.2 | ~0 (noise) |
| B: #7 add+rmsnorm fusion only | 38.3 | 350.0 | ~0 (noise) |
| C: #6 fused QKV/gate-up GEMM only | 38.9 | 355.8 | **+1.7%** |
| S5: all three | 39.1 | 356.1 | +1.7% |

**Finding: none is a regression, but #5 and #7 are net-zero — the whole S5 gain is #6 (fused GEMMs).**
Why #5/#7 are placebo here: RoPE and the add/rmsnorm elementwise kernels are tiny next to the GEMMs,
and at conc=32 the step is GEMM/HBM-bound, so eliminating their launches + redundant traffic doesn't
move the wall clock. #6 helps because folding 3→1 and 2→1 GEMMs gives cuBLAS a wider N (better tensor-
core utilization on the skinny M=32 decode shape) on top of fewer launches. Adding #5+#7 on top of #6
(C→S5) changes nothing (355.8→356.1). Kept all three anyway: correct, not negative, fewer launches for
the eventual CUDA-graph capture — but the lesson is **GEMM-shape wins are the lever, elementwise/launch
fusion is not** at this batch size.

<!-- Append one ### entry per landed step: What / Why / Change / Result (vs prev) / Lesson -->
