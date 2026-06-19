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
| S6 | FlashDecoding decode-attention kernel (split by request type) | 44.7 | **455.2** | **65.2%** | f0b1499 |
| S7 | WMMA tensor-core prefill attention | 44.9 | **493.4** | **70.7%** | 0dd4010 |
| S8 | attn_prefill occupancy (shrink shared, drop PVt) | 45.2 | **528.5** | **75.7%** | 392cda5 |
| S10 | scheduler: default max-batch-tokens 2048→1024 | ~45 | **589** | **84.4%** | 642cc28 |

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

### S6 — FlashDecoding decode-attention kernel (f0b1499)
- **Target**: the conc=32 ceiling the S5 profiling left open. Profiles used SHORT context; the tracked
  metric is 1024-ctx, where decode attention reads a ~1152-tok KV every step. The unified attention
  kernel handles q_len==1 (decode) terribly: only 1 of its 16 warps is active (~6% occupancy) and that
  warp serially scans all ~72 KV tiles. GEMM was already at 84% HBM BW (S5 profile), so attention was
  the remaining lever.
- **Change**: split attention by request type *within* the unified batch — the GEMMs/norms stay merged
  (attention is per-request anyway, so this is orthogonal to unified batching, NOT a return to
  prefill/decode-split forwards). Decode rows (q_len==1) → new `decode_attn_kernel`; prefill rows
  (q_len>1) → the existing tiled flash kernel. Both take a `rids[]` grid→request indirection so each
  runs on its request subset, writing disjoint rows of `attn_`. In steady conc=32 most steps are pure
  decode, so the decode kernel carries the common case.
  - `decode_attn_kernel`: one block per (head, decode-request); NW=8 warps split that request's KV
    [0,qpos] into strided slices; each warp online-softmaxes its slice in registers reading K/V
    **straight from the paged cache** (one query row has no cross-row reuse → shared-memory staging is
    pure overhead); then an in-block combine merges the NW partials (online-softmax). Full warp
    occupancy + NW-way KV parallelism vs the old 1-warp serial scan.
  - GQA K/V is still read per q-head (4× within a group), accepted: KV bytes are ~4% of GEMM and the
    4 q-heads of a group hit the same cache lines (L2 absorbs most of it).
- **Result vs S5**: conc=32 **357 → 455 tok/s (+27.5%, TPOT 84.9 → 65.3 ms)**; conc=1 39.1 → 44.7
  (+14%, TPOT 25.8 → 22.5 ms). **51% → 65% of vLLM**, 4.26× the `main` baseline. Greedy output coherent.
- **Lesson**: the S5 profiles (short context) under-counted attention; at the tracked 1024-ctx the
  hand-rolled decode attention WAS the conc=32 bottleneck after all — and unlike the spent S4 WMMA
  attempt (which fixed nothing real), FlashDecoding-style **KV-parallelism + occupancy** is the right
  fix. Confirms the rule from the profiling sweep: GEMM/elementwise/scheduling were dead ends; the one
  hand-rolled kernel with headroom delivered. Next gap to vLLM (455→698) is likely prefill-side
  attention + the GQA 4× redundant KV read (a GQA-shared decode kernel) and CUDA graphs for conc=1.

### S7 — WMMA tensor-core prefill attention (0dd4010)
- **Target**: after S6 split decode off, the prefill-only attention kernel still used CUDA-core FMA dot
  products (per-key warp-shuffle reductions) for what is a compute-bound 512×L×128 matmul — no tensor
  cores. S4's WMMA attempt failed because it was applied to *decode* (q_len=1 wastes 15/16 of a 16-row
  WMMA tile + per-kv-head grouping starved the grid); with decode now on its own kernel, WMMA on the
  prefill path (q_len up to 512, full tiles) is the right application and S4's failure mode is gone.
- **Change**: `attention_prefill_wmma_kernel` — FlashAttention-2 with WMMA 16×16×16 bf16 (fp32
  accumulate), one warp per (16-query-tile, head, request). Q read straight from the fused-QKV buffer,
  K/V straight from the paged cache (one 16-key tile == one page, since block_size==16), all as WMMA
  loads — no S materialization. Online softmax + deferred normalization; O kept in shared fp32 and
  rescaled per tile (portable — no WMMA fragment-layout assumptions). Dispatched via
  `launch_attention_prefill` on the prefill subset; FMA fallback when block_size!=16 or head_dim!=128.
- **Result vs S6**: conc=32 **455 → 493 tok/s (+8.5%, TPOT 65.3 → 59.8 ms)**; conc=1 flat (44.7 → 44.9
  — single-stream is decode-bound, prefill is a one-time cost amortized over the 128 outputs). **65% →
  71% of vLLM**, 4.6× the `main` baseline. Output verified coherent.
- **Lesson**: prefill attention WAS a meaningful conc=32 slice (the +8.5% is the proof), and the S4
  idea (tensor cores) was right all along — just mis-applied to decode. Splitting attention by request
  type (S6) is what unlocked it: each regime now gets the kernel it wants (FlashDecoding for decode,
  WMMA-FA for prefill). Remaining gap to vLLM (493→698): GQA-shared decode (4× redundant KV read —
  judged marginal, KV is ~4% of bytes + L2-cached), CUDA graphs for conc=1, and the 1-warp/block
  occupancy of this WMMA kernel (a bigger query tile / more warps per block could push it further).

### S8 — attn_prefill occupancy: shrink shared memory (392cda5)
- **Target**: the S7 WMMA prefill kernel was 1 warp/block with ~18 KB shared → only ~5 blocks/SM
  resident (~8% warp occupancy). With so few warps, the tensor cores stall during the per-tile softmax
  (CUDA-core work) — nothing else to overlap.
- **Change**: drop the `[16,128]` `PVt` temp (8 KB). Instead of "rescale all of O, then add the full
  P·V", fold both per 16×16 d-tile through a small `[16,16]` temp: `O[:,d] = O[:,d]*corr + (P·V)[:,d]`.
  Each O element is still rescaled once and gets its P·V once — bit-identical math. Shared ~18 KB →
  ~11 KB → ~9 blocks/SM (~2× resident warps).
- **Result vs S7**: conc=32 **493 → 528 tok/s (+7.1%, TPOT 59.8 → 55.5 ms)**; conc=1 flat (44.9 → 45.2).
  **71% → 76% of vLLM**, 4.94× the `main` baseline. Output coherent.
- **Lesson**: a hand-rolled WMMA kernel at 1 warp/block is occupancy-starved even though tensor-core
  throughput looks fine — freeing shared memory to double resident warps recovered 7%. More headroom
  likely remains (a multi-warp block / KV-split would push occupancy further), but that's a bigger
  rewrite with the usual WMMA-correctness risk; the cheap shared cut banked most of the easy gain.

### S10 — scheduler: default max-batch-tokens 2048 → 1024 (642cc28)
- **Target**: sweeping the scheduler knobs (`sched_sweep.sh`) surfaced `max-batch-tokens`
  (max_num_batched_tokens) as a big lever the earlier max-prefill-tokens sweep had missed.
- **Root cause found**: at conc=32 / 1024-in, the KV pool (~36.7k tokens) sits right at demand
  (32×1152 = 36864). The default mbt=2048 lets a step admit ~4 concurrent prefill chunks (4×512),
  spiking peak KV demand past the pool → recompute-preemption thrash → 527 tok/s. mbt=1024 serializes
  prefill enough to stay under the pool → 589 (+11%), with better TTFT/TPOT, at the **same 0.9 VRAM as
  vLLM's default** (fair).
- **Verified the mechanism** with a `--gpu-mem-fraction` sweep (plumbed through Go): enlarging the pool
  to 43.5k tokens fixes mbt=2048 (527 → 588) but does **not** beat mbt=1024's 589. So ~588 is the
  no-preemption ceiling — two paths (cap concurrency, or grow the pool) reach the same place, and the
  remaining gap to vLLM (698) is **compute/framework, not KV/preemption**.
- **Result vs S8**: conc=32 528 → **589 (+11%)**; conc=1 unchanged (~45, single-stream is
  mbt-independent). **76% → 84% of vLLM**, 5.5× the `main` baseline. (gpu-mem-fraction default kept at
  0.9; the flag is now exposed for users with other VRAM/models.)
- **Lesson**: input-length-coupled (the user's call) — the mbt sweet spot exists because the std test
  sits on the KV cliff; at 512-in (KV fits) mbt is irrelevant (~890 tok/s), at 2048-in (2× over) it
  collapses. The fair, free win is capping admission via mbt; chasing past 588 means compute, not memory.

#### inlen × mbt sweep — throughput is dominated by KV capacity vs demand (2026-06-20)
`mbt_inlen.sh`, conc=32, otps by (input length, max-batch-tokens):

| inlen | KV needed (32×(in+128)) | best otps | mbt effect |
|---|---|---|---|
| 512  | 20480 (57% of pool) | ~890 | flat (mbt irrelevant — KV fits) |
| 1024 | 36864 (≈pool, the cliff) | ~589 | small mbt wins (eases preemption) |
| 2048 | 69632 (190% of pool) | 78 → 7 | collapses; bigger mbt = worse (thrash) |

The tracked 1024/128 metric sits exactly on the KV-capacity cliff — which is why everything here is so
sensitive. Throughput is governed first by KV-pool capacity vs demand, then by kernels.

### S11 attempt — CUDA graphs for pure-decode steps (TRIED, REVERTED, 2026-06-20)
Captured the pure-decode forward (run_layers + lm_head + sample) into a CUDA graph per batch size B and
replayed it (split forward into eager upload_inputs + captureable compute; pinned a cuBLAS workspace and
fixed the block-table stride so launches stay valid; graceful fallback if capture fails). Capture
**succeeded** and output stayed coherent — but it was **net negative**: conc=1 flat (45.2 → 45.7), conc=32
**589 → 520 (−12%)**. Reverted.
- **Why no conc=1 win**: conc=1 decode is **weight-bandwidth-bound**, not launch-bound — each token reads
  all 17.3 GB of weights (~17.3 ms floor; our TPOT 22 ms ≈ 78% of HBM). The ~470 launches run async and
  overlap that 17 ms of GPU work, so eliding them saves ~0 wall-clock. (vLLM's conc=1 17.9 ms ≈ the
  17.3 ms floor; its edge is leaner *eager* per-step work — H2D + sync — which the graph doesn't cover.)
- **Why conc=32 regressed**: GPU is already ~97% busy (no launch gaps to hide), and B drains 32→1 over
  the run so each new batch size pays a one-time eager-warmup + graph-instantiate → net overhead.
- **Takeaway**: FlashQwen is not launch-bound at any concurrency (saturated at 32, bandwidth-bound at 1),
  so CUDA graphs — which only remove launch overhead — have nothing to recover here. This closes the
  search: the remaining gap to vLLM (589→698) is HBM bandwidth + vLLM's leaner per-step eager path, not
  a kernel/launch problem we can fix without quantization.

#### S9 attempt — attn_prefill head-dim split (TRIED, REVERTED, 2026-06-20)
After S8, tried 2 warps/block splitting the head dim (warp 0 dims [0,64), warp 1 [64,128); each
contracts its half for a partial S, warp 0 sums + softmaxes, each does P·V for its output half). Os
stays 8 KB (partitioned, not duplicated) so occupancy rose 9 → ~14 warps/SM. Output coherent, but
conc=32 528 → 534 (**+1.1%, within noise**). The partial-S combine's extra `__syncthreads` (coupling
the two warps) roughly cancels the occupancy gain — and S8 already banked the real occupancy win, so
attn_prefill is no longer occupancy-bound. Reverted (kept the simpler 1-warp S8). Lesson: occupancy
had one cheap win (S8's shared cut); past that, more warps don't pay here.

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
