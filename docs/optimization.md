# FlashQwen optimization log — catching up to vLLM

> 中文版见 [optimization.zh.md](optimization.zh.md)

The single, self-contained record of the throughput-optimization journey. One curated entry per
landed step: **what changed, the bottleneck it targeted, the measured effect, the lesson.** All
numbers and tables are inline; granular raw assets (per-run CSVs, profiler dumps, bench scripts) are
intentionally kept out of the repo.

## Executive summary

FlashQwen is a from-scratch Qwen3-8B inference engine (Go front-end + C++/CUDA backend over gRPC).
This log tracks closing the serving-throughput gap to **vLLM** on a single RTX 4090, at **bf16
parity (no quantization)**. Starting from `main` (INT8, a per-sequence scheduler) we rebuilt the
scheduler, the GEMM path, and the attention kernels.

**Final result** — fresh same-machine replay of every landed step, conc=32, output 128, vs a
**feature-matched** vLLM (`--no-enable-prefix-caching`, bf16, 0.9 mem):

| input | FlashQwen (S14) | vLLM (no prefix cache) | **% of vLLM** |
|---|---|---|---|
| 128 (decode-heavy) | 1341 | 1376 | **97.5%** |
| 512 | 908 | 944 | **96.2%** |
| 1024 (prefill-heavy) | 605 | 652 | **92.8%** |

conc=1 single-stream: 56.4 / 54.6 / 51.3 ≈ 92–97% of vLLM across the same inputs.

**Headline takeaways**
- **5.5×** over the `main` baseline at conc=32/1024 (107 → 605 tok/s).
- The gap to vLLM **grows with input length** (97.5% → 92.8%): the residual is entirely **prefill-side
  compute** (prefill GEMM at the cuBLAS/HBM limit + longer-context attention). Decode-heavy serving is
  essentially at vLLM parity.
- Decode, KV-cache capacity/eviction, and CPU/scheduling overhead were each measured and **ruled out**
  as the conc=32 bottleneck (see Conclusions).
- All comparisons here are against a **feature-matched** vLLM (no prefix caching). vLLM's default
  (prefix-cache ON, ~698/1031) was a non-equal earlier baseline — it caches the bench's shared
  chat-template prefix, which FlashQwen has no equivalent of — and is no longer used as the reference.

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

## Baselines & reference (canonical: the 2026-06-20 journey replay)

All numbers in this report are from the **single canonical dataset** — the same-machine journey replay
(2026-06-20): every landed step rebuilt clean and benched at 128/512/1024
input vs a **feature-matched vLLM** (`--no-enable-prefix-caching`, bf16, 0.9 mem). Aligned
`vllm bench serve --dataset-name random`, output 128, temp 0.

vLLM reference (conc=32 output tok/s): **128 → 1376, 512 → 944, 1024 → 652**.

Starting point — `main` (B0): per-sequence scheduler, INT8 weight-quant. Recorded at conc=32/1024 =
**106.9** tok/s (15% of the no-cache vLLM 652). B0 is INT8 + a different branch, so it is cited from
the original record, not re-run in the bf16 replay. INT8 makes its single-stream (conc=1) decode look
good (lighter weight traffic) but it **collapses under concurrency** — that collapse is what the
journey fixes.

> Earlier revisions of this log compared against vLLM with **prefix caching ON** (its default → ~698 /
> 1031), which is not feature-matched (FlashQwen has no prefix sharing). That comparison is deprecated
> and no longer used here.

## Progress

Canonical numbers from the 2026-06-20 journey replay, **standard test = 1024 input / 128 output**
(`% vLLM` is vs the feature-matched no-cache vLLM, conc=32 = 652). Cross-input (128/512) results are
in Final results below.

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

B0 is the original INT8 record (different branch/precision); R1→S14 are the fresh bf16 replay. S14 is
throughput-neutral on this metric (banks a memory/robustness/correctness fix). Reverted/not-landed
attempts S4, S9, S11, S13 are documented in the step entries, not the table.

## Step entries

> Note: absolute tok/s and "→ N" gap targets *inside* the entries below are **contemporaneous** — many
> were measured with the older harness and targeted the then-current (prefix-cache) vLLM baseline. They
> are the development narrative; the **canonical numbers are the Progress and Final-results tables**
> (2026-06-20 replay, vs no-cache vLLM). Relative deltas and lessons still hold.

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

### S12 — GQA-shared FlashDecoding decode attention (2026-06-20)
- **What**: the S6 decode kernel ran one block per (q-head, request); with Qwen3's 4:1 GQA each
  kv-head's K/V was re-read once per query head in its group (4×). New two-phase kernel: phase 1 = one
  block per (kv-head, request, **KV-split**) computes all G=4 q-heads of the group, loading each K/V
  element into registers **once** and reusing it across the 4 heads; phase 2 combines the per-split
  FlashDecoding partials. The `grid.z` KV-split (ksplit = clamp(128/n_decode, 1, 16)) keeps the 4090
  saturated despite n_kv being 4× fewer blocks than n_heads (conc=32: 8·32·4 = 1024 blocks × 8 warps =
  8192 warps = full). Old per-head kernel kept as fallback (head_dim≠128 or unsupported group).
- **Why it works now (vs the spent S4 GQA attempt)**: S4 grouped by kv-head but used only n_kv blocks
  → GPU starved → looked flat. The KV-split is what makes grouping a net win. L2 already absorbs most
  of the redundant *HBM* traffic from the 4× re-read, but the redundant L2 bandwidth + load
  instructions + score reductions still cost ~2.3× (microbench: 0.246→0.105 ms/layer).
- **Result**: conc=32 **589 → 608.8 (+3.4%, 84.4%→87.2% of vLLM)**, TPOT 54→52.1 ms; conc=1
  **45.7 → 51.3 (+12.2%)**, TPOT 22→19.6 ms. Greedy output verified bit-coherent (Rayleigh answer ok);
  standalone numerical check vs the reference kernel: max|Δ| ≤ 1e-4 (bf16 noise) for KVLEN ∈ {1..2000}.
- **Why conc=1 gains more than conc=32**: single-request KV (~4.7 MB/layer) fits in L2, so the 4×
  re-read was pure wasted L2 bandwidth — cleanly removed. At conc=32 the KV (151 MB/layer) overflows
  L2 (more HBM-bound) AND the 1024/128 workload is prefill-heavy (attn_decode is only ~13% of GPU
  time), so the end-to-end share is smaller.
- **Lesson**: a kernel-isolated speedup (2.3×) dilutes to +3.4% end-to-end because decode attention is
  a minority of GPU time on a prefill-heavy workload. The remaining hand-rolled hot kernel is
  **attn_prefill** (~11%), which still re-reads K/V 4× per GQA group — the next attention lever.

#### S13 attempt — GQA-shared PREFILL attention (TRIED via microbench, REJECTED, not landed)
After S12's decode win, tried the same GQA-sharing on `attn_prefill`: one block per (q-tile, kv-head,
request) with G=4 warps (one per q-head of the group) sharing one K/V page staged in shared, so each
K/V tile is read from global ONCE instead of 4× (once per q-head). Microbench (`prefill_probe.cu`,
R=4 reqs × qlen 512) verdict: **0.73× — 37% SLOWER** (0.684 vs 0.499 ms/call), and that's the
*favorable* build (Os shrunk to bf16; the natural fp32-Os version needs 51968 B shared > the 48 KB
static cap and won't even compile). Why it loses, unlike decode:
- **Prefill attention is compute(WMMA)/occupancy-bound, not K/V-load-bound.** Sharing the K/V *load*
  saves the thing that wasn't the bottleneck, while the 4× persistent `Os[16,128]` accumulator + the
  cross-warp `__syncthreads` barriers cost occupancy and stalls. (Decode has no big O accumulator —
  just DPL=4 registers/head — and gains because it IS bandwidth/L2-bound.)
- Consistent with S8 (prefill occupancy-bound; the win was *shrinking* shared) and S9 (more warps via
  head-dim split → flat). Prefill attention's occupancy is tapped; GQA-sharing only adds shared.
**Lesson: GQA K/V-sharing pays for the bandwidth-bound decode path (S12) but is net-negative for the
compute/occupancy-bound prefill path. Don't retry it on prefill on this architecture.**

<!-- Append one ### entry per landed step: What / Why / Change / Result (vs prev) / Lesson -->

### S14 — activation scratch sized to the real per-step bound (2026-06-20)
- **What**: `max_rows_` (the row count all activation buffers are sized to) was `max(max_ctx=4096,
  max_batch_tokens=1024)` = 4096, but a forward never exceeds `max_batch_tokens` rows (the scheduler's
  per-step budget caps T; prefills are chunked under it). Sized it to `max_batch_tokens + 16` instead.
- **Why the +16**: surfaced a latent OOB — the WMMA prefill-attention `load_matrix_sync` reads a full
  16-row Q tile, so a non-16-aligned last chunk over-reads up to 15 rows past T (harmless, masked by
  `grow<ql`). The old 4096 sizing absorbed it; sizing to exactly 1024 made T=1024 over-read past the
  buffer → illegal memory access (model_runtime.cu:272). +16 gives one Q-tile of headroom.
- **Result**: weights+activations 17.3→17.0 GB; KV pool **36,656 → 39,040 tokens** (2291→2440 blocks),
  now ABOVE the conc=32/1024 peak demand (36,864) → the KV-capacity cliff is eliminated. Throughput:
  conc=32 512 912.8→915.0, 1024 605.6→606.0 — **unchanged (noise)**. Coherent.
- **Lesson (settles the KV hypothesis)**: growing the pool past peak demand — guaranteeing zero
  preemption — moves throughput by 0. **KV cache size / eviction is NOT the conc=32 bottleneck**
  (re-confirms S10 post-S12, now with a fair memory fix rather than an unfair gpu-mem bump). Kept anyway:
  fixes the latent OOB, reclaims 0.36 GB (pool now ≈ vLLM's 40,816), and removes the cliff for
  robustness at higher concurrency / longer context.

### S15 — automatic prefix caching (branch `feat/prefix-caching`, fad12b7, 2026-06-22)
- **What**: vLLM-style content-hashed KV reuse — identical prompt prefixes are prefilled once and the
  cached KV blocks are spliced onto later requests. This lands the feature that the Conclusions had
  parked as an "off-table route past the gap" (route #1), and turns the long-standing caveat
  ("our headline gap was measured against a vLLM *with* prefix caching, which we lacked") into a
  feature-matched, both-on comparison.
- **Change**: three layers mirroring vLLM. **BlockPool** gains a per-block refcount + LRU free queue +
  a content-hash→block registry; a refcount-0 block is reclaimable but kept (hash mapping retained) so
  an identical prefix can resurrect it, and a fresh alloc pops the LRU front and evicts that block's
  mapping. **Scheduler** chains 64-bit block hashes over (prompt ++ output); on admission
  `acquire_prefix()` splices cached prefix blocks onto the block table and advances `computed_` past
  them (always leaving ≥1 token), and after each forward `cache_filled()` registers newly-full blocks
  (a finished request's blocks stay cached for the next identical prompt; a preempted request
  re-acquires its own still-cached prefix on resume). Correctness rests on absolute-position KV
  addressing + 16-aligned full-block reuse, so the chunked-prefill forward path is unchanged. On by
  default; **`FQ_PREFIX_CACHE=0` disables it** for A/B, and `[kvstat]` now reports prefix hit rate.
- **A/B (2026-06-22, same 口径 as the journey replay: `vllm bench serve --dataset-name random`,
  out=128, temp 0, chat endpoint, enable_thinking=false; harness `/root/bench-compare/prefix_test.sh`).
  conc=32 output tok/s, prefix cache OFF → ON for each engine:**

  | workload (conc=32) | FQ off → on | ΔFQ | vLLM off → on | ΔvLLM | FQ-on / vLLM-on |
  |---|---|---|---|---|---|
  | 128 random | 1350 → 1359 | +0.6% | 1406 → 1407 | ~0 | 96.6% |
  | 512 random | 920 → 953 | +3.6% | 962 → 982 | +2.0% | 97.1% |
  | 1024 random | 615 → 630 | +2.4% | 668 → 681 | +1.9% | 92.4% |
  | **512 shared + 512 random** | **621 → 844** | **+35.9%** | **666 → 915** | **+37.4%** | **92.2%** |

  conc=1 (single stream) is flat for both — only the template prefix is shared: FQ 56/55/51 ≈ 97/96/92%
  of vLLM 58/57/56. (FQ-off prefix_hit confirmed 0%; FQ-on ~8% on pure-random, ramping to ~22% under
  the shared-prefix probe — vLLM's reported hit rate tracks it: ~6% → ~19%.)
- **Result / Lesson**: on *pure-random* prompts the gain is modest (+0.6–3.6%) because the only shared
  prefix is the Qwen3 chat-template preamble (~8% of tokens) — and at conc=32 the bottleneck is
  prefill/decode compute, not the small prefill saved. The feature's real value appears when prompts
  **genuinely share a prefix** (system prompts, RAG context, few-shot, multi-turn): a 512-token shared
  prefix yields **+35.9% (FQ) ≈ +37.4% (vLLM)** — FlashQwen's prefix caching is **as effective as
  vLLM's**. Crucially, with both engines now feature-matched *including* prefix caching, FlashQwen holds
  **92–97% of vLLM** across inputs — the **same parity band** as the no-cache journey replay below. This
  closes the prefix-caching thread: the feature is no longer a missing capability, and the residual gap
  is still the prefill-side compute identified in the Conclusions, not the absent feature.

### S16 — prefill attention rewrite: WMMA → mma.sync FlashAttention-style (branch `feat/prefix-caching`, 2026-06-23)

**What.** An nsys re-profile of the tracked 1024/conc32 step (prefix-cache on) split GPU time into
**GEMM ~72% / prefill-attn ~11% / decode-attn ~12% / elementwise ~5%** — GEMM is the *identical* cuBLAS
path as vLLM (same `ampere_*_s1688gemm_*` kernels, same ~1.12 ms/call), so the only hand-rolled hot spot
with headroom is attention. A direct vLLM trace showed it runs **one** fused `flash_fwd_splitkv`
(FlashAttention) at ~10% vs FlashQwen's two hand-rolled kernels.

**Tried and rejected first — prefill split-K** (`FQ_PREFILL_KSPLIT`): mirror decode's KV-split onto the
WMMA prefill kernel. **0 e2e gain** — prefill already has thousands of independent blocks
(qtiles×heads×reqs) even at conc=1, so it's never block-starved; the wall is *per-block occupancy*
(the 8 KB shared O accumulator + 1 warp/block), which split-K doesn't change. Reverted.

**Change (the real fix).** Rewrote `attn_prefill_kernel` (WMMA, 1 warp/block, O in shared) as
`attn_prefill_mma_kernel`, porting FlashAttention's techniques for sm_89:
1. **O accumulator in registers** via raw `mma.sync.m16n8k16.f32.bf16.bf16.f32` (not `nvcuda::wmma`) —
   frees the 8 KB shared O that was the occupancy wall and keeps S in registers (no shared round-trip).
2. **64-row M tile = 4 warps/block** (each warp 16 rows).
3. **K/V staged to block-shared once per 16-key tile, reused by all 4 warps** (~4× less KV traffic — the
   biggest single lever); P passed through a tiny shared buffer to skip the C→A fragment repack; online
   softmax in registers via quad `__shfl_xor`.
4. **cp.async double-buffer** of the K/V stream (prefetch tile n+1 while computing tile n).
Validated against an fp32 reference in a standalone microbench (numerically identical to WMMA, max|Δ|~7e-4;
**2.5× without cp.async, 2.8× with**). Toggle `FQ_PREFILL_V2` (default 1); falls back to WMMA otherwise.

**Result — same-session A/B (`FQ_PREFILL_V2` 0=WMMA / 1=mma), conc=32 output tok/s, both cache states.**
The no-cache S14/WMMA row reproduces the historical journey replay (606 vs 605 @1024; vLLM 652 identical),
confirming the two are directly comparable.

*No prefix cache (feature-matched, extends the journey-replay table):*

| in | S14 (WMMA) | **S16 (mma)** | ΔFQ | vLLM | S16 % of vLLM |
|---|---|---|---|---|---|
| 128  | 1315 | 1318 | +0.2% | 1393 | 94.6% |
| 512  | 913  | 927  | +1.5% | 948  | 97.7% |
| **1024** | **606** | **640** | **+5.6%** | **652** | **93.0% → 98.1%** |

*Prefix cache ON (the branch default / deployed mode):*

| in | S15 (WMMA+cache) | **S16 (mma+cache)** | ΔFQ | vLLM+cache | S16 % of vLLM |
|---|---|---|---|---|---|
| 128  | 1351 | 1352 | ~0    | 1407 | 96.1% |
| 512  | 940  | 955  | +1.7% | 982  | 97.3% |
| **1024** | **621** | **655** | **+5.5%** | **681** | **92.4% → 96.2%** |
| 512 shared + 512 random | 831 | 870 | +4.7% | — | — |

in=128 and all conc=1 are flat (prefill-attn is a negligible slice there) — expected. The rewrite lifts
exactly the long-context, prefill-heavy regime it targets: **at 1024/conc32 the gap to vLLM closes from
~7% to ~2% (no-cache) / ~4% (cache-on)**, the smallest it has ever been at bf16 parity.

**Lesson.** The microbench 2.8× dilutes to +5.5% e2e exactly as the profile predicts (prefill-attn ~11% of
GPU → halved → ~5% realized); cp.async added only the last ~0.9% (diminishing once the first 2.5× shrank
attention's share). **Decode left unchanged and deliberately NOT unified into one mma kernel:** FA reuses
its mainloop for decode (splitkv) for library maintainability, but on q_len=1 mma wastes 15/16 of the M
tile (the S4 lesson) and FlashQwen's SIMT decode kernel has a GQA register-reuse advantage FA lacks — two
specialized kernels (S6's split-by-request-type) beat one unified mma kernel here. Full FA-vs-FlashQwen
comparison: [`attention-vs-flashattention.md`](attention-vs-flashattention.md).

## Final results — journey replay across input lengths (2026-06-20)

Every landed step `git checkout`'d, rebuilt clean, benched at 128/512/1024 input (output 128) on one
machine/day, vs a feature-matched vLLM (`--no-enable-prefix-caching`).

**conc=32 output tok/s (and % of feature-matched vLLM):**

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

(S16 row + its vLLM `%` are a same-session A/B re-measured 2026-06-23 — the S16-off/WMMA control reproduced
the historical S14 row: 606 vs 605 @1024, vLLM 652 identical — so the row is directly comparable. S15
(prefix caching) is omitted here because with prefix cache OFF it is identical to S14; its effect is shown
in the S15 entry's cache-on A/B above. vLLM 128/512 cells: historical / S16-session.)

**Reading the chart.** Two clean facts: (1) every optimization acts on the regime it targets and they
don't overlap — S6 (decode attn) lifts in=128 most; S7/S8 (prefill attn) lift in=1024 most; S10 (the
KV-cliff scheduler fix) moves *only* in=1024. (2) Through S14 the gap to vLLM was monotone in input length
(97.5% → 96.2% → 92.8%) — the residual concentrated at long input, i.e. prefill-side compute. **S16's
mma.sync prefill-attention rewrite then attacked exactly that long-input residual**, lifting 1024 from
92.8% to **98.1%** of vLLM (and 512 to 97.7%) — so the curve is now nearly flat across input length and
the long-context prefill gap, the last structural one, is largely closed at bf16 parity.

## Conclusions — what's exhausted and where the residual lives

> **Update (S16, 2026-06-23):** the prefill-attention rewrite below closed most of the long-input gap
> this section attributed to "prefill-side compute" — 1024/conc32 is now **98.1% of vLLM no-cache /
> 96.2% cache-on** (was 92.8%). The analysis below still holds for *why* the gap existed and which levers
> were dead; the one live attention lever it flagged has now been taken.


At bf16 parity on a 4090, the conc=32 bottleneck was chased to ground. Levers tried and **ruled out**
(each with data, not reasoning):

- **Decode GEMM** — at ~84% of HBM peak; both engines hit the same physical floor (only quantization
  beats it, off-table). cuBLAS `cublasGemmEx(DEFAULT)` already picks the aspect-ratio-optimal kernel
  per shape/M (split-K auto-tuned: large-K `down` always splits, large-N `gateup`/`lm_head` never;
  cuBLASLt autotune ties it for M≥8, wins only at M=1/conc-1). GEMM is closed for conc=32.
- **CPU/scheduling overlap (async scheduler)** — steady-state GPU is **98.7% busy**; the per-step host
  bubble is ~0.2%. Our 50 ms GPU steps dwarf ~0.12 ms host prep, so vLLM/SGLang-style "run one step
  ahead" buys ~nothing here.
- **CUDA graphs** (S11) — net-negative (−12% @conc=32): not launch-bound.
- **KV cache size / eviction** (S14) — growing the pool *past* peak demand (zero preemption possible)
  moved throughput by **0**. Not the bottleneck; the marginal cliff is absorbed by completion staggering.
- **prefill attention occupancy** (S8 banked it; S9 head-split flat; S13 GQA-shared prefill 0.73×
  *slower* — compute/occupancy-bound, can't afford the 4× O accumulator). Tapped.
- **prefill chunk size / gpu-mem-fraction** — flat / unfair. Done.

**The residual gap (in=1024 ~7%) is prefill-side compute** — the prefill GEMM is large-M and
compute-bound at the cuBLAS limit, and our hand-rolled WMMA prefill attention, while good, is a bit
behind vLLM's FlashAttention there. It shrinks toward 0 as input shortens (decode dominates).

**Only routes past it, both off the bf16-parity table:**
1. **Prefix caching** — **LANDED (S15, `feat/prefix-caching`).** Automatic content-hashed KV reuse,
   on by default. On pure-random inputs it recovers only the shared chat-template prefix (~2–4% at
   conc=32, matching what vLLM's default gains); on workloads with a real shared prefix it is a large
   win (**+36%** with a 512-token shared prefix, on par with vLLM's +37%). With both engines now
   feature-matched including prefix caching, FlashQwen holds the same 92–97%-of-vLLM band. A further
   step here would be cascade / shared-prefix attention (the FlashInfer decomposition) for the
   *attention* over a shared prefix, not just KV reuse.
2. **Quantization** (fp8/int weights or fp8 KV) — directly cuts the dominant weight-traffic / prefill-
   GEMM cost, but breaks strict bf16 parity.

Net: at equal precision + equal features, FlashQwen is **92.8–97.5% of vLLM** depending on input
length, **5.5×** over where it started, with the remaining gap localized, measured, and explained.
