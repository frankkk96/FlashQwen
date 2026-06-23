# Attention kernels: FlashAttention-2 vs FlashQwen — a learning comparison

A side-by-side of how FlashAttention-2's forward kernel is built vs FlashQwen's two hand-rolled
attention kernels, to (a) learn the production techniques and (b) pinpoint where FlashQwen's attention
is weaker. Citations: FA-2 = `flash-attention/csrc/flash_attn/src/`; FlashQwen = `engine/src/kernels.cu`.

Context: at the tracked 1024-in/conc=32 workload, attention is ~23% of FlashQwen's GPU time
(prefill-attn ~11% + decode-attn ~12%) while it's ~10% of vLLM's (one fused `flash_fwd_splitkv`).
GEMM is the same cuBLAS path on both. So attention is FlashQwen's one hand-rolled hot spot with room.

---

## TL;DR — the 5 things FlashAttention does that FlashQwen's PREFILL kernel doesn't

Ranked by likely impact on FlashQwen's `attn_prefill_kernel` (`kernels.cu:385-494`):

1. **O accumulator in registers, not shared.** FA keeps the whole output tile in registers
   (`flash_fwd_kernel.h:187`); FlashQwen parks `Os[16][128]` fp32 = 8 KB in shared (`kernels.cu:413`).
   That shared O is *the* occupancy wall — it caps the block at ~19% occupancy and forces a
   read-modify-write of shared every d-tile. (Our split-K experiment confirmed occupancy, not KV-chain
   length, is the bottleneck.)
2. **cp.async double-buffered K/V pipeline.** FA prefetches the *next* K/V tile with `cp.async` while
   the MMA on the *current* tile runs (`flash_fwd_kernel.h:271,305,317,335`). FlashQwen loads each K/V
   tile synchronously via `wmma::load_matrix_sync` right before using it (`kernels.cu:435,473`) — memory
   latency is fully exposed, nothing overlaps.
3. **Big M tile (64–128 query rows), not 16.** FA's `kBlockM` is 64–128 (`kernel_traits.h:66`); each
   K/V tile loaded once is reused across all those rows. FlashQwen's tile is 16 rows (one warp) — every
   K/V page is re-read for every 16 queries → 4–8× more KV traffic than necessary.
4. **S = Q·Kᵀ stays in registers through softmax into P·V.** FA computes S into a register accumulator,
   applies softmax in registers, feeds it straight to P·V (`flash_fwd_kernel.h:285,343,367`). FlashQwen
   round-trips S through shared: `store_matrix_sync(Ss)` → softmax reads/writes `Ss`/`Ps` →
   `load_matrix_sync(Ps)` for P·V, with **4 `__syncthreads` per K-tile** (`kernels.cu:438-483`).
5. **`mma.sync` (cute) with swizzled shared, not `nvcuda::wmma`.** FA uses raw MMA atoms
   (`SM80_16x8x16_F32F16F16F32_TN`, `kernel_traits.h:32`) + swizzled smem layouts to kill bank conflicts
   (`Swizzle<...>`, `kernel_traits.h:80`). `nvcuda::wmma` (FlashQwen) is higher-level: it forces operands
   through opaque fragments and effectively mandates the shared round-trips in #4.

These are **coupled**: you can't add cp.async prefetch (#2) without first moving O to registers (#1) and
dropping wmma (#5) to free the shared budget. That's why "use FA's ideas" really means "restructure the
kernel," not "patch one line."

---

## FA-2 forward kernel — the focal points (what to learn)

### 1. Tiling & block structure (`flash_fwd_kernel.h:52`, `kernel_traits.h:64-68`)
One block = one `kBlockM`-row Q tile, for one head, one batch element. `kNWarps` (4–8) warps cooperate;
`kNThreads = kNWarps*32`. N (keys) is tiled by `kBlockN` (32/64/128 by head dim). Bigger M tile = more
Q-row reuse per K/V load.

### 2. Shared-memory pipelining via cp.async (the big one) (`flash_fwd_kernel.h:250,271,305,317,335`)
Loads use `cp.async` (`SM80_CP_ASYNC_CACHEGLOBAL`, `kernel_traits.h:127`). Pattern is **fence-after-load,
wait-before-use**: `cp_async_fence()` right after issuing a tile's copy, `cp_async_wait<0>()` only when
the data is needed at the next GEMM. So the gmem→smem copy of K/V block *n+1* overlaps the MMA of block
*n*. Q is loaded once in the prologue; K/V stream in double-buffered.

### 3. MMA / tensor cores in registers (`kernel_traits.h:30-34,74-80`, `flash_fwd_kernel.h:187`)
`TiledMMA` over `SM80_16x8x16` atoms, warps laid out in the M dim. Operands (`tSrQ/tSrK/tOrVt`) live in
registers; shared K/V uses `Swizzle<kSwizzle,3,3>` so the MMA's shared reads hit different banks (no
conflicts). The S accumulator `acc_s` is registers and never written back to shared.

### 4. Online softmax fused with O rescale, in registers (`softmax.h:136-167,180`)
Running `row_max`/`row_sum` in registers. Per K-tile: new max via warp-shuffle `reduce_max`
(`Allreduce<4>`), `exp2` of (S − max), and the prior O is rescaled in registers by
`exp2((old_max−new_max)·scale)` *fused into the next P·V* — no aux memory, no extra passes. Final
`normalize_softmax_lse` divides O by the sum and emits the LSE.

### 5. O accumulator residency = registers (`flash_fwd_kernel.h:187,283,367,428`)
`partition_fragment_C(...)` puts the full `kBlockM×kHeadDim` O tile in registers (~32–64 fp32/thread),
updated in place by `gemm_rs()` (register P·V) each iteration. No shared round-trip → higher occupancy +
less traffic.

### 6. Causal masking = skip whole blocks (`flash_fwd_kernel.h:83-94`, `mask.h:188-191`)
Precompute `n_block_min/max`; fully-masked K/V blocks are **never loaded or computed** (early exit at
`:94`). Split into a short "masking" loop (only the diagonal blocks) + a "non-masking" loop (interior
blocks with *zero* mask overhead). Masking is applied in registers before softmax.

### 7. Split-KV decode + paged KV (`flash_fwd_kernel.h:499,585-594`, `flash_fwd_launch_template.h:102-161`)
For q_len≈1, split the KV range across `num_splits` blocks (grid.z); each writes partial O + LSE; a
combine kernel reduces them with the same online-softmax rescale. Paged KV: index via
`block_table[(n_block*kBlockN)/page_block_size]`, advance pointers by `block_table[next]-block_table[cur]`
between blocks → fragmented cache, no contiguity needed.

### 8. GQA without redundant K/V reads (`flash_fwd_kernel.h:148`, `flash.h:43`)
Shared kv-head index = `bidh / h_h_k_ratio` (precomputed ratio). K/V read once per kv-head, reused by all
query heads in the group.

---

## FlashQwen's two kernels — what they do today

### DECODE: `attn_decode_gqa_kernel` (`kernels.cu:233-313`) — already close to FA's approach
- Two-phase split-KV (grid `(n_kv, n_decode, ksplit)`) + combine kernel — same shape as FA's splitkv (#7).
- K/V loaded **once into registers**, reused across all G=4 GQA query heads (`kernels.cu:267-282`) — the
  GQA-shared idea (#8), and O/accumulators are in registers with warp-shuffle reduction (`:277`).
- This is the S12 kernel; it's genuinely FA-like and the profile shows it's the smaller problem.
- **Remaining gaps vs FA:** no `cp.async` overlap of the K/V stream (loads are synchronous), and it reads
  K/V straight from global per split. For q_len=1 there's no Q-reuse so staging matters less — decode is
  near-optimal already (it's KV-bandwidth-bound).

### PREFILL: `attn_prefill_kernel` (`kernels.cu:385-494`) — this is the weak one
- **1 warp per block** (`<<<grid,32>>>`, `:507`), 16-row Q tile. → gaps #1,#2,#3,#4,#5 above all apply.
- O in shared (`Os[16*128]` fp32, `:413`); S round-trips shared with 4 syncs/tile; `nvcuda::wmma`;
  synchronous K/V loads; per-element causal mask every tile (coarse tile-skip via `ntiles` only, `:409`).
- Softmax reduction uses only `lane<16` (`:442`) — half the warp idle, scalar per-row loop.

---

## UPDATE (2026-06-23): the rewrite landed — `attn_prefill_mma_kernel`

Steps 1–4 below were implemented for sm_89 (`engine/src/kernels.cu`, toggle `FQ_PREFILL_V2`, default on).
Microbench vs fp32 reference: numerically identical to WMMA (max|Δ|~7e-4), **2.5× without cp.async, 2.8×
with**. E2E (conc=32, prefix-cache on): **1024 +5.5% (621→655, 92.4%→96.2% of vLLM)**, 512 +1.7%,
512-shared-prefix +4.7%; conc=1 / in=128 flat. The biggest single lever was step 3 (block-shared K/V reused
by 4 warps = ~4× less KV traffic); cp.async added only the last ~0.9% e2e. See optimization.md S16.

**Decode was NOT rewritten / NOT unified into this kernel — on purpose.** FA reuses its mma mainloop for
decode (splitkv = "prefill with a 1-row M tile + KV split"), but doing that here regresses decode: mma
wastes 15/16 of the m16 tile on q_len=1 (the S4 lesson: WMMA-on-decode was −34%), and it loses FlashQwen's
decode GQA register-reuse (K/V loaded once, reused across the group's 4 q-heads; FA re-reads per q-head via
L2). decode is memory-bound, so two specialized kernels (S6 split-by-request-type) beat one unified mma
kernel. Also: cp.async (a global→shared instruction) can't simply be "added" to the current decode kernel,
which reads K/V global→registers with no shared staging.

## If we rewrite the prefill kernel (recommended order — DONE, see UPDATE above)

The changes are coupled, so a real rewrite ≈ a mini-FlashAttention for sm_89. Highest leverage first:

1. **Move O to registers + drop `nvcuda::wmma` for `mma.sync` (m16n8k16 bf16).** Frees the 8 KB shared O
   → occupancy unlocks, and lets S stay in registers (kills the shared round-trips + most syncs). This is
   the single biggest structural fix and the prerequisite for everything else.
2. **Grow the M tile to 64 (2–4 warps/block).** Reuse each K/V page across 64 queries, not 16 → cut KV
   traffic ~4×. Warps split the M rows; cross-warp softmax via shuffles.
3. **Add a cp.async double-buffer for the K/V stream.** Prefetch tile n+1 while MMA-ing tile n. Needs the
   shared budget freed by step 1. On Ada, opt into the 100 KB/SM dynamic shared (`cudaFuncAttribute`).
4. **Precise causal block-skip** (n_block_min/max + masking vs non-masking loops) — small but free.
5. **Swizzled shared layout** for K/V to avoid bank conflicts once on `mma.sync`.

Notes / honest expectations:
- This is essentially reimplementing FA-2's forward for sm_89 with FlashQwen's paged layout
  (`[num_blocks,16,kv_dim]`, block_size 16 — already compatible with a `kBlockN=64` = 4 pages).
- Upside is **bounded**: prefill-attn is ~11% of GPU time at conc=32; even halving it ≈ ~5% e2e. It would
  help more at prefill-heavy / longer-context and at conc=1. Validate with a microbench before judging
  e2e (microbench win ≠ serving win — see prior P8b/S7 lessons).
- sm_89 specifics to exploit: 100 KB/SM opt-in shared (room for bigger tiles + double-buffer), `cp.async`
  (cache-global), `mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32`.
