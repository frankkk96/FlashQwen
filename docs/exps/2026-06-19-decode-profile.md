# nsys profile — where conc=32 time goes (2026-06-19)

Goal: confirm the post-S5 lever. Two `nsys` captures (12 s windows, conc=32), kernel-time summary.

## Decode-dominated (input 32 / output 256 — negligible prefill)
GPU **~97% busy** (11.59 s kernels / 12 s wall) — *not* launch/CPU-gap bound in steady decode.

| bucket | % GPU | note |
|---|---|---|
| GEMM (all linear + lm_head) | **~76%** | cuBLAS, see below |
| attention_flash | **~19%** | our hand-rolled kernel |
| elementwise (add_rmsnorm, silu, head_norm_rope, store_kv, …) | **~4.5%** | confirms S5 #5/#7 are noise |
| sampling | 0.1% | |

**Decode GEMM is bandwidth-bound and near-optimal.** Weight bytes/step = 13.9 GB (36 layers ×
[qkv 50 + o 34 + gate_up 201 + down 101 MB]) + 1.24 GB (lm_head) = **15.1 GB**. Measured GEMM time
≈ 17.8 ms/step → **848 GB/s = ~84% of the 4090's ~1008 GB/s HBM peak.** cuBLAS is already at the wall.

→ **cuBLASLt autotuning is a dead end** (<16% headroom on the GEMM bucket, and GEMM is memory- not
kernel-bound). The only way past the decode-GEMM wall is reading fewer weight bytes (quantization —
out of scope, bf16-parity mandate) or a larger effective batch (more tokens amortizing each weight
read). At conc=32 decode we are ~84% of the hard physical floor.

## Mixed (input 512 / output 128 — prefill + decode)
attention jumps to **~38%** (prefill is O(n²)) and big-M prefill GEMMs appear (~1.4 ms each). So the
**1024/128 tracked metric is much more prefill-sensitive than pure decode** — its gap to vLLM is not
decode GEMM (bandwidth-bound for both engines) but prefill cost + prefill-chunk drag on decode steps
(a 512-token prefill chunk merged into a step stalls the 31 decodes behind it). Consistent with the
standing [[s2-gap-is-scheduling]] note.

## Takeaways / next levers (in priority)
1. **Decode is done** — at 84% HBM BW, no kernel win left without quantization. Drop the cuBLASLt idea.
2. **Attention** is the only hand-rolled hot kernel with headroom (19% decode, 38% mixed). A
   FlashDecoding / split-K decode kernel (parallelize over KV, unlike the q_len=1-wasteful S4 WMMA
   attempt) would help both regimes. Distinct lever from the spent S4 tensor-core experiment.
3. **Prefill-chunk drag** — smaller/smarter prefill chunking so 1024-token prompts don't stall decode
   (tune `--max-prefill-tokens`, or split prefill onto its own step). Zero/low code, matches vLLM's edge.

## Follow-up experiments (2026-06-19, same session)

### E1 — `--max-prefill-tokens` sweep on the tracked 1024/128 conc=32 workload — REFUTED
Plumbed `--max-batch-tokens` / `--max-prefill-tokens` through the Go layer (serve.go → common.go →
supervisor) so the chunk size could be swept. Fixed `--slots 32 --max-batch-tokens 2048`.

| max_prefill | conc=32 tok/s | TTFT | TPOT |
|---|---|---|---|
| 128 | 356.0 | 18.4 ms | 85.4 ms |
| 256 | 355.6 | 18.5 ms | 84.9 ms |
| 512 (default) | 357.1 | 18.1 ms | 84.9 ms |
| 1024 | 354.6 | 18.2 ms | 85.3 ms |

**All within noise — prefill chunk size does not move conc=32 throughput (nor TTFT/TPOT).** So the
prefill-chunk-drag hypothesis (lever #3 above) is *refuted* for this workload: how prefill is sliced
doesn't matter. (Script: `/root/bench-compare/prefill_sweep.sh`, CSV `results/prefill_sweep.csv`.)

### Open question — the long-context decode regime is still unprofiled
All three profiles above used SHORT context (32–512 tokens). The tracked metric is **1024-token
context**, where decode attention reads a ~1152-token KV every step — a regime none of these captures.
TPOT in the std test is ~85 ms vs ~24 ms/step in the short-context decode profile: a 3.5× gap that
must come from long-context attention (our decode kernel uses 1 active warp per (head,req) scanning
~72 KV tiles sequentially, and re-reads each kv-head's K/V 4× across the GQA group). **Next: profile
the exact 1024/128 conc=32 workload** (`/root/bench-compare/nsys_std.sh`, written but not yet run —
interrupted) to confirm whether long-context decode attention, not GEMM, is the real conc=32 ceiling.
If so, the lever is a FlashDecoding/split-K decode attention (distinct from the spent S4 WMMA attempt).

### What's been ruled out so far (conc=32, 1024/128)
- cuBLASLt GEMM autotuning — decode GEMM is at 84% HBM BW, no headroom.
- CUDA graphs / async scheduling — GPU is ~97% busy in decode, no launch gaps to fill.
- Elementwise/launch fusion (S5 #5/#7) — ~4% of GPU time, in the noise.
- Prefill chunk size (`--max-prefill-tokens`) — no effect (E1).
