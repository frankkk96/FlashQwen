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
