# FlashQwen optimization study

A replayable, stage-by-stage benchmark of the matmul/decode optimizations. Each stage is a
**tagged commit** on the `optimization-study` branch, so any state can be checked out, built,
and re-measured.

**Setup:** RTX 4090 (24 GB) · CUDA 12.8 (native sm_89) · Qwen3-8B, BF16 weights.
**Benchmark:** single stream (batch 1), greedy, output fixed at 128 tokens, sweep over input
length, 1 warmup + median of 3 runs. All numbers from `flashqwen benchmark`.

**Reproduce any stage:**

```bash
git checkout <tag>                 # e.g. bench-0-scalar
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
./build/flashqwen benchmark --model models/qwen3-8b
```

(`models/qwen3-8b` = a plain dir from `git clone https://huggingface.co/Qwen/Qwen3-8B`.)

## Summary (TTFT and TPOT in ms; decode in tok/s)

| stage | tag | TTFT@128 | TTFT@1024 | TPOT@128 | TPOT@1024 | decode@16 |
|---|---|---:|---:|---:|---:|---:|
| 0. scalar matmul | `bench-0-scalar` | 1531 | 12585 | 22.3 | 40.1 | 49.7 |
| 1. tensor-core prefill (WMMA) | `bench-1-wmma` | 89 | 1234 | 22.3 | 40.1 | 49.7 |
| 2. BF16 KV cache | `bench-2-bf16kv` | 89 | 1233 | 22.3 | 40.0 | 49.7 |
| 3. warp attention (no barrier) | `bench-3-attn` | 83 | 908 | 22.2 | 39.4 | 49.8 |
| 4. vectorized GEMV decode | `bench-4-gemv` | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

## Stages

### 0. scalar matmul (baseline) — `bench-0-scalar`

Plain warp-per-output-element matmul (one warp does one dot product) for **both** prefill
and decode. No tensor cores. This is the "before" state of the tensor-core change.

```
  input    TTFT      TPOT     decode     output      peak
  (tok)    (ms)    (ms/tok)  (tok/s)    (tok/s)    (tok/s)
     16     192.4    20.11      49.7       46.3       53.1
    128    1531.3    22.30      44.8       29.2       47.5
    512    6195.1    29.96      33.4       12.8       34.8
   1024   12584.9    40.09      24.9        7.2       25.7
```

### 1. tensor-core (WMMA) prefill — `bench-1-wmma`

Prefill (M>1) converts activations to BF16 and runs a tensor-core WMMA GEMM; decode (M=1)
keeps the scalar GEMV. **Prefill / TTFT drops ~10–17×**; decode is unchanged (it's
memory-bound, tensor cores don't help).

```
  input    TTFT      TPOT     decode     output      peak
  (tok)    (ms)    (ms/tok)  (tok/s)    (tok/s)    (tok/s)
     16      60.2    20.13      49.7       48.5       53.1
    128      88.5    22.34      44.8       43.4       47.5
    512     422.2    29.98      33.4       30.0       34.8
   1024    1234.4    40.12      24.9       20.1       25.7
```

### 2. BF16 KV cache — `bench-2-bf16kv`

KV cache stored as BF16 instead of FP32 (K/V converted on write, read back as float in
attention). **Speed is unchanged** vs stage 1 — and that's the expected result: the current
attention kernel is *latency-bound* (a serialized per-key `__syncthreads` block reduction),
not bandwidth-bound, so halving the K/V bytes doesn't move TPOT yet. The immediate win is
**memory**: the KV cache halves (~1.2 GB → ~0.6 GB at 4096 ctx), i.e. ~2× the max context
for the same VRAM. The *speed* payoff arrives in stage 3, once attention stops being
latency-bound (the new kernel then reads half the bytes).

```
  input    TTFT      TPOT     decode     output      peak
  (tok)    (ms)    (ms/tok)  (tok/s)    (tok/s)    (tok/s)
     16      60.2    20.14      49.7       48.5       52.9
    128      88.6    22.34      44.8       43.4       47.4
    512     420.4    29.93      33.4       30.1       34.9
   1024    1233.4    40.04      25.0       20.1       25.8
```

### 3. warp attention, no per-key barrier — `bench-3-attn`

Rewrote attention from "one block per (head,query) with a serialized per-key
`__syncthreads` block reduction" to **one warp per (head,query)**: each lane owns 4 dims
(head_dim/32), the per-key q·k dot product is a warp-shuffle reduction, online softmax in
registers — no barriers. Reading the BF16 KV cache from stage 2 now also costs half the
bytes.

Result: **prefill attention is much faster** (TTFT@1024 1233 → 908 ms ≈ −26 %, @512 420 →
339). Decode TPOT is roughly flat (40.0 → 39.4): at M=1 there are only 32 work-items (one
per head), so decode attention is *parallelism*-bound, not per-key-cost-bound — squeezing
it further needs flash-decoding split-K (split the key range across more blocks + combine),
which is left as future work.

> Note: a first cut used a runtime-bounded register array (`qreg[dpl]`), which spilled to
> local memory and *regressed* decode to ~47 ms. Fixing the per-lane dim count to a
> compile-time constant (4) kept it in registers and recovered the result below.

```
  input    TTFT      TPOT     decode     output      peak
  (tok)    (ms)    (ms/tok)  (tok/s)    (tok/s)    (tok/s)
     16      60.1    20.08      49.8       48.7       53.0
    128      83.2    22.21      45.0       43.7       47.6
    512     339.0    29.57      33.8       31.0       35.3
   1024     908.0    39.35      25.4       21.5       26.2
```
