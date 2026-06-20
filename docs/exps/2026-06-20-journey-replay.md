# Journey replay — every landed step × {128, 512, 1024} input vs vLLM (no prefix cache)

Goal: one clean, same-machine/same-day dataset showing the throughput progression of the whole
optimization journey across THREE input lengths (decode-heavy → prefill-heavy), measured against a
**feature-matched** vLLM (`--no-enable-prefix-caching`, bf16, 0.9 mem) — the fair denominator.

## What's replayed (landed commits only)

Each step is `git checkout`'d, rebuilt from clean (`rm -rf engine/build && make`), and benched with
**its own built-in defaults** (so e.g. pre-S10 runs at max-batch-tokens=2048 — the authentic behaviour
of that step). Output 128, greedy, thinking off.

| step | commit | what landed |
|---|---|---|
| R1 | 5065b2e | unified token-budget scheduler + GPU sampling (regression vs INT8 main) |
| R2 | 291cb74 | scheduler/kernel refactor (perf-neutral) |
| S3 | 7650654 | bf16 weights + cuBLAS GEMM + FlashAttention-2 |
| S5 | c2f98a0 | fused QKV/gate-up GEMM + add-rmsnorm + RoPE table |
| S6 | f0b1499 | FlashDecoding decode-attention (split by request type) |
| S7 | 0dd4010 | WMMA tensor-core prefill attention |
| S8 | 392cda5 | attn_prefill occupancy (shrink shared) |
| S10 | 642cc28 | scheduler default max-batch-tokens 2048→1024 |
| S12 | 240aaa1 | GQA-shared FlashDecoding (read K/V once/group + KV-split) |
| S14 | e5a99c8 | activation scratch sized to per-step bound (pool↑, KV-cliff gone) + latent WMMA OOB fix |

**Not replayable** (tried, reverted via working-tree checkout — no standalone build commit):
S4 (WMMA+GQA decode), S9 (prefill head-dim split), S11 (CUDA graphs), S13 (prefill GQA, microbench only).
Their negative results are documented in `optimization.md`; we don't rebuild them.

**B0** (= `main`, INT8 weight-quant) is NOT replayed here — it's a different branch + needs the Go
`ignore_eos` shim, and is a different precision regime. Cited from the recorded baseline (conc=32/1024
= 106.9). The fresh dataset starts at R1 (the bf16 unified-scheduler line).

## Matrix

- inputs: **128 / 512 / 1024**  (output 128)
- concurrency: **1** (single-stream, 6 prompts) and **32** (saturated, 96 prompts)
- 10 steps × 3 inputs × 2 conc = **60 FlashQwen cases** + **6 vLLM reference cases**

## Method / fairness

- Same machine, same day, one server at a time (killed between).
- vLLM: `--max-num-seqs 32 --gpu-memory-utilization 0.9 --no-enable-prefix-caching` (feature-matched).
- Metric: output tok/s (from `vllm bench serve --dataset-name random --seed 1234 --temperature 0`).
- Robust: continue-on-error per step (build/serve failures logged, not fatal); **resumable** (skips
  steps already present in the CSV); repo restored to `perf/vllm-catchup-v2` + rebuilt at the end.

## Run

```
! bash /root/bench-compare/journey_replay.sh
```

~1.5–2 h (10 clean rebuilds + 11 server loads + 66 benches). Output: `results/journey.csv`,
progress in `journey_run.log`. When done I render the progression chart (otps vs step, one line per
input, with the vLLM reference + % -of-vLLM).
