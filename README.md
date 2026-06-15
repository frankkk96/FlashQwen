# FlashQwen

A minimal, **from-scratch C++/CUDA inference engine for Qwen3-8B** — no external libraries
(no PyTorch, no cuBLAS, no tokenizers crate). Built for learning. Supports multi-turn
streaming chat and a benchmark mode (TTFT / TPOT / tok/s).

Chinese / 中文文档: [README.zh-CN.md](README.zh-CN.md)

---

## Usage

### Get the model

`git clone` the model repo into a directory, then point `--model` at it:

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-8B models/qwen3-8b
```

The directory needs `config.json`, the `*.safetensors` shards,
`model.safetensors.index.json`, `vocab.json`, and `merges.txt`. No offline conversion or
repacking — FlashQwen reads the BF16 files directly and quantizes the matmul weights to
INT8 in memory at load.

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Requires a CUDA toolkit (12.x recommended) and an NVIDIA GPU with ≥20 GB VRAM. The build
targets `sm_89` (RTX 4090 / Ada) by default; for a different GPU pass
`-DCMAKE_CUDA_ARCHITECTURES=<arch>` (e.g. `90` for Hopper).

### Run

`--model` is **required** and points at the model directory (e.g. `models/qwen3-8b` from
above; an HF hub cache dir also works). `--help` lists supported models and scans the local
hub cache for what's available.

```bash
# interactive multi-turn chat (default mode). KV cache persists across turns.
./build/flashqwen --model models/qwen3-8b

# chat with Qwen's recommended sampling
./build/flashqwen --model models/qwen3-8b --temperature 0.6 --top-p 0.95 --top-k 20

# benchmark (built-in fixed sweep over input lengths)
./build/flashqwen benchmark --model models/qwen3-8b

./build/flashqwen --help
```

**Modes:** bare command (or `chat`) → interactive chat; `benchmark` → metrics.

**In-chat commands:** `/exit`  `/quit`  `/reset` (clear history)  `/think on|off`.

**Common flags:** `--max-ctx N` (KV-cache size, default 4096), `--temperature`,
`--top-p`, `--top-k`, `--seed`, `--think` (enable Qwen3 thinking mode).

**Supported models:** any dense Qwen3 model (architecture `Qwen3ForCausalLM`): Qwen3-0.6B
/ 1.7B / 4B / 8B / 14B / 32B — dims are read from `config.json`. **Not supported:** Qwen3.5
(hybrid linear-attention + multimodal), Qwen3 MoE variants, and non-Qwen architectures.

> First launch loads ~16 GB of weights from disk (slow on network storage); subsequent
> runs hit the OS page cache and start in seconds.

---

## Local benchmark

**Hardware:** NVIDIA GeForce RTX 4090 (24 GB) · driver 580.76.05 (CUDA 13.0) · built with CUDA 12.8 (native sm_89)
**Model:** Qwen3-8B — matmul weights quantized to INT8 (per-row scale), BF16 embeddings,
FP32 activations; prefill on tensor cores (WMMA), decode attention via flash-decoding
split-K, single-token decode replayed from a CUDA graph.
**Method:** single stream (batch 1), greedy, fixed 128-token output (ignores EOS), 1 warmup
run + median of 3 measured runs. The sweep is built in — just `flashqwen benchmark --model <DIR>`.

Swept over input length (synthetic prompts), output fixed at 128 tokens:

| input tok | TTFT | TPOT | decode tok/s | output tok/s | peak tok/s |
|---:|---:|---:|---:|---:|---:|
| 16   | 69 ms  | 9.6 ms  | 104.2 | 98.7 | 105.1 |
| 128  | 97 ms  | 9.7 ms  | 102.6 | 95.3 | 103.5 |
| 512  | 0.40 s | 10.2 ms |  97.8 | 74.8 |  98.5 |
| 1024 | 0.95 s | 10.9 ms |  91.9 | 54.6 |  92.6 |

**Metric definitions:**
- **TTFT** — time to first token = prefill of the whole prompt + sampling token #1.
- **TPOT** — time per output token, averaged over the decode steps (after the first).
- **decode throughput** — tokens/s during decode only.
- **output throughput** — tokens/s including prefill (`n_out / total_time`).
- **peak output throughput** — `1 / fastest single-token latency`.

**Reading the numbers.** Decode is memory-bound — each token reads the whole model — so the
INT8 weights (~9 GB vs ~16 GB in BF16) put it around **~100 tok/s**, and flash-decoding
split-K keeps it nearly flat as context grows (TPOT 9.6 → 10.9 ms from 16 → 1024 tokens).
Prefill runs on tensor cores (WMMA); the INT8 weights are dequantized to BF16 first, which is
why TTFT is a touch higher than a pure-BF16 build. These are the numbers *after* the
optimization study below — which starts from a 49.7 tok/s scalar baseline and shows where
each gain came from.

---

## Dependencies & code structure

### Dependencies — none third-party

Only the **C++ standard library**, the **CUDA Runtime** (toolkit-bundled, not a 3rd-party
library), and **POSIX** system calls. `CMakeLists.txt` has no `target_link_libraries`.

With no dependencies to compile, a clean build (`-j8`) takes about **9 seconds**, and the
resulting binary is about **1.8 MB** (1.6 MB stripped). The model weights are loaded from
disk at runtime, so they are not part of the binary.

| usually a library | here |
|---|---|
| nlohmann/json, rapidjson | hand-written `src/json.hpp` |
| HF tokenizers / sentencepiece | hand-written byte-level BPE `src/tokenizer.*` |
| safetensors C++ lib | hand-written `src/safetensors.*` (mmap + header parse) |
| cuBLAS / CUTLASS | hand-written matmul (tensor-core WMMA for prefill, GEMV for decode) |
| PyTorch (any DL framework) | not used |

- C++ stdlib: `<vector> <string> <unordered_map> <chrono> <random> <cmath> <fstream>` …
- CUDA: `<cuda_runtime.h>`, `<cuda_bf16.h>` — **no** cuBLAS / cuDNN / cuRAND / Thrust.
- POSIX: `<sys/mman.h>` (mmap), `<fcntl.h>`, `<unistd.h>`.

### Code structure & line count

Each file has one job; `main.cpp` is a thin entry point that just parses arguments and
dispatches. Roughly 1940 lines total.

**Application layer**

| file | role | lines |
|---|---|---:|
| `src/main.cpp` | entry point: argument parsing + dispatch | 69 |
| `src/cli.cpp` / `.hpp` | architecture check + `--help` | 63 |
| `src/chat.cpp` / `.hpp` | interactive multi-turn chat | 55 |
| `src/benchmark.cpp` / `.hpp` | benchmark mode (input-length sweep) | 92 |
| `src/generate.cpp` / `.hpp` | shared prefill + decode loop | 66 |
| `src/sampler.cpp` / `.hpp` | token sampling (greedy / temp / top-k / top-p) | 50 |

**Core engine**

| file | role | lines |
|---|---|---:|
| `src/model.cu` / `.hpp` | weight loading (INT8 quant) + forward + KV cache + CUDA graph | 331 |
| `src/kernels.cu` / `.cuh` | CUDA kernels (INT8 GEMV / WMMA matmul, attention, split-K, …) | 431 |
| `src/tokenizer.cpp` / `.hpp` | byte-level BPE encode/decode | 394 |
| `src/safetensors.cpp` / `.hpp` | mmap + `.safetensors` header parse | 105 |
| `src/json.hpp` | minimal JSON parser | 203 |
| `src/config.hpp` | parse `config.json` | 45 |
| `CMakeLists.txt` | build | 35 |

Notably, the "usually-a-library" plumbing — tokenizer (394) + JSON (203) — is ~600 lines,
nearly half the project. The actual neural network — CUDA kernels (431) + forward (331) —
is ~760 lines, with the growth coming from the optimization stages below (INT8, split-K,
CUDA graph) rather than the base Qwen3 transformer, which is small and regular.

## Optimization study

The matmul/decode path was optimized in stages. Each stage is a **tagged commit on the
`optimization-study` branch**, so any version can be checked out and re-measured. **Stage 0
(scalar matmul) is the baseline** that everything below is compared against.

Measured on RTX 4090 (Qwen3-8B, BF16) — single stream, batch 1, greedy, output fixed at 128
tokens, swept over input length, median of 3 runs:

| stage | branch tag | TTFT@128 | TTFT@1024 | TPOT@1024 | decode@16 (tok/s) |
|---|---|---:|---:|---:|---:|
| **0 · scalar matmul (baseline)** | `bench-0-scalar` | 1531 ms | 12585 ms | 40.1 ms | 49.7 |
| 1 · tensor-core (WMMA) prefill | `bench-1-wmma` | 89 ms | 1234 ms | 40.1 ms | 49.7 |
| 2 · BF16 KV cache | `bench-2-bf16kv` | 89 ms | 1233 ms | 40.0 ms | 49.7 |
| 3 · warp attention (no barrier) | `bench-3-attn` | 83 ms | 908 ms | 39.4 ms | 49.8 |
| 4 · vectorized GEMV decode | `bench-4-gemv` | 83 ms | 907 ms | 38.8 ms | 51.1 |
| 5 · GPU argmax (greedy) | `bench-5-argmax` | 83 ms | 909 ms | 38.5 ms | 51.7 |
| 6 · CUDA graph (decode) | `bench-6-cudagraph` | 83 ms | 909 ms | 38.1 ms | 52.9 |
| 7 · flash-decoding split-K | `bench-7-splitk` | 83 ms | 910 ms | 18.9 ms | 56.9 |
| 8 · INT8 weight quantization | `bench-8-int8` | 97 ms | 954 ms | 10.9 ms | **104.2** |

vs the baseline: **~13–16× faster prefill**, and **~2× faster decode** that's nearly flat
across context — decode@16 49.7 → 104.2 tok/s, TPOT@1024 40.1 → 10.9 ms. (INT8 trades a
little prefill TTFT for the decode win.) Reproduce any stage:

```bash
git checkout bench-1-wmma     # or bench-0-scalar … bench-4-gemv
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
./build/flashqwen benchmark --model models/qwen3-8b
```

**Stage 0 — scalar matmul (baseline), `bench-0-scalar`.** One warp computes one output
element (a dot product) for every matmul, prefill and decode alike. No tensor cores. Prefill
is brutal: a 1024-token prompt takes ~12.6 s to first token, because the prefill GEMMs are
compute-bound and run with no tensor cores.

**Stage 1 — tensor-core (WMMA) prefill, `bench-1-wmma`.** Prefill (many tokens) converts
activations to BF16 and runs the matmul on tensor cores via WMMA (16×16×16, FP32 accumulate);
decode (one token) keeps the GEMV. **Prefill collapses ~14× (TTFT@1024 12585 → 1234 ms).**
Decode is untouched — it's memory-bound, tensor cores don't help.

**Stage 2 — BF16 KV cache, `bench-2-bf16kv`.** The KV cache is stored BF16 instead of FP32.
Speed is unchanged *here* — the attention kernel at this point is latency-bound (a serialized
per-key `__syncthreads` reduction), not bandwidth-bound — but the cache halves (~1.2 GB →
~0.6 GB at 4096 ctx), i.e. ~2× the max context. The byte savings only pay off in stage 3.

**Stage 3 — warp attention, `bench-3-attn`.** Attention is rewritten from "one block per
(head,query) with a per-key block reduction" to **one warp per (head,query)**: each lane owns
4 dims, the per-key q·k is a warp-shuffle reduction, online softmax in registers — no
barriers. **Prefill attention drops ~26 % (TTFT@1024 1233 → 908 ms).** Decode TPOT is roughly
flat: at batch 1 there are only 32 work-items (one per head), so decode attention is
parallelism-bound, not per-key-cost-bound (a real decode win needs flash-decoding split-K).

**Stage 4 — vectorized GEMV decode, `bench-4-gemv`.** The decode GEMV reads 8 elements per
step with 16-byte vectorized loads instead of one scalar BF16 at a time. A modest gain
(decode@16 49.8 → 51.1 tok/s) — the scalar version was already ~80 % of memory bandwidth.

**Stage 5 — GPU argmax for greedy, `bench-5-argmax`.** Instead of copying all 151936 logits
to the host every token and scanning them there, greedy decoding now runs an argmax
reduction on the GPU and copies back a single int. Saves ~0.2–0.3 ms/token (decode@16
51.1 → 51.7 tok/s). Sampling (temperature > 0) still copies the full logits.

**Stage 6 — CUDA graph for decode, `bench-6-cudagraph`.** A decode step launches ~430 small
kernels (36 layers × ~12); some are so short they're launch-overhead-bound rather than
hidden behind GPU work. The fixed single-token sequence is captured once into a CUDA graph
and replayed each step (token id / position / `past_len` live in device buffers the kernels
read, so the graph stays valid as context grows). Consistent ~0.4–0.5 ms/token (decode@16
51.7 → 52.9 tok/s, peak 55.2 → 56.5). Prefill (variable length) stays eager.

**Stage 7 — flash-decoding split-K, `bench-7-splitk`.** At M=1 the warp attention (stage 3)
had only 32 work-items (one per head), so decode attention was parallelism-bound and TPOT
grew with context (18.9 ms@16 → 38.1 ms@1024). Now each head's key range is split into
`ATTN_SPLITS` (16) chunks computed by separate blocks — each a partial online-softmax — and a
combine pass merges them, giving 16× the parallelism over the KV cache. **TPOT@1024
collapses 38.1 → 18.9 ms (decode 26.3 → 53.0 tok/s, ~2×)** and decode is now nearly flat
across context (~18–19 ms everywhere), i.e. bound by the weight reads (GEMV), not attention.
Prefill keeps the stage-3 kernel (already parallel enough).

**Stage 8 — INT8 weight quantization, `bench-8-int8`.** Decode was bandwidth-bound at the
bf16 ceiling (read all ~16 GB of weights per token → ~60 tok/s). The matmul weights
(attention + MLP projections + lm_head) are quantized to **INT8 with a per-output-row scale**
(symmetric, computed at load); embedding/norms stay as-is. Decode reads 1-byte weights and
dequantizes in-kernel → **decode@16 56.9 → 104.2 tok/s (~1.8×), TPOT@1024 18.9 → 10.9 ms**,
and the weight memory roughly halves (~16 → ~9 GB). Prefill dequantizes each weight to BF16
before the WMMA GEMM, which costs a little TTFT (e.g. @128 83 → 97 ms) — a fine trade since
prefill runs once but decode runs every token. Output stays coherent (per-channel INT8 is
mild for an 8B model).

**Still open:** INT4 / grouped quantization for more decode headroom, activation-INT8 tensor
cores to also speed prefill, and shared-memory tiling for the prefill WMMA.

## Batching (separate track, `feature/batching`)

The optimization study above is all **single-stream (batch 1)**. A separate track turns the
engine into a small serving backend that runs **many sequences at once**. The win is in decode:
decode is bound by reading the whole model per token, so serving `B` sequences in one step lets
that one weight read be **amortized across `B` tokens** instead of paid per token.

This is staged: **A — batched decode + static batching** (done), **B — continuous batching**
(dynamic scheduler, done), **C — PagedAttention** (paged KV, planned).

**Stage A — slot-based KV cache + batched decode.** The KV cache becomes a
`[B_max, max_ctx, kv_dim]` pool; `B_max` (the number of concurrent sequence slots) is whatever
fits under `--gpu-mem-fraction` (default 0.9) after weights and activations — e.g. **42 slots**
at `max_ctx=2048` on a 24 GB 4090. The decode path is rewritten to run a batch in one step:

- a **templated INT8 GEMV** where each warp reads a weight row once and computes all `B` dot
  products (`B` is compile-time so the accumulators stay in registers),
- per-slot `store_kv`, split-K decode attention with a batch dimension and per-sequence slot /
  `past_len`, and a batched argmax; `lm_head` runs in ≤4-sequence chunks to bound the
  `[B, vocab]` logits buffer.

Prefill stays single-sequence (it writes into a sequence's slot; prefill batching is a later
stage). The CUDA graph is dropped — batch shapes vary — so chat now runs as `B=1` through the
same decode path (a small, expected loss of the stage-6 graph win at batch 1).

Static-batch decode throughput (RTX 4090, Qwen3-8B INT8, greedy, 128 tok/seq, `input=128`):

| batch | per-seq TPOT | aggregate decode tok/s |
|---:|---:|---:|
| 1  | 10.2 ms | 98  |
| 8  | 28.3 ms | 283 |
| 16 | 49.7 ms | **322** |

So aggregate decode goes **~98 → ~322 tok/s (~3.3×)** at batch 16 — sublinear, and the reason
is instructive (and not the obvious one). A batch-16 step is **neither** weight-bandwidth-bound
(~9 GB in ~50 ms ≈ 190 GB/s, far under the 4090's ~1 TB/s) **nor** ALU-bound (FP32 math for the
step is ~2.5 ms of compute, ~5 % utilization). The limiter is **activation L2 re-reads**: the
GEMV computes one output row per warp and re-reads all `B` activation vectors for *every* row,
so activation traffic scales as `B × params` (~400 GB of L2 reads at B=16) — which is why the
step time grows ~linearly with `B`. The fix is activation *reuse* (shared-memory / tiled GEMV),
not tensor cores: a tried dequant→BF16 WMMA path was **slower** because the per-step INT8→BF16
weight dequant write (~14 GB, batch-independent) dominated. Left as future work. (`lm_head` is
fused with its argmax so its weight is read once per step.)

**Stage B — continuous batching.** There is only one execution primitive — the batched
`decode`, which takes an arbitrary running set (per-sequence KV slot + `past_len`). So "static"
vs "continuous" is a host-side *scheduling policy*, not a separate engine mode. Naive (static)
batching holds a fixed group's slots until its *slowest* sequence finishes, leaving short
requests idle behind long ones (head-of-line blocking). Continuous batching (`src/scheduler.*`)
instead keeps `n_slots` busy: admit a waiting request the instant a slot frees, and decode the
whole running set each step. On a varied-length workload it measured ~1.4× faster than the
static baseline, so continuous batching is the only serving path kept. Admission still prefills
one sequence at a time (interleaved chunked prefill is deferred).

Continuous-batching throughput on 32 requests (input 128, output 16–128 random), varying the
number of KV slots — `slots=1` is sequential serving (one request at a time):

| slots | wall | aggregate tok/s |
|---:|---:|---:|
| 1  | 22.4 s | 86  |
| 8  | 10.3 s | 187 |
| 16 |  9.8 s | **197** |

`feature/batching` is managed as a branch (no per-stage tags). To try it:

```bash
git checkout feature/batching
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
./build/flashqwen benchmark --model models/qwen3-8b   # adds static-batch + continuous sweeps
```
