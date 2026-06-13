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
`model.safetensors.index.json`, `vocab.json`, and `merges.txt`. Weights are used as-is
(BF16, no conversion or quantization).

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
**Model:** Qwen3-8B, BF16 weights, FP32 activations (the prefill matmul runs on tensor cores in BF16).
**Method:** single stream (batch 1), greedy, fixed 128-token output (ignores EOS), 1 warmup
run + median of 3 measured runs. The sweep is built in — just `flashqwen benchmark --model <DIR>`.

Swept over input length (synthetic prompts), output fixed at 128 tokens:

| input tok | TTFT | TPOT | decode tok/s | output tok/s | peak tok/s |
|---:|---:|---:|---:|---:|---:|
| 16   | 60 ms  | 20.2 ms | 49.6 | 48.5 | 52.9 |
| 128  | 89 ms  | 22.4 ms | 44.7 | 43.4 | 47.4 |
| 512  | 0.42 s | 30.0 ms | 33.3 | 30.1 | 34.8 |
| 1024 | 1.23 s | 40.1 ms | 24.9 | 20.1 | 25.7 |

**Metric definitions:**
- **TTFT** — time to first token = prefill of the whole prompt + sampling token #1.
- **TPOT** — time per output token, averaged over the decode steps (after the first).
- **decode throughput** — tokens/s during decode only.
- **output throughput** — tokens/s including prefill (`n_out / total_time`).
- **peak output throughput** — `1 / fastest single-token latency`.

**Reading the numbers.** Decode is memory-bound: each token reads all ~16 GB of weights
once, so TPOT ≈ 20 ms ≈ 50 tok/s ≈ 16 GB × 50 ≈ 800 GB/s — close to the 4090's ~1 TB/s
bandwidth, i.e. near the hardware limit for single-sequence decode. Prefill runs on a
**tensor-core (WMMA) GEMM**, so TTFT stays low even for long inputs (~1.2 s for 1024
tokens, vs ~13 s for the earlier scalar kernel). TPOT still creeps up with context
(20 → 40 ms) because the attention kernel rescans a longer KV cache every step — that, not
prefill, is now the main thing left to optimize.

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
dispatches. Roughly 1690 lines total.

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
| `src/model.cu` / `.hpp` | weight loading + forward pass + KV cache | 217 |
| `src/kernels.cu` / `.cuh` | CUDA kernels (WMMA/GEMV matmul, rmsnorm, rope, attention, …) | 295 |
| `src/tokenizer.cpp` / `.hpp` | byte-level BPE encode/decode | 394 |
| `src/safetensors.cpp` / `.hpp` | mmap + `.safetensors` header parse | 105 |
| `src/json.hpp` | minimal JSON parser | 203 |
| `src/config.hpp` | parse `config.json` | 45 |
| `CMakeLists.txt` | build | 35 |

Notably, the "usually-a-library" plumbing — tokenizer (394) + JSON (203) — is ~600 lines,
nearly half the project. The actual neural network — CUDA kernels (295) + forward (217) —
is ~510 lines, because Qwen3 is a clean, regular dense transformer.

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
