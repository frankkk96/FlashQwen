# FlashQwen

A minimal, **from-scratch C++/CUDA inference engine for Qwen3-8B** — no external libraries
(no PyTorch, no cuBLAS, no tokenizers crate). Built for learning. Supports multi-turn
streaming chat and a benchmark mode (TTFT / TPOT / tok/s).

Chinese / 中文文档: [README.zh-CN.md](README.zh-CN.md)

---

## Usage

### Get the model

The weights are used **directly as downloaded from HuggingFace** — no conversion,
quantization, or repacking step. Download Qwen3-8B with the official tooling:

```bash
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-8B
```

This fills an HF hub cache dir (default `~/.cache/huggingface/hub/models--Qwen--Qwen3-8B`);
pass that path to `--model`. FlashQwen reads `config.json`, the `*.safetensors` shards,
`vocab.json`, and `merges.txt` straight from it.

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Requires a CUDA toolkit (12.x recommended) and an NVIDIA GPU with ≥20 GB VRAM. The build
targets `sm_89` (RTX 4090 / Ada) by default; for a different GPU pass
`-DCMAKE_CUDA_ARCHITECTURES=<arch>` (e.g. `90` for Hopper).

### Run

`--model` is **required** — either an HF snapshot dir containing `config.json`, or an HF
hub cache dir like `.../models--Qwen--Qwen3-8B`. `--help` lists supported models and
scans the local hub cache for what's available.

```bash
# interactive multi-turn chat (default mode). KV cache persists across turns.
./build/flashqwen --model /path/to/models--Qwen--Qwen3-8B

# chat with Qwen's recommended sampling
./build/flashqwen --model <DIR> --temperature 0.6 --top-p 0.95 --top-k 20

# benchmark (built-in fixed sweep over input lengths)
./build/flashqwen benchmark --model <DIR>

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
- POSIX: `<sys/mman.h>` (mmap), `<fcntl.h>`, `<unistd.h>`, `<dirent.h>`.

### Code structure & line count

Each file has one job; `main.cpp` is a thin entry point that just parses arguments and
dispatches. Roughly 1720 lines total.

**Application layer**

| file | role | lines |
|---|---|---:|
| `src/main.cpp` | entry point: argument parsing + dispatch | 73 |
| `src/cli.cpp` / `.hpp` | model-dir resolution, arch check, `--help` | 117 |
| `src/chat.cpp` / `.hpp` | interactive multi-turn chat | 55 |
| `src/benchmark.cpp` / `.hpp` | benchmark mode | 52 |
| `src/generate.cpp` / `.hpp` | shared prefill + decode loop | 66 |
| `src/sampler.cpp` / `.hpp` | token sampling (greedy / temp / top-k / top-p) | 50 |

**Core engine**

| file | role | lines |
|---|---|---:|
| `src/model.cu` / `.hpp` | weight loading + forward pass + KV cache | 219 |
| `src/kernels.cu` / `.cuh` | CUDA kernels (WMMA/GEMV matmul, rmsnorm, rope, attention, …) | 270 |
| `src/tokenizer.cpp` / `.hpp` | byte-level BPE encode/decode | 394 |
| `src/safetensors.cpp` / `.hpp` | mmap + `.safetensors` header parse | 105 |
| `src/json.hpp` | minimal JSON parser | 203 |
| `src/config.hpp` | parse `config.json` | 45 |
| `CMakeLists.txt` | build | 35 |

Notably, the "usually-a-library" plumbing — tokenizer (394) + JSON (203) — is ~600 lines,
nearly half the project. The actual neural network — CUDA kernels (270) + forward (219) —
is ~490 lines, because Qwen3 is a clean, regular dense transformer.
