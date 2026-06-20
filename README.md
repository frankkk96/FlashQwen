# FlashQwen

A from-scratch **Qwen3-8B inference stack** (C++/CUDA + Go).

> **FlashQwen deliberately stays small, trading breadth of model support for a clean codebase.**
> We believe that in the age of AI-assisted programming, giving the AI clean context and a tight
> feedback loop is worth more than piling on dependencies, abstractions, adapters, and compromises.
> So there is no general framework here — just one narrow, complete inference path that is easy to
> read and easy to run.

中文: [README.zh-CN.md](README.zh-CN.md)

---

## 1. Project & version evolution

FlashQwen grew in three stages. Each stage is a self-contained version with its own goal; together
they trace the path from a single-file teaching engine to a serving runtime that is competitive with
vLLM.

### Stage 1 — from-scratch terminal engine
The earliest version (on `main`): a single C++/CUDA program, **under 2000 lines with zero
dependencies** (no PyTorch, no cuBLAS, no HuggingFace tokenizers). It loads Qwen3-8B weights directly
from `.safetensors`, runs prefill + decode + sampling, and is driven entirely from the **terminal**.
This stage also includes the single-stream optimization study (scalar matmul → tensor-core WMMA →
flash-decoding split-K → INT8 weight quantization).

### Stage 2 — scheduling + an OpenAI-compatible API service
Added a real serving layer: continuous batching + PagedAttention in the C++ engine, wrapped in a
**Go/Gin gateway that exposes the standard OpenAI API** over gRPC. From this stage on you start a
*service* (`flashqwen serve`) and talk to it with any OpenAI client, instead of running a one-shot
terminal program.

### Stage 3 — serving performance optimization (current)
The current focus: closing the serving-throughput gap to **vLLM** at **bf16 parity (no
quantization)**. Through a series of kernel and scheduler rewrites (bf16 + cuBLAS GEMM,
FlashAttention-2 prefill, GQA-shared FlashDecoding, a unified token-budget scheduler, …), FlashQwen
now reaches **~95% of vLLM's serving throughput with prefix caching disabled** on a single RTX 4090,
and ~5.5× its own Stage-1/2 baseline. Full report: [`docs/optimization.md`](docs/optimization.md).

### Architecture (Stage 2+)

Two layers communicating over gRPC:

- **Outer layer — a Go application** (the root module; builds to `./flashqwen`). Handles everything
  text-related: tokenisation, the Qwen3 ChatML template, tool-call detection, and serving both an
  OpenAI-compatible HTTP API and an interactive CLI. It bundles the compiled C++ engine via
  `//go:embed`, extracts it at runtime, and launches it as a child process — so you run one binary.
- **Inner layer — a C++/CUDA token engine** (`engine/`). All the GPU work: weight loading, prefill,
  decode, sampling, the paged KV cache, and the continuous-batching scheduler. It only accepts token
  ids and returns sampled token ids — it does no tokenisation and knows nothing about text formats.

---

## 2. Usage

### Get the model

`--model` points at a local directory; FlashQwen downloads nothing itself:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-8B --local-dir models/qwen3-8b
```

(Set `HF_ENDPOINT=https://hf-mirror.com` first if you need a mirror.) The directory must contain
`config.json`, the `*.safetensors` shards + `model.safetensors.index.json`, and the tokenizer files
(`tokenizer.json`, `vocab.json`, `merges.txt`, `generation_config.json`). The engine reads the BF16
safetensors directly — no offline conversion or repacking.

### Build

```bash
make            # build C++ engine → embed into Go → go build, producing ./flashqwen
```

`make` runs three steps: cmake builds the C++ engine, the engine binary is copied into
`internal/supervisor/bin/` (for `//go:embed`), and `go build` produces the final binary. Needs a CUDA
toolkit (12.x), Go 1.26+, gRPC/protobuf, **cuBLAS**, and an NVIDIA GPU with ≥20 GB. Targets `sm_89`
(RTX 4090 / Ada) by default; for other cards set `-DCMAKE_CUDA_ARCHITECTURES=<arch>` before
`make engine`.

### Run

```bash
./flashqwen serve --model models/qwen3-8b    # OpenAI-compatible server (default :8000)
./flashqwen chat  --model models/qwen3-8b    # interactive multi-turn chat
./flashqwen --help
```

- **OpenAI endpoints:** `POST /v1/chat/completions` (streaming / non-streaming / tools),
  `GET /v1/models`, `GET /healthz`.
- **Common flags:** `--max-ctx N` (KV/context length), `--slots N` (max concurrent sequences),
  `--max-batch-tokens N` (tokens computed per scheduler step), `--addr` (listen address). Sampling
  params (temperature, top_p) are per-request via the API.
- **In-chat commands:** `/exit` `/quit` leave, `/reset` clears context, `/think on|off` toggles
  thinking mode.
- **Supported models:** dense Qwen3 (`Qwen3ForCausalLM`). MoE / multimodal / non-Qwen are not
  supported.

---

## 3. Stages, characteristics & history

| Stage | Theme | Characteristics | Branch | Commits |
|---|---|---|---|---|
| **1** | From-scratch terminal engine | Single C++/CUDA binary, <2000 LOC, **zero deps**, terminal-only. INT8 weights, CUDA-graph decode, flash-decoding split-K. Single-stream study. | `main` | `7d6764c` (initial) → `89318c5` |
| **2** | Batching + OpenAI API service | Continuous batching + PagedAttention; Go/Gin **OpenAI-compatible API** over gRPC. Starts a service. | `main` | `566fac6` → `08392df` |
| **3** | Serving performance optimization | bf16 + cuBLAS GEMM, FlashAttention-2 prefill, GQA-shared FlashDecoding, unified token-budget scheduler. **~95% of vLLM** (no prefix cache), bf16 parity. | `perf/serving-optimization` | `08392df` (baseline) → `53ac297` |

### Stage 3 — optimization steps (branch `perf/serving-optimization`)

The detailed report — what each step changed, the bottleneck it targeted, the measured effect, and
the dead-ends — is in [`docs/optimization.md`](docs/optimization.md). Every landed step was rebuilt
clean and benched at **128 / 512 / 1024 input** (output 128) on one machine, vs a **feature-matched
vLLM** (`--no-enable-prefix-caching`, bf16, 0.9 mem).

**Saturated throughput (concurrency 32) — output tok/s (% of vLLM):**

| Step | change | commit | in=128 | in=512 | in=1024 |
|---|---|---|---|---|---|
| R1 | INT8 unified token-budget scheduler + GPU sampling | `5065b2e` | 248 (18.0%) | 139 (14.7%) | 86 (13.2%) |
| R2 | scheduler/kernel refactor (perf-neutral) | `291cb74` | 247 (18.0%) | 139 (14.8%) | 86 (13.2%) |
| S3 | bf16 weights + cuBLAS GEMM + FlashAttention-2 | `7650654` | 1094 (79.5%) | 628 (66.5%) | 348 (53.4%) |
| S5 | fused QKV / gate-up GEMM | `c2f98a0` | 1141 (82.9%) | 648 (68.6%) | 356 (54.6%) |
| S6 | FlashDecoding decode-attention (split by request type) | `f0b1499` | 1317 (95.7%) | 824 (87.3%) | 454 (69.6%) |
| S7 | WMMA tensor-core prefill attention | `0dd4010` | 1326 (96.4%) | 860 (91.1%) | 501 (76.8%) |
| S8 | prefill-attention occupancy | `392cda5` | 1328 (96.5%) | 878 (93.0%) | 531 (81.5%) |
| S10 | scheduler: max-batch-tokens 2048→1024 | `642cc28` | 1334 (97.0%) | 882 (93.4%) | 581 (89.1%) |
| S12 | GQA-shared FlashDecoding (read K/V once per group) | `240aaa1` | 1338 (97.2%) | 906 (96.0%) | 604 (92.7%) |
| **S14** | activation right-sizing + KV-pool / OOB fix | `e5a99c8` | **1341 (97.5%)** | **908 (96.2%)** | **605 (92.8%)** |
| **vLLM** (no prefix cache) | reference | — | **1376** | **944** | **652** |

**Single-stream (concurrency 1) — output tok/s:**

| Step | in=128 | in=512 | in=1024 |
|---|---|---|---|
| R1 / R2 (INT8) | 67 | 38 | 24 |
| S3 (bf16) | 52 | 45 | 38 |
| S6 | 56 | 51 | 44 |
| S8 | 56 | 51 | 45 |
| S12 | 56 | 55 | 51 |
| **S14** | **56** | **55** | **51** |
| **vLLM** | **58** | **57** | **55** |

Reading the data: each optimization acts on the regime it targets (S6/decode-attn lifts in=128 most;
S7/S8/prefill-attn lift in=1024 most; S10/the KV-cliff scheduler fix moves only in=1024). The gap to
vLLM **shrinks as input gets shorter** (more decode-heavy): **97.5% at 128, 96.2% at 512, 92.8% at
1024** — the residual is prefill-side compute. (Baseline B0 = `main` INT8, conc=32/1024 = 107 tok/s ≈
16%; not re-run across inputs since it is a different precision/branch.) See the report for the
per-step analysis and the levers measured and ruled out (CUDA graphs, async scheduling, KV-cache
sizing, GEMM autotuning).
