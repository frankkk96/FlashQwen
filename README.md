# FlashQwen

A from-scratch **Qwen3-8B inference stack** (C++/CUDA + Go).

> **FlashQwen deliberately stays small, trading breadth of model support for a clean codebase.**
> We believe that in the age of AI-assisted programming, giving the AI clean context and a tight
> feedback loop is worth more than piling on dependencies, abstractions, adapters, and compromises.
> So there is no general framework and no heavy dependencies here — just one narrow, complete
> inference path that is easy to read and easy to run.

中文: [README.zh-CN.md](README.zh-CN.md)

The project has two layers that communicate over gRPC:

- **The outer layer is a Go application** (the root module; it builds to `./flashqwen`). It handles
  everything text-related: tokenisation, rendering the Qwen3 ChatML template, detecting tool calls
  in the token stream, and serving both an OpenAI-compatible HTTP API and an interactive CLI. It
  bundles the compiled C++ engine into itself via `//go:embed`, then at runtime extracts it to a
  temp directory and launches it as a child process — so you start this one binary and never manage
  an engine process yourself.
- **The inner layer is a C++/CUDA token engine** (`engine/`). It does all the GPU work: loading
  weights, prefill, decode, and sampling. It only accepts token ids and returns sampled token ids —
  it does no tokenisation and knows nothing about text formats. Its model stack depends on no
  machine-learning library (no PyTorch, cuBLAS, or HuggingFace tokenizers); INT8 weight
  quantisation, continuous batching, and PagedAttention are all implemented from scratch.

---

## Usage

### 1. Get the model

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-8B models/qwen3-8b
```

The model directory must contain `config.json`, the `*.safetensors` shards,
`model.safetensors.index.json`, `tokenizer.json`, `vocab.json`, `merges.txt`, and
`generation_config.json`. The engine reads these BF16 files directly and quantises the matmul
weights to INT8 at load time, and the Go side loads the tokenizer from
`tokenizer.json` / `vocab.json` / `merges.txt` — so no offline conversion or repacking is needed.

### 2. Build

```bash
make            # build the C++ engine → embed → go build, producing ./flashqwen
```

`make` runs three steps in a fixed order: cmake builds the C++ engine, the engine binary is copied
into `internal/supervisor/bin/` (so `//go:embed` can bundle it), and `go build` produces the final
binary. Building needs a CUDA toolkit (12.x recommended), Go 1.26+, gRPC/protobuf, and an NVIDIA GPU
with ≥20 GB of memory. It targets `sm_89` (RTX 4090 / Ada) by default; for other cards set
`-DCMAKE_CUDA_ARCHITECTURES=<arch>` before `make engine`. The final binary is a single file, but at
runtime it still needs the CUDA and libgrpc++ shared libraries on the host (`//go:embed` bundles the
executable, not its `.so` dependencies).

### 3. Run

Each subcommand extracts and launches the embedded engine itself; `--model` is required:

```bash
./flashqwen serve     --model models/qwen3-8b   # start the OpenAI-compatible server (default :8000)
./flashqwen chat      --model models/qwen3-8b   # enter an interactive multi-turn chat
./flashqwen benchmark --model models/qwen3-8b   # run the end-to-end throughput benchmark (1/8/16)
./flashqwen --help
```

- **In-chat commands:** `/exit` and `/quit` leave, `/reset` clears the context, `/think on|off`
  toggles thinking mode.
- **Common flags:** `--max-ctx N` sets the KV cache size (the max context length), `--slots N` sets
  the max number of concurrent sequences, `--addr` sets the serve HTTP listen address; sampling
  params such as temperature and top_p are passed per request through the OpenAI API.
- **OpenAI endpoints:** `POST /v1/chat/completions` (streaming, non-streaming, and tools),
  `GET /v1/models`, `GET /healthz`.
- **Supported models:** any dense Qwen3 (architecture `Qwen3ForCausalLM` in `config.json`, with
  dimensions read from that file). Qwen3 MoE, multimodal models, and non-Qwen architectures are not
  supported.

---

## Code structure

The packages of the outer Go application (root module):

| Path | Responsibility | Lines |
|---|---|---:|
| `cmd/flashqwen/` | program entry point; dispatches the `serve` / `chat` / `benchmark` subcommands | 346 |
| `internal/engine/` | talks to the inner engine: a low-level token-stream gRPC client, and a high-level text `Generate` (render → tokenise → drive the engine → decode); the gRPC stubs `enginepb` are used only here | 225 |
| `internal/server/` | exposes the engine as an OpenAI-compatible HTTP service — OpenAI request/response decode/encode and SSE streaming | 293 |
| `internal/chatml/` | the bidirectional Qwen3 ChatML codec: renders messages into a prompt, and decodes the engine's token stream into text while detecting tool calls | 266 |
| `internal/tokenizer/` | a from-scratch byte-level BPE tokenizer; reads vocab / merges / tokenizer.json | 483 |
| `internal/supervisor/` | embeds the engine binary via `//go:embed`; extracts, launches, health-checks, and tears it down | 78 |

The source files of the inner C++/CUDA token engine (`engine/src/`), by descending line count:

| File | Responsibility | Lines |
|---|---|---:|
| `kernels.*` | all CUDA kernels: INT8 GEMV, WMMA matmul, attention, split-K decode | 529 |
| `model_runtime.*` | loads weights and quantises them to INT8 in VRAM; runs the forward pass | 435 |
| `scheduler.*` | the continuous-batching scheduler: chunked prefill, sequence admission, and preemption | 215 |
| `grpc_server.*` | the token-level gRPC service: takes token ids, streams back sampled token ids | 181 |
| `kv_cache.*` | the paged KV pool for PagedAttention | 127 |
| `safetensors.*` | mmaps and parses the `.safetensors` header with RapidJSON | 111 |
| `model_spec.hpp` | reads model dimensions and architecture from `config.json` (declarative, no GPU) | 64 |
| `sampler.*` | sampling: greedy, temperature, top-p | 54 |
| `main.cpp` / `args.*` | the engine process entry point and CLI argument parsing | 88 |

The interface between the two layers is defined by `proto/engine.proto`. The full single-stream
optimisation journey (scalar matmul → WMMA → split-K → INT8, 49.7 → 104 tok/s) and the staged
batching implementation live on the `optimization-study` and `feature/batching` branches.

---

## Benchmark

The table below is the end-to-end throughput measured by `./flashqwen benchmark`, over the real path
through gRPC and including tokenisation and sampling.

**Setup:** NVIDIA RTX 4090 (24 GB), Qwen3-8B with INT8-quantised matmul weights, continuous batching
+ PagedAttention.
**Config:** default flags — synthetic requests of 128 input / 128 output tokens, 32 requests per
concurrency level, greedy decoding.

| Concurrency | Requests | Wall | Aggregate throughput | Mean TTFT |
|---:|---:|---:|---:|---:|
| 1  | 32 | 44.9 s | 91 tok/s  | 99 ms  |
| 8  | 32 | 18.0 s | 228 tok/s | 286 ms |
| 16 | 32 | 16.7 s | **245 tok/s** | 647 ms |

Single-stream is about 91 tok/s: decode is memory-bound (every generated token reads the whole
model), and INT8 weights raise that ceiling to roughly this level. At concurrency 16, continuous
batching amortises one weight read across the sequences in a batch, lifting aggregate throughput to
about 245 tok/s (~2.7×); the cost is that time-to-first-token rises under load.
