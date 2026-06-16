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

`--model` takes either a Hugging Face repo id or a local directory. The simplest path is to pass the
repo id and let FlashQwen download it on first run:

```bash
./flashqwen chat --model Qwen/Qwen3-8B   # downloads to the Hugging Face cache, then runs
```

It fetches only the files it needs — `config.json`, `generation_config.json`, the tokenizer files,
and the safetensors shards — into the standard Hugging Face cache (`~/.cache/huggingface/hub`, or
`HF_HOME` / `HF_HUB_CACHE`), reusing anything already there and resuming interrupted downloads. Set
`HF_ENDPOINT` to use a mirror (e.g. `https://hf-mirror.com`), `HF_TOKEN` for gated or private repos,
and `HF_HUB_OFFLINE=1` to resolve from cache with no network call. Pin a revision with
`--model Qwen/Qwen3-8B@<branch-or-commit>`.

To use a model already on disk, pass its directory instead:

```bash
./flashqwen chat --model models/qwen3-8b
```

Either way the directory must contain `config.json`, the `*.safetensors` shards,
`model.safetensors.index.json`, `tokenizer.json`, `vocab.json`, `merges.txt`, and
`generation_config.json`. The engine reads these BF16 files directly and quantises the matmul weights
to INT8 at load time, and the Go side loads the tokenizer from
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

## How it works

### Bridging Go and C++

The compiled C++ engine binary is bundled into the Go binary with `//go:embed`. At startup the Go
supervisor writes it to a temp directory, picks a free localhost port, launches it as a child
process, polls `GetModel` until it answers, and sends it `SIGTERM` on exit — so there is one binary
to run and no engine process to manage by hand.

The two sides talk over gRPC, defined by `proto/engine.proto`. Only token ids cross the wire: Go
tokenises the prompt and sends `GenerateRequest{input_ids, max_tokens, temperature, top_p,
stop_token_ids}`; `Generate` is server-streaming, so the engine replies with a stream of `token_id`
events followed by a `Done{finish_reason, prompt_tokens, completion_tokens}`. `GetModel` reports the
engine's authoritative context window and vocab size. A client disconnect cancels the RPC, which the
engine turns into a sequence abort that frees the sequence's KV.

### Weight loading and INT8 quantisation

The engine mmaps the `.safetensors` shards and parses their header with RapidJSON, reading the
weights as BF16. At load time it quantises every matmul weight (the attention q/k/v/o projections,
the MLP gate/up/down, and `lm_head`) to INT8 with one scale per output row (symmetric, clamped to
[-127, 127]); embeddings and norms stay BF16. This roughly halves weight memory (~16 GB → ~9 GB).
Prefill dequantises the INT8 weights back to BF16 to feed the tensor cores; decode reads the INT8
bytes directly, which is what matters since decode is memory-bound.

### Prefill vs decode

A forward pass takes one of two paths. **Prefill** (many prompt tokens at once) is compute-bound, so
its matmuls run on the tensor cores via WMMA. **Decode** (one token per step) is memory-bound, so it
uses a vectorised INT8 GEMV; the fixed single-token kernel sequence is captured once into a CUDA
graph and replayed each step (the token id, position, and `past_len` live in device buffers, so the
graph stays valid as the context grows). Greedy decoding does the argmax on the GPU and returns just
one int; temperature / top-p sampling copies the logits to the host instead.

### Attention: flash-decoding split-K

Prefill attention assigns one warp per (head, query) and runs an online softmax entirely in
registers, with no block barriers. At decode time with batch 1 that scheme exposes only ~32 parallel
units (one per head), so each head's key range is split into `ATTN_SPLITS` (16) chunks computed by
separate blocks as partial online-softmaxes and then combined — about 16× more parallelism, which
keeps decode latency nearly flat as the context grows.

### Paged KV cache (PagedAttention)

Each layer's KV is a single `[num_blocks, BLOCK=16, kv_dim]` BF16 pool sized from the VRAM left after
weights and activations. A sequence owns a *block table* — a list of physical block ids — and a
logical position `p` maps to physical row `block_table[p/BLOCK]*BLOCK + p%BLOCK` (the addressing is
shared in `kv_cache.cuh`). Blocks are handed out on demand, so VRAM grows with the actual sequence
length and the number of concurrent sequences is decoupled from `max_ctx`: on a 24 GB 4090 the same
memory becomes one shared ~87k-token pool instead of a fixed per-sequence reservation. Three KV
kernels address through the block table: `store_kv_paged` (prefill and decode), `attention_paged`
(prefill), and the split-K `attention_decode_paged`.

### Continuous batching and preemption

The scheduler (`engine/src/scheduler.*`) keeps all `n_slots` busy: as soon as a slot frees it admits
a waiting request, and each iteration decodes the whole running set together. Prefill is chunked
(`PREFILL_CHUNK` = 256 tokens per iteration) and interleaved with decode, so a long prompt never
stalls the running sequences. When the block pool runs dry it preempts the youngest running sequence
— freeing its blocks and requeuing it — and recomputes its KV from prompt + output on resume (the
vLLM strategy), which keeps the engine correct under memory pressure rather than deadlocking. Each
request carries its own sampling parameters; an all-greedy batch uses the GPU argmax, and a single
sampling request makes the batch fall back to host-side per-row sampling.

The staged history behind these — the single-stream optimisation journey (scalar matmul → WMMA →
split-K → INT8, 49.7 → 104 tok/s) and the batching work — lives on the `optimization-study` and
`feature/batching` branches.

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
