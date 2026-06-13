# FlashQwen

从零实现、**零外部依赖的 Qwen3-8B C++/CUDA 推理引擎**(没有 PyTorch、没有 cuBLAS、没有
tokenizers),主要面向学习。支持多轮流式对话,以及 benchmark 模式(TTFT / TPOT / tok/s)。

English: [README.md](README.md)

---

## 用法

### 获取模型

用 `git clone` 把模型仓库拉到一个目录,然后把 `--model` 指向它:

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-8B models/qwen3-8b
```

这个目录需要包含 `config.json`、`*.safetensors` 分片、`model.safetensors.index.json`、
`vocab.json` 和 `merges.txt`。权重按原样使用(BF16,不转换、不量化)。

### 编译

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

需要 CUDA toolkit(建议 12.x)和一块 ≥20 GB 显存的 NVIDIA GPU。默认按 `sm_89`(RTX 4090 /
Ada)编译;其他显卡用 `-DCMAKE_CUDA_ARCHITECTURES=<arch>` 覆盖(如 Hopper 用 `90`)。

### 运行

`--model` 是**必填项**,指向模型目录(比如上面建好的 `models/qwen3-8b`;直接传 HF hub
缓存目录也可以)。`--help` 会列出支持的模型,并扫描本地 hub 缓存看有哪些可用。

```bash
# 交互式多轮对话(默认模式),KV cache 跨轮保留
./build/flashqwen --model models/qwen3-8b

# 用 Qwen 推荐的采样参数对话
./build/flashqwen --model models/qwen3-8b --temperature 0.6 --top-p 0.95 --top-k 20

# benchmark(内置的固定输入长度扫描)
./build/flashqwen benchmark --model models/qwen3-8b

./build/flashqwen --help
```

**模式:** 不带子命令(或写 `chat`)→ 交互对话;`benchmark` → 跑指标。

**对话内命令:** `/exit`  `/quit`  `/reset`(清空上下文)  `/think on|off`。

**常用参数:** `--max-ctx N`(KV cache 大小,默认 4096)、`--temperature`、`--top-p`、
`--top-k`、`--seed`、`--think`(开启 Qwen3 思考模式)。

**支持的模型:** 任意 dense 的 Qwen3 模型(架构为 `Qwen3ForCausalLM`):Qwen3-0.6B / 1.7B /
4B / 8B / 14B / 32B,各维度从 `config.json` 读取。**不支持:** Qwen3.5(混合线性注意力 +
多模态)、Qwen3 MoE 变体,以及非 Qwen 架构。

> 首次启动会从磁盘加载 ~16 GB 权重(网络盘上较慢);之后命中操作系统页缓存,几秒就能启动。

---

## 本地 Benchmark

**硬件:** NVIDIA GeForce RTX 4090(24 GB)· 驱动 580.76.05(CUDA 13.0)· 用 CUDA 12.8 编译(原生 sm_89)
**模型:** Qwen3-8B,BF16 权重,FP32 激活(prefill 的 matmul 用 tensor core,BF16 计算)。
**方法:** 单流(batch=1)、贪心、固定 128 token 输出(忽略 EOS)、1 次 warmup + 3 次测量取
中位数。扫描逻辑内置 —— 直接 `flashqwen benchmark --model <DIR>` 即可。

按输入长度扫描(合成 prompt),输出固定 128 token:

| 输入 tok | TTFT | TPOT | decode tok/s | output tok/s | peak tok/s |
|---:|---:|---:|---:|---:|---:|
| 16   | 60 ms  | 20.2 ms | 49.6 | 48.5 | 52.9 |
| 128  | 89 ms  | 22.4 ms | 44.7 | 43.4 | 47.4 |
| 512  | 0.42 s | 30.0 ms | 33.3 | 30.1 | 34.8 |
| 1024 | 1.23 s | 40.1 ms | 24.9 | 20.1 | 25.7 |

**指标定义:**
- **TTFT** — time to first token = 整个 prompt 的 prefill + 采样第 1 个 token。
- **TPOT** — time per output token,decode 阶段(第 1 个之后)每 token 的平均延迟。
- **decode 吞吐** — 仅 decode 阶段的 tok/s。
- **output 吞吐** — 含 prefill 的 tok/s(`n_out / total_time`)。
- **peak output 吞吐** — `1 / 最快单 token 延迟`。

**怎么看这些数字。** decode 是访存瓶颈:每个 token 都要把约 16 GB 权重完整读一遍,所以
TPOT ≈ 20 ms ≈ 50 tok/s ≈ 16 GB × 50 ≈ 800 GB/s,接近 4090 约 1 TB/s 的带宽——单序列
decode 已接近硬件上限。prefill 用的是 **tensor core(WMMA)GEMM**,所以即使输入很长 TTFT
依然很低(1024 token 约 1.2 s,而之前的标量 kernel 要 ~13 s)。TPOT 随上下文变长仍会略升
(20 → 40 ms),因为每步 attention 要扫描更长的 KV cache——现在主要待优化的是它,而不是 prefill。

---

## 依赖的库 & 代码结构

### 依赖——没有任何第三方库

只用了 **C++ 标准库**、**CUDA Runtime**(toolkit 自带,不算第三方库)和 **POSIX** 系统调用。
`CMakeLists.txt` 里没有任何 `target_link_libraries`。

由于没有任何依赖需要编译,一次干净编译(`-j8`)约 **9 秒**,产物二进制约 **1.8 MB**
(strip 后 1.6 MB)。模型权重是运行时从磁盘加载的,不打进二进制里。

| 通常要用的库 | 这里的做法 |
|---|---|
| nlohmann/json、rapidjson | 手写 `src/json.hpp` |
| HF tokenizers / sentencepiece | 手写 byte-level BPE `src/tokenizer.*` |
| safetensors C++ 库 | 手写 `src/safetensors.*`(mmap + 解析头) |
| cuBLAS / CUTLASS | 手写 matmul(prefill 用 tensor core WMMA,decode 用 GEMV) |
| PyTorch(任意深度学习框架) | 完全没用 |

- C++ 标准库:`<vector> <string> <unordered_map> <chrono> <random> <cmath> <fstream>` 等
- CUDA:`<cuda_runtime.h>`、`<cuda_bf16.h>` —— **没有** cuBLAS / cuDNN / cuRAND / Thrust
- POSIX:`<sys/mman.h>`(mmap)、`<fcntl.h>`、`<unistd.h>`

### 代码结构与行数

每个文件只做一件事;`main.cpp` 是个很薄的入口,只负责解析参数和分发。总共约 1690 行。

**应用层**

| 文件 | 作用 | 行数 |
|---|---|---:|
| `src/main.cpp` | 入口:参数解析 + 分发 | 69 |
| `src/cli.cpp` / `.hpp` | 架构检查 + `--help` | 63 |
| `src/chat.cpp` / `.hpp` | 交互式多轮对话 | 55 |
| `src/benchmark.cpp` / `.hpp` | benchmark 模式(扫描输入长度) | 92 |
| `src/generate.cpp` / `.hpp` | 共用的 prefill + decode 循环 | 66 |
| `src/sampler.cpp` / `.hpp` | 采样(greedy / temp / top-k / top-p) | 50 |

**核心引擎**

| 文件 | 作用 | 行数 |
|---|---|---:|
| `src/model.cu` / `.hpp` | 权重加载 + 前向 + KV cache | 217 |
| `src/kernels.cu` / `.cuh` | CUDA kernel(WMMA/GEMV matmul、rmsnorm、rope、attention…) | 295 |
| `src/tokenizer.cpp` / `.hpp` | byte-level BPE 编码/解码 | 394 |
| `src/safetensors.cpp` / `.hpp` | mmap + 解析 `.safetensors` 头 | 105 |
| `src/json.hpp` | 极简 JSON 解析器 | 203 |
| `src/config.hpp` | 解析 `config.json` | 45 |
| `CMakeLists.txt` | 构建 | 35 |

值得一提:那些"通常靠库"的胶水代码——分词器(394)+ JSON(203)——约 600 行,占了快一半。
真正的神经网络部分——CUDA kernel(295)+ 前向(217)——约 510 行,因为 Qwen3 是结构
非常规整的 dense transformer。

## 优化历程

matmul / decode 路径分阶段优化过。每个阶段都是 **`optimization-study` 分支上的一个 tag
commit**,可以 checkout 出来重新测。**阶段 0(标量 matmul)是 baseline**,下面所有都跟它对比。

测试环境 RTX 4090(Qwen3-8B,BF16)——单流、batch 1、贪心、输出固定 128 token、扫描输入长度、
取 3 次中位数:

| 阶段 | 分支 tag | TTFT@128 | TTFT@1024 | TPOT@1024 | decode@16 (tok/s) |
|---|---|---:|---:|---:|---:|
| **0 · 标量 matmul(baseline)** | `bench-0-scalar` | 1531 ms | 12585 ms | 40.1 ms | 49.7 |
| 1 · Tensor Core (WMMA) prefill | `bench-1-wmma` | 89 ms | 1234 ms | 40.1 ms | 49.7 |
| 2 · BF16 KV cache | `bench-2-bf16kv` | 89 ms | 1233 ms | 40.0 ms | 49.7 |
| 3 · warp attention(无 barrier) | `bench-3-attn` | 83 ms | 908 ms | 39.4 ms | 49.8 |
| 4 · 向量化 GEMV decode | `bench-4-gemv` | 83 ms | 907 ms | 38.8 ms | 51.1 |

相比 baseline,阶段 4 的 **prefill 快了约 14–18×**,decode 快约 3%。复现任一阶段:

```bash
git checkout bench-1-wmma     # 或 bench-0-scalar … bench-4-gemv
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
./build/flashqwen benchmark --model models/qwen3-8b
```

**阶段 0 — 标量 matmul(baseline),`bench-0-scalar`。** 每个 matmul(prefill 和 decode 都是)
都用"一个 warp 算一个输出元素"的点积,没有 tensor core。prefill 很惨:1024 token 的 prompt
要 ~12.6 s 才出第一个 token,因为 prefill 的 GEMM 是计算瓶颈又没用 tensor core。

**阶段 1 — Tensor Core (WMMA) prefill,`bench-1-wmma`。** prefill(多 token)把激活转成 BF16,
用 WMMA(16×16×16、FP32 累加)在 tensor core 上算 matmul;decode(单 token)保留 GEMV。
**prefill 暴降 ~14×(TTFT@1024 12585 → 1234 ms)。** decode 不动——它是访存瓶颈,tensor core 帮不上。

**阶段 2 — BF16 KV cache,`bench-2-bf16kv`。** KV cache 从 FP32 改存 BF16。这一步速度没变——
因为此时 attention kernel 是延迟瓶颈(每个 key 串行 `__syncthreads` 归约),不是带宽瓶颈——
但 cache 减半(4096 ctx 下 ~1.2 GB → ~0.6 GB),即最大上下文 ~2×。字节减半的收益要到阶段 3 才兑现。

**阶段 3 — warp attention,`bench-3-attn`。** attention 从"一个 block 管一个 (head,query)、每个
key 做块内归约"改成 **一个 warp 管一个 (head,query)**:每个 lane 负责 4 个维度,每个 key 的 q·k
用 warp shuffle 归约,online softmax 全在寄存器里——没有 barrier。**prefill attention 降 ~26%
(TTFT@1024 1233 → 908 ms)。** decode TPOT 基本持平:batch 1 时只有 32 个并行单元(每 head 一个),
所以 decode attention 是并行度瓶颈,不是单 key 成本瓶颈(真要快得上 flash-decoding split-K)。

**阶段 4 — 向量化 GEMV decode,`bench-4-gemv`。** decode 的 GEMV 改成每步用 16 字节向量化加载
读 8 个元素,而不是一次一个标量 BF16。提升不大(decode@16 49.8 → 51.1 tok/s)——标量版本已经到
~80% 显存带宽了。

**仍待优化:** decode attention 的 flash-decoding split-K,以及权重量化(INT8/INT4)来突破 decode
的带宽天花板(decode 每 token 要读完整 ~16 GB 权重,所以 bf16 下 ~60 tok/s 就是上限)。
