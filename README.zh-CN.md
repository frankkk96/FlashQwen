# FlashQwen

从零实现、**零外部依赖的 Qwen3-8B C++/CUDA 推理引擎**(没有 PyTorch、没有 cuBLAS、没有
tokenizers),主要面向学习。支持多轮流式对话,以及 benchmark 模式(TTFT / TPOT / tok/s)。

English: [README.md](README.md)

---

## 用法

### 获取模型

权重**直接用从 HuggingFace 拉下来的原始文件**,不做任何转换、量化或重打包。用官方工具下载
Qwen3-8B:

```bash
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-8B
```

下载后会得到一个 HF hub 缓存目录(默认 `~/.cache/huggingface/hub/models--Qwen--Qwen3-8B`),
把这个路径传给 `--model` 即可。FlashQwen 直接从里面读取 `config.json`、`*.safetensors`
分片、`vocab.json` 和 `merges.txt`。

### 编译

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

需要 CUDA toolkit(建议 12.x)和一块 ≥20 GB 显存的 NVIDIA GPU。默认按 `sm_89`(RTX 4090 /
Ada)编译;其他显卡用 `-DCMAKE_CUDA_ARCHITECTURES=<arch>` 覆盖(如 Hopper 用 `90`)。

### 运行

`--model` 是**必填项**——可以是带 `config.json` 的 HF snapshot 目录,或像
`.../models--Qwen--Qwen3-8B` 这样的 HF hub 缓存目录。`--help` 会列出支持的模型,并扫描
本地 hub 缓存看有哪些可用。

```bash
# 交互式多轮对话(默认模式),KV cache 跨轮保留
./build/flashqwen --model /path/to/models--Qwen--Qwen3-8B

# 用 Qwen 推荐的采样参数对话
./build/flashqwen --model <DIR> --temperature 0.6 --top-p 0.95 --top-k 20

# benchmark(内置的固定输入长度扫描)
./build/flashqwen benchmark --model <DIR>

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
- POSIX:`<sys/mman.h>`(mmap)、`<fcntl.h>`、`<unistd.h>`、`<dirent.h>`

### 代码结构与行数

每个文件只做一件事;`main.cpp` 是个很薄的入口,只负责解析参数和分发。总共约 1720 行。

**应用层**

| 文件 | 作用 | 行数 |
|---|---|---:|
| `src/main.cpp` | 入口:参数解析 + 分发 | 73 |
| `src/cli.cpp` / `.hpp` | 模型路径解析、架构检查、`--help` | 117 |
| `src/chat.cpp` / `.hpp` | 交互式多轮对话 | 55 |
| `src/benchmark.cpp` / `.hpp` | benchmark 模式 | 52 |
| `src/generate.cpp` / `.hpp` | 共用的 prefill + decode 循环 | 66 |
| `src/sampler.cpp` / `.hpp` | 采样(greedy / temp / top-k / top-p) | 50 |

**核心引擎**

| 文件 | 作用 | 行数 |
|---|---|---:|
| `src/model.cu` / `.hpp` | 权重加载 + 前向 + KV cache | 219 |
| `src/kernels.cu` / `.cuh` | CUDA kernel(WMMA/GEMV matmul、rmsnorm、rope、attention…) | 270 |
| `src/tokenizer.cpp` / `.hpp` | byte-level BPE 编码/解码 | 394 |
| `src/safetensors.cpp` / `.hpp` | mmap + 解析 `.safetensors` 头 | 105 |
| `src/json.hpp` | 极简 JSON 解析器 | 203 |
| `src/config.hpp` | 解析 `config.json` | 45 |
| `CMakeLists.txt` | 构建 | 35 |

值得一提:那些"通常靠库"的胶水代码——分词器(394)+ JSON(203)——约 600 行,占了快一半。
真正的神经网络部分——CUDA kernel(270)+ 前向(219)——约 490 行,因为 Qwen3 是结构
非常规整的 dense transformer。
