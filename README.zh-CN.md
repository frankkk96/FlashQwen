# FlashQwen

一个从零实现的 **Qwen3-8B 推理栈**(C++/CUDA + Go)。

> **FlashQwen 刻意保持小巧,用"支持的模型范围"换取一份干净的代码库。**
> 我们相信在 AI 辅助编程的时代,给 AI 干净的上下文和紧凑的反馈闭环,比堆砌依赖、抽象、适配器更有价值。
> 所以这里没有通用框架 —— 只有一条狭窄、完整、易读易跑的推理路径。

English: [README.md](README.md)

---

## 1. 项目介绍与版本演变

FlashQwen 经历了三个阶段,每个阶段都是一个目标明确的独立版本,合起来勾勒出从"单文件教学引擎"到"对标
vLLM 的服务运行时"的演进路径。

### 第一阶段 —— 从零实现的终端引擎
最早的版本(在 `main` 分支):一个单独的 C++/CUDA 程序,**代码量不到 2000 行、零依赖**(无 PyTorch、
无 cuBLAS、无 HuggingFace tokenizer)。直接从 `.safetensors` 读取 Qwen3-8B 权重,跑 prefill + decode +
采样,完全在**终端**里运行。这一阶段还包含单流优化研究(标量 matmul → tensor-core WMMA → flash-decoding
split-K → INT8 权重量化)。

### 第二阶段 —— 调度 + 标准 OpenAI API 服务
加入了真正的服务层:C++ 引擎里实现连续批处理 + PagedAttention,外面套一层 **Go/Gin 网关、通过 gRPC
对外提供标准 OpenAI API**。从这一阶段起,你启动的是一个*服务*(`flashqwen serve`),用任意 OpenAI 客户端
访问,而不再是跑一次性的终端程序。

### 第三阶段 —— 服务性能优化(当前)
当前重点:在 **bf16 对齐(不量化)** 的前提下,把服务吞吐追平 **vLLM**。通过一系列内核与调度重写(bf16 +
cuBLAS GEMM、FlashAttention-2 prefill、`mma.sync` FlashAttention 风格的 prefill-attention kernel、GQA
共享的 FlashDecoding、统一 token 预算调度器、自动 prefix caching),FlashQwen 目前在**关闭 prefix caching**
下、单张 RTX 4090 上达到 **vLLM 约 95–98% 的服务吞吐**,在**真实 ShareGPT 流量上约 97%**,相比自身第一/
二阶段基线约 **6×**。完整报告见 [`docs/optimization.md`](docs/optimization.md)。

### 架构(第二阶段起)

两层,通过 gRPC 通信:

- **外层 —— Go 应用**(根模块,产物 `./flashqwen`):负责所有与文本相关的事 —— 分词、Qwen3 ChatML 模板、
  工具调用识别,并对外提供 OpenAI 兼容 HTTP API 与交互式 CLI。它通过 `//go:embed` 把编译好的 C++ 引擎打包
  进自身,运行时释放并作为子进程拉起 —— 所以你只需运行一个二进制。
- **内层 —— C++/CUDA token 引擎**(`engine/`):承担全部 GPU 工作 —— 权重加载、prefill、decode、采样、
  分页 KV cache、连续批处理调度。它只接收 token id、返回采样的 token id,不做分词、不懂文本格式。

---

## 2. 使用指南

### 获取模型

`--model` 指向一个本地目录;FlashQwen 自身不下载任何东西:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-8B --local-dir models/qwen3-8b
```

(需要镜像就先设 `HF_ENDPOINT=https://hf-mirror.com`。)目录里需包含 `config.json`、`*.safetensors`
分片 + `model.safetensors.index.json`,以及 tokenizer 文件(`tokenizer.json`、`vocab.json`、
`merges.txt`、`generation_config.json`)。引擎直接读 BF16 safetensors,无需离线转换或重打包。

### 编译

```bash
make            # 编译 C++ 引擎 → 嵌入 Go → go build,产出 ./flashqwen
```

`make` 三步:cmake 编译 C++ 引擎、把引擎二进制拷进 `internal/supervisor/bin/`(供 `//go:embed`)、
`go build` 产出最终二进制。需要 CUDA 工具链(12.x)、Go 1.26+、gRPC/protobuf、**cuBLAS**,以及一张
≥20 GB 显存的 NVIDIA GPU。默认目标 `sm_89`(RTX 4090 / Ada);其它卡在 `make engine` 前设
`-DCMAKE_CUDA_ARCHITECTURES=<arch>`。

### 运行

```bash
./flashqwen serve --model models/qwen3-8b    # 启动 OpenAI 兼容服务(默认 :8000)
./flashqwen chat  --model models/qwen3-8b    # 进入交互式多轮对话
./flashqwen --help
```

- **OpenAI 接口:** `POST /v1/chat/completions`(流式/非流式/工具)、`GET /v1/models`、`GET /healthz`。
- **常用参数:** `--max-ctx N`(KV/上下文长度)、`--slots N`(最大并发序列数)、`--max-batch-tokens N`
  (每步计算的 token 数)、`--addr`(监听地址)。采样参数(temperature、top_p)按请求传入。
- **对话内命令:** `/exit` `/quit` 退出,`/reset` 清空上下文,`/think on|off` 切换思考模式。
- **支持的模型:** dense Qwen3(`Qwen3ForCausalLM`)。MoE / 多模态 / 非 Qwen 架构不支持。

---

## 3. 阶段细节与特点

| 阶段 | 主题 | 特点 | 分支 | 提交 |
|---|---|---|---|---|
| **1** | 从零实现的终端引擎 | 单个 C++/CUDA 二进制,<2000 行,**零依赖**,终端运行。INT8 权重、CUDA-graph decode、flash-decoding split-K。单流研究。 | `main` | `7d6764c`(初始)→ `89318c5` |
| **2** | 批处理 + OpenAI API 服务 | 连续批处理 + PagedAttention;Go/Gin **OpenAI 兼容 API**(gRPC)。启动为服务。 | `main` | `566fac6` → `08392df` |
| **3** | 服务性能优化 | bf16 + cuBLAS GEMM、FlashAttention-2 / `mma.sync` prefill、GQA 共享 FlashDecoding、统一 token 预算调度、自动 prefix caching。**vLLM 约 95–98%**(关 prefix cache)、**真实 ShareGPT 约 97%**,bf16 对齐。 | `feat/prefix-caching` | `08392df`(基线)→ `HEAD` |

### 第三阶段 —— 优化步骤(分支 `feat/prefix-caching`)

每一步改了什么、针对哪个瓶颈、实测效果、以及试过的死路,详见
[`docs/optimization.md`](docs/optimization.md)。每个落地步骤都干净重建,在同一台机器上分别测
**128 / 512 / 1024 输入**(输出 128),对比**功能对齐的 vLLM**(`--no-enable-prefix-caching`,bf16,0.9 显存)。

**饱和吞吐(并发 32)—— 输出 tok/s(占 vLLM 百分比):**

| 步骤 | 改动 | 提交 | in=128 | in=512 | in=1024 |
|---|---|---|---|---|---|
| R1 | INT8 统一 token 预算调度 + GPU 采样 | `5065b2e` | 248 (18.0%) | 139 (14.7%) | 86 (13.2%) |
| S3 | bf16 权重 + cuBLAS GEMM + FlashAttention-2 | `7650654` | 1094 (79.5%) | 628 (66.5%) | 348 (53.4%) |
| S5 | 融合 QKV / gate-up GEMM | `c2f98a0` | 1141 (82.9%) | 648 (68.6%) | 356 (54.6%) |
| S6 | FlashDecoding decode 注意力(按请求类型拆分) | `f0b1499` | 1317 (95.7%) | 824 (87.3%) | 454 (69.6%) |
| S7 | WMMA tensor-core prefill 注意力 | `0dd4010` | 1326 (96.4%) | 860 (91.1%) | 501 (76.8%) |
| S8 | prefill 注意力占用率 | `392cda5` | 1328 (96.5%) | 878 (93.0%) | 531 (81.5%) |
| S10 | 调度器:max-batch-tokens 2048→1024 | `642cc28` | 1334 (97.0%) | 882 (93.4%) | 581 (89.1%) |
| S12 | GQA 共享 FlashDecoding(每组 K/V 只读一次) | `240aaa1` | 1338 (97.2%) | 906 (96.0%) | 604 (92.7%) |
| S14 | activation 按需分配 + KV 池/越界修复 | `e5a99c8` | 1341 (97.5%) | 908 (96.2%) | 605 (92.8%) |
| S15 | 自动 prefix caching(内容哈希 KV 复用) | `fad12b7` | ≈S14* | ≈S14* | ≈S14* |
| **S16** | **prefill 注意力重写:WMMA → `mma.sync`(FlashAttention 风格)** | `feat/prefix-caching` | **1318 (94.6%)** | **927 (97.7%)** | **640 (98.1%)** |
| **vLLM**(无 prefix cache) | 参照 | — | **1376 / 1393** | **944 / 948** | **652** |

\* S15(prefix caching)在纯随机输入上中性(跨请求无公共前缀);只有在有共享前缀的负载上才有收益
(**512-token 共享前缀 +36%**,与 vLLM 的 +37% 相当)。S16 行是同会话 A/B(其 WMMA 对照复现了 S14 行),
第二个 vLLM 数字是该会话的参照。

**单流(并发 1):** 跨输入约为 vLLM 的 ~92–98%(`S14`/`S16`:56 / 55 / 51 vs vLLM 58 / 57 / 55 tok/s,
对应 128 / 512 / 1024);prefill 重写在 conc=1 基本持平(此处 prefill-attn 占比极小)。

怎么读这组数:每个优化都精确作用在它针对的 regime(S6/decode 注意力对 in=128 抬升最大;S7/S8/prefill 注意力
对 in=1024 抬升最大;S10/KV 悬崖的调度修复只动 in=1024)。到 S14 为止差距随输入长度单调(97.5% → 92.8%)——
残余在 prefill 侧 compute;**S16 的 `mma.sync` prefill 重写正是攻这一点,把 1024 从 92.8% → 98.1%**,曲线现在
几乎拉平。报告里有逐步分析和被实测排除的各条杠杆(CUDA graph、异步调度、KV cache 容量、GEMM autotuning)。

### 超出历程之外 —— 综合对比与真实数据

bf16 parity 下的宽口径对比(完整表见 [`docs/optimization.md`](docs/optimization.md)):

- **输入长度**(out=128, conc=32;2048 用 conc=16):128 / 512 / 1024 / 2048 = **97.1% / 98.3% / 97.8% / 91.9%**
  —— 长上下文 prefill(2048)是剩下的相对短板。
- **输出长度**(in=512, conc=32):out=128 98.3%、out=512 96.1%、**out=1024 107%** —— 长输出、decode 密集时
  FlashQwen 比 vLLM *更快*(其极低的 TTFT 在长生成里累积)。
- **真实数据 —— 1000 条 ShareGPT 对话**(conc=32,等输出预算):**vLLM 的 96.8%**(1286 vs 1328 tok/s)。
  **TTFT 更低**(94.6 vs 138.8 ms,vLLM 把排队的 chunked-prefill 前置),**TPOT 高约 4%**。Prefix caching 在
  ShareGPT 上对两边**都中性**(跨请求重叠≈0)—— 只有负载有真实共享前缀(多轮、RAG、复用 system prompt)时才有收益。

(基线 B0 = `main` INT8,conc=32/1024 ≈ 107 tok/s ≈ 16%;异精度/异分支,未跨输入重测。)
