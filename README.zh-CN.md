# FlashQwen

从零实现的 **Qwen3-8B 推理栈**(C++/CUDA + Go)。

> **FlashQwen 刻意保持精简,以模型适配的广度换取代码的干净。** 我们相信:在 AI 编程的时代,
> 为 AI 提供干净的上下文与高效的反馈循环,比堆砌复杂的依赖、抽象、适配层与妥协更有价值。
> 因此这里没有通用框架、没有重型依赖,只有一条窄而完整、读得懂也跑得动的推理路径。

English: [README.md](README.md)

该项目分为两层,二者通过 gRPC 通信:

- **外层是 Go 应用**(根模块,编译产物 `./flashqwen`),负责一切与文本相关的工作:分词、渲染
  Qwen3 ChatML 模板、在 token 流中识别工具调用,并对外提供 OpenAI 兼容 HTTP API 和交互式 CLI。
  它通过 `//go:embed` 把编译好的 C++ 引擎打包进自身,运行时释放到临时目录、作为子进程拉起,所以
  使用者只需启动这一个二进制,无需单独管理引擎进程。
- **内层是 C++/CUDA token 引擎**(`engine/`),负责 GPU 上的全部计算:加载权重、prefill、decode、
  采样。它只接收 token id、返回采样出的 token id,不做分词、不认识任何文本格式。它的模型栈不依赖
  任何机器学习库(没有 PyTorch、cuBLAS、HuggingFace tokenizers),INT8 权重量化、连续批处理、
  PagedAttention 都是自己实现的。

---

## 用法

### 1. 获取模型

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-8B models/qwen3-8b
```

模型目录需包含 `config.json`、`*.safetensors` 分片、`model.safetensors.index.json`、`tokenizer.json`、
`vocab.json`、`merges.txt`、`generation_config.json`。引擎直接读这些 BF16 文件、在加载时把 matmul 权重
量化成 INT8,Go 侧从 `tokenizer.json`/`vocab.json`/`merges.txt` 加载分词器,因此无需任何离线转换或重打包。

### 2. 编译

```bash
make            # 编译 C++ 引擎 → 嵌入 → go build,产出 ./flashqwen
```

`make` 按固定顺序执行三步:cmake 编译 C++ 引擎、把引擎二进制拷进 `internal/supervisor/bin/`(供
`//go:embed` 打包)、`go build` 产出最终二进制。编译需要 CUDA toolkit(建议 12.x)、Go 1.26+、
gRPC/protobuf,以及一块显存 ≥20 GB 的 NVIDIA GPU。默认按 `sm_89`(RTX 4090 / Ada 架构)编译,其他
显卡需在 `make engine` 前设置 `-DCMAKE_CUDA_ARCHITECTURES=<arch>`。最终二进制虽是单文件,运行时仍
需目标机器装有 CUDA 与 libgrpc++ 共享库(`//go:embed` 进去的是可执行文件,不含其 `.so` 依赖)。

### 3. 运行

三个子命令各自会释放并拉起内嵌的引擎,`--model` 为必填项:

```bash
./flashqwen serve     --model models/qwen3-8b   # 启动 OpenAI 兼容服务(默认监听 :8000)
./flashqwen chat      --model models/qwen3-8b   # 进入交互式多轮对话
./flashqwen benchmark --model models/qwen3-8b   # 跑端到端吞吐基准(并发 1/8/16)
./flashqwen --help
```

- **对话内命令:** `/exit` 与 `/quit` 退出,`/reset` 清空上下文,`/think on|off` 开关思考模式。
- **常用参数:** `--max-ctx N` 设 KV cache 容量(决定最大上下文),`--slots N` 设最大并发序列数,
  `--addr` 设 serve 的 HTTP 监听地址;temperature、top_p 等采样参数按每条 OpenAI 请求传入。
- **OpenAI 端点:** `POST /v1/chat/completions`(支持流式、非流式与 tools)、`GET /v1/models`、
  `GET /healthz`。
- **支持的模型:** 任意 dense 的 Qwen3(`config.json` 中架构为 `Qwen3ForCausalLM`,各维度从该文件读取)。
  不支持 Qwen3 MoE、多模态模型,以及非 Qwen 架构。

---

## 代码结构

外层 Go 应用(根模块)的各个包:

| 路径 | 负责 | 行数 |
|---|---|---:|
| `cmd/flashqwen/` | 程序入口,分发 `serve` / `chat` / `benchmark` 三个子命令 | 346 |
| `internal/engine/` | 对接内层引擎:底层是 token 级 gRPC 客户端,高层是文本级 `Generate`(渲染→分词→驱动引擎→解码);gRPC 桩 `enginepb` 只在此包内使用 | 225 |
| `internal/server/` | 把引擎暴露为 OpenAI 兼容 HTTP 服务,负责 OpenAI 格式的解析/编码与 SSE 流式输出 | 293 |
| `internal/chatml/` | Qwen3 ChatML 格式的双向 codec:把消息渲染成 prompt,把引擎的 token 流解码成文本并识别工具调用 | 266 |
| `internal/tokenizer/` | 从零实现的 byte-level BPE 分词器,读取 vocab / merges / tokenizer.json | 483 |
| `internal/supervisor/` | 用 `//go:embed` 内嵌引擎二进制,负责释放、拉起、健康检查与退出时收尾 | 78 |

内层 C++/CUDA token 引擎(`engine/src/`)的源文件,按行数降序:

| 文件 | 负责 | 行数 |
|---|---|---:|
| `kernels.*` | 全部 CUDA kernel:INT8 GEMV、WMMA matmul、attention、split-K decode | 529 |
| `model_runtime.*` | 加载权重并在显存里量化成 INT8,执行前向计算 | 435 |
| `scheduler.*` | 连续批处理调度器:分块 prefill、序列接纳与抢占 | 215 |
| `grpc_server.*` | token 级 gRPC 服务:接收 token id,流式返回采样出的 token id | 181 |
| `kv_cache.*` | PagedAttention 的分页 KV 池 | 127 |
| `safetensors.*` | mmap 并用 RapidJSON 解析 `.safetensors` 文件头 | 111 |
| `model_spec.hpp` | 从 `config.json` 读取模型维度与架构(声明层,不碰 GPU) | 64 |
| `sampler.*` | 采样:greedy、temperature、top-p | 54 |
| `main.cpp` / `args.*` | 引擎进程入口与命令行参数解析 | 88 |

两层之间的接口由 `proto/engine.proto` 定义。完整的单流优化历程(标量 matmul → WMMA → split-K →
INT8,49.7 → 104 tok/s)和批处理的分阶段实现分别保存在 `optimization-study` 与 `feature/batching`
两个分支。

---

## Benchmark

下表是 `./flashqwen benchmark` 测得的端到端吞吐,走的是经 gRPC 的真实路径,包含分词与采样。

**环境:** NVIDIA RTX 4090(24 GB),Qwen3-8B,matmul 权重 INT8 量化,连续批处理 + PagedAttention。
**配置:** 默认参数 —— 合成请求,每条 128 token 输入、128 token 输出,每个并发档位 32 条请求,贪心解码。

| 并发 | 请求数 | 墙钟 | 聚合吞吐 | 平均 TTFT |
|---:|---:|---:|---:|---:|
| 1  | 32 | 44.9 s | 91 tok/s  | 99 ms  |
| 8  | 32 | 18.0 s | 228 tok/s | 286 ms |
| 16 | 32 | 16.7 s | **245 tok/s** | 647 ms |

单流约 91 tok/s:decode 阶段是访存瓶颈(每生成一个 token 都要把整个模型读一遍),INT8 权重把上限推到
这个量级。并发提到 16 时,连续批处理把一次权重读取摊薄到同批的多个序列上,聚合吞吐升到约 245 tok/s
(约 2.7×);代价是首 token 延迟(TTFT)随并发升高。
