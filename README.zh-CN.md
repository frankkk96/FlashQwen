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

`--model` 指向一个存放模型的本地目录。FlashQwen 自身不负责下载;用官方的 Hugging Face CLI 把权重拉到
本地即可:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-8B --local-dir models/qwen3-8b
```

(需要镜像时,先设置 `HF_ENDPOINT=https://hf-mirror.com`。)然后让 FlashQwen 指向该目录:

```bash
./flashqwen chat --model models/qwen3-8b
```

该目录需包含 `config.json`、`*.safetensors` 分片、`model.safetensors.index.json`、
`tokenizer.json`、`vocab.json`、`merges.txt`、`generation_config.json`。引擎直接读这些 BF16 文件、在加载时
把 matmul 权重量化成 INT8,Go 侧从 `tokenizer.json`/`vocab.json`/`merges.txt` 加载分词器,因此无需任何
离线转换或重打包。

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

## 技术路线

### Go 与 C++ 的桥接

编译好的 C++ 引擎二进制通过 `//go:embed` 打包进 Go 二进制。启动时 Go 侧的 supervisor 把它写到临时
目录,选一个空闲的本地端口,作为子进程拉起,轮询 `GetModel` 直到引擎应答,退出时给它发 `SIGTERM`——
所以只有一个二进制要跑,没有引擎进程需要手动管理。

两层通过 gRPC 通信,接口由 `proto/engine.proto` 定义。线上只传 token id:Go 把 prompt 分词后发送
`GenerateRequest{input_ids, max_tokens, temperature, top_p, stop_token_ids}`;`Generate` 是
server-streaming,引擎先流式返回一串 `token_id` 事件,最后给一个
`Done{finish_reason, prompt_tokens, completion_tokens}`。`GetModel` 回报引擎权威的上下文窗口与词表
大小。客户端断开会取消该 RPC,引擎据此中止对应序列并释放它的 KV。

### 权重加载与 INT8 量化

引擎 mmap `.safetensors` 分片、用 RapidJSON 解析文件头,以 BF16 读入权重。加载时把每个 matmul 权重
(注意力的 q/k/v/o 投影、MLP 的 gate/up/down,以及 `lm_head`)量化成 INT8,**每个输出行一个 scale**
(对称,夹到 [-127, 127]);embedding 与 norm 保持 BF16。这把权重显存大致减半(~16 GB → ~9 GB)。
prefill 会把 INT8 权重反量化回 BF16 喂给 tensor core,decode 则直接读 INT8 字节——decode 是访存瓶颈,
这一点才是关键。

### 前向:prefill 与 decode

一次前向分两条路径。**prefill**(一次过很多 prompt token)是计算瓶颈,matmul 用 WMMA 走 tensor
core。**decode**(每步一个 token)是访存瓶颈,用向量化的 INT8 GEMV;固定的单 token kernel 序列只捕获
一次成 CUDA graph、每步重放(token id、位置、`past_len` 都放在设备 buffer 里,所以上下文增长时 graph
依然有效)。贪心解码在 GPU 上做 argmax、只回传一个 int;temperature / top-p 采样则把 logits 拷回主机。

### 注意力:flash-decoding split-K

prefill 的 attention 是一个 warp 负责一个 (head, query),online softmax 全在寄存器里完成,没有块内
barrier。decode 在 batch 1 时这种划分只有 ~32 个并行单元(每 head 一个),所以把每个 head 的 key 区间
切成 `ATTN_SPLITS`(16)块,交给不同 block 各算一个 partial online-softmax,再 combine 归并——并行度
约 16×,于是 decode 延迟几乎不随上下文增长。

### Paged KV cache(PagedAttention)

每层的 KV 是一个 `[num_blocks, BLOCK=16, kv_dim]` 的 BF16 池,大小由权重和激活之后剩余的显存决定。
每个序列持有一张**块表**(物理块 id 列表),逻辑位置 `p` 映射到物理行
`block_table[p/BLOCK]*BLOCK + p%BLOCK`(寻址公式共享在 `kv_cache.cuh`)。块按需分配,所以显存随**实际**
序列长度增长,并发序列数与 `max_ctx` 解耦:在 24 GB 的 4090 上,同样的显存变成一个共享的 ~8.7 万 token
池,而不是每序列预留固定一段。三个 KV kernel 都经块表寻址:`store_kv_paged`(prefill 与 decode 共用)、
`attention_paged`(prefill)、以及 split-K 的 `attention_decode_paged`。

### 连续批处理与抢占

调度器(`engine/src/scheduler.*`)让所有 `n_slots` 一直满载:一有槽位空出就接纳一个等待中的请求,每次
迭代把整个运行集合一起 decode。prefill 按固定大小分块(`PREFILL_CHUNK` = 每次迭代 256 token)、与 decode
交错进行,所以长 prompt 不会卡住正在运行的序列。块池耗尽时,抢占最年轻的运行序列——释放其块并重新
排队——待恢复时从 prompt + output **重算** KV(vLLM 的策略),从而在显存压力下保持正确而非死锁。每个请求
带自己的采样参数:全 greedy 的批走 GPU argmax,只要有一个请求要采样,整批就回退到主机侧逐行采样。

这些背后的分阶段历程——单流优化(标量 matmul → WMMA → split-K → INT8,49.7 → 104 tok/s)与批处理实现
——保存在 `optimization-study` 与 `feature/batching` 两个分支。

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
