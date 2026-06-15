# FlashQwen

从零实现的 **Qwen3-8B C++/CUDA 推理引擎**,模型栈零 ML 依赖(没有 PyTorch、没有 cuBLAS、没有
tokenizers;只有 header-only 的 JSON/命令行小工具),主要面向学习。支持多轮流式对话、benchmark
模式(TTFT / TPOT / tok/s)、连续批处理 + PagedAttention,以及 OpenAI 兼容 API 服务。

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
`vocab.json` 和 `merges.txt`。无需离线转换或重打包——FlashQwen 直接读 BF16 文件,并在加载时
把 matmul 权重在显存里量化成 INT8。

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

# 用 temperature + nucleus 采样对话
./build/flashqwen --model models/qwen3-8b --temperature 0.6 --top-p 0.95

# benchmark(内置的固定输入长度扫描)
./build/flashqwen benchmark --model models/qwen3-8b

./build/flashqwen --help
```

**模式:** 不带子命令(或写 `chat`)→ 交互对话;`benchmark` → 跑指标。

**对话内命令:** `/exit`  `/quit`  `/reset`(清空上下文)  `/think on|off`。

**常用参数:** `--max-ctx N`(KV cache 大小,默认 4096)、`--temperature`、`--top-p`、
`--seed`、`--think`(开启 Qwen3 思考模式)。

**支持的模型:** 任意 dense 的 Qwen3 模型(架构为 `Qwen3ForCausalLM`):Qwen3-0.6B / 1.7B /
4B / 8B / 14B / 32B,各维度从 `config.json` 读取。**不支持:** Qwen3.5(混合线性注意力 +
多模态)、Qwen3 MoE 变体,以及非 Qwen 架构。

> 首次启动会从磁盘加载 ~16 GB 权重(网络盘上较慢);之后命中操作系统页缓存,几秒就能启动。

---

## 本地 Benchmark

**硬件:** NVIDIA GeForce RTX 4090(24 GB)· 驱动 580.76.05(CUDA 13.0)· 用 CUDA 12.8 编译(原生 sm_89)
**模型:** Qwen3-8B —— matmul 权重量化为 INT8(每行一个 scale),BF16 embedding,FP32 激活;
prefill 走 tensor core(WMMA),decode attention 用 flash-decoding split-K,单 token decode
由 CUDA graph 重放。
**方法:** 单流(batch=1)、贪心、固定 128 token 输出(忽略 EOS)、1 次 warmup + 3 次测量取
中位数。扫描逻辑内置 —— 直接 `flashqwen benchmark --model <DIR>` 即可。

按输入长度扫描(合成 prompt),输出固定 128 token:

| 输入 tok | TTFT | TPOT | decode tok/s | output tok/s | peak tok/s |
|---:|---:|---:|---:|---:|---:|
| 16   | 69 ms  | 9.6 ms  | 104.2 | 98.7 | 105.1 |
| 128  | 97 ms  | 9.7 ms  | 102.6 | 95.3 | 103.5 |
| 512  | 0.40 s | 10.2 ms |  97.8 | 74.8 |  98.5 |
| 1024 | 0.95 s | 10.9 ms |  91.9 | 54.6 |  92.6 |

**指标定义:**
- **TTFT** — time to first token = 整个 prompt 的 prefill + 采样第 1 个 token。
- **TPOT** — time per output token,decode 阶段(第 1 个之后)每 token 的平均延迟。
- **decode 吞吐** — 仅 decode 阶段的 tok/s。
- **output 吞吐** — 含 prefill 的 tok/s(`n_out / total_time`)。
- **peak output 吞吐** — `1 / 最快单 token 延迟`。

**怎么看这些数字。** decode 是访存瓶颈——每个 token 都要读完整个模型——所以 INT8 权重
(~9 GB,bf16 是 ~16 GB)把它推到 **~100 tok/s**,而 flash-decoding split-K 让它几乎不随上下文
变化(TPOT 从 16 token 的 9.6 ms 到 1024 token 的 10.9 ms)。prefill 走 tensor core(WMMA);
INT8 权重要先反量化成 BF16,所以 TTFT 比纯 BF16 版略高。这些是下面「优化历程」**之后**的数字——
历程从 49.7 tok/s 的标量 baseline 开始,逐步展示每一处提升的来源。

---

## 依赖的库 & 代码结构

### 依赖——模型栈全部从零实现

整个**推理栈**——分词器、safetensors 加载、每一个 CUDA kernel——都只基于 **C++ 标准库**、
**CUDA Runtime**(toolkit 自带)和 **POSIX** 手写。唯一的第三方代码是 header-only 的**工具类**
解析库,vendored 在 `third_party/`(所以 `cmake --build` 离线即可编译):RapidJSON 解析那几个
JSON 文件、CLI11 解析命令行参数——它们都不碰张量。可选的 Go API 网关(`api/`)另外用了 Gin。

| 通常要用的库 | 这里的做法 |
|---|---|
| HF tokenizers / sentencepiece | 手写 byte-level BPE `src/tokenizer.*` |
| safetensors C++ 库 | 手写 `src/safetensors.*`(mmap + 解析头) |
| cuBLAS / CUTLASS | 手写 matmul(prefill 用 tensor core WMMA,decode 用 GEMV) |
| PyTorch(任意深度学习框架) | 完全没用 |
| JSON / 参数解析 | RapidJSON + CLI11(header-only,`third_party/`)—— 仅工具用途 |

- C++ 标准库:`<vector> <string> <unordered_map> <chrono> <random> <cmath> <fstream>` 等
- CUDA:`<cuda_runtime.h>`、`<cuda_bf16.h>` —— **没有** cuBLAS / cuDNN / cuRAND / Thrust
- POSIX:`<sys/mman.h>`(mmap)、`<fcntl.h>`、`<unistd.h>`

### 代码结构与行数

每个文件只做一件事;`main.cpp` 是个很薄的入口,只负责解析参数和分发。总共约 1940 行。

**应用层**

| 文件 | 作用 | 行数 |
|---|---|---:|
| `src/main.cpp` | 入口:参数解析 + 分发 | 69 |
| `src/cli.cpp` / `.hpp` | 架构检查 + `--help` | 63 |
| `src/chat.cpp` / `.hpp` | 交互式多轮对话 | 55 |
| `src/benchmark.cpp` / `.hpp` | benchmark 模式(扫描输入长度) | 92 |
| `src/generate.cpp` / `.hpp` | 共用的 prefill + decode 循环 | 66 |
| `src/sampler.cpp` / `.hpp` | 采样(greedy / temp / top-p) | 50 |

**核心引擎**

| 文件 | 作用 | 行数 |
|---|---|---:|
| `src/model.cu` / `.hpp` | 权重加载(INT8 量化)+ 前向 + KV cache + CUDA graph | 331 |
| `src/kernels.cu` / `.cuh` | CUDA kernel(INT8 GEMV / WMMA matmul、attention、split-K…) | 431 |
| `src/tokenizer.cpp` / `.hpp` | byte-level BPE 编码/解码 | 394 |
| `src/safetensors.cpp` / `.hpp` | mmap + 解析 `.safetensors` 头(RapidJSON) | 105 |
| `src/config.hpp` | 解析 `config.json`(RapidJSON) | 45 |
| `CMakeLists.txt` | 构建 | 36 |

连续批处理服务路径(`src/scheduler.*`、`src/server.*`)和 Go API 网关(`api/`)在下文各自的章节里介绍。

值得一提:手写分词器约 394 行;真正的神经网络部分——CUDA kernel(431)+ 前向(331)——约 760 行,
增长主要来自下面的优化阶段(INT8、split-K),而不是 Qwen3 本身——它结构小而规整。

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
| 5 · GPU argmax(greedy) | `bench-5-argmax` | 83 ms | 909 ms | 38.5 ms | 51.7 |
| 6 · CUDA graph(decode) | `bench-6-cudagraph` | 83 ms | 909 ms | 38.1 ms | 52.9 |
| 7 · flash-decoding split-K | `bench-7-splitk` | 83 ms | 910 ms | 18.9 ms | 56.9 |
| 8 · INT8 权重量化 | `bench-8-int8` | 97 ms | 954 ms | 10.9 ms | **104.2** |

相比 baseline:**prefill 快约 13–16×**,**decode 快约 2×**且几乎不随上下文变化——
decode@16 49.7 → 104.2 tok/s,TPOT@1024 40.1 → 10.9 ms。(INT8 用一点 prefill TTFT 换 decode
的大提升。)复现任一阶段:

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

**阶段 5 — GPU argmax(greedy),`bench-5-argmax`。** 贪心解码不再每个 token 把 151936 维 logits
全拷回主机再扫,而是在 GPU 上做 argmax 归约、只回传一个 int。省 ~0.2–0.3 ms/token
(decode@16 51.1 → 51.7 tok/s)。temperature 采样仍走完整 logits 拷贝。

**阶段 6 — CUDA graph(decode),`bench-6-cudagraph`。** 一个 decode step 要 launch ~430 个小
kernel(36 层 × ~12 个);有些 kernel 太短,launch 开销盖不住、暴露出来了。把固定的单 token
序列捕获成一个 CUDA graph、每步重放(token id / 位置 / `past_len` 都放在 kernel 读取的设备
buffer 里,所以上下文增长时 graph 依然有效)。稳定省 ~0.4–0.5 ms/token(decode@16 51.7 → 52.9
tok/s,peak 55.2 → 56.5)。prefill(长度可变)仍然 eager 执行。

**阶段 7 — flash-decoding split-K,`bench-7-splitk`。** M=1 时 warp attention(阶段 3)只有 32 个
并行单元(每 head 一个),所以 decode attention 受并行度限制、TPOT 随上下文增长(18.9 ms@16 →
38.1 ms@1024)。现在把每个 head 的 key 区间切成 `ATTN_SPLITS`(16)块,交给不同 block 各算一个
partial online-softmax,再用一个 combine pass 归并,等于对 KV cache 多了 16× 的并行度。
**TPOT@1024 从 38.1 暴降到 18.9 ms(decode 26.3 → 53.0 tok/s,约 2×)**,而且 decode 现在几乎
不随上下文变化(各长度都 ~18–19 ms),也就是又回到被权重读取(GEMV)主导,而不是 attention。
prefill 仍用阶段 3 的 kernel(并行度本来就够)。

**阶段 8 — INT8 权重量化,`bench-8-int8`。** decode 之前卡在 bf16 带宽天花板(每 token 读完整
~16 GB 权重 → ~60 tok/s 上限)。把 matmul 权重(注意力 + MLP 投影 + lm_head)量化成
**INT8 + 每输出行一个 scale**(对称,load 时计算);embedding/norm 保持原样。decode 读 1 字节
权重、kernel 内反量化 → **decode@16 56.9 → 104.2 tok/s(~1.8×),TPOT@1024 18.9 → 10.9 ms**,
权重显存大致减半(~16 → ~9 GB)。prefill 会先把权重反量化成 BF16 再走 WMMA,因此 TTFT 略升
(如 @128 83 → 97 ms)——这是划算的:prefill 只跑一次,decode 每个 token 都跑。输出依然连贯
(8B 模型上 per-channel INT8 影响很小)。

**仍待优化:** INT4 / 分组量化拿更多 decode 余量、激活也量化用 INT8 tensor core 加速 prefill、
以及给 prefill WMMA 加 shared-memory tiling。

## 批处理(独立分支 `feature/batching`)

上面的优化历程全是**单流(batch 1)**。另一条分支把引擎做成一个小型服务后端,能**同时跑多个序列**。
收益在 decode:decode 的瓶颈是每个 token 都要把整个模型读一遍,而一步同时服务 `B` 个序列,可以把
**这一次权重读取摊薄到 `B` 个 token 上**,而不是每 token 各读一次。

分三个阶段:**A——批处理 decode + 静态批处理**(已完成),**B——连续批处理**(动态调度器,已完成),
**C——PagedAttention**(分页 KV + 抢占,已完成)。

**阶段 A——槽位式 KV cache + 批处理 decode。** KV cache 改成 `[B_max, max_ctx, kv_dim]` 池;
`B_max`(并发序列槽数)由权重和激活之后、`--gpu-mem-fraction`(默认 0.9)以内剩余的显存决定——
例如 24GB 4090 上 `max_ctx=2048` 得到 **42 个槽**。decode 路径重写成一步跑一整批:

- **模板化 INT8 GEMV**:每个 warp 把一行权重读一次,算出全部 `B` 个点积(`B` 是编译期常量,
  累加器留在寄存器里);
- 按槽位的 `store_kv`、带 batch 维 + 每序列槽位 / `past_len` 的 split-K decode attention、
  批处理 argmax;`lm_head` 是一次批处理 GEMV 写出 `[B, vocab]` logits(每序列 ~0.6 MB),
  再走批处理 argmax(greedy)或逐序列的 host 采样。

prefill 仍是单序列(写入对应序列的槽位;prefill 批处理留到后续阶段)。CUDA graph 去掉——批处理
形状会变——所以 chat 现在作为 `B=1` 走同一条 decode 路径(在 batch 1 下损失了阶段 6 graph 的那点
收益,符合预期)。

静态批处理 decode 吞吐(RTX 4090,Qwen3-8B INT8,greedy,128 tok/序列,`input=128`):

| batch | 每序列 TPOT | 聚合 decode tok/s |
|---:|---:|---:|
| 1  | 10.2 ms | 98  |
| 8  | 28.3 ms | 283 |
| 16 | 49.7 ms | **322** |

聚合 decode 吞吐 **~98 → ~322 tok/s(~3.3×)**(batch 16)——次线性,而原因很有启发性(且不是想当然
的那个)。batch 16 的一步**既不是**权重带宽受限(~9 GB / ~50 ms ≈ 190 GB/s,远低于 4090 的 ~1 TB/s),
**也不是** ALU 受限(整步 FP32 计算理论只要 ~2.5 ms,利用率 ~5%)。真正的限制是**激活值的 L2 重读**:
GEMV 每个 warp 算一个输出行,每行都要把全部 `B` 个激活向量重读一遍,于是激活流量 ∝ `B × 参数量`
(B=16 时约 400 GB 的 L2 读取)——这也是 step 时间随 `B` 近似线性增长的原因。解法是**激活复用**
(共享内存 / tiled GEMV),而不是 tensor core:实测的 dequant→BF16 WMMA 路径反而**更慢**,因为每步把
INT8 权重反量化成 BF16 写回(~14 GB、与 batch 无关)的开销占了主导。留作后续工作。

**阶段 B——连续批处理。** 底层只有一个执行原语——批处理 `decode`,它接收任意的运行集合(每序列 KV
槽位 + `past_len`)。所以「静态」vs「连续」是**主机侧的调度策略**,而不是两种引擎模式。朴素(静态)
批处理一次处理固定一组,要等组里**最慢**的序列结束才释放槽位,短请求被长请求堵在后面(队头阻塞)。
连续批处理(`src/scheduler.*`)则让 `n_slots` 一直满载:一有槽位空出就立刻接纳等待中的请求,每步把
整个运行集合一起 decode。在一个变长工作负载上,它比静态基线快约 1.4×,因此只保留连续批处理这一条
服务路径。每个请求带自己的采样参数(temperature / top-p,或 greedy):全 greedy 的批走 GPU
argmax,只要批里有一个要采样就把 `[B, vocab]` logits 拷回主机逐行采样。接纳时一次只 prefill 一个
prompt,但按固定大小(256 token)分块、与 decode 交错进行,所以一个长 prompt 不会在整段 prefill 期间
卡住正在运行的序列——它们在分块之间持续 decode。

连续批处理吞吐(32 请求,input 128,output 16–128 随机),改变 KV 槽位数——`slots=1` 即顺序服务
(一次一个请求):

| 槽位 | 墙钟 | 聚合 tok/s |
|---:|---:|---:|
| 1  | 22.4 s | 86  |
| 8  | 10.4 s | 185 |
| 16 | 10.0 s | **193** |

**阶段 C——PagedAttention。** 阶段 A/B 给每个序列槽位预留一段**连续**的 `max_ctx` KV 区域,
也就是无论序列实际多长,都先按满上下文长度占住——`max_ctx=4096` 时每序列约 600 MB,直接限制了能并发
多少序列。阶段 C 换成**分页**池:每层 KV 是 `[num_blocks, BLOCK, kv_dim]`(BLOCK = 16 token),每个
序列持有一张**块表**(物理块 id 列表),逻辑位置 `p` 落在物理行 `block_table[p/BLOCK]*BLOCK + p%BLOCK`。
块按需分配,显存随**实际**长度增长,并发序列数与 `max_ctx` 解耦。同样的显存在 4090 上变成一个
**5207 块 × 16 token ≈ 8.3 万 token** 的共享池,而不是固定的 ~20×4096 预留。

三个 KV kernel 改成经块表寻址(算术与连续版完全一致):`store_kv_paged`(prefill 与 decode 共用)、
`attention_paged`(prefill)、以及 split-K 的 `attention_decode_paged`。调度器(`src/scheduler.*`)
持有空闲块列表,在跨 BLOCK 边界时为每个运行序列增长块表。池耗尽时**抢占**最年轻的运行序列——释放其块
并重新排队——待之后恢复时再从 `prompt + output` **重算** KV(与 vLLM 的重算策略一致)。这让引擎在显存
压力下保持正确而不是死锁;吞吐与阶段 B 持平(分页是显存效率的改动,每步的块表上传开销可忽略)。已验证
与单序列 greedy decode **逐 token 一致**——批处理(不相交块表)、连续、以及用刻意缩小的池**强制触发
125 次抢占**的压力测试,输出都与参考完全相同。chat 作为 `B=1` 走一张恒等块表。

`feature/batching` 用分支管理(不打每阶段的 tag)。试用:

```bash
git checkout feature/batching
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
./build/flashqwen benchmark --model models/qwen3-8b   # benchmark 多出静态批处理 + 连续批处理两段
```

## OpenAI 兼容 API 服务(Go 网关,`feature/api-server`)

把引擎通过 OpenAI Chat Completions API 暴露出来的双进程方案——Go 管服务、C++ 管模型,边界是 **gRPC**:

```
[OpenAI 客户端]
   | HTTP /v1/chat/completions (SSE)
   v
[Go + Gin 网关]   OpenAI 协议 <-> gRPC、SSE   (对模型零知识)
   | gRPC server-streaming   GenerateRequest{messages,tools,params} ->
   |   TextDelta / ToolCall / Done 事件流
   v
[C++ flashqwen serve]   Qwen3 模板 -> 分词 -> 连续批处理 decode
   |                     -> 反分词 -> token 级工具调用识别
   v
  GPU
```

契约是 `proto/inference.proto`。**C++ 引擎**(`flashqwen serve`,`src/grpc_server.cpp`)拥有所有模型
相关的东西:接收结构化 messages + tools,渲染 Qwen3 ChatML 模板(`src/chat_template.cpp`),分词,在
单个 GPU 线程上跑连续批处理 `Scheduler`,并在 **token 级**识别工具调用(`<tool_call>`/`</tool_call>`
是特殊 token),吐出 `TextDelta` / `ToolCall` / `Done` 三种事件。**Go 网关**(`api/`)是个薄适配层:
把 OpenAI 请求映射成 `GenerateRequest`,把事件作为 SSE chunk 流式吐出或攒成一整条,并报告 usage——
它**不含 chat 模板、也不解析工具格式**,所以换模型完全不用动 Go。gRPC 取消已打通:客户端断开 → RPC
取消 → 引擎丢弃在途序列(释放其 KV)。采样是 temperature + top-p。每个请求是独立序列,多客户端共享同一个
批处理引擎。

**运行**(两个终端):

```bash
# 1) 引擎(C++):加载模型,提供 gRPC 服务
./build/flashqwen serve --model models/qwen3-8b --address 127.0.0.1:50051 --slots 16

# 2) 网关(Go):在 :8000 上说 OpenAI 协议,经 gRPC 转发给引擎
cd api && go build -o flashqwen-api . && ./flashqwen-api --engine 127.0.0.1:50051 --addr :8000
```

编译引擎需要 gRPC + Protobuf C++(CMake 自动探测 `find_package(gRPC CONFIG)` 安装,例如装到
`/usr/local` 的源码构建;否则回退到发行版的 `libgrpc++-dev` 走 pkg-config)。Go 侧用 `go mod` 拉
`google.golang.org/grpc`。改了 proto 后重新生成:C++ stub 由 CMake 构建时用 `protoc` 生成,Go stub
在 `api/pb/`。

之后任意 OpenAI 客户端都能用(`base_url=http://localhost:8000/v1`):

```bash
# 非流式
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen3-8b","messages":[{"role":"user","content":"法国的首都?一个词回答。"}]}'

# 流式(SSE)
curl -N localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen3-8b","stream":true,"messages":[{"role":"user","content":"列出三种颜色。"}]}'

# function calling —— 模型返回 tool_calls;把结果作为 role:"tool" 消息再发回去
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen3-8b","messages":[{"role":"user","content":"巴黎天气怎么样?"}],
  "tools":[{"type":"function","function":{"name":"get_weather",
    "parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]}'
```

端点:`POST /v1/chat/completions`(流式 + 非流式、tools)、`GET /v1/models`、`GET /healthz`。
