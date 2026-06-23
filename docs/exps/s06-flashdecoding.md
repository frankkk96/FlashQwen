# S6 — FlashDecoding decode-attention kernel

## 背景/动机

S6 针对的是 S5 profiling 留下的 conc=32 天花板。此前的 profile 使用的是 SHORT context（短上下文），而我们追踪的标准指标是 1024-ctx——在这个场景下，decode attention 每一步都要读取约 1152-tok 的 KV。

统一的 attention kernel 处理 `q_len==1`（decode）的情况非常糟糕：它的 16 个 warp 里只有 1 个是活跃的（约 6% occupancy），而且这唯一的 warp 还要串行扫描全部约 72 个 KV tile。GEMM 在 S5 的 profile 里已经达到了 84% 的 HBM 带宽（HBM BW），因此 attention 成为剩下唯一可挖掘的 lever。

## 改动

按 request type（请求类型）在统一 batch *内部* 拆分 attention——GEMM/norm 仍然保持 merged（attention 本来就是 per-request 的，所以这与 unified batching 正交，**不是** 退回到 prefill/decode 拆分的 forward）。具体：

- Decode 行（`q_len==1`）→ 新的 `decode_attn_kernel`；
- Prefill 行（`q_len>1`）→ 已有的 tiled flash kernel。

两者都接收一个 `rids[]` 的 grid→request 间接索引，使得每个 kernel 各自运行在自己的 request 子集上，写入 `attn_` 中互不相交（disjoint）的行。在稳态的 conc=32 下，大多数 step 是纯 decode，因此 decode kernel 承载了 common case（常见情形）。

**`decode_attn_kernel` 细节：**
- 每个 (head, decode-request) 一个 block；NW=8 个 warp 把该 request 的 KV `[0,qpos]` 拆成跨步（strided）切片；
- 每个 warp 在寄存器中对自己的切片做 online-softmax，K/V **直接从 paged cache 读取**（单个 query 行没有跨行复用 → shared-memory staging 纯属 overhead）；
- 然后一个 in-block combine（块内合并）把 NW 个 partial 用 online-softmax 合并。
- 相比旧的 1-warp 串行扫描，实现了 full warp occupancy（满 warp 占用）+ NW 路 KV 并行。

GQA 的 K/V 仍然按 q-head 读取（一个 group 内 4×），这是可接受的：KV 字节量约为 GEMM 的 4%，而且一个 group 的 4 个 q-head 命中相同的 cache line（L2 吸收了大部分）。

## 实测结果

相对 S5：
- conc=32 **357 → 455 tok/s（+27.5%，TPOT 84.9 → 65.3 ms）**；
- conc=1 39.1 → 44.7（+14%，TPOT 25.8 → 22.5 ms）。
- **51% → 65% of vLLM**，达到 `main` 基线的 4.26×。
- 贪婪（greedy）输出连贯（coherent）。

commit: f0b1499。

在跨输入长度的 journey replay 中（conc=32 output tok/s，括号内为相对 feature-matched vLLM 的百分比）：

| step | in=128 | in=512 | in=1024 | 这一步带来了什么 |
|---|---|---|---|---|
| **S6** | 1317 (95.7%) | 824 (87.3%) | 454 (69.6%) | **FlashDecoding——短输入（decode）下提升最大** |

Progress 表中的对应行：

| Step | change | conc=1 | conc=32 | % vLLM | commit |
|---|---|---|---|---|---|
| S6 | FlashDecoding decode-attention kernel (split by request type) | 44.5 | **454** | **70%** | f0b1499 |

## 经验

S5 的 profile（短上下文）低估了 attention 的占比；在追踪的 1024-ctx 上，手写的 decode attention 终究确实是 conc=32 的瓶颈——而且与已经被榨干的 S4 WMMA 尝试（它没有修复任何真实问题）不同，FlashDecoding 式的 **KV-parallelism（KV 并行）+ occupancy（占用率）** 才是正确的修复方向。

这印证了 profiling sweep 得出的规律：GEMM/elementwise/scheduling 都是死路（dead end）；唯一还有 headroom 的手写 kernel 交付了成果。

接下来到 vLLM 的差距（455→698）很可能在于 prefill 侧的 attention + GQA 的 4× 冗余 KV 读取（一个 GQA-shared decode kernel），以及为 conc=1 引入 CUDA graphs。
