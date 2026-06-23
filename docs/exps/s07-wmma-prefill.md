# S7 — WMMA tensor-core prefill attention

## 背景/动机

在 S6 把 decode 拆分出去之后，prefill-only 的 attention kernel 仍然使用 CUDA-core 的 FMA dot product（per-key 的 warp-shuffle reduction）来完成本质上是 compute-bound 的 512×L×128 矩阵乘——没有用到 tensor core（张量核心）。

S4 的 WMMA 尝试之所以失败，是因为它被应用到了 *decode* 上（`q_len=1` 浪费了 16 行 WMMA tile 中的 15/16，且 per-kv-head grouping 让 grid 饥饿）；如今 decode 已经在自己专属的 kernel 上，把 WMMA 用在 prefill 路径（`q_len` 最高 512，full tile）才是正确的应用场景，S4 的失败模式也随之消失。

## 改动

`attention_prefill_wmma_kernel`——采用 FlashAttention-2，配合 WMMA 16×16×16 bf16（fp32 accumulate）：
- 每个 (16-query-tile, head, request) 一个 warp；
- Q 直接从 fused-QKV buffer 读取，K/V 直接从 paged cache 读取（一个 16-key tile == 一个 page，因为 block_size==16），全部以 WMMA load 形式——不对 S 做 materialization（物化）；
- Online softmax + deferred normalization（延迟归一化）；O 保存在 shared 的 fp32 中，按 tile rescale（可移植——不依赖任何 WMMA fragment-layout 假设）；
- 通过 `launch_attention_prefill` 在 prefill 子集上 dispatch；当 `block_size!=16` 或 `head_dim!=128` 时回退到 FMA fallback。

## 实测结果

相对 S6：
- conc=32 **455 → 493 tok/s（+8.5%，TPOT 65.3 → 59.8 ms）**；
- conc=1 持平（44.7 → 44.9——single-stream 是 decode-bound，prefill 是一次性成本，摊销到 128 个输出上）。
- **65% → 71% of vLLM**，达到 `main` 基线的 4.6×。
- 输出验证连贯（coherent）。

commit: 0dd4010。

在跨输入长度的 journey replay 中（conc=32 output tok/s，括号内为相对 feature-matched vLLM 的百分比）：

| step | in=128 | in=512 | in=1024 | 这一步带来了什么 |
|---|---|---|---|---|
| S7 | 1326 (96.4%) | 860 (91.1%) | 501 (76.8%) | WMMA prefill attn——提升长输入（@1024 +7pt） |

Progress 表中的对应行：

| Step | change | conc=1 | conc=32 | % vLLM | commit |
|---|---|---|---|---|---|
| S7 | WMMA tensor-core prefill attention | 44.8 | **501** | **77%** | 0dd4010 |

## 经验

Prefill attention 确实是 conc=32 中一块有意义的占比（+8.5% 就是证明），而且 S4 的想法（tensor core）一直都是对的——只是被错误地应用到了 decode 上。是按 request type 拆分 attention（S6）解锁了它：现在每种 regime（场景）都拿到了它想要的 kernel（decode 用 FlashDecoding，prefill 用 WMMA-FA）。

剩余到 vLLM 的差距（493→698）：GQA-shared decode（4× 冗余 KV 读取——被判定为边际收益，KV 仅约占字节量的 4% 且被 L2 缓存）、为 conc=1 引入 CUDA graphs，以及这个 WMMA kernel 的 1-warp/block occupancy（更大的 query tile / 每 block 更多 warp 可以把它推得更远）。
