# S12 — GQA-shared FlashDecoding decode attention

## 背景/动机

S6 的 decode kernel 是按 (q-head, request) 一个 block 来组织的;在 Qwen3 的 4:1 GQA 下,每个 kv-head 的 K/V 会被它所在 group 的每个 query head 各读一次(4×)。这是一笔可以省掉的冗余 K/V 流量。

注意这与早已废弃的 S4 GQA 尝试不同:S4 同样按 kv-head 分组,但只用了 `n_kv` 个 block → GPU 被饿死(starved) → 看起来收益持平。S12 要解决的正是“分组后 block 太少导致 GPU 饥饿”的问题。

## 改动

新的两阶段 kernel:

- **phase 1** = 一个 block 对应 (kv-head, request, **KV-split**),一次性计算该 group 的全部 G=4 个 q-head,把每个 K/V 元素**只载入寄存器一次**并在 4 个 head 间复用;
- **phase 2** 合并各 split 的 FlashDecoding partial。
- `grid.z` 的 KV-split(`ksplit = clamp(128/n_decode, 1, 16)`)让 4090 在 n_kv 比 n_heads 少 4× block 的情况下仍保持饱和(conc=32:8·32·4 = 1024 blocks × 8 warps = 8192 warps = 满)。
- 旧的 per-head kernel 保留为 fallback(head_dim≠128 或不支持的 group)。

**为什么现在能成(对比 S4)**:S4 按 kv-head 分组但只用 `n_kv` 个 block → GPU 饿死 → 看起来持平。KV-split 才是让分组变成净收益的关键。L2 其实已经吸收了 4× 重读带来的绝大部分冗余 *HBM* 流量,但冗余的 L2 bandwidth + load 指令 + score reduction 仍然要付出约 2.3× 的代价(microbench:0.246→0.105 ms/layer)。

## 实测结果

| 指标 | S10 | S12 | Δ |
|---|---|---|---|
| conc=32 | 589 | **608.8** | **+3.4%**(84.4%→87.2% of vLLM) |
| conc=32 TPOT | 54 ms | 52.1 ms | — |
| conc=1 | 45.7 | **51.3** | **+12.2%** |
| conc=1 TPOT | 22 ms | 19.6 ms | — |

Greedy 输出验证 bit-coherent(Rayleigh 问题答案 ok);独立的数值检查 vs 参考 kernel:对 KVLEN ∈ {1..2000},max|Δ| ≤ 1e-4(bf16 噪声)。

**为什么 conc=1 比 conc=32 收益更大**:单 request 的 KV(~4.7 MB/layer)能装进 L2,所以那 4× 重读是纯浪费的 L2 bandwidth——被干净地消除了。在 conc=32 时 KV(151 MB/layer)溢出 L2(更偏 HBM-bound),而且 1024/128 workload 是 prefill-heavy 的(attn_decode 只占 ~13% 的 GPU 时间),所以端到端占比更小。

## 经验

一个 kernel 级别的加速(2.3×)在端到端被稀释到 +3.4%,因为在 prefill-heavy 的 workload 上 decode attention 只是 GPU 时间的少数派。

剩下还在手写的热点 kernel 是 **attn_prefill**(~11%),它仍然按 GQA group 4× 重读 K/V——这是下一个 attention lever。
