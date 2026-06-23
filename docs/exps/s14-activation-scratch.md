# S14 — activation scratch 按真实每步上界裁剪

## 背景/动机

`max_rows_`(所有 activation buffer 的行数都按它来 size)原本是 `max(max_ctx=4096, max_batch_tokens=1024)` = 4096。但一次前向永远不会超过 `max_batch_tokens` 行——调度器的 per-step budget 把 T 卡住,prefill 也被 chunk 到这个预算之下。换句话说,activation scratch 一直按一个永远用不到的上界来分配,白白占用了显存,而这部分显存本可以让给 KV pool。

这也顺带是验证“KV cache 容量 / eviction 到底是不是 conc=32 瓶颈”这一假说的一次干净实验(此前 S10 用过 `--gpu-mem-fraction` 提升 pool,但那是不公平的内存加注;S14 是公平的内存修复)。

## 改动

把 `max_rows_` 改为按 `max_batch_tokens + 16` 来 size。

**为什么是 +16**:这暴露了一个潜伏的 OOB——WMMA prefill-attention 的 `load_matrix_sync` 会读取完整的 16 行 Q tile,所以一个非 16 对齐的末尾 chunk 会越界读取 T 之后最多 15 行(本身无害,被 `grow<ql` mask 掉)。旧的 4096 sizing 把它吸收掉了;而精确 size 到 1024 会让 T=1024 越过 buffer 越界读取 → illegal memory access(model_runtime.cu:272)。+16 给出一个 Q-tile 的 headroom。

## 实测结果

| 指标 | S12 | S14 | 说明 |
|---|---|---|---|
| weights+activations | 17.3 GB | 17.0 GB | 回收 0.36 GB |
| KV pool | 36,656 tokens(2291 blocks) | **39,040 tokens(2440 blocks)** | 现已 **高于** conc=32/1024 峰值需求(36,864) → KV-capacity cliff 被消除 |
| conc=32 / 512 | 912.8 | 915.0 | 持平(噪声) |
| conc=32 / 1024 | 605.6 | 606.0 | 持平(噪声) |

输出 coherent。

## 经验(给 KV 假说一个定论)

把 pool 增长到超过峰值需求——保证零 preemption——对吞吐的影响是 **0**。**KV cache 大小 / eviction 不是 conc=32 瓶颈**(在 S12 之后重新确认了 S10 的结论,而且这次用的是公平的内存修复,而非不公平的 gpu-mem 加注)。

这个改动仍然保留:它修掉了潜伏的 OOB,回收了 0.36 GB(pool 现在 ≈ vLLM 的 40,816),并为更高并发 / 更长 context 下的健壮性消除了 cliff。
