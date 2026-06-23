# S8 — prefill attention 占用率：压缩 shared memory

## 背景/动机

S7 的 WMMA prefill kernel 是 1 warp/block，约 18 KB shared → 每个 SM 只能驻留约 5 个 block（约 8% warp occupancy）。warp 数量如此之少，导致在 per-tile 的 softmax（CUDA-core work）期间 tensor core 会 stall（停顿）——没有别的工作可以 overlap（重叠）。

## 改动

去掉 `[16,128]` 的 `PVt` 临时缓冲（8 KB）。原本的做法是"先 rescale 整个 O，再加上完整的 P·V"；改为把两者按每个 16×16 d-tile 经由一个小的 `[16,16]` temp 折叠处理：`O[:,d] = O[:,d]*corr + (P·V)[:,d]`。

每个 O 元素仍然只被 rescale 一次，也只获得一次它的 P·V——bit-identical（按位一致）的数学。

Shared 从约 18 KB → 约 11 KB → 每个 SM 约 9 个 block（驻留 warp 数约翻倍）。

## 实测结果

相对 S7：
- conc=32 **493 → 528 tok/s（+7.1%，TPOT 59.8 → 55.5 ms）**；
- conc=1 持平（44.9 → 45.2）。
- **71% → 76% of vLLM**，达到 `main` 基线的 4.94×。
- 输出连贯（coherent）。

commit: 392cda5。

在跨输入长度的 journey replay 中（conc=32 output tok/s，括号内为相对 feature-matched vLLM 的百分比）：

| step | in=128 | in=512 | in=1024 | 这一步带来了什么 |
|---|---|---|---|---|
| S8 | 1328 (96.5%) | 878 (93.0%) | 531 (81.5%) | prefill occupancy——提升长输入（@1024 +4.7pt） |

Progress 表中的对应行：

| Step | change | conc=1 | conc=32 | % vLLM | commit |
|---|---|---|---|---|---|
| S8 | attn_prefill occupancy (shrink shared, drop PVt) | 45.0 | **531** | **81%** | 392cda5 |

## 经验

一个 1 warp/block 的手写 WMMA kernel，即便 tensor-core throughput 看起来不错，也会受困于 occupancy（occupancy-starved）——释放 shared memory 让驻留 warp 数翻倍，挽回了 7%。

很可能还有更多 headroom（一个 multi-warp block / KV-split 会把 occupancy 推得更远），但那是更大的改写，伴随着惯常的 WMMA 正确性风险；这次廉价的 shared 削减已经把容易拿到的收益（easy gain）大部分都拿到了。
