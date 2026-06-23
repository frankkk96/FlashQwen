# S15 — 自动 prefix caching(分支 `feat/prefix-caching`)

## 背景/动机

vLLM 风格的 content-hashed KV reuse:相同的 prompt 前缀只 prefill 一次,把缓存下来的 KV blocks 拼接(splice)到后续请求上。

这一步落地了 Conclusions 中曾被搁置为“绕过差距的 off-table 路线”(route #1)的特性,并且把长期存在的 caveat(“我们的 headline 差距是对着一个 *带* prefix caching 的 vLLM 测出来的,而我们没有这个特性”)转化为一次 feature-matched、两边都开的对比。

commit:`fad12b7`,2026-06-22。

## 改动

三层结构,镜像 vLLM:

- **BlockPool** 增加 per-block refcount + LRU free queue + 一个 content-hash→block registry。一个 refcount-0 的 block 可被回收但仍保留(hash mapping 保留),这样一个相同的前缀可以让它“复活”;而一次新的 alloc 会从 LRU 队首弹出并 evict 掉那个 block 的 mapping。
- **Scheduler** 在 (prompt ++ output) 上链式计算 64-bit block hash;admission 时 `acquire_prefix()` 把缓存的 prefix blocks 拼接到 block table 上,并把 `computed_` 推进到它们之后(始终至少留 ≥1 token);每次前向之后 `cache_filled()` 注册新写满的 block(一个完成的 request 的 blocks 会为下一个相同 prompt 保留缓存;一个被 preempt 的 request 在 resume 时重新 acquire 它自己仍被缓存的前缀)。
- 正确性依赖 absolute-position KV addressing + 16 对齐的整 block 复用,因此 chunked-prefill 的前向路径保持不变。
- 默认开启;**`FQ_PREFIX_CACHE=0` 关闭它**用于 A/B;`[kvstat]` 现在会报告 prefix hit rate。

## 实测结果

**A/B(2026-06-22,口径与 journey replay 相同:`vllm bench serve --dataset-name random`,out=128,temp 0,chat endpoint,enable_thinking=false;harness `/root/bench-compare/prefix_test.sh`)。conc=32 output tok/s,各引擎 prefix cache OFF → ON:**

| workload (conc=32) | FQ off → on | ΔFQ | vLLM off → on | ΔvLLM | FQ-on / vLLM-on |
|---|---|---|---|---|---|
| 128 random | 1350 → 1359 | +0.6% | 1406 → 1407 | ~0 | 96.6% |
| 512 random | 920 → 953 | +3.6% | 962 → 982 | +2.0% | 97.1% |
| 1024 random | 615 → 630 | +2.4% | 668 → 681 | +1.9% | 92.4% |
| **512 shared + 512 random** | **621 → 844** | **+35.9%** | **666 → 915** | **+37.4%** | **92.2%** |

conc=1(单流)两边都基本持平——因为只有 template prefix 被共享:FQ 56/55/51 ≈ vLLM 58/57/56 的 97/96/92%。

prefix_hit 验证:FQ-off 确认为 0%;FQ-on 在 pure-random 上 ~8%,在 shared-prefix probe 下爬升到 ~22%——vLLM 报告的 hit rate 与之一致:~6% → ~19%。

> 注:TTFT 在两边不可比(FQ 在 prefill 前就发出 SSE role chunk → ~2–12ms 的 artifact;vLLM 报告的是真实的 queueing TTFT)。跟踪 output tok/s。

## 经验

- 在 *pure-random* prompt 上收益很小(+0.6–3.6%),因为唯一被共享的前缀是 Qwen3 chat-template 的 preamble(~8% 的 tokens)——而且在 conc=32 时瓶颈是 prefill/decode 计算,而不是省下的那点 prefill。
- 这个特性的真正价值出现在 prompt **真正共享前缀**时(system prompt、RAG context、few-shot、multi-turn):一个 512-token 的共享前缀带来 **+35.9%(FQ)≈ +37.4%(vLLM)**——FlashQwen 的 prefix caching **和 vLLM 一样有效**。
- 关键在于:现在两个引擎在特性上 feature-matched(*包括* prefix caching),FlashQwen 跨各 input 保持 **92–97% of vLLM**——与下方 no-cache journey replay **同一个 parity 带**。
- 这关闭了 prefix-caching 这条线:该特性不再是一个缺失的能力,剩余差距仍然是 Conclusions 中识别出的 prefill-side compute,而不是这个曾经缺席的特性。
- 这条结论同时关闭了 [[unified-scheduler-done-kernel-bound]] 中的 open caveat(“那个 84%/698 的差距是对着一个 *带* prefix caching 的 vLLM 测的,而 FQ 没有”):现在两边都有了,在 equal features 下 FlashQwen 仍是 **92–97% of vLLM**,与 no-cache 对比同一 parity 带;残差是同样的 prefill-side compute gap,而非缺失的特性。
- 这一步记录在 `docs/optimization.md` 作为 **step S15**(entry + A/B table),并把 Conclusions route #1 更新为 “LANDED”(2026-06-22)。README 仍引用 no-cache replay(尚未更新)。
- 这条线之后还能再走一步:cascade / shared-prefix attention(FlashInfer 的分解)——针对共享前缀之上的 *attention* 本身,而不仅仅是 KV reuse。
