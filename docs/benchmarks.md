# 基准测试记录（简）

> 简单索引。**当前提交的详细快照**（环境 / 配置 / vLLM 版本 / 结果）见仓库根目录的 [`benchmark.md`](../benchmark.md)。
> 每步优化的**过程记录**见 [`exps/`](exps/)（S3→S19，一实验一文档，索引见 `exps/README.md`）。

口径：单张 RTX 4090，Qwen3-8B，bf16 严格对齐（无量化），`vllm bench serve`。追踪指标为 Output token throughput（tok/s，%vLLM）。

## 里程碑

| 日期 | 分支 / commit | 变更 | vs vLLM |
|---|---|---|---|
| 2026-06-23 | `feat/prefix-caching` | S17 全负载/全数据宽口径对比 + prefix-cache 修复 | 96–98%；in=2048 91.9%；ShareGPT 96.8%（等功修正后） |
| 2026-06-29 | `feat/cutlass-gemm-attn` (→main) | S18 CuTe attention + S19 decode graph 转正 | **97.6 / 100.1 / 99.6%**（in=128/512/1024） |
| 2026-07-16 | `feat/cutlass-fmha` | CUTLASS 转 submodule + layered FMHA 重构 | 逐字节相同、serving 持平 |
| 2026-07-22 | `feat/cutlass-fmha` @ `6a216d4` | CUTLASS 拉取搬进 CMake、Makefile 瘦身、attn 去 `fmha` 前缀；新增全网格 bench（`scripts/bench_grid.sh`，并发×输入长度×数据） | 本提交实测 vs vLLM 0.23.0：18 格中 16 格 **95–101%**；弱点 long-context prefill 低并发（in=2048@c1 **88.3%**）；ShareGPT c32 **97.4%**；conc≥8 时 FQ TTFT 全线更低 → 详见根 `benchmark.md` |
| 2026-08-05 | `feat/cutlass-fmha` @ `406409c` | prefill P 常驻寄存器（FA2 风格，去 smem 中转）+ `attn_sm80.cu` 注释/命名清理（逐字节相同）；全网格复测 | vs vLLM 0.23.0：19 格中 18 格 **95–101%**；唯一 <95% 为 in=2048@c1 **88.2%**；ShareGPT c32 **97.5%**；数字与 `6a216d4` 持平（decode-bound，P-in-reg serving-flat） → 详见根 `benchmark.md` |
