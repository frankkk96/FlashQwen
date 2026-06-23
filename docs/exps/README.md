# FlashQwen 优化实验详细记录

这里是每个优化步骤的**详细实验记录**(全中文,一实验一文档)。
总览见上层 [`../optimization.md`](../optimization.md)(英文)/ [`../optimization.zh.md`](../optimization.zh.md)(中文)。

口径:单张 RTX 4090,Qwen3-8B,bf16 严格对齐(无量化),`vllm bench serve` random 数据集,
out=128,conc=32,追踪指标为 1024-input 的 Output token throughput (tok/s)。

## 索引(按时间顺序)

| 步骤 | 文档 | 一句话 |
|---|---|---|
| B0/R1/R2 | [00-baseline-and-scheduler](00-baseline-and-scheduler.md) | 起点 = main(INT8、按序列调度);统一 token 预算调度器 + GPU 采样 |
| S3 | [s03-bf16-cublas-fa2](s03-bf16-cublas-fa2.md) | bf16 + cuBLAS GEMM + FlashAttention-2 |
| S4 | [s04-tensorcore-gqa-reverted](s04-tensorcore-gqa-reverted.md) | tensor-core + GQA 分组 attention(尝试,已回退) |
| S5 | [s05-kernel-fusion](s05-kernel-fusion.md) | 算子融合(QKV/gate-up GEMM、add+rmsnorm、RoPE 表) |
| S6 | [s06-flashdecoding](s06-flashdecoding.md) | FlashDecoding decode-attention kernel |
| S7 | [s07-wmma-prefill](s07-wmma-prefill.md) | WMMA tensor-core prefill attention |
| S8 | [s08-prefill-occupancy](s08-prefill-occupancy.md) | prefill 占用率:压缩 shared memory |
| S10 | [s10-max-batch-tokens](s10-max-batch-tokens.md) | 调度器默认 max-batch-tokens 2048 → 1024 |
| S11 | [s11-cuda-graphs-reverted](s11-cuda-graphs-reverted.md) | 纯 decode CUDA Graphs(尝试,已回退) |
| S12 | [s12-gqa-flashdecoding](s12-gqa-flashdecoding.md) | GQA-shared FlashDecoding decode attention |
| S14 | [s14-activation-scratch](s14-activation-scratch.md) | activation scratch 按真实每步上界裁剪 |
| S15 | [s15-prefix-caching](s15-prefix-caching.md) | 自动 prefix caching(分支 feat/prefix-caching) |
| S16 | [s16-prefill-mma-rewrite](s16-prefill-mma-rewrite.md) | prefill attention 重写:WMMA → mma.sync |
| S17 | [s17-comprehensive-comparison](s17-comprehensive-comparison.md) | 与 vLLM 全负载/全数据综合对比 + prefix-cache 修复 |
