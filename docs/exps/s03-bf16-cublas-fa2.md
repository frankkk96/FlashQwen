# S3 — bf16 + cuBLAS GEMM + FlashAttention-2

## 背景/动机

R1/R2 已经确认:unified-scheduler 的回归不是调度的问题,而是 **kernel 欠下的债**——尤其是 R1 引入的 no-split-K varlen attention kernel,以及 INT8 weight-quant + 手写 GEMM 路径。S3 是针对 R1/R2 指出的这些债务,做三个协调一致的 kernel 重写。

commit:7650654。

## 改动

三个协调一致的 kernel 重写:

1. **丢弃 INT8 weight-quant → 全程 bf16。** 权重以 bf16 加载(不再 dequant),整条 activation pipeline 也是 bf16(norm/rope/residual/silu 内部用 fp32 累加)。这样就和 vLLM 做到了**公平的 bf16-parity**——之前的精度 caveat 没有了。

2. **所有 matmul 都走 cuBLAS。** 一条 `cublasGemmEx` 路径(bf16 输入,fp32 累加,tensor cores)替换掉手写的 WMMA prefill GEMM + 每次调用的 full-weight dequant + batched INT8 decode GEMV。decode/prefill matmul 的拆分被删除——merged batch 只走一条路径。

3. **FlashAttention-2 paged kernel。** 替换掉那个 per-(head,row) 的 varlen kernel(它从 global 重新读全部 K/V、零复用、O(T²))。新 kernel:one block per (q-tile, head, request);BM=16 query rows(每行一个 warp)共享 staged 在 shared memory 里的 BLOCK-sized K/V tiles(复用因子 BM),online softmax 带 deferred normalization(只在最后才除以 running sum),causal whole-tile skip。Host 端按 request(qstart/qlen)对 batch 分组以形成 q-tile grid。

## 实测结果

- **vs R2**:conc=32 **87.4 → 350.0 tok/s(4.0×)**——**12.5% → 50.1% of vLLM**,是 `main` baseline(106.9)的 3.3×。
- conc=1:24.9 → 38.2(仍低于 main 的 INT8 56.7:bf16 权重读取的字节数是 INT8 的 2×,而单流 decode 是 weight-bandwidth-bound——这是预期内的 de-quant 代价;跟踪指标是 conc=32)。
- Greedy generation 验证连贯(coherent)。

Progress 表对应行(canonical 2026-06-20 replay):

| Step | change | conc=1 | conc=32 | % vLLM | commit |
|---|---|---|---|---|---|
| S3 | bf16 weights + cuBLAS GEMM + FlashAttention-2 | 37.9 | **348** | **53%** | 7650654 |

Final-results(conc=32 output tok/s,跨输入长度,括号内为 % of feature-matched vLLM):

| step | in=128 | in=512 | in=1024 | what this step bought |
|---|---|---|---|---|
| **S3** | 1094 (79.5%) | 628 (66.5%) | 348 (53.4%) | **bf16 + cuBLAS + FA2 — the engine; ~4× everywhere** |

## 经验

- R1/R2 是对的——差距在 kernel,不在 scheduling。unified-scheduler 的回归不仅被收回,还被远远甩在身后。
- conc=32 剩余的 ~2× 差距**不是 attention**(见 S4)——而是 GEMM(cuBLAS,接近最优)+ 每步的 launch/scheduling 开销(每步 ~720 个 kernel launch,没有 CUDA graphs、没有 persistent batch、没有 prefix caching)。
- 单流(conc=1)是另一个独立的、bandwidth-bound 的问题(bf16 weight traffic),如果以后要关心的话再说。
