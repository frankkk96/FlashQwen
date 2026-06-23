# S5 — 算子融合(QKV/gate-up GEMM、add+rmsnorm、RoPE 表)

## 背景/动机

S3 把 conc=32 剩余 lever 之一指认为每步的 launch + 冗余 traffic 开销(每步 ~720 个 kernel launch)。S5 针对这一点做三个融合,全部是 bf16 等价数学(bf16-identical math)。

commit:c2f98a0。

## 改动

1. **融合 QKV 和 gate|up GEMM。** q/k/v proj 权重在加载时按 OUT 维拼接为一个 `wqkv`([q_dim+2·kv_dim, H]),同样 gate+up 拼成 `wgateup`([2·I, H]);各用一个 `cublasGemmEx` 替换原来的 3 个和 2 个。下游 kernel 通过 offset/stride 读取交错的 fused buffer:一个新的 fused **per-head RMSNorm+RoPE** kernel 就地 normalize+rotate q 和 k 切片,`store_kv` 直接从 QKV buffer 读 k/v,attention 接受一个 `q_stride`,`silu_mul` 把 gate/up 当作一行的两半来读。

2. **融合 residual-add + RMSNorm。** `add_rmsnorm` 在一趟里做 `x += residual`(写回,把 residual 往前带)然后 `rmsnorm(x)`——替换掉分开的 `add` + `rmsnorm`,省掉对 x 的一次完整 H read/write 以及每个 residual 一次 launch。run_layers 被重构成:除第一个外的每个 norm 都消费 pending residual;一个收尾的 `add` 提交最后一层 MLP 的 residual。

3. **预计算 RoPE cos/sin 表。** 角度只依赖 (pos, i),且在全部 36 层都相同;旧 kernel 每层、每元素都重算 `powf/cosf/sinf`(36× 浪费)。现在在启动时一次性构建 `[max_ctx, head_dim/2]` 表并查表(折叠进 fused norm+rope kernel)。

净效果:每层 ~19 → 12 个 kernel launch(每步少 ~252 个,~720 → ~470)。

## 实测结果

- **vs S3**:conc=1 38.2 → **39.1(+2.3%)**;conc=32 350.0 → **356.1(+1.7%,在噪声内)**。Greedy 输出验证连贯。

Progress 表对应行:

| Step | change | conc=1 | conc=32 | % vLLM | commit |
|---|---|---|---|---|---|
| S5 | kernel fusion: fused QKV/gate-up GEMM, add+rmsnorm, RoPE table | 38.8 | 356 | 55% | c2f98a0 |

Final-results 对应行:

| step | in=128 | in=512 | in=1024 | what this step bought |
|---|---|---|---|---|
| S5 | 1141 (82.9%) | 648 (68.6%) | 356 (54.6%) | fused QKV/gate-up GEMM (the only fusion that paid) |

### S5 消融实验——三个融合里到底哪个真正有效(2026-06-19)

每个改动在 S3 base 上单独隔离(1024/128,temp 0);只有 engine 不同。

| variant | conc=1 | conc=32 | vs S3 (conc=32) |
|---|---|---|---|
| S3 base | 38.2 | 350.0 | — |
| A: #5 RoPE cos/sin table only | 38.2 | 350.2 | ~0 (noise) |
| B: #7 add+rmsnorm fusion only | 38.3 | 350.0 | ~0 (noise) |
| C: #6 fused QKV/gate-up GEMM only | 38.9 | 355.8 | **+1.7%** |
| S5: all three | 39.1 | 356.1 | +1.7% |

**结论:没有一个是回归,但 #5 和 #7 是 net-zero——整个 S5 的收益都来自 #6(fused GEMMs)。** 为什么 #5/#7 在这里是 placebo:RoPE 和 add/rmsnorm 这些 elementwise kernel 相比 GEMM 微不足道,而在 conc=32 这一步是 GEMM/HBM-bound 的,所以消除它们的 launch + 冗余 traffic 不动 wall clock。#6 之所以有效:把 3→1、2→1 地折叠 GEMM,给了 cuBLAS 更宽的 N(在 skinny M=32 的 decode shape 上获得更好的 tensor-core 利用率)外加更少 launch。在 #6 之上再叠加 #5+#7(C→S5)什么都没变(355.8→356.1)。三个都保留:正确、不负面、为后续 CUDA-graph capture 减少 launch——但教训是:**在这个 batch size 下,GEMM-shape 的赢是 lever,elementwise/launch 融合不是。**

## 经验

- 去掉 ~250 launch/step + 冗余 elementwise traffic 几乎不动 conc=32——所以在 conc=32 下,launch/elementwise 开销只占**很小一部分**;真正主导的是 skinny 的 decode **GEMM**(每步要从 HBM 流式读取全部 17 GB 的 bf16 权重)。
- 这重新校准了 roadmap:"~720 launches" 这个数字高估了这个 lever。**CUDA graphs 对 conc=1(latency-bound)的帮助会大于 conc=32**;在 conc=32 收掉剩余的 2× 需要 GEMM-side 的赢(cuBLASLt autotuning / 更好的 decode-GEMM shape)或者 vLLM 拥有的吞吐特性(更大的 effective batch、prefix caching),而不是只靠 launch elision。
- 融合无论如何都保留:正确、更干净,并预先铺好了 CUDA-graph capture 将会需要的 persistent-buffer 布局。
