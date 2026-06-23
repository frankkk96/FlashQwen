# S16 — prefill attention 重写:WMMA → mma.sync(FlashAttention 风格)

> branch `feat/prefix-caching`,2026-06-23。这是整个优化历程里最深的一次 kernel 重写之一:把 prefill attention 从 `nvcuda::wmma` 改写成裸 `mma.sync`、寄存器 O 累加器、block-shared K/V 复用、cp.async 流水线的 FlashAttention 风格实现,同时记录了一条被明确否决的 lever(prefill split-K)。

## 背景/动机

到 S14 为止,FlashQwen 在 bf16 parity 下已经做到 conc=32/1024 = 605 tok/s ≈ vLLM(no prefix cache)的 92.8%。从 journey replay 的曲线看,**与 vLLM 的差距随输入长度单调增大**(in=128 97.5% → in=512 96.2% → in=1024 92.8%),残差集中在长输入,也就是 **prefill 侧的 compute**。

一次针对 tracked 1024/conc32 步(prefix-cache on)的 nsys 重新 profile,把 GPU 时间按 kernel 拆开:

**nsys GPU kernel-time 拆分(steady-state,`cuda_gpu_kern_sum`):**
- **GEMM ~72%**(cuBLAS/CUTLASS ampere/cutlass tensorop bf16;prefill big-M compute-bound + decode M=32 HBM-bound — 两者都在 cuBLAS ceiling)
- **prefill attention 11.3%**(`attn_prefill_kernel`)
- **decode attention 11.8%**(`attn_decode_gqa_kernel` + combine;S12 已优化,KV-bandwidth-bound)
- **elementwise ~4.6%**(add_rmsnorm 1.8 / silu_mul 1.3 / head_norm_rope 0.9 / store_kv 0.3 / …)

这把过去那个混合负载下的 71/24/4.6 进一步细化:那 24% 的 attention 大约是一半 prefill、一半 decode。

关键结论:**GEMM 是与 vLLM 完全相同的 cuBLAS 路径**(两边跑的是同一批 `ampere_*_s1688gemm_*` kernel,同样 ~1.12 ms/call),所以唯一还有 headroom、且 on-parity 可处理的手写热点就是 attention。直接 trace vLLM 也证实了这一点:vLLM 跑的是 **一个** 融合的 `flash_fwd_splitkv`(FlashAttention)kernel,占 ~10%,而 FlashQwen 是 **两个** 手写 kernel(prefill + decode),合计约 23%。

> nsys 注意事项:系统自带的 nsys 2021.3 在本机 glibc 上注入失败(`__libc_dlclose GLIBC_PRIVATE`);要用 Nsight Compute 自带的 2025 版:`/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys`。

## 改动

### 先试后弃 —— prefill split-K(`FQ_PREFILL_KSPLIT`),确认为 NULL,已 revert

在动手做真正的重写之前,先验证了仅剩的一条 on-parity attention lever:把 decode 的 KV-split(FlashDecoding)照搬到 WMMA prefill kernel 上。

- phase-1 `attn_prefill_splitk_kernel`:把 `[0,qpos]` 的 key tile 跨 `ksplit` 个 block 分条(grid.z 打包 `req*ksplit`),每个 block 写出未归一化的 (m,l,O) partial(按 global query row)。
- phase-2 `attn_prefill_combine_kernel`:合并这些 partial。
- 由 env `FQ_PREFILL_KSPLIT` 控制(default 1 = base path)。

A/B(ksplit=1 vs 4,prefix-cache on,同 vllm-bench 口径,`/root/prof/splitk_test.sh`):

1024/c32 619→622,512/c32 938→934,128/c32 1351→1346,1024/c1 51.2→51.3,shared-prefix 831→835 —— **全部落在 ±0.5% noise 内,在每个 input/concurrency 上都是 0 e2e 收益。**(greedy text:2 个 prompt 里 1 个是 coherent paraphrase,属 strided-sum softmax merge 的 FP non-associativity,与 decode 已记录的 max|Δ|≤1e-4 同类 —— 不是 bug;这个 diff 同时也证明 split 路径确实被激活了。)

**为什么是 null(真正的 takeaway):** prefill-attn 的 grid = qtiles × n_heads × R,**即便在 conc=1 也是数千个独立 block**(1024-in 单请求 = 64 qtiles × 32 heads = 2048 blocks ≫ 4090 上约 1150 个 resident block)。GPU **从来不缺 block**,所以拆每个 block 的串行 KV chain 是在攻击一个非瓶颈。prefill-attn 的墙是 **per-block occupancy**(~11 KB shared、1 warp、~19% occ),而 split-K 不改变 shared/occupancy。ksplit 越大只会增加 combine 开销。**Lever 关闭:** 唯一 on-parity 能撬动 prefill-attn 的办法,是在 *不复制* `Os[16,128]` 累加器的前提下减少 per-block shared 或增加 warp —— 正是 S8/S13 那堵 occupancy 墙。

这条结论与历史一致:S8(prefill occupancy-bound;赢在 *缩小* shared)、S9(head-dim split 加 warp → 持平)、S13(GQA-shared prefill 0.73× *更慢*)。

### 真正的修复 —— `attn_prefill_kernel`(WMMA)→ `attn_prefill_mma_kernel`(`engine/src/kernels.cu`)

把旧的 WMMA(1 warp/block、O in shared)kernel 重写成 FlashAttention 风格,移植了 FA 的 4 项核心技术到 sm_89:

1. **O 累加器放进寄存器**,用裸 `mma.sync.m16n8k16.f32.bf16.bf16.f32` 的 C-fragment(而不是 `nvcuda::wmma`)。这释放掉了旧的 8 KB shared `Os[16][128]` —— 那正是 occupancy 墙;并且让 S 也留在寄存器里(消掉 per-tile 的 shared round-trip 和大部分 `__syncthreads`)。
2. **64-row M tile = 4 warps/block**,每个 warp 拥有 16 个 query row。
3. **K/V 每 16-key tile 只 stage 到 block-shared 一次,被 4 个 warp 共用**(KV global 流量约 4× 减少 —— 这是单项最大的 lever)。P 通过一个 tiny per-warp shared buffer 传递,以跳过 C→A 的 fragment repack;online softmax 在寄存器里用 quad `__shfl_xor` 完成。
4. **cp.async double-buffer** K/V 流(`__pipeline_memcpy_async` + `__pipeline_commit/wait_prior`):在计算 tile n 的同时预取 tile n+1。K/V 按 NATURAL 布局 stage(cp.async 不能转置);V 在 P@V 的 B-operand 里用转置索引读取。

**验证:** standalone microbench(`/root/prof/attn_bench/bench.cu`)对 fp32 reference,含所有非 64 倍数尺寸:与 WMMA kernel 数值一致(max|diff| ~7e-4,bf16 量级)。

**Microbench speedup:~2.5× 不带 cp.async,~2.8× 带 cp.async**(1024:WMMA 0.542 → mma 0.218 → +cp.async 0.192 ms;cp.async 在 kernel 上加了约 17%)。

通过 `FQ_PREFILL_V2` 开关(default 1);若 head_dim≠128 或 block_size≠16,或 `FQ_PREFILL_V2=0`,回退到 WMMA。

## 实测结果

同 session A/B(`FQ_PREFILL_V2` 0=WMMA / 1=mma),conc=32 output tok/s,两种 cache 状态都测。no-cache 的 S14/WMMA 行复现了历史 journey replay(606 vs 605 @1024;vLLM 652 完全一致),证明二者直接可比。

*No prefix cache(feature-matched,延续 journey-replay 表):*

| in | S14 (WMMA) | **S16 (mma)** | ΔFQ | vLLM | S16 % of vLLM |
|---|---|---|---|---|---|
| 128  | 1315 | 1318 | +0.2% | 1393 | 94.6% |
| 512  | 913  | 927  | +1.5% | 948  | 97.7% |
| **1024** | **606** | **640** | **+5.6%** | **652** | **93.0% → 98.1%** |

*Prefix cache ON(branch default / 部署模式):*

| in | S15 (WMMA+cache) | **S16 (mma+cache)** | ΔFQ | vLLM+cache | S16 % of vLLM |
|---|---|---|---|---|---|
| 128  | 1351 | 1352 | ~0    | 1407 | 96.1% |
| 512  | 940  | 955  | +1.7% | 982  | 97.3% |
| **1024** | **621** | **655** | **+5.5%** | **681** | **92.4% → 96.2%** |
| 512 shared + 512 random | 831 | 870 | +4.7% | — | — |

memory 文件里的 e2e A/B(`FQ_PREFILL_V2` 0/1,prefix-cache on,`/root/prof/v2_test.sh`)记录:
- **1024/conc32:621 → 655 tok/s(+5.5%)**,TPOT 51.0 → 48.3 ms → **96.2% of vLLM(681)**(原 92.4%)
- 512/conc32:940 → 955(+1.7%)→ 97.3% of vLLM;shared-prefix 831 → 870(+4.7%)
- 128/conc32 与所有 conc=1:持平(prefill-attn 在那里只占很小一片)—— 符合预期。

in=128 和所有 conc=1 持平(prefill-attn 在那些场景是可忽略的一片),符合预期。这次重写正好抬升了它所针对的 long-context、prefill-heavy 区间:**在 1024/conc32 上,与 vLLM 的差距从 ~7% 收到 ~2%(no-cache)/ ~4%(cache-on)**,是 bf16 parity 下有史以来最小的。

cp.async 相对不带 cp.async 的 mma 只多了约 +0.9% e2e(650 → 655):因为前面那 2.5× 已经把 prefill-attn 的占比缩小了,后续是 diminishing returns。greedy 正确性:长 prompt 输出 bit-identical;一个短 prompt 翻转了一个 near-tie 的列表项单词(reduction order 不同导致的 FP non-associativity,与 decode S12 / split-K 同类)—— 不是 bug。

## 经验

- **microbench 2.8× 精确地稀释成 +5.5% e2e,正如 profile 预测**:prefill-attn 占 GPU ~11%,被砍半 → 实现 ~5%。这是 "kernel 隔离加速会被它在端到端里的占比稀释" 这条规律的又一次印证(同 S12)。
- cp.async 的 overlap 只值约 0.9% e2e —— 这一点后来在 S17 的 bug 修复里很关键:正是它引入了一个跨 kernel 的异步时序 hazard,而砍掉它(改成 single-buffer 同步 staging)几乎不损失性能。
- **decode 故意没动、也故意没和 prefill 统一成一个 mma kernel。** 读了 FA 的 decode(splitkv)代码:它复用了和 prefill **同一个** mma+shared+cp.async mainloop(decode = "1-row M tile 的 prefill + KV split")。但把 FlashQwen 的两个 attention kernel 合并成一个 mma mainloop 会让 decode **回退**:
  - (a) mma 在 q_len=1 上浪费 m16 tile 的 15/16(S4 教训:WMMA-on-decode 曾 −34%);
  - (b) 它会丢掉 FlashQwen decode 的 GQA register-reuse(K/V 加载一次,在 group 的 4 个 q-head 间复用;FA 则经 L2 按 q-head 重读)。
  decode 是 memory-bound,FA 为了 library 可维护性接受这份 mma 浪费;而手写引擎更应该保留两个专用 kernel(S6 的 split-by-request-type:prefill compute-bound → tensor cores;decode memory-bound → SIMT + GQA-reuse + ksplit)。况且当前 decode kernel 根本无法 "加上" cp.async —— 它是 global→registers 读 K/V、没有 shared staging,而 cp.async 是 global→shared 指令。
- 完整的 FA-vs-FlashQwen 对照(FA 的 8 个 focal point + FlashQwen 的差距)见 [`docs/attention-vs-flashattention.md`](../attention-vs-flashattention.md)。

**状态:default ON,已集成,保留。** head_dim≠128 / block_size≠16 / `FQ_PREFILL_V2=0` 时回退 WMMA。
