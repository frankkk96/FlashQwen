// 独立微基准 + 正确性 A/B：喂同一份随机输入给 LaunchAttnPrefill，
// 打印输出 checksum（跨版本应一致）和 kernel 平均耗时。
// 用于对拍不同 attn_sm80.cu 版本（例如同步 vs cp.async 多级流水）的 checksum、对比耗时。
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "attn.h"
#include "model_spec.h"

using S = fq::ModelSpec;

#define CK(x)                                                               \
  do {                                                                      \
    cudaError_t e = (x);                                                    \
    if (e != cudaSuccess) {                                                 \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__,   \
             __LINE__);                                                     \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

// 确定性伪随机（LCG），[-1,1)，保证两版本喂到完全相同的输入。
static float lcg(unsigned& s) {
  s = s * 1664525u + 1013904223u;
  return (s >> 8) * (1.0f / 8388608.0f) - 1.0f;  // 24-bit → [-1,1)
}

int main(int argc, char** argv) {
  int L = argc > 1 ? atoi(argv[1]) : 1024;   // 序列长度（prefill token 数）
  int iters = argc > 2 ? atoi(argv[2]) : 200;

  const int kHead = S::kHeadDim, kNH = S::kNumHeads, kNKV = S::kNumKvHeads;
  const int kKvB = S::kKvBlock, kPlanes = 2;
  const int kv_dim = kNKV * kHead;                    // 1024
  const int qkv_dim = kNH * kHead + 2 * kv_dim;       // 6144
  const int out_dim = kNH * kHead;                    // 4096
  const int plane = kKvB * kv_dim;                    // 16384
  const int page = kPlanes * plane;                   // 32768
  const int npages = (L + kKvB - 1) / kKvB;

  size_t qN = (size_t)L * qkv_dim, kvN = (size_t)npages * page, oN = (size_t)L * out_dim;

  // ---- host 侧填充 ----
  std::vector<__nv_bfloat16> hq(qN), hkv(kvN), hout(oN);
  unsigned s = 12345u;
  for (auto& v : hq) v = __float2bfloat16(lcg(s));
  for (auto& v : hkv) v = __float2bfloat16(lcg(s));
  std::vector<int> hpos(L), hbt(npages), hqs{0}, hql{L}, hr{0};
  for (int i = 0; i < L; i++) hpos[i] = i;
  for (int i = 0; i < npages; i++) hbt[i] = i;

  // ---- device 侧 ----
  __nv_bfloat16 *dq, *dkv, *dout;
  int *dpos, *dbt, *dqs, *dql, *dr;
  CK(cudaMalloc(&dq, qN * 2));   CK(cudaMalloc(&dkv, kvN * 2));  CK(cudaMalloc(&dout, oN * 2));
  CK(cudaMalloc(&dpos, L * 4));  CK(cudaMalloc(&dbt, npages * 4));
  CK(cudaMalloc(&dqs, 4));       CK(cudaMalloc(&dql, 4));        CK(cudaMalloc(&dr, 4));
  CK(cudaMemcpy(dq, hq.data(), qN * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dkv, hkv.data(), kvN * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dpos, hpos.data(), L * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dbt, hbt.data(), npages * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dqs, hqs.data(), 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dql, hql.data(), 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dr, hr.data(), 4, cudaMemcpyHostToDevice));

  auto run = [&]() {
    fq::LaunchAttnPrefill(dq, dkv, dout, dpos, dqs, dql, dr, /*R=*/1,
                              /*max_qlen=*/L, dbt, /*max_blocks=*/npages, 0);
  };

  // ---- 正确性 checksum ----
  run();
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  CK(cudaMemcpy(hout.data(), dout, oN * 2, cudaMemcpyDeviceToHost));
  double sum = 0, sumabs = 0;
  for (auto v : hout) { float f = __bfloat162float(v); sum += f; sumabs += f < 0 ? -f : f; }

  // ---- 计时 ----
  for (int i = 0; i < 20; i++) run();  // warmup
  CK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a);
  for (int i = 0; i < iters; i++) run();
  cudaEventRecord(b);
  CK(cudaEventSynchronize(b));
  float ms = 0; cudaEventElapsedTime(&ms, a, b);

  printf("L=%d npages=%d iters=%d | checksum sum=%.6f sumabs=%.6f | avg %.4f ms/call\n",
         L, npages, iters, sum, sumabs, ms / iters);
  return 0;
}
