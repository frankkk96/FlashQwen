#include "kernels.cuh"
#include "kv_cache.cuh"   // kv_phys_row: the paged-KV addressing contract (shared with block_pool.cu)

// ---------------------------------------------------------------------------------------
// rmsnorm: one block per row, block-reduction over H. BF16 in/out, FP32 reduction + scale.
// ---------------------------------------------------------------------------------------
__global__ void rmsnorm_kernel(const bf16* __restrict__ x, const float* __restrict__ w,
                               bf16* __restrict__ out, int M, int H, float eps) {
    int m = blockIdx.x;
    const bf16* xr = x + (size_t)m * H;
    bf16* outr = out + (size_t)m * H;

    extern __shared__ float red[];
    float local = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) { float v = __bfloat162float(xr[i]); local += v * v; }
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        outr[i] = __float2bfloat16(__bfloat162float(xr[i]) * inv * w[i]);
}

void launch_rmsnorm(const bf16* x, const float* w, bf16* out, int M, int H, float eps, cudaStream_t s) {
    int block = 256;
    rmsnorm_kernel<<<M, block, block * sizeof(float), s>>>(x, w, out, M, H, eps);
}

// ---------------------------------------------------------------------------------------
// fused residual-add + rmsnorm: x += res (stored back, carries the residual forward), out = rmsnorm(x)
// ---------------------------------------------------------------------------------------
__global__ void add_rmsnorm_kernel(bf16* __restrict__ x, const bf16* __restrict__ res,
                                   const float* __restrict__ w, bf16* __restrict__ out,
                                   int M, int H, float eps) {
    int m = blockIdx.x;
    bf16* xr = x + (size_t)m * H;
    const bf16* rr = res + (size_t)m * H;
    bf16* outr = out + (size_t)m * H;

    extern __shared__ float red[];
    float local = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = __bfloat162float(xr[i]) + __bfloat162float(rr[i]);
        xr[i] = __float2bfloat16(v);          // updated residual stays in x for the next layer
        local += v * v;
    }
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        outr[i] = __float2bfloat16(__bfloat162float(xr[i]) * inv * w[i]);
}

void launch_add_rmsnorm(bf16* x, const bf16* res, const float* w, bf16* out,
                        int M, int H, float eps, cudaStream_t s) {
    int block = 256;
    add_rmsnorm_kernel<<<M, block, block * sizeof(float), s>>>(x, res, w, out, M, H, eps);
}

// ---------------------------------------------------------------------------------------
// embedding gather (BF16 -> BF16, straight copy)
// ---------------------------------------------------------------------------------------
__global__ void embed_kernel(const int* __restrict__ ids, const bf16* __restrict__ embed,
                             bf16* __restrict__ out, int M, int H) {
    int m = blockIdx.x;
    const bf16* src = embed + (size_t)ids[m] * H;
    bf16* dst = out + (size_t)m * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) dst[i] = src[i];
}

void launch_embed(const int* ids, const bf16* embed, bf16* out, int M, int H, cudaStream_t s) {
    embed_kernel<<<M, 256, 0, s>>>(ids, embed, out, M, H);
}

// ---------------------------------------------------------------------------------------
// Fused per-head RMSNorm + RoPE (rotate-half, matching HF Qwen). Operates in place on one sub-tensor
// (q or k) of a fused QKV row of width `stride`: head (m,h) is at buf + m*stride + h*head_dim. The
// rotation angles come from precomputed cos/sin tables ([max_pos, head_dim/2]) indexed by pos[m] —
// no per-call transcendental, and they're identical across all 36 layers so the table is built once.
// One block per (token, head); blockDim == head_dim. FP32 math.
// ---------------------------------------------------------------------------------------
__global__ void head_norm_rope_kernel(bf16* __restrict__ buf, const float* __restrict__ w,
                                      const float* __restrict__ cos_tab, const float* __restrict__ sin_tab,
                                      const int* __restrict__ pos, int n_heads, int head_dim,
                                      int stride, float eps) {
    int row = blockIdx.x;            // over M * n_heads
    int m = row / n_heads, h = row % n_heads;
    int t = threadIdx.x;             // 0..head_dim-1
    int half = head_dim >> 1;
    bf16* v = buf + (size_t)m * stride + (size_t)h * head_dim;

    extern __shared__ float sh[];    // [0,head_dim): reduction then normed values
    float x = __bfloat162float(v[t]);
    sh[t] = x * x;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / head_dim + eps);
    __syncthreads();
    sh[t] = x * inv * w[t];          // normed value
    __syncthreads();

    if (t < half) {
        const float* cb = cos_tab + (size_t)pos[m] * half;
        const float* sb = sin_tab + (size_t)pos[m] * half;
        float cs = cb[t], sn = sb[t];
        float x1 = sh[t], x2 = sh[t + half];
        v[t]        = __float2bfloat16(x1 * cs - x2 * sn);
        v[t + half] = __float2bfloat16(x2 * cs + x1 * sn);
    }
}

void launch_head_norm_rope(bf16* buf, const float* w, const float* cos_tab, const float* sin_tab,
                           const int* pos, int M, int n_heads, int head_dim, int stride, float eps,
                           cudaStream_t s) {
    head_norm_rope_kernel<<<M * n_heads, head_dim, head_dim * sizeof(float), s>>>(
        buf, w, cos_tab, sin_tab, pos, n_heads, head_dim, stride, eps);
}

// ---------------------------------------------------------------------------------------
// FlashAttention-2 style paged varlen attention.
//
// One thread block = (query-tile blockIdx.x, query head blockIdx.y, request blockIdx.z). The block
// owns BM query rows of that request/head (one warp each, DPL = head_dim/32 dims per lane). It walks
// the request's keys in BLOCK-sized tiles: all threads cooperatively stage one tile of K and V (this
// kv-head's slice) into shared memory, then every query warp reuses that tile — this is the K/V
// reuse the per-row kernel lacked. Softmax is online in registers with deferred normalization (divide
// by the running sum only at the very end, FA2's fewer-non-matmul-ops trick). Causal: query row at
// pos p attends keys [0,p]; whole key tiles past the block's max query pos are skipped, and within a
// tile each warp stops at its own pos.
// ---------------------------------------------------------------------------------------
template<int BM, int DPL>
__global__ void attention_flash_kernel(const bf16* __restrict__ q, int q_stride,
                                       const bf16* __restrict__ cache_k,
                                       const bf16* __restrict__ cache_v, bf16* __restrict__ out,
                                       int n_heads, int n_kv, int head_dim,
                                       const int* __restrict__ pos, const int* __restrict__ qstart,
                                       const int* __restrict__ qlen, const int* __restrict__ bt,
                                       int max_blocks, int block_size, float scale) {
    int r     = blockIdx.z;            // request
    int h     = blockIdx.y;            // query head
    int qtile = blockIdx.x;            // query-row tile within the request
    int warp  = threadIdx.x >> 5;      // 0..BM-1: this warp's query row within the tile
    int lane  = threadIdx.x & 31;

    int ql = qlen[r];
    int row_in_req = qtile * BM + warp;
    bool active = row_in_req < ql;     // last tile may have idle warps (they still help stage K/V)

    int kvh = h / (n_heads / n_kv);
    int kv_dim = n_kv * head_dim;
    const int* btr = bt + (size_t)r * max_blocks;

    int flat = qstart[r] + row_in_req;            // this warp's flattened query-row index
    int qpos = active ? pos[flat] : -1;           // its absolute position (attends keys [0, qpos])

    float qreg[DPL], acc[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) acc[i] = 0.f;
    if (active) {
        const bf16* qv = q + (size_t)flat * q_stride + (size_t)h * head_dim;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) qreg[i] = __bfloat162float(qv[lane + (i << 5)]);
    } else {
        #pragma unroll
        for (int i = 0; i < DPL; ++i) qreg[i] = 0.f;
    }
    float m_run = -1e30f, l_run = 0.f;

    // Block-wide key extent: the max query pos over this tile's active rows (the last row's pos, since
    // pos increases within a request). Key tiles beyond it are fully causally masked for every warp.
    int last_row = min((qtile + 1) * BM, ql) - 1;
    int block_qpos_max = last_row >= qtile * BM ? pos[qstart[r] + last_row] : -1;
    int ntiles = (block_qpos_max + block_size) / block_size;   // ceil((block_qpos_max+1)/block_size)

    extern __shared__ bf16 smem[];
    bf16* ks = smem;                              // [block_size, head_dim]
    bf16* vs = smem + block_size * head_dim;      // [block_size, head_dim]

    for (int j = 0; j < ntiles; ++j) {
        // cooperatively stage one key block's K/V (this kv-head's head_dim slice) into shared memory
        size_t base = (size_t)btr[j] * block_size * kv_dim + (size_t)kvh * head_dim;
        for (int e = threadIdx.x; e < block_size * head_dim; e += blockDim.x) {
            int c = e / head_dim, d = e % head_dim;
            ks[e] = cache_k[base + (size_t)c * kv_dim + d];
            vs[e] = cache_v[base + (size_t)c * kv_dim + d];
        }
        __syncthreads();

        if (active) {
            int base_pos = j * block_size;
            int cmax = min(block_size, qpos - base_pos + 1);   // keys with key_pos <= qpos
            for (int c = 0; c < cmax; ++c) {
                const bf16* kc = ks + c * head_dim;
                float partial = 0.f;
                #pragma unroll
                for (int i = 0; i < DPL; ++i) partial += qreg[i] * __bfloat162float(kc[lane + (i << 5)]);
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, o);
                float score = partial * scale;            // identical on all lanes

                float m_new = fmaxf(m_run, score);
                float corr  = __expf(m_run - m_new);
                float p     = __expf(score - m_new);
                l_run = l_run * corr + p;
                const bf16* vc = vs + c * head_dim;
                #pragma unroll
                for (int i = 0; i < DPL; ++i) acc[i] = acc[i] * corr + p * __bfloat162float(vc[lane + (i << 5)]);
                m_run = m_new;
            }
        }
        __syncthreads();   // all warps done reading the tile before it is overwritten
    }

    if (active) {
        float inv = l_run > 0.f ? 1.0f / l_run : 0.f;
        bf16* o = out + ((size_t)flat * n_heads + h) * head_dim;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) o[lane + (i << 5)] = __float2bfloat16(acc[i] * inv);
    }
}

void launch_attention_flash(const bf16* q, int q_stride, const bf16* cache_k, const bf16* cache_v,
                            bf16* out, int n_heads, int n_kv, int head_dim,
                            const int* pos, const int* qstart, const int* qlen,
                            int R, int max_qlen,
                            const int* bt, int max_blocks, int block_size, float scale,
                            cudaStream_t s) {
    constexpr int BM = 16;                         // query rows (warps) per block
    int qtiles = (max_qlen + BM - 1) / BM;
    dim3 grid(qtiles, n_heads, R);
    size_t shmem = 2 * (size_t)block_size * head_dim * sizeof(bf16);
    // head_dim 128 -> DPL 4 (one warp covers 128 dims, 4 per lane). Qwen3 head_dim is always 128.
    attention_flash_kernel<BM, 4><<<grid, BM * 32, shmem, s>>>(
        q, q_stride, cache_k, cache_v, out, n_heads, n_kv, head_dim, pos, qstart, qlen, bt,
        max_blocks, block_size, scale);
}

// Gather S rows: out[i, :] = x[rows[i], :]   (one block per gathered row, BF16)
__global__ void gather_rows_kernel(const bf16* __restrict__ x, const int* __restrict__ rows,
                                   bf16* __restrict__ out, int H) {
    int i = blockIdx.x;
    const bf16* src = x + (size_t)rows[i] * H;
    bf16* dst = out + (size_t)i * H;
    for (int j = threadIdx.x; j < H; j += blockDim.x) dst[j] = src[j];
}

void launch_gather_rows(const bf16* x, const int* rows, bf16* out, int S, int H, cudaStream_t s) {
    if (S > 0) gather_rows_kernel<<<S, 256, 0, s>>>(x, rows, out, H);
}

// ---------------------------------------------------------------------------------------
// elementwise helpers (BF16 in/out, FP32 math)
// ---------------------------------------------------------------------------------------
__global__ void add_kernel(bf16* out, const bf16* in, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = __float2bfloat16(__bfloat162float(out[i]) + __bfloat162float(in[i]));
}
void launch_add(bf16* out, const bf16* in, int N, cudaStream_t s) {
    int block = 256;
    add_kernel<<<(N + block - 1) / block, block, 0, s>>>(out, in, N);
}

// h[m,i] = silu(gateup[m,i]) * gateup[m, I+i] — gate and up are the two halves of a fused row (width 2I).
__global__ void silu_mul_kernel(const bf16* __restrict__ gateup, bf16* __restrict__ h, int M, int I) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)M * I) return;
    int m = idx / I, i = idx % I;
    const bf16* row = gateup + (size_t)m * 2 * I;
    float g = __bfloat162float(row[i]);
    h[idx] = __float2bfloat16((g / (1.0f + expf(-g))) * __bfloat162float(row[I + i]));
}
void launch_silu_mul(const bf16* gateup, bf16* h, int M, int I, cudaStream_t s) {
    int block = 256;
    long N = (long)M * I;
    silu_mul_kernel<<<(N + block - 1) / block, block, 0, s>>>(gateup, h, M, I);
}

// Batched sampling: one block per row of logits[B, N]. Greedy (invT[b] <= 0) reduces an argmax;
// otherwise temperature softmax + inverse-CDF categorical sampling, restricted to the nucleus when
// top_p[b] < 1. Each thread owns a CONTIGUOUS index range so the cumulative scan is in token-id
// order (any consistent order gives the same distribution; contiguous makes the per-thread prefix a
// true prefix over token ids). Nucleus = the smallest set of highest-prob tokens with cumulative
// prob >= top_p; found by binary-searching a weight threshold wt with sum_{w_i >= wt} w_i >= top_p*Z
// (full distribution is the wt == 0 case). No global sort — all work stays inside one block.
static constexpr int kSampleThreads = 256;
__global__ void sample_batch_kernel(const float* __restrict__ logits, int N,
                                    const float* __restrict__ invT,
                                    const float* __restrict__ topp,
                                    const float* __restrict__ u,
                                    int* __restrict__ out) {
    int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const float* lg = logits + (size_t)b * N;
    float it = invT[b];

    __shared__ float sval[kSampleThreads];   // per-thread reduction / partial-sum scratch
    __shared__ int   sidx[kSampleThreads];

    // ---- greedy: argmax over a strided sweep ----
    if (it <= 0.0f) {
        float best = -1e30f; int bi = 0;
        for (int i = tid; i < N; i += nt) { float v = lg[i]; if (v > best) { best = v; bi = i; } }
        sval[tid] = best; sidx[tid] = bi;
        __syncthreads();
        for (int s = nt >> 1; s > 0; s >>= 1) {
            if (tid < s && sval[tid + s] > sval[tid]) { sval[tid] = sval[tid + s]; sidx[tid] = sidx[tid + s]; }
            __syncthreads();
        }
        if (tid == 0) out[b] = sidx[0];
        return;
    }

    // ---- temperature categorical (full distribution, or nucleus when top_p < 1) ----
    int chunk = (N + nt - 1) / nt;
    int lo = min(tid * chunk, N), hi = min(lo + chunk, N);

    // pass 1: max logit (numerical stability), block max-reduce
    float lmax = -1e30f;
    for (int i = lo; i < hi; ++i) lmax = fmaxf(lmax, lg[i]);
    sval[tid] = lmax;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) sval[tid] = fmaxf(sval[tid], sval[tid + s]);
        __syncthreads();
    }
    float m = sval[0];
    __syncthreads();

    // pass 2: per-thread exp-weight sum over its chunk; prefix -> base[], total Z. w_i in (0,1]
    // since the max logit gives w = exp(0) = 1.
    __shared__ float base[kSampleThreads];
    __shared__ float total;            // Z first, then the nucleus mass M
    float lsum = 0.0f;
    for (int i = lo; i < hi; ++i) lsum += expf((lg[i] - m) * it);
    sval[tid] = lsum;
    __syncthreads();
    if (tid == 0) { float acc = 0.0f; for (int t = 0; t < nt; ++t) { base[t] = acc; acc += sval[t]; } total = acc; }
    __syncthreads();
    float Z = total;

    // nucleus: binary-search the largest weight threshold wt whose kept mass still covers top_p*Z
    // (largest wt => smallest set), then rebuild base[]/M over the kept tokens. wt stays 0 (keep
    // all) when top_p >= 1.
    float wt = 0.0f;
    if (topp[b] < 1.0f) {
        float need = topp[b] * Z, wlo = 0.0f, whi = 1.0f;
        for (int it2 = 0; it2 < 32; ++it2) {
            float mid = 0.5f * (wlo + whi);
            float mmass = 0.0f;
            for (int i = lo; i < hi; ++i) { float w = expf((lg[i] - m) * it); if (w >= mid) mmass += w; }
            sval[tid] = mmass;
            __syncthreads();
            for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) sval[tid] += sval[tid + s]; __syncthreads(); }
            float mass = sval[0];
            __syncthreads();
            if (mass >= need) wlo = mid; else whi = mid;
        }
        wt = wlo;
        float km = 0.0f;
        for (int i = lo; i < hi; ++i) { float w = expf((lg[i] - m) * it); if (w >= wt) km += w; }
        sval[tid] = km;
        __syncthreads();
        if (tid == 0) { float acc = 0.0f; for (int t = 0; t < nt; ++t) { base[t] = acc; acc += sval[t]; } total = acc; }
        __syncthreads();
    }

    // inverse-CDF within the kept set (all tokens when wt == 0): the one thread whose
    // [base, base+sum) interval straddles the target re-scans its chunk in token-id order.
    __shared__ int result;
    if (tid == 0) result = N - 1;   // defensive fallback (target ~ total)
    __syncthreads();
    float target = u[b] * total;
    if (target >= base[tid] && target < base[tid] + sval[tid]) {
        float acc = base[tid];
        for (int i = lo; i < hi; ++i) {
            float w = expf((lg[i] - m) * it);
            if (w >= wt) { acc += w; if (acc >= target) { result = i; break; } }
        }
    }
    __syncthreads();
    if (tid == 0) out[b] = result;
}

void launch_sample_batch(const float* logits, int B, int N,
                         const float* invT, const float* topp, const float* u,
                         int* out, cudaStream_t s) {
    sample_batch_kernel<<<B, kSampleThreads, 0, s>>>(logits, N, invT, topp, u, out);
}
