#include "kernels.cuh"
#include "kv_cache.cuh"   // kv_phys_row: the paged-KV addressing contract (shared with block_pool.cu)
#include <mma.h>
using namespace nvcuda;

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

// ======================================================================================
// Paged-KV attention. Split by request type (dispatched from run_layers): prefill rows
// (q_len>1) go to attn_prefill (WMMA tensor cores), decode rows (q_len==1) to attn_decode
// (FlashDecoding). Both read the paged KV pool [num_blocks, BLOCK, kv_dim] via per-request
// block tables `bt` (stride max_blocks); q comes from the fused-QKV buffer at row stride
// q_stride. `rids[grid_slot]` -> actual request id, so each runs on its request subset.
// ======================================================================================

// --- decode: one block per (head, decode-request); NW warps split the request's KV [0,qpos] into
// strided slices (warp w takes key positions w, w+NW, ...), each online-softmaxing its slice in
// registers — K/V read straight from the cache (a single query row has no cross-row reuse, so
// shared staging would only add traffic + syncs) — then an in-block combine merges the NW partials.
template<int NW, int DPL>
__global__ void attn_decode_kernel(const bf16* __restrict__ q, int q_stride,
                                   const bf16* __restrict__ cache_k, const bf16* __restrict__ cache_v,
                                   bf16* __restrict__ out, int n_heads, int n_kv, int head_dim,
                                   const int* __restrict__ pos, const int* __restrict__ qstart,
                                   const int* __restrict__ decode_rids, const int* __restrict__ bt,
                                   int max_blocks, int block_size, float scale) {
    int h    = blockIdx.x;             // query head
    int di   = blockIdx.y;             // decode-request slot
    int w    = threadIdx.x >> 5;       // 0..NW-1: this warp's KV split
    int lane = threadIdx.x & 31;

    int r    = decode_rids[di];
    int flat = qstart[r];              // the single query row (q_len == 1)
    int qpos = pos[flat];              // attends keys [0, qpos]
    int kvh  = h / (n_heads / n_kv);
    int kv_dim = n_kv * head_dim;
    const int* btr = bt + (size_t)r * max_blocks;

    float qreg[DPL];
    const bf16* qv = q + (size_t)flat * q_stride + (size_t)h * head_dim;
    #pragma unroll
    for (int i = 0; i < DPL; ++i) qreg[i] = __bfloat162float(qv[lane + (i << 5)]);

    float m_run = -1e30f, l_run = 0.f, acc[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) acc[i] = 0.f;

    // warp w streams key positions w, w+NW, w+2NW, ... <= qpos, reading K/V directly from the pool.
    for (int kpos = w; kpos <= qpos; kpos += NW) {
        size_t base = (size_t)btr[kpos / block_size] * block_size * kv_dim
                    + (size_t)(kpos % block_size) * kv_dim + (size_t)kvh * head_dim;
        const bf16* kc = cache_k + base;
        float partial = 0.f;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) partial += qreg[i] * __bfloat162float(kc[lane + (i << 5)]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, o);
        float score = partial * scale;

        float m_new = fmaxf(m_run, score);
        float corr  = __expf(m_run - m_new);
        float p     = __expf(score - m_new);
        l_run = l_run * corr + p;
        const bf16* vc = cache_v + base;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) acc[i] = acc[i] * corr + p * __bfloat162float(vc[lane + (i << 5)]);
        m_run = m_new;
    }

    // in-block combine of the NW partials (online-softmax merge), done by warp 0.
    __shared__ float sm[NW], sl[NW], sacc[NW][DPL * 32];
    sm[w] = m_run; sl[w] = l_run;
    #pragma unroll
    for (int i = 0; i < DPL; ++i) sacc[w][lane + (i << 5)] = acc[i];
    __syncthreads();

    if (w == 0) {
        float gm = -1e30f;
        #pragma unroll
        for (int s = 0; s < NW; ++s) gm = fmaxf(gm, sm[s]);
        float gl = 0.f, gacc[DPL];
        #pragma unroll
        for (int i = 0; i < DPL; ++i) gacc[i] = 0.f;
        for (int s = 0; s < NW; ++s) {
            float sc = __expf(sm[s] - gm);
            gl += sl[s] * sc;
            #pragma unroll
            for (int i = 0; i < DPL; ++i) gacc[i] += sacc[s][lane + (i << 5)] * sc;
        }
        float inv = gl > 0.f ? 1.0f / gl : 0.f;
        bf16* o = out + ((size_t)flat * n_heads + h) * head_dim;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) o[lane + (i << 5)] = __float2bfloat16(gacc[i] * inv);
    }
}

void launch_attn_decode(const bf16* q, int q_stride, const bf16* cache_k, const bf16* cache_v,
                        bf16* out, int n_heads, int n_kv, int head_dim,
                        const int* pos, const int* qstart, const int* decode_rids, int n_decode,
                        const int* bt, int max_blocks, int block_size, float scale, cudaStream_t s) {
    if (n_decode <= 0) return;
    constexpr int NW = 8;              // warps per block = KV splits
    dim3 grid(n_heads, n_decode);
    attn_decode_kernel<NW, 4><<<grid, NW * 32, 0, s>>>(
        q, q_stride, cache_k, cache_v, out, n_heads, n_kv, head_dim, pos, qstart, decode_rids, bt,
        max_blocks, block_size, scale);
}

// --- prefill: tensor-core (WMMA 16x16x16 bf16 in / fp32 accumulate) FlashAttention-2. One warp per
// (16-query-tile, head, request); streams K/V in 16-key tiles (one tile == one paged block, since
// block_size==16), no S materialization, online softmax with deferred normalization, O kept in shared
// fp32 and rescaled per tile (portable — no WMMA fragment-layout assumptions). HD==128 (8 d-steps).
__global__ void attn_prefill_kernel(const bf16* __restrict__ q, int q_stride,
                                    const bf16* __restrict__ cache_k,
                                    const bf16* __restrict__ cache_v, bf16* __restrict__ out,
                                    int n_heads, int n_kv,
                                    const int* __restrict__ pos, const int* __restrict__ qstart,
                                    const int* __restrict__ qlen, const int* __restrict__ rids,
                                    const int* __restrict__ bt, int max_blocks, float scale) {
    constexpr int HD = 128, ND = HD / 16, TILE = 16;
    int r  = rids[blockIdx.z];
    int h  = blockIdx.y;
    int qt = blockIdx.x;
    int lane = threadIdx.x & 31;

    int ql = qlen[r];
    int q0 = qt * TILE;                       // first query row (within request) of this tile
    if (q0 >= ql) return;                     // tile entirely past the request's rows
    int qs  = qstart[r];
    int kvh = h / (n_heads / n_kv);
    int kv_dim = n_kv * HD;
    const int* btr = bt + (size_t)r * max_blocks;

    // last active row's position bounds the key range (positions increase within a request).
    int last = min(q0 + TILE - 1, ql - 1);
    int maxqpos = pos[qs + last];
    int ntiles = maxqpos / TILE + 1;

    __shared__ float Ss[TILE * TILE];         // S tile scratch (fp32)
    __shared__ __nv_bfloat16 Ps[TILE * TILE]; // softmaxed P tile (bf16) for P*V
    __shared__ float Os[TILE * HD];           // O accumulator (fp32, persists across K tiles)
    __shared__ float Ot[TILE * TILE];         // one 16x16 P*V d-tile (folded into Os immediately)
    __shared__ float m_[TILE], l_[TILE], corr_[TILE];

    for (int e = lane; e < TILE * HD; e += 32) Os[e] = 0.f;
    if (lane < TILE) { m_[lane] = -1e30f; l_[lane] = 0.f; }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;

    for (int kt = 0; kt < ntiles; ++kt) {
        size_t kvbase = (size_t)btr[kt] * TILE * kv_dim + (size_t)kvh * HD;   // page kt, this kv-head

        // S = Q * K^T  (accumulate over the 8 head-dim steps)
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sacc;
        wmma::fill_fragment(sacc, 0.f);
        #pragma unroll
        for (int d = 0; d < ND; ++d) {
            wmma::load_matrix_sync(qf, q + (size_t)(qs + q0) * q_stride + (size_t)h * HD + d * 16, q_stride);
            wmma::load_matrix_sync(kf, cache_k + kvbase + d * 16, kv_dim);
            wmma::mma_sync(sacc, qf, kf, sacc);
        }
        wmma::store_matrix_sync(Ss, sacc, TILE, wmma::mem_row_major);
        __syncthreads();

        // online softmax over this tile's 16 keys (one warp lane per query row)
        if (lane < TILE) {
            int grow = q0 + lane;
            if (grow < ql) {
                int qpos = pos[qs + grow];
                float sc[TILE], rmax = -1e30f;
                #pragma unroll
                for (int c = 0; c < TILE; ++c) {
                    int kpos = kt * TILE + c;
                    sc[c] = (kpos <= qpos) ? Ss[lane * TILE + c] * scale : -1e30f;
                    rmax = fmaxf(rmax, sc[c]);
                }
                float mold = m_[lane], mnew = fmaxf(mold, rmax), corr = __expf(mold - mnew), rsum = 0.f;
                #pragma unroll
                for (int c = 0; c < TILE; ++c) { float p = __expf(sc[c] - mnew); Ps[lane * TILE + c] = __float2bfloat16(p); rsum += p; }
                l_[lane] = l_[lane] * corr + rsum; m_[lane] = mnew; corr_[lane] = corr;
            } else {
                #pragma unroll
                for (int c = 0; c < TILE; ++c) Ps[lane * TILE + c] = __float2bfloat16(0.f);
                corr_[lane] = 1.f;
            }
        }
        __syncthreads();

        // P*V folded with the running-O rescale, one 16x16 d-tile at a time (no full [16,128] temp):
        // Os[:,d] = Os[:,d]*corr + (P*V)[:,d]. Each Os element is rescaled exactly once (when its own
        // d-tile is processed). Shrinks shared (~18KB -> ~11KB) so more blocks/warps stay resident.
        #pragma unroll
        for (int d = 0; d < ND; ++d) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv;
            wmma::fill_fragment(pv, 0.f);
            wmma::load_matrix_sync(pf, Ps, TILE);
            wmma::load_matrix_sync(vf, cache_v + kvbase + d * 16, kv_dim);
            wmma::mma_sync(pv, pf, vf, pv);
            wmma::store_matrix_sync(Ot, pv, TILE, wmma::mem_row_major);
            __syncthreads();
            for (int e = lane; e < TILE * TILE; e += 32) {
                int row = e / TILE, col = e % TILE;
                int oidx = row * HD + d * 16 + col;
                Os[oidx] = Os[oidx] * corr_[row] + Ot[e];
            }
            __syncthreads();
        }
    }

    // normalize and write out the active rows
    for (int e = lane; e < TILE * HD; e += 32) {
        int row = e / HD, d = e % HD, grow = q0 + row;
        if (grow < ql) {
            float inv = l_[row] > 0.f ? 1.0f / l_[row] : 0.f;
            out[((size_t)(qs + grow) * n_heads + h) * HD + d] = __float2bfloat16(Os[e] * inv);
        }
    }
}

void launch_attn_prefill(const bf16* q, int q_stride, const bf16* cache_k, const bf16* cache_v,
                         bf16* out, int n_heads, int n_kv, int head_dim,
                         const int* pos, const int* qstart, const int* qlen,
                         const int* rids, int R, int max_qlen,
                         const int* bt, int max_blocks, int block_size, float scale,
                         cudaStream_t s) {
    if (R <= 0) return;
    // The WMMA kernel hardwires head_dim==128 (8 d-steps) and one 16-key tile per paged block
    // (block_size==16) — the engine's only supported layout (Qwen3, BlockPool::BLOCK==16).
    int qtiles = (max_qlen + 15) / 16;
    dim3 grid(qtiles, n_heads, R);
    attn_prefill_kernel<<<grid, 32, 0, s>>>(
        q, q_stride, cache_k, cache_v, out, n_heads, n_kv, pos, qstart, qlen, rids, bt, max_blocks, scale);
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
