// QSA block indexing, selection, and masked prefill/decode attention.

// ---- QSA indexer: which 4-token blocks a query past the budget sees ----
// Raw index keys (one per position) are cached; a block's key is the mean
// of its `ratio` raw keys, RMS-normed and roped at the block's first
// position. A query's `inh` index heads are normed and roped at its
// position; block score = sum over heads of relu(q_h . key) / sqrt(ihd).
// The top `k` blocks (ties: lowest index) plus the incomplete tail are
// visible. Rows with at most k complete blocks attend densely.
struct IndexParams {
    uint ihd;        // index head dim
    uint inh;        // index query heads
    uint qk_dim;     // index_qk projection width (inh*ihd + ihd)
    uint ratio;      // tokens per block
    uint rot;        // rotary dims
    float theta;
    float eps;
    uint base_pos;
    uint nb;
    uint b0;         // first block to (re)compute
    uint b1;         // one past the last
    uint k;          // blocks kept (budget / ratio)
    uint max_blocks; // score row stride
    uint vis_stride; // visible token list row stride (k*ratio + ratio)
    uint mask_words; // block bitmask row stride (u32 words)
};

// ikc[base_pos + b][d] = iqk[b][inh*ihd + d], kept as half precision.
kernel void fn_index_append(
    device const float* iqk [[buffer(0)]],
    device half*        ikc [[buffer(1)]],
    constant IndexParams& p [[buffer(2)]],
    uint gi [[thread_position_in_grid]])
{
    if (gi >= p.nb * p.ihd) return;
    const uint b = gi / p.ihd;
    const uint d = gi % p.ihd;
    ikc[(ulong)(p.base_pos + b) * p.ihd + d] =
        (half)iqk[(ulong)b * p.qk_dim + p.inh * p.ihd + d];
}

// RMSNorm of the `ihd` values held one per thread, times w, then partial
// rope (rotate_half pairing over the first `rot` dims) at `pos`; result
// written to dst (float or half), then a per-`rot` half-step rope. The block
// keys (half) and the roped queries (float) both use this helper.
template <typename Dst>
static inline void fn_index_norm_rope(
    float v, device const bfloat* w, device Dst* dst, constant IndexParams& p, uint pos,
    threadgroup float* red, threadgroup float* vals, uint d, uint sgid, uint lane)
{
    float ss = simd_sum(v * v);
    if (lane == 0) red[sgid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < (p.ihd + 31) / 32; i++) total += red[i];
    const float inv = rsqrt(total / (float)p.ihd + p.eps);
    const float val = v * inv * (float)w[d];
    vals[d] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint hr = p.rot / 2;
    if (d < hr) {
        const float inv_freq = pow(p.theta, -2.0f * (float)d / (float)p.rot);
        const float angle = (float)pos * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const float a = vals[d];
        const float bb = vals[d + hr];
        dst[d] = (Dst)(a * c - bb * s);
        dst[d + hr] = (Dst)(bb * c + a * s);
    } else if (d >= p.rot) {
        dst[d] = (Dst)val;
    }
}

// One threadgroup (ihd threads) per block in [b0, b1): its key from the
// cached raw keys.
kernel void fn_index_blocks(
    device const half*   ikc [[buffer(0)]],
    device half*         blk [[buffer(1)]],
    device const bfloat* w   [[buffer(2)]],
    constant IndexParams& p  [[buffer(3)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint d    [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[32];
    threadgroup float vals[256];
    const uint b = p.b0 + tg;
    if (b >= p.b1) return;
    float acc = 0.0f;
    for (uint t = 0; t < p.ratio; t++)
        acc += (float)ikc[(ulong)(b * p.ratio + t) * p.ihd + d];
    acc /= (float)p.ratio;
    fn_index_norm_rope(acc, w, blk + (ulong)b * p.ihd, p, b * p.ratio, red,
                       vals, d, sgid, lane);
}

// One threadgroup (ihd threads) per (row, index head): the roped query.
kernel void fn_index_q(
    device const float*  iqk [[buffer(0)]],
    device float*        iq  [[buffer(1)]],
    device const bfloat* w   [[buffer(2)]],
    constant IndexParams& p  [[buffer(3)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint d    [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[32];
    threadgroup float vals[256];
    const uint b = tg / p.inh;
    const uint h = tg % p.inh;
    const float v = iqk[(ulong)b * p.qk_dim + h * p.ihd + d];
    fn_index_norm_rope(v, w, iq + ((ulong)b * p.inh + h) * p.ihd, p, p.base_pos + b, red,
                       vals, d, sgid, lane);
}

// score[b][j] = sum_h relu(iq[b][h] . blk[j]) / sqrt(ihd) for the complete
// blocks of row b. Threadgroups of 256 blocks per row.
kernel void fn_index_score(
    device const float*   iq    [[buffer(0)]],
    device const half*    blk   [[buffer(1)]],
    device float*         score [[buffer(2)]],
    constant IndexParams& p     [[buffer(3)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float q[1024];
    const uint per = (p.max_blocks + 255) / 256;
    const uint b = tg / per;
    const uint j = (tg % per) * 256 + tid;
    const uint qn = p.inh * p.ihd;
    for (uint i = tid; i < qn; i += 256) q[i] = iq[(ulong)b * qn + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint blocks = (p.base_pos + b + 1) / p.ratio;
    if (j >= blocks) return;
    device const half* key = blk + (ulong)j * p.ihd;
    float acc = 0.0f;
    for (uint h = 0; h < p.inh; h++) {
        float dot = 0.0f;
        for (uint d = 0; d < p.ihd; d++) dot += q[h * p.ihd + d] * (float)key[d];
        acc += max(dot, 0.0f);
    }
    score[(ulong)b * p.max_blocks + j] = acc / sqrt((float)p.ihd);
}

// Top-k blocks of a row (ties: lowest index) plus the incomplete tail, as
// an ascending token list vis[b][..nvis[b]] and as a block bitmask
// vmask[b] (for the prefill softmax). One 1024-thread threadgroup per
// row; rows within the budget are left alone (dense attention). Radix
// select over the score bits (scores are >= 0, so their bit patterns
// order like the values), 4 bits per pass from the top.
kernel void fn_index_select(
    device const float* score [[buffer(0)]],
    device uint*        vis   [[buffer(1)]],
    device uint*        nvis  [[buffer(2)]],
    constant IndexParams& p   [[buffer(3)]],
    device uint*        vmask [[buffer(4)]],
    uint b    [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup atomic_uint hist[16];
    threadgroup uint sh_digit;
    threadgroup uint sh_above;
    threadgroup uint sh_sum[32];
    const uint t_len = p.base_pos + b + 1;
    const uint n = t_len / p.ratio;
    if (n <= p.k) return;
    device const float* sc = score + (ulong)b * p.max_blocks;
    device uint* mrow = vmask + (ulong)b * p.mask_words;
    for (uint w = tid; w < p.mask_words; w += 1024) mrow[w] = 0u;
    threadgroup_barrier(mem_flags::mem_device);
    uint prefix = 0;
    uint mask = 0;
    uint remaining = p.k;
    for (int shift = 28; shift >= 0; shift -= 4) {
        if (tid < 16) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += 1024) {
            const uint key = as_type<uint>(sc[i]);
            if ((key & mask) == prefix) {
                atomic_fetch_add_explicit(&hist[(key >> shift) & 15u], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint cum = 0;
            uint digit = 0;
            uint above = 0;
            for (int dg = 15; dg >= 0; dg--) {
                const uint c = atomic_load_explicit(&hist[dg], memory_order_relaxed);
                if (cum + c >= remaining) { digit = (uint)dg; above = cum; break; }
                cum += c;
            }
            sh_digit = digit;
            sh_above = above;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= sh_digit << shift;
        mask |= 0xFu << shift;
        remaining -= sh_above;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint T = prefix;
    // Contiguous index chunks per thread keep ascending order.
    const uint chunk = (n + 1023) / 1024;
    const uint i0 = min(tid * chunk, n);
    const uint i1 = min(i0 + chunk, n);
    uint local_gt = 0;
    uint local_eq = 0;
    for (uint i = i0; i < i1; i++) {
        const uint key = as_type<uint>(sc[i]);
        local_gt += key > T;
        local_eq += key == T;
    }
    // Exclusive prefix of local_eq across the threadgroup, and the totals.
    uint eq_pre = simd_prefix_exclusive_sum(local_eq);
    uint gt_sum = simd_sum(local_gt);
    if (lane == 31) sh_sum[sgid] = eq_pre + local_eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint eq_base = 0;
    for (uint s = 0; s < sgid; s++) eq_base += sh_sum[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) sh_sum[sgid] = gt_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count_gt = 0;
    for (uint s = 0; s < 32; s++) count_gt += sh_sum[s];
    const uint need = p.k - count_gt;   // keys == T taken lowest-index first
    uint eq_rank = eq_base + eq_pre;
    uint local_sel = 0;
    for (uint i = i0; i < i1; i++) {
        const uint key = as_type<uint>(sc[i]);
        if (key > T) local_sel++;
        else if (key == T) { if (eq_rank < need) local_sel++; eq_rank++; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint sel_pre = simd_prefix_exclusive_sum(local_sel);
    if (lane == 31) sh_sum[sgid] = sel_pre + local_sel;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint out = sel_pre;
    for (uint s = 0; s < sgid; s++) out += sh_sum[s];
    device uint* row = vis + (ulong)b * p.vis_stride;
    eq_rank = eq_base + eq_pre;
    for (uint i = i0; i < i1; i++) {
        const uint key = as_type<uint>(sc[i]);
        bool take = key > T;
        if (key == T) { take = eq_rank < need; eq_rank++; }
        if (take) {
            for (uint t = 0; t < p.ratio; t++) row[out * p.ratio + t] = i * p.ratio + t;
            atomic_fetch_or_explicit((device atomic_uint*)(mrow + (i >> 5)), 1u << (i & 31u), memory_order_relaxed);
            out++;
        }
    }
    if (tid == 0) {
        const uint tail0 = n * p.ratio;
        for (uint t = tail0; t < t_len; t++) row[p.k * p.ratio + (t - tail0)] = t;
        nvis[b] = p.k * p.ratio + (t_len - tail0);
    }
}

struct FnAttnPartParams {
    uint n_heads;
    uint n_kv;
    uint head_dim;
    uint t_len;      // visible tokens
    uint q_stride;
    uint q_off;
    uint max_blk;
    float scale;
};

// attn_part2_q8 (common/attention.metal) over a visible token list instead of the
// contiguous prefix: split-T flash-decode partials over the q8 KV cache.
kernel void fn_attn_part2_q8_sel(
    device const float* q     [[buffer(0)]],
    device const char*  kc    [[buffer(1)]],
    device const char*  vc    [[buffer(2)]],
    device float*       part  [[buffer(3)]],
    constant FnAttnPartParams& p [[buffer(4)]],
    constant uint&      n_wg  [[buffer(5)]],
    constant uint&      scale_off [[buffer(6)]],
    device const uint*  vis   [[buffer(7)]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint hd = p.head_dim;
    const uint kv_row = p.n_kv * hd;
    const uint blocks_per_row = kv_row / 32;
    const uint n_rep = p.n_heads / p.n_kv;
    const uint hk = tgpos % p.n_kv;
    const uint iwg = tgpos / p.n_kv;
    const uint head = hk * n_rep + sgid;
    if (sgid >= n_rep) return;

    device const half* ks = (device const half*)(kc + scale_off);
    device const half* vs = (device const half*)(vc + scale_off);
    const uint e0 = hk * hd + lane * 4;
    const uint e1 = hk * hd + (lane + 32) * 4;
    const uint b0 = e0 / 32;
    const uint b1 = e1 / 32;

    device const float* qh = q + p.q_off + (ulong)head * p.q_stride;
    const float4 qa = *(device const float4*)(qh + lane * 4);
    const float4 qb = *(device const float4*)(qh + (lane + 32) * 4);

    float m = -INFINITY;
    float s = 0.0f;
    float4 oa = 0.0f;
    float4 ob = 0.0f;

    const uint n_chunks = (p.t_len + 31) / 32;
    for (uint c = iwg; c < n_chunks; c += n_wg) {
        const uint c0 = c * 32;
        const uint cn = min(p.t_len - c0, 32u);
        for (uint i = 0; i < cn; i++) {
            const ulong t = vis[c0 + i];
            float ds0 = (float)ks[t * blocks_per_row + b0];
            float ds1 = (float)ks[t * blocks_per_row + b1];
            float4 ka = float4(*(device const char4*)(kc + t * kv_row + e0)) * ds0;
            float4 kb = float4(*(device const char4*)(kc + t * kv_row + e1)) * ds1;
            float partial = dot(qa, ka) + dot(qb, kb);
            float sc = simd_sum(partial) * p.scale;
            float mnew = max(m, sc);
            float factor = exp2(m - mnew);
            float e = exp2(sc - mnew);
            float dv0 = (float)vs[t * blocks_per_row + b0];
            float dv1 = (float)vs[t * blocks_per_row + b1];
            float4 va = float4(*(device const char4*)(vc + t * kv_row + e0)) * dv0;
            float4 vb = float4(*(device const char4*)(vc + t * kv_row + e1)) * dv1;
            oa = oa * factor + e * va;
            ob = ob * factor + e * vb;
            s = s * factor + e;
            m = mnew;
        }
    }

    device float* pb = part + ((ulong)head * p.max_blk + iwg) * (2 + hd);
    if (lane == 0) {
        pb[0] = m;
        pb[1] = s;
    }
    *(device float4*)(pb + 2 + lane * 4) = oa;
    *(device float4*)(pb + 2 + (lane + 32) * 4) = ob;
}

struct FnAttnGemmParams {
    uint kv_row;
    uint blocks_per_row;
    uint head;
    uint kl;
    uint kl_pad;
    uint scale_off;
    uint base;       // context length before this query sub-chunk
    uint nb;         // queries in the sub-chunk
    uint n_rep;
    float scale;
};

struct SelMaskParams {
    uint row0;       // first query's row in the block bitmask
    uint mask_words;
    uint ratio;
    uint k;
};

// Prefill row softmax with the QSA visibility mask: a query
// with more than k complete blocks sees only its selected blocks (bit
// set in vmask) and the incomplete tail.
kernel void fn_attn_softmax_sel(
    device const float* S [[buffer(0)]],   // [m][kl_pad]
    device half*        P [[buffer(1)]],   // [m][kl_pad]
    constant FnAttnGemmParams& p [[buffer(2)]],
    device const uint*  vmask [[buffer(3)]],
    constant SelMaskParams& sm [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];
    device const float* r = S + (ulong)row * p.kl_pad;
    device half* o = P + (ulong)row * p.kl_pad;
    const uint t = row % p.nb;
    const uint e = p.base + t + 1;  // causal extent
    const uint blocks = e / sm.ratio;
    const bool selective = blocks > sm.k;
    const uint tail0 = blocks * sm.ratio;
    device const uint* mrow = vmask + (ulong)(sm.row0 + t) * sm.mask_words;
    #define FN_VIS(i) (!selective || (i) >= tail0 || ((mrow[((i) / sm.ratio) >> 5] >> (((i) / sm.ratio) & 31u)) & 1u))
    float m = -INFINITY;
    for (uint i = tid; i < e; i += tpg) if (FN_VIS(i)) m = max(m, r[i]);
    m = simd_max(m);
    if (lane == 0) partial[sgid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float v = (lane < (tpg + 31) / 32) ? partial[lane] : -INFINITY;
        v = simd_max(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = partial[0];
    float s = 0.0f;
    for (uint i = tid; i < e; i += tpg) if (FN_VIS(i)) s += exp2(r[i] - m);
    s = simd_sum(s);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partial[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float v = (lane < (tpg + 31) / 32) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / partial[0];
    for (uint i = tid; i < e; i += tpg) o[i] = FN_VIS(i) ? (half)(exp2(r[i] - m) * inv) : 0.0h;
    for (uint i = e + tid; i < p.kl_pad; i += tpg) o[i] = 0.0h;
    #undef FN_VIS
}
