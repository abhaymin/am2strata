// am2Strata's native Metal kernels (macOS): the same products and steps as kernels.rs, written in MSL so they can use
// SIMD-group reductions and run a whole token inside one command buffer. Weights stay in their GGUF encoding and are
// dequantized as they are read, exactly as strata_pack::quants does on the CPU; only the order of the sums differs.
//
// The matrix-vector layout is the usual Apple-GPU shape: a threadgroup of NSG SIMD groups, each SIMD group owning NR consecutive rows; the 32 lanes split a row's
// 32-value groups between them, every lane loads its x values once and applies them to all NR rows, and each row's
// partial sums are added with simd_sum.

#include <metal_stdlib>
using namespace metal;

#define NSG 2   // SIMD groups per threadgroup in the products
#define NR 4    // rows per SIMD group

struct MvArgs {
    uint rows;
    uint cols;
    uint row_bytes;
    uint kind;      // the weight encoding: kernels::kind
};

// kernels::kind
#define K_F32 0
#define K_F16 1
#define K_BF16 2
#define K_Q8_0 3
#define K_Q4_0 4
#define K_Q4_K 5
#define K_Q5_K 6
#define K_Q6_K 7

static inline float f16_at(device const uchar* p) {
    ushort u = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(as_type<half>(u));
}

static inline float bf16_at(device const uchar* p) {
    uint u = (uint(p[0]) | (uint(p[1]) << 8)) << 16;
    return as_type<float>(u);
}

static inline float f32_at(device const uchar* p) {
    uint u = uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
    return as_type<float>(u);
}

// the 6-bit scale and min of 32-value group g of a Q4_K / Q5_K block whose 12 scale bytes start at s
static inline void scale_min_k4(device const uchar* s, uint g, thread float& sc, thread float& mn) {
    uint j = g % 4;
    uint a, b;
    if (g < 4) {
        a = s[j] & 0x3f;
        b = s[4 + j] & 0x3f;
    } else {
        a = (s[8 + j] & 0x0f) | ((s[j] >> 2) & 0x30);
        b = (s[8 + j] >> 4) | ((s[4 + j] >> 2) & 0x30);
    }
    sc = float(a);
    mn = float(b);
}

// The dot product of 32-value group g of the block at o with xv (the 32 x values, sum xs).
static inline float group_dot(device const uchar* o, uint g, thread const float* xv, float xs, uint kind) {
    float acc = 0.0f;
    if (kind == K_Q4_K || kind == K_Q5_K) {
        float d = f16_at(o), dmin = f16_at(o + 2);
        float sc, mn;
        scale_min_k4(o + 4, g, sc, mn);
        device const uchar* qs = o + (kind == K_Q5_K ? 48 : 16) + (g / 2) * 32;
        uint sh = (g % 2) * 4;
        float part = 0.0f;
        if (kind == K_Q4_K) {
            for (uint l = 0; l < 32; l++) {
                part += float((qs[l] >> sh) & 0x0f) * xv[l];
            }
        } else {
            device const uchar* qh = o + 16;
            for (uint l = 0; l < 32; l++) {
                uint q = ((qs[l] >> sh) & 0x0f) | (((qh[l] >> g) & 1) << 4);
                part += float(q) * xv[l];
            }
        }
        acc = d * sc * part - dmin * mn * xs;
    } else if (kind == K_Q6_K) {
        float d = f16_at(o + 208);
        uint hh = g / 4, kk = g % 4;
        device const uchar* lo = o + hh * 64 + (kk % 2) * 32;
        uint lo_sh = 4 * (kk / 2);
        device const uchar* hi = o + 128 + hh * 32;
        float s0 = d * float(as_type<char>(o[192 + g * 2]));
        float s1 = d * float(as_type<char>(o[192 + g * 2 + 1]));
        float p0 = 0.0f, p1 = 0.0f;
        for (uint l = 0; l < 16; l++) {
            int q = int(((lo[l] >> lo_sh) & 0x0f) | (((hi[l] >> (2 * kk)) & 3) << 4)) - 32;
            p0 += float(q) * xv[l];
        }
        for (uint l = 16; l < 32; l++) {
            int q = int(((lo[l] >> lo_sh) & 0x0f) | (((hi[l] >> (2 * kk)) & 3) << 4)) - 32;
            p1 += float(q) * xv[l];
        }
        acc = s0 * p0 + s1 * p1;
    } else if (kind == K_Q8_0) {
        float d = f16_at(o);
        for (uint l = 0; l < 32; l++) {
            acc += float(as_type<char>(o[2 + l])) * xv[l];
        }
        acc *= d;
    } else if (kind == K_Q4_0) {
        float d = f16_at(o);
        for (uint l = 0; l < 16; l++) {
            uint b = o[2 + l];
            acc += (float(b & 0x0f) - 8.0f) * xv[l] + (float(b >> 4) - 8.0f) * xv[l + 16];
        }
        acc *= d;
    } else if (kind == K_F16) {
        for (uint l = 0; l < 32; l++) acc += f16_at(o + 2 * l) * xv[l];
    } else if (kind == K_BF16) {
        for (uint l = 0; l < 32; l++) acc += bf16_at(o + 2 * l) * xv[l];
    } else { // K_F32
        for (uint l = 0; l < 32; l++) acc += f32_at(o + 4 * l) * xv[l];
    }
    return acc;
}

// y[t * rows + r] = W[r] . x[t * cols ..]: threadgroup (i, t) takes rows i * NSG * NR .. for token t.
// Lane `lane` takes the 32-value groups lane, lane + 32, ... of each row: for 256-value blocks that is block
// lane / 8 + 4k, group lane % 8; for 32-value blocks, block lane + 32k.
kernel void mv(device const uchar* w [[buffer(0)]],
               device const float* x [[buffer(1)]],
               device float* y [[buffer(2)]],
               constant MvArgs& a [[buffer(3)]],
               uint2 tg [[threadgroup_position_in_grid]],
               uint sg [[simdgroup_index_in_threadgroup]],
               uint lane [[thread_index_in_simdgroup]]) {
    const uint r0 = (tg.x * NSG + sg) * NR;
    const uint t = tg.y;
    const bool big = a.kind >= K_Q4_K;
    const uint vpb = big ? 256 : 32;                // values per block
    const uint bpb = a.row_bytes / (a.cols / vpb);  // bytes per block
    const uint ng = a.cols / 32;
    device const float* xt = x + t * a.cols;
    float acc[NR];
    for (uint r = 0; r < NR; r++) acc[r] = 0.0f;
    for (uint gi = lane; gi < ng; gi += 32) {
        const uint b = big ? gi / 8 : gi;
        const uint g = big ? gi % 8 : 0;
        float xv[32];
        float xs = 0.0f;
        device const float4* xp = (device const float4*)(xt + gi * 32);
        for (uint i = 0; i < 8; i++) {
            float4 v = xp[i];
            xv[4 * i] = v.x; xv[4 * i + 1] = v.y; xv[4 * i + 2] = v.z; xv[4 * i + 3] = v.w;
            xs += v.x + v.y + v.z + v.w;
        }
        for (uint r = 0; r < NR; r++) {
            const uint row = r0 + r;
            if (row < a.rows) {
                acc[r] += group_dot(w + row * a.row_bytes + b * bpb, g, xv, xs, a.kind);
            }
        }
    }
    for (uint r = 0; r < NR; r++) {
        const float s = simd_sum(acc[r]);
        const uint row = r0 + r;
        if (lane == 0 && row < a.rows) {
            y[t * a.rows + row] = s;
        }
    }
}

// ------------------------------------------------------------------ the steps between the products

struct NormArgs { uint n; float eps; };

// y = x * w / sqrt(mean(x^2) + eps), one vector per threadgroup (threadgroup t takes vector t): 1024 threads.
kernel void rms_norm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device float* y [[buffer(2)]],
                     constant NormArgs& a [[buffer(3)]],
                     uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                     uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    device const float* xt = x + t * a.n;
    float acc = 0.0f;
    for (uint j = tid; j < a.n; j += ntg) acc += xt[j] * xt[j];
    acc = simd_sum(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < (ntg + 31) / 32 ? part[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) part[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float k = 1.0f / sqrt(part[0] / float(a.n) + a.eps);
    for (uint j = tid; j < a.n; j += ntg) y[t * a.n + j] = xt[j] * k * w[j];
}

struct NArgs { uint n; uint at; };

kernel void add_into(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant NArgs& a [[buffer(2)]],
                     uint j [[thread_position_in_grid]]) {
    if (j < a.n) y[j] = y[j] + x[j];
}

kernel void silu_gate(device const float* gate [[buffer(0)]], device float* up [[buffer(1)]], constant NArgs& a [[buffer(2)]],
                      uint j [[thread_position_in_grid]]) {
    if (j < a.n) {
        const float g = gate[j];
        up[j] = up[j] * (g / (1.0f + exp(-g)));
    }
}

kernel void kv_append(device const float* x [[buffer(0)]], device float* cache [[buffer(1)]], constant NArgs& a [[buffer(2)]],
                      uint j [[thread_position_in_grid]]) {
    if (j < a.n) cache[a.at + j] = x[j];
}

struct RopeArgs { uint hd; uint n_rot; uint neox; };

// Rotates each head of x by the (cos, sin) pairs of tab: threadgroup = head, thread = pair.
kernel void rope(device const float* tab [[buffer(0)]], device float* x [[buffer(1)]], constant RopeArgs& a [[buffer(2)]],
                 uint h [[threadgroup_position_in_grid]], uint p [[thread_position_in_threadgroup]]) {
    if (p < a.n_rot / 2) {
        const float c = tab[p * 2], s = tab[p * 2 + 1];
        uint i0 = h * a.hd + p * 2, i1 = i0 + 1;
        if (a.neox == 1) {
            i0 = h * a.hd + p;
            i1 = i0 + a.n_rot / 2;
        }
        const float x0 = x[i0], x1 = x[i1];
        x[i0] = x0 * c - x1 * s;
        x[i1] = x0 * s + x1 * c;
    }
}

struct AttnArgs { uint n; uint n_kv; uint group; uint hd; float scale; };

// One query head's attention over n positions, decoding (one token): threadgroup = query head, 256 threads.
// Scores go to s[hh * n ..] (device scratch, so the context is not limited by threadgroup memory), then the softmax, then
// o = sum_t p_t v_t with the threads split between dimensions and position stripes.
#define ATG 256
kernel void attend(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
                   device float* s [[buffer(3)]], device float* o [[buffer(4)]], constant AttnArgs& a [[buffer(5)]],
                   uint hh [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[ATG / 32];
    threadgroup float accs[ATG];
    const uint kh = hh / a.group;
    device const float* qh = q + hh * a.hd;
    device float* sh = s + hh * a.n;
    // scores: SIMD group sg takes positions sg, sg + 8, ...; its lanes split the head's values
    for (uint t = sg; t < a.n; t += ATG / 32) {
        device const float* kt = k + (t * a.n_kv + kh) * a.hd;
        float acc = 0.0f;
        for (uint j = lane; j < a.hd; j += 32) acc += qh[j] * kt[j];
        acc = simd_sum(acc);
        if (lane == 0) sh[t] = acc * a.scale;
    }
    threadgroup_barrier(mem_flags::mem_device);
    // the largest score
    float top = -1.0e30f;
    for (uint t = tid; t < a.n; t += ATG) top = max(top, sh[t]);
    top = simd_max(top);
    if (lane == 0) red[sg] = top;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    top = red[0];
    for (uint i = 1; i < ATG / 32; i++) top = max(top, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // exp and their sum
    float total = 0.0f;
    for (uint t = tid; t < a.n; t += ATG) {
        const float e = exp(sh[t] - top);
        sh[t] = e;
        total += e;
    }
    total = simd_sum(total);
    if (lane == 0) red[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    total = 0.0f;
    for (uint i = 0; i < ATG / 32; i++) total += red[i];
    const float inv = 1.0f / total;
    // o[j] = sum_t p_t v[t][j]: thread tid takes dimension tid % hd and the positions stripe tid / hd
    const uint stripes = a.hd <= ATG ? ATG / a.hd : 1;
    for (uint j0 = 0; j0 < a.hd; j0 += ATG / stripes) {
        const uint j = j0 + tid % (ATG / stripes);
        const uint stripe = tid / (ATG / stripes);
        float acc = 0.0f;
        if (j < a.hd && stripe < stripes) {
            for (uint t = stripe; t < a.n; t += stripes) acc += sh[t] * v[(t * a.n_kv + kh) * a.hd + j];
        }
        accs[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (stripe == 0 && j < a.hd) {
            float sum = 0.0f;
            for (uint i = 0; i < stripes; i++) sum += accs[i * (ATG / stripes) + tid];
            o[hh * a.hd + j] = sum * inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ------------------------------------------------------------------ a prompt chunk: several tokens per dispatch

// The 32 values of group g of the block at o, dequantized (the same values group_dot multiplies).
static inline void group_values(device const uchar* o, uint g, uint kind, thread float* wv) {
    if (kind == K_Q4_K || kind == K_Q5_K) {
        float d = f16_at(o), dmin = f16_at(o + 2);
        float sc, mn;
        scale_min_k4(o + 4, g, sc, mn);
        const float a = d * sc, b = dmin * mn;
        device const uchar* qs = o + (kind == K_Q5_K ? 48 : 16) + (g / 2) * 32;
        const uint sh = (g % 2) * 4;
        if (kind == K_Q4_K) {
            for (uint l = 0; l < 32; l++) wv[l] = a * float((qs[l] >> sh) & 0x0f) - b;
        } else {
            device const uchar* qh = o + 16;
            for (uint l = 0; l < 32; l++) wv[l] = a * float(((qs[l] >> sh) & 0x0f) | (((qh[l] >> g) & 1) << 4)) - b;
        }
    } else if (kind == K_Q6_K) {
        const float d = f16_at(o + 208);
        const uint hh = g / 4, kk = g % 4;
        device const uchar* lo = o + hh * 64 + (kk % 2) * 32;
        const uint lo_sh = 4 * (kk / 2);
        device const uchar* hi = o + 128 + hh * 32;
        const float s0 = d * float(as_type<char>(o[192 + g * 2]));
        const float s1 = d * float(as_type<char>(o[192 + g * 2 + 1]));
        for (uint l = 0; l < 32; l++) {
            const int q = int(((lo[l] >> lo_sh) & 0x0f) | (((hi[l] >> (2 * kk)) & 3) << 4)) - 32;
            wv[l] = (l < 16 ? s0 : s1) * float(q);
        }
    } else if (kind == K_Q8_0) {
        const float d = f16_at(o);
        for (uint l = 0; l < 32; l++) wv[l] = d * float(as_type<char>(o[2 + l]));
    } else if (kind == K_Q4_0) {
        const float d = f16_at(o);
        for (uint l = 0; l < 16; l++) {
            const uint b = o[2 + l];
            wv[l] = d * (float(b & 0x0f) - 8.0f);
            wv[l + 16] = d * (float(b >> 4) - 8.0f);
        }
    } else if (kind == K_F16) {
        for (uint l = 0; l < 32; l++) wv[l] = f16_at(o + 2 * l);
    } else if (kind == K_BF16) {
        for (uint l = 0; l < 32; l++) wv[l] = bf16_at(o + 2 * l);
    } else {
        for (uint l = 0; l < 32; l++) wv[l] = f32_at(o + 4 * l);
    }
}

#define MM_BM 64   // rows per threadgroup
#define MM_BN 32   // tokens per threadgroup
#define MM_THREADS 128

struct MmArgs { uint rows; uint cols; uint row_bytes; uint kind; uint tokens; };

// Y[t][r] = W[r] . X[t] for a chunk of tokens, on the SIMD-group matrix units: threadgroup (i, j) computes rows
// i * 64 .. and tokens j * 32 ..; per 32-column step, 64 threads dequantize one row's 32-value group each into
// threadgroup memory while the other 64 load the tokens' 32 values, then each of the 4 SIMD groups multiplies its
// 32 x 16 corner in 8 x 8 tiles (f32 throughout, so the sums are the ones `mv` makes up to their order).
kernel void mm(device const uchar* w [[buffer(0)]],
               device const float* x [[buffer(1)]],
               device float* y [[buffer(2)]],
               constant MmArgs& a [[buffer(3)]],
               uint2 tg [[threadgroup_position_in_grid]],
               uint2 tid2 [[thread_position_in_threadgroup]],
               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[MM_BM * 32];   // weights [row][k]
    threadgroup float sb[MM_BN * 32];   // tokens [token][k]
    threadgroup float sc[MM_BM * MM_BN];
    const uint tid = tid2.x;
    const uint r0 = tg.x * MM_BM, t0 = tg.y * MM_BN;
    const bool big = a.kind >= K_Q4_K;
    const uint vpb = big ? 256 : 32;
    const uint bpb = a.row_bytes / (a.cols / vpb);
    const uint ng = a.cols / 32;
    const uint mr = (sg / 2) * 32, mt = (sg % 2) * 16;   // this SIMD group's corner
    simdgroup_float8x8 c[4][2];
    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 2; j++) c[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint gi = 0; gi < ng; gi++) {
        if (tid < MM_BM) {
            const uint row = r0 + tid;
            float wv[32];
            if (row < a.rows) {
                group_values(w + row * a.row_bytes + (big ? gi / 8 : gi) * bpb, big ? gi % 8 : 0, a.kind, wv);
            } else {
                for (uint l = 0; l < 32; l++) wv[l] = 0.0f;
            }
            for (uint l = 0; l < 32; l++) sa[tid * 32 + l] = wv[l];
        } else {
            // 64 threads, 32 tokens x 32 values: 16 each
            const uint i = tid - MM_BM;
            const uint tt = i / 2, half_ = (i % 2) * 16;
            const uint t = t0 + tt;
            for (uint l = 0; l < 16; l++) {
                sb[tt * 32 + half_ + l] = t < a.tokens ? x[t * a.cols + gi * 32 + half_ + l] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 am[4], bm[2];
            for (uint i = 0; i < 4; i++) simdgroup_load(am[i], sa + (mr + i * 8) * 32 + kk, 32);
            for (uint j = 0; j < 2; j++) simdgroup_load(bm[j], sb + (mt + j * 8) * 32 + kk, 32, ulong2(0, 0), true);
            for (uint i = 0; i < 4; i++) for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(c[i][j], am[i], bm[j], c[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 2; j++) simdgroup_store(c[i][j], sc + (mr + i * 8) * MM_BN + mt + j * 8, MM_BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < MM_BM * MM_BN; e += MM_THREADS) {
        const uint rr = e / MM_BN, tt = e % MM_BN;
        if (r0 + rr < a.rows && t0 + tt < a.tokens) y[(t0 + tt) * a.rows + r0 + rr] = sc[e];
    }
}

// rope for a chunk: threadgroup (head, token); tab holds each token's n_rot / 2 (cos, sin) pairs one after another,
// x holds each token's heads one after another.
struct RopeTArgs { uint hd; uint n_rot; uint neox; uint heads; };
kernel void rope_tokens(device const float* tab [[buffer(0)]], device float* x [[buffer(1)]], constant RopeTArgs& a [[buffer(2)]],
                        uint2 g [[threadgroup_position_in_grid]], uint2 p2 [[thread_position_in_threadgroup]]) {
    const uint p = p2.x;
    if (p < a.n_rot / 2) {
        device const float* tt = tab + g.y * a.n_rot;
        const float c = tt[p * 2], s = tt[p * 2 + 1];
        const uint at = (g.y * a.heads + g.x) * a.hd;
        uint i0 = at + p * 2, i1 = i0 + 1;
        if (a.neox == 1) {
            i0 = at + p;
            i1 = i0 + a.n_rot / 2;
        }
        const float x0 = x[i0], x1 = x[i1];
        x[i0] = x0 * c - x1 * s;
        x[i1] = x0 * s + x1 * c;
    }
}

// y[j] = x[at + j]: one row out of a chunk (the last token's, for the logits).
kernel void copy_from(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant NArgs& a [[buffer(2)]],
                      uint j [[thread_position_in_grid]]) {
    if (j < a.n) y[j] = x[a.at + j];
}

// Causal attention for a chunk of query tokens (positions base .. base + tokens), one threadgroup per (query head,
// token): keys 0 ..= base + t in blocks of ATG, an online softmax (running max and sum), V blended as it goes.
// scales[t] is token t's scale (temperature included).
struct AttnTArgs { uint base; uint n_head; uint n_kv; uint group; uint hd; };
kernel void attend_tokens(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
                          device const float* scales [[buffer(3)]], device float* o [[buffer(4)]], constant AttnTArgs& a [[buffer(5)]],
                          uint2 g [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint tid = tid2.x;
    threadgroup float qs[256];
    threadgroup float p[ATG];
    threadgroup float red[ATG / 32];
    threadgroup float accs[ATG];
    const uint hh = g.x, t = g.y;
    const uint kh = hh / a.group;
    const uint n = a.base + t + 1;
    const float scale = scales[t];
    device const float* qh = q + (t * a.n_head + hh) * a.hd;
    for (uint j = tid; j < a.hd; j += ATG) qs[j] = qh[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint stripes = a.hd <= ATG ? ATG / a.hd : 1;
    const uint width = ATG / stripes;           // dimensions per stripe pass
    const uint jd = tid % width, stripe = tid / width;
    float m = -1.0e30f, l = 0.0f;
    float acc0 = 0.0f;                          // this thread's dimension jd (hd <= ATG), its stripe's positions
    for (uint b0 = 0; b0 < n; b0 += ATG) {
        const uint nb = min(uint(ATG), n - b0);
        // scores of this block: thread tid takes position b0 + tid
        float sc = -1.0e30f;
        if (tid < nb) {
            device const float* kt = k + ((b0 + tid) * a.n_kv + kh) * a.hd;
            float s = 0.0f;
            for (uint j = 0; j < a.hd; j++) s += qs[j] * kt[j];
            sc = s * scale;
        }
        // the block's max
        float bm = simd_max(sc);
        if (lane == 0) red[sg] = bm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bm = red[0];
        for (uint i = 1; i < ATG / 32; i++) bm = max(bm, red[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float m2 = max(m, bm);
        const float corr = exp(m - m2);
        const float e = tid < nb ? exp(sc - m2) : 0.0f;
        p[tid] = e;
        float bs = simd_sum(e);
        if (lane == 0) red[sg] = bs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bs = 0.0f;
        for (uint i = 0; i < ATG / 32; i++) bs += red[i];
        l = l * corr + bs;
        m = m2;
        // V: dimension jd, positions stripe, stripe + stripes, ... of the block
        acc0 *= corr;
        if (jd < a.hd) {
            for (uint i = stripe; i < nb; i += stripes) acc0 += p[i] * v[((b0 + i) * a.n_kv + kh) * a.hd + jd];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    accs[tid] = acc0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (stripe == 0 && jd < a.hd) {
        float sum = 0.0f;
        for (uint i = 0; i < stripes; i++) sum += accs[i * width + jd];
        o[(t * a.n_head + hh) * a.hd + jd] = sum / l;
    }
}
