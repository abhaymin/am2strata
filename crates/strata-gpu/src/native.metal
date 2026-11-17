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
        // the blocks are 144 / 176 bytes and rows whole blocks: the quants are read a word (four values) at a time
        device const uint* qs = (device const uint*)(o + (kind == K_Q5_K ? 48 : 16) + (g / 2) * 32);
        uint sh = (g % 2) * 4;
        float part = 0.0f;
        if (kind == K_Q4_K) {
            for (uint i = 0; i < 8; i++) {
                const uint q = qs[i] >> sh;
                part += float(q & 0x0f) * xv[4 * i] + float((q >> 8) & 0x0f) * xv[4 * i + 1]
                      + float((q >> 16) & 0x0f) * xv[4 * i + 2] + float((q >> 24) & 0x0f) * xv[4 * i + 3];
            }
        } else {
            device const uint* qh = (device const uint*)(o + 16);
            for (uint i = 0; i < 8; i++) {
                const uint q = qs[i] >> sh;
                const uint h = qh[i] >> g;
                part += float((q & 0x0f) | ((h & 1) << 4)) * xv[4 * i] + float(((q >> 8) & 0x0f) | (((h >> 8) & 1) << 4)) * xv[4 * i + 1]
                      + float(((q >> 16) & 0x0f) | (((h >> 16) & 1) << 4)) * xv[4 * i + 2]
                      + float(((q >> 24) & 0x0f) | (((h >> 24) & 1) << 4)) * xv[4 * i + 3];
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
        // 210-byte blocks: two bytes at a time
        device const ushort* lo2 = (device const ushort*)lo;
        device const ushort* hi2 = (device const ushort*)hi;
        float p0 = 0.0f, p1 = 0.0f;
        for (uint i = 0; i < 16; i++) {
            const uint l = lo2[i] >> lo_sh, h = hi2[i] >> (2 * kk);
            const float v = float(int((l & 0x0f) | ((h & 3) << 4)) - 32) * xv[2 * i]
                          + float(int(((l >> 8) & 0x0f) | (((h >> 8) & 3) << 4)) - 32) * xv[2 * i + 1];
            if (i < 8) p0 += v; else p1 += v;
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
template <uint NR>
static inline void mv_rows(device const uchar* w, device const float* x, device float* y, constant MvArgs& a,
                           uint2 tg, uint sg, uint lane) {
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

// NR rows per SIMD group: 4 for big matrices (x reused across them), 1 for small ones (more threadgroups in flight)
kernel void mv(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
               constant MvArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    mv_rows<4>(w, x, y, a, tg, sg, lane);
}

kernel void mv1(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                constant MvArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    mv_rows<1>(w, x, y, a, tg, sg, lane);
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

// K and V caches are half on the native backend (as the CPU path rounds them, and ggml keeps them): half the memory and
// half the reads of a long context.
kernel void kv_append(device const float* x [[buffer(0)]], device half* cache [[buffer(1)]], constant NArgs& a [[buffer(2)]],
                      uint j [[thread_position_in_grid]]) {
    if (j < a.n) cache[a.at + j] = half(x[j]);
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
kernel void attend(device const float* q [[buffer(0)]], device const half* k [[buffer(1)]], device const half* v [[buffer(2)]],
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
        device const half* kt = k + (t * a.n_kv + kh) * a.hd;
        float acc = 0.0f;
        for (uint j = lane; j < a.hd; j += 32) acc += qh[j] * float(kt[j]);
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
            for (uint t = stripe; t < a.n; t += stripes) acc += sh[t] * float(v[(t * a.n_kv + kh) * a.hd + j]);
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
        device const uint* qs = (device const uint*)(o + (kind == K_Q5_K ? 48 : 16) + (g / 2) * 32);
        const uint sh = (g % 2) * 4;
        if (kind == K_Q4_K) {
            for (uint i = 0; i < 8; i++) {
                const uint q = qs[i] >> sh;
                wv[4 * i] = a * float(q & 0x0f) - b;
                wv[4 * i + 1] = a * float((q >> 8) & 0x0f) - b;
                wv[4 * i + 2] = a * float((q >> 16) & 0x0f) - b;
                wv[4 * i + 3] = a * float((q >> 24) & 0x0f) - b;
            }
        } else {
            device const uint* qh = (device const uint*)(o + 16);
            for (uint i = 0; i < 8; i++) {
                const uint q = qs[i] >> sh, h = qh[i] >> g;
                wv[4 * i] = a * float((q & 0x0f) | ((h & 1) << 4)) - b;
                wv[4 * i + 1] = a * float(((q >> 8) & 0x0f) | (((h >> 8) & 1) << 4)) - b;
                wv[4 * i + 2] = a * float(((q >> 16) & 0x0f) | (((h >> 16) & 1) << 4)) - b;
                wv[4 * i + 3] = a * float(((q >> 24) & 0x0f) | (((h >> 24) & 1) << 4)) - b;
            }
        }
    } else if (kind == K_Q6_K) {
        const float d = f16_at(o + 208);
        const uint hh = g / 4, kk = g % 4;
        device const uchar* lo = o + hh * 64 + (kk % 2) * 32;
        const uint lo_sh = 4 * (kk / 2);
        device const uchar* hi = o + 128 + hh * 32;
        const float s0 = d * float(as_type<char>(o[192 + g * 2]));
        const float s1 = d * float(as_type<char>(o[192 + g * 2 + 1]));
        device const ushort* lo2 = (device const ushort*)lo;
        device const ushort* hi2 = (device const ushort*)hi;
        for (uint i = 0; i < 16; i++) {
            const uint l = lo2[i] >> lo_sh, h = hi2[i] >> (2 * kk);
            const float sc = i < 8 ? s0 : s1;
            wv[2 * i] = sc * float(int((l & 0x0f) | ((h & 3) << 4)) - 32);
            wv[2 * i + 1] = sc * float(int(((l >> 8) & 0x0f) | (((h >> 8) & 3) << 4)) - 32);
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
// threadgroup memory (as half: a dequantized weight is a half scale times a small integer, as ggml's Metal GEMM keeps it)
// while the other 64 load the tokens' 32 values (f32), then each of the 4 SIMD groups multiplies its 32 x 16 corner in
// 8 x 8 tiles, accumulating in f32. One 8 KiB array holds both tiles and then the results, so several threadgroups fit
// on a core.
kernel void mm(device const uchar* w [[buffer(0)]],
               device const float* x [[buffer(1)]],
               device float* y [[buffer(2)]],
               constant MmArgs& a [[buffer(3)]],
               uint2 tg [[threadgroup_position_in_grid]],
               uint2 tid2 [[thread_position_in_threadgroup]],
               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float mem[MM_BM * MM_BN];                          // 8 KiB
    threadgroup half* sa = (threadgroup half*)mem;                 // weights [row][k]: 64 x 32 half = 4 KiB
    threadgroup float* sb = mem + MM_BM * 32 / 2;                  // tokens [token][k]: 32 x 32 f32 = 4 KiB
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
            threadgroup half4* sp = (threadgroup half4*)(sa + tid * 32);
            for (uint l = 0; l < 8; l++) sp[l] = half4(wv[4 * l], wv[4 * l + 1], wv[4 * l + 2], wv[4 * l + 3]);
        } else {
            const uint i = tid - MM_BM;
            const uint tt = i / 2, h = (i % 2) * 16;
            const uint t = t0 + tt;
            if (t < a.tokens) {
                device const float4* xp = (device const float4*)(x + t * a.cols + gi * 32 + h);
                threadgroup float4* sp = (threadgroup float4*)(sb + tt * 32 + h);
                for (uint l = 0; l < 4; l++) sp[l] = xp[l];
            } else {
                for (uint l = 0; l < 16; l++) sb[tt * 32 + h + l] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_half8x8 am[4];
            simdgroup_float8x8 bm[2];
            for (uint i = 0; i < 4; i++) simdgroup_load(am[i], sa + (mr + i * 8) * 32 + kk, 32);
            for (uint j = 0; j < 2; j++) simdgroup_load(bm[j], sb + (mt + j * 8) * 32 + kk, 32, ulong2(0, 0), true);
            for (uint i = 0; i < 4; i++) for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(c[i][j], am[i], bm[j], c[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* sc = mem;
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
kernel void attend_tokens(device const float* q [[buffer(0)]], device const half* k [[buffer(1)]], device const half* v [[buffer(2)]],
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
            device const half* kt = k + ((b0 + tid) * a.n_kv + kh) * a.hd;
            float s = 0.0f;
            for (uint j = 0; j < a.hd; j++) s += qs[j] * float(kt[j]);
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
            for (uint i = stripe; i < nb; i += stripes) acc0 += p[i] * float(v[((b0 + i) * a.n_kv + kh) * a.hd + jd]);
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

// ------------------------------------------------------------------ decoding attention over a long context
// Split by positions (flash decoding): SIMD group p of the grid takes positions p * DP .. p * DP + DP of one KV head, for
// all `group` query heads that read it (so K and V are read once per KV head, not once per query head), keeping a
// running max, sum and weighted V per query head (an online softmax). Its partial result goes to `part`; attend_merge
// then combines the parts of each query head.

#define DP 32          // positions per SIMD group
#define DG 8           // most query heads per KV head
#define DV 8           // most values per lane (head dimension <= 256)

struct SplitArgs { uint n; uint n_kv; uint group; uint hd; float scale; uint parts; };

// part[(hh * parts + p) * (hd + 2) ..] = (max, sum, sum_t e^(s_t - max) v_t[0 .. hd])
kernel void attend_split(device const float* q [[buffer(0)]], device const half* k [[buffer(1)]], device const half* v [[buffer(2)]],
                         device float* part [[buffer(3)]], constant SplitArgs& a [[buffer(4)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint kh = tg.x;
    const uint p = tg.y * nsg + sg;
    if (p >= a.parts) return;
    const uint per = a.hd / 32;                 // values per lane
    float qv[DG][DV], acc[DG][DV], m[DG], l[DG];
    for (uint h = 0; h < a.group; h++) {
        device const float* qh = q + (kh * a.group + h) * a.hd;
        for (uint i = 0; i < per; i++) { qv[h][i] = qh[lane * per + i]; acc[h][i] = 0.0f; }
        m[h] = -1.0e30f;
        l[h] = 0.0f;
    }
    const uint t1 = min(a.n, (p + 1) * DP);
    for (uint t = p * DP; t < t1; t++) {
        device const half* kt = k + (t * a.n_kv + kh) * a.hd + lane * per;
        device const half* vt = v + (t * a.n_kv + kh) * a.hd + lane * per;
        float kv[DV], vv[DV];
        if (per == 4) {
            const float4 k4 = float4(*(device const half4*)kt), v4 = float4(*(device const half4*)vt);
            kv[0] = k4.x; kv[1] = k4.y; kv[2] = k4.z; kv[3] = k4.w;
            vv[0] = v4.x; vv[1] = v4.y; vv[2] = v4.z; vv[3] = v4.w;
        } else {
            for (uint i = 0; i < per; i++) { kv[i] = float(kt[i]); vv[i] = float(vt[i]); }
        }
        for (uint h = 0; h < a.group; h++) {
            float s = 0.0f;
            for (uint i = 0; i < per; i++) s += qv[h][i] * kv[i];
            s = simd_sum(s) * a.scale;
            const float m2 = max(m[h], s);
            const float corr = exp(m[h] - m2), e = exp(s - m2);
            l[h] = l[h] * corr + e;
            for (uint i = 0; i < per; i++) acc[h][i] = acc[h][i] * corr + e * vv[i];
            m[h] = m2;
        }
    }
    for (uint h = 0; h < a.group; h++) {
        device float* out = part + ((kh * a.group + h) * a.parts + p) * (a.hd + 2);
        if (lane == 0) { out[0] = m[h]; out[1] = l[h]; }
        for (uint i = 0; i < per; i++) out[2 + lane * per + i] = acc[h][i];
    }
}

// o[hh][j] = (sum_p e^(m_p - M) acc_p[j]) / (sum_p e^(m_p - M) l_p): threadgroup = query head, thread = value.
struct MergeArgs { uint hd; uint parts; };
kernel void attend_merge(device const float* part [[buffer(0)]], device float* o [[buffer(1)]], constant MergeArgs& a [[buffer(2)]],
                         uint hh [[threadgroup_position_in_grid]], uint j [[thread_position_in_threadgroup]]) {
    if (j >= a.hd) return;
    device const float* ph = part + hh * a.parts * (a.hd + 2);
    float big = -1.0e30f;
    for (uint p = 0; p < a.parts; p++) big = max(big, ph[p * (a.hd + 2)]);
    float sum = 0.0f, acc = 0.0f;
    for (uint p = 0; p < a.parts; p++) {
        device const float* pp = ph + p * (a.hd + 2);
        const float f = exp(pp[0] - big);
        sum += f * pp[1];
        acc += f * pp[2 + j];
    }
    o[hh * a.hd + j] = acc / sum;
}

// ------------------------------------------------------------------ a prompt chunk's attention on the matrix units
// Flash attention: threadgroup (h, j) takes query head h for tokens j * 32 .. j * 32 + 32 (each of its 4 SIMD groups owns 8
// of them) and walks the keys in blocks of 32: S = Q K^T in 8 x 8 tiles, an online softmax per row (causal: a token sees
// positions up to its own), then O = diag(corr) O + P V. Q, K and V go through threadgroup memory as half; S, P and O stay
// f32. Heads of up to 128 values.
#define FQ 32
#define FK 32
#define FHD 128

struct FlashArgs { uint base; uint tokens; uint n_head; uint n_kv; uint group; uint hd; };

kernel void attend_flash(device const float* q [[buffer(0)]], device const half* k [[buffer(1)]], device const half* v [[buffer(2)]],
                         device const float* scales [[buffer(3)]], device float* o [[buffer(4)]], constant FlashArgs& a [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half qs[FQ * FHD];
    threadgroup half kvs[FK * FHD];
    threadgroup float ss[FQ * FK];
    threadgroup float dg[4 * 64];
    const uint tid = tid2.x;
    const uint h = tg.x, kh = h / a.group;
    const uint t0 = tg.y * FQ;
    const uint hd = a.hd, nd = hd / 8;
    // Q rows for this block's tokens
    for (uint e = tid; e < FQ * hd; e += 128) {
        const uint r = e / hd, d = e % hd;
        qs[r * FHD + d] = t0 + r < a.tokens ? half(q[((t0 + r) * a.n_head + h) * hd + d]) : half(0.0f);
    }
    simdgroup_float8x8 oc[FHD / 8];
    for (uint i = 0; i < nd; i++) oc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint row0 = sg * 8;                     // this SIMD group's 8 query rows
    float m[8], l[8];
    for (uint r = 0; r < 8; r++) { m[r] = -1.0e30f; l[r] = 0.0f; }
    const uint last = min(a.tokens, t0 + FQ) - 1; // the block's last token
    const uint n = a.base + last + 1;             // positions any token of the block sees
    threadgroup float* d8 = dg + sg * 64;
    for (uint p0 = 0; p0 < n; p0 += FK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < FK * hd; e += 128) {
            const uint r = e / hd, d = e % hd;
            kvs[r * FHD + d] = p0 + r < n ? k[((p0 + r) * a.n_kv + kh) * hd + d] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S rows row0 .. row0 + 8, 32 positions
        for (uint c = 0; c < FK / 8; c++) {
            simdgroup_float8x8 sc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint kk = 0; kk < nd; kk++) {
                simdgroup_half8x8 qa, kb;
                simdgroup_load(qa, qs + row0 * FHD + kk * 8, FHD);
                simdgroup_load(kb, kvs + c * 8 * FHD + kk * 8, FHD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sc, qa, kb, sc);
            }
            simdgroup_store(sc, ss + row0 * FK + c * 8, FK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // the online softmax, row by row: lane j is position p0 + j
        for (uint r = 0; r < 8; r++) {
            const uint t = t0 + row0 + r;
            const uint pos = p0 + lane;
            const bool live = t < a.tokens && pos <= a.base + t;
            const float s = live ? ss[(row0 + r) * FK + lane] * scales[min(t, a.tokens - 1)] : -1.0e30f;
            const float m2 = max(m[r], simd_max(s));
            const float corr = exp(m[r] - m2);
            const float e = live ? exp(s - m2) : 0.0f;
            l[r] = l[r] * corr + simd_sum(e);
            m[r] = m2;
            ss[(row0 + r) * FK + lane] = e;
            if (lane < 8) d8[r * 8 + lane] = lane == r ? corr : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dm;
        simdgroup_load(dm, d8, 8);
        for (uint i = 0; i < nd; i++) simdgroup_multiply(oc[i], dm, oc[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < FK * hd; e += 128) {
            const uint r = e / hd, d = e % hd;
            kvs[r * FHD + d] = p0 + r < n ? v[((p0 + r) * a.n_kv + kh) * hd + d] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < FK / 8; kk++) {
            simdgroup_float8x8 pa;
            simdgroup_load(pa, ss + row0 * FK + kk * 8, FK);
            for (uint i = 0; i < nd; i++) {
                simdgroup_half8x8 vb;
                simdgroup_load(vb, kvs + kk * 8 * FHD + i * 8, FHD);
                simdgroup_multiply_accumulate(oc[i], pa, vb, oc[i]);
            }
        }
    }
    // O / l, then out: rows of tokens t, each n_head * hd apart
    for (uint r = 0; r < 8; r++) if (lane < 8) d8[r * 8 + lane] = lane == r ? (l[r] > 0.0f ? 1.0f / l[r] : 0.0f) : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dm;
    simdgroup_load(dm, d8, 8);
    for (uint i = 0; i < nd; i++) {
        simdgroup_multiply(oc[i], dm, oc[i]);
        simdgroup_store(oc[i], d8, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64; e += 32) {
            const uint r = e / 8, c = e % 8;
            const uint t = t0 + row0 + r;
            if (t < a.tokens) o[(t * a.n_head + h) * hd + i * 8 + c] = d8[e];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// The same split decoding attention on the matrix units: threadgroup (one SIMD group) = (KV head, part of DPM positions).
// The query heads of the KV head (up to 8, padded to 8 rows) times 8 positions at a time: S = Q K^T in 8 x 8 tiles with
// K read straight from the cache, an online softmax per row, O = diag(corr) O + P V. Its part goes to `part` as in
// attend_split, and attend_merge combines them. The caches carry 8 spare rows (kv_cache), so a last block of 8 may read
// past the context (those positions are masked).
#define DPM 128

kernel void attend_split_mma(device const float* q [[buffer(0)]], device const half* k [[buffer(1)]], device const half* v [[buffer(2)]],
                             device float* part [[buffer(3)]], constant SplitArgs& a [[buffer(4)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half qs[8 * FHD];
    threadgroup float ss[64];
    threadgroup float d8[64];
    threadgroup float os[8 * FHD];
    const uint kh = tg.x, p = tg.y;
    const uint hd = a.hd, nd = hd / 8, G = a.group;
    const uint stride = a.n_kv * hd;
    for (uint e = lane; e < 8 * hd; e += 32) {
        const uint r = e / hd, d = e % hd;
        qs[r * FHD + d] = r < G ? half(q[(kh * G + r) * hd + d]) : half(0.0f);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 oc[FHD / 8];
    for (uint i = 0; i < nd; i++) oc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    // lane holds elements e0 = lane (row lane / 8) and e1 = lane + 32 (row lane / 8 + 4) of each 8 x 8 tile
    const uint e0 = lane, e1 = lane + 32, c = lane % 8;
    float m0 = -1.0e30f, m1 = -1.0e30f, l0 = 0.0f, l1 = 0.0f;
    const uint pend = min(a.n, (p + 1) * DPM);
    for (uint p0 = p * DPM; p0 < pend; p0 += 8) {
        simdgroup_float8x8 sc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        device const half* kb0 = k + p0 * stride + kh * hd;
        for (uint kk = 0; kk < nd; kk++) {
            simdgroup_half8x8 qa, kb;
            simdgroup_load(qa, qs + kk * 8, FHD);
            simdgroup_load(kb, kb0 + kk * 8, stride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(sc, qa, kb, sc);
        }
        simdgroup_store(sc, ss, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const bool live = p0 + c < pend;
        const float s0 = live ? ss[e0] * a.scale : -1.0e30f;
        const float s1 = live ? ss[e1] * a.scale : -1.0e30f;
        // row maxima and sums over the 8 lanes of a row
        float b0 = s0, b1 = s1;
        for (uint o = 1; o < 8; o <<= 1) { b0 = max(b0, simd_shuffle_xor(b0, o)); b1 = max(b1, simd_shuffle_xor(b1, o)); }
        const float n0 = max(m0, b0), n1 = max(m1, b1);
        const float c0 = exp(m0 - n0), c1 = exp(m1 - n1);
        const float x0 = live ? exp(s0 - n0) : 0.0f, x1 = live ? exp(s1 - n1) : 0.0f;
        float t0 = x0, t1 = x1;
        for (uint o = 1; o < 8; o <<= 1) { t0 += simd_shuffle_xor(t0, o); t1 += simd_shuffle_xor(t1, o); }
        l0 = l0 * c0 + t0; l1 = l1 * c1 + t1;
        m0 = n0; m1 = n1;
        ss[e0] = x0; ss[e1] = x1;
        d8[e0] = (e0 % 8 == e0 / 8) ? c0 : 0.0f;
        d8[e1] = (e1 % 8 == e1 / 8) ? c1 : 0.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dm, pa;
        simdgroup_load(dm, d8, 8);
        simdgroup_load(pa, ss, 8);
        device const half* vb0 = v + p0 * stride + kh * hd;
        for (uint i = 0; i < nd; i++) {
            simdgroup_half8x8 vb;
            simdgroup_load(vb, vb0 + i * 8, stride);
            simdgroup_multiply(oc[i], dm, oc[i]);
            simdgroup_multiply_accumulate(oc[i], pa, vb, oc[i]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < nd; i++) simdgroup_store(oc[i], os + i * 8, FHD);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // row r's (max, sum) live in lanes r * 8 .. (rows 0-3, e0) and (rows 4-7, e1)
    for (uint r = 0; r < G; r++) {
        device float* out = part + ((kh * G + r) * a.parts + p) * (hd + 2);
        if (lane == (r % 4) * 8) {
            out[0] = r < 4 ? m0 : m1;
            out[1] = r < 4 ? l0 : l1;
        }
        for (uint d = lane; d < hd; d += 32) out[2 + d] = os[r * FHD + d];
    }
}

// A prompt chunk's attention, a SIMD group per (query head, 8 tokens): threadgroup (KV head, 8 tokens) holds the `group`
// query heads that read that KV head, so they walk the same K and V together (read straight from the cache in 8 x 8
// tiles: no staging, no barriers between the SIMD groups). Per block of 8 positions: S = Q K^T, an online softmax per
// row (causal), O = diag(corr) O + P V.
kernel void attend_flash2(device const float* q [[buffer(0)]], device const half* k [[buffer(1)]], device const half* v [[buffer(2)]],
                          device const float* scales [[buffer(3)]], device float* o [[buffer(4)]], constant FlashArgs& a [[buffer(5)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half qall[8 * 8 * FHD];       // per SIMD group: its 8 rows of Q
    threadgroup float sall[8 * 64];           // per SIMD group: an 8 x 8 tile
    threadgroup float dall[8 * 64];
    const uint kh = tg.x, h = kh * a.group + sg;
    const uint t0 = tg.y * 8;
    const uint hd = a.hd, nd = hd / 8;
    const uint stride = a.n_kv * hd;
    threadgroup half* qs = qall + sg * 8 * FHD;
    threadgroup float* ss = sall + sg * 64;
    threadgroup float* d8 = dall + sg * 64;
    for (uint e = lane; e < 8 * hd; e += 32) {
        const uint r = e / hd, d = e % hd;
        qs[r * FHD + d] = t0 + r < a.tokens ? half(q[((t0 + r) * a.n_head + h) * hd + d]) : half(0.0f);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 oc[FHD / 8];
    for (uint i = 0; i < nd; i++) oc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint e0 = lane, e1 = lane + 32, c = lane % 8;
    const uint ta = t0 + e0 / 8, tb = t0 + e1 / 8;          // the tokens of this lane's two rows
    const float sca = scales[min(ta, a.tokens - 1)], scb = scales[min(tb, a.tokens - 1)];
    float m0 = -1.0e30f, m1 = -1.0e30f, l0 = 0.0f, l1 = 0.0f;
    const uint pend = a.base + min(a.tokens, t0 + 8);       // positions the block's last token sees
    for (uint p0 = 0; p0 < pend; p0 += 8) {
        simdgroup_float8x8 sc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        device const half* kb0 = k + p0 * stride + kh * hd;
        for (uint kk = 0; kk < nd; kk++) {
            simdgroup_half8x8 qa, kb;
            simdgroup_load(qa, qs + kk * 8, FHD);
            simdgroup_load(kb, kb0 + kk * 8, stride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(sc, qa, kb, sc);
        }
        simdgroup_store(sc, ss, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint pos = p0 + c;
        const bool la = ta < a.tokens && pos <= a.base + ta;
        const bool lb = tb < a.tokens && pos <= a.base + tb;
        const float s0 = la ? ss[e0] * sca : -1.0e30f;
        const float s1 = lb ? ss[e1] * scb : -1.0e30f;
        float b0 = s0, b1 = s1;
        for (uint off = 1; off < 8; off <<= 1) { b0 = max(b0, simd_shuffle_xor(b0, off)); b1 = max(b1, simd_shuffle_xor(b1, off)); }
        const float n0 = max(m0, b0), n1 = max(m1, b1);
        const float c0 = exp(m0 - n0), c1 = exp(m1 - n1);
        const float x0 = la ? exp(s0 - n0) : 0.0f, x1 = lb ? exp(s1 - n1) : 0.0f;
        float u0 = x0, u1 = x1;
        for (uint off = 1; off < 8; off <<= 1) { u0 += simd_shuffle_xor(u0, off); u1 += simd_shuffle_xor(u1, off); }
        l0 = l0 * c0 + u0; l1 = l1 * c1 + u1;
        m0 = n0; m1 = n1;
        ss[e0] = x0; ss[e1] = x1;
        d8[e0] = (e0 % 8 == e0 / 8) ? c0 : 0.0f;
        d8[e1] = (e1 % 8 == e1 / 8) ? c1 : 0.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dm, pa;
        simdgroup_load(dm, d8, 8);
        simdgroup_load(pa, ss, 8);
        device const half* vb0 = v + p0 * stride + kh * hd;
        for (uint i = 0; i < nd; i++) {
            simdgroup_half8x8 vb;
            simdgroup_load(vb, vb0 + i * 8, stride);
            simdgroup_multiply(oc[i], dm, oc[i]);
            simdgroup_multiply_accumulate(oc[i], pa, vb, oc[i]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // O / l: the row sums live in lanes r * 8 (rows 0-3: l0, rows 4-7: l1)
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    d8[e0] = (e0 % 8 == e0 / 8) ? i0 : 0.0f;
    d8[e1] = (e1 % 8 == e1 / 8) ? i1 : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dm;
    simdgroup_load(dm, d8, 8);
    for (uint i = 0; i < nd; i++) {
        simdgroup_multiply(oc[i], dm, oc[i]);
        simdgroup_store(oc[i], ss, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64; e += 32) {
            const uint t = t0 + e / 8;
            if (t < a.tokens) o[(t * a.n_head + h) * hd + i * 8 + e % 8] = ss[e];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
