//! The kernels, in CubeCL: written once, compiled at run time for CUDA, HIP, Vulkan (wgpu) or the CPU.
//!
//! Weights stay in their GGUF encoding on the device, uploaded as bytes in a `u32` array (k-quant blocks are not
//! 4-byte aligned: Q6_K's are 210 bytes), and are dequantized as they are read, exactly as strata_pack::quants does on
//! the CPU. The activations arrive already rounded the way ggml rounds them (the caller's Vdt), so the products
//! are the CPU engine's; only the order of the sums differs.

#![allow(unused_assignments)] // a `let mut` given a value in every branch: how CubeCL kernels declare it

use cubecl::prelude::*;

/// Units per cube for the row kernels (a power of two, at most every backend's limit).
pub const CUBE: u32 = 128;

/// Weight encodings with a kernel.
pub mod kind {
    pub const F32: u32 = 0;
    pub const F16: u32 = 1;
    pub const BF16: u32 = 2;
    pub const Q8_0: u32 = 3;
    pub const Q4_0: u32 = 4;
    pub const Q4_K: u32 = 5;
    pub const Q5_K: u32 = 6;
    pub const Q6_K: u32 = 7;
}

/// (values per block, bytes per block, kind) of a GGUF type name this crate has a kernel for.
pub fn geometry(ty: &str) -> Option<(u32, u32, u32)> {
    Some(match ty {
        "F32" => (32, 128, kind::F32),
        "F16" => (32, 64, kind::F16),
        "BF16" => (32, 64, kind::BF16),
        "Q8_0" => (32, 34, kind::Q8_0),
        "Q4_0" => (32, 18, kind::Q4_0),
        "Q4_K" => (256, 144, kind::Q4_K),
        "Q5_K" => (256, 176, kind::Q5_K),
        "Q6_K" => (256, 210, kind::Q6_K),
        _ => return None,
    })
}

/// Byte `off` of the weights.
#[cube]
fn byte(w: &Array<u32>, off: u32) -> u32 {
    (w[(off / 4) as usize] >> ((off % 4) * 8)) & 0xff
}

/// The f16 at byte `off` (little-endian), as f32.
#[cube]
fn half(w: &Array<u32>, off: u32) -> f32 {
    let h = byte(w, off) | (byte(w, off + 1) << 8);
    let e = (h >> 10) & 31;
    let m = h & 1023;
    let mut v = f32::new(0.0);
    if e == 0 {
        v = f32::cast_from(m) * 5.9604645e-8f32; // subnormal: m * 2^-24
    } else if e == 31 {
        v = f32::reinterpret(0x7f80_0000u32 | (m << 13));
    } else {
        v = f32::reinterpret(((e + 112) << 23) | (m << 13));
    }
    if (h >> 15) == 1 {
        v = -v;
    }
    v
}

/// A byte as a signed 8-bit value.
#[cube]
fn signed(b: u32) -> f32 {
    f32::cast_from(b) - f32::cast_from((b >> 7) * 256)
}

/// The dot product of one block (at byte `o`) with x[xo..].
#[cube]
fn block_dot(w: &Array<u32>, o: u32, x: &Array<f32>, xo: u32, #[comptime] k: u32) -> f32 {
    let mut acc = f32::new(0.0);
    if comptime![k == kind::F32] {
        for j in 0..32u32 {
            let bits = byte(w, o + 4 * j) | (byte(w, o + 4 * j + 1) << 8) | (byte(w, o + 4 * j + 2) << 16) | (byte(w, o + 4 * j + 3) << 24);
            acc += f32::reinterpret(bits) * x[(xo + j) as usize];
        }
    } else if comptime![k == kind::F16] {
        for j in 0..32u32 {
            acc += half(w, o + 2 * j) * x[(xo + j) as usize];
        }
    } else if comptime![k == kind::BF16] {
        for j in 0..32u32 {
            let bits = (byte(w, o + 2 * j) | (byte(w, o + 2 * j + 1) << 8)) << 16;
            acc += f32::reinterpret(bits) * x[(xo + j) as usize];
        }
    } else if comptime![k == kind::Q8_0] {
        let d = half(w, o);
        for j in 0..32u32 {
            acc += signed(byte(w, o + 2 + j)) * x[(xo + j) as usize];
        }
        acc *= d;
    } else if comptime![k == kind::Q4_0] {
        let d = half(w, o);
        for j in 0..16u32 {
            let b = byte(w, o + 2 + j);
            acc += (f32::cast_from(b & 0x0f) - 8.0) * x[(xo + j) as usize];
            acc += (f32::cast_from(b >> 4) - 8.0) * x[(xo + 16 + j) as usize];
        }
        acc *= d;
    } else if comptime![k == kind::Q4_K || k == kind::Q5_K] {
        // 8 groups of 32: a 6-bit scale and min each (packed in 12 bytes), then 4-bit values (+ a 5th bit for Q5_K)
        let d = half(w, o);
        let dmin = half(w, o + 2);
        let s = o + 4;
        let qs = if comptime![k == kind::Q5_K] { o + 48 } else { o + 16 };
        let qh = o + 16;
        for g in 0..8u32 {
            let j = g % 4;
            let mut sc = u32::new(0);
            let mut mn = u32::new(0);
            if g < 4 {
                sc = byte(w, s + j) & 0x3f;
                mn = byte(w, s + 4 + j) & 0x3f;
            } else {
                sc = (byte(w, s + 8 + j) & 0x0f) | ((byte(w, s + j) >> 2) & 0x30);
                mn = (byte(w, s + 8 + j) >> 4) | ((byte(w, s + 4 + j) >> 2) & 0x30);
            }
            let dd = d * f32::cast_from(sc);
            let dm = dmin * f32::cast_from(mn);
            let c = g / 2;
            let sh = (g % 2) * 4;
            let mut part = f32::new(0.0);
            let mut xs = f32::new(0.0);
            for l in 0..32u32 {
                let mut q = (byte(w, qs + c * 32 + l) >> sh) & 0x0f;
                if comptime![k == kind::Q5_K] {
                    q |= ((byte(w, qh + l) >> g) & 1) << 4;
                }
                let xv = x[(xo + g * 32 + l) as usize];
                part += f32::cast_from(q) * xv;
                xs += xv;
            }
            acc += dd * part - dm * xs;
        }
    } else if comptime![k == kind::Q6_K] {
        // 16 groups of 16 with signed 8-bit scales; 4 low bits in ql, 2 high bits in qh
        let d = half(w, o + 208);
        for hh in 0..2u32 {
            for kk in 0..4u32 {
                for l in 0..32u32 {
                    let lo = (byte(w, o + hh * 64 + (kk % 2) * 32 + l) >> (4 * (kk / 2))) & 0x0f;
                    let hi = (byte(w, o + 128 + hh * 32 + l) >> (2 * kk)) & 0x03;
                    let q = f32::cast_from(lo | (hi << 4)) - 32.0;
                    let e = (hh * 4 + kk) * 32 + l;
                    acc += d * signed(byte(w, o + 192 + e / 16)) * q * x[(xo + e) as usize];
                }
            }
        }
    }
    acc
}

/// The dot product of 32 values - group `g` of the block at byte `o` - with x[xo..xo + 32]: the unit of work of
/// [`matvec`] (a row's groups are spread over 32 units, so even a short row keeps them busy).
#[cube]
fn group_dot(w: &Array<u32>, o: u32, g: u32, x: &Array<f32>, xo: u32, #[comptime] k: u32) -> f32 {
    let mut acc = f32::new(0.0);
    if comptime![k == kind::F32 || k == kind::F16 || k == kind::BF16 || k == kind::Q8_0 || k == kind::Q4_0] {
        // these blocks are 32 values: the group is the block
        acc = block_dot(w, o, x, xo, k);
    } else if comptime![k == kind::Q4_K || k == kind::Q5_K] {
        let d = half(w, o);
        let dmin = half(w, o + 2);
        let s = o + 4;
        let j = g % 4;
        let mut sc = u32::new(0);
        let mut mn = u32::new(0);
        if g < 4 {
            sc = byte(w, s + j) & 0x3f;
            mn = byte(w, s + 4 + j) & 0x3f;
        } else {
            sc = (byte(w, s + 8 + j) & 0x0f) | ((byte(w, s + j) >> 2) & 0x30);
            mn = (byte(w, s + 8 + j) >> 4) | ((byte(w, s + 4 + j) >> 2) & 0x30);
        }
        let qs = if comptime![k == kind::Q5_K] { o + 48 } else { o + 16 } + (g / 2) * 32;
        let sh = (g % 2) * 4;
        let mut part = f32::new(0.0);
        let mut xs = f32::new(0.0);
        // the blocks are 144 / 176 bytes: whole words, four values per load
        for i in 0..8u32 {
            let word = w[(qs / 4 + i) as usize];
            let mut hb = u32::new(0);
            if comptime![k == kind::Q5_K] {
                hb = w[((o + 16) / 4 + i) as usize];
            }
            for b in 0..4u32 {
                let l = i * 4 + b;
                let mut q = (word >> (b * 8 + sh)) & 0x0f;
                if comptime![k == kind::Q5_K] {
                    q |= ((hb >> (b * 8 + g)) & 1) << 4;
                }
                let xv = x[(xo + l) as usize];
                part += f32::cast_from(q) * xv;
                xs += xv;
            }
        }
        acc = d * f32::cast_from(sc) * part - dmin * f32::cast_from(mn) * xs;
    } else if comptime![k == kind::Q6_K] {
        let d = half(w, o + 208);
        let hh = g / 4;
        let kk = g % 4;
        let lo_at = o + hh * 64 + (kk % 2) * 32;
        let lo_sh = 4 * (kk / 2);
        let hi_at = o + 128 + hh * 32;
        let s0 = d * signed(byte(w, o + 192 + g * 2));
        let s1 = d * signed(byte(w, o + 192 + g * 2 + 1));
        for l in 0..32u32 {
            let lo = (byte(w, lo_at + l) >> lo_sh) & 0x0f;
            let hi = (byte(w, hi_at + l) >> (2 * kk)) & 0x03;
            let q = f32::cast_from(lo | (hi << 4)) - 32.0;
            let sc = if l < 16 { s0 } else { s1 };
            acc += sc * q * x[(xo + l) as usize];
        }
    }
    acc
}

/// Rows per cube in [`matvec`], 32 units each.
pub const ROWS_PER_CUBE: u32 = CUBE / 32;

/// y[t * rows + r] = W[r] . x[t * cols ..], for `tokens` vectors at once (1 for decoding, the prompt's for prefill).
/// One cube per (row, token); its units take the row's blocks in turn and sum in shared memory.
/// y[t * rows + r] = W[r] . x[t * cols ..], for `tokens` vectors at once (1 for decoding, the prompt's for prefill).
/// A cube takes ROWS_PER_CUBE rows; each row's 32-value groups are spread over 32 units, summed in shared memory.
#[cube(launch_unchecked)]
pub fn matvec(w: &Array<u32>, x: &Array<f32>, y: &mut Array<f32>, rows: u32, cols: u32, row_bytes: u32,
              #[comptime] values_per_block: u32, #[comptime] bytes_per_block: u32, #[comptime] k: u32) {
    let cube = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    let lane = UNIT_POS % 32;
    let row = cube * ROWS_PER_CUBE + UNIT_POS / 32;
    let t = CUBE_POS_Z;
    let groups_per_block = values_per_block / 32;
    let mut acc = f32::new(0.0);
    if row < rows {
        let ng = cols / 32;
        let mut gi = lane;
        while gi < ng {
            let b = gi / groups_per_block;
            let g = gi % groups_per_block;
            acc += group_dot(w, row * row_bytes + b * bytes_per_block, g, x, t * cols + gi * 32, k);
            gi += 32;
        }
    }
    let mut sum = SharedMemory::<f32>::new(CUBE as usize);
    sum[UNIT_POS as usize] = acc;
    sync_cube();
    let mut stride = u32::new(16);
    while stride > 0 {
        if lane < stride {
            sum[UNIT_POS as usize] = sum[UNIT_POS as usize] + sum[(UNIT_POS + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && row < rows {
        y[(t * rows + row) as usize] = sum[UNIT_POS as usize];
    }
}

// ------------------------------------------------------------------ a whole token on the device (docs/ENGINE-PLAN.md phase 1)
// These run between the matrix products, so the activations stay on the device and only the logits come back.

/// y = x * w / sqrt(mean(x^2) + eps): the RMS norm of one vector of `n` values. One cube.
#[cube(launch_unchecked)]
pub fn rms_norm(x: &Array<f32>, w: &Array<f32>, y: &mut Array<f32>, n: u32, eps: f32) {
    let mut sum = SharedMemory::<f32>::new(CUBE as usize);
    let mut acc = f32::new(0.0);
    let mut j = UNIT_POS;
    while j < n {
        let v = x[j as usize];
        acc += v * v;
        j += CUBE;
    }
    sum[UNIT_POS as usize] = acc;
    sync_cube();
    let mut stride = u32::new(64);
    while stride > 0 {
        if UNIT_POS < stride {
            sum[UNIT_POS as usize] = sum[UNIT_POS as usize] + sum[(UNIT_POS + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    let k = f32::new(1.0) / f32::sqrt(sum[0usize] / f32::cast_from(n) + eps);
    let mut j = UNIT_POS;
    while j < n {
        y[j as usize] = x[j as usize] * k * w[j as usize];
        j += CUBE;
    }
}

/// y += x, over `n` values.
#[cube(launch_unchecked)]
pub fn add_into(x: &Array<f32>, y: &mut Array<f32>, n: u32) {
    let j = CUBE_POS_X * CUBE + UNIT_POS;
    if j < n {
        y[j as usize] = y[j as usize] + x[j as usize];
    }
}

/// up *= silu(gate), over `n` values (the gated feed-forward).
#[cube(launch_unchecked)]
pub fn silu_gate(gate: &Array<f32>, up: &mut Array<f32>, n: u32) {
    let j = CUBE_POS_X * CUBE + UNIT_POS;
    if j < n {
        let g = gate[j as usize];
        up[j as usize] = up[j as usize] * (g / (f32::new(1.0) + f32::exp(-g)));
    }
}

/// Rotates each of the head vectors of x (`hd` values each) by the (cos, sin) pairs of `tab`: `n_rot / 2` pairs for this
/// position, adjacent values (`neox` 0) or the two halves (`neox` 1). A cube per head.
#[cube(launch_unchecked)]
pub fn rope(tab: &Array<f32>, x: &mut Array<f32>, hd: u32, n_rot: u32, neox: u32) {
    let at = CUBE_POS_X * hd;
    let p = UNIT_POS;
    if p < n_rot / 2 {
        let c = tab[(p * 2) as usize];
        let s = tab[(p * 2 + 1) as usize];
        let mut a = at + p * 2;
        let mut b = a + 1;
        if neox == 1 {
            a = at + p;
            b = a + n_rot / 2;
        }
        let x0 = x[a as usize];
        let x1 = x[b as usize];
        x[a as usize] = x0 * c - x1 * s;
        x[b as usize] = x0 * s + x1 * c;
    }
}

/// cache[at + i] = x[i], over `n` values: one position of a layer's K or V.
#[cube(launch_unchecked)]
pub fn kv_append(x: &Array<f32>, cache: &mut Array<f32>, at: u32, n: u32) {
    let j = CUBE_POS_X * CUBE + UNIT_POS;
    if j < n {
        cache[(at + j) as usize] = x[j as usize];
    }
}

/// Attention scores: s[hh * n + t] = (q_hh . k_t) * scale * ts for every query head and position (a cube per pair).
/// K and V hold `n_kv` heads per position, and query head hh reads KV head hh / group.
#[cube(launch_unchecked)]
pub fn attn_scores(q: &Array<f32>, k: &Array<f32>, s: &mut Array<f32>, n: u32, n_kv: u32, group: u32, hd: u32, scale: f32) {
    let t = CUBE_POS_X;
    let hh = CUBE_POS_Y;
    let kh = hh / group;
    let qo = hh * hd;
    let ko = (t * n_kv + kh) * hd;
    let mut acc = f32::new(0.0);
    let mut j = UNIT_POS;
    while j < hd {
        acc += q[(qo + j) as usize] * k[(ko + j) as usize];
        j += CUBE;
    }
    let mut sum = SharedMemory::<f32>::new(CUBE as usize);
    sum[UNIT_POS as usize] = acc;
    sync_cube();
    let mut stride = u32::new(64);
    while stride > 0 {
        if UNIT_POS < stride {
            sum[UNIT_POS as usize] = sum[UNIT_POS as usize] + sum[(UNIT_POS + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    if UNIT_POS == 0 {
        s[(hh * n + t) as usize] = sum[0usize] * scale;
    }
}

/// s[hh * n..] -> softmax in place, for one query head per cube.
#[cube(launch_unchecked)]
pub fn attn_softmax(s: &mut Array<f32>, n: u32) {
    let hh = CUBE_POS_X;
    let base = hh * n;
    let mut m = SharedMemory::<f32>::new(CUBE as usize);
    let mut top = f32::new(-1.0e30);
    let mut t = UNIT_POS;
    while t < n {
        top = f32::max(top, s[(base + t) as usize]);
        t += CUBE;
    }
    m[UNIT_POS as usize] = top;
    sync_cube();
    let mut stride = u32::new(64);
    while stride > 0 {
        if UNIT_POS < stride {
            m[UNIT_POS as usize] = f32::max(m[UNIT_POS as usize], m[(UNIT_POS + stride) as usize]);
        }
        sync_cube();
        stride /= 2;
    }
    let top = m[0usize];
    sync_cube();
    let mut acc = f32::new(0.0);
    let mut t = UNIT_POS;
    while t < n {
        let e = f32::exp(s[(base + t) as usize] - top);
        s[(base + t) as usize] = e;
        acc += e;
        t += CUBE;
    }
    m[UNIT_POS as usize] = acc;
    sync_cube();
    let mut stride = u32::new(64);
    while stride > 0 {
        if UNIT_POS < stride {
            m[UNIT_POS as usize] = m[UNIT_POS as usize] + m[(UNIT_POS + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    let total = m[0usize];
    let mut t = UNIT_POS;
    while t < n {
        s[(base + t) as usize] = s[(base + t) as usize] / total;
        t += CUBE;
    }
}

/// o[hh * hd + j] = sum over t of p[hh, t] * v[t, hh / group][j]: a cube per query head, a unit per `j % CUBE`.
#[cube(launch_unchecked)]
pub fn attn_blend(p: &Array<f32>, v: &Array<f32>, o: &mut Array<f32>, n: u32, n_kv: u32, group: u32, hd: u32) {
    let hh = CUBE_POS_X;
    let kh = hh / group;
    let mut j = UNIT_POS;
    while j < hd {
        let mut acc = f32::new(0.0);
        let mut t = u32::new(0);
        while t < n {
            acc += p[(hh * n + t) as usize] * v[((t * n_kv + kh) * hd + j) as usize];
            t += 1;
        }
        o[(hh * hd + j) as usize] = acc;
        j += CUBE;
    }
}
