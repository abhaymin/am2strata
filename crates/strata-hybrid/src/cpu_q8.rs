//! Integer row products for the K-quant weights (Q4_K, Q5_K, Q6_K) against activations quantized to Q8_K, and for Q8_0
//! weights against activations quantized to Q8_0, the way ggml's CPU backend computes them (ggml_vec_dot_q4_K_q8_K,
//! ggml_vec_dot_q5_K_q8_K, ggml_vec_dot_q6_K_q8_K, ggml_vec_dot_q8_0_q8_0).
//!
//! The activations are quantized once per matrix product, with exactly `Vdt::Q8K`'s rule (`model::Vdt::round_trip`):
//! per block of 256, iscale = -127 / (the value with the largest magnitude), q = round-half-even(iscale * x), at most
//! 127, d = 1 / iscale. So the values multiplied are the same as the f32 path's (`q * d`). Only the order of the sums
//! differs, and the inner sums are exact in i32.
//!
//! One plain kernel body is compiled several times: with AVX2 on x86-64 and with the dot-product extension on
//! aarch64, each chosen at run time, plus the baseline. The compiler vectorizes the fixed-size loops; there are no
//! hand-written intrinsics to get wrong on a platform this PC cannot run.

use crate::cpu_dot::Fast;
use strata_pack::num::f16_to_f32;

/// A row of activations quantized to Q8_K: i8 values, one scale per 256, and each 32's sum (for Q4_K's minimums).
pub struct Q8K {
    pub q: Vec<i8>,
    pub d: Vec<f32>,
    pub sum32: Vec<i32>,
}

impl Q8K {
    /// None when `x` is not whole blocks of 256 (the caller keeps the f32 path).
    pub fn quantize(x: &[f32]) -> Option<Q8K> {
        if x.is_empty() || x.len() % 256 != 0 {
            return None;
        }
        let mut q = Vec::with_capacity(x.len());
        let mut d = Vec::with_capacity(x.len() / 256);
        for b in x.chunks_exact(256) {
            let (mut amax, mut max) = (0f32, 0f32);
            for v in b {
                if v.abs() > amax {
                    amax = v.abs();
                    max = *v;
                }
            }
            if amax == 0.0 {
                q.extend(std::iter::repeat(0i8).take(256));
                d.push(0.0);
                continue;
            }
            let iscale = -127.0 / max;
            q.extend(b.iter().map(|v| (iscale * v).round_ties_even().min(127.0) as i8));
            d.push(1.0 / iscale);
        }
        let sum32 = q.chunks_exact(32).map(|c| c.iter().map(|&v| v as i32).sum()).collect();
        Some(Q8K { q, d, sum32 })
    }

    /// Q8_0's rule instead (`Vdt::Q8_0`, quantize_row_q8_0_ref): per block of 32, d = amax / 127 stored as f16,
    /// q = round-half-away(x / d). `d` then has one scale per 32 and `sum32` is empty.
    pub fn quantize_q8_0(x: &[f32]) -> Option<Q8K> {
        use strata_pack::num::f32_to_f16;
        if x.is_empty() || x.len() % 32 != 0 {
            return None;
        }
        let mut q = Vec::with_capacity(x.len());
        let mut d = Vec::with_capacity(x.len() / 32);
        for b in x.chunks_exact(32) {
            let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
            let dd = amax / 127.0;
            let id = if dd != 0.0 { 1.0 / dd } else { 0.0 };
            q.extend(b.iter().map(|v| (v * id).round() as i8));
            d.push(f16_to_f32(f32_to_f16(dd)));
        }
        Some(Q8K { q, d, sum32: Vec::new() })
    }

    /// The activations this weight type multiplies with (Q8_0 for Q8_0, Q8_K for the K-quants).
    pub fn quantize_for(t: Fast, x: &[f32]) -> Option<Q8K> {
        if t == Fast::Q8_0 { Q8K::quantize_q8_0(x) } else { Q8K::quantize(x) }
    }
}

/// Whether this weight type has an integer kernel here.
pub fn handles(t: Fast) -> bool {
    matches!(t, Fast::Q4K | Fast::Q5K | Fast::Q6K | Fast::Q8_0)
}

#[inline(always)]
fn h(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

/// Σ a[i] * b[i] over 32 values, exact in i32 (|a|, |b| <= 128: at most 2^19).
#[inline(always)]
fn dot32(a: &[i8; 32], b: &[i8]) -> i32 {
    let b: &[i8; 32] = b.try_into().unwrap();
    let mut s = 0i32;
    for i in 0..32 {
        s += a[i] as i32 * b[i] as i32;
    }
    s
}

#[inline(always)]
fn dot16(a: &[i8], b: &[i8]) -> i32 {
    let (a, b): (&[i8; 16], &[i8; 16]) = (a.try_into().unwrap(), b.try_into().unwrap());
    let mut s = 0i32;
    for i in 0..16 {
        s += a[i] as i32 * b[i] as i32;
    }
    s
}

/// Q4_K block (144 bytes, 256 values): d, dmin (f16), 12 bytes of 6-bit scales and mins, 128 bytes of nibbles. Value
/// = d * scale_g * q - dmin * min_g for group g of 32 (the layout row_dot and dequantize_into read).
#[inline(always)]
fn q4k(row: &[u8], x: &Q8K) -> f32 {
    let mut s = 0f32;
    for (bi, b) in row.chunks_exact(144).enumerate() {
        let (d, dmin) = (h(b, 0), h(b, 2));
        let sc = &b[4..16];
        let qs = &b[16..144];
        let xq = &x.q[bi * 256..bi * 256 + 256];
        let mut acc = 0f32;
        for c in 0..4 {
            // one 32-byte run holds groups 2c (low nibbles) and 2c+1 (high nibbles)
            let mut lo = [0i8; 32];
            let mut hi = [0i8; 32];
            for l in 0..32 {
                let v = qs[c * 32 + l];
                lo[l] = (v & 0x0f) as i8;
                hi[l] = (v >> 4) as i8;
            }
            for (g, vals) in [(2 * c, &lo), (2 * c + 1, &hi)] {
                let j = g % 4;
                let (scale, min) = if g < 4 {
                    (sc[j] & 0x3f, sc[4 + j] & 0x3f)
                } else {
                    ((sc[8 + j] & 0x0f) | ((sc[j] >> 2) & 0x30), (sc[8 + j] >> 4) | ((sc[4 + j] >> 2) & 0x30))
                };
                let p = dot32(vals, &xq[g * 32..g * 32 + 32]);
                acc += d * scale as f32 * p as f32 - dmin * min as f32 * x.sum32[bi * 8 + g] as f32;
            }
        }
        s += acc * x.d[bi];
    }
    s
}

/// Q6_K block (210 bytes, 256 values): 128 bytes of low 4 bits, 64 of high 2 bits, 16 i8 scales (one per 16), d (f16).
/// Value = d * scale_(e/16) * (q - 32) (the layout row_dot and dequantize_into read).
#[inline(always)]
fn q6k(row: &[u8], x: &Q8K) -> f32 {
    let mut s = 0f32;
    for (bi, b) in row.chunks_exact(210).enumerate() {
        let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
        let d = h(b, 208);
        let xq = &x.q[bi * 256..bi * 256 + 256];
        let mut acc = 0i32;
        for hh in 0..2 {
            for k in 0..4 {
                let g = hh * 4 + k;
                let mut v = [0i8; 32];
                for l in 0..32 {
                    let lo = (ql[hh * 64 + (k % 2) * 32 + l] >> (4 * (k / 2))) & 0x0f;
                    let hi = (qh[hh * 32 + l] >> (2 * k)) & 0x03;
                    v[l] = (lo | (hi << 4)) as i8 - 32;
                }
                let xs = &xq[g * 32..g * 32 + 32];
                acc += sc[2 * g] as i8 as i32 * dot16(&v[0..16], &xs[0..16]);
                acc += sc[2 * g + 1] as i8 as i32 * dot16(&v[16..32], &xs[16..32]);
            }
        }
        s += d * x.d[bi] * acc as f32;
    }
    s
}

/// Q4_K for several inputs: each block's nibbles and scales unpacked once, then dotted with every input. Each input's
/// sums are taken in exactly `q4k`'s order, so every result equals `q4k`'s bit for bit.
#[inline(always)]
fn q4k_many(row: &[u8], xs: &[Q8K], out: &mut [f32]) {
    out.iter_mut().for_each(|o| *o = 0.0);
    let mut vals = [[0i8; 32]; 8];
    let mut ds = [0f32; 8];
    let mut ms = [0f32; 8];
    for (bi, b) in row.chunks_exact(144).enumerate() {
        let (d, dmin) = (h(b, 0), h(b, 2));
        let sc = &b[4..16];
        let qs = &b[16..144];
        for c in 0..4 {
            for l in 0..32 {
                let v = qs[c * 32 + l];
                vals[2 * c][l] = (v & 0x0f) as i8;
                vals[2 * c + 1][l] = (v >> 4) as i8;
            }
        }
        for g in 0..8 {
            let j = g % 4;
            let (scale, min) = if g < 4 {
                (sc[j] & 0x3f, sc[4 + j] & 0x3f)
            } else {
                ((sc[8 + j] & 0x0f) | ((sc[j] >> 2) & 0x30), (sc[8 + j] >> 4) | ((sc[4 + j] >> 2) & 0x30))
            };
            ds[g] = d * scale as f32;
            ms[g] = dmin * min as f32;
        }
        for (x, o) in xs.iter().zip(out.iter_mut()) {
            let xq = &x.q[bi * 256..bi * 256 + 256];
            let mut acc = 0f32;
            for c in 0..4 {
                for g in [2 * c, 2 * c + 1] {
                    let p = dot32(&vals[g], &xq[g * 32..g * 32 + 32]);
                    acc += ds[g] * p as f32 - ms[g] * x.sum32[bi * 8 + g] as f32;
                }
            }
            *o += acc * x.d[bi];
        }
    }
}

/// Q6_K for several inputs, unpacked once per block; each result equals `q6k`'s bit for bit.
#[inline(always)]
fn q6k_many(row: &[u8], xs: &[Q8K], out: &mut [f32]) {
    out.iter_mut().for_each(|o| *o = 0.0);
    let mut vals = [[0i8; 32]; 8];
    for (bi, b) in row.chunks_exact(210).enumerate() {
        let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
        let d = h(b, 208);
        for hh in 0..2 {
            for k in 0..4 {
                let g = hh * 4 + k;
                for l in 0..32 {
                    let lo = (ql[hh * 64 + (k % 2) * 32 + l] >> (4 * (k / 2))) & 0x0f;
                    let hi = (qh[hh * 32 + l] >> (2 * k)) & 0x03;
                    vals[g][l] = (lo | (hi << 4)) as i8 - 32;
                }
            }
        }
        for (x, o) in xs.iter().zip(out.iter_mut()) {
            let xq = &x.q[bi * 256..bi * 256 + 256];
            let mut acc = 0i32;
            for g in 0..8 {
                let xs32 = &xq[g * 32..g * 32 + 32];
                acc += sc[2 * g] as i8 as i32 * dot16(&vals[g][0..16], &xs32[0..16]);
                acc += sc[2 * g + 1] as i8 as i32 * dot16(&vals[g][16..32], &xs32[16..32]);
            }
            *o += d * x.d[bi] * acc as f32;
        }
    }
}

/// Q5_K block (176 bytes, 256 values): d, dmin (f16), 12 bytes of 6-bit scales and mins, 32 bytes of fifth bits, 128
/// of nibbles. Value = d * scale_g * (nibble | bit << 4) - dmin * min_g (the layout cpu_dot's row_dot reads). Several
/// inputs, the block unpacked once; one input is the same function (`q5k`), so the results are identical either way.
#[inline(always)]
fn q5k_many(row: &[u8], xs: &[Q8K], out: &mut [f32]) {
    out.iter_mut().for_each(|o| *o = 0.0);
    let mut vals = [[0i8; 32]; 8];
    let mut ds = [0f32; 8];
    let mut ms = [0f32; 8];
    for (bi, b) in row.chunks_exact(176).enumerate() {
        let (d, dmin) = (h(b, 0), h(b, 2));
        let sc = &b[4..16];
        let (qh, qs) = (&b[16..48], &b[48..176]);
        for c in 0..4 {
            for l in 0..32 {
                let v = qs[c * 32 + l];
                vals[2 * c][l] = ((v & 0x0f) | (((qh[l] >> (2 * c)) & 1) << 4)) as i8;
                vals[2 * c + 1][l] = ((v >> 4) | (((qh[l] >> (2 * c + 1)) & 1) << 4)) as i8;
            }
        }
        for g in 0..8 {
            let j = g % 4;
            let (scale, min) = if g < 4 {
                (sc[j] & 0x3f, sc[4 + j] & 0x3f)
            } else {
                ((sc[8 + j] & 0x0f) | ((sc[j] >> 2) & 0x30), (sc[8 + j] >> 4) | ((sc[4 + j] >> 2) & 0x30))
            };
            ds[g] = d * scale as f32;
            ms[g] = dmin * min as f32;
        }
        for (x, o) in xs.iter().zip(out.iter_mut()) {
            let xq = &x.q[bi * 256..bi * 256 + 256];
            let mut acc = 0f32;
            for g in 0..8 {
                let p = dot32(&vals[g], &xq[g * 32..g * 32 + 32]);
                acc += ds[g] * p as f32 - ms[g] * x.sum32[bi * 8 + g] as f32;
            }
            *o += acc * x.d[bi];
        }
    }
}

#[inline(always)]
fn q5k(row: &[u8], x: &Q8K) -> f32 {
    let mut o = [0f32];
    q5k_many(row, std::slice::from_ref(x), &mut o);
    o[0]
}

/// Q8_0 block (34 bytes, 32 values): d (f16), 32 i8. Against Q8_0 activations (`Q8K::quantize_q8_0`): Σ d_w d_x Σ q_w q_x,
/// the inner sum exact in i32. One input is the same function, so the results are identical either way.
#[inline(always)]
fn q80_many(row: &[u8], xs: &[Q8K], out: &mut [f32]) {
    out.iter_mut().for_each(|o| *o = 0.0);
    let mut w = [0i8; 32];
    for (bi, b) in row.chunks_exact(34).enumerate() {
        let d = h(b, 0);
        for j in 0..32 {
            w[j] = b[2 + j] as i8;
        }
        for (x, o) in xs.iter().zip(out.iter_mut()) {
            let p = dot32(&w, &x.q[bi * 32..bi * 32 + 32]);
            *o += d * x.d[bi] * p as f32;
        }
    }
}

#[inline(always)]
fn q80(row: &[u8], x: &Q8K) -> f32 {
    let mut o = [0f32];
    q80_many(row, std::slice::from_ref(x), &mut o);
    o[0]
}

#[inline(always)]
fn body_many(t: Fast, row: &[u8], xs: &[Q8K], out: &mut [f32]) {
    match t {
        Fast::Q4K => q4k_many(row, xs, out),
        Fast::Q5K => q5k_many(row, xs, out),
        Fast::Q6K => q6k_many(row, xs, out),
        Fast::Q8_0 => q80_many(row, xs, out),
        _ => unreachable!("cpu_q8 handles Q4_K, Q5_K, Q6_K and Q8_0"),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_dot_many_avx2(t: Fast, row: &[u8], xs: &[Q8K], out: &mut [f32]) {
    body_many(t, row, xs, out)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn row_dot_many_dotprod(t: Fast, row: &[u8], xs: &[Q8K], out: &mut [f32]) {
    body_many(t, row, xs, out)
}

/// One weight row dotted with several inputs (a prompt chunk): out[i] = row_dot(t, row, &xs[i]), the row unpacked once.
#[inline]
pub fn row_dot_many(t: Fast, row: &[u8], xs: &[Q8K], out: &mut [f32]) {
    debug_assert_eq!(xs.len(), out.len());
    match path() {
        Path::Baseline => body_many(t, row, xs, out),
        // SAFETY: path() returns these only when the CPU has the features
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => unsafe { row_dot_many_avx2(t, row, xs, out) },
        #[cfg(target_arch = "aarch64")]
        Path::DotProd => unsafe { row_dot_many_dotprod(t, row, xs, out) },
    }
}

#[inline(always)]
fn body(t: Fast, row: &[u8], x: &Q8K) -> f32 {
    match t {
        Fast::Q4K => q4k(row, x),
        Fast::Q5K => q5k(row, x),
        Fast::Q6K => q6k(row, x),
        Fast::Q8_0 => q80(row, x),
        _ => unreachable!("cpu_q8 handles Q4_K, Q5_K, Q6_K and Q8_0"),
    }
}

/// The baseline build of the kernel (any CPU).
pub fn row_dot_baseline(t: Fast, row: &[u8], x: &Q8K) -> f32 {
    body(t, row, x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_dot_avx2(t: Fast, row: &[u8], x: &Q8K) -> f32 {
    body(t, row, x)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn row_dot_dotprod(t: Fast, row: &[u8], x: &Q8K) -> f32 {
    body(t, row, x)
}

/// Which build of the kernel this CPU runs, checked once.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Path {
    Baseline,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    DotProd,
}

pub fn path() -> Path {
    static P: std::sync::OnceLock<Path> = std::sync::OnceLock::new();
    *P.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return Path::Avx2;
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return Path::DotProd;
        }
        Path::Baseline
    })
}

/// row . x for a Q4_K or Q6_K weight row, with x quantized by `Q8K::quantize`.
#[inline]
pub fn row_dot(t: Fast, row: &[u8], x: &Q8K) -> f32 {
    match path() {
        Path::Baseline => body(t, row, x),
        // SAFETY: path() returns these only when the CPU has the features
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => unsafe { row_dot_avx2(t, row, x) },
        #[cfg(target_arch = "aarch64")]
        Path::DotProd => unsafe { row_dot_dotprod(t, row, x) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    /// Several inputs at once give exactly what one at a time gives (each input's sums in the same order).
    const TYPES: [&str; 4] = ["Q4_K", "Q5_K", "Q6_K", "Q8_0"];

    /// `blocks` x 256 values of random bytes with sane f16 scales where each type keeps them.
    fn random_row(ty: &str, blocks: usize, seed: &mut u64, scale: f32) -> Vec<u8> {
        let (per256, bb, at): (usize, usize, &[usize]) = match ty {
            "Q4_K" => (1, 144, &[0, 2]),
            "Q5_K" => (1, 176, &[0, 2]),
            "Q6_K" => (1, 210, &[208]),
            "Q8_0" => (8, 34, &[0]),
            _ => unreachable!(),
        };
        let mut row: Vec<u8> = (0..blocks * per256 * bb).map(|_| rng(seed) as u8).collect();
        for b in 0..blocks * per256 {
            for (k, &a) in at.iter().enumerate() {
                let v = scale / (k + 1) as f32;
                row[b * bb + a..b * bb + a + 2].copy_from_slice(&strata_pack::num::f32_to_f16(v).to_le_bytes());
            }
        }
        row
    }

    #[test]
    fn many_inputs_equal_one_at_a_time_exactly() {
        let mut seed = 0x0DDB_A11C_0FFE_E123u64;
        for ty in TYPES {
            let t = Fast::of(ty).unwrap();
            let blocks = 3;
            let row = random_row(ty, blocks, &mut seed, 0.02);
            let xs: Vec<Q8K> = (0..7).map(|_| {
                let x: Vec<f32> = (0..blocks * 256).map(|_| (rng(&mut seed) % 4001) as f32 / 1000.0 - 2.0).collect();
                Q8K::quantize_for(t, &x).unwrap()
            }).collect();
            let mut many = vec![0f32; xs.len()];
            row_dot_many(t, &row, &xs, &mut many);
            for (i, x) in xs.iter().enumerate() {
                assert_eq!(many[i].to_bits(), row_dot(t, &row, x).to_bits(), "{ty} input {i}");
            }
        }
    }

    /// The quantized activations are exactly Vdt::Q8K's (q * d equals round_trip's values).
    #[test]
    fn quantize_matches_the_f32_paths_rounding() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        let mut x: Vec<f32> = (0..512).map(|_| (rng(&mut seed) % 2001) as f32 / 100.0 - 10.0).collect();
        x[300..400].iter_mut().for_each(|v| *v = 0.0);
        x[256..512].iter_mut().for_each(|v| *v = 0.0); // a whole zero block
        let q = Q8K::quantize(&x).unwrap();
        let want = crate::model::Vdt::Q8K.round_trip(&x);
        for (i, w) in want.iter().enumerate() {
            assert_eq!(q.q[i] as f32 * q.d[i / 256], *w, "value {i}");
        }
        assert!(Q8K::quantize(&x[..100]).is_none());
        // and Q8_0's rule, a scale per 32 (the zero blocks included)
        let q = Q8K::quantize_q8_0(&x).unwrap();
        let want = crate::model::Vdt::Q8_0.round_trip(&x);
        for (i, w) in want.iter().enumerate() {
            assert_eq!(q.q[i] as f32 * q.d[i / 32], *w, "Q8_0 value {i}");
        }
        assert!(Q8K::quantize_q8_0(&x[..100]).is_none());
    }

    /// Every build of the kernel this CPU can run equals the f32 path (row_dot over the same rounded activations),
    /// within float rounding, on random blocks.
    #[test]
    fn integer_kernels_equal_the_f32_path() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for ty in TYPES {
            let t = Fast::of(ty).unwrap();
            for _ in 0..50 {
                let blocks = 4;
                let row = random_row(ty, blocks, &mut seed, 0.013);
                let x: Vec<f32> = (0..blocks * 256).map(|_| (rng(&mut seed) % 4001) as f32 / 1000.0 - 2.0).collect();
                let xq = Q8K::quantize_for(t, &x).unwrap();
                let rounded = crate::model::Vdt::of(ty).round_trip(&x);
                let want = crate::cpu_dot::row_dot(t, &row, &rounded) as f64;
                let mut deq = vec![0f32; blocks * 256];
                strata_pack::quants::dequantize_into(ty, &row, &mut deq).unwrap();
                let mag: f64 = deq.iter().zip(&rounded).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
                let tol = 1e-5 * mag.max(1e-6);
                let got = row_dot_baseline(t, &row, &xq) as f64;
                assert!((got - want).abs() <= tol, "{ty} baseline: {got} vs {want} (tol {tol})");
                let got = row_dot(t, &row, &xq) as f64;
                assert!((got - want).abs() <= tol, "{ty} {:?}: {got} vs {want} (tol {tol})", path());
            }
        }
    }
}
