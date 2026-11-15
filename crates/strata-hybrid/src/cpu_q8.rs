//! Integer row products for the K-quant weights (Q4_K, Q6_K) against activations quantized to Q8_K, the way ggml's CPU
//! backend computes them (ggml_vec_dot_q4_K_q8_K, ggml_vec_dot_q6_K_q8_K).
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
}

/// Whether this weight type has an integer kernel here.
pub fn handles(t: Fast) -> bool {
    matches!(t, Fast::Q4K | Fast::Q6K)
}

#[inline(always)]
fn h(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

/// Σ a[i] * b[i] over 32 values, exact in i32 (|a| < 64, |b| <= 127).
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

#[inline(always)]
fn body(t: Fast, row: &[u8], x: &Q8K) -> f32 {
    match t {
        Fast::Q4K => q4k(row, x),
        Fast::Q6K => q6k(row, x),
        _ => unreachable!("cpu_q8 handles Q4_K and Q6_K"),
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
    }

    /// Every build of the kernel this CPU can run equals the f32 path (row_dot over the same rounded activations),
    /// within float rounding, on random blocks.
    #[test]
    fn integer_kernels_equal_the_f32_path() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for (ty, bb) in [("Q4_K", 144usize), ("Q6_K", 210usize)] {
            let t = Fast::of(ty).unwrap();
            for _ in 0..50 {
                let blocks = 4;
                let mut row: Vec<u8> = (0..blocks * bb).map(|_| rng(&mut seed) as u8).collect();
                for b in 0..blocks {
                    let o = b * bb;
                    let mut set = |at: usize, v: f32| row[o + at..o + at + 2].copy_from_slice(&strata_pack::num::f32_to_f16(v).to_le_bytes());
                    if ty == "Q4_K" { set(0, 0.013); set(2, 0.007); } else { set(208, 0.011); }
                }
                let x: Vec<f32> = (0..blocks * 256).map(|_| (rng(&mut seed) % 4001) as f32 / 1000.0 - 2.0).collect();
                let xq = Q8K::quantize(&x).unwrap();
                let rounded = crate::model::Vdt::Q8K.round_trip(&x);
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
