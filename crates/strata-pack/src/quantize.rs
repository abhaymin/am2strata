//! ggml reference quantizers (Q8_0, Q4_0, Q4_K, Q5_K, Q6_K): the inverse of `quants::dequantize`, packing fixed-width
//! f32 weights into the byte layouts strata-gguf's `block_geometry` reports. This is a faithful port of llama.cpp's
//! `ggml-quants.c` reference quantizers (commit 20d839db, `quantize_row_q8_0_ref`, `quantize_row_q4_0_ref`,
//! `quantize_row_q4_K_ref`, `quantize_row_q5_K_ref`, `quantize_row_q6_K_ref`, and the shared helpers
//! `make_qkx2_quants` / `make_qx_quants`) as written there; the block layouts are byte for byte those blocks, so with
//! no quantization weights the output is bit-identical to llama.cpp's `llama-quantize`. ggml is MIT:
//!
//!   Copyright (c) 2023-2026 The ggml authors
//!   MIT License (see .agent/ref/ggml/LICENSE-llama.cpp and the project NOTICE).

// The names here (L, Laux, Q8_0) are ggml's, carried over from ggml-quants.c.
#![allow(non_snake_case)]

use crate::num::{f16_to_f32, f32_to_f16};

// block sizes from ggml-common.h
const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12; // blocks of 16 within a super-block, packed into 12 bytes
const GROUP_MAX_EPS: f32 = 1e-15f32;

/// ggml's reference `nearest_int`: add 2^22 and truncate (round-half-to-even via the IEEE round mode).
#[inline]
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12_582_912.0_f32;
    let i = val.to_bits() as i32;
    (i & 0x007F_FFFF) - 0x0040_0000
}

#[inline]
fn mnI(a: i32, b: i32) -> i32 { if a < b { a } else { b } }
#[inline]
fn mxI(a: i32, b: i32) -> i32 { if a > b { a } else { b } }

/// ggml's `make_qx_quants` (a single-group best-scale search; weights optional), as written in ggml-quants.c.
fn make_qx_quants(n: usize, nmax: i32, x: &[f32], L: &mut [i8], mut rmse_type: i32, qw: Option<&[f32]>) -> f32 {
    let (mut max, mut amax) = (0.0_f32, 0.0_f32);
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax { amax = ax; max = x[i]; }
    }
    if amax < GROUP_MAX_EPS {
        L.iter_mut().for_each(|l| *l = 0);
        return 0.0_f32;
    }
    let mut iscale = -nmax as f32 / max;
    if rmse_type == 0 {
        for i in 0..n {
            let l = nearest_int(iscale * x[i]);
            L[i] = (nmax + mxI(-nmax, mnI(nmax - 1, l))) as i8;
        }
        return 1.0 / iscale;
    }
    let mut return_early = false;
    if rmse_type < 0 { rmse_type = -rmse_type; return_early = true; }
    let weight = |i: usize| -> f32 {
        match qw {
            Some(qw) => qw[i],
            None => if rmse_type == 1 { x[i] * x[i] } else if rmse_type == 2 { 1.0 } else if rmse_type == 3 { x[i].abs() } else { x[i].sqrt() },
        }
    };
    let (mut sumlx, mut suml2) = (0.0_f32, 0.0_f32);
    for i in 0..n {
        let l = nearest_int(iscale * x[i]);
        let l = mnI(nmax - 1, mxI(-nmax, l));
        L[i] = (l + nmax) as i8;
        let w = weight(i);
        sumlx += w * x[i] * l as f32;
        suml2 += w * (l as f32) * (l as f32);
    }
    let mut scale = if suml2 > 0.0 { sumlx / suml2 } else { 0.0_f32 };
    if return_early { return if suml2 > 0.0 { 0.5 * (scale + 1.0 / iscale) } else { 1.0 / iscale }; }
    let mut best = scale * sumlx;
    for is in -9..=9 {
        if is == 0 { continue; }
        iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        sumlx = 0.0; suml2 = 0.0;
        for i in 0..n {
            let l = nearest_int(iscale * x[i]);
            let l = mnI(nmax - 1, mxI(-nmax, l));
            let w = weight(i);
            sumlx += w * x[i] * l as f32;
            suml2 += w * (l as f32) * (l as f32);
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let l = nearest_int(iscale * x[i]);
                L[i] = (nmax + mxI(-nmax, mnI(nmax - 1, l))) as i8;
            }
            scale = sumlx / suml2; best = scale * sumlx;
        }
    }
    scale
}

/// ggml's `make_qkx2_quants` (per-16-element grouped scale+min search), as written in ggml-quants.c.
fn make_qkx2_quants(n: usize, nmax: i32, x: &[f32], weights: &[f32], L: &mut [u8], the_min: &mut f32, Laux: &mut [u8],
                    rmin: f32, rdelta: f32, nstep: i32, use_mad: bool) -> f32 {
    let (mut min, mut max) = (x[0], x[0]);
    let (mut sum_w, mut sum_x) = (weights[0], weights[0] * x[0]);
    for i in 1..n {
        if x[i] < min { min = x[i]; }
        if x[i] > max { max = x[i]; }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 { min = 0.0; }
    if max == min {
        L.iter_mut().for_each(|l| *l = 0);
        *the_min = -min;
        return 0.0_f32;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0_f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min));
        L[i] = mxI(0, mnI(nmax, l)) as u8;
        let diff = scale * (L[i] as f32) + min - x[i];
        let diff = if use_mad { diff.abs() } else { diff * diff };
        best_error += weights[i] * diff;
    }
    if nstep < 1 { *the_min = -min; return scale; }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0_f32, 0.0_f32, 0.0_f32);
        for i in 0..n {
            let l = mxI(0, mnI(nmax, nearest_int(iscale * (x[i] - min))));
            Laux[i] = l as u8;
            let w = weights[i];
            sum_l += w * l as f32;
            sum_l2 += w * (l as f32) * (l as f32);
            sum_xl += w * (l as f32) * x[i];
        }
        let D = sum_w * sum_l2 - sum_l * sum_l;
        if D > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if this_min > 0.0 { this_min = 0.0; this_scale = sum_xl / sum_l2; }
            let mut cur_error = 0.0_f32;
            for i in 0..n {
                let diff = this_scale * (Laux[i] as f32) + this_min - x[i];
                let diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error += weights[i] * diff;
            }
            if cur_error < best_error {
                for i in 0..n { L[i] = Laux[i]; }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// ggml's `get_scale_min_k4`, decode one of the eight super-block scales/mins from its 12 packed bytes.
fn get_scale_min_k4(j: usize, q: &[u8], d: &mut u8, m: &mut u8) {
    if j < 4 { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); } // q[j-0] == q[j]
}

/// ggml reference Q8_0 as `quantize_row_q8_0_ref`: d = amax/127, q = round(x/d).
pub fn q8_0(x: &[f32], out: &mut Vec<u8>) {
    debug_assert!(x.len() % 32 == 0);
    for b in x.chunks_exact(32) {
        let amax = b.iter().fold(0.0_f32, |m, v| if v.abs() > m { v.abs() } else { m });
        let d = amax / 127.0_f32;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for j in 0..32 { out.push((b[j] * id).round() as i8 as u8); }
    }
}

/// ggml reference Q4_0 as `quantize_row_q4_0_ref`: d = (signed max)/-8, code = min(15, trunc(x/d + 8.5)).
pub fn q4_0(x: &[f32], out: &mut Vec<u8>) {
    debug_assert!(x.len() % 32 == 0);
    for b in x.chunks_exact(32) {
        let mut idx = 0usize;
        for (i, v) in b.iter().enumerate() { if v.abs() > b[idx].abs() { idx = i; } }
        let d = b[idx] / -8.0_f32;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for j in 0..16 {
            let xi0 = ((b[j] * id + 8.5_f32).trunc() as i8).min(15);
            let xi1 = ((b[16 + j] * id + 8.5_f32).trunc() as i8).min(15);
            out.push(xi0 as u8 | (xi1 as u8) << 4);
        }
    }
}

/// ggml reference Q4_K as `quantize_row_q4_K_ref`.
pub fn q4_K(x: &[f32], out: &mut Vec<u8>) {
    debug_assert!(x.len() % QK_K == 0);
    for b in x.chunks_exact(QK_K) {
        let mut L = [0u8; QK_K];
        let mut Laux = [0u8; 32];
        let mut weights = [0.0_f32; 32];
        let (mut mins, mut scales) = ([0.0_f32; 8], [0.0_f32; 8]);
        let (mut max_scale, mut max_min) = (0.0_f32, 0.0_f32);

        for j in 0..8 {
            let mut sum_x2 = 0.0_f32;
            for l in 0..32 { sum_x2 += b[32 * j + l] * b[32 * j + l]; }
            let av_x = (sum_x2 / 32.0).sqrt();
            for l in 0..32 { weights[l] = av_x + b[32 * j + l].abs(); }
            scales[j] = make_qkx2_quants(32, 15, &b[32 * j..32 * j + 32], &weights[..], &mut L[32 * j..32 * j + 32],
                                         &mut mins[j], &mut Laux[..], -1.0_f32, 0.1_f32, 20, false);
            let s = scales[j]; if s > max_scale { max_scale = s; }
            let mn = mins[j]; if mn > max_min { max_min = mn; }
        }

        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        let mut sc = [0u8; K_SCALE_SIZE];
        for j in 0..8 {
            let ls = mnI(63, nearest_int(inv_scale * scales[j]));
            let lm = mnI(63, nearest_int(inv_min * mins[j]));
            if j < 4 { sc[j] = ls as u8; sc[j + 4] = lm as u8; }
            else { sc[j + 4] = (ls & 0xF) as u8 | ((lm & 0xF) as u8) << 4; sc[j - 4] |= ((ls >> 4) as u8) << 6; sc[j] |= ((lm >> 4) as u8) << 6; }
        }

        let d16 = f32_to_f16(max_scale / 63.0_f32);
        let dmin16 = f32_to_f16(max_min / 63.0_f32);
        out.extend_from_slice(&d16.to_le_bytes());
        out.extend_from_slice(&dmin16.to_le_bytes());
        out.extend_from_slice(&sc);

        let dd = f16_to_f32(d16);
        let dm = f16_to_f32(dmin16);
        for j in 0..8 {
            let (mut sct, mut mt) = (0u8, 0u8);
            get_scale_min_k4(j, &sc, &mut sct, &mut mt);
            let d = dd * (sct as f32);
            if d == 0.0 { continue; }
            let dmj = dm * (mt as f32);
            for ii in 0..32 {
                let l = mxI(0, mnI(15, nearest_int((b[32 * j + ii] + dmj) / d)));
                L[32 * j + ii] = l as u8;
            }
        }

        for j in (0..QK_K).step_by(64) {
            for l in 0..32 { out.push(L[j + l] | (L[j + l + 32] << 4)); }
        }
    }
}

/// ggml reference Q5_K as `quantize_row_q5_K_ref`.
pub fn q5_K(x: &[f32], out: &mut Vec<u8>) {
    debug_assert!(x.len() % QK_K == 0);
    for b in x.chunks_exact(QK_K) {
        let mut L = [0u8; QK_K];
        let mut Laux = [0u8; 32];
        let mut weights = [0.0_f32; 32];
        let (mut mins, mut scales) = ([0.0_f32; 8], [0.0_f32; 8]);
        let (mut max_scale, mut max_min) = (0.0_f32, 0.0_f32);

        for j in 0..8 {
            let mut sum_x2 = 0.0_f32;
            for l in 0..32 { sum_x2 += b[32 * j + l] * b[32 * j + l]; }
            let av_x = (sum_x2 / 32.0).sqrt();
            for l in 0..32 { weights[l] = av_x + b[32 * j + l].abs(); }
            scales[j] = make_qkx2_quants(32, 31, &b[32 * j..32 * j + 32], &weights[..], &mut L[32 * j..32 * j + 32],
                                         &mut mins[j], &mut Laux[..], -0.5_f32, 0.1_f32, 15, false);
            let s = scales[j]; if s > max_scale { max_scale = s; }
            let mn = mins[j]; if mn > max_min { max_min = mn; }
        }

        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        let mut sc = [0u8; K_SCALE_SIZE];
        for j in 0..8 {
            let ls = mnI(63, nearest_int(inv_scale * scales[j]));
            let lm = mnI(63, nearest_int(inv_min * mins[j]));
            if j < 4 { sc[j] = ls as u8; sc[j + 4] = lm as u8; }
            else { sc[j + 4] = (ls & 0xF) as u8 | ((lm & 0xF) as u8) << 4; sc[j - 4] |= ((ls >> 4) as u8) << 6; sc[j] |= ((lm >> 4) as u8) << 6; }
        }

        let d16 = f32_to_f16(max_scale / 63.0_f32);
        let dmin16 = f32_to_f16(max_min / 63.0_f32);
        out.extend_from_slice(&d16.to_le_bytes());
        out.extend_from_slice(&dmin16.to_le_bytes());
        out.extend_from_slice(&sc);

        let dd = f16_to_f32(d16);
        let dm = f16_to_f32(dmin16);
        for j in 0..8 {
            let (mut sct, mut mt) = (0u8, 0u8);
            get_scale_min_k4(j, &sc, &mut sct, &mut mt);
            let d = dd * (sct as f32);
            if d == 0.0 { continue; }
            let dmj = dm * (mt as f32);
            for ii in 0..32 {
                let l = mxI(0, mnI(31, nearest_int((b[32 * j + ii] + dmj) / d)));
                L[32 * j + ii] = l as u8;
            }
        }

        let mut qh = [0u8; QK_K / 8];
        let mut qlbuf = [0u8; QK_K / 2];
        let (mut m1, mut m2) = (1u8, 2u8);
        for g in 0..4 {
            let n = g * 64;
            for j in 0..32 {
                let mut l1 = L[n + j];
                if l1 > 15 { l1 -= 16; qh[j] |= m1; }
                let mut l2 = L[n + j + 32];
                if l2 > 15 { l2 -= 16; qh[j] |= m2; }
                qlbuf[g * 32 + j] = l1 | (l2 << 4);
            }
            m1 <<= 2; m2 <<= 2;
        }

        out.extend_from_slice(&qh);
        out.extend_from_slice(&qlbuf);
    }
}

/// ggml reference Q6_K as `quantize_row_q6_K_ref`.
pub fn q6_K(x: &[f32], out: &mut Vec<u8>) {
    debug_assert!(x.len() % QK_K == 0);
    for b in x.chunks_exact(QK_K) {
        let mut L = [0i8; QK_K];
        let (mut sc_f, mut scales) = ([0.0_f32; QK_K / 16], [0u8; QK_K / 16]);
        let (mut max_scale, mut max_abs_scale) = (0.0_f32, 0.0_f32);

        for ib in 0..(QK_K / 16) {
            let scale = make_qx_quants(16, 32, &b[16 * ib..16 * ib + 16], &mut L[16 * ib..16 * ib + 16], 1, None);
            sc_f[ib] = scale;
            let abs_scale = scale.abs();
            if abs_scale > max_abs_scale { max_abs_scale = abs_scale; max_scale = scale; }
        }

        if max_abs_scale < GROUP_MAX_EPS {
            out.extend(std::iter::repeat(0u8).take(2 + 64 + 128)); // memset(block,0) then d=fp16(0) == all zero anyway
            return;
        }

        let iscale = -128.0_f32 / max_scale;
        let d16 = f32_to_f16(1.0 / iscale);
        for ib in 0..(QK_K / 16) { scales[ib] = mnI(127, nearest_int(iscale * sc_f[ib])) as i8 as u8; }

        // ggml's block_q6_K layout: ql, qh, then scales, then d last
        let dd = f16_to_f32(d16);
        for j in 0..(QK_K / 16) {
            let d = dd * (scales[j] as i8 as f32);
            if d == 0.0 { continue; }
            for ii in 0..16 {
                let l = mnI(31, mxI(-32, nearest_int(b[16 * j + ii] / d)));
                L[16 * j + ii] = (l + 32) as i8;
            }
        }

        let mut qlbuf = [0u8; QK_K / 2];
        let mut qhbuf = [0u8; QK_K / 4];
        for gi in 0..2 {
            let j = gi * 128;
            for l in 0..32 {
                let q1 = (L[j + l] as i32) & 0xF;
                let q2 = (L[j + l + 32] as i32) & 0xF;
                let q3 = (L[j + l + 64] as i32) & 0xF;
                let q4 = (L[j + l + 96] as i32) & 0xF;
                qlbuf[gi * 64 + l] = q1 as u8 | (q3 as u8) << 4;
                qlbuf[gi * 64 + l + 32] = q2 as u8 | (q4 as u8) << 4;
                let hi = ((L[j + l] as i32) >> 4) | (((L[j + l + 32] as i32) >> 4) << 2) | (((L[j + l + 64] as i32) >> 4) << 4) | (((L[j + l + 96] as i32) >> 4) << 6);
                qhbuf[gi * 32 + l] = hi as u8;
            }
        }

        out.extend_from_slice(&qlbuf);
        out.extend_from_slice(&qhbuf);
        out.extend_from_slice(&scales);
        out.extend_from_slice(&d16.to_le_bytes());
    }
}

/// Quantize `x` (a whole number of the encoding's blocks) to `out`. Q8_0 / Q4_0 are bit-identical to ggml's reference;
/// with no quantization weights the K encodings match `llama-quantize` byte for byte.
pub fn quantize(kind: &str, x: &[f32], out: &mut Vec<u8>) -> Result<(), String> {
    match kind {
        "Q8_0" => q8_0(x, out),
        "Q4_0" => q4_0(x, out),
        "Q4_K" => q4_K(x, out),
        "Q5_K" => q5_K(x, out),
        "Q6_K" => q6_K(x, out),
        _ => return Err(format!("unsupported quantization {kind} (expected Q8_0 Q4_0 Q4_K Q5_K Q6_K)")),
    }
    Ok(())
}
