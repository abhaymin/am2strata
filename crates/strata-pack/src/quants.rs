//! ggml block dequantization, value for value as gguf-py's numpy dequantizers compute it (the element order of their
//! reshapes and the float32 operation order: d·scale first, then ·q, then − min), which iq_pack.py --compat-bf16 uses.
//! tests/quants_golden.rs compares every type with gguf-py on recorded blocks.

use crate::grids::*;
use crate::num::f16_to_f32;

/// The encodings this module dequantizes.
pub const SUPPORTED: [&str; 16] = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_XXS",
                                   "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS"];

fn h(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}
fn u32le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}
fn sign(bit: u8) -> f32 {
    if bit == 0 { 1.0 } else { -1.0 }
}

/// The values of `data` (whole blocks of `type_name`) as f32.
pub fn dequantize(type_name: &str, data: &[u8]) -> Result<Vec<f32>, String> {
    let (be, bb) = strata_gguf::block_geometry(type_name).ok_or_else(|| format!("unknown encoding {type_name}"))?;
    let (be, bb) = (be as usize, bb as usize);
    if data.len() % bb != 0 {
        return Err(format!("{} bytes are not whole {type_name} blocks", data.len()));
    }
    let mut out = vec![0f32; data.len() / bb * be];
    for (b, y) in data.chunks_exact(bb).zip(out.chunks_exact_mut(be)) {
        block(type_name, b, y)?;
    }
    Ok(out)
}

fn block(t: &str, b: &[u8], y: &mut [f32]) -> Result<(), String> {
    match t {
        "Q4_0" => {
            let d = h(b, 0);
            for j in 0..16 {
                y[j] = d * ((b[2 + j] & 0x0f) as i8 - 8) as f32;
                y[16 + j] = d * ((b[2 + j] >> 4) as i8 - 8) as f32;
            }
        }
        "Q4_1" => {
            let (d, m) = (h(b, 0), h(b, 2));
            for j in 0..16 {
                y[j] = d * (b[4 + j] & 0x0f) as f32 + m;
                y[16 + j] = d * (b[4 + j] >> 4) as f32 + m;
            }
        }
        "Q5_0" | "Q5_1" => {
            let q1 = t == "Q5_1";
            let d = h(b, 0);
            let m = if q1 { h(b, 2) } else { 0.0 };
            let o = if q1 { 4 } else { 2 };
            let qh = u32le(b, o);
            let qs = &b[o + 4..];
            for i in 0..32 {
                let ql = if i < 16 { qs[i] & 0x0f } else { qs[i - 16] >> 4 };
                let q = ql | (((qh >> i) & 1) as u8) << 4;
                y[i] = if q1 { d * q as f32 + m } else { d * (q as i8 - 16) as f32 };
            }
        }
        "Q8_0" => {
            let d = h(b, 0);
            for j in 0..32 {
                y[j] = (b[2 + j] as i8) as f32 * d;
            }
        }
        "Q2_K" => {
            let (sc, qs) = (&b[0..16], &b[16..80]);
            let (d, dmin) = (h(b, 80), h(b, 82));
            for hh in 0..2 {
                for s in 0..4 {
                    for l in 0..32 {
                        let flat = (hh * 4 + s) * 32 + l;
                        let g = flat / 16;
                        let q = ((qs[hh * 32 + l] >> (2 * s)) & 3) as f32;
                        let dl = d * (sc[g] & 0x0f) as f32;
                        let ml = dmin * (sc[g] >> 4) as f32;
                        y[flat] = dl * q - ml;
                    }
                }
            }
        }
        "Q3_K" => {
            let (hm, qs, sc) = (&b[0..32], &b[32..96], &b[96..108]);
            let d = h(b, 108);
            let mut dl = [0f32; 16];
            for (k, v) in dl.iter_mut().enumerate() {
                let ls = (sc[k % 8] >> (4 * (k / 8))) & 0x0f;
                let hs = (sc[8 + k % 4] >> (2 * (k / 4))) & 0x03;
                *v = d * ((ls | (hs << 4)) as i8 - 32) as f32;
            }
            for hh in 0..2 {
                for s in 0..4 {
                    for l in 0..32 {
                        let flat = (hh * 4 + s) * 32 + l;
                        let ql = (qs[hh * 32 + l] >> (2 * s)) & 3;
                        let qhb = ((hm[flat % 32] >> (flat / 32)) & 1) ^ 1;
                        let q = (ql as i8 - (qhb << 2) as i8) as f32;
                        y[flat] = dl[flat / 16] * q;
                    }
                }
            }
        }
        "Q4_K" | "Q5_K" => {
            let (d, dmin) = (h(b, 0), h(b, 2));
            let s = &b[4..16];
            let mut dd = [0f32; 8];
            let mut dm = [0f32; 8];
            for j in 0..4 {
                let (sc_lo, mn_lo) = (s[j] & 0x3f, s[4 + j] & 0x3f);
                let sc_hi = (s[8 + j] & 0x0f) | ((s[j] >> 2) & 0x30);
                let mn_hi = (s[8 + j] >> 4) | ((s[4 + j] >> 2) & 0x30);
                dd[j] = d * sc_lo as f32;
                dd[4 + j] = d * sc_hi as f32;
                dm[j] = dmin * mn_lo as f32;
                dm[4 + j] = dmin * mn_hi as f32;
            }
            let five = t == "Q5_K";
            let (qh, qs) = if five { (&b[16..48], &b[48..176]) } else { (&b[0..0], &b[16..144]) };
            for c in 0..4 {
                for sh in 0..2 {
                    let g = c * 2 + sh;
                    for l in 0..32 {
                        let mut q = (qs[c * 32 + l] >> (4 * sh)) & 0x0f;
                        if five {
                            q |= ((qh[l] >> g) & 1) << 4;
                        }
                        y[g * 32 + l] = dd[g] * q as f32 - dm[g];
                    }
                }
            }
        }
        "Q6_K" => {
            let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
            let d = h(b, 208);
            let mut dsc = [0f32; 16];
            for (i, v) in dsc.iter_mut().enumerate() {
                *v = d * (sc[i] as i8) as f32;
            }
            for hh in 0..2 {
                for k in 0..4 {
                    let g = hh * 4 + k;
                    for l in 0..32 {
                        let lo = (ql[hh * 64 + (k % 2) * 32 + l] >> (4 * (k / 2))) & 0x0f;
                        let hi = (qh[hh * 32 + l] >> (2 * k)) & 0x03;
                        let q = ((lo | (hi << 4)) as i8 - 32) as f32;
                        let e = g * 32 + l;
                        y[e] = dsc[e / 16] * q;
                    }
                }
            }
        }
        "IQ2_XXS" => {
            let d = h(b, 0);
            for ib in 0..8 {
                let a0 = &b[2 + 8 * ib..2 + 8 * ib + 4];
                let a1 = u32le(b, 2 + 8 * ib + 4);
                let db = d * (0.5f32 + (a1 >> 28) as f32) * 0.25f32;
                for l in 0..4 {
                    let ks = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 0x7f) as usize];
                    let grid = IQ2XXS_GRID[a0[l] as usize].to_le_bytes();
                    for j in 0..8 {
                        y[ib * 32 + l * 8 + j] = db * grid[j] as f32 * sign((ks >> j) & 1);
                    }
                }
            }
        }
        "IQ2_XS" => {
            let d = h(b, 0);
            let sc = &b[66..74];
            for i in 0..32 {
                let k = i / 2;
                let s = (sc[k / 2] >> (4 * (k % 2))) & 0x0f;
                let db = d * (0.5f32 + s as f32) * 0.25f32;
                let q = u16::from_le_bytes([b[2 + 2 * i], b[3 + 2 * i]]);
                let ks = KSIGNS_IQ2XS[(q >> 9) as usize];
                let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
                for j in 0..8 {
                    y[i * 8 + j] = db * grid[j] as f32 * sign((ks >> j) & 1);
                }
            }
        }
        "IQ3_XXS" => {
            let d = h(b, 0);
            let qs = &b[2..66];
            for ib in 0..8 {
                let sc = u32le(b, 66 + 4 * ib);
                let db = d * (0.5f32 + (sc >> 28) as f32) * 0.5f32;
                for l in 0..4 {
                    let ks = KSIGNS_IQ2XS[((sc >> (7 * l)) & 0x7f) as usize];
                    for j in 0..8 {
                        let flat = ib * 32 + l * 8 + j;
                        let g = IQ3XXS_GRID[qs[flat / 4] as usize].to_le_bytes()[flat % 4];
                        y[flat] = db * g as f32 * sign((ks >> j) & 1);
                    }
                }
            }
        }
        "IQ3_S" => {
            let d = h(b, 0);
            let (qs, qh, signs, sc) = (&b[2..66], &b[66..74], &b[74..106], &b[106..110]);
            for k in 0..8 {
                let s = (sc[k / 2] >> (4 * (k % 2))) & 0x0f;
                let db = d * (1u8 + 2 * s) as f32;
                for l in 0..4 {
                    let sb = signs[k * 4 + l];
                    for j in 0..8 {
                        let flat = k * 32 + l * 8 + j;
                        let i = flat / 4;
                        let idx = qs[i] as usize | ((((qh[i / 8] >> (i % 8)) & 1) as usize) << 8);
                        let g = IQ3S_GRID[idx].to_le_bytes()[flat % 4];
                        y[flat] = db * g as f32 * sign((sb >> j) & 1);
                    }
                }
            }
        }
        "IQ4_NL" => {
            let d = h(b, 0);
            for j in 0..16 {
                y[j] = d * KVALUES_IQ4NL[(b[2 + j] & 0x0f) as usize] as f32;
                y[16 + j] = d * KVALUES_IQ4NL[(b[2 + j] >> 4) as usize] as f32;
            }
        }
        "IQ4_XS" => {
            let d = h(b, 0);
            let sh = u16::from_le_bytes([b[2], b[3]]);
            let (sl, qs) = (&b[4..8], &b[8..136]);
            for k in 0..8 {
                let lo = (sl[k / 2] >> (4 * (k % 2))) & 0x0f;
                let hi = ((sh >> (2 * k)) as u8) & 0x03;
                let dl = d * ((lo | (hi << 4)) as i8 - 32) as f32;
                for s in 0..2 {
                    for l in 0..16 {
                        let q = (qs[k * 16 + l] >> (4 * s)) & 0x0f;
                        y[k * 32 + s * 16 + l] = dl * KVALUES_IQ4NL[q as usize] as f32;
                    }
                }
            }
        }
        _ => return Err(format!("dequantizing {t} is not supported (supported: {})", SUPPORTED.join(", "))),
    }
    Ok(())
}
