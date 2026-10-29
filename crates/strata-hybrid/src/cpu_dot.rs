//! The CPU's row products without a row buffer: a weight row's blocks multiplied with x as they are read, the type
//! chosen once per matrix (strata_pack::quants::dequantize_into matches on the type's name for every block and
//! writes the whole row out first). The values are the dequantizer's - d * scale * q - min, per element - so the
//! products are the same; only the order of the sums differs, as on the GPU.

use strata_pack::num::f16_to_f32;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Fast {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

impl Fast {
    pub fn of(ty: &str) -> Option<Fast> {
        Some(match ty {
            "F32" => Fast::F32,
            "F16" => Fast::F16,
            "BF16" => Fast::Bf16,
            "Q8_0" => Fast::Q8_0,
            "Q4_K" => Fast::Q4K,
            "Q5_K" => Fast::Q5K,
            "Q6_K" => Fast::Q6K,
            _ => return None,
        })
    }
}

#[inline]
fn h(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

/// Eight running sums (vectorizes; the same as model::dot's).
#[inline]
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    for (ca, cb) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for j in 0..8 {
            acc[j] += ca[j] * cb[j];
        }
    }
    acc.iter().sum()
}

/// row . x for one weight row of type `t`.
pub fn row_dot(t: Fast, row: &[u8], x: &[f32]) -> f32 {
    let mut s = 0f32;
    let mut q = [0f32; 32];
    match t {
        Fast::F32 => {
            for (c, xs) in row.chunks_exact(128).zip(x.chunks_exact(32)) {
                for j in 0..32 {
                    q[j] = f32::from_le_bytes([c[4 * j], c[4 * j + 1], c[4 * j + 2], c[4 * j + 3]]);
                }
                s += dot8(&q, xs);
            }
        }
        Fast::F16 => {
            for (c, xs) in row.chunks_exact(64).zip(x.chunks_exact(32)) {
                for j in 0..32 {
                    q[j] = h(c, 2 * j);
                }
                s += dot8(&q, xs);
            }
        }
        Fast::Bf16 => {
            for (c, xs) in row.chunks_exact(64).zip(x.chunks_exact(32)) {
                for j in 0..32 {
                    q[j] = f32::from_bits((u16::from_le_bytes([c[2 * j], c[2 * j + 1]]) as u32) << 16);
                }
                s += dot8(&q, xs);
            }
        }
        Fast::Q8_0 => {
            for (b, xs) in row.chunks_exact(34).zip(x.chunks_exact(32)) {
                let d = h(b, 0);
                for j in 0..32 {
                    q[j] = (b[2 + j] as i8) as f32 * d;
                }
                s += dot8(&q, xs);
            }
        }
        Fast::Q4K | Fast::Q5K => {
            let five = t == Fast::Q5K;
            let bb = if five { 176 } else { 144 };
            for (b, xs) in row.chunks_exact(bb).zip(x.chunks_exact(256)) {
                let (d, dmin) = (h(b, 0), h(b, 2));
                let sc = &b[4..16];
                let (qh, qs) = if five { (&b[16..48], &b[48..176]) } else { (&b[0..0], &b[16..144]) };
                for g in 0..8 {
                    let j = g % 4;
                    let (scale, min) = if g < 4 {
                        (sc[j] & 0x3f, sc[4 + j] & 0x3f)
                    } else {
                        ((sc[8 + j] & 0x0f) | ((sc[j] >> 2) & 0x30), (sc[8 + j] >> 4) | ((sc[4 + j] >> 2) & 0x30))
                    };
                    let (dd, dm) = (d * scale as f32, dmin * min as f32);
                    let (c, sh) = (g / 2, 4 * (g % 2));
                    for l in 0..32 {
                        let mut v = (qs[c * 32 + l] >> sh) & 0x0f;
                        if five {
                            v |= ((qh[l] >> g) & 1) << 4;
                        }
                        q[l] = dd * v as f32 - dm;
                    }
                    s += dot8(&q, &xs[g * 32..g * 32 + 32]);
                }
            }
        }
        Fast::Q6K => {
            for (b, xs) in row.chunks_exact(210).zip(x.chunks_exact(256)) {
                let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
                let d = h(b, 208);
                for hh in 0..2 {
                    for k in 0..4 {
                        let g = hh * 4 + k;
                        for l in 0..32 {
                            let lo = (ql[hh * 64 + (k % 2) * 32 + l] >> (4 * (k / 2))) & 0x0f;
                            let hi = (qh[hh * 32 + l] >> (2 * k)) & 0x03;
                            let e = g * 32 + l;
                            q[l] = d * (sc[e / 16] as i8) as f32 * ((lo | (hi << 4)) as i8 - 32) as f32;
                        }
                        s += dot8(&q, &xs[g * 32..g * 32 + 32]);
                    }
                }
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fused_products_are_the_dequantizers() {
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut next = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
        for (ty, bb, be) in [("Q8_0", 34, 32), ("Q4_K", 144, 256), ("Q5_K", 176, 256), ("Q6_K", 210, 256), ("F16", 64, 32), ("BF16", 64, 32)] {
            let blocks = 512 / be;
            let mut row: Vec<u8> = (0..blocks * bb).map(|_| next() as u8).collect();
            // valid f16 scales
            for b in 0..blocks {
                let o = b * bb;
                let fix = |row: &mut Vec<u8>, at: usize| row[at..at + 2].copy_from_slice(&strata_pack::num::f32_to_f16(0.01).to_le_bytes());
                match ty {
                    "Q8_0" => fix(&mut row, o),
                    "Q4_K" | "Q5_K" => { fix(&mut row, o); fix(&mut row, o + 2) }
                    "Q6_K" => fix(&mut row, o + 208),
                    _ => for j in 0..32 { fix(&mut row, o + 2 * j) },
                }
            }
            let x: Vec<f32> = (0..512).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
            let mut deq = vec![0f32; 512];
            strata_pack::quants::dequantize_into(ty, &row, &mut deq).unwrap();
            let want: f64 = deq.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
            let scale: f64 = deq.iter().zip(&x).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
            let got = row_dot(Fast::of(ty).unwrap(), &row, &x) as f64;
            assert!((got - want).abs() <= 1e-5 * scale.max(1e-9), "{ty}: {got} vs {want}");
        }
    }
}
