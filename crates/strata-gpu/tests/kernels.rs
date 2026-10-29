//! Every kernel against the CPU's dequantizer (strata_pack::quants), on every device here: the phase 2 gate
//! (docs/GPU-PLAN.md) - within 1e-3 of the sum of |w x|, so only the order of the sums differs.

use strata_gpu::{list, Gpu};

/// A tiny deterministic generator (no rand dependency).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 20001) as f32 / 10000.0 - 1.0
    }
}

fn f16_bits(v: f32) -> u16 {
    strata_pack::num::f32_to_f16(v)
}

/// `rows` rows of random blocks of `ty`, with valid f16 scales where the type has them.
fn weights(ty: &str, rows: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
    let (be, bb, _) = strata_gpu::kernels::geometry(ty).unwrap();
    let blocks = rows * cols / be as usize;
    let mut out = Vec::with_capacity(blocks * bb as usize);
    for _ in 0..blocks {
        let mut b: Vec<u8> = (0..bb).map(|_| rng.next() as u8).collect();
        let put = |b: &mut Vec<u8>, at: usize, v: f32| b[at..at + 2].copy_from_slice(&f16_bits(v).to_le_bytes());
        match ty {
            "F32" => for j in 0..32 { b[4 * j..4 * j + 4].copy_from_slice(&(rng.unit()).to_le_bytes()) },
            "F16" => for j in 0..32 { let v = rng.unit(); put(&mut b, 2 * j, v) },
            "BF16" => for j in 0..32 { let v = rng.unit(); b[2 * j..2 * j + 2].copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes()) },
            "Q8_0" | "Q4_0" => put(&mut b, 0, rng.unit() * 0.05),
            "Q4_K" | "Q5_K" => { put(&mut b, 0, rng.unit() * 0.01); put(&mut b, 2, rng.unit() * 0.01); }
            "Q6_K" => put(&mut b, 208, rng.unit() * 0.01),
            _ => unreachable!(),
        }
        out.extend(b);
    }
    out
}

#[test]
fn every_kernel_matches_the_cpu_on_every_device() {
    // CubeCL's own CPU runtime compiles slowly (minutes per kernel): STRATA_GPU_TEST_CPU=1 includes it
    let devices: Vec<_> = list().into_iter().filter(|d| d.name != "cpu" || std::env::var_os("STRATA_GPU_TEST_CPU").is_some()).collect();
    eprintln!("devices: {:?}", devices.iter().map(|d| d.name.as_str()).collect::<Vec<_>>());
    let (rows, cols, tokens) = (70, 512, 3); // 70 rows: not a multiple of the cube; 3 vectors at once
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let x: Vec<f32> = (0..cols * tokens).map(|_| rng.unit()).collect();
    for ty in ["F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_K", "Q5_K", "Q6_K"] {
        let w = weights(ty, rows, cols, &mut rng);
        let row_bytes = w.len() / rows;
        // the reference: each row dequantized on the CPU, a dot in f64
        let mut want = vec![0f32; rows * tokens];
        let mut scale = vec![0f64; rows * tokens];
        let mut deq = vec![0f32; cols];
        for r in 0..rows {
            strata_pack::quants::dequantize_into(ty, &w[r * row_bytes..(r + 1) * row_bytes], &mut deq).unwrap();
            for t in 0..tokens {
                let xs = &x[t * cols..(t + 1) * cols];
                want[t * rows + r] = deq.iter().zip(xs).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32;
                scale[t * rows + r] = deq.iter().zip(xs).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum::<f64>();
            }
        }
        for d in &devices {
            let g = Gpu::open(&d.name).unwrap();
            let wt = g.upload(ty, &w, rows, cols).unwrap();
            let got = g.matvec(&wt, &x);
            assert_eq!(got.len(), rows * tokens);
            let worst = got.iter().zip(&want).zip(&scale).map(|((g, w), s)| ((*g - *w).abs() as f64) / s.max(1e-12)).fold(0f64, f64::max);
            eprintln!("{ty:5} on {:18} worst relative error {worst:.2e}", d.name);
            assert!(worst < 1e-3, "{ty} on {}: {worst}", d.name);
        }
    }
}
