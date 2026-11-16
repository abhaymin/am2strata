//! Every kernel against the CPU's dequantizer (strata_pack::quants), on every device here: the phase 2 gate
//! (docs/GPU-PLAN.md) - within 1e-3 of the sum of |w x|, so only the order of the sums differs.

use std::sync::atomic::Ordering;
use strata_gpu::{list, Gpu, WIDE_ROWS};

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
            for (kernel, limit) in [("plain", u32::MAX), ("wide", 0)] {
                WIDE_ROWS.store(limit, Ordering::Relaxed);
                let got = g.matvec(&wt, &x);
                assert_eq!(got.len(), rows * tokens);
                let worst = got.iter().zip(&want).zip(&scale).map(|((g, w), s)| ((*g - *w).abs() as f64) / s.max(1e-12)).fold(0f64, f64::max);
                eprintln!("{ty:5} on {:18} {kernel:6} worst relative error {worst:.2e}", d.name);
                assert!(worst < 1e-3, "{ty} on {} ({kernel}): {worst}", d.name);
            }
        }
    }
}

/// A token's chain of kernels against the same arithmetic on the CPU (docs/ENGINE-PLAN.md phase 1).
#[test]
fn the_chain_of_kernels_matches_the_cpu() {
    let devices: Vec<_> = list().into_iter().filter(|d| d.name != "cpu" || std::env::var_os("STRATA_GPU_TEST_CPU").is_some()).collect();
    let mut rng = Rng(42);
    let (hd, n_head, n_kv, n) = (128usize, 8usize, 2usize, 37usize);
    let q: Vec<f32> = (0..n_head * hd).map(|_| rng.unit()).collect();
    let k: Vec<f32> = (0..n * n_kv * hd).map(|_| rng.unit()).collect();
    let v: Vec<f32> = (0..n * n_kv * hd).map(|_| rng.unit()).collect();
    let x: Vec<f32> = (0..3072).map(|_| rng.unit() * 3.0).collect();
    let w: Vec<f32> = (0..3072).map(|_| rng.unit()).collect();
    let scale = 1.0 / (hd as f32).sqrt();
    // the CPU's answers
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let kk = 1.0 / (ms + 1e-5).sqrt();
    let norm: Vec<f32> = x.iter().zip(&w).map(|(a, b)| a * kk * b).collect();
    let group = n_head / n_kv;
    let mut att = vec![0f32; n_head * hd];
    for hh in 0..n_head {
        let kh = hh / group;
        let mut s: Vec<f32> = (0..n).map(|t| (0..hd).map(|j| q[hh * hd + j] * k[(t * n_kv + kh) * hd + j]).sum::<f32>() * scale).collect();
        let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        s.iter_mut().for_each(|v| *v = (*v - m).exp());
        let t: f32 = s.iter().sum();
        for (tt, p) in s.iter().enumerate() {
            for j in 0..hd {
                att[hh * hd + j] += p / t * v[(tt * n_kv + kh) * hd + j];
            }
        }
    }
    for d in &devices {
        let g = Gpu::open(&d.name).unwrap();
        let close = |got: &[f32], want: &[f32], what: &str| {
            let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(got.len() == want.len() && worst < 2e-4, "{} {what}: worst {worst}", d.name);
        };
        close(&g.read(&g.rms_norm(&g.buffer(&x), &g.buffer(&w), 1e-5)), &norm, "rms_norm");
        let y = g.buffer(&x);
        g.add_into(&g.buffer(&w), &y);
        close(&g.read(&y), &x.iter().zip(&w).map(|(a, b)| a + b).collect::<Vec<_>>(), "add_into");
        let up = g.buffer(&x);
        g.silu_gate(&g.buffer(&w), &up);
        close(&g.read(&up), &x.iter().zip(&w).map(|(u, g)| u * (g / (1.0 + (-g).exp()))).collect::<Vec<_>>(), "silu_gate");
        let tab: Vec<f32> = (0..64).flat_map(|i| { let a = i as f32 * 0.1; [a.cos(), a.sin()] }).collect();
        let r = g.buffer(&q);
        g.rope(&g.buffer(&tab), &r, hd as u32, hd as u32, false);
        let mut want = q.clone();
        for h in want.chunks_mut(hd) {
            for p in 0..64 {
                let (c, s) = (tab[p * 2], tab[p * 2 + 1]);
                let (a, b) = (h[p * 2], h[p * 2 + 1]);
                h[p * 2] = a * c - b * s;
                h[p * 2 + 1] = a * s + b * c;
            }
        }
        close(&g.read(&r), &want, "rope");
        let cache = g.scratch(k.len());
        for t in 0..n {
            let row = n_kv * hd;
            g.kv_append(&g.buffer(&k[t * row..(t + 1) * row]), &cache, t * row);
        }
        close(&g.read(&cache), &k, "kv_append");
        let s = g.scratch(n_head * n);
        let o = g.attend(&g.buffer(&q), &cache, &g.buffer(&v), &s, n as u32, n_head as u32, n_kv as u32, hd as u32, scale);
        close(&g.read(&o), &att, "attend");
        eprintln!("{}: the chain matches", d.name);
    }
}

/// The product kernels' speed on one device: AM2_DEVICE=cuda:0 cargo test --release product_speed -- --ignored --nocapture
#[test]
#[ignore]
fn product_speed() {
    let g = Gpu::open(&std::env::var("AM2_DEVICE").unwrap_or("cuda:0".into())).unwrap();
    let mut rng = Rng(0x9e3779b97f4a7c15);
    for (rows, cols) in [(5120, 2048), (2048, 2048), (512, 2048), (64, 2048)] {
        let w = weights("Q4_K", rows, cols, &mut rng);
        let wt = g.upload("Q4_K", &w, rows, cols).unwrap();
        let x = g.buffer(&(0..cols).map(|_| rng.unit()).collect::<Vec<f32>>());
        for (kernel, limit) in [("plain", u32::MAX), ("wide", 0)] {
            WIDE_ROWS.store(limit, Ordering::Relaxed);
            g.read(&g.mv(&wt, &x)); // compiled and warm
            let t = std::time::Instant::now();
            let n = 400;
            let mut y = g.mv(&wt, &x);
            for _ in 1..n {
                y = g.mv(&wt, &x);
            }
            g.read(&y);
            let s = t.elapsed().as_secs_f64();
            eprintln!("{rows}x{cols} {kernel:6} {:7.1} us/product {:6.1} GB/s", s * 1e6 / n as f64, (w.len() * n) as f64 / s / 1e9);
        }
    }
}

/// A model's real sizes: the output projection of a 3B model is about 131072 x 3072 (Q6_K: ~330 MB), far above
/// the small matrices above and above wgpu's default 128 MiB storage-binding limit. Found when Metal on a Mac M4 gave
/// wrong tokens although every small kernel test passed. Rows from the start, the middle and the end are checked
/// against the CPU, so a binding cut short or a grid that wraps shows as the rows it gets wrong.
#[test]
fn a_model_sized_matrix_matches_the_cpu_on_every_device() {
    // STRATA_GPU_TEST_ONLY=<device> checks one device (a card that is full with a running model would only OOM)
    let only = std::env::var("STRATA_GPU_TEST_ONLY").ok();
    let devices: Vec<_> = list().into_iter().filter(|d| d.name != "cpu" && only.as_deref().is_none_or(|o| o == d.name)).collect();
    let (rows, cols) = (131072usize, 3072usize);
    let mut rng = Rng(0x2545f4914f6cdd1d);
    let ty = "Q6_K";
    let w = weights(ty, rows, cols, &mut rng);
    let row_bytes = w.len() / rows;
    let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
    let pick: Vec<usize> = (0..16).chain(rows / 2..rows / 2 + 16).chain(rows - 16..rows).collect();
    let want: Vec<(f32, f64)> = pick.iter().map(|&r| {
        let mut row = vec![0f32; cols];
        strata_pack::quants::dequantize_into(ty, &w[r * row_bytes..(r + 1) * row_bytes], &mut row).unwrap();
        let s: f64 = row.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
        let mag: f64 = row.iter().zip(&x).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
        (s as f32, mag.max(1e-9))
    }).collect();
    eprintln!("a {rows} x {cols} {ty} matrix: {} MB", w.len() / 1_000_000);
    for d in &devices {
        let gpu = Gpu::open(&d.name).unwrap();
        let up = match gpu.upload(ty, &w, rows, cols) { Ok(u) => u, Err(e) => { eprintln!("{}: upload refused: {e}", d.name); continue } };
        let y = gpu.matvec(&up, &x);
        let mut wrong = Vec::new();
        for (k, &r) in pick.iter().enumerate() {
            let e = (y[r] as f64 - want[k].0 as f64).abs() / want[k].1;
            if !(e < 1e-3) { wrong.push(r); }
        }
        eprintln!("{}: {} of {} checked rows wrong{}", d.name, wrong.len(), pick.len(),
                  if wrong.is_empty() { String::new() } else { format!(" (first: {:?})", &wrong[..wrong.len().min(6)]) });
        assert!(wrong.is_empty(), "{}: a model-sized {ty} matrix gives wrong rows {:?}", d.name, &wrong[..wrong.len().min(10)]);
    }
}

/// The engine's q/k/v and gate/up products go through `matvec_many`: one upload, the kernels launched back to back,
/// then each result read. The single-matrix tests never run that path; here three matrices of different heights share
/// one input, as q, k and v do, and each must match the CPU.
#[test]
fn several_products_at_once_match_the_cpu_on_every_device() {
    let only = std::env::var("STRATA_GPU_TEST_ONLY").ok();
    let devices: Vec<_> = list().into_iter().filter(|d| d.name != "cpu" && only.as_deref().is_none_or(|o| o == d.name)).collect();
    let mut rng = Rng(0x0ddba11c0ffee123);
    let cols = 3072usize;
    let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
    let shapes = [("Q4_K", 4096usize), ("Q4_K", 1024), ("Q6_K", 1024)];
    let ws: Vec<Vec<u8>> = shapes.iter().map(|(ty, rows)| weights(ty, *rows, cols, &mut rng)).collect();
    let want: Vec<Vec<(f32, f64)>> = shapes.iter().zip(&ws).map(|((ty, rows), w)| {
        let rb = w.len() / rows;
        (0..*rows).map(|r| {
            let mut row = vec![0f32; cols];
            strata_pack::quants::dequantize_into(ty, &w[r * rb..(r + 1) * rb], &mut row).unwrap();
            let s: f64 = row.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
            let mag: f64 = row.iter().zip(&x).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
            (s as f32, mag.max(1e-9))
        }).collect()
    }).collect();
    for d in &devices {
        let gpu = Gpu::open(&d.name).unwrap();
        let ups: Vec<_> = shapes.iter().zip(&ws).map(|((ty, rows), w)| gpu.upload(ty, w, *rows, cols).unwrap()).collect();
        for round in 0..3 {
            let got = gpu.matvec_many(&ups.iter().collect::<Vec<_>>(), &x);
            for (m, (g, wn)) in got.iter().zip(&want).enumerate() {
                let wrong = g.iter().zip(wn).filter(|(a, (b, mag))| !(((**a as f64 - *b as f64).abs() / mag) < 1e-3)).count();
                eprintln!("{} round {round} matrix {m} ({} rows): {wrong} wrong", d.name, wn.len());
                assert_eq!(wrong, 0, "{}: matvec_many matrix {m} has {wrong} wrong rows (round {round})", d.name);
            }
        }
    }
}

/// A Q4_K matrix too wide for the wide kernel's shared memory on a device with a 32 KiB limit (Metal): 3072 rows x
/// 9216 columns, a 3B model's ffn_down, which gave a Mac M4 wrong tokens. Whatever kernel is chosen, it must match.
#[test]
fn a_wide_q4k_ffn_down_matches_the_cpu_on_every_device() {
    let only = std::env::var("STRATA_GPU_TEST_ONLY").ok();
    let devices: Vec<_> = list().into_iter().filter(|d| d.name != "cpu" && only.as_deref().is_none_or(|o| o == d.name)).collect();
    let (rows, cols) = (3072usize, 9216usize);
    eprintln!("wide kernel needs {} bytes of shared memory for {cols} columns", strata_gpu::kernels::wide_shared_bytes(cols as u32));
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let w = weights("Q4_K", rows, cols, &mut rng);
    let rb = w.len() / rows;
    let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
    let want: Vec<(f32, f64)> = (0..rows).map(|r| {
        let mut row = vec![0f32; cols];
        strata_pack::quants::dequantize_into("Q4_K", &w[r * rb..(r + 1) * rb], &mut row).unwrap();
        let s: f64 = row.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
        let mag: f64 = row.iter().zip(&x).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
        (s as f32, mag.max(1e-9))
    }).collect();
    for d in &devices {
        let gpu = Gpu::open(&d.name).unwrap();
        let up = gpu.upload("Q4_K", &w, rows, cols).unwrap();
        let y = gpu.matvec(&up, &x);
        let wrong = y.iter().zip(&want).filter(|(a, (b, m))| !(((**a as f64 - *b as f64).abs() / m) < 1e-3)).count();
        eprintln!("{}: {wrong} of {rows} rows wrong", d.name);
        assert_eq!(wrong, 0, "{}: a 3072 x 9216 Q4_K product has {wrong} wrong rows", d.name);
    }
}
