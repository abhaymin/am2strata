//! The engine's pattern: products of several shapes in turn (cargo run --release --example bench_mix -- cuda:0)
fn main() {
    let dev = std::env::args().nth(1).unwrap_or("cuda:0".into());
    let g = strata_gpu::Gpu::open(&dev).unwrap();
    let shapes = [("Q4_K", 3072, 3072), ("Q4_K", 1024, 3072), ("Q4_K", 9216, 3072), ("Q4_K", 3072, 9216), ("Q6_K", 131072, 3072)];
    let ws: Vec<_> = shapes.iter().map(|&(ty, r, c)| {
        let (be, bb, _) = strata_gpu::kernels::geometry(ty).unwrap();
        let data: Vec<u8> = (0..r * c / be as usize * bb as usize).map(|i| (i * 2654435761 >> 13) as u8 & 0x3f).collect();
        (g.upload(ty, &data, r, c).unwrap(), c, ty, r)
    }).collect();
    for (w, c, _, _) in &ws { let _ = g.matvec(w, &vec![0.01; *c]); }
    for (w, c, ty, r) in &ws {
        let t0 = std::time::Instant::now();
        for _ in 0..50 { std::hint::black_box(g.matvec(w, &vec![0.01; *c])); }
        println!("{dev} {ty} {r}x{c}: {:.3} ms each (alone)", t0.elapsed().as_secs_f64() * 1e3 / 50.0);
    }
    let t0 = std::time::Instant::now();
    for _ in 0..20 { for (w, c, _, _) in &ws { std::hint::black_box(g.matvec(w, &vec![0.01; *c])); } }
    println!("{dev} mixed: {:.3} ms per round of {}", t0.elapsed().as_secs_f64() * 1e3 / 20.0, ws.len());
}
