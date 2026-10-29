//! One matrix x vector, repeated: per-call time on a device (cargo run --release --example bench -- cuda:0 Q4_K 3072 3072)
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dev = a.get(1).map(String::as_str).unwrap_or("cuda:0");
    let ty = a.get(2).map(String::as_str).unwrap_or("Q4_K");
    let rows: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(3072);
    let cols: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(3072);
    let (be, bb, _) = strata_gpu::kernels::geometry(ty).unwrap();
    let mut data = vec![0u8; rows * cols / be as usize * bb as usize];
    for (i, b) in data.iter_mut().enumerate() { *b = (i * 2654435761 >> 13) as u8 & 0x3f; }
    let g = strata_gpu::Gpu::open(dev).unwrap();
    let w = g.upload(ty, &data, rows, cols).unwrap();
    let x = vec![0.01f32; cols];
    let _ = g.matvec(&w, &x); // the first call compiles the kernel
    let n = 200;
    let t0 = std::time::Instant::now();
    for _ in 0..n { std::hint::black_box(g.matvec(&w, &x)); }
    let per = t0.elapsed().as_secs_f64() / n as f64;
    println!("{dev} {ty} {rows}x{cols}: {:.3} ms per matvec, {:.1} GB/s of weights", per * 1e3, data.len() as f64 / per / 1e9);
}
