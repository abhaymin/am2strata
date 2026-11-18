//! Per-block recompute and the GPU devices: both must give the plain CPU backward pass's loss and gradients.
//! The GPU half runs only in a build with `--features metal` (or `cuda`) on a machine that has that GPU.

use candle_core::{Device, Tensor};
use strata_workshop::model::{BaseDtype, Ctx, LlamaModel, LoraSpec, Target};
use strata_workshop::testmodel::{write_chat_model, TinyLlama};

const TOKENS: usize = 40;

fn model_dir() -> tempfile_dir::Dir {
    let d = tempfile_dir::Dir::new("recompute");
    // three blocks, grouped-query attention, a vocabulary big enough for the head to be chunked
    let tiny = TinyLlama { num_layers: 3, vocab_size: 259, hidden: 64, intermediate: 128, num_attention_heads: 8, num_key_value_heads: 2,
                           seed: 11, ..Default::default() };
    write_chat_model(d.path(), &tiny).unwrap();
    d
}

/// Loss and every adapter gradient, plain or recomputed, on `dev`.
fn run(dir: &std::path::Path, dev: &Device, dtype: BaseDtype, recompute: bool, head_rows: usize, train_layers: Option<usize>) -> (f64, Vec<(String, Vec<f32>)>) {
    let mut model = LlamaModel::load(dir, dtype, dev).unwrap();
    // dropout on: the recomputed blocks must draw the very same masks
    model.add_lora(&LoraSpec { r: 4, alpha: 8.0, dropout: 0.1, use_rslora: false, targets: Target::ALL.to_vec(), train_layers, seed: 7 }).unwrap();
    for (name, var) in model.lora_vars() {
        if name.ends_with("lora_B.weight") {
            let n = var.as_tensor().elem_count();
            let vals: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 2000.0).collect();
            var.set(&Tensor::from_vec(vals, var.as_tensor().shape(), dev).unwrap()).unwrap();
        }
    }
    let tokens: Vec<u32> = (0..TOKENS).map(|i| ((i * 7 + 3) % 259) as u32).collect();
    let supervised: Vec<bool> = (0..TOKENS).map(|p| p >= 6).collect();
    let (names, vars): (Vec<String>, Vec<_>) = model.lora_vars().into_iter().unzip();
    let h = model.lower(&tokens).unwrap();
    let ctx = Ctx::train(1234);
    let (loss, grads) = if recompute {
        model.loss_and_grads_recompute(&h, &tokens, &supervised, &ctx, &vars, head_rows).unwrap()
    } else {
        let loss = model.loss_from_hidden(&h, &tokens, &supervised, &ctx).unwrap();
        let store = loss.backward().unwrap();
        (loss.to_scalar::<f32>().unwrap() as f64, vars.iter().map(|v| store.get(v).cloned()).collect())
    };
    let g = names.into_iter().zip(grads)
        .map(|(n, g)| (n.clone(), g.unwrap_or_else(|| panic!("{n}: no gradient")).to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()))
        .collect();
    (loss, g)
}

fn worst(a: &[(String, Vec<f32>)], b: &[(String, Vec<f32>)]) -> f32 {
    let mut worst = 0.0f32;
    for ((name, x), (_, y)) in a.iter().zip(b) {
        assert!(x.iter().any(|v| *v != 0.0), "{name}: the gradient is alive");
        let scale = x.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-6);
        worst = worst.max(x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0.0, f32::max) / scale);
    }
    worst
}

#[test]
fn recompute_gives_the_plain_backward_pass() {
    let d = model_dir();
    for train_layers in [None, Some(2)] {
        let (l0, g0) = run(d.path(), &Device::Cpu, BaseDtype::F32, false, 0, train_layers);
        // head rows 7: the 34 scored rows split unevenly
        let (l1, g1) = run(d.path(), &Device::Cpu, BaseDtype::F32, true, 7, train_layers);
        let w = worst(&g0, &g1);
        println!("train_layers {train_layers:?}: loss {l0:.6} vs {l1:.6}; gradients differ by {w:.2e} of their largest value");
        assert!((l0 - l1).abs() < 1e-5, "loss {l0} vs {l1}");
        assert!(w < 1e-4, "recomputed gradients differ by {w}");
    }
}

/// The first GPU the build has, if this machine has it, and its name for the output.
fn gpu_named() -> Option<(&'static str, Device)> {
    if cfg!(feature = "metal") {
        Device::new_metal(0).ok().map(|d| ("Metal", d))
    } else if cfg!(feature = "cuda") {
        Device::new_cuda(0).ok().map(|d| ("CUDA", d))
    } else {
        None
    }
}

fn gpu() -> Option<Device> {
    gpu_named().map(|(_, d)| d)
}

#[test]
fn the_gpu_trains_like_the_cpu() {
    let Some((name, dev)) = gpu_named() else {
        println!("no GPU in this build or machine: skipped");
        return;
    };
    let d = model_dir();
    for (dtype, recompute) in [(BaseDtype::F32, false), (BaseDtype::F32, true), (BaseDtype::BF16, true)] {
        let (l0, g0) = run(d.path(), &Device::Cpu, dtype, recompute, 16, None);
        let (l1, g1) = run(d.path(), &dev, dtype, recompute, 16, None);
        let w = worst(&g0, &g1);
        println!("{dtype:?} recompute {recompute}: CPU loss {l0:.6}, {name} {l1:.6}; gradients differ by {w:.2e} of their largest value");
        assert!((l0 - l1).abs() < 1e-3, "loss {l0} vs {l1}");
        assert!(w < 1e-2, "GPU gradients differ by {w}");
    }
}

/// A scratch folder removed at the end (no tempfile dependency in this crate).
mod tempfile_dir {
    pub struct Dir(std::path::PathBuf);
    impl Dir {
        pub fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!("strata-workshop-{tag}-{}-{:?}", std::process::id(), std::thread::current().id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Dir(p)
        }
        pub fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

/// Seconds per training step (forward + backward) of a mid-size model, CPU against the GPU: `cargo test --release
/// --features metal --test recompute_gpu -- --ignored --nocapture`.
#[test]
#[ignore]
fn step_time_cpu_against_gpu() {
    let d = tempfile_dir::Dir::new("bench");
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (hidden, layers) = (env("BENCH_HIDDEN", 1024), env("BENCH_LAYERS", 8));
    let dropout = env("BENCH_DROPOUT_PCT", 5) as f64 / 100.0;
    let tiny = TinyLlama { num_layers: layers, vocab_size: 259, hidden, intermediate: hidden * 11 / 4, num_attention_heads: 16, num_key_value_heads: 4, seed: 3, ..Default::default() };
    write_chat_model(d.path(), &tiny).unwrap();
    let seq = 512;
    let tokens: Vec<u32> = (0..seq).map(|i| ((i * 7 + 3) % 259) as u32).collect();
    let supervised: Vec<bool> = (0..seq).map(|p| p >= 8).collect();
    let mut devs = vec![("cpu", Device::Cpu)];
    if let Some((name, g)) = gpu_named() { devs.push((name, g)); }
    for (label, dev) in devs {
        for recompute in [false, true] {
            let mut model = LlamaModel::load(d.path(), BaseDtype::BF16, &dev).unwrap();
            model.add_lora(&LoraSpec { r: 16, alpha: 32.0, dropout, use_rslora: false, targets: Target::ALL.to_vec(), train_layers: None, seed: 1 }).unwrap();
            let vars: Vec<_> = model.lora_vars().into_iter().map(|(_, v)| v).collect();
            let h = model.lower(&tokens).unwrap();
            let mut times = Vec::new();
            for i in 0..4 {
                let t = std::time::Instant::now();
                let ctx = Ctx::train(i);
                let g = if recompute {
                    model.loss_and_grads_recompute(&h, &tokens, &supervised, &ctx, &vars, 256).unwrap().1
                } else {
                    let s = model.loss_from_hidden(&h, &tokens, &supervised, &ctx).unwrap().backward().unwrap();
                    vars.iter().map(|v| s.get(v).cloned()).collect()
                };
                // wait for the GPU: read one value back
                let _ = g[0].as_ref().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
                times.push(t.elapsed().as_secs_f64());
            }
            println!("hidden {hidden} x{layers} dropout {dropout} {label} recompute {recompute}: {:.3} s per {seq}-token step (first {:.3} s)", times[1..].iter().sum::<f64>() / 3.0, times[0]);
        }
    }
}
