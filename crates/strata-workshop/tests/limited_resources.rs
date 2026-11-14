//! The limited-resource modes, each with what it saves: a 16-bit base, LoRA in the top N blocks only (the rest runs
//! without a graph), and the activation cache (the frozen blocks run once per conversation, then leave memory).

mod common;

use std::path::PathBuf;

use candle_core::Device;
use strata_workshop::finetune::{self, Event, TrainSettings};
use strata_workshop::model::{BaseDtype, Ctx, LlamaModel, LoraSpec, Target};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn settings(model: &std::path::Path, data: &std::path::Path, out: &std::path::Path) -> TrainSettings {
    let mut s = TrainSettings::new(model, data, out);
    s.lora = LoraSpec { r: 4, alpha: 8.0, dropout: 0.05, use_rslora: true, targets: Target::ALL.to_vec(), train_layers: Some(1), seed: 3 };
    s.lr = 2e-2;
    s.epochs = 3.0;
    s.accumulation = 2;
    s.max_len = 64;
    s.val_split = 0.125;
    s.save_steps = 0;
    s.logging_steps = 1;
    s.seed = 5;
    s
}

#[test]
fn a_half_precision_base_holds_half_the_bytes_and_gives_the_same_gradients() {
    // the fixture's weights are BF16 on disk: held as BF16 they take 2 bytes each, and widening them is exact, so the
    // gradients through the 16-bit path (FrozenLinear) must equal the f32 path's
    let dir = fixtures().join("tiny-llama-256-bf16-hf");
    let spec = LoraSpec { r: 4, alpha: 8.0, dropout: 0.0, use_rslora: false, targets: Target::ALL.to_vec(), train_layers: None, seed: 9 };
    let tokens: Vec<u32> = (0..12).map(|i| (i * 7 + 3) % 32).collect();
    let supervised: Vec<bool> = (0..12).map(|p| p >= 5).collect();
    let mut grads = Vec::new();
    let mut bytes = Vec::new();
    for dtype in [BaseDtype::F32, BaseDtype::BF16] {
        let mut model = LlamaModel::load(&dir, dtype, &Device::Cpu).unwrap();
        model.add_lora(&spec).unwrap();
        // B starts at zero, which hides A's gradient; give B values so every gradient is alive
        for (name, var) in model.lora_vars() {
            if name.ends_with("lora_B.weight") {
                let n = var.as_tensor().elem_count();
                let vals: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 2000.0).collect();
                var.set(&candle_core::Tensor::from_vec(vals, var.as_tensor().shape(), &Device::Cpu).unwrap()).unwrap();
            }
        }
        let loss = model.loss(&tokens, &supervised, &Ctx::eval()).unwrap();
        let store = loss.backward().unwrap();
        let g: Vec<(String, Vec<f32>)> = model.lora_vars().into_iter()
            .map(|(n, v)| (n, store.get(&v).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap())).collect();
        grads.push((loss.to_scalar::<f32>().unwrap(), g));
        bytes.push(model.base_bytes());
    }
    let (f32_bytes, bf16_bytes) = (bytes[0], bytes[1]);
    // matrices: embed + head 2·32·256, one block 4·256·256 + 3·256·512; norms (3 × 256 values) stay f32
    let (matrices, vectors) = (2 * 32 * 256 + 4 * 256 * 256 + 3 * 256 * 512, 3 * 256u64);
    assert_eq!(f32_bytes, matrices as u64 * 4 + vectors * 4);
    assert_eq!(bf16_bytes, matrices as u64 * 2 + vectors * 4);
    println!("base in memory: {f32_bytes} bytes as f32, {bf16_bytes} as bf16 ({:.1} %)", 100.0 * bf16_bytes as f64 / f32_bytes as f64);
    assert!((grads[0].0 - grads[1].0).abs() < 1e-6, "same loss");
    let mut worst = 0.0f32;
    for ((name, a), (_, b)) in grads[0].1.iter().zip(&grads[1].1) {
        let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        assert!(a.iter().any(|x| *x != 0.0), "{name}: the gradient is alive");
        worst = worst.max(d);
    }
    println!("gradients f32 vs bf16 base: max abs difference {worst:.3e}");
    assert!(worst < 1e-6, "the 16-bit path's gradients differ by {worst}");
}

#[test]
fn a_half_precision_base_trains() {
    let (root, model, data) = common::model_and_data("f16");
    let mut s = settings(&model, &data, &root.join("out"));
    s.lora.train_layers = None;
    s.base_dtype = BaseDtype::F16;
    s.epochs = 4.0;
    let r = finetune::train(&s, &mut |_| {}, &|| false).unwrap();
    let n = r.losses.len();
    let (first, last) = (r.losses[..4].iter().sum::<f64>() / 4.0, r.losses[n - 4..].iter().sum::<f64>() / 4.0);
    println!("f16 base: loss {first:.3} → {last:.3}");
    assert!(last < 0.8 * first, "{first} → {last}");
    // the adapter is f32 whatever the base is
    let t = candle_core::safetensors::load(root.join("out/adapter_model.safetensors"), &Device::Cpu).unwrap();
    assert!(t.values().all(|t| t.dtype() == candle_core::DType::F32));
}

#[test]
fn the_top_blocks_alone_train_and_the_adapter_says_which() {
    let (root, model, data) = common::model_and_data("top");
    let mut s = settings(&model, &data, &root.join("out"));
    s.lora.train_layers = Some(2);
    let r = finetune::train(&s, &mut |_| {}, &|| false).unwrap();
    // 3 blocks, the top 2 carry adapters: r·(in + out) over the seven targets
    // q 32→32, k and v 32→16, o 32→32, gate and up 32→64, down 64→32
    let per_block = 4 * ((32 + 32) + 2 * (32 + 16) + (32 + 32) + 2 * (32 + 64) + (64 + 32));
    assert_eq!(r.trainable_params, 2 * per_block as u64);
    let t = candle_core::safetensors::load(root.join("out/adapter_model.safetensors"), &Device::Cpu).unwrap();
    assert!(t.keys().all(|k| !k.contains(".layers.0.")), "block 0 has no adapter");
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("out/adapter_config.json")).unwrap()).unwrap();
    assert_eq!(cfg["layers_to_transform"], serde_json::json!([1, 2]), "PEFT is told which blocks the adapter covers");
    // --train-layers beyond the model is refused, not clamped
    s.lora.train_layers = Some(9);
    assert!(finetune::train(&s, &mut |_| {}, &|| false).unwrap_err().to_string().contains("3 blocks"));
}

#[test]
fn the_activation_cache_changes_nothing_but_the_work() {
    let (root, model, data) = common::model_and_data("cache");
    // without the cache: the frozen blocks run for every conversation, every epoch
    let plain = settings(&model, &data, &root.join("plain"));
    let a = finetune::train(&plain, &mut |_| {}, &|| false).unwrap();
    assert!(a.lower_in_memory);
    assert_eq!((a.cache_hits, a.cache_misses), (0, 0));

    // with it: each conversation's hidden state is computed once, stored, and read back from then on
    let mut cached = plain.clone();
    cached.out = root.join("cached");
    cached.cache_dir = Some(root.join("activations"));
    let mut logs = Vec::new();
    let b = finetune::train(&cached, &mut |e| if let Event::Log(m) = e { logs.push(m) }, &|| false).unwrap();
    assert_eq!(b.losses, a.losses, "the loss curve is the same, step for step, to the last bit");
    assert_eq!(b.eval_loss, a.eval_loss);
    assert_eq!(b.cache_misses, 24, "one state per conversation (21 trained on, 3 held out)");
    // 21 conversations × 3 epochs read back, and the 3 held-out ones once
    assert_eq!(b.cache_hits, 21 * 3 + 3);
    assert!(!b.lower_in_memory, "the frozen blocks were freed once the cache was full");
    assert!(logs.iter().any(|m| m.contains("blocks 0 to 1 freed")), "{logs:?}");

    // a second run finds the cache complete and never loads the frozen blocks; the curve is still the same
    let mut again = cached.clone();
    again.out = root.join("again");
    let mut logs = Vec::new();
    let c = finetune::train(&again, &mut |e| if let Event::Log(m) = e { logs.push(m) }, &|| false).unwrap();
    assert_eq!(c.losses, a.losses);
    assert_eq!((c.cache_misses, c.cache_hits), (0, 21 * 3 + 3));
    assert!(!c.lower_in_memory);
    assert!(logs.iter().any(|m| m.contains("were not loaded")), "{logs:?}");

    // and the adapters of all three runs are the same bytes
    let load = |d: &str| std::fs::read(root.join(d).join("adapter_model.safetensors")).unwrap();
    let (pa, pb, pc) = (load("plain"), load("cached"), load("again"));
    let tensors = |b: &[u8]| -> Vec<(String, Vec<f32>)> {
        let mut v: Vec<(String, Vec<f32>)> = candle_core::safetensors::load_buffer(b, &Device::Cpu).unwrap().into_iter()
            .map(|(k, t)| (k, t.flatten_all().unwrap().to_vec1().unwrap())).collect();
        v.sort_by(|x, y| x.0.cmp(&y.0));
        v
    };
    assert_eq!(tensors(&pa), tensors(&pb));
    assert_eq!(tensors(&pa), tensors(&pc));

    // a cache made for other settings is refused, not reused
    let mut other = cached.clone();
    other.out = root.join("other");
    other.lora.train_layers = Some(2);
    assert!(finetune::train(&other, &mut |_| {}, &|| false).unwrap_err().to_string().contains("made for another model"));
    // and with every block training there is nothing to cache: said, not failed
    let mut all = cached.clone();
    all.out = root.join("all");
    all.cache_dir = Some(root.join("activations-all"));
    all.lora.train_layers = None;
    all.max_steps = Some(2);
    let mut logs = Vec::new();
    finetune::train(&all, &mut |e| if let Event::Log(m) = e { logs.push(m) }, &|| false).unwrap();
    assert!(logs.iter().any(|m| m.contains("activation cache is off")), "{logs:?}");
}

#[test]
fn the_frozen_blocks_leave_memory_with_the_cache() {
    let root = common::temp_dir("drop");
    let dir = root.join("model");
    common::chat_model_sized(&dir, 6, 64, 128);
    let mut model = LlamaModel::load(&dir, BaseDtype::F32, &Device::Cpu).unwrap();
    model.add_lora(&LoraSpec { r: 2, alpha: 4.0, dropout: 0.0, use_rslora: false, targets: vec![Target::Q, Target::V], train_layers: Some(1), seed: 1 }).unwrap();
    let full = model.base_bytes();
    let tokens: Vec<u32> = (3..40).collect();
    let h = model.lower(&tokens).unwrap();
    let want = model.upper(&h, &Ctx::eval()).unwrap();
    model.drop_lower();
    // what stays: one block of six, the head and the final norm (the embeddings went too: the head is its own tensor)
    let block = 4 * (64 * 64 + 2 * 32 * 64 + 64 * 64 + 3 * 64 * 128) + 4 * 2 * 64;
    assert_eq!(model.base_bytes(), (block + 4 * 259 * 64 + 4 * 64) as u64);
    println!("base in memory: {full} bytes with every block, {} with the top block only ({:.1} %)", model.base_bytes(), 100.0 * model.base_bytes() as f64 / full as f64);
    assert!(!model.has_lower());
    // the trained part still works from the stored state, and says so plainly when asked for the part that is gone
    let got = model.upper(&h, &Ctx::eval()).unwrap();
    assert_eq!(got.to_vec2::<f32>().unwrap(), want.to_vec2::<f32>().unwrap());
    assert!(model.forward(&tokens, &Ctx::eval()).unwrap_err().to_string().contains("not in memory"));
    // a model loaded without its lower part is the same thing
    let partial = LlamaModel::load_from(&dir, BaseDtype::F32, &Device::Cpu, 5).unwrap();
    assert_eq!(partial.base_bytes(), model.base_bytes());
}

#[test]
fn four_bit_training_is_not_claimed() {
    assert!(!strata_workshop::QLORA_READY);
    let e = BaseDtype::from_s("4bit").unwrap_err().to_string();
    assert!(e.contains("QLoRA") && e.contains("not done"), "{e}");
}
