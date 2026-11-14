//! Step 5 — merge a LoRA adapter into its base safetensors, shard by shard. Gates (each green before the next commit):
//! 1. train the tiny model 8 steps, checkpoint it, and merge it into the base;
//! 2. the merged base's plain forward (B = 0 adapters) reproduces the adapter model's logits (<= 1e-4);
//! 3. untargeted tensors (norms, MLP, embeddings) are byte-identical to the base after merge;
//! 4. rsLoRA on/off picks the right scaling — the correct factor passes gate 2, a deliberately wrong one fails it.

use std::collections::BTreeMap;

use anyhow::anyhow;
use candle_core::{Device, Tensor};
use strata_workshop::checkpoint;
use strata_workshop::lora::{LoraConfig, TinyLlama};
use strata_workshop::merge::{self, Merger};
use strata_workshop::testmodel::TinyLlama as Fixture;
use strata_workshop::train::{make_train_data, Trainer, TrainingConfig};

/// f32 buffer element-wise max abs difference.
fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
    let va: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
    let vb: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(va.len(), vb.len());
    va.iter().zip(&vb).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("am2-workshop-merge-{tag}-{nanos}"))
}

/// A base fixture written to disk, plus the config the adapter trains with.
fn setup() -> (std::path::PathBuf, LoraConfig) {
    // rsLoRA on: exercises the alpha / sqrt(r) scaling path and proves it reproduces the adapter logits in gate 2.
    let cfg = LoraConfig { r: 8, lora_alpha: 16.0, use_rslora: true, target_modules: vec!["q_proj".into(), "k_proj".into(), "v_proj".into(), "o_proj".into()] };
    let dir = temp_dir("base");
    Fixture::default().write(&dir).unwrap();
    (dir, cfg)
}

/// Train 8 steps on the base fixture, checkpoint the adapter, and return the trained model + adapter dir.
fn trained(_base: &std::path::Path, dev: &Device, cfg: &LoraConfig) -> (TinyLlama, std::path::PathBuf) {
    let base_tensors: BTreeMap<String, Tensor> = Fixture::default().generate().into_iter().map(|t| (t.name, Tensor::from_vec(t.values, t.shape.clone(), dev).unwrap())).collect();
    let model = TinyLlama::load(&base_tensors, cfg, dev).unwrap();
    let mut trainer = Trainer::new(&model, TrainingConfig { lr: 0.1, warmup_steps: 3, total_steps: 20, accumulation_steps: 2 }).unwrap();
    let data = make_train_data(7, 20, 8, 32, 2);
    for step in 0..8 { trainer.train_step(&model, &data, step).unwrap(); }

    let adapter_dir = temp_dir("adapter");
    checkpoint::save(&adapter_dir, &model, cfg, &trainer, 8).unwrap();
    (model, adapter_dir)
}

/// The logits the merged base produces with its adapters folded in but held at zero (i.e. plain frozen-forward), for `tok`.
fn merged_plain_logits(merged: &std::path::Path, tok: &Tensor, cfg: &LoraConfig, dev: &Device) -> Tensor {
    // load whatever shard(s) the merge wrote; single-shard base => one file here
    let mut base = BTreeMap::new();
    for sf in strata_workshop::safetensors_io::shards(&merged.join("model-00001-of-00001.safetensors")) {
        for (k, v) in candle_core::safetensors::load(&sf, dev).unwrap() {
            base.insert(k, v);
        }
    }
    let mut model = TinyLlama::load(&base, cfg, dev).unwrap();
    // zero every B so the adapter contributes nothing — only the folded weight remains.
    for li in 0..model.num_layers {
        for lin in model.attn[li].iter_mut() {
            lin.set_b_values(&vec![0f32; cfg.r * hidden(dev)]).unwrap();
        }
    }
    model.forward(tok).unwrap()
}

fn hidden(dev: &Device) -> usize {
    // the fixture's hidden size (see testmodel::Default); kept here rather than importing the fixture internals.
    let _ = dev;
    16usize
}

/// The base weights a LoRA adapter targets — the attention projections, one per layer per q/k/v/o.
fn targeted_names() -> Vec<String> {
    let mut v = Vec::new();
    for li in 0..2 {
        for m in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            v.push(format!("model.layers.{li}.self_attn.{m}.weight"));
        }
    }
    v
}

#[test]
fn train_then_merge_is_a_checkpoint_and_logits_match() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let (base, cfg) = setup();
    let (model, adapter_dir) = trained(&base, &dev, &cfg);

    // the merged base sits on disk next to the original fixture
    let merged = temp_dir("merged");
    merge::merge(&base, &adapter_dir, &merged).unwrap();
    let merged_shard = merged.join("model-00001-of-00001.safetensors");
    assert!(merged_shard.exists(), "the merge wrote a shard under the base's name");

    // gate 2: the merged base's plain forward reproduces the adapter model's logits within tolerance.
    let tok = Tensor::from_vec(vec![3u32, 1, 9, 15, 7], (5,), &dev)?;
    let adapted_logits = model.forward(&tok)?;
    let plain_logits = merged_plain_logits(&merged, &tok, &cfg, &dev);
    assert!(max_abs(&adapted_logits, &plain_logits) <= 1e-4, "gate 2 failed: merged plain logits differ from the adapter logits by {}", max_abs(&adapted_logits, &plain_logits));

    let _ = std::fs::remove_dir_all(&base);
    let _ = std::fs::remove_dir_all(&merged);
    Ok(())
}

#[test]
fn untargeted_tensors_are_byte_identical_to_the_base() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let (base, cfg) = setup();
    let (_model, adapter_dir) = trained(&base, &dev, &cfg);

    let merged = temp_dir("merged");
    merge::merge(&base, &adapter_dir, &merged).unwrap();

    // read both the original base shard and the merged shard as tensors; untargeted names must match exactly.
    let orig: BTreeMap<String, Tensor> = candle_core::safetensors::load(base.join("model-00001-of-00001.safetensors"), &dev)?.into_iter().collect();
    let post: BTreeMap<String, Tensor> = candle_core::safetensors::load(merged.join("model-00001-of-00001.safetensors"), &dev)?.into_iter().collect();

    for (name, t) in &orig {
        if targeted_names().contains(name) {
            continue; // targeted tensors are expected to change
        }
        let got = post.get(name).ok_or_else(|| anyhow!("{name} vanished from the merged shard"))?;
        let a: Vec<f32> = t.flatten_all()?.to_vec1()?;
        let b: Vec<f32> = got.flatten_all()?.to_vec1()?;
        assert_eq!(a, b, "{name}: untargeted tensor must be byte-identical to the base");
    }

    // and the merge report flags the targeted tensors as changed and every other one as exactly zero.
    let report = merge::merge(&base, &adapter_dir, &temp_dir("merged-report")).unwrap();
    let by_name: std::collections::HashMap<String, f64> = report.per_tensor.into_iter().map(|d| (d.name, d.delta_rel_norm)).collect();
    for name in targeted_names() {
        assert!(by_name[&name] > 0.0, "targeted {name} should have a non-zero relative-norm delta");
    }
    for (name, rel) in &by_name {
        if !targeted_names().contains(name) {
            assert_eq!(*rel, 0.0, "{name} relative-norm delta must be exactly 0");
        }
    }

    let _ = std::fs::remove_dir_all(&base);
    Ok(())
}

#[test]
fn rslora_scaling_is_the_one_that_reproduces_logits() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let (base, cfg) = setup(); // use_rslora == true => correct scaling is alpha / sqrt(r)
    let (model, adapter_dir) = trained(&base, &dev, &cfg);

    let tok = Tensor::from_vec(vec![2u32, 11, 5, 20], (4,), &dev)?;
    let adapted_logits = model.forward(&tok)?;

    // correct scaling: gate 2 holds.
    let ok = temp_dir("merged-ok");
    merge::merge(&base, &adapter_dir, &ok).unwrap();
    let plain_ok = merged_plain_logits(&ok, &tok, &cfg, &dev);
    assert!(max_abs(&adapted_logits, &plain_ok) <= 1e-4, "gate 2 must pass at the correct (rsLoRA) scaling");

    // wrong scaling: fold in alpha / r instead of alpha / sqrt(r). The delta is off and gate 2 must fail.
    let wrong = temp_dir("merged-wrong");
    let mut g = Merger::new(&base, &adapter_dir, &wrong);
    g.set_scaling(cfg.lora_alpha / cfg.r as f32);
    g.run().unwrap();
    let plain_wrong = merged_plain_logits(&wrong, &tok, &cfg, &dev);
    assert!(max_abs(&adapted_logits, &plain_wrong) > 1e-4, "gate 2 must fail at a wrong scaling (diff {})", max_abs(&adapted_logits, &plain_wrong));

    let _ = std::fs::remove_dir_all(&base);
    Ok(())
}
