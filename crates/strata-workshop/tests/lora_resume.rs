//! Step 3, part D: a checkpoint is what PEFT expects on disk, and resuming from it is exact: the steps after a resume
//! give the same losses as the uninterrupted run (adapter weights, AdamW moments and step counts all restored).

use std::collections::BTreeMap;

use candle_core::{Device, Tensor};
use strata_workshop::checkpoint;
use strata_workshop::lora::{LoraConfig, TinyLlama};
use strata_workshop::testmodel::TinyLlama as Fixture;
use strata_workshop::train::{make_train_data, Trainer, TrainingConfig};

fn base(dev: &Device) -> BTreeMap<String, Tensor> {
    Fixture::default().generate().into_iter().map(|t| (t.name, Tensor::from_vec(t.values, t.shape.clone(), dev).unwrap())).collect()
}

fn cfgs() -> (LoraConfig, TrainingConfig) {
    (LoraConfig { r: 8, lora_alpha: 16.0, use_rslora: true, target_modules: vec!["q_proj".into(), "k_proj".into(), "v_proj".into(), "o_proj".into()] },
     TrainingConfig { lr: 0.05, warmup_steps: 2, total_steps: 10, accumulation_steps: 2 })
}

#[test]
fn resume_reproduces_the_uninterrupted_run() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let (lcfg, tcfg) = cfgs();
    let data = make_train_data(11, 12, 8, 32, 2);
    let dir = std::env::temp_dir().join(format!("am2-workshop-resume-{}-{}", std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()));

    // the uninterrupted run: 4 steps, checkpoint, then steps 4 and 5
    let model = TinyLlama::load(&base(&dev), &lcfg, &dev)?;
    let mut trainer = Trainer::new(&model, tcfg.clone())?;
    for step in 0..4 { trainer.train_step(&model, &data, step)?; }
    checkpoint::save(&dir, &model, &lcfg, &trainer, 4)?;
    let after: Vec<f64> = (4..6).map(|s| trainer.train_step(&model, &data, s)).collect::<Result<_, _>>()?;

    // a fresh model and trainer, resumed from the checkpoint, take the same two steps
    let mut model2 = TinyLlama::load(&base(&dev), &lcfg, &dev)?;
    let mut trainer2 = Trainer::new(&model2, tcfg)?;
    let step = checkpoint::load(&dir, &mut model2, &mut trainer2)?;
    assert_eq!(step, 4);
    let resumed: Vec<f64> = (step..step + 2).map(|s| trainer2.train_step(&model2, &data, s)).collect::<Result<_, _>>()?;
    for (a, b) in after.iter().zip(&resumed) {
        assert!((a - b).abs() < 1e-6, "resumed losses {resumed:?} differ from the uninterrupted run {after:?}");
    }

    // what is on disk is PEFT's layout
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(checkpoint::CONFIG_FILE))?)?;
    assert_eq!((cfg["peft_type"].as_str(), cfg["r"].as_u64(), cfg["use_rslora"].as_bool()), (Some("LORA"), Some(8), Some(true)));
    let adapter = candle_core::safetensors::load(dir.join(checkpoint::ADAPTER_FILE), &dev)?;
    assert!(adapter.contains_key("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"));
    assert!(adapter.contains_key("base_model.model.model.layers.1.self_attn.o_proj.lora_B.weight"));
    assert_eq!(adapter["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"].dims(), &[8, 16]);
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

#[test]
fn a_checkpoint_for_another_rank_is_refused() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let (lcfg, tcfg) = cfgs();
    let dir = std::env::temp_dir().join(format!("am2-workshop-rank-{}", std::process::id()));
    let model = TinyLlama::load(&base(&dev), &lcfg, &dev)?;
    checkpoint::save(&dir, &model, &lcfg, &Trainer::new(&model, tcfg.clone())?, 0)?;
    let other = LoraConfig { r: 4, ..lcfg };
    let mut model4 = TinyLlama::load(&base(&dev), &other, &dev)?;
    let mut trainer4 = Trainer::new(&model4, tcfg)?;
    let err = checkpoint::load(&dir, &mut model4, &mut trainer4).unwrap_err().to_string();
    assert!(err.contains("shape"), "a rank mismatch names the shapes: {err}");
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
