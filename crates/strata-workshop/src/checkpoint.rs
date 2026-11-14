//! Step 3, part D: a training checkpoint that PEFT can read and that resumes exactly.
//!
//! A checkpoint folder holds:
//! - `adapter_model.safetensors`: the LoRA A/B matrices under PEFT's names
//!   (`base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight`, ...), f32;
//! - `adapter_config.json`: the hyper-parameters as PEFT records them (`peft_type`, `r`, `lora_alpha`, `use_rslora`,
//!   `target_modules`, ...), so `PeftModel.from_pretrained` or Unsloth can load the adapter;
//! - `optimizer.safetensors`: AdamW's first and second moments per adapter matrix (`<name>.exp_avg`, `<name>.exp_avg_sq`);
//! - `trainer_state.json`: the global step and AdamW's own step count.
//!
//! The last two are ours (HF's Trainer keeps the same facts in `optimizer.pt` / `trainer_state.json`); with them a
//! resumed run takes exactly the steps an uninterrupted run would.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};
use serde_json::json;

use crate::lora::{LoraConfig, TinyLlama};
use crate::train::{trainable_params, Trainer};

type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub const ADAPTER_FILE: &str = "adapter_model.safetensors";
pub const CONFIG_FILE: &str = "adapter_config.json";
pub const OPTIMIZER_FILE: &str = "optimizer.safetensors";
pub const STATE_FILE: &str = "trainer_state.json";

/// Write a checkpoint of `model`'s adapters and `trainer`'s optimizer state at `global_step` into `dir`.
pub fn save(dir: &Path, model: &TinyLlama, cfg: &LoraConfig, trainer: &Trainer, global_step: usize) -> Res<()> {
    std::fs::create_dir_all(dir)?;
    let (_vars, names) = trainable_params(model)?;

    // adapter weights under PEFT names, in PEFT order
    let adapter: Vec<(String, Tensor)> = names.iter().cloned()
        .zip(trainable_params(model)?.0.iter().map(|v| v.as_tensor().clone()))
        .collect();
    candle_core::safetensors::save(&adapter.into_iter().collect::<HashMap<_, _>>(), dir.join(ADAPTER_FILE))?;

    let config = json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": cfg.r,
        "lora_alpha": cfg.lora_alpha,
        "use_rslora": cfg.use_rslora,
        "target_modules": cfg.target_modules,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": false,
        "inference_mode": false,
    });
    std::fs::write(dir.join(CONFIG_FILE), serde_json::to_string_pretty(&config)?)?;

    // AdamW moments, named after the matrix they belong to
    let mut moments: HashMap<String, Tensor> = HashMap::new();
    for (name, (_theta, m, v)) in names.iter().zip(trainer.optimizer.slots.iter()) {
        moments.insert(format!("{name}.exp_avg"), m.as_tensor().clone());
        moments.insert(format!("{name}.exp_avg_sq"), v.as_tensor().clone());
    }
    candle_core::safetensors::save(&moments, dir.join(OPTIMIZER_FILE))?;

    let state = json!({ "global_step": global_step, "adam_step": trainer.optimizer.step_t });
    std::fs::write(dir.join(STATE_FILE), serde_json::to_string_pretty(&state)?)?;
    Ok(())
}

/// Restore `model`'s adapters and `trainer`'s optimizer state from `dir`; returns the global step to continue from.
/// Every name and shape must match what this model expects, or the load is refused (nothing half-loaded).
pub fn load(dir: &Path, model: &mut TinyLlama, trainer: &mut Trainer) -> Res<usize> {
    let adapter = candle_core::safetensors::load(dir.join(ADAPTER_FILE), &Device::Cpu)?;
    let moments = candle_core::safetensors::load(dir.join(OPTIMIZER_FILE), &Device::Cpu)?;
    let state: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(STATE_FILE))?)?;
    let (vars, names) = trainable_params(model)?;

    // check everything before changing anything
    for (name, var) in names.iter().zip(vars.iter()) {
        for (key, map) in [(name.clone(), &adapter), (format!("{name}.exp_avg"), &moments), (format!("{name}.exp_avg_sq"), &moments)] {
            let t = map.get(&key).ok_or_else(|| format!("checkpoint is missing {key}"))?;
            if t.dims() != var.as_tensor().dims() {
                return Err(format!("{key}: shape {:?} in the checkpoint, {:?} in the model", t.dims(), var.as_tensor().dims()).into());
            }
        }
    }
    let global_step = state["global_step"].as_u64().ok_or("trainer_state.json: no global_step")? as usize;
    let adam_step = state["adam_step"].as_u64().ok_or("trainer_state.json: no adam_step")? as usize;

    let dev = model.device();
    for (name, var) in names.iter().zip(vars.iter()) {
        var.set(&adapter[name].to_device(&dev)?)?;
    }
    for (name, (_theta, m, v)) in names.iter().zip(trainer.optimizer.slots.iter()) {
        m.set(&moments[&format!("{name}.exp_avg")].to_device(&dev)?)?;
        v.set(&moments[&format!("{name}.exp_avg_sq")].to_device(&dev)?)?;
    }
    trainer.optimizer.step_t = adam_step;
    Ok(global_step)
}
