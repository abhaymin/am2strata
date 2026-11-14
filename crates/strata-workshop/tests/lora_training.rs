//! Step 3, part B — the training loop. Gates: (1) cosine-with-warm-up schedule has the right warm-up ramp and decay to
//! ~0 at the end; (2) masked next-token cross-entropy excludes the prompt positions from its average; (3) 5+ AdamW steps
//! on 20 lines drives the masked loss down. No downloads — the frozen base comes from the tiny fixture's tensors and the
//! tokens are synthetic but deterministic.

use std::collections::BTreeMap;

use candle_core::{Device, Tensor};
use strata_workshop::lora::{LoraConfig, TinyLlama};
use strata_workshop::testmodel::TinyLlama as Fixture;
use strata_workshop::train::{cosine_with_warmup_lr, masked_loss_scalar, make_train_data, Trainer, TrainingConfig};

fn base_tensors(dev: &Device) -> BTreeMap<String, Tensor> {
    Fixture::default().generate().into_iter().map(|t| (t.name, Tensor::from_vec(t.values, t.shape.clone(), dev).unwrap())).collect()
}

#[test]
fn schedule_warms_up_then_decays_to_zero() {
    let peak = 0.1;
    let warmup = 5;
    let total = 20;
    // linear ramp: 0 at step 0, fraction of peak mid-warm-up
    assert!((cosine_with_warmup_lr(peak, 0, warmup, total)).abs() < 1e-9, "step 0 starts at 0");
    assert!((cosine_with_warmup_lr(peak, 4, warmup, total) - peak * 4.0 / 5.0).abs() < 1e-6, "warm-up ramps toward peak");
    // end of warm-up is the peak (cosine progress == 0 => factor 1)
    assert!((cosine_with_warmup_lr(peak, warmup, warmup, total) - peak).abs() < 1e-9, "peak at end of warm-up");
    // cosine envelope returns to ~0 at the final step
    let last = cosine_with_warmup_lr(peak, total, warmup, total);
    assert!(last.abs() < 1e-6, "cosine decays to ~0 at the last step (got {last})");
}

#[test]
fn masked_loss_excludes_prompt_positions() -> Result<(), Box<dyn std::error::Error>> {
    // With the whole sequence as the prompt (prompt_len == seq_len) every position is masked, so the per-sequence loss
    // must be exactly 0 regardless of the logits; a smaller prompt yields a non-zero mean. The base is random so a full
    // mask cannot produce a positive mean over zero positions — this pins the masking.
    let dev = Device::Cpu;
    let cfg = LoraConfig { r: 8, lora_alpha: 16.0, use_rslora: false, target_modules: vec!["q_proj".into(), "k_proj".into(), "v_proj".into(), "o_proj".into()] };
    let model = TinyLlama::load(&base_tensors(&dev), &cfg, &dev).unwrap();
    let data = make_train_data(1, 3, /* seq_len */ 6, /* vocab */ 16, /* prompt_len */ 6);

    // forward a row and mask it fully — the reported scalar loss must be 0.
    let tokens = Tensor::from_vec(data.tokens[0].clone(), (data.seq_len,), &dev).unwrap();
    let logits = model.forward(&tokens)?;
    let full = masked_loss_scalar(&logits, &data.tokens[0], data.seq_len, &dev)?;
    assert!(full.abs() < 1e-6, "full prompt masks out the whole loss (got {full})");
    // a short prompt leaves real (positive, finite) supervision over the rest.
    let some = masked_loss_scalar(&logits, &data.tokens[0], 2, &dev)?;
    assert!(some > 0.0 && some.is_finite(), "masked loss over unprompted positions is a positive finite number (got {some})");
    Ok(())
}

#[test]
fn masked_loss_counts_exactly_the_answer_tokens() -> Result<(), Box<dyn std::error::Error>> {
    // As PEFT/HF with prompt labels at -100: position p predicts token p+1; only targets at index >= prompt_len count,
    // i.e. positions prompt_len-1 ..= seq_len-2. The last position has no target (no wrap-around to token 0).
    let dev = Device::Cpu;
    let model = TinyLlama::load(&base_tensors(&dev), &LoraConfig::default(), &dev).unwrap();
    let data = make_train_data(3, 1, /* seq_len */ 7, /* vocab */ 16, /* prompt_len */ 3);
    let toks = &data.tokens[0];
    let logits = model.forward(&Tensor::from_vec(toks.clone(), (toks.len(),), &dev)?)?;
    let logp = candle_nn::ops::log_softmax(&logits, 1)?.to_vec2::<f32>()?;
    let by_hand: f64 = (2..=5).map(|p| -(logp[p][toks[p + 1] as usize] as f64)).sum::<f64>() / 4.0;
    let got = masked_loss_scalar(&logits, toks, 3, &dev)?;
    assert!((got - by_hand).abs() < 1e-5, "masked loss {got} != mean NLL over positions 2..=5 ({by_hand})");
    Ok(())
}

#[test]
fn training_over_five_steps_falls() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    // r == hidden gives full-rank adapters, so the LoRA block has enough capacity to fit the (learnable) next-token
    // signal on these 20 short lines within a handful of AdamW steps.
    let cfg = LoraConfig { r: 16, lora_alpha: 8.0, use_rslora: false, target_modules: vec!["q_proj".into(), "k_proj".into(), "v_proj".into(), "o_proj".into()] };
    let train_cfg = TrainingConfig { lr: 0.1, warmup_steps: 3, total_steps: 20, accumulation_steps: 2 };

    let model = TinyLlama::load(&base_tensors(&dev), &cfg, &dev).unwrap();
    let data = make_train_data(7, /* num_sequences */ 20, /* seq_len */ 8, /* vocab */ 32, /* prompt_len */ 2);

    let mut trainer = Trainer::new(&model, train_cfg)?;
    let n_steps = 8usize;
    let mut losses = Vec::with_capacity(n_steps);
    for step in 0..n_steps {
        let l = trainer.train_step(&model, &data, step)?;
        losses.push(l);
        eprintln!("step {step} lr={} loss={l:.5}", trainer.learning_rate());
    }

    // the training signal must fall: final is well below the start, and mid-run already below the start.
    let first = losses[0];
    assert!(losses[n_steps - 1] < first * 0.75, "final loss {}/{} did not fall enough", losses[n_steps - 1], first);
    assert!(losses[4] < first, "loss should be dropping by step 4 ({} vs {})", losses[4], first);
    Ok(())
}
