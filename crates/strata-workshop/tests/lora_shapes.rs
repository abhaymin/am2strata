//! Step 3, part A: LoRA layers on a frozen base. These pin the three things the adapters must get right before training —
//! the PEFT tensor names/shapes (`...q_proj.lora_{A,B}.weight`), the rsLoRA/normal scaling factor, and that a trained
//! adapter's output is exactly `base + scaling * (B A) x`. No downloads; the base comes from the tiny fixture's tensors.

use std::collections::BTreeMap;

use candle_core::{Device, Tensor};
use strata_workshop::lora::{LoraConfig, LoraLinear, TinyLlama};
use strata_workshop::testmodel::TinyLlama as Fixture;

fn base_tensors(dev: &Device) -> BTreeMap<String, Tensor> {
    // the tiny fixture's tensors, laid out by name exactly as an HF checkpoint (and as `TinyLlama::load` reads them)
    Fixture::default().generate().into_iter().map(|t| (t.name, Tensor::from_vec(t.values, t.shape.clone(), dev).unwrap())).collect()
}

/// Absolute difference of two f32 buffers, element-wise max.
fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let va: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
    let vb: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(va.len(), vb.len());
    va.iter().zip(&vb).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

#[test]
fn scaling_is_alpha_over_r_or_alpha_over_sqrt_r() {
    // the default adapter: r = 4, lora_alpha = 8
    let cfg = LoraConfig::default();
    assert_eq!(cfg.scaling(), 2.0, "ordinary LoRA scales by alpha / r");

    let mut rs = cfg;
    rs.use_rslora = true;
    assert!((rs.scaling() - 4.0).abs() < 1e-6, "rsLoRA scales by alpha / sqrt(r) = 8 / 2");
}

#[test]
fn adapter_names_shapes_and_order_are_peft() {
    let dev = Device::Cpu;
    let cfg = LoraConfig { r: 8, lora_alpha: 16.0, use_rslora: false, target_modules: vec!["q_proj".into(), "k_proj".into(), "v_proj".into(), "o_proj".into()] };
    let model = TinyLlama::load(&base_tensors(&dev), &cfg, &dev).unwrap();

    // the exact tensor names PEFT expects (`base_model.model.model.layers.{i}.self_attn.{m}_proj.lora_{A,B}.weight`)
    // and shapes A:[r, in] = [8, 16], B:[out, r] = [16, 8], layer by layer, module q/k/v/o, A before B.
    let mut expected = Vec::new();
    for li in 0..model.num_layers {
        for m in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            expected.push((format!("base_model.model.model.layers.{li}.self_attn.{m}.lora_A.weight"), (8usize, 16usize)));
            expected.push((format!("base_model.model.model.layers.{li}.self_attn.{m}.lora_B.weight"), (16usize, 8usize)));
        }
    }
    let snap = model.snapshot().unwrap();
    assert_eq!(snap.len(), expected.len());
    for ((en, es), (gn, gs, _vals)) in expected.iter().zip(snap.iter()) {
        assert_eq!(en, gn, "adapter name mismatch (got {gn}, want {en})");
        assert_eq!(*es, *gs, "adapter shape mismatch for {en}");
    }

    // and the config we serialize carries the same hyper-parameters PEFT records in adapter_config.json
    assert_eq!(cfg.r, 8);
    assert_eq!(cfg.lora_alpha, 16.0);
    assert!(!cfg.use_rslora);
    assert_eq!(cfg.target_modules, vec!["q_proj", "k_proj", "v_proj", "o_proj"]);
}

#[test]
fn lora_output_is_base_plus_scaled_delta() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let cfg = LoraConfig::default(); // r = 4, alpha = 8 => scaling 2.0
    let (in_, out_, r, seq) = (16usize, 16usize, 4usize, 3usize);

    // a constant base weight and an input; with B untrained (all zero) the adapter must contribute nothing
    let base_w = Tensor::from_vec(vec![0.01_f32; out_ * in_], (out_, in_), &dev).unwrap();
    let x = Tensor::from_vec((0..(seq * in_) as usize).map(|i| i as f32).collect(), (seq, in_), &dev).unwrap();

    let mut lora = LoraLinear::new(base_w.clone(), &cfg, &dev).unwrap();
    let base_out = x.matmul(&base_w.t()?).unwrap();
    assert!(max_abs_diff(&lora.forward(&x)?, &base_out) < 1e-4, "untrained adapter (B=0) must equal the frozen base");

    // set a known B; A stays at its 0.5 initial value. The forward must be `base + scaling * (B A) x`.
    let b_flat: Vec<f32> = (0..(out_ * r) as usize).map(|i| ((i % 5) as i32 - 2) as f32).collect();
    lora.set_b_values(&b_flat).unwrap();

    let a = Tensor::from_vec(vec![0.5_f32; r * in_], (r, in_), &dev).unwrap();
    let b = Tensor::from_vec(b_flat, (out_, r), &dev).unwrap();
    let ba = b.matmul(&a)?; // [out, in]
    let exp_delta = (x.matmul(&ba.t()?)? * cfg.scaling() as f64)?;
    let expected = base_out.add(&exp_delta)?;

    assert!(max_abs_diff(&lora.forward(&x)?, &expected) < 1e-4, "trained output must equal base + scaling*(B A)x");
    Ok(())
}

#[test]
fn snapshot_round_trips_through_load_snapshot() {
    let dev = Device::Cpu;
    let cfg = LoraConfig { r: 8, lora_alpha: 16.0, use_rslora: true, target_modules: vec!["q_proj".into(), "k_proj".into(), "v_proj".into(), "o_proj".into()] };

    // drive A and B away from their initial values so there is something to restore, then snapshot it in PEFT order
    let mut model = TinyLlama::load(&base_tensors(&dev), &cfg, &dev).unwrap();
    for li in 0..model.num_layers {
        for lin in &mut model.attn[li] {
            let mut a_vals: Vec<f32> = (0..(cfg.r * 16) as usize).map(|i| ((i % 7) as i32 - 3) as f32).collect();
            for v in &mut a_vals {
                *v *= -1.0;
            }
            lin.a.set(candle_core::Var::from_slice(&a_vals, (cfg.r, 16), &dev).unwrap().as_tensor()).unwrap();
        }
    }
    let saved = model.snapshot().unwrap();

    // a fresh model with the same seed has identical A; restoring restores B exactly and leaves A's values intact
    let mut restored = TinyLlama::load(&base_tensors(&dev), &cfg, &dev).unwrap();
    restored.load_snapshot(&saved).unwrap();

    let after = restored.snapshot().unwrap();
    assert_eq!(after.len(), saved.len());
    for ((sn, ss, sv), (rn, rs, rv)) in saved.iter().zip(after.iter()) {
        assert_eq!(sn, rn, "name mismatch on restore");
        assert_eq!(ss, rs, "shape mismatch on restore");
        assert_eq!(sv.len(), rv.len());
        for (a, b) in sv.iter().zip(rv.iter()) {
            assert!((a - b).abs() < 1e-6, "value drift on restore for {sn}");
        }
    }
}
