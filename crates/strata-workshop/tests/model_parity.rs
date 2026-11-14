//! The parity gate of the Workshop's own forward pass: for a fixed token sequence its logits equal transformers'
//! (recorded once in `fixtures/forward-reference.json` by `fixtures/make-forward-reference.py`; no Python runs here).

use std::path::PathBuf;

use candle_core::Device;
use strata_workshop::model::{BaseDtype, Ctx, LlamaModel, LoraSpec, Target};

/// max abs difference allowed on F32 weights
const TOL: f32 = 1e-3;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn reference(name: &str) -> (Vec<u32>, Vec<Vec<f32>>) {
    let all: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixtures().join("forward-reference.json")).unwrap()).unwrap();
    let e = &all[name];
    let tokens = e["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
    let logits = e["logits"].as_array().unwrap().iter()
        .map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()).collect();
    (tokens, logits)
}

fn max_abs(got: &candle_core::Tensor, want: &[Vec<f32>]) -> f32 {
    let got: Vec<Vec<f32>> = got.to_vec2().unwrap();
    assert_eq!((got.len(), got[0].len()), (want.len(), want[0].len()), "logit shape");
    got.iter().flatten().zip(want.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
}

fn parity(name: &str, dtype: BaseDtype) -> f32 {
    let (tokens, want) = reference(name);
    let model = LlamaModel::load(&fixtures().join(name), dtype, &Device::Cpu).unwrap();
    let diff = max_abs(&model.forward(&tokens, &Ctx::eval()).unwrap(), &want);
    println!("{name} as {}: max abs logit difference {diff:.3e}", dtype.as_str());
    diff
}

#[test]
fn llama_two_layers_matches_transformers() {
    assert!(parity("tiny-llama-hf", BaseDtype::F32) < TOL);
}

#[test]
fn llama_sixteen_heads_matches_transformers() {
    assert!(parity("tiny-llama-256-hf", BaseDtype::F32) < TOL);
}

#[test]
fn grouped_query_attention_matches_transformers() {
    assert!(parity("tiny-llama-gqa-hf", BaseDtype::F32) < TOL);
}

#[test]
fn bf16_safetensors_match_transformers() {
    // the file is BF16; widened to f32 it holds the same values transformers computed with
    assert!(parity("tiny-llama-256-bf16-hf", BaseDtype::F32) < TOL);
    // and kept in 16 bits (widened per product) the result is the same: BF16 → f32 is exact
    assert!(parity("tiny-llama-256-bf16-hf", BaseDtype::BF16) < TOL);
}

#[test]
fn qwen2_biases_match_transformers() {
    assert!(parity("tiny-qwen2-hf", BaseDtype::F32) < TOL);
}

#[test]
fn qwen3_head_norms_and_head_dim_match_transformers() {
    assert!(parity("tiny-qwen3-hf", BaseDtype::F32) < TOL);
}

#[test]
fn tied_embeddings_match_transformers() {
    assert!(parity("tiny-llama-tied-hf", BaseDtype::F32) < TOL);
}

#[test]
fn half_precision_base_stays_close() {
    // rounding F32 weights to 16 bits moves the logits a little; this bounds how much on the tiny models
    for (name, dtype) in [("tiny-llama-256-hf", BaseDtype::F16), ("tiny-llama-256-hf", BaseDtype::BF16), ("tiny-qwen3-hf", BaseDtype::F16)] {
        let d = parity(name, dtype);
        assert!(d < 0.05, "{name} {dtype:?}: {d}");
    }
}

#[test]
fn a_peft_adapter_gives_pefts_logits() {
    // PEFT's own adapter (rsLoRA, r 4, alpha 8, all seven targets, random non-zero A and B) on the grouped-attention model
    let (tokens, want) = reference("tiny-gqa-peft-adapter");
    let mut model = LlamaModel::load(&fixtures().join("tiny-llama-gqa-hf"), BaseDtype::F32, &Device::Cpu).unwrap();
    let spec = model.load_adapter(&fixtures().join("tiny-gqa-peft-adapter")).unwrap();
    assert_eq!((spec.r, spec.alpha, spec.use_rslora), (4, 8.0, true));
    assert_eq!(spec.targets, Target::ALL.to_vec());
    assert!((spec.scaling() - 4.0).abs() < 1e-12, "rsLoRA: 8 / sqrt(4)");
    let diff = max_abs(&model.forward(&tokens, &Ctx::eval()).unwrap(), &want);
    println!("PEFT adapter: max abs logit difference {diff:.3e}");
    assert!(diff < TOL);
    // and the adapter matters: without it the logits are the base's, visibly different
    let (_, base) = reference("tiny-llama-gqa-hf");
    assert!(max_abs(&model.forward(&tokens, &Ctx::eval()).unwrap(), &base) > 0.01);
}

#[test]
fn a_fresh_adapter_changes_nothing_and_only_adapters_are_variables() {
    let (tokens, want) = reference("tiny-qwen3-hf");
    let mut model = LlamaModel::load(&fixtures().join("tiny-qwen3-hf"), BaseDtype::F32, &Device::Cpu).unwrap();
    let spec = LoraSpec { r: 4, alpha: 8.0, dropout: 0.0, use_rslora: true, targets: Target::ALL.to_vec(), train_layers: Some(2), seed: 1 };
    model.add_lora(&spec).unwrap();
    // B is zero: the logits are still transformers'
    assert!(max_abs(&model.forward(&tokens, &Ctx::eval()).unwrap(), &want) < TOL);
    // 3 blocks, the top 2 train, 7 targets, A and B each
    let vars = model.lora_vars();
    assert_eq!(vars.len(), 2 * 7 * 2);
    assert!(vars.iter().all(|(n, _)| n.contains(".layers.1.") || n.contains(".layers.2.")), "block 0 carries no adapter");
    assert_eq!(model.first_trained(), 1);
    let n: u64 = vars.iter().map(|(_, v)| v.as_tensor().elem_count() as u64).sum();
    assert_eq!(n, spec.trainable_params(&model.cfg));
    // the hidden state entering block 1 needs no gradient
    assert!(!model.lower(&tokens).unwrap().track_op());
    // a backward pass reaches every B (and A's gradient is zero while B is zero, as in PEFT)
    let sup: Vec<bool> = (0..tokens.len()).map(|p| p >= 4).collect();
    let grads = model.loss(&tokens, &sup, &Ctx::eval()).unwrap().backward().unwrap();
    let missing: Vec<&String> = vars.iter().filter(|(_, v)| grads.get(v).is_none()).map(|(n, _)| n).collect();
    assert!(missing.is_empty(), "no gradient for {missing:?}");
    for (name, var) in &vars {
        let g = grads.get(var).unwrap();
        let norm = g.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        if name.ends_with("lora_B.weight") {
            assert!(norm > 0.0, "{name} has a zero gradient");
        }
    }
}

#[test]
fn unsupported_models_are_refused_by_name() {
    use strata_workshop::model::ModelConfig;
    let base = serde_json::json!({"model_type": "llama", "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 2, "num_attention_heads": 2});
    assert!(ModelConfig::from_json(&base).is_ok());
    let with = |k: &str, v: serde_json::Value| { let mut c = base.clone(); c[k] = v; ModelConfig::from_json(&c).unwrap_err().to_string() };
    assert!(with("num_local_experts", 8.into()).contains("mixture-of-experts"));
    assert!(with("vision_config", serde_json::json!({})).contains("multimodal"));
    assert!(with("model_type", "gemma3".into()).contains("gemma3"));
    assert!(with("rope_scaling", serde_json::json!({"rope_type": "yarn"})).contains("yarn"));
    assert!(BaseDtype::from_s("4bit").unwrap_err().to_string().contains("QLoRA"));
}
