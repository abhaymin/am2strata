//! Pure-math checks for the merge fold — run before the disk/integration gates. They pin the two delta formulas (the
//! 2-D projection product and the fused-MoE expert einsum) against references, plus the name mapping and the rsLoRA
//! scaling factor that gate 4 keys off. The expert einsum is checked against an independent PyTorch/PEFT reference
//! (`tests/fixtures/peft-expert-delta.safetensors`), not a re-typing of the code — see lesson in TASK-8h.

use super::{add_delta_2d, add_expert_delta, apply_scaling, peft_to_base_name};

use candle_core::{Device, Tensor};
use std::collections::HashMap;

/// Absolute element-wise max of two f32 slices (both equal length).
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

/// Every tensor in the reviewer's PyTorch/PEFT reference fixture, keyed by name — read once per test.
fn peft_reference() -> HashMap<String, Tensor> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/peft-expert-delta.safetensors");
    candle_core::safetensors::load(&path, &Device::Cpu).unwrap()
}

/// A fixture tensor flattened to f32 (row-major), the order PEFT's einsum writes into.
fn reference(name: &str) -> Vec<f32> {
    peft_reference().get(name).and_then(|t| t.flatten_all().ok()).and_then(|v| v.to_vec1::<f32>().ok())
        .unwrap_or_else(|| panic!("{name}: missing from the PEFT reference fixture"))
}

#[test]
fn peft_name_maps_to_base_weight() {
    assert_eq!(peft_to_base_name("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight").as_deref(), Some("model.layers.0.self_attn.q_proj.weight"));
    assert_eq!(peft_to_base_name("base_model.model.model.layers.1.self_attn.o_proj.lora_B.weight").as_deref(), Some("model.layers.1.self_attn.o_proj.weight"));
    // a plain base weight, carrying no LoRA suffix, is not mapped.
    assert!(peft_to_base_name("model.embed_tokens.weight").is_none());
}

#[test]
fn rslora_and_normal_scaling_factors() {
    assert!((apply_scaling(8.0, 4, false) - 2.0).abs() < 1e-6, "ordinary LoRA: alpha / r"); // 8/4
    assert!((apply_scaling(8.0, 4, true) - 4.0).abs() < 1e-6, "rsLoRA: alpha / sqrt(r)"); // 8/2
}

#[test]
fn d2_delta_is_scaled_ba() {
    // out=3, in=2, r=1 keeps the product visible by hand: B:[3,1], A:[1,2]; BA is a 3x2 with scaling folded in.
    let (w, delta) = add_delta_2d(&vec![0f32; 6], &[2.0, -1.0], &[-1.0, 4.0, 0.5], 3, 2, 1, 0.5);
    // B A row-major then *0.5: [[-2,1],[-8,4],[1,-0.5]]*0.5 = [[-1,0.5],[4,-2],[0.5,-0.25]]
    let expected: Vec<f32> = vec![-1.0, 0.5, 4.0, -2.0, 0.5, -0.25];
    assert!(max_abs(&delta, &expected) < 1e-6, "d2 delta {delta:?} != hand product {expected:?}");
    for (wi, ei) in w.iter().zip(expected) {
        assert!((wi - ei).abs() < 1e-6, "folded weight equals the zero base + delta");
    }
}

#[test]
fn expert_einsum_matches_peft_default_layout() {
    // The identical adapter (lora_A [E*r,in]=[6,5], lora_B [out,r*E]=[4,6]) must fold into a default-layout target of
    // shape [E, in, out] = [3, 5, 4] exactly as PEFT's `einsum("o r e, e r i -> e i o")` produced `delta_eio`.
    let a = reference("lora_A");
    let b = reference("lora_B");
    let scaling = reference("scaling")[0];
    assert!((scaling - 0.75).abs() < 1e-6, "fixture scaling is the reviewer's 0.75");

    let (_, delta) = add_expert_delta("delta_eio", &vec![0f32; 60], &a, &b, 3, /*d1=*/5, /*d2=*/4, scaling).unwrap();
    assert!(max_abs(&delta, &reference("delta_eio")) < 1e-6, "default e-i-o expert delta differs from PEFT by {}", max_abs(&delta, &reference("delta_eio")));
}

#[test]
fn expert_einsum_matches_peft_swapped_layout() {
    // Same adapter against a swapped-layout target of shape [E, out, in] = [3, 4, 5]; PEFT's `einsum("o r e, e r i -> e o i")`
    // is the only layout whose (a_in,b_out)==(d2,d1) match here, folding into exactly `delta_eoi`.
    let a = reference("lora_A");
    let b = reference("lora_B");
    let scaling = reference("scaling")[0];

    let (_, delta) = add_expert_delta("delta_eoi", &vec![0f32; 60], &a, &b, 3, /*d1=*/4, /*d2=*/5, scaling).unwrap();
    assert!(max_abs(&delta, &reference("delta_eoi")) < 1e-6, "swapped e-o-i expert delta differs from PEFT by {}", max_abs(&delta, &reference("delta_eoi")));
}

#[test]
fn square_expert_parameter_is_refused_not_guess() {
    // A [3, 5, 5] target is square: with a_in=b_out=5 the two expert layouts are indistinguishable from the shapes alone,
    // so add_expert_delta must refuse rather than pick one.
    let a = reference("lora_A");
    let b = reference("lora_B");
    let scaling = reference("scaling")[0];

    let err = add_expert_delta("square_param", &vec![0f32; 75], &a, &b, 3, /*d1=*/5, /*d2=*/5, scaling).unwrap_err();
    assert!(err.to_string().contains("square"), "square target should be refused with a clear message, got: {err}");
}
