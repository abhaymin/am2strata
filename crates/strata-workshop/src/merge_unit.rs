//! Pure-math checks for the merge fold — run before the disk/integration gates. They pin the two delta formulas (the
//! 2-D projection product and the fused-MoE expert einsum) against hand-written references, plus the name mapping and the
//! rsLoRA scaling factor that gate 4 keys off.

use super::{add_delta_2d, add_expert_delta, apply_scaling, peft_to_base_name};

/// Absolute element-wise max of two f32 slices (both equal length).
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
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
fn expert_einsum_matches_naive_loop() {
    // E=3 experts, out=4, in=5, rank r=2: random A flat [E,r,i], B flat [o,E,r], delta expected per-expert B_e @ A_e.
    let (e, o, i, r) = (3usize, 4usize, 5usize, 2usize);
    let scaling = 1.0 / r as f32; // ordinary LoRA scaling

    let mut seed = 0x5EED_u64;
    let rand = |state: &mut u64| -> f32 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (((z ^ (z >> 31)) >> 40) as f32) / 16_777_216.0 * 2.0 - 1.0
    };
    let a: Vec<f32> = (0..e * r * i).map(|_| rand(&mut seed)).collect();
    let b: Vec<f32> = (0..o * e * r).map(|_| rand(&mut seed)).collect();

    // the einsum path under test
    let (_w, got) = add_expert_delta(&vec![0f32; e * o * i], &a, &b, e, o, i, r, scaling);

    // an independent per-expert loop: for each expert, reshape B to [o,r] and A to [r,i], then matrix-multiply.
    let mut want = vec![0f32; e * o * i];
    for ee in 0..e {
        for oo in 0..o {
            for ii in 0..i {
                let mut acc = 0.0_f32;
                for rr in 0..r {
                    // B reshaped [out,r,E]: element[oo][rr][ee] lives at flat ((oo*r + rr)*E + ee)
                    let be = b[((oo * r) + rr) * e + ee];
                    // A reshaped [E,r,i]: element[ee][rr][ii] lives at flat (ee*r + rr)*i + ii
                    let ae = a[(ee * r + rr) * i + ii];
                    acc += be * ae;
                }
                want[ee * o * i + oo * i + ii] = scaling * acc;
            }
        }
    }

    assert!(max_abs(&got, &want) < 1e-6, "fused MoE einsum delta differs from the naive per-expert loop by {}", max_abs(&got, &want));
}
