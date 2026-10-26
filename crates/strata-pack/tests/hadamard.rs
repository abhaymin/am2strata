//! A GGUF with Hadamard-folded experts (PRISM, `prism.hadamard.version`) belongs to Strata 0.2, whose pack format is
//! native_experts.txt v5 with hadamard.txt: iq-pack (Strata 0.1.45's, as PolyStrata ships it) refuses it by name.

fn gguf(path: &std::path::Path) {
    let mut b = Vec::new();
    let s = |b: &mut Vec<u8>, t: &str| {
        b.extend((t.len() as u64).to_le_bytes());
        b.extend(t.as_bytes());
    };
    b.extend(b"GGUF");
    b.extend(3u32.to_le_bytes());
    b.extend(1u64.to_le_bytes()); // one tensor
    b.extend(2u64.to_le_bytes()); // two key/values
    s(&mut b, "general.architecture");
    b.extend(8u32.to_le_bytes());
    s(&mut b, "qwen35moe");
    s(&mut b, "prism.hadamard.version");
    b.extend(4u32.to_le_bytes());
    b.extend(1u32.to_le_bytes());
    s(&mut b, "token_embd.weight");
    b.extend(1u32.to_le_bytes());
    b.extend(32u64.to_le_bytes());
    b.extend(0u32.to_le_bytes()); // F32
    b.extend(0u64.to_le_bytes());
    while b.len() % 32 != 0 {
        b.push(0);
    }
    b.extend(vec![0u8; 128]);
    std::fs::write(path, b).unwrap();
}

#[test]
fn hadamard_folded_models_are_refused_by_name() {
    let d = std::env::temp_dir().join(format!("am2-hadamard-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let g = d.join("prism.gguf");
    gguf(&g);
    let o = strata_pack::iq_pack::Options { gguf: g, out: d.join("pack"), base: None, skip_experts: false, compat_bf16: false,
                                            experts_bin: false };
    let e = strata_pack::iq_pack::run(&o).unwrap_err();
    assert!(e.contains("Hadamard") && e.contains("Strata 0.2"), "{e}");
    let _ = std::fs::remove_dir_all(&d);
}
