//! Typed key/values round-trip through `strata_gguf` (an independent reader): numbers stay numbers, token lists stay
//! typed arrays, so a converted model's metadata reads as llama.cpp's converter writes it. The plain-string `create`
//! must keep writing exactly what it always did (the MTP draft writer relies on it).

use strata_pack::gguf_write::{Kv, TensorSpec, Writer};

fn write(path: &std::path::Path, kv: &[(&str, Kv)]) {
    let raw: Vec<u8> = (0..4u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let mut w = Writer::create_typed(path, kv, vec![TensorSpec { name: "t".into(), shape: vec![4], ggml_type: 0, nbytes: 16 }]).unwrap();
    w.tensor_data(&raw).unwrap();
    w.end_tensor().unwrap();
    w.finish().unwrap();
}

#[test]
fn typed_values_read_back_with_their_types() {
    let p = std::env::temp_dir().join(format!("am2-gguf-kv-{}.gguf", std::process::id()));
    write(&p, &[
        ("general.architecture", Kv::Str("llama".into())),
        ("llama.block_count", Kv::U32(2)),
        ("llama.rope.freq_base", Kv::F32(10000.0)),
        ("tokenizer.ggml.add_bos_token", Kv::Bool(true)),
        ("tokenizer.ggml.tokens", Kv::Strings(vec!["<s>".into(), "a".into(), "é".into()])),
        ("tokenizer.ggml.token_type", Kv::Ints(vec![3, 1, 1])),
        ("tokenizer.ggml.scores", Kv::Floats(vec![0.0, -1.5, -2.25])),
        ("some.signed", Kv::I32(-7)),
    ]);
    let g = strata_gguf::Gguf::open(&p).unwrap();
    let m = &g.metadata;
    assert_eq!(m["general.architecture"], "llama");
    assert_eq!(m["llama.block_count"], 2);
    assert_eq!(m["llama.rope.freq_base"].as_f64(), Some(10000.0));
    assert_eq!(m["tokenizer.ggml.add_bos_token"], true);
    assert_eq!(m["tokenizer.ggml.tokens"], serde_json::json!(["<s>", "a", "é"]));
    assert_eq!(m["tokenizer.ggml.token_type"], serde_json::json!([3, 1, 1]));
    assert_eq!(m["tokenizer.ggml.scores"], serde_json::json!([0.0, -1.5, -2.25]));
    assert_eq!(m["some.signed"], -7);
    assert_eq!(g.tensors.len(), 1);
    assert_eq!(g.tensors[0].name, "t");
    let _ = std::fs::remove_file(&p);
}

#[test]
fn string_create_is_unchanged() {
    // the bytes the plain-string writer has always produced: magic, version 3, 0 tensors, 1 kv, then key + STRING(8) + value
    let p = std::env::temp_dir().join(format!("am2-gguf-str-{}.gguf", std::process::id()));
    Writer::create(&p, &[("general.architecture", "x".to_string())], vec![]).unwrap().finish().unwrap();
    let got = std::fs::read(&p).unwrap();
    let mut want: Vec<u8> = Vec::new();
    want.extend(b"GGUF"); want.extend(3u32.to_le_bytes()); want.extend(0u64.to_le_bytes()); want.extend(1u64.to_le_bytes());
    want.extend(20u64.to_le_bytes()); want.extend(b"general.architecture"); want.extend(8u32.to_le_bytes());
    want.extend(1u64.to_le_bytes()); want.extend(b"x");
    want.extend(vec![0u8; (32 - want.len() % 32) % 32]);
    assert_eq!(got, want);
    let _ = std::fs::remove_file(&p);
}
