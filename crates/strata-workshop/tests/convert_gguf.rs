//! Step 6, part B: convert the tiny fixture to GGUF (F32) and open it with `strata_gguf`. This exercises the whole
//! mapping pipeline end-to-end — HF tensor-name map, q/k rotary permute, llama.* metadata, and tokenizer arrays — and
//! pins that what comes off the disk is a valid GGUF whose tensor names/shapes and metadata match what llama.cpp reads.
//! (The name map and the q/k permute bytes are covered by the unit tests in `convert.rs`; here we assert on the GGUF.)

use strata_gguf::Gguf;
use strata_workshop::convert::{convert, OutType};
use strata_workshop::testmodel::TinyLlama as Fixture;

/// A fresh temp dir, removed at the end of each test.
fn dir(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("am2-workshop-convert-gguf-{tag}-{nanos}"))
}

/// `strata_gguf` reads tensor dims in file order (innermost-first), the reverse of a tensor's numpy shape; flip them
/// back so we compare against the canonical [out, in] shapes the fixture writes.
fn numpy_dims(dims: &[u64]) -> Vec<u64> {
    let mut d = dims.to_vec();
    d.reverse();
    d
}

#[test]
fn converted_gguf_has_expected_tensors_metadata_and_tokenizer() -> Result<(), Box<dyn std::error::Error>> {
    let fdir = dir("ckpt");
    let written = Fixture::default().write(&fdir)?;
    let out = fdir.join("model.f32.gguf");

    convert(&written.dir, &out, OutType::F32, Some("gpt-2"))?;
    let g = Gguf::open(&out)?;

    // strata_gguf reads dims in file order (innermost-first), the reverse of a tensor's numpy shape; flip them back so
    // we compare against the canonical [out, in] shapes the fixture writes. BTreeMap by name keeps the assertions tidy.
    let shapes: std::collections::BTreeMap<String, Vec<u64>> = g
        .tensors
        .iter()
        .map(|t| (t.name.clone(), numpy_dims(&t.dims)))
        .collect();

    // 1 shared embed + 1 shared output-norm + 1 tied output head, then 9 tensors per layer x 2 layers.
    assert_eq!(g.tensors.len(), 3 + 2 * 9, "one tensor per HF weight, mapped to GGUF names");
    let dim = |name: &str| shapes.get(name).cloned().unwrap_or_default();

    // shared tensors keep their HF shapes (gguf names carry llama.cpp's .weight suffix).
    assert_eq!(dim("token_embd.weight"), vec![32u64, 16]);
    assert_eq!(dim("output_norm.weight"), vec![16]);
    assert_eq!(dim("output.weight"), vec![32u64, 16]);

    // block tensors: attn_q/k/v/output [hidden,hidden], ffn_gate/up [intermediate,hidden], ffn_down [hidden,intermediate].
    for n in 0..2 {
        assert_eq!(dim(&format!("blk.{n}.attn_norm.weight")), vec![16]);
        assert_eq!(dim(&format!("blk.{n}.attn_q.weight")), vec![16, 16]);
        assert_eq!(dim(&format!("blk.{n}.attn_k.weight")), vec![16, 16]);
        assert_eq!(dim(&format!("blk.{n}.attn_v.weight")), vec![16, 16]);
        assert_eq!(dim(&format!("blk.{n}.attn_output.weight")), vec![16, 16]);
        assert_eq!(dim(&format!("blk.{n}.ffn_norm.weight")), vec![16]);
        assert_eq!(dim(&format!("blk.{n}.ffn_gate.weight")), vec![32, 16]);
        assert_eq!(dim(&format!("blk.{n}.ffn_up.weight")), vec![32, 16]);
        assert_eq!(dim(&format!("blk.{n}.ffn_down.weight")), vec![16, 32]);
    }

    // metadata llama.cpp reads for a llama model (values from the fixture's config.json).
    assert_eq!(g.str("general.architecture"), Some("llama"));
    assert_eq!(g.u64("llama.context_length"), Some(512));
    assert_eq!(g.u64("llama.embedding_length"), Some(16));
    assert_eq!(g.u64("llama.block_count"), Some(2));
    assert_eq!(g.u64("llama.feed_forward_length"), Some(32));
    assert_eq!(g.u64("llama.attention.head_count"), Some(2));
    assert_eq!(g.u64("llama.attention.head_count_kv"), Some(2));
    let eps = g.metadata.get("llama.attention.layer_norm_rms_epsilon").unwrap().as_f64().unwrap();
    assert!((eps - 1e-5).abs() < 1e-9, "rms epsilon ~= 1e-5, got {eps}");
    // llama.cpp stores rope.freq_base as FLOAT32 (not UINT32), so read it back as a float like the rms epsilon above.
    let freq = g.metadata.get("llama.rope.freq_base").unwrap().as_f64().unwrap();
    assert!((freq - 10000.0).abs() < 1e-3, "rope freq base ~= 10000, got {freq}");
    assert_eq!(g.u64("llama.vocab_size"), Some(32));

    // tokenizer: model + sorted vocab + special-token flags + chat template. The tiny fixture is a BPE tokenizer.json, so
    // llama.cpp (and the reference GGUF) record "gpt2" here, not "llama".
    assert_eq!(g.str("tokenizer.ggml.model"), Some("gpt2"));
    let toks = g.metadata.get("tokenizer.ggml.tokens").unwrap().as_array().unwrap();
    assert_eq!(toks.len(), 32);
    assert_eq!(toks[0], serde_json::json!("tok_0"));
    assert_eq!(toks[31], serde_json::json!("tok_31"));
    let tt = g.metadata.get("tokenizer.ggml.token_type").unwrap().as_array().unwrap();
    assert_eq!(tt[0], serde_json::json!(3)); // bos (CONTROL)
    assert_eq!(tt[1], serde_json::json!(3)); // eos (CONTROL)
    assert_eq!(tt[2], serde_json::json!(1)); // a normal token
    assert_eq!(g.u64("tokenizer.ggml.bos_token_id"), Some(0));
    assert_eq!(g.u64("tokenizer.ggml.eos_token_id"), Some(1));
    assert!(g.str("tokenizer.chat_template").is_some() && !g.str("tokenizer.chat_template").unwrap().is_empty());

    std::fs::remove_dir_all(&fdir).ok();
    Ok(())
}
