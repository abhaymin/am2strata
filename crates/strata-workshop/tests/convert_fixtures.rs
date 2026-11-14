//! Step 6, part A: the tiny synthetic Llama fixture (`testmodel::TinyLlama`) writes a real HuggingFace checkpoint
//! folder to disk — the `config.json`, `tokenizer_config.json`, `tokenizer.json` and F32 safetensors shards the step-6
//! converter reads. This pins that the fixture is exactly what the converter expects before any GGUF is written; no
//! downloads, no Python.

use strata_workshop::convert::read_tensors;
use strata_workshop::testmodel::TinyLlama as Fixture;

/// A fresh temp dir tagged so concurrent runs don't collide; removed at the end of each test.
fn dir(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("am2-workshop-convert-{tag}-{nanos}"))
}

/// The checkpoint written by `TinyLlama::write` is exactly the HF folder the converter loads: a `config.json`,
/// `tokenizer_config.json` and a readable set of F32 safetensors shards (one per configured shard), nothing else.
#[test]
fn fixture_writes_a_readable_llama_checkpoint() {
    let d = dir("ckpt");
    let written = Fixture::default().write(&d).expect("tiny fixture writes");

    // exactly the expected files, no more — the converter scans these by name/extension.
    assert!(d.join("config.json").is_file(), "config.json must exist on disk");
    assert!(d.join("tokenizer_config.json").is_file(), "tokenizer_config.json must exist on disk");
    assert!(d.join("tokenizer.json").is_file(), "tokenizer.json must exist on disk");
    let on_disk: Vec<_> = std::fs::read_dir(&d)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    assert_eq!(on_disk.len(), written.shard_files.len() + 3, "only shards + config/tokenizer files");

    // config.json is a real LlamaForCausalLM config with every hyper-parameter the converter reads from it.
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(d.join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg["architectures"][0], "LlamaForCausalLM");
    for k in ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size",
        "rms_norm_eps", "vocab_size", "max_position_embeddings", "rope_theta", "tie_word_embeddings"]
    {
        assert!(cfg.get(k).is_some(), "config.json is missing {k}");
    }

    // tokenizer_config.json carries the special tokens and chat template the converter and llama.cpp both read.
    let tc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(d.join("tokenizer_config.json")).unwrap()).unwrap();
    assert!(tc.get("bos_token").is_some(), "tokenizer_config.json must have bos_token");
    assert!(tc.get("eos_token").is_some(), "tokenizer_config.json must have eos_token");
    assert!(tc.get("chat_template").is_some(), "tokenizer_config.json must have a chat template");

    // the shards are real, readable F32 tensors — what read_tensors consumes.
    let tensors = read_tensors(&d).expect("F32 shards read back");
    assert!(!tensors.is_empty(), "the fixture writes at least one tensor");
    for name in ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"] {
        assert!(tensors.contains_key(name), "checkpoint must contain {name}");
    }

    std::fs::remove_dir_all(&d).ok();
}
