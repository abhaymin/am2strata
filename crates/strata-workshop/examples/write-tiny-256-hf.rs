//! Generates the wide tiny HF checkpoint under `tests/fixtures/tiny-llama-256-hf/` the same deterministic way
//! [`testmodel::TinyLlama`] writes the first workshop fixture, so its `config.json`, tokenizer and safetensors are real
//! files (and byte-for-byte reproducible) that llama.cpp can convert and quantize. Sizes: hidden 256, intermediate 512,
//! one layer, vocab 32 — wide enough that every 2-D weight holds whole 32- and 256-element blocks, so `llama-quantize`
//! actually quantizes instead of falling back to F16. Run once to (re)write the committed folder:
//!
//!     cargo run -p strata-workshop --example write-tiny-256-hf

use std::path::PathBuf;

use strata_workshop::testmodel::TinyLlama as Fixture;

fn main() -> anyhow::Result<()> {
    let wide = Fixture {
        seed: 0x5EED,
        num_layers: 1,
        vocab_size: 32,
        hidden: 256,
        intermediate: 512,
        num_attention_heads: 16,
        num_key_value_heads: 16,
        shard_count: 1,
    };
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join("tiny-llama-256-hf");
    let written = wide.write(&dir)?;
    println!("wrote tiny-llama-256-hf checkpoint to {}", written.dir.display());
    Ok(())
}
