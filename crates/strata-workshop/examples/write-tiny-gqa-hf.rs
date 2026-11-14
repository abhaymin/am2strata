//! Generates the grouped-attention tiny HF checkpoint under `tests/fixtures/tiny-llama-gqa-hf/` the same deterministic way
//! [`testmodel::TinyLlama`] writes the other workshop fixtures, so its `config.json`, tokenizer and safetensors are real
//! files (byte-for-byte reproducible) that llama.cpp can convert. Sizes: 16 attention heads, 4 kv heads (GQA), hidden
//! 256 (head_dim 16, so the kv dim is 4*16 = 64), intermediate 512, one layer, vocab 32 - enough that every 2-D weight
//! holds whole blocks and k/v projections carry fewer rows than q. Run once to (re)write the checkpoint the reviewer's
//! `convert_hf_to_gguf.py` reads:
//!
//!     cargo run -p strata-workshop --example write-tiny-gqa-hf

use std::path::PathBuf;

use strata_workshop::testmodel::TinyLlama as Fixture;

fn main() -> anyhow::Result<()> {
    let gqa = Fixture {
        seed: 0x5EED,
        num_layers: 1,
        vocab_size: 32,
        hidden: 256,
        intermediate: 512,
        num_attention_heads: 16,
        num_key_value_heads: 4,
        shard_count: 1,
    };
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join("tiny-llama-gqa-hf");
    let written = gqa.write(&dir)?;
    println!("wrote tiny-llama-gqa-hf checkpoint (16 attn / 4 kv heads) to {}", written.dir.display());
    Ok(())
}
