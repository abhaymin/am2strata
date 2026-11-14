//! Tiny synthetic Llama-shaped safetensors fixture + a `tokenizer.json`, built deterministically from a fixed seed — no
//! downloads and no Python. Every workshop step's test reuses this to get a small, well-shaped model on disk, so the
//! round-trips exercise real code paths. The golden references that do need Python or llama.cpp (the
//! `convert_hf_to_gguf` output of this tiny model, the `llama-quantize` bytes) are made once by the reviewer and
//! committed under `tests/fixtures/` with a README; tests only read them.

use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use serde_json::Value as Json;
use serde_json::json;
use safetensors::tensor as st;

/// A tiny random Llama (2 layers by default) laid out the way a real checkpoint's tensors are named. All weights are
/// f32 and generated from `seed` so a round-trip read compares against identical values. Sizes scale the fixture up or
/// down but never download anything.
#[derive(Debug, Clone)]
pub struct TinyLlama {
    pub seed: u64,
    /// how many layers; the plan's "2-layer" proof uses 2
    pub num_layers: usize,
    pub vocab_size: usize,
    pub hidden: usize,
    pub intermediate: usize,
    /// attention heads (hidden is a multiple of these), written to config.json and used for the rotary q/k permute
    pub num_attention_heads: usize,
    /// GQA key/value heads (defaults to full attention here)
    pub num_key_value_heads: usize,
    /// how many shard files to write (1 = one file; >=2 lets step 1 exercise shard enumeration)
    pub shard_count: usize,
}

impl Default for TinyLlama {
    fn default() -> Self {
        Self { seed: 0x5EED, num_layers: 2, vocab_size: 32, hidden: 16, intermediate: 32, num_attention_heads: 2, num_key_value_heads: 2, shard_count: 1 }
    }
}

/// One generated tensor: its name, shape and raw f32 contents (row-major).
#[derive(Debug, Clone)]
pub struct TensorData {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl TinyLlama {
    /// The canonical tensor list, in a fixed order. `write` and the round-trip test's "expected" values come from here,
    /// so a read always matches what was written regardless of on-disk ordering.
    pub fn generate(&self) -> Vec<TensorData> {
        let mut gen = SplitMix64::new(self.seed);
        let mut out = Vec::new();
        // realistic scales, so the model's outputs can move: norm gains 1.0 (as Llama initialises them), embeddings
        // about ±1, projections about ±1/sqrt(fan-in). Everything at ±0.02, norm gains included, made the tied head's
        // logits ~0.006, i.e. a uniform output no adapter could change.
        let w = 1.0 / (self.hidden as f32).sqrt();
        // embeddings + output norm are shared across layers
        out.push(TensorData { name: "model.embed_tokens.weight".into(), shape: vec![self.vocab_size, self.hidden], values: mat(&mut gen, self.vocab_size, self.hidden, 1.0) });
        for layer in 0..self.num_layers {
            let p = |key: &str| format!("model.layers.{layer}.{key}");
            out.push(TensorData { name: p("input_layernorm.weight"), shape: vec![self.hidden], values: vec![1.0; self.hidden] });
            out.push(TensorData { name: p("self_attn.q_proj.weight"), shape: vec![self.hidden, self.hidden], values: mat(&mut gen, self.hidden, self.hidden, w) });
            out.push(TensorData { name: p("self_attn.k_proj.weight"), shape: vec![self.hidden, self.hidden], values: mat(&mut gen, self.hidden, self.hidden, w) });
            out.push(TensorData { name: p("self_attn.v_proj.weight"), shape: vec![self.hidden, self.hidden], values: mat(&mut gen, self.hidden, self.hidden, w) });
            out.push(TensorData { name: p("self_attn.o_proj.weight"), shape: vec![self.hidden, self.hidden], values: mat(&mut gen, self.hidden, self.hidden, w) });
            out.push(TensorData { name: p("post_attention_layernorm.weight"), shape: vec![self.hidden], values: vec![1.0; self.hidden] });
            out.push(TensorData { name: p("mlp.gate_proj.weight"), shape: vec![self.intermediate, self.hidden], values: mat(&mut gen, self.intermediate, self.hidden, w) });
            out.push(TensorData { name: p("mlp.up_proj.weight"), shape: vec![self.intermediate, self.hidden], values: mat(&mut gen, self.intermediate, self.hidden, w) });
            out.push(TensorData { name: p("mlp.down_proj.weight"), shape: vec![self.hidden, self.intermediate], values: mat(&mut gen, self.hidden, self.intermediate, 1.0 / (self.intermediate as f32).sqrt()) });
        }
        out.push(TensorData { name: "model.norm.weight".into(), shape: vec![self.hidden], values: vec![1.0; self.hidden] });
        // a distinct output head (tie_word_embeddings = false in config.json), so the converter writes an `output` tensor
        out.push(TensorData { name: "lm_head.weight".into(), shape: vec![self.vocab_size, self.hidden], values: mat(&mut gen, self.vocab_size, self.hidden, 1.0 / (self.hidden as f32).sqrt()) });
        out
    }

    /// Write the fixture into `dir`: `<count>` safetensors shards named like an HF checkpoint, a real `config.json` and
    /// `tokenizer_config.json`, plus a real `tokenizer.json`. Returns the shard files (in order). The config and
    /// tokenizer are exactly what the step-6 converter reads to build the GGUF and what llama.cpp's converter produced
    /// the reference from; they are byte-identical to `tests/fixtures/tiny-llama-hf/`.
    pub fn write(&self, dir: &Path) -> anyhow::Result<ShardFiles> {
        let tensors = self.generate();
        std::fs::create_dir_all(dir)?;
        let shards = partition(&tensors, self.shard_count);
        let mut shard_files = Vec::new();
        for (i, chunk) in shards.iter().enumerate() {
            let name = format!("model-{:05}-of-{:05}.safetensors", i + 1, shards.len());
            write_shard(dir.join(&name), chunk)?;
            shard_files.push(dir.join(name));
        }
        write_config_json(dir, &self.hf_config())?;
        write_tokenizer_config_json(dir)?;
        write_tokenizer_json(dir, self.vocab_size)?;
        Ok(ShardFiles { dir: dir.to_path_buf(), shard_files })
    }

    /// The config.json contents this fixture writes, as a serde_json value — so the converter can be tested against it.
    pub fn hf_config(&self) -> serde_json::Value {
        json!({
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": self.hidden,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate,
            "rms_norm_eps": 1e-5,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": 512,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
        })
    }
}

/// Where a written fixture landed.
pub struct ShardFiles {
    pub dir: PathBuf,
    /// all shards in order (1 entry when `shard_count == 1`)
    pub shard_files: Vec<PathBuf>,
}

/// Partition `tensors` into `n` contiguous chunks (the last may hold the remainder). One safetensors file holds one
/// chunk; reading the shards in order reproduces the full tensor list.
fn partition(tensors: &[TensorData], n: usize) -> Vec<Vec<TensorData>> {
    if n <= 1 {
        return vec![tensors.to_vec()];
    }
    let per = tensors.len().div_ceil(n);
    tensors.chunks(per).map(|c| c.to_vec()).collect()
}

/// Serialize a chunk to `path` with the safetensors crate (row-major f32; dtype "F32").
fn write_shard(path: PathBuf, chunk: &[TensorData]) -> anyhow::Result<()> {
    let mut owned: Vec<Tensor> = Vec::with_capacity(chunk.len());
    for t in chunk {
        owned.push(Tensor::from_vec(t.values.clone(), t.shape.clone(), &Device::Cpu)?);
    }
    // views borrow the tensors above; they are all built before this immutable borrow, so serialize can read them.
    let views: Vec<(&str, &Tensor)> = (0..chunk.len()).map(|i| (chunk[i].name.as_str(), &owned[i])).collect();
    st::serialize_to_file(views, None, &path)?;
    Ok(())
}

/// A minimal but valid HuggingFace `tokenizer.json` (ByteLevel BPE, no merges) so later steps that read a tokenizer
/// from the fixture have something loadable. Vocab tokens are derived from the seed.
fn write_config_json(dir: &Path, cfg: &serde_json::Value) -> anyhow::Result<()> {
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(cfg)?)?;
    Ok(())
}

/// Write the HF `tokenizer_config.json` (chat template, bos/eos, add flags) so the converter and llama.cpp agree on the
/// tokenizer's special tokens. bos is tok_0 / eos is tok_1 (plain strings, as a real `PreTrainedTokenizerFast` writes),
/// matching `tests/fixtures/tiny-llama-hf/tokenizer_config.json` byte for byte.
fn write_tokenizer_config_json(dir: &Path) -> anyhow::Result<()> {
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_string_pretty(&tokenizer_config_json())?,
    )?;
    Ok(())
}

/// Write a real Hugging Face `tokenizer.json` (version, truncation/padding, `added_tokens` for tok_0/tok_1 as special,
/// normalizer, ByteLevel pre_tokenizer and decoder, and a full BPE model). Identical to the one committed in
/// `tests/fixtures/tiny-llama-hf/tokenizer.json`.
fn write_tokenizer_json(dir: &Path, vocab_size: usize) -> anyhow::Result<()> {
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string_pretty(&tokenizer_json(vocab_size))?,
    )?;
    Ok(())
}

/// One entry of `tokenizer.json`'s `added_tokens`: tok_0 / tok_1, special and otherwise inert. Kept as a builder so both
/// the writer and the reference-equality test construct it identically.
fn added_token(id: u64, content: &str, special: bool) -> Json {
    json!({
        "id": id,
        "content": content,
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": special,
    })
}

/// The `tokenizer.json` this fixture writes — the exact structure committed alongside the reference GGUF. Built as a
/// value so a test can assert (with `serde_json::Value` equality) that the writer never drifts back to an old form and
/// destroys the fixture on disk.
fn tokenizer_json(vocab_size: usize) -> Json {
    let mut vocab = serde_json::Map::new();
    for i in 0..vocab_size {
        vocab.insert(format!("tok_{i}"), json!(i));
    }
    let mut added_tokens = Vec::new();
    if vocab_size > 0 {
        added_tokens.push(added_token(0, "tok_0", true));
    }
    if vocab_size > 1 {
        added_tokens.push(added_token(1, "tok_1", true));
    }
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true },
        "post_processor": null,
        "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": Json::Object(vocab),
            "merges": [],
        },
    })
}

/// The `tokenizer_config.json` this fixture writes (bos/eos as plain strings). Matches the committed reference exactly.
fn tokenizer_config_json() -> Json {
    json!({
        "add_bos_token": true,
        "add_eos_token": false,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "tok_0",
        "eos_token": "tok_1",
        "chat_template": "{% for message in messages %}{% if loop.first and message['role'] != 'system' %}{{ system_prompt }}{% endif %}{% if message['role'] == 'system' %}{{ message['content'] }}{% else %}{{ message['content'] }}{% endif %}{% endfor %}",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The committed fixture tokenizer files are the reference; re-derive the expected value in the test and compare
    /// with `serde_json::Value` equality so a drift in [`tokenizer_json`] / [`tokenizer_config_json`] is caught here
    /// (and, crucially, before [`save_tiny_checkpoint`](crate::tests::convert_reference::save_tiny_checkpoint) overwrites
    /// the folder and hides it).
    #[test]
    fn tokenizer_files_match_reference() {
        let fx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-llama-hf");
        let got_tok: Json = serde_json::from_str(
            &std::fs::read_to_string(fx.join("tokenizer.json")).expect("committed tokenizer.json parses"),
        )
        .expect("committed tokenizer.json parses");
        let got_cfg: Json = serde_json::from_str(
            &std::fs::read_to_string(fx.join("tokenizer_config.json")).expect("committed tokenizer_config.json parses"),
        )
        .expect("committed tokenizer_config.json parses");
        assert_eq!(got_tok, tokenizer_json(32), "committed tokenizer.json differs from what TinyLlama writes");
        assert_eq!(got_cfg, tokenizer_config_json(), "committed tokenizer_config.json differs from what TinyLlama writes");
    }
}

/// Deterministic generator so `write` and a test's expected values agree exactly.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // top 24 bits
        bits as f32 / 16_777_216.0
    }
}

fn mat(gen: &mut SplitMix64, rows: usize, cols: usize, scale: f32) -> Vec<f32> {
    (0..rows * cols).map(|_| scale * (gen.next_f32() * 2.0 - 1.0)).collect()
}
