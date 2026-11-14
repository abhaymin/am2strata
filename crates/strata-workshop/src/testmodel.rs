//! Tiny synthetic Llama-shaped safetensors fixture + a `tokenizer.json`, built deterministically from a fixed seed — no
//! downloads and no Python. Every workshop step's test reuses this to get a small, well-shaped model on disk, so the
//! round-trips exercise real code paths. The golden references that do need Python or llama.cpp (the
//! `convert_hf_to_gguf` output of this tiny model, the `llama-quantize` bytes) are made once by the reviewer and
//! committed under `tests/fixtures/` with a README; tests only read them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
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
    /// number of shard files to write (1 = one file; >=2 lets step 1 exercise shard enumeration)
    pub shard_count: usize,
}

impl Default for TinyLlama {
    fn default() -> Self {
        Self { seed: 0x5EED, num_layers: 2, vocab_size: 32, hidden: 16, intermediate: 32, shard_count: 1 }
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
            out.push(TensorData { name: p("post_layernorm.weight"), shape: vec![self.hidden], values: vec![1.0; self.hidden] });
            out.push(TensorData { name: p("mlp.gate_proj.weight"), shape: vec![self.intermediate, self.hidden], values: mat(&mut gen, self.intermediate, self.hidden, w) });
            out.push(TensorData { name: p("mlp.up_proj.weight"), shape: vec![self.intermediate, self.hidden], values: mat(&mut gen, self.intermediate, self.hidden, w) });
            out.push(TensorData { name: p("mlp.down_proj.weight"), shape: vec![self.hidden, self.intermediate], values: mat(&mut gen, self.hidden, self.intermediate, 1.0 / (self.intermediate as f32).sqrt()) });
        }
        out.push(TensorData { name: "model.norm.weight".into(), shape: vec![self.hidden], values: vec![1.0; self.hidden] });
        out
    }

    /// Write the fixture into `dir`: `<count>` safetensors shards named like an HF checkpoint plus a minimal
    /// `tokenizer.json`. Returns the shard files (in order) and the tokenizer path.
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
        write_tokenizer_json(dir)?;
        Ok(ShardFiles { dir: dir.to_path_buf(), shard_files })
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
fn write_tokenizer_json(dir: &Path) -> anyhow::Result<()> {
    let vocab_size = 8usize; // keeps it tiny; shape is what matters for reader reuse
    let mut vocab = BTreeMap::new();
    for i in 0..vocab_size {
        vocab.insert(format!("tok_{i}"), i);
    }
    // hand-build the JSON so the structure (model.vocab / model.merges / pre_tokenizer) is explicit and correct
    let mut pairs = String::from("{ ");
    for (i, (k, v)) in vocab.iter().enumerate() {
        if i > 0 {
            pairs.push_str(", ");
        }
        pairs.push_str(&format!("\"{k}\": {v}"));
    }
    pairs.push_str(" }");
    let text = format!(
        "{{\n  \"version\": \"1.1\",\n  \"add_bos_token\": false,\n  \"add_eos_token\": false,\n  \
         \"model\": {{ \"type\": \"BPE\", \"dropout\": null, \"vocab\": {pairs}, \"merges\": [] }},\n  \
         \"pre_tokenizer\": {{ \"type\": \"ByteLevel\", \"use_regex\": true }},\n  \
         \"decoders\": null,\n  \"special_tokens\": {{ \"bos\": null, \"eos\": null }}\n}}\n"
    );
    std::fs::write(dir.join("tokenizer.json"), text)?;
    Ok(())
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
