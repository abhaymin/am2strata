//! A real Llama-family decoder in candle, with LoRA hooks — the model the Workshop trains.
//!
//! Loads a Hugging Face folder (`config.json` + safetensors in F32, F16 or BF16) for the dense decoders the converter
//! handles: Llama and Mistral (RMSNorm, RoPE, grouped-query attention, SwiGLU), Qwen2 (the same with q/k/v biases) and
//! Qwen3 (the same with a per-head RMSNorm on q and k and an explicit `head_dim`); embeddings tied to the output head
//! or not. One sequence at a time (batch 1), causal mask, logits out. It follows transformers' modules step by step and
//! is checked against transformers' own logits (`tests/model_parity.rs`, fixtures made once with
//! `tests/fixtures/make-forward-reference.py`).
//!
//! **LoRA.** Any of `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` gets `y = W x + s · B A drop(x)`
//! with `s = alpha / r`, or `alpha / sqrt(r)` with rsLoRA. A is random (PEFT's bound, seeded), B is zero, so an
//! untrained adapter leaves the model unchanged. Only A and B are `Var`s; the base never is.
//!
//! **What costs no gradient.** candle records an operation only when one of its inputs descends from a `Var`. With
//! LoRA in the top N blocks only ([`LoraSpec::train_layers`]), the embedding and the blocks below hold no `Var`, so
//! they run as plain inference: nothing is kept for a backward pass. [`LlamaModel::lower`] returns the hidden state
//! entering the first trained block; [`LlamaModel::upper`] continues from it. That split is what the activation cache
//! (`crate::finetune`) stores, and what lets the lower blocks leave memory ([`LlamaModel::drop_lower`]).
//!
//! **A half-precision base.** With [`BaseDtype::F16`] or [`BaseDtype::BF16`] every weight matrix stays in 16 bits and
//! is widened to f32 only for the one product that uses it; activations, LoRA and the loss are f32. Inside the trained
//! region a plain `matmul` would keep that widened copy alive in the graph until the backward pass (f16 + f32 of the
//! whole region: worse than f32 alone), so there the product goes through [`FrozenLinear`], a custom operation that
//! keeps only the 16-bit weight and widens it again, briefly, to send the gradient back.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context as _, Result};
use candle_core::{CpuStorage, CustomOp1, DType, Device, Layout, Shape, Storage, Tensor, Var, D};
use serde_json::Value;

/// The decoder families this forward pass covers (each checked against transformers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Llama,
    Mistral,
    Qwen2,
    Qwen3,
}

impl Family {
    pub fn as_str(self) -> &'static str {
        match self {
            Family::Llama => "llama",
            Family::Mistral => "mistral",
            Family::Qwen2 => "qwen2",
            Family::Qwen3 => "qwen3",
        }
    }
}

/// What `config.json` says about the model, as far as the forward pass and the memory estimate need it.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub family: Family,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// `head_dim` when the config names it (Qwen3, some Mistral), else hidden / heads
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl ModelConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        let text = std::fs::read_to_string(&p).with_context(|| format!("{}: not a Hugging Face model folder (no config.json)", dir.display()))?;
        let v: Value = serde_json::from_str(&text).with_context(|| format!("{}", p.display()))?;
        Self::from_json(&v).with_context(|| format!("{}", p.display()))
    }

    /// Read a config, refusing by name what this trainer cannot run (rather than training something else quietly).
    pub fn from_json(c: &Value) -> Result<Self> {
        let arch = c["architectures"].as_array().and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("");
        let model_type = c["model_type"].as_str().unwrap_or("");
        for key in ["num_experts", "num_local_experts", "num_experts_per_tok", "moe_intermediate_size"] {
            if c.get(key).is_some_and(|v| !v.is_null() && v.as_u64() != Some(0)) {
                bail!("a mixture-of-experts model ({key} in config.json): training expert adapters is not done; only dense Llama, Mistral, Qwen2 and Qwen3 decoders train here");
            }
        }
        for key in ["vision_config", "audio_config", "vision_tower"] {
            if c.get(key).is_some_and(|v| !v.is_null()) {
                bail!("a multimodal model ({key} in config.json): vision and audio towers are not trained here; only text decoders are");
            }
        }
        let family = match (model_type, arch) {
            ("llama", _) => Family::Llama,
            ("mistral", _) => Family::Mistral,
            ("qwen2", _) => Family::Qwen2,
            ("qwen3", _) => Family::Qwen3,
            ("", a) if a.starts_with("Llama") => Family::Llama,
            ("", a) if a.starts_with("Mistral") => Family::Mistral,
            ("", a) if a.starts_with("Qwen2") => Family::Qwen2,
            ("", a) if a.starts_with("Qwen3") => Family::Qwen3,
            _ => bail!("architecture {arch:?} (model_type {model_type:?}) is not one this trainer runs: Llama, Mistral, Qwen2, Qwen3"),
        };
        let n = |k: &str| c[k].as_u64().map(|x| x as usize).ok_or_else(|| anyhow!("config.{k} is missing or not a number"));
        // transformers 5 writes rope_parameters {rope_theta, rope_type}; before that rope_theta and rope_scaling
        let rope = &c["rope_parameters"];
        if let Some(t) = rope["rope_type"].as_str().or(c["rope_scaling"]["rope_type"].as_str()).or(c["rope_scaling"]["type"].as_str()) {
            if t != "default" {
                bail!("rope scaling of type {t:?} is not implemented in this forward pass");
            }
        } else if c.get("rope_scaling").is_some_and(|v| !v.is_null()) {
            bail!("config.rope_scaling is set, which this forward pass does not implement");
        }
        if let Some(act) = c["hidden_act"].as_str() {
            if act != "silu" {
                bail!("hidden_act {act:?} is not implemented (silu only)");
            }
        }
        let hidden_size = n("hidden_size")?;
        let num_attention_heads = n("num_attention_heads")?;
        if num_attention_heads == 0 {
            bail!("config.num_attention_heads is 0");
        }
        let num_key_value_heads = c["num_key_value_heads"].as_u64().map(|x| x as usize).unwrap_or(num_attention_heads);
        if num_key_value_heads == 0 || num_attention_heads % num_key_value_heads != 0 {
            bail!("num_attention_heads {num_attention_heads} is not a multiple of num_key_value_heads {num_key_value_heads}");
        }
        let head_dim = c["head_dim"].as_u64().map(|x| x as usize).unwrap_or(hidden_size / num_attention_heads);
        if head_dim == 0 || head_dim % 2 != 0 {
            bail!("head_dim {head_dim} must be even and above 0 (rotary embeddings pair its halves)");
        }
        Ok(ModelConfig {
            family,
            vocab_size: n("vocab_size")?,
            hidden_size,
            intermediate_size: n("intermediate_size")?,
            num_hidden_layers: n("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            rope_theta: rope["rope_theta"].as_f64().or(c["rope_theta"].as_f64()).unwrap_or(10000.0),
            max_position_embeddings: c["max_position_embeddings"].as_u64().map(|x| x as usize).unwrap_or(2048),
            tie_word_embeddings: c["tie_word_embeddings"].as_bool().unwrap_or(false),
        })
    }

    /// `(out, in)` of one target's weight.
    pub fn target_shape(&self, t: Target) -> (usize, usize) {
        let (h, i) = (self.hidden_size, self.intermediate_size);
        let (q, kv) = (self.num_attention_heads * self.head_dim, self.num_key_value_heads * self.head_dim);
        match t {
            Target::Q => (q, h),
            Target::K | Target::V => (kv, h),
            Target::O => (h, q),
            Target::Gate | Target::Up => (i, h),
            Target::Down => (h, i),
        }
    }

    /// Parameters of one decoder block (matrices, norms, and the family's biases and q/k norms).
    pub fn block_params(&self) -> u64 {
        let matrices: u64 = Target::ALL.iter().map(|t| { let (o, i) = self.target_shape(*t); (o * i) as u64 }).sum();
        matrices + self.block_small_params()
    }

    /// The 1-D parameters of one block: two norms, plus Qwen2's q/k/v biases or Qwen3's q/k norms.
    pub fn block_small_params(&self) -> u64 {
        let (q, kv) = (self.num_attention_heads * self.head_dim, self.num_key_value_heads * self.head_dim);
        let extra = match self.family {
            Family::Qwen2 => q + 2 * kv,
            Family::Qwen3 => 2 * self.head_dim,
            _ => 0,
        };
        (2 * self.hidden_size + extra) as u64
    }

    /// All parameters: embeddings, blocks, the final norm and the output head unless it is tied to the embeddings.
    pub fn total_params(&self) -> u64 {
        let emb = (self.vocab_size * self.hidden_size) as u64;
        emb + self.num_hidden_layers as u64 * self.block_params() + self.hidden_size as u64 + if self.tie_word_embeddings { 0 } else { emb }
    }
}

/// A projection a LoRA adapter can sit on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Target {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

impl Target {
    pub const ALL: [Target; 7] = [Target::Q, Target::K, Target::V, Target::O, Target::Gate, Target::Up, Target::Down];

    pub fn as_str(self) -> &'static str {
        match self {
            Target::Q => "q_proj",
            Target::K => "k_proj",
            Target::V => "v_proj",
            Target::O => "o_proj",
            Target::Gate => "gate_proj",
            Target::Up => "up_proj",
            Target::Down => "down_proj",
        }
    }

    pub fn from_s(s: &str) -> Result<Target> {
        Target::ALL.into_iter().find(|t| t.as_str() == s.trim())
            .ok_or_else(|| anyhow!("{s:?} is not a LoRA target; choose from q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"))
    }

    /// `q_proj,v_proj` → targets, in the fixed module order, without repeats.
    pub fn parse_list(s: &str) -> Result<Vec<Target>> {
        let mut out: Vec<Target> = s.split(',').filter(|p| !p.trim().is_empty()).map(Target::from_s).collect::<Result<_>>()?;
        out.sort();
        out.dedup();
        if out.is_empty() {
            bail!("no LoRA target modules named");
        }
        Ok(out)
    }

    /// The module path inside a block, as Hugging Face names it.
    fn module(self) -> &'static str {
        match self {
            Target::Q | Target::K | Target::V | Target::O => "self_attn",
            _ => "mlp",
        }
    }
}

/// LoRA settings (what `adapter_config.json` stores, plus which blocks train).
#[derive(Debug, Clone, PartialEq)]
pub struct LoraSpec {
    pub r: usize,
    pub alpha: f64,
    pub dropout: f64,
    pub use_rslora: bool,
    pub targets: Vec<Target>,
    /// LoRA only in the top N blocks; `None` trains every block
    pub train_layers: Option<usize>,
    /// seeds A's initial values
    pub seed: u64,
}

impl Default for LoraSpec {
    fn default() -> Self {
        Self { r: 16, alpha: 32.0, dropout: 0.0, use_rslora: false, targets: vec![Target::Q, Target::V], train_layers: None, seed: 42 }
    }
}

impl LoraSpec {
    /// `alpha / r`, or `alpha / sqrt(r)` with rsLoRA.
    pub fn scaling(&self) -> f64 {
        if self.use_rslora { self.alpha / (self.r as f64).sqrt() } else { self.alpha / self.r as f64 }
    }

    /// The index of the first block that carries adapters.
    pub fn first_trained(&self, num_layers: usize) -> usize {
        match self.train_layers {
            Some(n) => num_layers.saturating_sub(n.max(1)),
            None => 0,
        }
    }

    /// Adapter parameters for `cfg`: per trained block and target, `r · (in + out)`.
    pub fn trainable_params(&self, cfg: &ModelConfig) -> u64 {
        let blocks = (cfg.num_hidden_layers - self.first_trained(cfg.num_hidden_layers)) as u64;
        let per: u64 = self.targets.iter().map(|t| { let (o, i) = cfg.target_shape(*t); (self.r * (o + i)) as u64 }).sum();
        blocks * per
    }
}

/// How the frozen base is held in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseDtype {
    F32,
    F16,
    BF16,
}

impl BaseDtype {
    pub fn from_s(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "float32" => Ok(BaseDtype::F32),
            "f16" | "fp16" | "float16" => Ok(BaseDtype::F16),
            "bf16" | "bfloat16" => Ok(BaseDtype::BF16),
            "4bit" | "8bit" | "q4" | "int4" | "int8" => bail!("a {s} base (QLoRA) is not done: the base trains in f32, f16 or bf16"),
            _ => bail!("base dtype {s:?}: choose f32, f16 or bf16"),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            BaseDtype::F32 => "f32",
            BaseDtype::F16 => "f16",
            BaseDtype::BF16 => "bf16",
        }
    }
    pub fn bytes(self) -> u64 {
        if self == BaseDtype::F32 { 4 } else { 2 }
    }
    fn dtype(self) -> DType {
        match self {
            BaseDtype::F32 => DType::F32,
            BaseDtype::F16 => DType::F16,
            BaseDtype::BF16 => DType::BF16,
        }
    }
}

/// A frozen product `x Wᵀ` whose weight stays in 16 bits inside the graph. The value is computed outside the graph
/// (the widened weight is dropped at once) and handed over here; the backward pass widens the weight again for
/// `grad · W`. Without this, candle's `matmul` node would hold the f32 copy of every weight of the trained region.
struct FrozenLinear {
    /// the forward value, taken (not copied twice) by the forward call
    y: Mutex<Option<Tensor>>,
    /// `[out, in]`, 16-bit
    w: Tensor,
}

impl CustomOp1 for FrozenLinear {
    fn name(&self) -> &'static str {
        "frozen-half-linear"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        let y = self.y.lock().map_err(|_| candle_core::Error::Msg("frozen-half-linear: lock".into()))?.take()
            .ok_or_else(|| candle_core::Error::Msg("frozen-half-linear: forward ran twice".into()))?;
        let y = y.contiguous()?;
        let shape = y.shape().clone();
        let (storage, layout) = y.storage_and_layout();
        match &*storage {
            Storage::Cpu(c) if layout.start_offset() == 0 => Ok((c.clone(), shape)),
            _ => Err(candle_core::Error::Msg("frozen-half-linear: the value is not a plain CPU tensor".into())),
        }
    }

    fn bwd(&self, _arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> candle_core::Result<Option<Tensor>> {
        Ok(Some(grad_res.matmul(&self.w.to_dtype(DType::F32)?)?))
    }
}

/// One LoRA adapter: `A` `[r, in]`, `B` `[out, r]`.
#[derive(Debug, Clone)]
pub struct Lora {
    pub a: Var,
    pub b: Var,
    pub scaling: f64,
    pub dropout: f64,
}

/// A frozen projection, with an optional bias (Qwen2) and an optional adapter.
#[derive(Debug, Clone)]
pub struct Linear {
    /// `[out, in]` in the base dtype
    w: Tensor,
    bias: Option<Tensor>,
    pub lora: Option<Lora>,
}

/// Per-forward switches: training turns dropout on, and `seed` makes its masks repeat exactly.
#[derive(Debug, Default)]
pub struct Ctx {
    pub train: bool,
    pub seed: u64,
    calls: Cell<u64>,
}

impl Ctx {
    pub fn eval() -> Self {
        Self::default()
    }
    pub fn train(seed: u64) -> Self {
        Self { train: true, seed, calls: Cell::new(0) }
    }
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in [0, 1) from the top 24 bits.
fn unit(state: &mut u64) -> f32 {
    (splitmix(state) >> 40) as f32 / 16_777_216.0
}

impl Linear {
    fn frozen(&self, x: &Tensor) -> Result<Tensor> {
        let y = if self.w.dtype() == DType::F32 {
            x.matmul(&self.w.t()?)?
        } else {
            // widen for this one product; the copy is gone when this block ends
            let value = x.detach().matmul(&self.w.to_dtype(DType::F32)?.t()?)?;
            if x.track_op() {
                x.apply_op1(FrozenLinear { y: Mutex::new(Some(value)), w: self.w.clone() })?
            } else {
                value
            }
        };
        Ok(match &self.bias {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        })
    }

    /// `x` is `[rows, in]`, f32.
    pub fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let base = self.frozen(x)?;
        let Some(l) = &self.lora else { return Ok(base) };
        let x = if ctx.train && l.dropout > 0.0 {
            // our own mask (candle's CPU random numbers cannot be seeded): the same seed gives the same run
            let n = ctx.calls.get();
            ctx.calls.set(n + 1);
            let mut state = ctx.seed ^ n.wrapping_mul(0xD6E8_FEB8_6659_FD93);
            let keep = 1.0 - l.dropout as f32;
            let mask: Vec<f32> = (0..x.elem_count()).map(|_| if unit(&mut state) < keep { 1.0 / keep } else { 0.0 }).collect();
            x.mul(&Tensor::from_vec(mask, x.shape(), x.device())?)?
        } else {
            x.clone()
        };
        let delta = (x.matmul(&l.a.as_tensor().t()?)?.matmul(&l.b.as_tensor().t()?)? * l.scaling)?;
        Ok(base.add(&delta)?)
    }
}

/// One decoder block.
#[derive(Debug, Clone)]
struct Block {
    /// q, k, v, o, gate, up, down — in [`Target::ALL`] order
    lin: Vec<Linear>,
    norm1: Tensor,
    norm2: Tensor,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
}

/// The decoder: frozen base, adapters in the trained blocks.
pub struct LlamaModel {
    pub cfg: ModelConfig,
    pub base_dtype: BaseDtype,
    /// `[vocab, hidden]`; absent once the lower part was dropped and the head is not tied to it
    embed: Option<Tensor>,
    blocks: Vec<Option<Block>>,
    norm: Tensor,
    /// absent when tied (the embeddings are the head)
    lm_head: Option<Tensor>,
    first_trained: usize,
    spec: Option<LoraSpec>,
    dev: Device,
}

/// The weight files of a model folder, in order: `model.safetensors`, or the `model-0000i-of-0000n` shards.
pub fn weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir).with_context(|| format!("{}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with("model-") && name.ends_with(".safetensors")
        })
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("{}: no model.safetensors or model-*.safetensors (a GGUF or .bin model has to be converted to safetensors first)", dir.display());
    }
    Ok(files)
}

fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(x.broadcast_div(&(var + eps)?.sqrt()?)?.broadcast_mul(w)?)
}

impl LlamaModel {
    /// Load the frozen base, holding its matrices as `base_dtype`. `skip_lower` leaves out everything below block
    /// `first` (the embeddings too, unless they are the output head): for a run whose activation cache is complete.
    pub fn load(dir: &Path, base_dtype: BaseDtype, dev: &Device) -> Result<Self> {
        Self::load_from(dir, base_dtype, dev, 0)
    }

    /// As [`load`], without the blocks below `skip_below` (and without the embeddings when the head is its own tensor).
    pub fn load_from(dir: &Path, base_dtype: BaseDtype, dev: &Device, skip_below: usize) -> Result<Self> {
        let cfg = ModelConfig::load(dir)?;
        let files = weight_files(dir)?;
        // mapped, not read: each tensor is copied out (and narrowed to the base dtype) one at a time
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&files) }.with_context(|| format!("{}", dir.display()))?;
        let names: HashMap<String, ()> = st.tensors().into_iter().map(|(n, _)| (n, ())).collect();
        let prefix = if names.contains_key("model.embed_tokens.weight") { "model." } else if names.contains_key("embed_tokens.weight") { "" }
            else { bail!("{}: no model.embed_tokens.weight (not a Llama-family checkpoint)", dir.display()) };
        let matrix = |name: &str| -> Result<Tensor> {
            let t = st.load(name, dev).with_context(|| format!("{}: tensor {name}", dir.display()))?;
            if !matches!(t.dtype(), DType::F32 | DType::F16 | DType::BF16) {
                bail!("{name} is {:?}: the base must be F32, F16 or BF16 safetensors", t.dtype());
            }
            Ok(t.to_dtype(base_dtype.dtype())?)
        };
        let vector = |name: &str| -> Result<Tensor> { Ok(st.load(name, dev).with_context(|| format!("{}: tensor {name}", dir.display()))?.to_dtype(DType::F32)?) };
        let expect = |t: &Tensor, dims: &[usize], name: &str| -> Result<()> {
            if t.dims() != dims { bail!("{name} has shape {:?}, the config says {dims:?}", t.dims()) } else { Ok(()) }
        };

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            if i < skip_below {
                blocks.push(None);
                continue;
            }
            let p = |k: &str| format!("{prefix}layers.{i}.{k}");
            let mut lin = Vec::with_capacity(7);
            for t in Target::ALL {
                let name = p(&format!("{}.{}.weight", t.module(), t.as_str()));
                let w = matrix(&name)?;
                let (o, in_) = cfg.target_shape(t);
                expect(&w, &[o, in_], &name)?;
                let bias_name = p(&format!("{}.{}.bias", t.module(), t.as_str()));
                let bias = if names.contains_key(&bias_name) { Some(vector(&bias_name)?) } else { None };
                lin.push(Linear { w, bias, lora: None });
            }
            let opt = |k: &str| -> Result<Option<Tensor>> { let n = p(k); if names.contains_key(&n) { Ok(Some(vector(&n)?)) } else { Ok(None) } };
            let (q_norm, k_norm) = (opt("self_attn.q_norm.weight")?, opt("self_attn.k_norm.weight")?);
            if cfg.family == Family::Qwen3 && (q_norm.is_none() || k_norm.is_none()) {
                bail!("a Qwen3 model without self_attn.q_norm / k_norm in block {i}");
            }
            blocks.push(Some(Block { lin, norm1: vector(&p("input_layernorm.weight"))?, norm2: vector(&p("post_attention_layernorm.weight"))?, q_norm, k_norm }));
        }
        let has_head = names.contains_key("lm_head.weight");
        if !has_head && !cfg.tie_word_embeddings {
            bail!("{}: no lm_head.weight and tie_word_embeddings is false", dir.display());
        }
        let need_embed = skip_below == 0 || !has_head;
        let embed = if need_embed {
            let e = matrix(&format!("{prefix}embed_tokens.weight"))?;
            expect(&e, &[cfg.vocab_size, cfg.hidden_size], "embed_tokens.weight")?;
            Some(e)
        } else {
            None
        };
        let lm_head = if has_head { Some(matrix("lm_head.weight")?) } else { None };
        let norm = vector(&format!("{prefix}norm.weight"))?;
        Ok(Self { cfg, base_dtype, embed, blocks, norm, lm_head, first_trained: 0, spec: None, dev: dev.clone() })
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// The first block carrying adapters (0 without adapters or when every block trains).
    pub fn first_trained(&self) -> usize {
        self.first_trained
    }

    pub fn lora_spec(&self) -> Option<&LoraSpec> {
        self.spec.as_ref()
    }

    /// Put fresh adapters on the chosen targets of the trained blocks: A random in ±1/sqrt(in) (PEFT's
    /// `kaiming_uniform_(a=√5)` bound), seeded by `spec.seed`, the block and the target; B zero.
    pub fn add_lora(&mut self, spec: &LoraSpec) -> Result<()> {
        if spec.r == 0 {
            bail!("LoRA rank 0 would be full fine-tuning, which this trainer does not do; use a rank of 1 or more");
        }
        if !(0.0..1.0).contains(&spec.dropout) {
            bail!("LoRA dropout {} must be in [0, 1)", spec.dropout);
        }
        if let Some(n) = spec.train_layers {
            if n == 0 || n > self.cfg.num_hidden_layers {
                bail!("--train-layers {n}: the model has {} blocks; choose 1 to {}", self.cfg.num_hidden_layers, self.cfg.num_hidden_layers);
            }
        }
        let first = spec.first_trained(self.cfg.num_hidden_layers);
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let Some(block) = block else { continue };
            for (ti, t) in Target::ALL.into_iter().enumerate() {
                block.lin[ti].lora = None;
                if i < first || !spec.targets.contains(&t) {
                    continue;
                }
                let (out, in_) = self.cfg.target_shape(t);
                let mut state = spec.seed ^ ((i as u64) << 32) ^ ((ti as u64) << 24) ^ 0xA5A5_5A5A;
                let bound = 1.0 / (in_ as f32).sqrt();
                let a: Vec<f32> = (0..spec.r * in_).map(|_| bound * (2.0 * unit(&mut state) - 1.0)).collect();
                block.lin[ti].lora = Some(Lora {
                    a: Var::from_tensor(&Tensor::from_vec(a, (spec.r, in_), &self.dev)?)?,
                    b: Var::from_tensor(&Tensor::zeros((out, spec.r), DType::F32, &self.dev)?)?,
                    scaling: spec.scaling(),
                    dropout: spec.dropout,
                });
            }
        }
        self.first_trained = first;
        self.spec = Some(spec.clone());
        Ok(())
    }

    /// Every adapter matrix under its PEFT name (`base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight`),
    /// block by block, target by target, A then B.
    pub fn lora_vars(&self) -> Vec<(String, Var)> {
        let mut out = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            let Some(block) = block else { continue };
            for (ti, t) in Target::ALL.into_iter().enumerate() {
                if let Some(l) = &block.lin[ti].lora {
                    let stem = format!("base_model.model.model.layers.{i}.{}.{}", t.module(), t.as_str());
                    out.push((format!("{stem}.lora_A.weight"), l.a.clone()));
                    out.push((format!("{stem}.lora_B.weight"), l.b.clone()));
                }
            }
        }
        out
    }

    /// Attach a PEFT adapter folder (`adapter_config.json` + `adapter_model.safetensors`) as it is: the blocks and
    /// targets its tensors name, its rank, alpha and rsLoRA flag. Returns the settings it carried.
    pub fn load_adapter(&mut self, dir: &Path) -> Result<LoraSpec> {
        let cfg: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("adapter_config.json"))
            .with_context(|| format!("{}: no adapter_config.json", dir.display()))?)?;
        if cfg["use_dora"].as_bool() == Some(true) {
            bail!("a DoRA adapter: only plain LoRA and rsLoRA adapters load here");
        }
        let r = cfg["r"].as_u64().ok_or_else(|| anyhow!("adapter_config.json has no r"))? as usize;
        let alpha = cfg["lora_alpha"].as_f64().ok_or_else(|| anyhow!("adapter_config.json has no lora_alpha"))?;
        let mut spec = LoraSpec { r, alpha, dropout: cfg["lora_dropout"].as_f64().unwrap_or(0.0), use_rslora: cfg["use_rslora"].as_bool().unwrap_or(false),
                                  targets: Vec::new(), train_layers: None, seed: 0 };
        let tensors = candle_core::safetensors::load(dir.join("adapter_model.safetensors"), &self.dev)?;
        let mut used = 0;
        let mut first = self.cfg.num_hidden_layers;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let Some(block) = block else { continue };
            for (ti, t) in Target::ALL.into_iter().enumerate() {
                let stem = format!("base_model.model.model.layers.{i}.{}.{}", t.module(), t.as_str());
                let (a, b) = (tensors.get(&format!("{stem}.lora_A.weight")), tensors.get(&format!("{stem}.lora_B.weight")));
                block.lin[ti].lora = match (a, b) {
                    (Some(a), Some(b)) => {
                        let (out, in_) = self.cfg.target_shape(t);
                        if a.dims() != [r, in_] || b.dims() != [out, r] {
                            bail!("{stem}: adapter shapes {:?} and {:?} do not fit a [{out}, {in_}] weight at rank {r}", a.dims(), b.dims());
                        }
                        used += 2;
                        first = first.min(i);
                        if !spec.targets.contains(&t) {
                            spec.targets.push(t);
                        }
                        Some(Lora { a: Var::from_tensor(&a.to_dtype(DType::F32)?)?, b: Var::from_tensor(&b.to_dtype(DType::F32)?)?, scaling: spec.scaling(), dropout: spec.dropout })
                    }
                    (None, None) => None,
                    _ => bail!("{stem}: the adapter has only one of lora_A and lora_B"),
                };
            }
        }
        if used != tensors.len() {
            let unknown: Vec<&String> = tensors.keys().filter(|k| !k.contains(".lora_A.weight") && !k.contains(".lora_B.weight") || !k.contains(".layers.")).take(3).collect();
            bail!("the adapter holds {} tensors, {used} of them fit this model (for example not: {unknown:?}); it was made for another model or is not a plain LoRA", tensors.len());
        }
        spec.targets.sort();
        self.first_trained = if used == 0 { 0 } else { first };
        if self.first_trained > 0 {
            spec.train_layers = Some(self.cfg.num_hidden_layers - self.first_trained);
        }
        self.spec = Some(spec.clone());
        Ok(spec)
    }

    /// Free the embeddings (when the head is its own tensor) and the blocks below the first trained one. After this
    /// only [`upper`] works: the hidden states must come from the activation cache.
    pub fn drop_lower(&mut self) {
        for b in self.blocks.iter_mut().take(self.first_trained) {
            *b = None;
        }
        if self.lm_head.is_some() && self.first_trained > 0 {
            self.embed = None;
        }
    }

    /// Whether the embeddings and every block below the first trained one are in memory.
    pub fn has_lower(&self) -> bool {
        self.embed.is_some() && self.blocks.iter().take(self.first_trained).all(Option::is_some)
    }

    fn rope(&self, seq: usize) -> Result<(Tensor, Tensor)> {
        let hd = self.cfg.head_dim;
        // as transformers: inv_freq in f32, angles = position × inv_freq, each repeated over both halves
        let inv: Vec<f32> = (0..hd / 2).map(|i| 1.0 / (self.cfg.rope_theta as f32).powf(2.0 * i as f32 / hd as f32)).collect();
        let mut cos = Vec::with_capacity(seq * hd);
        let mut sin = Vec::with_capacity(seq * hd);
        for p in 0..seq {
            for half in 0..2 {
                let _ = half;
                for f in &inv {
                    let a = p as f32 * f;
                    cos.push(a.cos());
                    sin.push(a.sin());
                }
            }
        }
        Ok((Tensor::from_vec(cos, (seq, hd), &self.dev)?, Tensor::from_vec(sin, (seq, hd), &self.dev)?))
    }

    fn block_forward(&self, block: &Block, x: &Tensor, rope: &(Tensor, Tensor), mask: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let c = &self.cfg;
        let (seq, hd, nh, nkv) = (x.dim(0)?, c.head_dim, c.num_attention_heads, c.num_key_value_heads);
        let h = rms_norm(x, &block.norm1, c.rms_norm_eps)?;
        // [seq, heads·hd] → [heads, seq, hd], with Qwen3's per-head norm before the rotation
        let heads = |t: Tensor, n: usize, norm: &Option<Tensor>| -> Result<Tensor> {
            let t = t.reshape((seq, n, hd))?;
            let t = match norm {
                Some(w) => rms_norm(&t, w, c.rms_norm_eps)?,
                None => t,
            };
            Ok(t.transpose(0, 1)?.contiguous()?)
        };
        let rotate = |t: &Tensor| -> Result<Tensor> {
            let (x1, x2) = (t.narrow(D::Minus1, 0, hd / 2)?, t.narrow(D::Minus1, hd / 2, hd / 2)?);
            let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
            Ok((t.broadcast_mul(&rope.0)? + rot.broadcast_mul(&rope.1)?)?)
        };
        let q = rotate(&heads(block.lin[0].forward(&h, ctx)?, nh, &block.q_norm)?)?;
        let k = rotate(&heads(block.lin[1].forward(&h, ctx)?, nkv, &block.k_norm)?)?;
        let v = heads(block.lin[2].forward(&h, ctx)?, nkv, &None)?;
        // grouped-query attention: each key/value head serves heads/kv_heads query heads
        let rep = nh / nkv;
        let spread = |t: Tensor| -> Result<Tensor> {
            if rep == 1 { Ok(t) } else { Ok(t.unsqueeze(1)?.broadcast_as((nkv, rep, seq, hd))?.reshape((nh, seq, hd))?) }
        };
        let (k, v) = (spread(k)?, spread(v)?);
        let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * (1.0 / (hd as f64).sqrt()))?.broadcast_add(mask)?;
        // candle_nn's `softmax_last_dim` is faster but has no backward pass (it cuts the graph without an error, and the
        // q and k adapters would then never train); this one is built from differentiable operations
        let att = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = att.matmul(&v)?.transpose(0, 1)?.reshape((seq, nh * hd))?;
        let x = (x + block.lin[3].forward(&out, ctx)?)?;

        let h = rms_norm(&x, &block.norm2, c.rms_norm_eps)?;
        let gated = (candle_nn::ops::silu(&block.lin[4].forward(&h, ctx)?)? * block.lin[5].forward(&h, ctx)?)?;
        Ok((x + block.lin[6].forward(&gated, ctx)?)?)
    }

    fn causal_mask(&self, seq: usize) -> Result<Tensor> {
        let m: Vec<f32> = (0..seq).flat_map(|i| (0..seq).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })).collect();
        Ok(Tensor::from_vec(m, (seq, seq), &self.dev)?)
    }

    fn run_blocks(&self, mut x: Tensor, range: std::ops::Range<usize>, ctx: &Ctx) -> Result<Tensor> {
        let seq = x.dim(0)?;
        if seq == 0 {
            bail!("an empty token sequence");
        }
        let (rope, mask) = (self.rope(seq)?, self.causal_mask(seq)?);
        for i in range {
            let block = self.blocks[i].as_ref().ok_or_else(|| anyhow!("block {i} is not in memory (it was dropped for the activation cache)"))?;
            x = self.block_forward(block, &x, &rope, &mask, ctx)?;
        }
        Ok(x)
    }

    /// The hidden state `[seq, hidden]` entering the first trained block: embeddings and the frozen blocks below it.
    /// No adapter is involved, so nothing is recorded for a backward pass; the result is detached.
    pub fn lower(&self, tokens: &[u32]) -> Result<Tensor> {
        let embed = self.embed.as_ref().ok_or_else(|| anyhow!("the embeddings are not in memory (dropped for the activation cache)"))?;
        if let Some(bad) = tokens.iter().find(|t| **t as usize >= self.cfg.vocab_size) {
            bail!("token id {bad} is outside the vocabulary of {}", self.cfg.vocab_size);
        }
        let ids = Tensor::from_vec(tokens.to_vec(), (tokens.len(),), &self.dev)?;
        let x = embed.index_select(&ids, 0)?.to_dtype(DType::F32)?;
        Ok(self.run_blocks(x, 0..self.first_trained, &Ctx::eval())?.detach())
    }

    /// From that hidden state through the trained blocks to the final norm: `[seq, hidden]`.
    pub fn upper_hidden(&self, h: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let x = self.run_blocks(h.clone(), self.first_trained..self.cfg.num_hidden_layers, ctx)?;
        rms_norm(&x, &self.norm, self.cfg.rms_norm_eps)
    }

    /// The output head on `[rows, hidden]` → `[rows, vocab]`.
    pub fn head(&self, h: &Tensor) -> Result<Tensor> {
        let w = self.lm_head.as_ref().or(self.embed.as_ref()).ok_or_else(|| anyhow!("no output head in memory"))?.clone();
        Linear { w, bias: None, lora: None }.forward(h, &Ctx::eval())
    }

    /// From the cached hidden state to logits `[seq, vocab]`.
    pub fn upper(&self, h: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        self.head(&self.upper_hidden(h, ctx)?)
    }

    /// Token ids → logits `[seq, vocab]` (f32).
    pub fn forward(&self, tokens: &[u32], ctx: &Ctx) -> Result<Tensor> {
        self.upper(&self.lower(tokens)?, ctx)
    }

    /// The mean next-token cross-entropy over the supervised positions: `supervised[p]` says token `p` is a target
    /// (predicted from position `p − 1`). The head runs only on the rows that are scored.
    pub fn loss_from_hidden(&self, h: &Tensor, tokens: &[u32], supervised: &[bool], ctx: &Ctx) -> Result<Tensor> {
        let rows: Vec<u32> = (1..tokens.len()).filter(|p| supervised[*p]).map(|p| (p - 1) as u32).collect();
        if rows.is_empty() {
            bail!("an example with no supervised token");
        }
        let targets: Vec<u32> = rows.iter().map(|r| tokens[*r as usize + 1]).collect();
        let n = rows.len();
        let hidden = self.upper_hidden(h, ctx)?.index_select(&Tensor::from_vec(rows, (n,), &self.dev)?, 0)?;
        let logits = self.head(&hidden)?;
        Ok(candle_nn::loss::cross_entropy(&logits, &Tensor::from_vec(targets, (n,), &self.dev)?)?)
    }

    pub fn loss(&self, tokens: &[u32], supervised: &[bool], ctx: &Ctx) -> Result<Tensor> {
        self.loss_from_hidden(&self.lower(tokens)?, tokens, supervised, ctx)
    }
}

/// What PEFT reads from `adapter_config.json`, for `spec` on `cfg` (`base` is recorded as the base model's path).
pub fn adapter_config(spec: &LoraSpec, cfg: &ModelConfig, base: &str) -> Value {
    let first = spec.first_trained(cfg.num_hidden_layers);
    let layers: Value = if first == 0 { Value::Null } else { serde_json::json!((first..cfg.num_hidden_layers).collect::<Vec<_>>()) };
    serde_json::json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base,
        "r": spec.r,
        "lora_alpha": spec.alpha,
        "lora_dropout": spec.dropout,
        "use_rslora": spec.use_rslora,
        "use_dora": false,
        "target_modules": spec.targets.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
        "layers_to_transform": layers,
        "bias": "none",
        "fan_in_fan_out": false,
        "inference_mode": true,
        "init_lora_weights": true,
        "modules_to_save": Value::Null,
    })
}

/// The adapter tensors as an ordered map, for saving.
pub fn adapter_tensors(model: &LlamaModel) -> BTreeMap<String, Tensor> {
    model.lora_vars().into_iter().map(|(n, v)| (n, v.as_tensor().detach())).collect()
}
