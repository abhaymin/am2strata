//! LoRA layers on top of a frozen base, and a 2-layer Llama-shaped forward pass that threads them through (attention
//! q/k/v/o get low-rank adapters, the rest of the base stays frozen). This is written by hand in this crate rather than
//! pulled from candle-transformers: candle-transformers is not a dependency of strata-workshop here (only candle-core/
//! candle-nn-mold are), and the proof model is tiny — 2 decoder layers. The plan (`docs/MODEL-WORKSHOP-PLAN.md`) allows
//! exactly this ("write the tiny forward pass yourself ... it is 2 layers"). Rotary embeddings are also omitted in this
//! proof block; they do not change the LoRA/adapters/loss/masking machinery being exercised here.
//!
//! LoRA: for a frozen projection `y = W x`, the adapter adds `scaling * (B A) x`. A is a trainable `Var` of shape `[r,
//! in]`, B one of `[out, r]`; the base weight `W` is a frozen plain `Tensor` and never appears in the optimizer. PEFT
//! names these `...q_proj.lora_A.weight` / `.lora_B.weight`, which [`LoraLinear::param_refs`] returns so a checkpoint
//! matches PEFT byte for byte. Scaling is `alpha / r`, or `alpha / sqrt(r)` with rsLoRA ([`LoraConfig::scaling`]).

use std::collections::BTreeMap;

use candle_core::{Device, Tensor, Var};
use anyhow::anyhow;

/// Any error in this module (candle ops + a few hand-built messages). `Box<dyn Error>` accepts both candle's error and
/// `anyhow!`-built messages without conversion at each call site.
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// LoRA hyper-parameters, also what `adapter_config.json` stores.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// low rank
    pub r: usize,
    /// scaling multiplier (lora_alpha in PEFT)
    pub lora_alpha: f32,
    /// rsLoRA uses `alpha / sqrt(r)` instead of `alpha / r`
    pub use_rslora: bool,
    /// which projections get adapters: attention q/k/v/o here
    pub target_modules: Vec<String>,
}

impl LoraConfig {
    /// The LoRA scaling factor. rsLoRA normalizes by `sqrt(r)`; ordinary LoRA by `r`.
    pub fn scaling(&self) -> f32 {
        if self.use_rslora {
            self.lora_alpha / (self.r as f32).sqrt()
        } else {
            self.lora_alpha / self.r as f32
        }
    }
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self { r: 4, lora_alpha: 8.0, use_rslora: false, target_modules: vec!["q_proj".into(), "k_proj".into(), "v_proj".into(), "o_proj".into()] }
    }
}

/// One LoRA adapter on top of a frozen linear `W` (weight shape `[out, in]`). Forward: `W x + scaling * (B A) x`.
#[derive(Debug, Clone)]
pub struct LoraLinear {
    /// frozen base weight, `[out, in]`
    base_w: Tensor,
    /// trainable adapter matrix A, `[r, in]` (pub so a test can drive arbitrary values without extra setters)
    pub a: Var,
    /// trainable adapter matrix B, `[out, r]`; starts at zero so the adapter is inactive until trained (`y = W x`) (pub for tests)
    pub b: Var,
    a_shape: (usize, usize),
    b_shape: (usize, usize),
    scaling: f32,
    dev: Device,
}

impl LoraLinear {
    pub fn new(base_w: Tensor, cfg: &LoraConfig, dev: &Device) -> Res<Self> {
        let out = base_w.dim(0)?;
        let in_ = base_w.dim(1)?;
        // A starts non-zero (constant), B at zero: the very first backward has a nonzero gradient on B so both learn,
        // and untrained output equals the frozen base exactly.
        let a = Var::from_slice(&vec![0.5_f32; cfg.r * in_], (cfg.r, in_), dev)?;
        let b = Var::from_slice(&vec![0f32; out * cfg.r], (out, cfg.r), dev)?;
        Ok(Self { base_w, a, b, a_shape: (cfg.r, in_), b_shape: (out, cfg.r), scaling: cfg.scaling(), dev: dev.clone() })
    }

    pub fn forward(&self, x: &Tensor) -> Res<Tensor> {
        // frozen base projection: candle_nn Linear convention is weight `[out, in]`, applied as `x W^T`
        let base = x.matmul(&self.base_w.t()?)?;
        // adapter delta: `(B A) x`, scaling folded into the product so one scalar mul closes it out
        let ba = self.b.clone().matmul(&self.a.clone())?; // `[out, in]`
        let delta = (x.matmul(&ba.t()?)? * self.scaling as f64)?;
        base.add(&delta).map_err(Into::into)
    }

    /// Overwrite the B matrix from flat f32 (row-major). Used by tests to pin the delta math; in real training candle's
    /// optimizer updates it in place.
    pub fn set_b_values(&mut self, vals: &[f32]) -> Res<()> {
        let v = Var::from_slice(vals, self.b_shape, &self.dev)?;
        self.b.set(v.as_tensor())?;
        Ok(())
    }

    /// `(name, shape)` for each trainable matrix (A then B), in PEFT order, so a checkpoint reads back identically.
    pub fn param_refs(&self) -> Vec<(&'static str, (usize, usize), &Var)> {
        vec![("lora_A.weight", self.a_shape, &self.a), ("lora_B.weight", self.b_shape, &self.b)]
    }
}

/// A tiny 2-layer Llama decoder block + full model built on frozen base weights and LoRA attention adapters.
#[derive(Debug, Clone)]
pub struct TinyLlama {
    pub num_layers: usize,
    pub hidden: usize,
    pub num_heads: usize,
    norm_final: Tensor,                 // `[hidden]`, `model.norm.weight` (frozen)
    /// per layer, in order: `[input_layernorm, post_layernorm]` (`[hidden]` each, frozen)
    norms: Vec<[Tensor; 2]>,
    /// per layer, in order: `[gate, up, down]` for the MLP (frozen)
    mlp: Vec<[Tensor; 3]>,
    /// per layer, in order: `[q, k, v, o]` LoRA adapters (pub so a test can drive the adapter values without extra setters)
    pub attn: Vec<[LoraLinear; 4]>,
    embed: Tensor,                      // `[vocab, hidden]`, also the tied LM head (frozen)
    dev: Device,
}

impl TinyLlama {
    /// Assemble the model from a frozen base's tensors (`key -> weight tensor`) and LoRA config. Keys are an HF
    /// checkpoint's names: `model.embed_tokens.weight`, `model.layers.{i}.{input,output}_layernorm.weight`,
    /// `...self_attn.{q,k,v,o}_proj.weight`, `...mlp.{gate,up,down}_proj.weight`, `model.norm.weight`.
    pub fn load(base: &BTreeMap<String, Tensor>, cfg: &LoraConfig, dev: &Device) -> Res<Self> {
        let hidden = base.get("model.embed_tokens.weight").ok_or_else(|| anyhow!("no embed"))?.dim(1)?;
        let num_layers = 2; // the proof model has exactly two layers (see testmodel::TinyLlama)
        let n_heads = 2;
        let mut norms = Vec::with_capacity(num_layers);
        let mut mlp = Vec::with_capacity(num_layers);
        let mut attn = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let p = |k: &str| format!("model.layers.{i}.{k}");
            norms.push([
                base.get(&p("input_layernorm.weight")).ok_or_else(|| anyhow!("missing {}", p("input_layernorm.weight")))?.clone(),
                base.get(&p("post_layernorm.weight")).ok_or_else(|| anyhow!("missing {}", p("post_layernorm.weight")))?.clone(),
            ]);
            mlp.push([
                base.get(&p("mlp.gate_proj.weight")).ok_or_else(|| anyhow!("missing {}", p("mlp.gate_proj.weight")))?.clone(),
                base.get(&p("mlp.up_proj.weight")).ok_or_else(|| anyhow!("missing {}", p("mlp.up_proj.weight")))?.clone(),
                base.get(&p("mlp.down_proj.weight")).ok_or_else(|| anyhow!("missing {}", p("mlp.down_proj.weight")))?.clone(),
            ]);
            let q = LoraLinear::new(base.get(&p("self_attn.q_proj.weight")).ok_or_else(|| anyhow!("missing q_proj"))?.clone(), cfg, dev)?;
            let k = LoraLinear::new(base.get(&p("self_attn.k_proj.weight")).ok_or_else(|| anyhow!("missing k_proj"))?.clone(), cfg, dev)?;
            let v = LoraLinear::new(base.get(&p("self_attn.v_proj.weight")).ok_or_else(|| anyhow!("missing v_proj"))?.clone(), cfg, dev)?;
            let o = LoraLinear::new(base.get(&p("self_attn.o_proj.weight")).ok_or_else(|| anyhow!("missing o_proj"))?.clone(), cfg, dev)?;
            attn.push([q, k, v, o]);
        }
        Ok(Self { num_layers, hidden, num_heads: n_heads, norm_final: base.get("model.norm.weight").ok_or_else(|| anyhow!("no norm"))?.clone(), norms, mlp, attn, embed: base.get("model.embed_tokens.weight").ok_or_else(|| anyhow!("no embed"))?.clone(), dev: dev.clone() })
    }

    /// Every adapter matrix as `(layer, module, (rows, cols), &Var)`, so a checkpoint is serializable in order.
    pub fn lora_params(&self) -> Vec<(usize /*layer*/, &'static str, (usize, usize), &Var)> {
        let mut out = Vec::new();
        for (li, layer) in self.attn.iter().enumerate() {
            // (&mod_, &LoraLinear): destructure the borrowed pairs without requiring them to be Copy
            for (mod_, lin) in [("q_proj", &layer[0]), ("k_proj", &layer[1]), ("v_proj", &layer[2]), ("o_proj", &layer[3])] {
                for (_nm, shape, var) in lin.param_refs() {
                    out.push((li, mod_, shape, var));
                }
            }
        }
        out
    }

    /// The PEFT adapter tensor names and their shapes — enough to validate the adapter's PEFT compatibility.
    pub fn param_shapes(&self) -> Vec<(String, (usize, usize))> {
        self.lora_params()
            .into_iter()
            .map(|(li, mod_, shape, _)| (format!("base_model.model.model.layers.{li}.self_attn.{mod_}"), shape))
            .collect()
    }

    fn rmsnorm(&self, x: &Tensor, gamma: &Tensor) -> Res<Tensor> {
        // inputs are `[rows, hidden]`; reduce squares over the last (hidden) dim, keep it as a column
        let d = x.dim(1)? as f64;                        // hidden size
        let summed = x.sqr()?.sum_keepdim(1)?;           // [rows, 1], sum of squares per row
        let mean_sq = (summed * (1.0 / d))?;             // [rows, 1], mean of squares
        // normalized then gain-scaled: multiply by 1/sqrt(mean-of-squares); epsilon omitted (negligible for this proof)
        let inv_rms = mean_sq.powf(-0.5)?;               // [rows, 1]
        let normed = x.mul(&inv_rms)?;                   // [rows, hidden] broadcasts over the trailing dim
        normed.mul(gamma).map_err(Into::into)            // gain-scale: [rows, hidden] * [hidden] broadcasts to [rows, hidden]
    }

    /// Multi-head attention with LoRA on q/k/v/o. This tiny proof model omits rotary embeddings (noted in the plan).
    fn attn_layer(&self, x: &Tensor, li: usize) -> Res<Tensor> {
        let head_dim = self.hidden / self.num_heads;
        let scale = 1.0 / (head_dim as f32);
        let mut out = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let xb = x.chunk(self.num_heads, 1)?; // split `[seq, hidden]` into `num_heads` head slices along dim 1
            let q = self.attn[li][0].forward(&xb[h])?;
            let k = self.attn[li][1].forward(&xb[h])?;
            let v = self.attn[li][2].forward(&xb[h])?;
            // scaled dot-product attention over the sequence dimension of keys/values
            let scores = (q.matmul(&k.t()?)? * scale as f64)?; // `[seq, seq]`
            let a = softmax(&scores)?;
            out.push(a.matmul(&v)?); // `[seq, head_dim]`
        }
        let heads = Tensor::cat(&out, 1)?; // back to `[seq, hidden]`, then the (LoRA'd) o projection mixes heads out
        self.attn[li][3].forward(&heads)
    }

    fn mlp_layer(&self, x: &Tensor, li: usize) -> Res<Tensor> {
        let (gate, up, down) = (&self.mlp[li][0], &self.mlp[li][1], &self.mlp[li][2]);
        // SwiGLU-style gate: `down(silu(gate x) * (up x))`, no biases in the fixture
        let g = x.matmul(&gate.t()?)?;
        let u = x.matmul(&up.t()?)?;
        let up_g = silu(&g)?;
        let gated = up_g.mul(&u)?;
        down.matmul(&gated).map_err(|e| anyhow!("mlp down: {e}").into())
    }

    fn decoder_layer(&self, x: &Tensor, li: usize) -> Res<Tensor> {
        // pre-norm llama block with residuals (no attention bias in the fixture)
        let nx = self.rmsnorm(x, &self.norms[li][0])?;
        let x = x.add(&self.attn_layer(&nx, li)?)?;
        let nx = self.rmsnorm(&x, &self.norms[li][1])?;
        x.add(&self.mlp_layer(&nx, li)?).map_err(Into::into)
    }

    /// Token ids `[seq]` (int64) -> logits `[seq, vocab]`. Training-friendly: every position produces a next-token logit.
    pub fn forward(&self, tokens: &Tensor) -> Res<Tensor> {
        let emb = Tensor::index_select(&self.embed, tokens, 0)?; // `[seq, hidden]`
        let mut h = emb;
        for li in 0..self.num_layers {
            h = self.decoder_layer(&h, li)?;
        }
        let h = self.rmsnorm(&h, &self.norm_final)?;
        // tied LM head: logits = h @ embed^T
        h.matmul(&self.embed.t()?).map_err(Into::into)
    }

    /// Snapshot every adapter matrix as flat f32 in PEFT name order (`...q_proj.lora_A.weight` then `.lora_B.weight`,
    /// layer by layer), each with its shape, so a checkpoint round-trips exactly.
    pub fn snapshot(&self) -> Res<Vec<(String, (usize, usize), Vec<f32>)>> {
        let mut full = Vec::new();
        for (li, layer) in self.attn.iter().enumerate() {
            // (&mod_, &LoraLinear): destructure the borrowed pairs without requiring them to be Copy
            for (mod_, lin) in [("q_proj", &layer[0]), ("k_proj", &layer[1]), ("v_proj", &layer[2]), ("o_proj", &layer[3])] {
                let nm = format!("base_model.model.model.layers.{li}.self_attn.{mod_}");
                // param_refs yields lora_A (shape [r, in]) then lora_B ([out, r]) for this module
                for (nm_suf, shape, var) in lin.param_refs() {
                    // A and B are 2-D ([r,in] / [out,r]); flatten before reading as a flat f32 buffer so the checkpoint is row-major.
                    let vals = var.as_tensor().flatten_all()?.to_vec1::<f32>()?;
                    full.push((format!("{nm}.{nm_suf}"), shape, vals));
                }
            }
        }
        Ok(full)
    }

    /// Load a snapshot produced by [`snapshot`] back into the adapter matrices, restoring training state from disk.
    pub fn load_snapshot(&mut self, snap: &[(String, (usize, usize), Vec<f32>)]) -> Res<()> {
        let mut idx = 0usize;
        for li in 0..self.num_layers {
            // layer's attention projections in PEFT order: q,k,v,o, each with an A (shape [r,in]) then a B ([out,r])
            for mi in 0..4 {
                let lin = &mut self.attn[li][mi];
                let (_anm, a_exp, a_vals) = &snap[idx];
                idx += 1;
                if *a_exp != lin.a_shape {
                    return Err(anyhow!("lora_A shape {:?} != saved {:?}", lin.a_shape, *a_exp).into());
                }
                let va = Var::from_slice(a_vals.as_slice(), lin.a_shape, &self.dev)?;
                lin.a.set(va.as_tensor())?;
                let (_bnm, b_exp, b_vals) = &snap[idx];
                idx += 1;
                if *b_exp != lin.b_shape {
                    return Err(anyhow!("lora_B shape {:?} != saved {:?}", lin.b_shape, *b_exp).into());
                }
                let vb = Var::from_slice(b_vals.as_slice(), lin.b_shape, &self.dev)?;
                lin.b.set(vb.as_tensor())?;
            }
        }
        Ok(())
    }
}

fn softmax(x: &Tensor) -> Res<Tensor> {
    let m = x.max(1)?; // `[B]` over sequence dim
    let e = x.sub(&m)?.exp()?;
    let s = e.sum(1)?;
    e.div(&s).map_err(Into::into)
}

fn silu(x: &Tensor) -> Res<Tensor> {
    // x * sigmoid(x), with sigmoid computed as e^-x / (1 + e^-x); no unary-neg on a Tensor, no dedicated activation op
    let nx = (x.clone() * -1.0_f64)?;            // -x via scalar mul (Tensor does not impl Neg)
    let enx = nx.exp()?;                          // e^-x
    let den = Tensor::ones(x.shape(), x.dtype(), x.device())?.add(&enx)?;  // 1 + e^-x : x.shape() already yields &Shape
    let sig = den.powf(-1.0)?;                    // sigmoid
    x.mul(&sig).map_err(Into::into)
}
