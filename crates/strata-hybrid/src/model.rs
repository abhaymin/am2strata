//! The reference forward pass on the CPU, from the quantized GGUF weights (memory-mapped, decoded row by row), for the
//! architectures this engine runs:
//!   - `nemotron_h_moe` / `nemotron_h` (NVIDIA Nemotron-H: Mamba-2, attention without positions, latent MoE or relu^2
//!     feed-forwards; one block per layer, its kind read as llama.cpp's nemotron-h.cpp reads it),
//!   - `llama` (Mistral, Devstral, Mixtral: GQA with RoPE, a SiLU-gated feed-forward or a softmax-routed MoE),
//!   - `mistral3` (Ministral 3, Mistral Small / Medium 3.x: `llama` plus YaRN and the attention temperature),
//!   - `qwen3` / `qwen3moe` (Qwen3 dense and MoE: per-head q/k RMS norms, NEOX RoPE; the MoE is Mixtral's softmax
//!     routing, so these models also test that path),
//!   - `deepseek2` (GLM-4.7-Flash, DeepSeek V2/V3 and their family: MLA attention with the absorbed K/V
//!     projections, leading dense layers, then sigmoid- or softmax-routed experts with a shared expert).
//! Each layer is a list of (norm, block) pairs applied as x += block(rms_norm(x)); every formula follows llama.cpp,
//! which the parity tests compare against (tools/reference-llamacpp).

use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use strata_gguf::{Gguf, TensorInfo};

/// One weight matrix in the GGUF: `rows` output features of `cols` inputs, in its stored encoding.
#[derive(Clone)]
pub struct QMat {
    pub data: &'static [u8],
    ty: String,
    vdt: Vdt,
    pub rows: usize,
    pub cols: usize,
    row_bytes: usize,
}

impl QMat {
    fn row_bytes(ty: &str, cols: usize) -> usize {
        let (be, bb) = strata_gguf::block_geometry(ty).unwrap_or((1, 4));
        cols / be as usize * bb as usize
    }

    /// Slice `index` (an expert, or a head) of a 3-D tensor stored as `rows`-row matrices, as a matrix of its own.
    pub fn slice(&self, index: usize, rows: usize) -> QMat {
        let start = index * rows * self.row_bytes;
        QMat { data: &self.data[start..start + rows * self.row_bytes], ty: self.ty.clone(), vdt: self.vdt, rows, cols: self.cols, row_bytes: self.row_bytes }
    }

    /// y = W x, rows in parallel.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(y.len(), self.rows);
        let xq = self.vdt.round_trip(x);
        let x = &xq[..];
        let chunk = (self.rows / (rayon::current_num_threads() * 4)).max(16);
        y.par_chunks_mut(chunk).enumerate().for_each(|(ci, ys)| {
            let mut row = vec![0f32; self.cols];
            for (k, out) in ys.iter_mut().enumerate() {
                let r = ci * chunk + k;
                let bytes = &self.data[r * self.row_bytes..(r + 1) * self.row_bytes];
                strata_pack::quants::dequantize_into(&self.ty, bytes, &mut row).expect("weight row");
                *out = dot(&row, x);
            }
        });
    }

    /// The same matrix stored differently (a pack's copy of a float tensor).
    pub fn with_data(&self, data: &'static [u8], ty: &str) -> Result<QMat, String> {
        let row_bytes = QMat::row_bytes(ty, self.cols);
        if data.len() != row_bytes * self.rows {
            return Err(format!("{} bytes of {ty} for a {}x{} tensor", data.len(), self.rows, self.cols));
        }
        Ok(QMat { data, ty: ty.to_string(), vdt: Vdt::of(ty), rows: self.rows, cols: self.cols, row_bytes })
    }

    fn mul(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0f32; self.rows];
        self.matvec(x, &mut y);
        y
    }

    /// One row as f32 (an embedding).
    pub fn row(&self, r: usize) -> Vec<f32> {
        let mut out = vec![0f32; self.cols];
        strata_pack::quants::dequantize_into(&self.ty, &self.data[r * self.row_bytes..(r + 1) * self.row_bytes], &mut out).expect("row");
        out
    }

    fn all_f32(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.rows * self.cols];
        strata_pack::quants::dequantize_into(&self.ty, self.data, &mut out).expect("tensor");
        out
    }
}

/// The encoding ggml's CPU backend converts the activations to before a dot product with a weight of a given type
/// (type_traits_cpu[..].vec_dot_type). Doing the same - quantize, then compute with the values that survive - keeps
/// this engine's logits within float rounding of llama.cpp's, which is what the parity tests compare.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Vdt {
    F32,
    F16,
    Bf16,
    /// blocks of 32, a scale stored as f16, round half away from zero (quantize_row_q8_0_ref / q8_1)
    Q8_0,
    /// blocks of 256, an f32 scale from the signed maximum, round half to even (quantize_row_q8_K_ref)
    Q8K,
}

impl Vdt {
    pub fn of(ty: &str) -> Vdt {
        match ty {
            "F32" => Vdt::F32,
            "F16" => Vdt::F16,
            "BF16" => Vdt::Bf16,
            "Q4_0" | "Q4_1" | "Q5_0" | "Q5_1" | "Q8_0" | "Q8_1" | "IQ4_NL" | "MXFP4" | "NVFP4" => Vdt::Q8_0,
            _ => Vdt::Q8K,
        }
    }

    pub fn round_trip(self, x: &[f32]) -> Vec<f32> {
        use strata_pack::num::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};
        match self {
            Vdt::F32 => x.to_vec(),
            Vdt::F16 => x.iter().map(|v| f16_to_f32(f32_to_f16(*v))).collect(),
            Vdt::Bf16 => x.iter().map(|v| bf16_to_f32(f32_to_bf16(*v))).collect(),
            Vdt::Q8_0 if x.len() % 32 == 0 => {
                let mut out = Vec::with_capacity(x.len());
                for b in x.chunks(32) {
                    let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let d = amax / 127.0;
                    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                    let d16 = f16_to_f32(f32_to_f16(d));
                    out.extend(b.iter().map(|v| (v * id).round() * d16));
                }
                out
            }
            Vdt::Q8K if x.len() % 256 == 0 => {
                let mut out = Vec::with_capacity(x.len());
                for b in x.chunks(256) {
                    let (mut amax, mut max) = (0f32, 0f32);
                    for v in b {
                        if v.abs() > amax {
                            amax = v.abs();
                            max = *v;
                        }
                    }
                    if amax == 0.0 {
                        out.extend(std::iter::repeat(0.0).take(256));
                        continue;
                    }
                    let iscale = -127.0 / max;
                    let d = 1.0 / iscale;
                    out.extend(b.iter().map(|v| ((iscale * v).round_ties_even()).min(127.0) * d));
                }
                out
            }
            _ => x.to_vec(),
        }
    }
}

fn f16_round(v: &mut [f32]) {
    v.iter_mut().for_each(|x| *x = strata_pack::num::f16_to_f32(strata_pack::num::f32_to_f16(*x)));
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    // eight accumulators: vectorizes, and does not lose much to one long serial sum
    let mut acc = [0f32; 8];
    let n = a.len() / 8 * 8;
    for i in (0..n).step_by(8) {
        for j in 0..8 {
            acc[j] += a[i + j] * b[i + j];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for i in n..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let k = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * k * g).collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn relu_sqr(x: f32) -> f32 {
    let r = x.max(0.0);
    r * r
}

fn add_bias(v: &mut [f32], b: &Option<Vec<f32>>) {
    if let Some(b) = b {
        v.iter_mut().zip(b).for_each(|(a, c)| *a += c);
    }
}

/// Rotary position embedding as ggml computes it, with YaRN and llama3-style frequency factors.
pub struct Rope {
    n_dims: usize,
    base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    corr: [f32; 2],
    neox: bool,
    factors: Option<Vec<f32>>,
}

impl Rope {
    fn from_gguf(g: &Gguf, arch: &str, n_dims: usize, neox: bool, factors: Option<Vec<f32>>) -> Rope {
        let f = |n: &str| g.metadata.get(&format!("{arch}.{n}")).and_then(|v| v.as_f64()).map(|v| v as f32);
        let base = f("rope.freq_base").unwrap_or(10000.0);
        let kind = g.metadata.get(&format!("{arch}.rope.scaling.type")).and_then(|v| v.as_str()).unwrap_or("none").to_string();
        let factor = f("rope.scaling.factor").filter(|v| *v > 0.0).unwrap_or(1.0);
        let freq_scale = if kind == "linear" || kind == "yarn" { 1.0 / factor } else { 1.0 };
        let n_ctx_orig = f("rope.scaling.original_context_length").or_else(|| f("context_length")).unwrap_or(4096.0);
        let (beta_fast, beta_slow) = (f("rope.scaling.yarn_beta_fast").unwrap_or(32.0), f("rope.scaling.yarn_beta_slow").unwrap_or(1.0));
        let log_mul = f("rope.scaling.yarn_log_multiplier").unwrap_or(0.0);
        // llama_context's defaults (llama-context.cpp): YaRN extrapolates fully, and its magnitude correction is
        // folded into attn_factor so that ggml's own (1 + 0.1 ln s) cancels out
        let ext_factor = if kind == "yarn" { 1.0 } else { 0.0 };
        let mut attn_factor = 1.0f32;
        if ext_factor != 0.0 {
            let gm = |s: f32, m: f32| if s <= 1.0 { 1.0 } else { 0.1 * m * s.ln() + 1.0 };
            let fac = 1.0 / freq_scale;
            attn_factor = if log_mul != 0.0 {
                let m = if arch == "deepseek2" && log_mul != 1.0 { log_mul } else { 1.0 };
                gm(fac, m) / gm(fac, log_mul)
            } else {
                gm(fac, 1.0)
            };
            attn_factor *= 1.0 / (1.0 + 0.1 * fac.ln());
        }
        attn_factor *= f("rope.scaling.attn_factor").unwrap_or(1.0);
        let corr_dim = |rot: f32| n_dims as f32 * (n_ctx_orig / (rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln());
        let corr = [corr_dim(beta_fast).floor().max(0.0), corr_dim(beta_slow).ceil().min(n_dims as f32 - 1.0)];
        Rope { n_dims, base, freq_scale, ext_factor, attn_factor, corr, neox, factors }
    }

    /// Rotates the first n_dims values of one head's vector for position `pos`.
    fn apply(&self, x: &mut [f32], pos: usize) {
        let theta_scale = self.base.powf(-2.0 / self.n_dims as f32);
        let mut theta = pos as f32;
        let half = self.n_dims / 2;
        for i0 in (0..self.n_dims).step_by(2) {
            let ff = self.factors.as_ref().map(|f| f[i0 / 2]).unwrap_or(1.0);
            let extrap = theta / ff;
            let interp = self.freq_scale * extrap;
            let (mut th, mut mscale) = (interp, self.attn_factor);
            if self.ext_factor != 0.0 {
                let y = ((i0 / 2) as f32 - self.corr[0]) / (self.corr[1] - self.corr[0]).max(0.001);
                let ramp = (1.0 - y.clamp(0.0, 1.0)) * self.ext_factor;
                th = interp * (1.0 - ramp) + extrap * ramp;
                mscale *= 1.0 + 0.1 * (1.0 / self.freq_scale).ln();
            }
            let (c, s) = (th.cos() * mscale, th.sin() * mscale);
            let (a, b) = if self.neox { (i0 / 2, i0 / 2 + half) } else { (i0, i0 + 1) };
            let (x0, x1) = (x[a], x[b]);
            x[a] = x0 * c - x1 * s;
            x[b] = x0 * s + x1 * c;
            theta *= theta_scale;
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Act {
    Silu,
    ReluSqr,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Gating {
    Softmax,
    Sigmoid,
}

/// A feed-forward: down(act(gate x) * up x), or down(act(up x)) without a gate.
pub struct Dense {
    gate: Option<QMat>,
    up: QMat,
    down: QMat,
    act: Act,
}

impl Dense {
    fn forward(&self, h: &[f32]) -> Vec<f32> {
        let mut mid = self.up.mul(h);
        match &self.gate {
            Some(g) => {
                let gv = g.mul(h);
                mid.iter_mut().zip(&gv).for_each(|(u, g)| *u *= act(self.act, *g));
            }
            None => mid.iter_mut().for_each(|v| *v = act(self.act, *v)),
        }
        self.down.mul(&mid)
    }
}

fn act(a: Act, x: f32) -> f32 {
    match a {
        Act::Silu => silu(x),
        Act::ReluSqr => relu_sqr(x),
    }
}

pub struct Moe {
    /// the layer's index in the expert profile
    pub layer: usize,
    router: Vec<f32>,
    bias: Option<Vec<f32>>,
    gating: Gating,
    pub n_expert: usize,
    top_k: usize,
    w_norm: bool,
    w_scale: f32,
    /// Nemotron's latent MoE: experts work in a smaller space between these two projections
    latent: Option<(QMat, QMat)>,
    pub gate_exps: Option<QMat>,
    pub up_exps: QMat,
    pub down_exps: QMat,
    pub ff: usize,
    act: Act,
    shared: Option<Dense>,
}

pub struct Attn {
    q: QMat,
    k: QMat,
    v: QMat,
    o: QMat,
    q_b: Option<Vec<f32>>,
    k_b: Option<Vec<f32>>,
    v_b: Option<Vec<f32>>,
    o_b: Option<Vec<f32>>,
    /// per-head RMS norms of q and k before RoPE (Qwen3)
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    /// no positions at all (Nemotron-H)
    rope: bool,
}

/// DeepSeek's multi-head latent attention, with the K and V up-projections absorbed (as llama.cpp's is_mla path):
/// the cache holds the normalized latent (kv_lora values) and the shared rotary key per position.
pub struct Mla {
    q_a: Option<(QMat, Vec<f32>)>,
    q_b: QMat,
    kv_a: QMat,
    kv_norm: Vec<f32>,
    k_b: QMat,
    v_b: QMat,
    o: QMat,
    n_head: usize,
    nope: usize,
    rope_dim: usize,
    kv_lora: usize,
    v_dim: usize,
    kq_scale: f32,
}

pub struct Mamba {
    ssm_in: QMat,
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    dt_b: Vec<f32>,
    a: Vec<f32>,
    d: Vec<f32>,
    ssm_norm: Vec<f32>,
    ssm_out: QMat,
}

pub enum Block {
    Mamba(Mamba),
    Attn(Attn),
    Mla(Mla),
    Dense(Dense),
    Moe(Moe),
}

pub struct Layer {
    pub parts: Vec<(Vec<f32>, Block)>,
}

pub struct Model {
    pub arch: String,
    pub n_embd: usize,
    d_inner: usize,
    d_state: usize,
    n_ssm_head: usize,
    n_group: usize,
    d_conv: usize,
    eps: f32,
    rope: Option<Rope>,
    /// Llama 4 / Mistral 3 attention temperature: (scale, floor) - q *= 1 + scale * ln(floor(pos / floor) + 1)
    attn_temp: Option<(f32, f32)>,
    pub stops: Vec<u32>,
    embd: QMat,
    out_norm: Vec<f32>,
    output: QMat,
    pub layers: Vec<Layer>,
    /// the expert profile's shape: one row per layer, `profile_experts` columns
    pub profile_experts: usize,
    /// how often each (layer, expert) was routed to, for the expert profile
    pub heat: Vec<AtomicU32>,
}

/// Per-conversation state: per layer the attention cache (K, V - or MLA's latent and rotary key), and the Mamba
/// layers' conv window and SSM state.
pub struct State {
    pub pos: usize,
    kv: HashMap<usize, (Vec<f32>, Vec<f32>)>,
    conv: HashMap<usize, Vec<f32>>,
    ssm: HashMap<usize, Vec<f32>>,
}

fn arr_u64(g: &Gguf, key: &str, i: usize) -> u64 {
    match g.metadata.get(key) {
        Some(serde_json::Value::Array(a)) => a.get(i).and_then(|v| v.as_u64()).unwrap_or(0),
        Some(v) => v.as_u64().unwrap_or(0),
        None => 0,
    }
}

pub const ARCHS: &[&str] = &["nemotron_h_moe", "nemotron_h", "llama", "mistral3", "qwen3", "qwen3moe", "deepseek2"];

impl Model {
    /// Loads the GGUF (memory-mapped); with a pack, its float tensors replace the GGUF's. Also returns how many
    /// tensors came from the pack.
    pub fn load(first: &Path, pack: Option<&crate::pack::Pack>) -> Result<(Model, usize), String> {
        let shards = strata_gguf::shards(first);
        let mut tensors: HashMap<String, QMat> = HashMap::new();
        let mut offsets: HashMap<String, u64> = HashMap::new();
        let mut meta: Option<Gguf> = None;
        for p in &shards {
            let g = Gguf::open(p).map_err(|e| e.to_string())?;
            let f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
            let mm: &'static memmap2::Mmap = Box::leak(Box::new(unsafe { memmap2::Mmap::map(&f) }.map_err(|e| e.to_string())?));
            for t in &g.tensors {
                tensors.insert(t.name.clone(), qmat(mm, &g, t)?);
                offsets.insert(t.name.clone(), g.data_start + t.offset);
            }
            if meta.is_none() {
                meta = Some(g);
            }
        }
        let g = meta.ok_or("no shards")?;
        let from_pack = match pack {
            Some(p) => p.apply(&mut tensors, &offsets)?,
            None => 0,
        };
        let arch = g.architecture().unwrap_or("").to_string();
        if !ARCHS.contains(&arch.as_str()) {
            return Err(format!("this engine runs {} models, not {arch}", ARCHS.join(", ")));
        }
        let k = |n: &str| format!("{arch}.{n}");
        let u = |n: &str| g.u64(&k(n)).ok_or_else(|| format!("{} is missing", k(n)));
        let fl = |n: &str| g.metadata.get(&k(n)).and_then(|v| v.as_f64()).map(|v| v as f32);
        let take = |name: &str| tensors.get(name).cloned().ok_or_else(|| format!("tensor {name} is missing"));
        let opt = |name: &str| tensors.get(name).cloned();
        let vecf = |name: &str| take(name).map(|t| t.all_f32());
        let optf = |name: &str| opt(name).map(|t| t.all_f32());
        let n_layer = u("block_count")? as usize;
        let n_embd = u("embedding_length")? as usize;
        let eps = fl("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let w_scale = fl("expert_weights_scale").unwrap_or(0.0);
        let w_norm_meta = g.metadata.get(&k("expert_weights_norm")).and_then(|v| v.as_bool());
        let gating = match u("expert_gating_func").unwrap_or(1) {
            2 => Gating::Sigmoid,
            _ => Gating::Softmax,
        };
        let nemotron = arch.starts_with("nemotron_h");
        let mut layers = Vec::with_capacity(n_layer);
        let mut max_expert = 0usize;
        let dense_lead = u("leading_dense_block_count").unwrap_or(0) as usize;
        for l in 0..n_layer {
            let p = |n: &str| format!("blk.{l}.{n}");
            let n_head = arr_u64(&g, &k("attention.head_count"), l) as usize;
            let n_kv = arr_u64(&g, &k("attention.head_count_kv"), l) as usize;
            let head_dim = u("attention.key_length").map(|v| v as usize).unwrap_or(if n_head > 0 { n_embd / n_head } else { 0 });
            let attn = |rope: bool| -> Result<Block, String> {
                Ok(Block::Attn(Attn {
                    q: take(&p("attn_q.weight"))?, k: take(&p("attn_k.weight"))?, v: take(&p("attn_v.weight"))?,
                    o: take(&p("attn_output.weight"))?, q_b: optf(&p("attn_q.bias")), k_b: optf(&p("attn_k.bias")),
                    v_b: optf(&p("attn_v.bias")), o_b: optf(&p("attn_output.bias")), q_norm: optf(&p("attn_q_norm.weight")),
                    k_norm: optf(&p("attn_k_norm.weight")), n_head, n_kv, head_dim, rope,
                }))
            };
            // the feed-forward of a layer that has one: experts when there is a router, else dense
            let ffn = |act_: Act, max_expert: &mut usize| -> Result<Block, String> {
                let Some(router) = opt(&p("ffn_gate_inp.weight")) else {
                    return Ok(Block::Dense(Dense { gate: opt(&p("ffn_gate.weight")), up: take(&p("ffn_up.weight"))?, down: take(&p("ffn_down.weight"))?, act: act_ }));
                };
                let up_exps = take(&p("ffn_up_exps.weight"))?;
                let n_expert = router.rows;
                *max_expert = (*max_expert).max(n_expert);
                let shared = match opt(&p("ffn_up_shexp.weight")) {
                    Some(up) => Some(Dense { gate: opt(&p("ffn_gate_shexp.weight")), up, down: take(&p("ffn_down_shexp.weight"))?, act: act_ }),
                    None => None,
                };
                let latent = match (opt(&p("ffn_latent_down.weight")), opt(&p("ffn_latent_up.weight"))) {
                    (Some(a), Some(b)) => Some((a, b)),
                    _ => None,
                };
                if opt(&p("ffn_gate_up_exps.weight")).is_some() {
                    return Err(format!("layer {l}: merged gate_up experts are not supported yet"));
                }
                Ok(Block::Moe(Moe {
                    layer: l, router: router.all_f32(), bias: optf(&p("exp_probs_b.bias")),
                    gating: if nemotron { Gating::Sigmoid } else if arch == "deepseek2" { gating } else { Gating::Softmax },
                    n_expert, top_k: arr_u64(&g, &k("expert_used_count"), l) as usize,
                    // llama.cpp: the llama family always renormalizes the chosen experts' weights
                    w_norm: if matches!(arch.as_str(), "llama" | "mistral3" | "qwen3moe") { true } else { w_norm_meta.unwrap_or(false) },
                    w_scale, latent, gate_exps: opt(&p("ffn_gate_exps.weight")), ff: up_exps.rows / n_expert, up_exps,
                    down_exps: take(&p("ffn_down_exps.weight"))?, act: act_, shared,
                }))
            };
            let mut parts = Vec::new();
            if nemotron {
                // one block per layer: recurrent with neither heads nor a feed-forward, attention with KV heads
                let norm = vecf(&p("attn_norm.weight"))?;
                let ff = arr_u64(&g, &k("feed_forward_length"), l) as usize;
                let block = if n_kv == 0 && ff == 0 {
                    Block::Mamba(Mamba {
                        ssm_in: take(&p("ssm_in.weight"))?, conv_w: vecf(&p("ssm_conv1d.weight"))?, conv_b: vecf(&p("ssm_conv1d.bias"))?,
                        dt_b: vecf(&p("ssm_dt.bias"))?, a: vecf(&p("ssm_a"))?, d: vecf(&p("ssm_d"))?,
                        ssm_norm: vecf(&p("ssm_norm.weight"))?, ssm_out: take(&p("ssm_out.weight"))?,
                    })
                } else if ff == 0 {
                    attn(false)?
                } else {
                    ffn(Act::ReluSqr, &mut max_expert)?
                };
                parts.push((norm, block));
            } else if arch == "deepseek2" {
                let rope_dim = u("rope.dimension_count")? as usize;
                let k_mla = u("attention.key_length_mla").or_else(|_| u("attention.key_length"))? as usize;
                let v_dim = u("attention.value_length_mla").or_else(|_| u("attention.value_length"))? as usize;
                let kv_lora = u("attention.kv_lora_rank")? as usize;
                let q_a = match opt(&p("attn_q_a.weight")) {
                    Some(a) => Some((a, vecf(&p("attn_q_a_norm.weight"))?)),
                    None => None,
                };
                let q_b = if q_a.is_some() { take(&p("attn_q_b.weight"))? } else { take(&p("attn_q.weight"))? };
                if opt(&p("attn_k_b.weight")).is_none() {
                    return Err("this deepseek2 GGUF has the old attn_kv_b tensor; re-convert it with a llama.cpp from 2025 or later".into());
                }
                // llama.cpp's deepseek2 graph pre-scales kq by YaRN's magnitude, as DeepSeek's reference does
                let fs = 1.0 / fl("rope.scaling.factor").filter(|v| *v > 0.0).unwrap_or(1.0);
                let rope_tmp = Rope::from_gguf(&g, &arch, rope_dim, false, None);
                let af_org = rope_tmp.attn_factor * (1.0 + 0.1 * (1.0 / fs).ln());
                let mscale = af_org * (1.0 + 0.1 * fl("rope.scaling.yarn_log_multiplier").unwrap_or(0.0) * (1.0 / fs).ln());
                parts.push((vecf(&p("attn_norm.weight"))?, Block::Mla(Mla {
                    q_a, q_b, kv_a: take(&p("attn_kv_a_mqa.weight"))?, kv_norm: vecf(&p("attn_kv_a_norm.weight"))?,
                    k_b: take(&p("attn_k_b.weight"))?, v_b: take(&p("attn_v_b.weight"))?, o: take(&p("attn_output.weight"))?,
                    n_head, nope: k_mla - rope_dim, rope_dim, kv_lora, v_dim, kq_scale: mscale * mscale / (k_mla as f32).sqrt(),
                })));
                let f = if l < dense_lead {
                    Block::Dense(Dense { gate: opt(&p("ffn_gate.weight")), up: take(&p("ffn_up.weight"))?, down: take(&p("ffn_down.weight"))?, act: Act::Silu })
                } else {
                    ffn(Act::Silu, &mut max_expert)?
                };
                parts.push((vecf(&p("ffn_norm.weight"))?, f));
            } else {
                parts.push((vecf(&p("attn_norm.weight"))?, attn(true)?));
                parts.push((vecf(&p("ffn_norm.weight"))?, ffn(Act::Silu, &mut max_expert)?));
            }
            layers.push(Layer { parts });
        }
        let rope = if nemotron {
            None
        } else {
            // llama.cpp's default n_rot is the head size (attention.key_length when given), not n_embd / n_head
            let n_head0 = arr_u64(&g, &k("attention.head_count"), 0).max(1) as usize;
            let head_k = u("attention.key_length").map(|v| v as usize).unwrap_or(n_embd / n_head0);
            let n_rot = u("rope.dimension_count").map(|v| v as usize).unwrap_or(head_k);
            // llama.cpp's rope types: NORM (adjacent pairs) for the llama family and deepseek2, NEOX (halves) for Qwen
            let neox = arch.starts_with("qwen");
            Some(Rope::from_gguf(&g, &arch, n_rot, neox, optf("rope_freqs.weight")))
        };
        let attn_temp = match fl("attention.temperature_scale") {
            Some(s) if s != 0.0 && (arch == "mistral3" || arch == "deepseek2") => {
                let floor = fl("rope.scaling.original_context_length").ok_or("attention.temperature_scale needs rope.scaling.original_context_length")?;
                Some((s, floor))
            }
            _ => None,
        };
        let mut stops = vec![];
        for key in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
            if let Some(v) = g.u64(key) {
                if !stops.contains(&(v as u32)) {
                    stops.push(v as u32);
                }
            }
        }
        let embd = take("token_embd.weight")?;
        let output = opt("output.weight").unwrap_or_else(|| embd.clone());
        let profile_experts = (u("expert_count").unwrap_or(0) as usize).max(max_expert);
        Ok((Model {
            n_embd,
            d_inner: u("ssm.inner_size").unwrap_or(0) as usize, d_state: u("ssm.state_size").unwrap_or(0) as usize,
            n_ssm_head: u("ssm.time_step_rank").unwrap_or(0) as usize, n_group: u("ssm.group_count").unwrap_or(1) as usize,
            d_conv: u("ssm.conv_kernel").unwrap_or(0) as usize, eps, rope, attn_temp, stops,
            embd, out_norm: vecf("output_norm.weight")?, output,
            heat: (0..n_layer * profile_experts).map(|_| AtomicU32::new(0)).collect(),
            profile_experts, layers, arch,
        }, from_pack))
    }

    pub fn new_state(&self) -> State {
        State { pos: 0, kv: HashMap::new(), conv: HashMap::new(), ssm: HashMap::new() }
    }

    /// One token through the model: the logits of the next one.
    pub fn forward(&self, st: &mut State, token: u32) -> Vec<f32> {
        let mut x = self.embd.row(token as usize);
        for (l, layer) in self.layers.iter().enumerate() {
            for (pi, (norm, block)) in layer.parts.iter().enumerate() {
                let h = rms_norm(&x, norm, self.eps);
                let out = match block {
                    Block::Mamba(b) => self.mamba(l, b, &h, st),
                    Block::Attn(b) => self.attention(l * 2 + pi, b, &h, st),
                    Block::Mla(b) => self.mla(l * 2 + pi, b, &h, st),
                    Block::Dense(b) => b.forward(&h),
                    Block::Moe(b) => self.moe(b, &h),
                };
                x.iter_mut().zip(&out).for_each(|(a, b)| *a += b);
            }
        }
        st.pos += 1;
        let h = rms_norm(&x, &self.out_norm, self.eps);
        self.output.mul(&h)
    }

    fn temp_scale(&self, pos: usize) -> f32 {
        match self.attn_temp {
            Some((s, floor)) => ((pos as f32 / floor).floor() + 1.0).ln() * s + 1.0,
            None => 1.0,
        }
    }

    fn mamba(&self, l: usize, m: &Mamba, h: &[f32], st: &mut State) -> Vec<f32> {
        let Mamba { ssm_in, conv_w, conv_b, dt_b, a, d, ssm_norm, ssm_out } = m;
        let (di, ds, nh, ng, dc) = (self.d_inner, self.d_state, self.n_ssm_head, self.n_group, self.d_conv);
        let n_conv = di + 2 * ng * ds;
        let head_dim = di / nh;
        let zx = ssm_in.mul(h);
        let (z, rest) = zx.split_at(di);
        let (xbc, dt) = rest.split_at(n_conv);
        // causal conv1d over the last d_conv positions (the window keeps d_conv - 1 earlier ones, oldest first)
        let win = st.conv.entry(l).or_insert_with(|| vec![0f32; (dc - 1) * n_conv]);
        let mut conv = vec![0f32; n_conv];
        for c in 0..n_conv {
            let mut s = 0f32;
            for t in 0..dc - 1 {
                s += win[t * n_conv + c] * conv_w[c * dc + t];
            }
            s += xbc[c] * conv_w[c * dc + dc - 1];
            conv[c] = silu(s + conv_b[c]);
        }
        win.copy_within(n_conv.., 0);
        let last = (dc - 2) * n_conv;
        win[last..last + n_conv].copy_from_slice(xbc);
        let (xs, bc) = conv.split_at(di);
        let (bm, cm) = bc.split_at(ng * ds);
        // the selective scan: one scalar decay per head, B and C shared by the heads of a group
        let state = st.ssm.entry(l).or_insert_with(|| vec![0f32; nh * head_dim * ds]);
        let mut y = vec![0f32; di];
        for hh in 0..nh {
            let dts = softplus(dt[hh] + dt_b[hh]);
            let da = (dts * a[hh]).exp();
            let g = hh / (nh / ng);
            let bg = &bm[g * ds..(g + 1) * ds];
            let cg = &cm[g * ds..(g + 1) * ds];
            for dd in 0..head_dim {
                let i = hh * head_dim + dd;
                let x_dt = xs[i] * dts;
                let s = &mut state[i * ds..(i + 1) * ds];
                let mut acc = 0f32;
                for n in 0..ds {
                    let v = s[n] * da + bg[n] * x_dt;
                    s[n] = v;
                    acc += v * cg[n];
                }
                y[i] = acc + xs[i] * d[hh];
            }
        }
        // gate with silu(z), then the grouped RMS norm
        for (v, zz) in y.iter_mut().zip(z) {
            *v *= silu(*zz);
        }
        let gs = di / ng;
        for g in 0..ng {
            let seg = &mut y[g * gs..(g + 1) * gs];
            let ms = seg.iter().map(|v| v * v).sum::<f32>() / gs as f32;
            let k = 1.0 / (ms + self.eps).sqrt();
            for (j, v) in seg.iter_mut().enumerate() {
                *v = *v * k * ssm_norm[g * gs + j];
            }
        }
        ssm_out.mul(&y)
    }

    fn attention(&self, slot: usize, a: &Attn, h: &[f32], st: &mut State) -> Vec<f32> {
        let hd = a.head_dim;
        let pos = st.pos;
        let (mut qv, mut kv, mut vv) = (a.q.mul(h), a.k.mul(h), a.v.mul(h));
        add_bias(&mut qv, &a.q_b);
        add_bias(&mut kv, &a.k_b);
        add_bias(&mut vv, &a.v_b);
        let eps = self.eps;
        if let Some(w) = &a.q_norm {
            qv.chunks_mut(hd).for_each(|q| q.copy_from_slice(&rms_norm(q, w, eps)));
        }
        if let Some(w) = &a.k_norm {
            kv.chunks_mut(hd).for_each(|k| k.copy_from_slice(&rms_norm(k, w, eps)));
        }
        if a.rope {
            let rope = self.rope.as_ref().expect("rope");
            qv.chunks_mut(hd).for_each(|q| rope.apply(q, pos));
            kv.chunks_mut(hd).for_each(|k| rope.apply(k, pos));
        }
        let ts = self.temp_scale(pos);
        if ts != 1.0 {
            qv.iter_mut().for_each(|v| *v *= ts);
        }
        // the cache holds f16 values (llama.cpp's default cache type: half the memory of f32)
        f16_round(&mut kv);
        f16_round(&mut vv);
        let cache = st.kv.entry(slot).or_default();
        cache.0.extend_from_slice(&kv);
        cache.1.extend_from_slice(&vv);
        let (kc, vc) = (&cache.0, &cache.1);
        let n_kv = a.n_kv;
        let n = kc.len() / (n_kv * hd);
        let scale = 1.0 / (hd as f32).sqrt();
        let group = a.n_head / n_kv;
        let mut att = vec![0f32; a.n_head * hd];
        att.par_chunks_mut(hd).enumerate().for_each(|(hh, oh)| {
            let kh = hh / group;
            let qh = &qv[hh * hd..(hh + 1) * hd];
            let mut scores: Vec<f32> = (0..n).map(|t| dot(qh, &kc[(t * n_kv + kh) * hd..(t * n_kv + kh + 1) * hd]) * scale).collect();
            softmax(&mut scores);
            for (t, p) in scores.iter().enumerate() {
                let vt = &vc[(t * n_kv + kh) * hd..(t * n_kv + kh + 1) * hd];
                oh.iter_mut().zip(vt).for_each(|(o, v)| *o += p * v);
            }
        });
        let mut out = a.o.mul(&att);
        add_bias(&mut out, &a.o_b);
        out
    }

    fn mla(&self, slot: usize, m: &Mla, h: &[f32], st: &mut State) -> Vec<f32> {
        let pos = st.pos;
        let rope = self.rope.as_ref().expect("rope");
        let q = match &m.q_a {
            Some((qa, norm)) => m.q_b.mul(&rms_norm(&qa.mul(h), norm, self.eps)),
            None => m.q_b.mul(h),
        };
        let kvr = m.kv_a.mul(h);
        let mut k_pe = kvr[m.kv_lora..m.kv_lora + m.rope_dim].to_vec();
        rope.apply(&mut k_pe, pos);
        let mut kv = rms_norm(&kvr[..m.kv_lora], &m.kv_norm, self.eps);
        f16_round(&mut kv);
        f16_round(&mut k_pe);
        let cache = st.kv.entry(slot).or_default();
        cache.0.extend_from_slice(&kv);
        cache.1.extend_from_slice(&k_pe);
        let (lat, pe) = (&cache.0, &cache.1);
        let n = lat.len() / m.kv_lora;
        let ts = self.temp_scale(pos);
        let qk = m.nope + m.rope_dim;
        let mut out_heads = vec![0f32; m.n_head * m.v_dim];
        out_heads.par_chunks_mut(m.v_dim).enumerate().for_each(|(hh, oh)| {
            let qh = &q[hh * qk..(hh + 1) * qk];
            // the absorbed query: K_b(head) maps the no-position part into the latent space
            let mut q_lat = m.k_b.slice(hh, m.kv_lora).mul(&qh[..m.nope]);
            let mut q_pe = qh[m.nope..].to_vec();
            rope.apply(&mut q_pe, pos);
            if ts != 1.0 {
                q_lat.iter_mut().chain(q_pe.iter_mut()).for_each(|v| *v *= ts);
            }
            let mut scores: Vec<f32> = (0..n)
                .map(|t| (dot(&q_lat, &lat[t * m.kv_lora..(t + 1) * m.kv_lora]) + dot(&q_pe, &pe[t * m.rope_dim..(t + 1) * m.rope_dim])) * m.kq_scale)
                .collect();
            softmax(&mut scores);
            let mut acc = vec![0f32; m.kv_lora];
            for (t, p) in scores.iter().enumerate() {
                acc.iter_mut().zip(&lat[t * m.kv_lora..(t + 1) * m.kv_lora]).for_each(|(a, v)| *a += p * v);
            }
            m.v_b.slice(hh, m.v_dim).matvec(&acc, oh);
        });
        m.o.mul(&out_heads)
    }

    fn moe(&self, m: &Moe, h: &[f32]) -> Vec<f32> {
        let d = self.n_embd;
        let logits: Vec<f32> = (0..m.n_expert).map(|e| dot(&m.router[e * d..(e + 1) * d], h)).collect();
        let probs: Vec<f32> = match m.gating {
            Gating::Sigmoid => logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
            Gating::Softmax => {
                let mut p = logits.clone();
                softmax(&mut p);
                p
            }
        };
        // the bias only chooses; the weights are the unbiased probabilities
        let sel = |e: usize| probs[e] + m.bias.as_ref().map(|b| b[e]).unwrap_or(0.0);
        let mut order: Vec<usize> = (0..m.n_expert).collect();
        order.sort_by(|&a, &b| sel(b).total_cmp(&sel(a)));
        let chosen = &order[..m.top_k];
        let mut w: Vec<f32> = chosen.iter().map(|&e| probs[e]).collect();
        if m.w_norm {
            let s = w.iter().sum::<f32>().max(6.103515625e-5);
            w.iter_mut().for_each(|v| *v /= s);
        }
        if m.w_scale != 0.0 && m.w_scale != 1.0 {
            w.iter_mut().for_each(|v| *v *= m.w_scale);
        }
        for &e in chosen {
            self.heat[m.layer * self.profile_experts + e].fetch_add(1, Ordering::Relaxed);
        }
        let x = match &m.latent {
            Some((down, _)) => down.mul(h),
            None => h.to_vec(),
        };
        let out_dim = m.down_exps.rows / m.n_expert;
        let mut acc = vec![0f32; out_dim];
        for (&e, &we) in chosen.iter().zip(&w) {
            let mut mid = m.up_exps.slice(e, m.ff).mul(&x);
            match &m.gate_exps {
                Some(g) => {
                    let gv = g.slice(e, m.ff).mul(&x);
                    mid.iter_mut().zip(&gv).for_each(|(u, g)| *u *= act(m.act, *g));
                }
                None => mid.iter_mut().for_each(|v| *v = act(m.act, *v)),
            }
            let o = m.down_exps.slice(e, out_dim).mul(&mid);
            acc.iter_mut().zip(&o).for_each(|(a, v)| *a += we * v);
        }
        let mut out = match &m.latent {
            Some((_, up)) => up.mul(&acc),
            None => acc,
        };
        if let Some(sh) = &m.shared {
            out.iter_mut().zip(&sh.forward(h)).for_each(|(a, b)| *a += b);
        }
        out
    }

    /// The MoE layers, in order.
    pub fn moes(&self) -> impl Iterator<Item = &Moe> {
        self.layers.iter().flat_map(|l| l.parts.iter()).filter_map(|(_, b)| if let Block::Moe(m) = b { Some(m) } else { None })
    }

    /// A one-line summary of the layers, for INFO.
    pub fn summary(&self) -> String {
        let mut c: Vec<(&str, usize)> = vec![];
        for (_, b) in self.layers.iter().flat_map(|l| l.parts.iter()) {
            let name = match b {
                Block::Mamba(_) => "mamba",
                Block::Attn(_) => "attention",
                Block::Mla(_) => "mla",
                Block::Dense(_) => "dense_ffn",
                Block::Moe(_) => "moe",
            };
            match c.iter_mut().find(|(n, _)| *n == name) {
                Some(e) => e.1 += 1,
                None => c.push((name, 1)),
            }
        }
        c.iter().map(|(n, k)| format!("{n}_layers={k}")).collect::<Vec<_>>().join(" ")
    }
}

fn softmax(v: &mut [f32]) {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for s in v.iter_mut() {
        *s = (*s - m).exp();
        sum += *s;
    }
    v.iter_mut().for_each(|s| *s /= sum);
}

fn qmat(mm: &'static memmap2::Mmap, g: &Gguf, t: &TensorInfo) -> Result<QMat, String> {
    let n = t.expected_bytes().ok_or_else(|| format!("{}: unsupported type {}", t.name, t.type_name()))? as usize;
    let start = (g.data_start + t.offset) as usize;
    let data: &'static [u8] = &mm[start..start + n];
    let cols = t.dims[0] as usize;
    let rows = (t.elements() as usize) / cols;
    let ty = t.type_name();
    Ok(QMat { data, row_bytes: QMat::row_bytes(&ty, cols), vdt: Vdt::of(&ty), ty, rows, cols })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_rounding_as_ggml() {
        // Q8_K: the signed maximum maps to -127 x d, everything else rounds half to even on that grid
        let mut x = vec![0f32; 256];
        x[0] = -2.0;
        x[1] = 1.0;
        x[2] = 0.5 * 2.0 / 127.0; // exactly half a step: rounds to 0 (even)
        let y = Vdt::Q8K.round_trip(&x);
        assert_eq!((y[0], y[2]), (-2.0, 0.0));
        assert!((y[1] - 1.0).abs() < 2.0 / 127.0);
        // Q8_0: the scale is stored as f16, and halves round away from zero
        let mut x = vec![0f32; 32];
        x[0] = 127.0;
        x[1] = 0.5;
        let y = Vdt::Q8_0.round_trip(&x);
        assert_eq!((y[0], y[1]), (127.0, 1.0));
        // a length that is not whole blocks passes through
        assert_eq!(Vdt::Q8K.round_trip(&[0.3, 0.7]), vec![0.3, 0.7]);
    }

    #[test]
    fn rope_pairs() {
        let r = |neox| Rope { n_dims: 4, base: 10000.0, freq_scale: 1.0, ext_factor: 0.0, attn_factor: 1.0, corr: [0.0, 3.0], neox, factors: None };
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut a = x;
        r(false).apply(&mut a, 0);
        assert_eq!(a, x); // position 0 is the identity
        // NORM rotates (x0, x1) by pos * base^0; NEOX rotates (x0, x2)
        let (c, s) = (1f32.cos(), 1f32.sin());
        let mut a = x;
        r(false).apply(&mut a, 1);
        assert!((a[0] - (c - 2.0 * s)).abs() < 1e-6 && (a[1] - (s + 2.0 * c)).abs() < 1e-6);
        let mut a = x;
        r(true).apply(&mut a, 1);
        assert!((a[0] - (c - 3.0 * s)).abs() < 1e-6 && (a[2] - (s + 3.0 * c)).abs() < 1e-6);
    }
}
