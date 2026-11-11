//! ACE-Step 1.5's diffusion transformer (turbo) and its condition encoder, as in its `modeling_acestep_v15_turbo.py`:
//! - attention is Qwen3's (q/k RMS norms, rotary positions on self-attention, 16 query heads over 8 KV heads),
//!   bidirectional, over all positions ("full_attention" layers) or within 128 positions ("sliding_attention");
//! - the condition encoder: the caption projected, the lyrics through 8 encoder layers, the timbre (a reference's
//!   latents, or ACE-Step's silence latent) through 4, packed as lyrics, timbre, caption;
//! - the DiT: [context, x_t] (192 channels) in patches of 2 frames, 24 layers of AdaLN-modulated self-attention,
//!   cross-attention to the conditions and an MLP, conditioned on t and t - r;
//! - the sampler: Euler steps of the flow, the last one to x_0, each followed by DCW (a one-level Haar split along time,
//!   each band pushed from the predicted clean latent).

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, RmsNorm, VarBuilder};

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: usize,
    pub layer_types: Vec<String>,
    pub num_hidden_layers: usize,
    pub num_lyric_encoder_hidden_layers: usize,
    pub num_timbre_encoder_hidden_layers: usize,
    pub in_channels: usize,
    pub audio_acoustic_hidden_dim: usize,
    pub text_hidden_dim: usize,
    pub timbre_hidden_dim: usize,
    pub patch_size: usize,
}

fn rms(dim: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    candle_nn::rms_norm(dim, eps, vb)
}

/// cos and sin [len, head_dim/2] for positions 0..len
struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    fn new(cfg: &Config, len: usize, dtype: DType, dev: &Device) -> Result<Self> {
        let half = cfg.head_dim / 2;
        let inv: Vec<f32> = (0..half).map(|i| 1.0 / (cfg.rope_theta as f32).powf(2.0 * i as f32 / cfg.head_dim as f32)).collect();
        let inv = Tensor::from_vec(inv, (1, half), dev)?;
        let pos = Tensor::arange(0u32, len as u32, dev)?.to_dtype(DType::F32)?.unsqueeze(1)?;
        let f = pos.matmul(&inv)?;
        Ok(Self { cos: f.cos()?.to_dtype(dtype)?, sin: f.sin()?.to_dtype(dtype)? })
    }
}

/// An additive mask [1, 1, len, len]: 0 within `window` positions either way, -inf beyond (None: no mask).
fn window_mask(len: usize, window: usize, dtype: DType, dev: &Device) -> Result<Option<Tensor>> {
    if len <= window + 1 {
        return Ok(None);
    }
    let v: Vec<f32> = (0..len).flat_map(|i| (0..len).map(move |j| if i.abs_diff(j) <= window { 0.0 } else { f32::NEG_INFINITY })).collect();
    Ok(Some(Tensor::from_vec(v, (1, 1, len, len), dev)?.to_dtype(dtype)?))
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (h, kv, d) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        Ok(Self {
            q: linear_no_bias(cfg.hidden_size, h * d, vb.pp("q_proj"))?,
            k: linear_no_bias(cfg.hidden_size, kv * d, vb.pp("k_proj"))?,
            v: linear_no_bias(cfg.hidden_size, kv * d, vb.pp("v_proj"))?,
            o: linear_no_bias(h * d, cfg.hidden_size, vb.pp("o_proj"))?,
            q_norm: rms(d, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms(d, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            heads: h,
            kv_heads: kv,
            head_dim: d,
        })
    }

    /// Self-attention when `kv_from` is None (with rotary positions), else cross-attention to it (without).
    fn forward(&self, x: &Tensor, kv_from: Option<&Tensor>, rope: Option<&Rope>, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let src = kv_from.unwrap_or(x);
        let s = src.dim(1)?;
        let q = self.q.forward(x)?.reshape((b, l, self.heads, self.head_dim))?.apply(&self.q_norm)?.transpose(1, 2)?.contiguous()?;
        let k = self.k.forward(src)?.reshape((b, s, self.kv_heads, self.head_dim))?.apply(&self.k_norm)?.transpose(1, 2)?.contiguous()?;
        let v = self.v.forward(src)?.reshape((b, s, self.kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let (q, k) = match rope {
            Some(r) => (candle_nn::rotary_emb::rope(&q, &r.cos, &r.sin)?, candle_nn::rotary_emb::rope(&k, &r.cos, &r.sin)?),
            None => (q, k),
        };
        // grouped-query: each KV head serves heads/kv_heads query heads
        let rep = self.heads / self.kv_heads;
        let (k, v) = if rep > 1 {
            let k = k.unsqueeze(2)?.expand((b, self.kv_heads, rep, s, self.head_dim))?.reshape((b, self.heads, s, self.head_dim))?;
            let v = v.unsqueeze(2)?.expand((b, self.kv_heads, rep, s, self.head_dim))?.reshape((b, self.heads, s, self.head_dim))?;
            (k, v)
        } else {
            (k, v)
        };
        let mut w = (q.matmul(&k.t()?)? * (1.0 / (self.head_dim as f64).sqrt()))?;
        if let Some(m) = mask {
            w = w.broadcast_add(m)?;
        }
        let w = candle_nn::ops::softmax_last_dim(&w.to_dtype(DType::F32)?)?.to_dtype(v.dtype())?;
        let y = w.matmul(&v)?.transpose(1, 2)?.reshape((b, l, self.heads * self.head_dim))?;
        self.o.forward(&y)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down.forward(&(candle_nn::ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?)
    }
}

/// A pre-norm encoder layer (the lyric and timbre encoders).
struct EncoderLayer {
    norm1: RmsNorm,
    attn: Attention,
    norm2: RmsNorm,
    mlp: Mlp,
    sliding: bool,
}

impl EncoderLayer {
    fn new(cfg: &Config, i: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: rms(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            attn: Attention::new(cfg, vb.pp("self_attn"))?,
            norm2: rms(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            sliding: cfg.layer_types[i] == "sliding_attention",
        })
    }

    fn forward(&self, x: &Tensor, rope: &Rope, sliding_mask: Option<&Tensor>) -> Result<Tensor> {
        let mask = if self.sliding { sliding_mask } else { None };
        let x = (x + self.attn.forward(&x.apply(&self.norm1)?, None, Some(rope), mask)?)?;
        &x + self.mlp.forward(&x.apply(&self.norm2)?)?
    }
}

/// FSQ levels of ACE-Step's audio codes (`fsq_input_levels`): a code is 6 digits in these bases, 64,000 codes.
pub const FSQ_LEVELS: [u32; 6] = [8, 8, 8, 5, 5, 5];
pub const CODEBOOK_SIZE: u32 = 8 * 8 * 8 * 5 * 5 * 5;
/// 25 Hz latent frames per 5 Hz code (`pool_window_size`)
pub const POOL: usize = 5;

/// The planner's codes [n] back to 25 Hz latents [1, n * 5, 64], the DiT's source when the planner sets the song's
/// structure: the residual FSQ's code values (symmetric: each digit d of level L as d * 2 / (L - 1) - 1; one quantizer, scale 1)
/// and its output projection, then the detokenizer (each code as 5 frames: the projected code plus learnt per-frame
/// tokens, 2 encoder layers within those 5, a norm, a projection to 64 channels).
pub struct Detokenizer {
    project_out: Linear,
    embed: Linear,
    special: Tensor,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    proj_out: Linear,
    cfg: Config,
}

impl Detokenizer {
    /// `vb`: the turbo checkpoint's root (`tokenizer.quantizer.*`, `detokenizer.*`).
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d = vb.pp("detokenizer");
        Ok(Self {
            project_out: linear(FSQ_LEVELS.len(), cfg.hidden_size, vb.pp("tokenizer.quantizer.project_out"))?,
            embed: linear(cfg.hidden_size, cfg.hidden_size, d.pp("embed_tokens"))?,
            special: d.get((1, POOL, cfg.hidden_size), "special_tokens")?,
            layers: (0..2).map(|i| EncoderLayer::new(cfg, i, d.pp(format!("layers.{i}")))).collect::<Result<_>>()?,
            norm: rms(cfg.hidden_size, cfg.rms_norm_eps, d.pp("norm"))?,
            proj_out: linear(cfg.hidden_size, cfg.audio_acoustic_hidden_dim, d.pp("proj_out"))?,
            cfg: cfg.clone(),
        })
    }

    /// The quantizer's output for `codes` [1, n, 2048].
    pub fn quantized(&self, codes: &[u32], dtype: DType, dev: &Device) -> Result<Tensor> {
        let mut v = Vec::with_capacity(codes.len() * FSQ_LEVELS.len());
        for &c in codes {
            let mut basis = 1u32;
            for &l in &FSQ_LEVELS {
                let digit = (c / basis) % l;
                v.push(digit as f32 * 2.0 / (l - 1) as f32 - 1.0);
                basis *= l;
            }
        }
        let z = Tensor::from_vec(v, (1, codes.len(), FSQ_LEVELS.len()), dev)?.to_dtype(dtype)?;
        self.project_out.forward(&z)
    }

    /// The 25 Hz hints [1, n * 5, 64] for `codes`.
    pub fn hints(&self, codes: &[u32], dtype: DType, dev: &Device) -> Result<Tensor> {
        let q = self.quantized(codes, dtype, dev)?;
        let n = codes.len();
        let h = self.cfg.hidden_size;
        // [1, n, h] -> [n, 5, h]: every code repeated over its 5 frames, plus the frame tokens
        let x = self.embed.forward(&q)?.squeeze(0)?.unsqueeze(1)?.expand((n, POOL, h))?;
        let mut x = x.broadcast_add(&self.special.to_dtype(x.dtype())?)?.contiguous()?;
        let rope = Rope::new(&self.cfg, POOL, x.dtype(), dev)?;
        for layer in &self.layers {
            x = layer.forward(&x, &rope, None)?;
        }
        let x = self.proj_out.forward(&x.apply(&self.norm)?)?;
        x.reshape((1, n * POOL, self.cfg.audio_acoustic_hidden_dim))
    }
}

/// The lyric or the timbre encoder: a projection in, encoder layers, a final norm.
struct Encoder {
    embed: Linear,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    cfg: Config,
}

impl Encoder {
    fn new(cfg: &Config, input: usize, layers: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embed: linear(input, cfg.hidden_size, vb.pp("embed_tokens"))?,
            layers: (0..layers).map(|i| EncoderLayer::new(cfg, i, vb.pp(format!("layers.{i}")))).collect::<Result<_>>()?,
            norm: rms(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?,
            cfg: cfg.clone(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.embed.forward(x)?;
        let l = h.dim(1)?;
        let rope = Rope::new(&self.cfg, l, h.dtype(), h.device())?;
        let mask = window_mask(l, self.cfg.sliding_window, h.dtype(), h.device())?;
        for layer in &self.layers {
            h = layer.forward(&h, &rope, mask.as_ref())?;
        }
        h.apply(&self.norm)
    }
}

pub struct ConditionEncoder {
    text_projector: Linear,
    lyric: Encoder,
    timbre: Encoder,
}

impl ConditionEncoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            text_projector: linear_no_bias(cfg.text_hidden_dim, cfg.hidden_size, vb.pp("text_projector"))?,
            lyric: Encoder::new(cfg, cfg.text_hidden_dim, cfg.num_lyric_encoder_hidden_layers, vb.pp("lyric_encoder"))?,
            timbre: Encoder::new(cfg, cfg.timbre_hidden_dim, cfg.num_timbre_encoder_hidden_layers, vb.pp("timbre_encoder"))?,
        })
    }

    /// The conditions the DiT attends to: [lyrics, timbre (one position), caption], [1, L + 1 + T, hidden].
    /// `caption`: the caption encoder's states [1, T, 1024]; `lyrics`: the lyric token embeddings [1, L, 1024];
    /// `timbre`: reference latents [1, frames, 64] (ACE-Step's silence latent, its first 750 frames, for none).
    pub fn forward(&self, caption: &Tensor, lyrics: &Tensor, timbre: &Tensor) -> Result<Tensor> {
        let text = self.text_projector.forward(caption)?;
        let lyr = self.lyric.forward(lyrics)?;
        let tim = self.timbre.forward(timbre)?.narrow(1, 0, 1)?;
        Tensor::cat(&[&lyr, &tim, &text], 1)
    }
}

/// Sinusoidal embedding of t (times 1000, 256 channels) through an MLP: (temb [B, hidden], proj [B, 6, hidden]).
struct TimestepEmbedding {
    l1: Linear,
    l2: Linear,
    proj: Linear,
    hidden: usize,
}

impl TimestepEmbedding {
    fn new(hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            l1: linear(256, hidden, vb.pp("linear_1"))?,
            l2: linear(hidden, hidden, vb.pp("linear_2"))?,
            proj: linear(hidden, hidden * 6, vb.pp("time_proj"))?,
            hidden,
        })
    }

    fn forward(&self, t: f32, dtype: DType, dev: &Device) -> Result<(Tensor, Tensor)> {
        let half = 128;
        let args: Vec<f32> = (0..half).map(|i| t * 1000.0 * (-(10000f32).ln() * i as f32 / half as f32).exp()).collect();
        let emb: Vec<f32> = args.iter().map(|a| a.cos()).chain(args.iter().map(|a| a.sin())).collect();
        let emb = Tensor::from_vec(emb, (1, 256), dev)?.to_dtype(dtype)?;
        let temb = self.l2.forward(&candle_nn::ops::silu(&self.l1.forward(&emb)?)?)?;
        let proj = self.proj.forward(&candle_nn::ops::silu(&temb)?)?.reshape((1, 6, self.hidden))?;
        Ok((temb, proj))
    }
}

struct DitLayer {
    self_norm: RmsNorm,
    self_attn: Attention,
    cross_norm: RmsNorm,
    cross_attn: Attention,
    mlp_norm: RmsNorm,
    mlp: Mlp,
    table: Tensor,
    sliding: bool,
}

impl DitLayer {
    fn new(cfg: &Config, i: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_norm: rms(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("self_attn_norm"))?,
            self_attn: Attention::new(cfg, vb.pp("self_attn"))?,
            cross_norm: rms(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("cross_attn_norm"))?,
            cross_attn: Attention::new(cfg, vb.pp("cross_attn"))?,
            mlp_norm: rms(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("mlp_norm"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            table: vb.get((1, 6, cfg.hidden_size), "scale_shift_table")?,
            sliding: cfg.layer_types[i] == "sliding_attention",
        })
    }

    fn forward(&self, x: &Tensor, proj: &Tensor, cond: &Tensor, rope: &Rope, sliding_mask: Option<&Tensor>) -> Result<Tensor> {
        let m = self.table.broadcast_add(proj)?;
        let part = |i: usize| m.narrow(1, i, 1);
        let (shift, scale, gate, c_shift, c_scale, c_gate) = (part(0)?, part(1)?, part(2)?, part(3)?, part(4)?, part(5)?);
        let h = x.apply(&self.self_norm)?.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
        let mask = if self.sliding { sliding_mask } else { None };
        let x = (x + self.self_attn.forward(&h, None, Some(rope), mask)?.broadcast_mul(&gate)?)?;
        let x = (&x + self.cross_attn.forward(&x.apply(&self.cross_norm)?, Some(cond), None, None)?)?;
        let h = x.apply(&self.mlp_norm)?.broadcast_mul(&(c_scale + 1.0)?)?.broadcast_add(&c_shift)?;
        &x + self.mlp.forward(&h)?.broadcast_mul(&c_gate)?
    }
}

pub struct Dit {
    cfg: Config,
    proj_in: Conv1d,
    time_embed: TimestepEmbedding,
    time_embed_r: TimestepEmbedding,
    condition_embedder: Linear,
    layers: Vec<DitLayer>,
    norm_out: RmsNorm,
    proj_out: ConvTranspose1d,
    table: Tensor,
}

impl Dit {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let p = cfg.patch_size;
        let w_in = vb.get((cfg.hidden_size, cfg.in_channels, p), "proj_in.1.weight")?;
        let b_in = vb.get(cfg.hidden_size, "proj_in.1.bias")?;
        let w_out = vb.get((cfg.hidden_size, cfg.audio_acoustic_hidden_dim, p), "proj_out.1.weight")?;
        let b_out = vb.get(cfg.audio_acoustic_hidden_dim, "proj_out.1.bias")?;
        Ok(Self {
            proj_in: Conv1d::new(w_in, Some(b_in), Conv1dConfig { stride: p, ..Default::default() }),
            time_embed: TimestepEmbedding::new(cfg.hidden_size, vb.pp("time_embed"))?,
            time_embed_r: TimestepEmbedding::new(cfg.hidden_size, vb.pp("time_embed_r"))?,
            condition_embedder: linear(cfg.hidden_size, cfg.hidden_size, vb.pp("condition_embedder"))?,
            layers: (0..cfg.num_hidden_layers).map(|i| DitLayer::new(cfg, i, vb.pp(format!("layers.{i}")))).collect::<Result<_>>()?,
            norm_out: rms(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm_out"))?,
            proj_out: ConvTranspose1d::new(w_out, Some(b_out), ConvTranspose1dConfig { stride: p, ..Default::default() }),
            table: vb.get((1, 2, cfg.hidden_size), "scale_shift_table")?,
            cfg: cfg.clone(),
        })
    }

    /// The velocity at time `t` (with r = t, as the turbo sampler calls it): x_t [1, frames, 64], context [1, frames,
    /// 128] (source latents and chunk mask), conditions [1, n, hidden] (the condition encoder's output).
    pub fn forward(&self, xt: &Tensor, t: f32, context: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let (dtype, dev) = (xt.dtype(), xt.device());
        let (temb_t, proj_t) = self.time_embed.forward(t, dtype, dev)?;
        let (temb_r, proj_r) = self.time_embed_r.forward(0.0, dtype, dev)?;
        let (temb, proj) = ((temb_t + temb_r)?, (proj_t + proj_r)?);
        let mut h = Tensor::cat(&[context, xt], D::Minus1)?;
        let frames = h.dim(1)?;
        let p = self.cfg.patch_size;
        if frames % p != 0 {
            h = h.pad_with_zeros(1, 0, p - frames % p)?;
        }
        let mut h = h.transpose(1, 2)?.apply(&self.proj_in)?.transpose(1, 2)?;
        let cond = self.condition_embedder.forward(cond)?;
        let l = h.dim(1)?;
        let rope = Rope::new(&self.cfg, l, dtype, dev)?;
        let mask = window_mask(l, self.cfg.sliding_window, dtype, dev)?;
        for layer in &self.layers {
            h = layer.forward(&h, &proj, &cond, &rope, mask.as_ref())?;
        }
        let m = self.table.broadcast_add(&temb.unsqueeze(1)?)?;
        let (shift, scale) = (m.narrow(1, 0, 1)?, m.narrow(1, 1, 1)?);
        let h = h.apply(&self.norm_out)?.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
        h.transpose(1, 2)?.apply(&self.proj_out)?.transpose(1, 2)?.narrow(1, 0, frames)
    }
}

/// The turbo schedule for `steps` steps and `shift` (ACE-Step's variable-step rule: 1 - i/n, then shifted).
pub fn schedule(steps: usize, shift: f32) -> Vec<f32> {
    (0..steps).map(|i| {
        let t = 1.0 - i as f32 / steps as f32;
        if shift == 1.0 { t } else { shift * t / (1.0 + (shift - 1.0) * t) }
    }).collect()
}

/// DCW "double": the one-level Haar bands of x pushed from those of `denoised` (low by `low`, high by `high`).
/// For a pair of frames that is: x + low * mean(d) + high * (d - mean(d)), d = x - denoised; an odd last frame pairs
/// with a zero, as the reference pads it.
pub fn dcw_double(x: &Tensor, denoised: &Tensor, low: f32, high: f32) -> Result<Tensor> {
    let d = (x - denoised)?;
    let frames = d.dim(1)?;
    let dp = if frames % 2 == 1 { d.pad_with_zeros(1, 0, 1)? } else { d.clone() };
    let (b, n, c) = dp.dims3()?;
    let pairs = dp.reshape((b, n / 2, 2, c))?;
    let mean = pairs.mean_keepdim(2)?.broadcast_as((b, n / 2, 2, c))?.reshape((b, n, c))?.narrow(1, 0, frames)?;
    let corr = ((&mean * low as f64)? + ((&d - &mean)? * high as f64)?)?;
    x + corr
}

/// The turbo sampler from `noise` [1, frames, 64]: Euler steps of the flow, the last one to x_0, DCW after each
/// (scaler 0.05 for the low band, 0.02 for the high, as ACE-Step's defaults). `on_step` sees each step.
pub fn sample(dit: &Dit, noise: &Tensor, context: &Tensor, cond: &Tensor, ts: &[f32], on_step: &mut dyn FnMut(usize, usize)) -> Result<Tensor> {
    sample_repaint(dit, noise, context, cond, ts, None, on_step)
}

/// A repaint (ACE-Step's "repaint" task, here run past the end of a source to continue it): the frames to make and the
/// source they continue.
pub struct Repaint<'a> {
    /// [1, frames, 1]: 1 where frames are made, 0 where the source is kept
    pub mask: &'a Tensor,
    /// the source's latent [1, frames, 64] (silence where nothing was)
    pub clean: &'a Tensor,
    /// the kept frames forced back to the noised source for this share of the steps (ACE-Step: 0.5)
    pub injection_ratio: f32,
    /// the frames over which the boundary is crossfaded at the end (ACE-Step's handler: 12)
    pub crossfade: usize,
}

/// The repaint's soft mask: 1 where frames are made, ramps of `cf` frames into the kept frames on each side of the
/// region (as ACE-Step's _repaint_boundary_blend: linspace without its end points).
pub fn soft_mask(region: (usize, usize), frames: usize, cf: usize) -> Vec<f32> {
    let (left, right) = region;
    let mut m = vec![0f32; frames];
    m[left..right].iter_mut().for_each(|x| *x = 1.0);
    let fs = left.saturating_sub(cf);
    let n = left - fs;
    for j in 0..n {
        m[fs + j] = (j + 1) as f32 / (n + 1) as f32;
    }
    let fe = (right + cf).min(frames);
    let n = fe - right;
    for j in 0..n {
        m[right + j] = 1.0 - (j + 1) as f32 / (n + 1) as f32;
    }
    m
}

/// `sample` with an optional repaint: after each of the first injection_ratio of the steps the kept frames are set
/// back to the source noised to the next step's level (t·noise + (1-t)·source, the same noise), and at the end the
/// result is blended with the source through the soft mask (ACE-Step's _repaint_step_injection and
/// _repaint_boundary_blend).
pub fn sample_repaint(dit: &Dit, noise: &Tensor, context: &Tensor, cond: &Tensor, ts: &[f32], repaint: Option<&Repaint>,
                      on_step: &mut dyn FnMut(usize, usize)) -> Result<Tensor> {
    let mut x = noise.clone();
    let cutoff = repaint.map(|r| (r.injection_ratio * ts.len() as f32).round_ties_even() as usize).unwrap_or(0);
    for (i, &t) in ts.iter().enumerate() {
        let v = dit.forward(&x, t, context, cond)?;
        let denoised = (&x - (&v * t as f64)?)?;
        x = if i + 1 == ts.len() { denoised.clone() } else { (&x - (&v * (t - ts[i + 1]) as f64)?)? };
        x = dcw_double(&x, &denoised, t * 0.05, (1.0 - t) * 0.02)?;
        if let (Some(r), true) = (repaint, i + 1 < ts.len() && i < cutoff) {
            let tn = ts[i + 1] as f64;
            let zt = ((noise * tn)? + (r.clean * (1.0 - tn))?)?;
            let keep = r.mask.ones_like()?.sub(r.mask)?;
            x = (x.broadcast_mul(r.mask)? + zt.broadcast_mul(&keep)?)?;
        }
        on_step(i + 1, ts.len());
    }
    if let Some(r) = repaint.filter(|r| r.crossfade > 0) {
        let m = r.mask.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let (Some(left), Some(last)) = (m.iter().position(|v| *v > 0.5), m.iter().rposition(|v| *v > 0.5)) else { return Ok(x) };
        let soft = soft_mask((left, last + 1), m.len(), r.crossfade);
        let soft = Tensor::from_vec(soft, (1, m.len(), 1), x.device())?.to_dtype(x.dtype())?;
        let keep = soft.ones_like()?.sub(&soft)?;
        x = (x.broadcast_mul(&soft)? + r.clean.broadcast_mul(&keep)?)?;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> std::path::PathBuf {
        std::env::var("ACESTEP_REF").map(Into::into)
            .unwrap_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("Development/ref"))
    }

    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let (a, b) = (a.flatten_all().unwrap(), b.flatten_all().unwrap());
        let dot = (&a * &b).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let na = a.sqr().unwrap().sum_all().unwrap().sqrt().unwrap().to_scalar::<f32>().unwrap();
        let nb = b.sqr().unwrap().sum_all().unwrap().sqrt().unwrap().to_scalar::<f32>().unwrap();
        dot / (na * nb)
    }

    #[test]
    fn schedules_are_acesteps() {
        assert_eq!(schedule(8, 1.0), [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]);
        let s3 = schedule(8, 3.0);
        assert!((s3[1] - 0.954_545_5).abs() < 1e-6 && (s3[7] - 0.3).abs() < 1e-6, "{s3:?}");
    }

    #[test]
    #[ignore = "needs ACE-Step's checkpoints and the golden files (ACESTEP_REF)"]
    fn conditions_steps_and_the_song_match_the_reference() {
        let r = reference();
        let dev = Device::Cpu;
        let ck = r.join("ACE-Step-1.5/checkpoints/acestep-v15-turbo");
        let cfg: Config = serde_json::from_reader(std::fs::File::open(ck.join("config.json")).unwrap()).unwrap();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[ck.join("model.safetensors")], DType::F32, &dev) }.unwrap();
        let enc = ConditionEncoder::new(&cfg, vb.pp("encoder")).unwrap();
        let dit = Dit::new(&cfg, vb.pp("decoder")).unwrap();
        let g = |f: &str| candle_core::safetensors::load(r.join(format!("acestep-golden/{f}.safetensors")), &dev).unwrap();
        let (genr, encg, steps) = (g("generate"), g("encoder"), g("dit_steps"));
        // the condition encoder
        let timbre = genr["in.refer_audio_acoustic_hidden_states_packed"].clone();
        let cond = enc.forward(&genr["in.text_hidden_states"], &genr["in.lyric_hidden_states"], &timbre).unwrap();
        let c = cosine(&cond, &encg["0.out.0"]);
        eprintln!("conditions {:?}: cosine {c:.5}", cond.dims());
        assert!(c > 0.999, "conditions: cosine {c}");
        // step 0 against ACE-Step's own DiT in float32 on the CPU (~/Development/ref/acestep-f32-step.py): the port is
        // checked tightly there; the recording itself is bf16 on the GPU (ACE-Step's f32 output is at cosine 0.9926 to
        // it), so the steps below are checked against it with a bf16 margin
        let f32_ref = r.join("acestep-golden/dit_step0_f32.safetensors");
        if f32_ref.exists() {
            let want = &candle_core::safetensors::load(&f32_ref, &dev).unwrap()["v"];
            let v = dit.forward(&steps["0.kwargs.hidden_states"], 1.0, &steps["0.kwargs.context_latents"], &steps["0.kwargs.encoder_hidden_states"]).unwrap();
            let c = cosine(&v, want);
            eprintln!("step 0 against ACE-Step in f32: cosine {c:.6}");
            assert!(c > 0.9999, "step 0 against the f32 reference: cosine {c}");
        }
        // each DiT call, from its recorded inputs
        let ts = schedule(8, 1.0);
        for i in 0..8 {
            let xt = &steps[&format!("{i}.kwargs.hidden_states")];
            let ctx = &steps[&format!("{i}.kwargs.context_latents")];
            let t = steps[&format!("{i}.kwargs.timestep")].to_vec1::<f32>().unwrap()[0];
            assert!((t - ts[i]).abs() < 1e-3, "step {i}: t {t} against {}", ts[i]);
            let t0 = std::time::Instant::now();
            let v = dit.forward(xt, t, ctx, &steps[&format!("{i}.kwargs.encoder_hidden_states")]).unwrap();
            let c = cosine(&v, &steps[&format!("{i}.out.0")]);
            eprintln!("step {i} (t {t:.3}): velocity cosine {c:.5} in {:.1} s", t0.elapsed().as_secs_f64());
            assert!(c > 0.98, "step {i}: velocity cosine {c}");
        }
        // the whole sampler from the recorded noise, against the final latent
        let noise = &steps["0.kwargs.hidden_states"];
        let x0 = sample(&dit, noise, &steps["0.kwargs.context_latents"], &encg["0.out.0"], &ts, &mut |_, _| {}).unwrap();
        let c = cosine(&x0, &genr["out.target_latents"]);
        eprintln!("final latent: cosine {c:.5}");
        assert!(c > 0.97, "final latent: cosine {c}");
    }
}
