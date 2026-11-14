//! Step 3, part B — the training loop for the LoRA adapters built in [`crate::lora`]. This threads a masked next-token
//! loss through the frozen base + LoRA forward pass, accumulates gradients over micro-batches before one optimizer step,
//! runs AdamW with a cosine-with-warm-up learning-rate schedule, and exposes the adapter parameters by PEFT name so a
//! checkpoint can be saved and resumed (part C). No downloads: the base comes from the tiny fixture's tensors, and the
//! training tokens are synthetic but deterministic.

use candle_core::{Device, Tensor, Var};
use candle_nn::ParamsAdamW;
use anyhow::Context as _;

/// Any error in this module (candle ops + a few hand-built messages).
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// AdamW hyper-parameters and the learning-rate schedule that drives the training loop. `lr` is the peak rate reached at
/// the end of warm-up, then cosine-decayed to zero over `total_steps`.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// peak learning rate (reached after `warmup_steps`, then cosine decayed)
    pub lr: f64,
    /// steps at which the peak LR is reached, before cosine decay
    pub warmup_steps: usize,
    /// total number of optimizer steps; sets the end of the cosine envelope
    pub total_steps: usize,
    /// micro-batches whose gradients are summed before one AdamW step (accumulation)
    pub accumulation_steps: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self { lr: 0.05, warmup_steps: 2, total_steps: 12, accumulation_steps: 4 }
    }
}

/// Cosine schedule with linear warm-up. Returns the learning rate at global `step` (0-indexed): ramps from 0 to `peak`
/// over `warmup_steps`, then cosine-decays toward 0, reaching ~0 at `total_steps`.
pub fn cosine_with_warmup_lr(peak: f64, step: usize, warmup_steps: usize, total_steps: usize) -> f64 {
    let warmup = warmup_steps.max(1);
    if step < warmup {
        peak * (step as f64 / warmup as f64)
    } else {
        let total = total_steps.saturating_sub(warmup).max(1);
        let progress = ((step - warmup) as f64 / total as f64).min(1.0);
        peak * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

/// Deterministic training data: `num_sequences` sequences of `seq_len` token ids, each split into a `prompt_len`-token
/// prompt (masked from the loss) and the rest (whose next tokens are supervised). Synthetic — no tokenizer needed for
/// this CPU proof; what matters is that a learnable signal exists so masked CE falls as the LoRA adapters train.
#[derive(Debug, Clone)]
pub struct TrainData {
    /// token ids per sequence, length `seq_len`
    pub tokens: Vec<Vec<u32>>,
    /// prompt length (positions before this index are excluded from the loss) per sequence
    pub prompt_lens: Vec<usize>,
    /// fixed sequence length of every row
    pub seq_len: usize,
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u32_in(&mut self, hi: u32) -> u32 {
        ((self.next_u64() >> 40) as u32) % hi
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Build a deterministic, **learnable** dataset from `seed`: each row starts at a random token and every next token is
/// a fixed function of the current one (`next = (5 * cur + 3) mod vocab`). Uniformly random tokens would not do: their
/// best possible loss is ln(vocab), so "the loss falls" could only pass by luck.
pub fn make_train_data(seed: u64, num_sequences: usize, seq_len: usize, vocab: usize, prompt_len: usize) -> TrainData {
    let mut gen = SplitMix64::new(seed);
    let v = vocab as u32;
    let tokens = (0..num_sequences)
        .map(|_| {
            let mut cur = gen.next_u32_in(v);
            (0..seq_len).map(|_| { let t = cur; cur = (5 * cur + 3) % v; t }).collect()
        })
        .collect();
    TrainData { tokens, prompt_lens: vec![prompt_len; num_sequences], seq_len }
}

impl TrainData {
    /// The row indices for micro-batch `group` when the dataset is split into `accumulation_steps` groups.
    pub fn group(&self, group: usize, accumulation_steps: usize) -> Vec<usize> {
        let n = self.tokens.len();
        if n == 0 {
            return vec![];
        }
        let per = n.div_ceil(accumulation_steps);
        (group * per..(group * per + per).min(n)).collect()
    }

    /// Extract row `r`'s token slice as a rank-1 tensor of ids.
    fn row_tensor(&self, r: usize, dev: &Device) -> Res<Tensor> {
        Ok(Tensor::from_vec(self.tokens[r].clone(), (self.seq_len,), dev)?)
    }
}

/// AdamW with its state in the open, so a checkpoint can save and restore it (candle_nn's `AdamW` keeps its moments
/// private, and a resume without them is not exact). The update is candle_nn's, line for line: decoupled weight decay,
/// bias-corrected first and second moments.
pub struct AdamWState {
    pub params: ParamsAdamW,
    /// optimizer steps taken (the bias-correction exponent)
    pub step_t: usize,
    /// per trainable variable: (the variable, first moment `exp_avg`, second moment `exp_avg_sq`)
    pub slots: Vec<(Var, Var, Var)>,
}

impl AdamWState {
    pub fn new(vars: Vec<Var>, params: ParamsAdamW) -> Res<Self> {
        let slots = vars.into_iter().map(|v| {
            let t = v.as_tensor();
            let m = Var::zeros(t.shape(), t.dtype(), t.device())?;
            let s = Var::zeros(t.shape(), t.dtype(), t.device())?;
            Ok::<_, candle_core::Error>((v, m, s))
        }).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { params, step_t: 0, slots })
    }

    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Res<()> {
        self.step_t += 1;
        let p = &self.params;
        let scale_m = 1f64 / (1f64 - p.beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - p.beta2.powi(self.step_t as i32));
        for (theta, m, v) in &self.slots {
            if let Some(g) = grads.get(theta) {
                let next_m = ((m.as_tensor() * p.beta1)? + (g * (1.0 - p.beta1))?)?;
                let next_v = ((v.as_tensor() * p.beta2)? + (g.sqr()? * (1.0 - p.beta2))?)?;
                let adjusted = ((&next_m * scale_m)? / ((&next_v * scale_v)?.sqrt()? + p.eps)?)?;
                let next_theta = ((theta.as_tensor() * (1f64 - p.lr * p.weight_decay))? - (adjusted * p.lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

/// AdamW optimizer over the LoRA adapter variables, paired with the schedule.
pub struct Trainer {
    pub optimizer: AdamWState,
    cfg: TrainingConfig,
}

impl Trainer {
    /// The current learning rate the schedule has set on the optimizer.
    pub fn learning_rate(&self) -> f64 {
        self.optimizer.params.lr
    }

    /// Build AdamW from the model's trainable A/B variables (see [`trainable_params`]) with a cosine-with-warm-up schedule.
    pub fn new(model: &crate::lora::TinyLlama, cfg: TrainingConfig) -> Res<Self> {
        let (vars, _names) = trainable_params(model)?;
        let mut params = ParamsAdamW::default();
        params.lr = cfg.lr;
        let optimizer = AdamWState::new(vars, params)?;
        Ok(Self { optimizer, cfg })
    }

    /// One optimizer step. Accumulates the masked next-token loss over `accumulation_steps` micro-batches (averaged),
    /// takes a single backward, then steps with the schedule's LR at the current global step. Returns the mean masked CE
    /// over the whole dataset at these parameters — the training-loss signal that should fall.
    pub fn train_step(&mut self, model: &crate::lora::TinyLlama, data: &TrainData, global_step: usize) -> Res<f64> {
        let accum = self.cfg.accumulation_steps.clamp(1, data.tokens.len().max(1));
        let lr = cosine_with_warmup_lr(self.cfg.lr, global_step, self.cfg.warmup_steps, self.cfg.total_steps);
        self.optimizer.params.lr = lr;

        // sum the per-micro-step losses (each scaled by 1/accum) so one backward propagates the accumulated gradient.
        let mut acc_loss: Option<Tensor> = None;
        for group in 0..accum {
            let rows = data.group(group, accum);
            let loss = self.batch_loss(model, data, &rows)?; // scalar mean-masked-CE over this micro-batch
            acc_loss = Some(match acc_loss {
                Some(a) => a.add(&loss.affine(1.0 / accum as f64, 0.0)?)?,
                None => loss.affine(1.0 / accum as f64, 0.0)?,
            });
        }

        // report the mean masked CE over the whole dataset at these parameters — read before `acc_loss` moves into backward.
        // candle keeps the loss F32 through `affine`; cast to f64 only when reporting the scalar.
        // each micro-batch loss was already scaled by 1/accum above, so the sum IS the mean: no second division.
        let loss_scalar = acc_loss.as_ref().context("accumulated loss")?.to_scalar::<f32>()? as f64;
        let grads = acc_loss.context("accumulated loss")?.backward().context("backward")?;
        self.optimizer.step(&grads)?;
        Ok(loss_scalar)
    }

    /// Mean masked next-token cross-entropy over the given row indices. Prompt positions are excluded from the average.
    fn batch_loss(&self, model: &crate::lora::TinyLlama, data: &TrainData, rows: &[usize]) -> Res<Tensor> {
        let dev = model.device();
        let mut acc: Option<Tensor> = None;
        for r in rows {
            let logits = model.forward(&data.row_tensor(*r, &dev)?)?; // [seq_len, vocab]
            let seq_loss = masked_cross_entropy(&logits, &data.tokens[*r], data.prompt_lens[*r], &dev)?;
            acc = Some(match acc {
                Some(a) => a.add(&seq_loss)?,
                None => seq_loss,
            });
        }
        let n = rows.len() as f64;
        Ok(acc.unwrap().affine(1.0 / n, 0.0)?)
    }
}

/// The trainable adapter variables and their PEFT tensor names (`...q_proj.lora_{A,B}.weight`, layer by layer). Used to
/// build the optimizer and to save/load a checkpoint in PEFT name order.
pub fn trainable_params(model: &crate::lora::TinyLlama) -> Res<(Vec<Var>, Vec<String>)> {
    let mut vars = Vec::new();
    let mut names = Vec::new();
    for (li, layer) in model.attn.iter().enumerate() {
        let mods = ["q_proj", "k_proj", "v_proj", "o_proj"];
        for (mod_, lin) in mods.iter().zip(layer.iter()) {
            // param_refs yields lora_A then lora_B (the suffix), so the composed names match PEFT exactly.
            for (suf, _shape, var) in lin.param_refs() {
                names.push(format!("base_model.model.model.layers.{li}.self_attn.{mod_}.{suf}"));
                vars.push(var.clone());
            }
        }
    }
    Ok((vars, names))
}

/// Masked next-token cross-entropy for one sequence, as PEFT/HF compute it with prompt labels set to -100.
/// `logits` is `[seq_len, vocab]`; position `p` predicts token `p + 1`. Only answer tokens are supervised, i.e. targets
/// at index `>= prompt_len`, which are predicted from positions `prompt_len - 1 ..= seq_len - 2`. The last position has
/// no next token and is never counted (no wrap-around).
/// The target log-prob is picked with an explicit one-hot so every intermediate shape stays obvious and broadcast-free.
fn masked_cross_entropy(logits: &Tensor, tokens: &[u32], prompt_len: usize, dev: &Device) -> Res<Tensor> {
    let seq_len = tokens.len();
    let vocab = logits.dim(1)?;
    // the last row's target is unused (masked below); 0 keeps the one-hot well-formed
    let targets = (0..seq_len).map(|p| if p + 1 < seq_len { tokens[p + 1] } else { 0 }).collect::<Vec<u32>>();

    // log probabilities over the vocab keep the class pick numerically stable.
    let log_probs = candle_nn::ops::log_softmax(logits, 1)?; // [seq_len, vocab]
    // one-hot over the target token at each row: `log_probs * onehot` summed over the class dim is exactly the target
    // log-prob per row. Built explicitly so every intermediate shape is obvious ([seq_len, vocab]).
    let mut hot: Vec<f32> = vec![0f32; seq_len * vocab];
    for (r, &t) in targets.iter().enumerate() {
        hot[r * vocab + t as usize] = 1.0;
    }
    let onehot = Tensor::from_vec(hot, (seq_len, vocab), dev)?; // [seq_len, vocab]
    let prods = log_probs.mul(&onehot)?; // [seq_len, vocab] elementwise
    let logp_at_target = prods.sum(1)?; // [seq_len], chosen target log-prob per row

    // only rows that predict an answer token count: p in prompt_len-1 ..= seq_len-2 (weight 1); prompt rows and the
    // last row weigh 0.
    let first = prompt_len.saturating_sub(1);
    let mut mask: Vec<f32> = vec![0f32; seq_len];
    for p in first..seq_len.saturating_sub(1) {
        mask[p] = 1.0;
    }
    let count = mask.iter().filter(|m| **m > 0.0).count().max(1) as f64;
    let mask_t = Tensor::from_vec(mask, (seq_len,), dev)?; // [seq_len]
    let neg_logp = logp_at_target.affine(-1.0, 0.0)?; // [seq_len], per-row negative log-likelihood
    let masked = neg_logp.mul(&mask_t)?; // [seq_len], 0 on unsupervised rows
    Ok(masked.sum_all()?.affine(1.0 / count, 0.0)?)
}

/// Public view of [`masked_cross_entropy`] returning the loss scalar as an `f64` for tests and logging.
pub fn masked_loss_scalar(logits: &Tensor, tokens: &[u32], prompt_len: usize, dev: &Device) -> Res<f64> {
    Ok(masked_cross_entropy(logits, tokens, prompt_len, dev)?.to_scalar::<f32>()? as f64)
}
