//! LoRA fine-tuning of a real model ([`crate::model::LlamaModel`]) on chat data — the Workshop's `train`.
//!
//! One conversation at a time (batch 1). A step is `accumulation` conversations: each one's loss is back-propagated
//! on its own and the gradients are summed, so only one conversation's graph is ever alive; then the mean gradient is
//! clipped and AdamW ([`crate::train::AdamWState`], the crate's own, with its moments in the open) takes one step at
//! the rate of a warm-up + cosine schedule. The run is a pure function of the settings and the seed: the order of the
//! conversations, the dropout masks and the rate depend only on (seed, step), which is what makes a resumed run take
//! exactly the steps an uninterrupted one would.
//!
//! **Output.** `<out>/adapter_model.safetensors` + `adapter_config.json` in PEFT's format (the Workshop's `merge`,
//! PEFT and Unsloth read it), `training_report.json`, and `checkpoint-<step>/` folders holding the adapter, AdamW's
//! moments and the step, to resume from.
//!
//! **Limited resources.** See [`TrainSettings`]: a 16-bit base, LoRA in the top N blocks only (the blocks below run
//! as plain inference), and the activation cache ([`ActCache`]): the hidden state entering the first trained block is
//! computed once per conversation and stored on disk, so later epochs never run the lower blocks — which can then
//! leave memory, or never be loaded when the cache is already complete.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context as _, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::ParamsAdamW;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::chatdata::{self, ChatTokenizer, DataFormat, Example};
use crate::model::{self, BaseDtype, Ctx, LlamaModel, LoraSpec, ModelConfig};
use crate::train::{hf_cosine_lr, AdamWState};

pub const ADAPTER_FILE: &str = "adapter_model.safetensors";
pub const CONFIG_FILE: &str = "adapter_config.json";
pub const OPTIMIZER_FILE: &str = "optimizer.safetensors";
pub const STATE_FILE: &str = "trainer_state.json";
pub const REPORT_FILE: &str = "training_report.json";

/// Where a run continues from.
#[derive(Debug, Clone, PartialEq)]
pub enum Resume {
    /// the newest `checkpoint-<step>` in the output folder, if there is one
    Auto,
    Dir(PathBuf),
}

/// Everything a training run depends on.
#[derive(Debug, Clone)]
pub struct TrainSettings {
    /// the base model: a Hugging Face folder
    pub model: PathBuf,
    pub data: PathBuf,
    pub data_format: DataFormat,
    /// a second file to measure on; otherwise `val_split` of the data is held out
    pub eval_data: Option<PathBuf>,
    pub val_split: f64,
    pub out: PathBuf,
    pub lora: LoraSpec,
    /// f16 / bf16 hold the frozen base in half the memory (LoRA and the loss stay f32)
    pub base_dtype: BaseDtype,
    /// peak learning rate
    pub lr: f64,
    pub epochs: f64,
    /// stop after this many optimizer steps, whatever `epochs` says
    pub max_steps: Option<usize>,
    /// conversations per optimizer step (the batch is always 1; this is the effective batch)
    pub accumulation: usize,
    /// longer conversations are cut to this many tokens
    pub max_len: usize,
    pub warmup_ratio: f64,
    pub weight_decay: f64,
    /// clip the gradient's norm to this (0: no clipping)
    pub max_grad_norm: f64,
    pub seed: u64,
    /// the activation cache's folder (needs `lora.train_layers`)
    pub cache_dir: Option<PathBuf>,
    /// write a checkpoint every N steps (0: only when stopped and at the end)
    pub save_steps: usize,
    /// measure the held-out loss every N steps (0: only at the end)
    pub eval_steps: usize,
    pub logging_steps: usize,
    pub resume: Option<Resume>,
    /// a chat template file, for a model that has none
    pub chat_template: Option<PathBuf>,
    /// score every token, not only the assistant's (Soup's `train_on_responses_only: false`)
    pub score_everything: bool,
}

impl TrainSettings {
    /// Soup's defaults where it has them (epochs 3, warm-up 3 %, weight decay 0.01, clip 1.0, accumulation 4,
    /// 10 % held out, 2048 tokens), LoRA's usual rate.
    pub fn new(model: impl Into<PathBuf>, data: impl Into<PathBuf>, out: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(), data: data.into(), data_format: DataFormat::Auto, eval_data: None, val_split: 0.1, out: out.into(),
            lora: LoraSpec::default(), base_dtype: BaseDtype::F32, lr: 2e-4, epochs: 3.0, max_steps: None, accumulation: 4, max_len: 2048,
            warmup_ratio: 0.03, weight_decay: 0.01, max_grad_norm: 1.0, seed: 42, cache_dir: None, save_steps: 100, eval_steps: 0,
            logging_steps: 10, resume: None, chat_template: None, score_everything: false,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !(self.lr > 0.0 && self.lr.is_finite()) {
            bail!("the learning rate must be above 0");
        }
        if !(self.epochs > 0.0) {
            bail!("epochs must be above 0");
        }
        if self.accumulation == 0 {
            bail!("gradient accumulation must be 1 or more");
        }
        if self.max_len < 8 {
            bail!("a sequence length of {} is too short to hold a question and an answer", self.max_len);
        }
        if !(0.0..=0.5).contains(&self.val_split) {
            bail!("the validation split must be between 0 and 0.5");
        }
        if !(0.0..=0.5).contains(&self.warmup_ratio) {
            bail!("the warm-up ratio must be between 0 and 0.5");
        }
        Ok(())
    }

    /// The facts a checkpoint must share with the run that resumes it, or the steps would not be the same steps.
    fn fingerprint(&self, n_train: usize, total_steps: usize) -> Value {
        json!({
            "lr": self.lr, "accumulation": self.accumulation, "max_len": self.max_len, "seed": self.seed, "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay, "max_grad_norm": self.max_grad_norm, "train_examples": n_train, "total_steps": total_steps,
            "r": self.lora.r, "alpha": self.lora.alpha, "dropout": self.lora.dropout, "use_rslora": self.lora.use_rslora,
            "targets": self.lora.targets.iter().map(|t| t.as_str()).collect::<Vec<_>>(), "train_layers": self.lora.train_layers,
            "base_dtype": self.base_dtype.as_str(), "score_everything": self.score_everything,
        })
    }
}

/// What a run tells its caller while it works.
#[derive(Debug, Clone)]
pub enum Event {
    Log(String),
    /// an optimizer step finished: `loss` is the mean over its conversations, before the update
    Step { step: usize, total: usize, loss: f64, lr: f64, eta_seconds: f64 },
    Eval { step: usize, loss: f64 },
    Checkpoint { step: usize, dir: PathBuf },
    /// the activation cache is being filled
    Cache { done: usize, total: usize },
}

/// The outcome of [`train`].
#[derive(Debug, Clone)]
pub struct TrainReport {
    pub out: PathBuf,
    pub steps_done: usize,
    pub total_steps: usize,
    /// true when a stop request ended the run early (a checkpoint was written)
    pub stopped: bool,
    /// the loss of every step taken in this process (before its update)
    pub losses: Vec<f64>,
    pub eval_loss: Option<f64>,
    pub train_examples: usize,
    pub eval_examples: usize,
    pub trainable_params: u64,
    pub seconds: f64,
    pub tokens_seen: u64,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_bytes: u64,
    /// the lower blocks were never loaded (the cache was complete) or were freed after filling it
    pub lower_in_memory: bool,
    pub peak_rss_bytes: Option<u64>,
}

/// The process's peak resident memory (Linux `VmHWM`), when the OS says.
pub fn peak_rss_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb: u64 = s.lines().find(|l| l.starts_with("VmHWM:"))?.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A permutation of `0..n` that depends only on `(seed, salt)` (Fisher–Yates).
pub fn permutation(seed: u64, salt: u64, n: usize) -> Vec<usize> {
    let mut state = seed ^ salt.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ 0x51_7C_C1_B7_27_22_0A_95;
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (splitmix(&mut state) % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

// ---------------------------------------------------------------------------------------------------------------------
// the activation cache
// ---------------------------------------------------------------------------------------------------------------------

/// The hidden states entering the first trained block, one file per conversation, keyed by its token ids.
///
/// The folder's `manifest.json` records what the states depend on — the model's files, the first trained block and
/// the base dtype — and a cache made for something else is refused, not reused. A state is stored as f32, exactly as
/// computed, so a run reading the cache has the same losses as one computing the states.
pub struct ActCache {
    dir: PathBuf,
    pub hits: usize,
    pub misses: usize,
    pub bytes_written: u64,
}

/// What the cached states depend on: the config, each weight file's name, size and first mebibyte, the first trained
/// block and the base dtype.
pub fn cache_fingerprint(model_dir: &Path, first_trained: usize, base_dtype: BaseDtype) -> Result<Value> {
    use std::io::Read as _;
    let mut weights = Vec::new();
    for f in model::weight_files(model_dir)? {
        let mut head = vec![0u8; 1 << 20];
        let mut file = std::fs::File::open(&f)?;
        let mut n = 0;
        while n < head.len() {
            let got = file.read(&mut head[n..])?;
            if got == 0 {
                break;
            }
            n += got;
        }
        weights.push(json!({"file": f.file_name().and_then(|n| n.to_str()), "bytes": file.metadata()?.len(), "head_sha256": format!("{:x}", Sha256::digest(&head[..n]))}));
    }
    let config = std::fs::read(model_dir.join("config.json"))?;
    Ok(json!({"format": 1, "config_sha256": format!("{:x}", Sha256::digest(&config)), "weights": weights,
              "first_trained_block": first_trained, "base_dtype": base_dtype.as_str()}))
}

impl ActCache {
    /// Open (or start) a cache folder for `fingerprint`.
    pub fn open(dir: &Path, fingerprint: &Value) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("{}", dir.display()))?;
        let manifest = dir.join("manifest.json");
        match std::fs::read_to_string(&manifest) {
            Ok(text) => {
                let have: Value = serde_json::from_str(&text).with_context(|| format!("{}", manifest.display()))?;
                if &have != fingerprint {
                    bail!("the activation cache in {} was made for another model, base dtype or number of trained blocks; choose another folder or delete this one", dir.display());
                }
            }
            Err(_) => std::fs::write(&manifest, serde_json::to_string_pretty(fingerprint)?)?,
        }
        Ok(Self { dir: dir.to_path_buf(), hits: 0, misses: 0, bytes_written: 0 })
    }

    fn path(&self, tokens: &[u32]) -> PathBuf {
        // FNV-1a over the ids; the file also holds the ids, so a clash is detected, never used
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for t in tokens {
            for b in t.to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        self.dir.join(format!("{h:016x}-{}.safetensors", tokens.len()))
    }

    pub fn has(&self, tokens: &[u32]) -> bool {
        self.path(tokens).exists()
    }

    pub fn get(&mut self, tokens: &[u32], dev: &Device) -> Result<Option<Tensor>> {
        let p = self.path(tokens);
        if !p.exists() {
            return Ok(None);
        }
        let mut t = candle_core::safetensors::load(&p, dev).with_context(|| format!("{}", p.display()))?;
        let ids: Vec<u32> = t.remove("tokens").ok_or_else(|| anyhow!("{}: no tokens", p.display()))?.to_vec1()?;
        if ids != tokens {
            return Ok(None);
        }
        self.hits += 1;
        Ok(Some(t.remove("h").ok_or_else(|| anyhow!("{}: no hidden state", p.display()))?))
    }

    pub fn put(&mut self, tokens: &[u32], h: &Tensor) -> Result<()> {
        let p = self.path(tokens);
        let tmp = p.with_extension("tmp");
        let map: HashMap<String, Tensor> = [("h".to_string(), h.to_dtype(DType::F32)?), ("tokens".to_string(), Tensor::from_vec(tokens.to_vec(), (tokens.len(),), h.device())?)].into();
        candle_core::safetensors::save(&map, &tmp)?;
        self.bytes_written += std::fs::metadata(&tmp)?.len();
        std::fs::rename(&tmp, &p)?;
        self.misses += 1;
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// data
// ---------------------------------------------------------------------------------------------------------------------

/// The tokenised training and held-out conversations of a run.
#[derive(Debug, Clone)]
pub struct Data {
    pub train: Vec<Example>,
    pub eval: Vec<Example>,
    /// what was left out or cut, for the log
    pub notes: Vec<String>,
}

/// Read, tokenise and split the data as `settings` says.
pub fn prepare_data(s: &TrainSettings) -> Result<Data> {
    let tok = ChatTokenizer::load(&s.model, s.chat_template.as_deref())?;
    let mut notes = Vec::new();
    let mut load = |path: &Path| -> Result<Vec<Example>> {
        let (records, bad) = chatdata::read_jsonl(path, s.data_format)?;
        if !bad.is_empty() {
            notes.push(format!("{}: {} unreadable rows left out (line {}: {})", path.display(), bad.len(), bad[0].0, bad[0].1));
        }
        let t = chatdata::tokenize(&tok, &records, s.max_len, s.score_everything);
        if t.skipped_count > 0 {
            notes.push(format!("{}: {} conversations left out ({})", path.display(), t.skipped_count, t.skipped.join("; ")));
        }
        if t.truncated > 0 {
            notes.push(format!("{}: {} conversations reach the {}-token cap and were cut there", path.display(), t.truncated, s.max_len));
        }
        if t.examples.is_empty() {
            bail!("{}: no usable conversation", path.display());
        }
        Ok(t.examples)
    };
    let all = load(&s.data)?;
    let (train, eval) = match &s.eval_data {
        Some(p) => (all, load(p)?),
        None => {
            let n = all.len();
            let held = if s.val_split > 0.0 && n >= 2 { ((n as f64 * s.val_split).round() as usize).clamp(1, n - 1) } else { 0 };
            let order = permutation(s.seed, 0xE7A1, n);
            let is_eval: std::collections::HashSet<usize> = order[n - held..].iter().copied().collect();
            let (mut tr, mut ev) = (Vec::new(), Vec::new());
            for (i, ex) in all.into_iter().enumerate() {
                if is_eval.contains(&i) { ev.push(ex) } else { tr.push(ex) }
            }
            (tr, ev)
        }
    };
    Ok(Data { train, eval, notes })
}

/// Optimizer steps for `n` conversations: `ceil(ceil(n · epochs) / accumulation)`, capped by `max_steps`.
pub fn total_steps(n: usize, epochs: f64, accumulation: usize, max_steps: Option<usize>) -> usize {
    let micro = (n as f64 * epochs).ceil() as usize;
    let steps = micro.div_ceil(accumulation.max(1)).max(1);
    max_steps.map_or(steps, |m| steps.min(m.max(1)))
}

// ---------------------------------------------------------------------------------------------------------------------
// the run
// ---------------------------------------------------------------------------------------------------------------------

/// A loaded model with its adapters, optimizer and data: what `train` and `find_lr` both drive.
struct Runner {
    model: LlamaModel,
    names: Vec<String>,
    vars: Vec<Var>,
    opt: AdamWState,
    data: Data,
    cache: Option<ActCache>,
    seed: u64,
    accumulation: usize,
    max_grad_norm: f64,
    /// per-epoch order of the training conversations
    orders: HashMap<usize, Vec<usize>>,
    tokens_seen: u64,
    lower_in_memory: bool,
}

impl Runner {
    fn new(s: &TrainSettings, data: Data, on_event: &mut dyn FnMut(Event)) -> Result<Self> {
        let dev = Device::Cpu;
        let cfg = ModelConfig::load(&s.model)?;
        let first = s.lora.first_trained(cfg.num_hidden_layers);
        if let Some(n) = s.lora.train_layers {
            if n == 0 || n > cfg.num_hidden_layers {
                bail!("--train-layers {n}: the model has {} blocks; choose 1 to {}", cfg.num_hidden_layers, cfg.num_hidden_layers);
            }
        }
        let mut cache = match &s.cache_dir {
            Some(_) if first == 0 => {
                on_event(Event::Log("the activation cache is off: every block trains, so there is no frozen part to cache (use --train-layers N)".into()));
                None
            }
            Some(dir) => Some(ActCache::open(dir, &cache_fingerprint(&s.model, first, s.base_dtype)?)?),
            None => None,
        };
        // with every state already on disk the lower blocks are never loaded
        let complete = cache.as_ref().is_some_and(|c| data.train.iter().chain(&data.eval).all(|ex| c.has(&ex.tokens)));
        let mut model = LlamaModel::load_from(&s.model, s.base_dtype, &dev, if complete { first } else { 0 })?;
        model.add_lora(&s.lora)?;
        let mut lower_in_memory = !complete;
        if let Some(c) = cache.as_mut() {
            if complete {
                on_event(Event::Log(format!("activation cache complete: blocks 0 to {} were not loaded", first - 1)));
            } else {
                let all: Vec<&Example> = data.train.iter().chain(&data.eval).collect();
                for (i, ex) in all.iter().enumerate() {
                    if !c.has(&ex.tokens) {
                        c.put(&ex.tokens, &model.lower(&ex.tokens)?)?;
                    }
                    if (i + 1) % 16 == 0 || i + 1 == all.len() {
                        on_event(Event::Cache { done: i + 1, total: all.len() });
                    }
                }
                // from here on the states come from disk: the frozen blocks can leave memory
                model.drop_lower();
                lower_in_memory = false;
                on_event(Event::Log(format!("activation cache filled ({} states, {:.1} MB): blocks 0 to {} freed", c.misses, c.bytes_written as f64 / 1e6, first - 1)));
            }
        }
        let (names, vars): (Vec<String>, Vec<Var>) = model.lora_vars().into_iter().unzip();
        let mut params = ParamsAdamW::default();
        params.lr = s.lr;
        params.weight_decay = s.weight_decay;
        let opt = AdamWState::new(vars.clone(), params).map_err(|e| anyhow!("{e}"))?;
        Ok(Self { model, names, vars, opt, data, cache, seed: s.seed, accumulation: s.accumulation, max_grad_norm: s.max_grad_norm, orders: HashMap::new(),
                  tokens_seen: 0, lower_in_memory })
    }

    fn hidden(&mut self, ex: &Example) -> Result<Tensor> {
        if let Some(c) = self.cache.as_mut() {
            if let Some(h) = c.get(&ex.tokens, self.model.device())? {
                return Ok(h);
            }
            let h = self.model.lower(&ex.tokens)?;
            c.put(&ex.tokens, &h)?;
            return Ok(h);
        }
        self.model.lower(&ex.tokens)
    }

    /// The training conversation that micro-batch `m` of the whole run uses.
    fn example_at(&mut self, m: usize) -> usize {
        let n = self.data.train.len();
        let (epoch, seed) = (m / n, self.seed);
        self.orders.entry(epoch).or_insert_with(|| permutation(seed, epoch as u64 + 1, n))[m % n]
    }

    /// One optimizer step at rate `lr` over micro-batches `step·accumulation ..`, as many as `count`. Returns the mean
    /// loss before the update. A loss that is not a number stops the run without touching the adapters.
    fn step(&mut self, step: usize, count: usize, lr: f64) -> Result<f64> {
        let mut sum: Vec<Option<Tensor>> = vec![None; self.vars.len()];
        let mut loss_sum = 0.0;
        for k in 0..count {
            let m = step * self.accumulation + k;
            let index = self.example_at(m);
            let ex = self.data.train[index].clone();
            let h = self.hidden(&ex)?;
            let mut dropout_seed = self.seed ^ (m as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let ctx = Ctx::train(splitmix(&mut dropout_seed));
            let loss = self.model.loss_from_hidden(&h, &ex.tokens, &ex.supervised, &ctx)?;
            let value = loss.to_scalar::<f32>()? as f64;
            if !value.is_finite() {
                bail!("the loss became {value} at step {} (rate {lr:.3e}): lower the learning rate", step + 1);
            }
            loss_sum += value;
            self.tokens_seen += ex.tokens.len() as u64;
            let grads = loss.backward()?;
            for (slot, var) in sum.iter_mut().zip(&self.vars) {
                if let Some(g) = grads.get(var) {
                    *slot = Some(match slot.take() {
                        Some(acc) => (acc + g)?,
                        None => g.clone(),
                    });
                }
            }
        }
        // the mean gradient over the step's conversations, clipped by its norm over all adapters
        let mut norm_sq = 0.0f64;
        for g in sum.iter_mut().flatten() {
            *g = (&*g / count as f64)?;
            norm_sq += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        }
        let norm = norm_sq.sqrt();
        if self.max_grad_norm > 0.0 && norm > self.max_grad_norm {
            for g in sum.iter_mut().flatten() {
                *g = (&*g * (self.max_grad_norm / norm))?;
            }
        }
        self.opt.params.lr = lr;
        self.opt.step_grads(&sum).map_err(|e| anyhow!("{e}"))?;
        Ok(loss_sum / count as f64)
    }

    fn eval_loss(&mut self) -> Result<Option<f64>> {
        if self.data.eval.is_empty() {
            return Ok(None);
        }
        let mut sum = 0.0;
        for i in 0..self.data.eval.len() {
            let ex = self.data.eval[i].clone();
            let h = self.hidden(&ex)?;
            sum += self.model.loss_from_hidden(&h, &ex.tokens, &ex.supervised, &Ctx::eval())?.to_scalar::<f32>()? as f64;
        }
        Ok(Some(sum / self.data.eval.len() as f64))
    }

    fn save_adapter(&self, dir: &Path, base: &Path, spec: &LoraSpec) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let tensors: HashMap<String, Tensor> = model::adapter_tensors(&self.model).into_iter().collect();
        candle_core::safetensors::save(&tensors, dir.join(ADAPTER_FILE))?;
        std::fs::write(dir.join(CONFIG_FILE), serde_json::to_string_pretty(&model::adapter_config(spec, &self.model.cfg, &base.display().to_string()))?)?;
        Ok(())
    }

    fn save_checkpoint(&self, dir: &Path, s: &TrainSettings, step: usize, fingerprint: &Value, losses: &[f64]) -> Result<()> {
        self.save_adapter(dir, &s.model, &s.lora)?;
        let mut moments: HashMap<String, Tensor> = HashMap::new();
        for (name, (_, m, v)) in self.names.iter().zip(&self.opt.slots) {
            moments.insert(format!("{name}.exp_avg"), m.as_tensor().detach());
            moments.insert(format!("{name}.exp_avg_sq"), v.as_tensor().detach());
        }
        candle_core::safetensors::save(&moments, dir.join(OPTIMIZER_FILE))?;
        let state = json!({"global_step": step, "adam_step": self.opt.step_t, "settings": fingerprint, "losses": losses});
        std::fs::write(dir.join(STATE_FILE), serde_json::to_string_pretty(&state)?)?;
        Ok(())
    }

    /// Restore adapters, moments and the step from a checkpoint made with the same settings; returns the step and
    /// the losses so far. Everything is checked before anything is changed.
    fn load_checkpoint(&mut self, dir: &Path, fingerprint: &Value) -> Result<(usize, Vec<f64>)> {
        let state: Value = serde_json::from_str(&std::fs::read_to_string(dir.join(STATE_FILE)).with_context(|| format!("{}: not a checkpoint (no {STATE_FILE})", dir.display()))?)?;
        if &state["settings"] != fingerprint {
            let differ: Vec<String> = fingerprint.as_object().into_iter().flatten()
                .filter(|(k, v)| &&state["settings"][k.as_str()] != v)
                .map(|(k, v)| format!("{k}: checkpoint {}, now {v}", state["settings"][k.as_str()])).collect();
            bail!("{} was made with other settings ({}); a resumed run must repeat them", dir.display(), differ.join("; "));
        }
        let dev = self.model.device().clone();
        let adapter = candle_core::safetensors::load(dir.join(ADAPTER_FILE), &dev)?;
        let moments = candle_core::safetensors::load(dir.join(OPTIMIZER_FILE), &dev)?;
        for (name, var) in self.names.iter().zip(&self.vars) {
            for (key, map) in [(name.clone(), &adapter), (format!("{name}.exp_avg"), &moments), (format!("{name}.exp_avg_sq"), &moments)] {
                let t = map.get(&key).ok_or_else(|| anyhow!("{}: no tensor {key}", dir.display()))?;
                if t.dims() != var.as_tensor().dims() {
                    bail!("{key}: shape {:?} in the checkpoint, {:?} in the model", t.dims(), var.as_tensor().dims());
                }
            }
        }
        let step = state["global_step"].as_u64().ok_or_else(|| anyhow!("{STATE_FILE}: no global_step"))? as usize;
        let adam = state["adam_step"].as_u64().ok_or_else(|| anyhow!("{STATE_FILE}: no adam_step"))? as usize;
        for (name, (theta, m, v)) in self.names.iter().zip(&self.opt.slots) {
            theta.set(&adapter[name])?;
            m.set(&moments[&format!("{name}.exp_avg")])?;
            v.set(&moments[&format!("{name}.exp_avg_sq")])?;
        }
        self.opt.step_t = adam;
        let losses = state["losses"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
        Ok((step, losses))
    }
}

/// The newest `checkpoint-<step>` folder of an output folder.
pub fn latest_checkpoint(out: &Path) -> Option<PathBuf> {
    std::fs::read_dir(out).ok()?.flatten()
        .filter_map(|e| {
            let step: usize = e.file_name().to_str()?.strip_prefix("checkpoint-")?.parse().ok()?;
            e.path().join(STATE_FILE).exists().then_some((step, e.path()))
        })
        .max_by_key(|(step, _)| *step).map(|(_, p)| p)
}

/// Train a LoRA adapter. `on_event` hears the progress; `should_stop` is asked after every step, and a yes writes a
/// checkpoint and ends the run (`stopped` in the report).
pub fn train(s: &TrainSettings, on_event: &mut dyn FnMut(Event), should_stop: &dyn Fn() -> bool) -> Result<TrainReport> {
    s.validate()?;
    let t0 = Instant::now();
    let data = prepare_data(s)?;
    for n in &data.notes {
        on_event(Event::Log(n.clone()));
    }
    let (n_train, n_eval) = (data.train.len(), data.eval.len());
    let total = total_steps(n_train, s.epochs, s.accumulation, s.max_steps);
    let total_micro = ((n_train as f64 * s.epochs).ceil() as usize).min(total * s.accumulation);
    let warmup = (s.warmup_ratio * total as f64).ceil() as usize;
    let fingerprint = s.fingerprint(n_train, total);
    let mut run = Runner::new(s, data, on_event)?;
    let trainable: u64 = run.vars.iter().map(|v| v.as_tensor().elem_count() as u64).sum();
    on_event(Event::Log(format!(
        "{} conversations to train on, {} held out; {} steps of {} ({} warm-up); {} adapter parameters in blocks {} to {} of a {} base",
        n_train, n_eval, total, s.accumulation, warmup, trainable, run.model.first_trained(), run.model.cfg.num_hidden_layers - 1, s.base_dtype.as_str())));

    let (mut step, mut losses) = (0usize, Vec::new());
    let resume_dir = match &s.resume {
        Some(Resume::Dir(d)) => Some(d.clone()),
        Some(Resume::Auto) => latest_checkpoint(&s.out),
        None => None,
    };
    if let Some(dir) = resume_dir {
        (step, losses) = run.load_checkpoint(&dir, &fingerprint)?;
        on_event(Event::Log(format!("resumed from {} at step {step}", dir.display())));
    }
    let start_step = step;
    let mut stopped = false;
    let mut eval = None;
    let t_steps = Instant::now();
    while step < total {
        let count = s.accumulation.min(total_micro - step * s.accumulation).max(1);
        let lr = hf_cosine_lr(s.lr, step, warmup, total);
        let loss = run.step(step, count, lr)?;
        losses.push(loss);
        step += 1;
        let done = (step - start_step) as f64;
        let eta = t_steps.elapsed().as_secs_f64() / done * (total - step) as f64;
        if s.logging_steps <= 1 || step % s.logging_steps == 0 || step == total || step == start_step + 1 {
            on_event(Event::Step { step, total, loss, lr, eta_seconds: eta });
        }
        if s.eval_steps > 0 && step % s.eval_steps == 0 && step < total {
            if let Some(l) = run.eval_loss()? {
                on_event(Event::Eval { step, loss: l });
            }
        }
        let stop_now = step < total && should_stop();
        if stop_now || (s.save_steps > 0 && step % s.save_steps == 0 && step < total) {
            let dir = s.out.join(format!("checkpoint-{step}"));
            run.save_checkpoint(&dir, s, step, &fingerprint, &losses)?;
            on_event(Event::Checkpoint { step, dir });
        }
        if stop_now {
            stopped = true;
            break;
        }
    }
    if !stopped {
        eval = run.eval_loss()?;
        if let Some(l) = eval {
            on_event(Event::Eval { step, loss: l });
        }
        // the finished adapter, in PEFT's format, at the top of the output folder
        run.save_adapter(&s.out, &s.model, &s.lora)?;
    }
    let (hits, misses, bytes) = run.cache.as_ref().map_or((0, 0, 0), |c| (c.hits, c.misses, c.bytes_written));
    let report = TrainReport {
        out: s.out.clone(), steps_done: step, total_steps: total, stopped, losses: losses.clone(), eval_loss: eval, train_examples: n_train,
        eval_examples: n_eval, trainable_params: trainable, seconds: t0.elapsed().as_secs_f64(), tokens_seen: run.tokens_seen, cache_hits: hits,
        cache_misses: misses, cache_bytes: bytes, lower_in_memory: run.lower_in_memory, peak_rss_bytes: peak_rss_bytes(),
    };
    std::fs::create_dir_all(&s.out)?;
    std::fs::write(s.out.join(REPORT_FILE), serde_json::to_string_pretty(&json!({
        "steps_done": step, "total_steps": total, "stopped": stopped, "losses": losses, "eval_loss": eval, "train_examples": n_train,
        "eval_examples": n_eval, "trainable_params": trainable, "seconds": report.seconds, "tokens_seen": report.tokens_seen,
        "peak_rss_bytes": report.peak_rss_bytes, "settings": fingerprint, "base_model": s.model.display().to_string(),
        "activation_cache": {"hits": hits, "written": misses, "bytes_written": bytes, "lower_blocks_in_memory": report.lower_in_memory},
    }))?)?;
    Ok(report)
}

// ---------------------------------------------------------------------------------------------------------------------
// the learning-rate range test
// ---------------------------------------------------------------------------------------------------------------------

/// The range test's settings: the rate rises geometrically from `start` to `end` over `steps` optimizer steps.
#[derive(Debug, Clone, Copy)]
pub struct FindLr {
    pub start: f64,
    pub end: f64,
    pub steps: usize,
}

impl Default for FindLr {
    fn default() -> Self {
        Self { start: 1e-7, end: 1.0, steps: 40 }
    }
}

/// What the curve says.
#[derive(Debug, Clone, PartialEq)]
pub struct LrSuggestion {
    /// the rate where the smoothed loss was falling fastest; `None` when it never fell
    pub suggested: Option<f64>,
    /// the rate at the lowest smoothed loss (too high to train at: the loss turns there)
    pub min_loss_lr: f64,
    /// the first rate after the minimum whose smoothed loss is over 4× the minimum
    pub diverged_at: Option<f64>,
    pub smoothed: Vec<f64>,
}

/// The outcome of [`find_lr`].
#[derive(Debug, Clone)]
pub struct LrReport {
    pub lrs: Vec<f64>,
    pub losses: Vec<f64>,
    pub suggestion: LrSuggestion,
    /// the run ended early because the loss blew up (which is what a range test is for)
    pub diverged: bool,
}

/// Read a suggestion off a range test's (rate, loss) curve: smooth the losses (an exponential average, bias
/// corrected), find the minimum, and before it the place where the loss falls fastest per unit of log-rate. The rate
/// at the start of that fall is the suggestion (a step back from the steepest point, as Soup and fastai advise).
pub fn suggest_lr(lrs: &[f64], losses: &[f64]) -> Result<LrSuggestion> {
    if lrs.len() != losses.len() {
        bail!("{} rates for {} losses", lrs.len(), losses.len());
    }
    if lrs.len() < 4 {
        bail!("a range test needs at least 4 steps, this one has {}", lrs.len());
    }
    const BETA: f64 = 0.7;
    let mut avg = 0.0;
    let smoothed: Vec<f64> = losses.iter().enumerate().map(|(i, l)| {
        avg = BETA * avg + (1.0 - BETA) * l;
        avg / (1.0 - BETA.powi(i as i32 + 1))
    }).collect();
    let min_idx = (0..smoothed.len()).min_by(|a, b| smoothed[*a].total_cmp(&smoothed[*b])).unwrap_or(0);
    let diverged_at = (min_idx + 1..smoothed.len()).find(|i| smoothed[*i].abs() > 4.0 * smoothed[min_idx].abs()).map(|i| lrs[i]);
    let mut best = (0.0f64, 0usize);
    for i in 1..=min_idx {
        let d_lr = lrs[i].ln() - lrs[i - 1].ln();
        if d_lr > 0.0 {
            let slope = (smoothed[i] - smoothed[i - 1]) / d_lr;
            if slope < best.0 {
                best = (slope, i);
            }
        }
    }
    Ok(LrSuggestion { suggested: (best.0 < 0.0).then(|| lrs[best.1 - 1]), min_loss_lr: lrs[min_idx], diverged_at, smoothed })
}

/// The learning-rate range test: a short run whose rate rises geometrically, recording each step's loss. Nothing is
/// written; the adapters trained here are thrown away.
pub fn find_lr(s: &TrainSettings, opts: FindLr, on_event: &mut dyn FnMut(Event)) -> Result<LrReport> {
    s.validate()?;
    if !(opts.start > 0.0 && opts.end > opts.start) || opts.steps < 4 {
        bail!("a range test needs 0 < start < end and at least 4 steps");
    }
    let data = prepare_data(s)?;
    for n in &data.notes {
        on_event(Event::Log(n.clone()));
    }
    let mut run = Runner::new(s, data, on_event)?;
    let (mut lrs, mut losses) = (Vec::new(), Vec::new());
    let mut best = f64::INFINITY;
    let mut diverged = false;
    for step in 0..opts.steps {
        let lr = opts.start * (opts.end / opts.start).powf(step as f64 / (opts.steps - 1) as f64);
        let loss = match run.step(step, s.accumulation, lr) {
            Ok(l) => l,
            // a loss that is not a number is the end of the curve, not a failure
            Err(e) if e.to_string().contains("the loss became") => {
                diverged = true;
                break;
            }
            Err(e) => return Err(e),
        };
        lrs.push(lr);
        losses.push(loss);
        on_event(Event::Step { step: step + 1, total: opts.steps, loss, lr, eta_seconds: 0.0 });
        best = best.min(loss);
        if step >= 4 && loss > 4.0 * best {
            diverged = true;
            break;
        }
    }
    let suggestion = suggest_lr(&lrs, &losses)?;
    Ok(LrReport { lrs, losses, suggestion, diverged })
}
