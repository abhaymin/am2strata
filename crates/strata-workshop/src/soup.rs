//! Read a Soup config (`soup.yaml`) into this trainer's settings.
//!
//! Soup (<https://github.com/MakazhanAlpamys/Soup>, Apache-2.0) is the Python fine-tuning CLI this trainer is modelled
//! on; its config format is the reference here, so a `soup.yaml` written for `soup train` can drive the Workshop. Only
//! the format is shared: nothing of Soup runs, and none of its code is used.
//!
//! What is read: `base`, `task: sft`, `data.{train, format, val_split, max_length, train_on_responses_only}`,
//! `training.{epochs, lr, batch_size, gradient_accumulation_steps, seed, warmup_ratio, weight_decay, max_grad_norm,
//! scheduler, optimizer, quantization, save_steps, logging_steps}`, `training.lora.{r, alpha, dropout, target_modules,
//! use_rslora}` and `output`, with Soup's defaults for what is left out.
//!
//! **Everything else is refused, by name.** Soup has hundreds of settings (DPO, GRPO, DoRA, vision, quantised bases…);
//! a config that sets one this trainer does not implement is not run as if it had not: [`import`] returns
//! [`SoupError::Unsupported`] with every such key. A key that is merely switched off (`false`, `null`, empty) is
//! noted and ignored. Two settings that almost every Soup config carries have a sound stand-in here, used only when
//! the caller asks (`relax`): `quantization: 4bit | 8bit` (Soup's default; QLoRA is not done) becomes a bf16 base, and
//! an optimizer other than AdamW becomes AdamW. Both are said in the notes.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::chatdata::DataFormat;
use crate::finetune::TrainSettings;
use crate::model::{BaseDtype, LoraSpec, Target};

/// Why a Soup config was not imported.
#[derive(Debug, Clone, PartialEq)]
pub enum SoupError {
    /// not YAML, not a mapping, or a value of the wrong kind
    Invalid(String),
    /// settings this trainer does not implement, each as `path: value (why)`
    Unsupported(Vec<String>),
}

impl std::fmt::Display for SoupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SoupError::Invalid(m) => write!(f, "soup.yaml: {m}"),
            SoupError::Unsupported(keys) => write!(f, "soup.yaml sets {} thing{} this trainer does not do: {}", keys.len(), if keys.len() == 1 { "" } else { "s" }, keys.join("; ")),
        }
    }
}

impl std::error::Error for SoupError {}

/// An imported config.
#[derive(Debug, Clone)]
pub struct SoupImport {
    pub settings: TrainSettings,
    /// `base` as written (a folder, or a Hub id — which this trainer does not download)
    pub base: String,
    /// what was ignored, assumed or replaced, one line each
    pub notes: Vec<String>,
}

/// A number, also when YAML read it as text (PyYAML reads `2e-5` as a string and Soup accepts it).
fn number(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
}

fn is_off(v: &Value) -> bool {
    match v {
        Value::Null | Value::Bool(false) => true,
        Value::String(s) => s.is_empty() || s == "none",
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        _ => false,
    }
}

fn short(v: &Value) -> String {
    let s = match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    if s.chars().count() > 40 { format!("{}…", s.chars().take(40).collect::<String>()) } else { s }
}

struct Reader<'a> {
    relax: bool,
    unsupported: Vec<String>,
    notes: Vec<String>,
    invalid: Option<String>,
    root: &'a Value,
}

impl Reader<'_> {
    fn refuse(&mut self, path: &str, value: &Value, why: &str) {
        self.unsupported.push(format!("{path}: {} ({why})", short(value)));
    }

    /// Refuse (or note, when switched off) every key of `map` that is not in `known`.
    fn unknown(&mut self, prefix: &str, map: &Value, known: &[&str]) {
        for (k, v) in map.as_object().into_iter().flatten() {
            if known.contains(&k.as_str()) {
                continue;
            }
            let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
            if is_off(v) {
                self.notes.push(format!("{path}: {} — switched off, ignored", short(v)));
            } else {
                self.refuse(&path, v, "not implemented");
            }
        }
    }

    fn num(&mut self, path: &str, v: &Value, default: f64) -> f64 {
        if v.is_null() {
            return default;
        }
        match number(v) {
            Some(n) => n,
            None => {
                self.invalid.get_or_insert(format!("{path} must be a number, not {}", short(v)));
                default
            }
        }
    }
}

/// Parse and convert a Soup config. Relative `base`, `data.train` and `output` paths are taken from `dir` (the
/// config's folder). With `relax`, a quantised base becomes bf16 and another optimizer becomes AdamW (see the module).
pub fn import(yaml: &str, dir: &Path, relax: bool) -> Result<SoupImport, SoupError> {
    let root: Value = serde_norway::from_str(yaml).map_err(|e| SoupError::Invalid(format!("not valid YAML: {e}")))?;
    if !root.is_object() {
        return Err(SoupError::Invalid("the config must be a mapping (base:, data:, training:, ...)".into()));
    }
    let mut r = Reader { relax, unsupported: Vec::new(), notes: Vec::new(), invalid: None, root: &root };
    let at = |p: &Path| if p.is_absolute() { p.to_path_buf() } else { dir.join(p) };

    let base = root["base"].as_str().ok_or_else(|| SoupError::Invalid("base (the model) is missing".into()))?.to_string();
    let data = &root["data"];
    let train_file = data["train"].as_str().ok_or_else(|| SoupError::Invalid("data.train (the training file) is missing".into()))?;
    let out = root["output"].as_str().unwrap_or("./output");
    let mut s = TrainSettings::new(at(Path::new(&base)), at(Path::new(train_file)), at(Path::new(out)));
    if !s.model.is_dir() {
        r.notes.push(format!("base: {base} is not a folder here (a Hugging Face id?) — this trainer downloads nothing: fetch the model and pass its folder with --model"));
        s.model = PathBuf::from(&base);
    }

    r.unknown("", r.root, &["base", "task", "modality", "backend", "data", "training", "output", "experiment_name", "lora"]);
    if let Some(task) = root["task"].as_str() {
        if task != "sft" {
            r.refuse("task", &root["task"], "only supervised fine-tuning, task: sft");
        }
    }
    if let Some(m) = root["modality"].as_str() {
        if m != "text" {
            r.refuse("modality", &root["modality"], "text only: vision and audio towers are not trained");
        }
    }
    if let Some(b) = root["backend"].as_str() {
        r.notes.push(format!("backend: {b} — ignored: this is the Rust trainer"));
    }
    if !root["experiment_name"].is_null() {
        r.notes.push("experiment_name — ignored: no experiment tracker here".into());
    }

    // data
    r.unknown("data", data, &["train", "format", "val_split", "max_length", "train_on_responses_only"]);
    match data["format"].as_str().unwrap_or("auto") {
        "auto" => {}
        f => match DataFormat::from_s(f) {
            Ok(fmt) => s.data_format = fmt,
            Err(_) => r.refuse("data.format", &data["format"], "this trainer reads auto, chatml, alpaca and sharegpt rows"),
        },
    }
    s.val_split = r.num("data.val_split", &data["val_split"], 0.1);
    s.max_len = r.num("data.max_length", &data["max_length"], 2048.0) as usize;
    s.score_everything = data["train_on_responses_only"].as_bool() == Some(false);
    if s.score_everything {
        r.notes.push("data.train_on_responses_only: false — every token is scored, not only the assistant's".into());
    }

    // training
    let t = &root["training"];
    r.unknown("training", t, &["epochs", "lr", "batch_size", "gradient_accumulation_steps", "seed", "warmup_ratio", "weight_decay", "max_grad_norm",
                               "scheduler", "optimizer", "quantization", "save_steps", "logging_steps", "lora"]);
    s.epochs = r.num("training.epochs", &t["epochs"], 3.0);
    s.lr = r.num("training.lr", &t["lr"], 2e-5);
    let accumulation = r.num("training.gradient_accumulation_steps", &t["gradient_accumulation_steps"], 4.0) as usize;
    let batch = match &t["batch_size"] {
        Value::Null => 1,
        Value::String(a) if a == "auto" => {
            r.notes.push("training.batch_size: auto — the batch here is always 1 (least memory); only gradient_accumulation_steps conversations make a step".into());
            1
        }
        v => {
            let n = r.num("training.batch_size", v, 1.0) as usize;
            if n > 1 {
                r.notes.push(format!("training.batch_size: {n} × gradient_accumulation_steps {accumulation} = {} conversations per step, taken one at a time (the batch here is always 1)", n * accumulation));
            }
            n.max(1)
        }
    };
    s.accumulation = batch * accumulation.max(1);
    s.seed = r.num("training.seed", &t["seed"], 42.0) as u64;
    s.warmup_ratio = r.num("training.warmup_ratio", &t["warmup_ratio"], 0.03);
    s.weight_decay = r.num("training.weight_decay", &t["weight_decay"], 0.01);
    s.max_grad_norm = r.num("training.max_grad_norm", &t["max_grad_norm"], 1.0);
    s.save_steps = r.num("training.save_steps", &t["save_steps"], 100.0) as usize;
    s.logging_steps = r.num("training.logging_steps", &t["logging_steps"], 10.0) as usize;
    if let Some(sched) = t["scheduler"].as_str() {
        if sched != "cosine" {
            r.refuse("training.scheduler", &t["scheduler"], "warm-up + cosine only");
        }
    }
    if let Some(opt) = t["optimizer"].as_str() {
        if !matches!(opt, "adamw" | "adamw_torch" | "adamw_torch_fused") {
            if r.relax {
                r.notes.push(format!("training.optimizer: {opt} — replaced by AdamW (f32 moments), the one optimizer here"));
            } else {
                r.refuse("training.optimizer", &t["optimizer"], "AdamW only; --soup-relax uses AdamW instead");
            }
        }
    }
    // Soup's default is 4bit: a config that does not say `quantization: none` asks for a quantised base
    let quant = t["quantization"].as_str().unwrap_or("4bit");
    if quant != "none" {
        let shown = if t["quantization"].is_null() { Value::String("4bit, Soup's default when unset".into()) } else { t["quantization"].clone() };
        if r.relax && matches!(quant, "4bit" | "8bit") {
            s.base_dtype = BaseDtype::BF16;
            r.notes.push(format!("training.quantization: {} — QLoRA (a quantised base) is not done: the base is held in bf16 instead, which takes about {} the memory", short(&shown),
                                 if quant == "4bit" { "4 times" } else { "twice" }));
        } else {
            r.refuse("training.quantization", &shown, "QLoRA is not done: set quantization: none, or --soup-relax holds the base in bf16 instead");
        }
    }

    // lora: under training (Soup's schema); Soup's own bundled example writes it at the top level, where Soup ignores it
    let lora = match (&t["lora"], &root["lora"]) {
        (Value::Null, Value::Null) => &Value::Null,
        (l, Value::Null) => l,
        (Value::Null, top) => {
            r.notes.push("lora: at the top level — Soup's schema has it under training: (there Soup ignores this block and uses r 64); read here as training.lora, since that is what the file says".into());
            top
        }
        (l, top) => {
            r.refuse("lora", top, "set both at the top level and under training: keep one");
            l
        }
    };
    let lora_path = if t["lora"].is_null() { "lora" } else { "training.lora" };
    r.unknown(lora_path, lora, &["r", "alpha", "dropout", "target_modules", "use_rslora"]);
    let rank = r.num(&format!("{lora_path}.r"), &lora["r"], 64.0) as usize;
    if rank == 0 {
        r.refuse(&format!("{lora_path}.r"), &lora["r"], "rank 0 is full fine-tuning, which is not done");
    }
    let targets = match &lora["target_modules"] {
        Value::Null => vec![Target::Q, Target::V],
        Value::String(a) if a == "auto" => {
            r.notes.push(format!("{lora_path}.target_modules: auto — q_proj and v_proj, PEFT's own default for this family"));
            vec![Target::Q, Target::V]
        }
        Value::Array(list) => {
            let mut out = Vec::new();
            for m in list {
                match m.as_str().map(Target::from_s) {
                    Some(Ok(tg)) => out.push(tg),
                    _ => r.refuse(&format!("{lora_path}.target_modules"), m, "adapters go on q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"),
                }
            }
            out.sort();
            out.dedup();
            out
        }
        other => {
            r.refuse(&format!("{lora_path}.target_modules"), other, "a list of module names, or auto");
            vec![Target::Q, Target::V]
        }
    };
    s.lora = LoraSpec {
        r: rank.max(1), alpha: r.num(&format!("{lora_path}.alpha"), &lora["alpha"], 16.0), dropout: r.num(&format!("{lora_path}.dropout"), &lora["dropout"], 0.05),
        use_rslora: lora["use_rslora"].as_bool().unwrap_or(false), targets, train_layers: None, seed: s.seed,
    };

    if let Some(m) = r.invalid {
        return Err(SoupError::Invalid(m));
    }
    if !r.unsupported.is_empty() {
        return Err(SoupError::Unsupported(r.unsupported));
    }
    s.validate().map_err(|e| SoupError::Invalid(e.to_string()))?;
    Ok(SoupImport { settings: s, base, notes: r.notes })
}

/// [`import`] from a file.
pub fn import_file(path: &Path, relax: bool) -> Result<SoupImport, SoupError> {
    let text = std::fs::read_to_string(path).map_err(|e| SoupError::Invalid(format!("{}: {e}", path.display())))?;
    import(&text, path.parent().unwrap_or(Path::new(".")), relax)
}
