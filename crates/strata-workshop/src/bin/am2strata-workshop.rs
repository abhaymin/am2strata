//! `am2strata-workshop`: the Workshop's jobs as a command, in Rust (no Python).
//!
//! ```text
//! am2strata-workshop info <model.gguf | hf-folder>
//! am2strata-workshop convert <hf-folder> --out <model.gguf> [--outtype f16|f32] [--pre-tokenizer NAME]
//! am2strata-workshop quantize <in.gguf> --out <out.gguf> --type q8_0|q6_k|q5_k|q4_k|q4_0|f16
//! am2strata-workshop merge <base-hf-folder> <adapter-folder> --out <merged-hf-folder>
//! am2strata-workshop train --model <hf-folder> --data <chat.jsonl> --out <adapter-folder> [--soup soup.yaml] [...]
//! am2strata-workshop find-lr --model <hf-folder> --data <chat.jsonl> [--steps 40]
//! am2strata-workshop profile (--model <hf-folder> | --preset llama-2-7b) [--data <chat.jsonl>] [...]
//! am2strata-workshop autopilot --model <hf-folder> --data <chat.jsonl> [--time-budget 2h] [--ram-budget 12GB]
//! ```
//!
//! `train` fine-tunes a LoRA adapter on the CPU (after Soup, in Rust: docs/MODEL-WORKSHOP-PLAN.md). Its progress
//! events carry `step`, `total`, `loss`, `lr` and `eta` beside `done`/`total`/`message`; SIGTERM writes a checkpoint
//! and stops, and `--resume auto` continues from it.
//!
//! With `--json` every line on stdout is one event: `{"event":"start",...}`, `{"event":"log","message":...}`,
//! `{"event":"done","output":...,"bytes":...,"seconds":...}` or `{"event":"error","message":...}`. The server's Workshop
//! runs jobs this way (a job that runs out of memory cannot take the server down); people get readable lines.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Args, Parser, Subcommand};
use serde_json::{json, Value};
use strata_workshop::chatdata::{self, ChatTokenizer, DataFormat};
use strata_workshop::finetune::{self, Event, FindLr, Resume, TrainSettings};
use strata_workshop::model::{BaseDtype, ModelConfig, Target};
use strata_workshop::profile::{self, AutopilotInput, Machine, ProfileInput};
use strata_workshop::{convert, datagen, merge, quantize_gguf, soup};

#[derive(Parser)]
#[command(name = "am2strata-workshop", about = "Convert, quantize and merge models in Rust (am2Strata's Workshop)")]
struct Cli {
    /// one JSON event per line on stdout (for the server and scripts)
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// what a GGUF or a Hugging Face model folder holds
    Info { path: PathBuf },
    /// a Hugging Face model folder (safetensors + config + tokenizer) to one GGUF, as llama.cpp's converter writes it
    Convert {
        dir: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// f16 (half the size) or f32
        #[arg(long, default_value = "f16")]
        outtype: String,
        /// name the BPE pre-tokenizer when its hash is not a known one (e.g. "llama-bpe", "qwen2", "gpt-2")
        #[arg(long)]
        pre_tokenizer: Option<String>,
    },
    /// a GGUF to another type, as `llama-quantize --pure` does
    Quantize {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// q8_0, q6_k, q5_k, q4_k, q4_0 or f16
        #[arg(long = "type")]
        ty: String,
    },
    /// training data written by the server's loaded model (AM2STRATA_URL, AM2STRATA_API_KEY): from documents (qa,
    /// chat) or from seed examples (instruct), filtered, de-duplicated and optionally judged
    Datagen {
        /// qa, chat or instruct
        #[arg(long, default_value = "qa")]
        recipe: String,
        /// documents: files or folders (text, Markdown, code, HTML, PDF)
        #[arg(long = "source")]
        sources: Vec<PathBuf>,
        /// instruct: a JSONL of example chat records
        #[arg(long)]
        seeds: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 100)]
        count: usize,
        #[arg(long, default_value_t = 4)]
        per_chunk: usize,
        #[arg(long, default_value_t = 3000)]
        chunk_chars: usize,
        #[arg(long, default_value_t = 200)]
        overlap: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f64,
        /// judge each answer 1-5 against its source and keep those at least this (0: no judge)
        #[arg(long, default_value_t = 0)]
        min_score: u8,
        #[arg(long)]
        topic: Option<String>,
        #[arg(long)]
        language: Option<String>,
    },
    /// a LoRA adapter (PEFT or Unsloth: adapter_model.safetensors + adapter_config.json) into its base model
    Merge {
        base: PathBuf,
        adapter: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// fine-tune a LoRA adapter on chat data (CPU; Llama, Mistral, Qwen2, Qwen3 decoders; a 4-bit base is not done)
    Train {
        #[command(flatten)]
        run: RunArgs,
        /// run although the estimate says it will not fit in the free memory
        #[arg(long)]
        force: bool,
    },
    /// the learning-rate range test: a short run with a rising rate, and the rate it suggests (writes nothing)
    FindLr {
        #[command(flatten)]
        run: RunArgs,
        #[arg(long, default_value_t = 1e-7)]
        start_lr: f64,
        #[arg(long, default_value_t = 1.0)]
        end_lr: f64,
        #[arg(long, default_value_t = 40)]
        steps: usize,
        /// write the curve and the suggestion here as JSON
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// what a training run needs (parameters, memory part by part, a rough time) against this machine, before it runs
    Profile {
        #[command(flatten)]
        run: RunArgs,
        /// a known model's shape instead of --model (no files needed): llama-2-7b, mistral-7b, llama-3.1-8b, qwen2.5-7b, qwen2.5-0.5b
        #[arg(long)]
        preset: Option<String>,
        /// conversations, when no --data is given to count them
        #[arg(long)]
        examples: Option<usize>,
        /// mean tokens per conversation, when no --data is given
        #[arg(long)]
        avg_tokens: Option<usize>,
        /// this CPU's matrix-product speed, instead of measuring it
        #[arg(long)]
        cpu_gflops: Option<f64>,
    },
    /// settings that fit this machine, the data and a time budget, each with its reason, and the train command for them
    Autopilot {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        data: PathBuf,
        #[arg(long, default_value = "auto")]
        format: String,
        /// where the adapter would go (for the printed command)
        #[arg(long)]
        out: Option<PathBuf>,
        /// how long the run may take: 90m, 2h, 1d, or seconds
        #[arg(long, default_value = "8h")]
        time_budget: String,
        /// memory the run may use: 12GB, 800MB; the free RAM when not given
        #[arg(long)]
        ram_budget: Option<String>,
        #[arg(long)]
        cpu_gflops: Option<f64>,
        #[arg(long)]
        chat_template: Option<PathBuf>,
        #[arg(long)]
        threads: Option<usize>,
    },
}

/// The settings of a run, shared by train, find-lr and profile. A `--soup` file gives them all; a flag given beside
/// it wins. Without a Soup file, what is not given takes Soup's default (see `TrainSettings::new`).
#[derive(Args, Clone, Default)]
struct RunArgs {
    /// the base model: a Hugging Face folder (config.json, safetensors, tokenizer.json)
    #[arg(long)]
    model: Option<PathBuf>,
    /// chat JSONL: {"messages": [...]} rows (also alpaca, sharegpt, prompt/completion)
    #[arg(long)]
    data: Option<PathBuf>,
    /// the adapter's folder
    #[arg(long)]
    out: Option<PathBuf>,
    /// a Soup config (soup.yaml) to take the settings from; what it sets that is not supported is refused by name
    #[arg(long)]
    soup: Option<PathBuf>,
    /// with --soup: hold a `quantization: 4bit|8bit` base in bf16 and use AdamW for another optimizer, saying so
    #[arg(long)]
    soup_relax: bool,
    /// auto, chatml, alpaca or sharegpt
    #[arg(long)]
    format: Option<String>,
    #[arg(long)]
    eval_data: Option<PathBuf>,
    /// share of the data held out to measure on, when no --eval-data
    #[arg(long)]
    val_split: Option<f64>,
    #[arg(long)]
    rank: Option<usize>,
    #[arg(long)]
    alpha: Option<f64>,
    #[arg(long)]
    dropout: Option<f64>,
    /// rsLoRA scaling: alpha / sqrt(rank)
    #[arg(long)]
    rslora: bool,
    /// of q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
    #[arg(long)]
    targets: Option<String>,
    /// LoRA in the top N blocks only: the blocks below run without a backward pass
    #[arg(long)]
    train_layers: Option<usize>,
    /// with --train-layers: store the hidden state entering the first trained block here, once per conversation
    #[arg(long)]
    cache_activations: Option<PathBuf>,
    /// hold the frozen base as f32, f16 or bf16 (half the memory; LoRA and the loss stay f32)
    #[arg(long)]
    base_dtype: Option<String>,
    #[arg(long)]
    lr: Option<f64>,
    #[arg(long)]
    epochs: Option<f64>,
    #[arg(long)]
    max_steps: Option<usize>,
    /// conversations per optimizer step (the batch is always 1)
    #[arg(long, alias = "grad-accum")]
    accumulation: Option<usize>,
    /// longer conversations are cut to this many tokens
    #[arg(long)]
    max_len: Option<usize>,
    #[arg(long)]
    warmup_ratio: Option<f64>,
    #[arg(long)]
    weight_decay: Option<f64>,
    /// 0 turns clipping off
    #[arg(long)]
    max_grad_norm: Option<f64>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    save_steps: Option<usize>,
    #[arg(long)]
    eval_steps: Option<usize>,
    #[arg(long)]
    logging_steps: Option<usize>,
    /// continue from a checkpoint folder, or `auto` for the newest in --out
    #[arg(long)]
    resume: Option<String>,
    /// a chat template file, for a model without one
    #[arg(long)]
    chat_template: Option<PathBuf>,
    /// score every token, not only the assistant's
    #[arg(long)]
    score_everything: bool,
    /// CPU threads (all cores when not given)
    #[arg(long)]
    threads: Option<usize>,
    /// cpu (GPU runs are not part of this trainer yet)
    #[arg(long, default_value = "cpu")]
    device: String,
}

impl RunArgs {
    /// The settings these flags (and the Soup file) describe, and what to say about them.
    fn settings(&self, need_data: bool) -> Result<(TrainSettings, Vec<String>), String> {
        if self.device != "cpu" {
            return Err(format!("--device {}: this trainer runs on the CPU only for now (GPU runs through the server's queue are the next milestone)", self.device));
        }
        let (mut s, mut notes) = match &self.soup {
            Some(path) => {
                let imp = soup::import_file(path, self.soup_relax).map_err(|e| e.to_string())?;
                (imp.settings, imp.notes.into_iter().map(|n| format!("soup: {n}")).collect())
            }
            None => (TrainSettings::new(PathBuf::new(), PathBuf::new(), PathBuf::new()), Vec::new()),
        };
        let soup_given = self.soup.is_some();
        if let Some(m) = &self.model { s.model = m.clone(); }
        if let Some(d) = &self.data { s.data = d.clone(); }
        if let Some(o) = &self.out { s.out = o.clone(); }
        if s.model.as_os_str().is_empty() { return Err("--model: the base model's Hugging Face folder is needed".into()); }
        if need_data && s.data.as_os_str().is_empty() { return Err("--data: the training conversations (JSONL) are needed".into()); }
        if s.out.as_os_str().is_empty() { s.out = PathBuf::from("./output"); }
        if let Some(f) = &self.format { s.data_format = DataFormat::from_s(f).map_err(|e| e.to_string())?; }
        if let Some(e) = &self.eval_data { s.eval_data = Some(e.clone()); }
        if let Some(v) = self.val_split { s.val_split = v; }
        if !soup_given {
            // LoRA's usual settings when nothing says otherwise (Soup's own default, r 64 at 2e-5, suits its 4-bit GPU runs)
            s.lora.r = 16;
            s.lora.alpha = 32.0;
            s.lora.dropout = 0.05;
        }
        if let Some(r) = self.rank { s.lora.r = r; if self.alpha.is_none() && !soup_given { s.lora.alpha = 2.0 * r as f64; } }
        if let Some(a) = self.alpha { s.lora.alpha = a; }
        if let Some(d) = self.dropout { s.lora.dropout = d; }
        if self.rslora { s.lora.use_rslora = true; }
        if let Some(t) = &self.targets { s.lora.targets = Target::parse_list(t).map_err(|e| e.to_string())?; }
        if let Some(n) = self.train_layers { s.lora.train_layers = Some(n); }
        if let Some(c) = &self.cache_activations { s.cache_dir = Some(c.clone()); }
        if let Some(d) = &self.base_dtype { s.base_dtype = BaseDtype::from_s(d).map_err(|e| e.to_string())?; }
        if let Some(v) = self.lr { s.lr = v; }
        if let Some(v) = self.epochs { s.epochs = v; }
        if let Some(v) = self.max_steps { s.max_steps = Some(v); }
        if let Some(v) = self.accumulation { s.accumulation = v; }
        if let Some(v) = self.max_len { s.max_len = v; }
        if let Some(v) = self.warmup_ratio { s.warmup_ratio = v; }
        if let Some(v) = self.weight_decay { s.weight_decay = v; }
        if let Some(v) = self.max_grad_norm { s.max_grad_norm = v; }
        if let Some(v) = self.seed { s.seed = v; s.lora.seed = v; }
        if let Some(v) = self.save_steps { s.save_steps = v; }
        if let Some(v) = self.eval_steps { s.eval_steps = v; }
        if let Some(v) = self.logging_steps { s.logging_steps = v; }
        if let Some(r) = &self.resume { s.resume = Some(if r == "auto" { Resume::Auto } else { Resume::Dir(PathBuf::from(r)) }); }
        if let Some(c) = &self.chat_template { s.chat_template = Some(c.clone()); }
        if self.score_everything { s.score_everything = true; }
        if s.cache_dir.is_some() && s.lora.train_layers.is_none() {
            notes.push("--cache-activations needs --train-layers N: with every block training there is no frozen part to cache".into());
        }
        s.validate().map_err(|e| e.to_string())?;
        Ok((s, notes))
    }
}

/// `90m`, `2h`, `1d`, `45s` or plain seconds.
fn parse_duration(s: &str) -> Result<f64, String> {
    let t = s.trim().to_ascii_lowercase();
    let (num, unit) = match t.char_indices().find(|(_, c)| c.is_ascii_alphabetic()) {
        Some((i, _)) => (&t[..i], &t[i..]),
        None => (t.as_str(), "s"),
    };
    let n: f64 = num.trim().parse().map_err(|_| format!("{s:?} is not a duration (90m, 2h, 1d, or seconds)"))?;
    Ok(n * match unit { "s" | "sec" => 1.0, "m" | "min" => 60.0, "h" => 3600.0, "d" => 86400.0, _ => return Err(format!("{s:?}: use s, m, h or d")) })
}

/// `12GB`, `800MB`, or bytes.
fn parse_bytes(s: &str) -> Result<u64, String> {
    let t = s.trim().to_ascii_lowercase();
    let (num, unit) = match t.char_indices().find(|(_, c)| c.is_ascii_alphabetic()) {
        Some((i, _)) => (&t[..i], &t[i..]),
        None => (t.as_str(), "b"),
    };
    let n: f64 = num.trim().parse().map_err(|_| format!("{s:?} is not a size (12GB, 800MB)"))?;
    Ok((n * match unit { "b" => 1.0, "kb" | "k" => 1e3, "mb" | "m" => 1e6, "gb" | "g" => 1e9, "gib" => (1u64 << 30) as f64, "mib" => (1u64 << 20) as f64, _ => return Err(format!("{s:?}: use MB or GB")) }) as u64)
}

/// Conversations, mean and 95th-percentile token counts of a data file under the model's tokenizer and template.
fn data_stats(s: &TrainSettings) -> Result<(usize, usize, usize), String> {
    let tok = ChatTokenizer::load(&s.model, s.chat_template.as_deref()).map_err(|e| e.to_string())?;
    let (records, _) = chatdata::read_jsonl(&s.data, s.data_format).map_err(|e| e.to_string())?;
    let mut counts: Vec<usize> = records.iter().filter_map(|r| tok.count(r).ok()).collect();
    if counts.is_empty() {
        return Err(format!("{}: no usable conversation", s.data.display()));
    }
    counts.sort_unstable();
    let p95 = counts[((counts.len() as f64 * 0.95).ceil() as usize).clamp(1, counts.len()) - 1];
    Ok((counts.len(), counts.iter().sum::<usize>() / counts.len(), p95))
}

/// The estimate's input for these settings: the data's size from the file when there is one.
fn profile_input(s: &TrainSettings, cfg: ModelConfig, examples: Option<usize>, avg_tokens: Option<usize>) -> Result<ProfileInput, String> {
    let (n, avg) = if s.data.as_os_str().is_empty() {
        (examples.unwrap_or(1000), avg_tokens.unwrap_or(s.max_len / 2))
    } else {
        let (n, avg, _) = data_stats(s)?;
        // what is held out is not trained on
        let held = if s.eval_data.is_none() && s.val_split > 0.0 && n >= 2 { ((n as f64 * s.val_split).round() as usize).clamp(1, n - 1) } else { 0 };
        (examples.unwrap_or(n - held), avg_tokens.unwrap_or(avg))
    };
    let max_len = s.max_len.min(cfg.max_position_embeddings.max(8));
    Ok(ProfileInput { lora: s.lora.clone(), base_dtype: s.base_dtype, max_len, accumulation: s.accumulation,
                      cache: s.cache_dir.is_some() && s.lora.first_trained(cfg.num_hidden_layers) > 0, examples: n, epochs: s.epochs, avg_tokens: avg.min(max_len), cfg })
}

fn os_threads() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status").ok()?.lines().find(|l| l.starts_with("Threads:"))?.split_whitespace().nth(1)?.parse().ok()
}

/// set by SIGTERM (a cancel from the server): datagen stops after the current request and writes what it made
static TERM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn on_term(_: libc::c_int) {
    TERM.store(true, std::sync::atomic::Ordering::SeqCst);
}

struct Out {
    json: bool,
}

impl Out {
    fn event(&self, v: Value) {
        if self.json {
            println!("{v}");
            return;
        }
        match v["event"].as_str() {
            Some("start") => println!("{} …", v["what"].as_str().unwrap_or("")),
            Some("log") => println!("  {}", v["message"].as_str().unwrap_or("")),
            Some("progress") => println!("  [{}/{}] {}", v["done"], v["total"], v["message"].as_str().unwrap_or("")),
            Some("done") => println!("done in {:.1} s: {} ({})", v["seconds"].as_f64().unwrap_or(0.0),
                                     v["output"].as_str().unwrap_or(""), human(v["bytes"].as_u64().unwrap_or(0))),
            Some("error") => eprintln!("error: {}", v["message"].as_str().unwrap_or("")),
            Some("info") => println!("{}", serde_json::to_string_pretty(&v["info"]).unwrap_or_default()),
            Some("profile") | Some("plan") | Some("find-lr") => {
                for l in v["lines"].as_array().into_iter().flatten() {
                    println!("{}", l.as_str().unwrap_or(""));
                }
            }
            _ => println!("{v}"),
        }
    }
}

fn human(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 { format!("{:.2} GB", b / 1e9) } else if b >= 1e6 { format!("{:.1} MB", b / 1e6) } else { format!("{:.0} KB", b / 1e3) }
}

fn size_of(p: &Path) -> u64 {
    if p.is_dir() {
        std::fs::read_dir(p).into_iter().flatten().flatten().map(|e| size_of(&e.path())).sum()
    } else {
        std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
    }
}

/// Free memory (MemAvailable) in bytes, when the OS says.
fn mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = s.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// The f32 working size of a model folder's safetensors (the converter holds them as f32 today).
fn f32_bytes_of(dir: &Path) -> u64 {
    let mut n = 0u64;
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) == Some("safetensors") {
            if let Ok(s) = strata_workshop::safetensors_io::Shard::open(&p) {
                for t in s.tensors() {
                    n += t.shape().iter().product::<usize>() as u64 * 4;
                }
            }
        }
    }
    n
}

/// Refuse a job whose working set will not fit in free memory, before it starts (not halfway, with the PC swapping).
fn fits(need: u64, what: &str) -> Result<(), String> {
    match mem_available() {
        Some(free) if need + (2 << 30) > free => Err(format!(
            "{what} needs about {} of memory and {} is free; free memory first (unload the text model from Settings › Models) or use a smaller model",
            human(need), human(free))),
        _ => Ok(()),
    }
}

fn info(path: &Path) -> Result<Value, String> {
    if path.is_dir() {
        let cfg: Value = std::fs::read_to_string(path.join("config.json")).ok().and_then(|t| serde_json::from_str(&t).ok())
            .ok_or_else(|| format!("{}: no config.json (not a Hugging Face model folder)", path.display()))?;
        let adapter = path.join("adapter_config.json").exists();
        return Ok(json!({
            "kind": if adapter { "lora-adapter" } else { "hf-model" },
            "architectures": cfg["architectures"], "model_type": cfg["model_type"],
            "hidden_size": cfg["hidden_size"], "layers": cfg["num_hidden_layers"], "heads": cfg["num_attention_heads"],
            "kv_heads": cfg["num_key_value_heads"], "vocab_size": cfg["vocab_size"],
            "f32_bytes": f32_bytes_of(path), "bytes": size_of(path),
        }));
    }
    let g = strata_gguf::Gguf::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut types: std::collections::BTreeMap<String, usize> = Default::default();
    for t in &g.tensors {
        *types.entry(strata_gguf::type_name(t.ggml_type)).or_default() += 1;
    }
    let m = &g.metadata;
    let arch = m.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("").to_string();
    Ok(json!({
        "kind": "gguf", "architecture": arch, "name": m.get("general.name"), "file_type": m.get("general.file_type"),
        "tensors": g.tensors.len(), "tensor_types": types, "bytes": size_of(path),
        "context_length": m.get(&format!("{arch}.context_length")), "block_count": m.get(&format!("{arch}.block_count")),
        "tokenizer": m.get("tokenizer.ggml.model"),
    }))
}

/// What a training event looks like on the wire: progress with step/total/loss/lr/eta, the rest as log lines.
fn training_event(out: &Out, e: Event) {
    match e {
        Event::Log(m) => out.event(json!({"event": "log", "message": m})),
        Event::Step { step, total, loss, lr, eta_seconds } => out.event(json!({
            "event": "progress", "done": step, "total": total, "step": step, "loss": loss, "lr": lr, "eta": eta_seconds,
            "message": format!("loss {loss:.4} · rate {lr:.3e} · {} left", profile::human_seconds(eta_seconds))})),
        Event::Eval { step, loss } => out.event(json!({"event": "log", "message": format!("held-out loss {loss:.4} at step {step}"), "step": step, "eval_loss": loss})),
        Event::Checkpoint { step, dir } => out.event(json!({"event": "log", "message": format!("checkpoint at step {step}: {}", dir.display()), "step": step, "checkpoint": dir.display().to_string()})),
        Event::Cache { done, total } => out.event(json!({"event": "progress", "done": done, "total": total, "message": "filling the activation cache"})),
    }
}

fn install_term_handler() {
    // a cancel (SIGTERM from the server) is noticed after the current step: a checkpoint is written, then the run ends
    #[cfg(unix)]
    unsafe { libc::signal(libc::SIGTERM, on_term as extern "C" fn(libc::c_int) as libc::sighandler_t); }
}

fn run(cmd: &Cmd, out: &Out) -> Result<(PathBuf, String, Value), String> {
    let plain = |r: Result<(PathBuf, String), String>| r.map(|(p, k)| (p, k, json!({})));
    match cmd {
        Cmd::Train { run, force } => {
            let (s, notes) = run.settings(true)?;
            let cfg = ModelConfig::load(&s.model).map_err(|e| format!("{e:#}"))?;
            out.event(json!({"event": "start", "what": format!("train a LoRA adapter on {} with {} into {}", s.model.display(), s.data.display(), s.out.display()),
                             "threads": run.threads, "device": "cpu", "base_dtype": s.base_dtype.as_str()}));
            for n in notes {
                out.event(json!({"event": "log", "message": n}));
            }
            out.event(json!({"event": "log", "message": format!("the base is held in {}; a 4-bit base (QLoRA) is not done in this trainer", s.base_dtype.as_str())}));
            // the estimate first: a run that will not fit is refused before it loads anything
            let p = profile_input(&s, cfg, None, None)?;
            let est = profile::estimate(&p);
            let machine = Machine::detect();
            for l in profile::report_lines(&p, &est, &machine).into_iter().skip(1).take(4) {
                out.event(json!({"event": "log", "message": l}));
            }
            let v = profile::verdict(&est, &machine);
            if v.fits_ram == Some(false) {
                let msg = format!("this run needs about {} of memory and {} is free", profile::human_bytes(v.need_bytes), profile::human_bytes(machine.ram_available));
                if !*force {
                    return Err(format!("{msg}: `am2strata-workshop autopilot` picks settings that fit (a 16-bit base, fewer trained blocks, a shorter sequence); --force runs it anyway"));
                }
                out.event(json!({"event": "log", "message": format!("{msg}; running anyway (--force)")}));
            }
            install_term_handler();
            let report = finetune::train(&s, &mut |e| training_event(out, e), &|| TERM.load(std::sync::atomic::Ordering::SeqCst)).map_err(|e| format!("{e:#}"))?;
            let extra = json!({
                "steps": report.steps_done, "total_steps": report.total_steps, "stopped": report.stopped, "final_loss": report.losses.last(),
                "first_loss": report.losses.first(), "eval_loss": report.eval_loss, "trainable_params": report.trainable_params,
                "tokens_seen": report.tokens_seen, "peak_rss_bytes": report.peak_rss_bytes, "os_threads": os_threads(),
                "estimated_bytes": est.total_bytes,
            });
            if report.stopped {
                out.event(json!({"event": "log", "message": format!("stopped at step {} of {}: continue with --resume auto", report.steps_done, report.total_steps)}));
                return Ok((s.out.join(format!("checkpoint-{}", report.steps_done)), "checkpoint".into(), extra));
            }
            Ok((s.out, "lora-adapter".into(), extra))
        }
        Cmd::FindLr { run, start_lr, end_lr, steps, report } => {
            let (s, notes) = run.settings(true)?;
            out.event(json!({"event": "start", "what": format!("learning-rate range test on {} with {} ({steps} steps, {start_lr:e} to {end_lr:e})", s.model.display(), s.data.display())}));
            for n in notes {
                out.event(json!({"event": "log", "message": n}));
            }
            let r = finetune::find_lr(&s, FindLr { start: *start_lr, end: *end_lr, steps: *steps }, &mut |e| training_event(out, e)).map_err(|e| format!("{e:#}"))?;
            let sg = &r.suggestion;
            let mut lines = vec![match sg.suggested {
                Some(lr) => format!("suggested learning rate: {lr:.2e} (where the loss was falling fastest)"),
                None => "no suggestion: the loss never fell in this range — try a lower --start-lr or more --steps".to_string(),
            }];
            lines.push(format!("lowest loss at {:.2e} (too high to train at: the loss turns there){}", sg.min_loss_lr,
                               sg.diverged_at.map(|d| format!("; the loss blew up from {d:.2e}")).unwrap_or_default()));
            if r.diverged {
                lines.push(format!("stopped after {} of {steps} steps: the loss diverged, which ends a range test", r.lrs.len()));
            }
            let v = json!({"event": "find-lr", "suggested_lr": sg.suggested, "min_loss_lr": sg.min_loss_lr, "diverged_at": sg.diverged_at, "diverged": r.diverged,
                           "lrs": r.lrs, "losses": r.losses, "smoothed": sg.smoothed, "lines": lines});
            if let Some(p) = report {
                std::fs::write(p, serde_json::to_string_pretty(&v).unwrap_or_default()).map_err(|e| format!("{}: {e}", p.display()))?;
            }
            out.event(v);
            Ok((s.model, "info".into(), json!({})))
        }
        Cmd::Profile { run, preset, examples, avg_tokens, cpu_gflops } => {
            let mut run = run.clone();
            let cfg = match preset {
                Some(name) => {
                    run.model.get_or_insert_with(|| PathBuf::from(name));
                    profile::preset(name).ok_or_else(|| format!("no preset {name:?}: {}", profile::PRESETS.join(", ")))?
                }
                None => ModelConfig::load(run.model.as_ref().ok_or("--model DIR or --preset NAME is needed")?).map_err(|e| format!("{e:#}"))?,
            };
            if preset.is_some() && run.data.is_some() {
                return Err("--preset has no tokenizer to count --data with: give --examples and --avg-tokens".into());
            }
            let (s, notes) = run.settings(false)?;
            let p = profile_input(&s, cfg, *examples, *avg_tokens)?;
            let est = profile::estimate(&p);
            let mut machine = Machine::detect();
            if let Some(g) = cpu_gflops {
                machine.cpu_gflops = *g;
            }
            let mut lines = profile::report_lines(&p, &est, &machine);
            lines.extend(notes);
            lines.push("a 4-bit base (QLoRA), expert adapters of mixture-of-experts models, several GPUs and vision towers are not done".into());
            let mut v = profile::estimate_json(&p, &est, &machine);
            v["event"] = json!("profile");
            v["lines"] = json!(lines);
            out.event(v);
            Ok((s.model, "info".into(), json!({})))
        }
        Cmd::Autopilot { model, data, format, out: dst, time_budget, ram_budget, cpu_gflops, chat_template, threads: _ } => {
            let cfg = ModelConfig::load(model).map_err(|e| format!("{e:#}"))?;
            let mut s = TrainSettings::new(model, data, dst.clone().unwrap_or_else(|| PathBuf::from("./output")));
            s.data_format = DataFormat::from_s(format).map_err(|e| e.to_string())?;
            s.chat_template = chat_template.clone();
            let (n, avg, p95) = data_stats(&s)?;
            let machine = Machine::detect();
            let budget = match ram_budget {
                Some(b) => parse_bytes(b)?,
                None if machine.ram_available > 0 => machine.ram_available,
                None => return Err("this system does not say how much RAM is free: pass --ram-budget (for example 12GB)".into()),
            };
            let held = if n >= 2 { ((n as f64 * s.val_split).round() as usize).clamp(1, n - 1) } else { 0 };
            let plan = profile::autopilot(&AutopilotInput {
                cfg, source_dtype: profile::source_dtype(model).map_err(|e| format!("{e:#}"))?, examples: n - held, p95_tokens: p95, avg_tokens: avg, budget_bytes: budget,
                time_budget_seconds: parse_duration(time_budget)?, cpu_gflops: cpu_gflops.unwrap_or(machine.cpu_gflops),
            }).map_err(|e| e.to_string())?;
            let mut command = format!("am2strata-workshop train --model {} --data {} --out {} --rank {} --alpha {} --dropout {} --targets {} --base-dtype {} --max-len {} --accumulation {} --epochs {} --lr {:e}",
                model.display(), data.display(), s.out.display(), plan.lora.r, plan.lora.alpha, plan.lora.dropout,
                plan.lora.targets.iter().map(|t| t.as_str()).collect::<Vec<_>>().join(","), plan.base_dtype.as_str(), plan.max_len, plan.accumulation, plan.epochs, plan.lr);
            if let Some(nl) = plan.lora.train_layers {
                command.push_str(&format!(" --train-layers {nl}"));
                if plan.cache {
                    command.push_str(&format!(" --cache-activations {}", s.out.join("activations").display()));
                }
            }
            let mut lines = vec![format!("{} conversations to train on ({held} held out), {avg} tokens on average, 95 % within {p95}; {} of memory and {} to work with",
                                         n - held, profile::human_bytes(budget), profile::human_seconds(parse_duration(time_budget)?))];
            lines.extend(plan.choices.iter().map(|c| format!("{:<17}{:<16}{}", c.setting, c.value, c.why)));
            lines.push(format!("estimate: {} of memory with headroom, about {} on this CPU, {} steps",
                               profile::human_bytes(profile::need_bytes(&plan.estimate)), profile::human_seconds(plan.seconds), plan.estimate.steps));
            if !plan.fits_memory {
                lines.push(format!("THIS DOES NOT FIT the memory: even the smallest plan needs {}; a smaller model is the way (a 4-bit base, QLoRA, is not done)", profile::human_bytes(profile::need_bytes(&plan.estimate))));
            }
            if !plan.fits_time {
                lines.push(format!("THIS DOES NOT FIT the time: the shortest plan takes about {}; fewer conversations, or a GPU run when that lands", profile::human_seconds(plan.seconds)));
            }
            lines.push(command.clone());
            out.event(json!({"event": "plan", "fits_memory": plan.fits_memory, "fits_time": plan.fits_time, "command": command, "lines": lines,
                "settings": {"rank": plan.lora.r, "alpha": plan.lora.alpha, "dropout": plan.lora.dropout, "targets": plan.lora.targets.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
                             "train_layers": plan.lora.train_layers, "base_dtype": plan.base_dtype.as_str(), "max_len": plan.max_len, "accumulation": plan.accumulation,
                             "epochs": plan.epochs, "lr": plan.lr, "cache_activations": plan.cache},
                "choices": plan.choices.iter().map(|c| json!({"setting": c.setting, "value": c.value, "why": c.why})).collect::<Vec<_>>(),
                "estimate": {"bytes": plan.estimate.total_bytes, "needed_with_headroom": profile::need_bytes(&plan.estimate), "seconds": plan.seconds, "steps": plan.estimate.steps,
                             "trainable_parameters": plan.estimate.trainable_params, "cache_disk_bytes": plan.estimate.cache_disk_bytes}}));
            Ok((model.clone(), "info".into(), json!({})))
        }
        other => plain(run_tools(other, out)),
    }
}

fn run_tools(cmd: &Cmd, out: &Out) -> Result<(PathBuf, String), String> {
    match cmd {
        Cmd::Train { .. } | Cmd::FindLr { .. } | Cmd::Profile { .. } | Cmd::Autopilot { .. } => unreachable!("handled by run"),
        Cmd::Info { path } => {
            out.event(json!({"event": "info", "info": info(path)?}));
            Ok((path.clone(), "info".into()))
        }
        Cmd::Convert { dir, out: dst, outtype, pre_tokenizer } => {
            let ot = convert::OutType::from_s(outtype).map_err(|e| e.to_string())?;
            fits(f32_bytes_of(dir), "converting this model")?;
            out.event(json!({"event": "start", "what": format!("convert {} to {} ({outtype})", dir.display(), dst.display())}));
            convert::convert(dir, dst, ot, pre_tokenizer.as_deref()).map_err(|e| e.to_string())?;
            Ok((dst.clone(), "gguf".into()))
        }
        Cmd::Quantize { input, out: dst, ty } => {
            let t = quantize_gguf::PureType::from_s(ty).map_err(|e| e.to_string())?;
            fits(size_of(input) * 2, "quantizing this model")?;
            out.event(json!({"event": "start", "what": format!("quantize {} to {} ({ty})", input.display(), dst.display())}));
            quantize_gguf::quantize(input, dst, t).map_err(|e| e.to_string())?;
            Ok((dst.clone(), "gguf".into()))
        }
        Cmd::Datagen { recipe, sources, seeds, out: dst, count, per_chunk, chunk_chars, overlap, temperature, min_score, topic, language } => {
            let settings = datagen::Settings {
                recipe: datagen::Recipe::from_s(recipe).map_err(|e| e.to_string())?, sources: sources.clone(), seeds: seeds.clone(),
                count: *count, per_chunk: (*per_chunk).clamp(1, 20), chunk_chars: (*chunk_chars).clamp(500, 20_000),
                overlap: *overlap, temperature: *temperature, min_score: (*min_score).min(5), topic: topic.clone(),
                language: language.clone(), out: dst.clone(),
            };
            let model = datagen::Model::from_env().map_err(|e| e.to_string())?;
            out.event(json!({"event": "start", "what": format!("{} data with {} into {}", recipe, model.name, dst.display())}));
            // a cancel (SIGTERM from the server) stops after the current request and still writes what was made
            #[cfg(unix)]
            unsafe { libc::signal(libc::SIGTERM, on_term as extern "C" fn(libc::c_int) as libc::sighandler_t); }
            let r = datagen::run(&settings, &model,
                |done, total, msg| out.event(json!({"event": "progress", "done": done, "total": total, "message": msg})),
                || TERM.load(std::sync::atomic::Ordering::SeqCst)).map_err(|e| e.to_string())?;
            out.event(json!({"event": "log", "message": format!("kept {} of {} written ({} filtered, {} duplicates, {} judged out)",
                r.kept, r.generated, r.filtered.values().sum::<usize>(), r.duplicates, r.judged_out)}));
            if r.kept == 0 && !TERM.load(std::sync::atomic::Ordering::SeqCst) {
                let why = if r.generated == 0 {
                    format!("{} of {} replies from {} held no records in the asked-for JSON: try a larger or instruction-tuned model",
                            r.unparsable, r.asked, model.name)
                } else {
                    "every record was dropped by the filters, as duplicates or by the judge: see the dataset card".to_string()
                };
                // nothing to train on: no empty dataset is left behind (its card would only repeat this)
                std::fs::remove_file(dst).ok();
                std::fs::remove_file(datagen::card_path(dst)).ok();
                return Err(format!("no records kept: {why}"));
            }
            Ok((dst.clone(), "dataset".into()))
        }
        Cmd::Merge { base, adapter, out: dst } => {
            out.event(json!({"event": "start", "what": format!("merge {} into {} as {}", adapter.display(), base.display(), dst.display())}));
            let r = merge::merge(base, adapter, dst).map_err(|e| e.to_string())?;
            let touched: Vec<&merge::TensorDelta> = r.per_tensor.iter().filter(|t| t.delta_rel_norm > 0.0).collect();
            out.event(json!({"event": "log", "message": format!("{} of {} tensors changed by the adapter", touched.len(), r.per_tensor.len())}));
            for t in touched.iter().take(8) {
                out.event(json!({"event": "log", "message": format!("{}: |ΔW|/|W| = {:.4}", t.name, t.delta_rel_norm)}));
            }
            Ok((r.out_dir, "hf-model".into()))
        }
    }
}

/// `--threads N`: candle and its matrix products size their thread pools from `RAYON_NUM_THREADS` and
/// `CANDLE_NUM_THREADS` the first time they compute. This process never changes its own environment (other threads
/// may read it, which makes `set_var` unsound); when the two variables do not say N yet, it starts itself again with
/// them set for the new process only. On Unix that is `exec`: the same process (the same PID, so the server's SIGTERM
/// still reaches the run). Elsewhere a child runs and its exit code is passed on. `None`: they say N, carry on.
fn with_threads(n: usize) -> Option<ExitCode> {
    let want = n.to_string();
    let set = |k: &str| std::env::var(k).ok().as_deref() == Some(want.as_str());
    if set("RAYON_NUM_THREADS") && set("CANDLE_NUM_THREADS") {
        return None;
    }
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(_) => return None, // no way to start again: all cores, said by the start event's thread count being absent
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.args(std::env::args_os().skip(1)).env("RAYON_NUM_THREADS", &want).env("CANDLE_NUM_THREADS", &want);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        let e = cmd.exec(); // returns only on failure
        eprintln!("--threads {n}: could not restart with the thread count set ({e}); running on all cores");
        None
    }
    #[cfg(not(unix))]
    {
        match cmd.status() {
            Ok(st) => Some(ExitCode::from(st.code().unwrap_or(1).clamp(0, 255) as u8)),
            Err(e) => {
                eprintln!("--threads {n}: could not restart with the thread count set ({e}); running on all cores");
                None
            }
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let threads = match &cli.cmd {
        Cmd::Train { run, .. } | Cmd::FindLr { run, .. } | Cmd::Profile { run, .. } => run.threads,
        Cmd::Autopilot { threads, .. } => *threads,
        _ => None,
    };
    if let Some(n) = threads.filter(|n| *n > 0) {
        if let Some(code) = with_threads(n) {
            return code;
        }
    }
    let out = Out { json: cli.json };
    let t0 = Instant::now();
    match run(&cli.cmd, &out) {
        Ok((path, kind, extra)) => {
            if kind != "info" {
                let mut done = json!({"event": "done", "output": path.display().to_string(), "kind": kind,
                                      "bytes": size_of(&path), "seconds": t0.elapsed().as_secs_f64()});
                for (k, v) in extra.as_object().into_iter().flatten() {
                    done[k.as_str()] = v.clone();
                }
                out.event(done);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            out.event(json!({"event": "error", "message": e}));
            ExitCode::from(1)
        }
    }
}
