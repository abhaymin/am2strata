//! `am2strata-workshop`: the Workshop's jobs as a command, in Rust (no Python).
//!
//! ```text
//! am2strata-workshop info <model.gguf | hf-folder>
//! am2strata-workshop convert <hf-folder> --out <model.gguf> [--outtype f16|f32] [--pre-tokenizer NAME]
//! am2strata-workshop quantize <in.gguf> --out <out.gguf> --type q8_0|q6_k|q5_k|q4_k|q4_0|f16
//! am2strata-workshop merge <base-hf-folder> <adapter-folder> --out <merged-hf-folder>
//! ```
//!
//! With `--json` every line on stdout is one event: `{"event":"start",...}`, `{"event":"log","message":...}`,
//! `{"event":"done","output":...,"bytes":...,"seconds":...}` or `{"event":"error","message":...}`. The server's Workshop
//! runs jobs this way (a job that runs out of memory cannot take the server down); people get readable lines.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use strata_workshop::{convert, datagen, merge, quantize_gguf};

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

fn run(cmd: &Cmd, out: &Out) -> Result<(PathBuf, String), String> {
    match cmd {
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
            unsafe { libc::signal(libc::SIGTERM, on_term as *const () as libc::sighandler_t); }
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

fn main() -> ExitCode {
    let cli = Cli::parse();
    let out = Out { json: cli.json };
    let t0 = Instant::now();
    match run(&cli.cmd, &out) {
        Ok((path, kind)) => {
            if kind != "info" {
                out.event(json!({"event": "done", "output": path.display().to_string(), "kind": kind,
                                 "bytes": size_of(&path), "seconds": t0.elapsed().as_secs_f64()}));
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            out.event(json!({"event": "error", "message": e}));
            ExitCode::from(1)
        }
    }
}
