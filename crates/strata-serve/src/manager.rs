//! The installed models (every `strata-*.json` run config in the models directory) and the one in memory.
//! A switch stops the loaded engine and image encoder first, so two models are never in memory together, then starts
//! the new one with launch settings adapted to the GPU (see `adapt`). On failure the earlier model is started again.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use strata_engine::hardware::{arch_supported, engine_archs, Hardware, VISION_GPU_MIN_GB};
use strata_engine::{child_env, Engine, RunConfig, Vision};
use strata_frontend::{PromptBuilder, ToolOrders};
use strata_template::ChatTemplate;
use strata_tokenizer::Tokenizer;
use tokio::sync::{Mutex, OwnedMutexGuard, RwLock};

#[derive(Clone)]
pub struct ModelEntry {
    pub id: String,
    pub path: PathBuf,
    pub cfg: RunConfig,
}

impl ModelEntry {
    pub fn display_name(&self) -> String {
        self.cfg.display_name.clone().unwrap_or_else(|| self.cfg.model_name.clone())
    }
}

pub struct Loaded {
    pub entry: ModelEntry,
    pub tok: Arc<Tokenizer>,
    pub prompts: PromptBuilder,
    pub engine: Engine,
    pub vision: Option<Vision>,
    pub turn_ids: Vec<u32>,
    /// how the model writes tool calls (by its tokenizer's pre-tokenizer)
    pub call_format: strata_frontend::formats::CallFormat,
    /// the tokens that end the reply (serve/server.py stop_ids)
    pub stop_ids: std::collections::HashSet<u32>,
    /// Gemma's own opening of its reasoning: a prompt that ends on it starts the reply inside the reasoning
    pub think_open_ids: Vec<u32>,
    pub max_context: usize,
    pub family: Option<String>,
    /// image placeholder token, the marker before it, and the placeholder's text ids
    pub pad: Option<u32>,
    pub vision_start: Option<u32>,
    pub pad_literal: Vec<u32>,
    pub tool_orders: ToolOrders,
    /// what the launch adapted to this machine (shown in /health and the log)
    pub adapted: Vec<String>,
    /// the Chat settings shared with other apps (POST /settings), kept next to the run config
    pub shared: std::sync::RwLock<crate::settings::Defaults>,
    /// the run config's `sampling` block
    pub sampling_defaults: crate::settings::Defaults,
}

impl Loaded {
    /// The sampling defaults of a request: the model card's for this thinking state, the config's block, then the
    /// shared Chat settings (the request's own fields win over all of them).
    pub fn defaults(&self, thinking: bool) -> crate::settings::Defaults {
        let mut d = crate::run::model_card_defaults(self.family.as_deref(), thinking);
        d.extend(self.sampling_defaults.clone());
        d.extend(self.shared.read().unwrap().clone());
        d
    }

    /// The thinking level of a request that names none: the shared Chat settings first, then the config's.
    pub fn config_effort(&self) -> Option<String> {
        self.entry.cfg.rest.get("reasoning_effort").and_then(serde_json::Value::as_str).map(str::to_owned)
    }
}

pub struct Manager {
    pub models: Vec<ModelEntry>,
    pub hw: Hardware,
    pub default_id: String,
    current: RwLock<Option<Arc<Loaded>>>,
    /// one request at a time on the GPU (generation and image encoding), and switches wait for it
    pub turn: Arc<Mutex<()>>,
    switching: Mutex<()>,
    pub ready_timeout: Duration,
    pub monitor: Arc<crate::monitor::Monitor>,
}

/// Every run config in `dir` that names an engine (strata-*.json), sorted by id.
pub fn scan(dir: &Path) -> Vec<ModelEntry> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    for e in rd.flatten() {
        let p = e.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !(name.starts_with("strata-") && name.ends_with(".json")) {
            continue;
        }
        match RunConfig::load(&p) {
            Ok(cfg) if cfg.exe.exists() || cfg.exe.is_relative() => out.push(ModelEntry { id: cfg.model_name.clone(), path: p, cfg }),
            Ok(_) | Err(_) => tracing::debug!(path = %p.display(), "not a usable run config"),
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// Launch settings for this machine. The engine sizes its expert cache to the free VRAM by itself; here: the
/// engine binary must be built for this GPU, and the image encoder goes to the CPU below 15.5 GB of VRAM so a
/// 12 GB card keeps its VRAM for experts (PolyStrata setup's rule).
pub fn adapt(cfg: &RunConfig, hw: &Hardware) -> Result<(RunConfig, bool, Vec<String>)> {
    let mut cfg = cfg.clone();
    let mut notes = Vec::new();
    let gpu = hw.gpu(cfg.first_gpu());
    if let Some(g) = gpu {
        let archs = engine_archs(&cfg.exe);
        if !arch_supported(&archs, g.arch) {
            bail!(
                "{} is compiled for sm_{} but this GPU ({}) is sm_{}: rebuild the engine with CUDA architecture {} \
                 included (e.g. CMAKE_CUDA_ARCHITECTURES=\"86;120\" for both an RTX 3060 and an RTX 5060 Ti)",
                cfg.exe.display(),
                archs.iter().map(u32::to_string).collect::<Vec<_>>().join("/sm_"),
                g.name, g.arch, g.arch
            );
        }
        notes.push(format!("GPU {}: {} MiB, sm_{}", g.name, g.vram_mib, g.arch));
    }
    let vram_gb = gpu.map(|g| g.vram_mib as f64 / 1024.0);
    let mut vision_gpu = cfg.vision.as_ref().is_some_and(|v| v.gpu);
    if vision_gpu && vram_gb.is_some_and(|v| v < VISION_GPU_MIN_GB) {
        vision_gpu = false;
        notes.push(format!("image encoder on the CPU: {:.1} GB of VRAM is under {VISION_GPU_MIN_GB} GB, so it stays with the experts",
                           vram_gb.unwrap()));
    }
    let poly = cfg.exe.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains("strata-poly"));
    if !poly && cfg.vision.is_some() && !cfg.args.iter().any(|a| a == "--vision") {
        cfg.args.push("--vision".into());
        if vision_gpu && !cfg.args.iter().any(|a| a == "--vram-reserve-mib") {
            cfg.args.extend(["--vram-reserve-mib".into(), "700".into()]);
        }
        notes.push("added --vision (the config has a vision section)".into());
    }
    Ok((cfg, vision_gpu, notes))
}

impl Manager {
    pub fn new(models: Vec<ModelEntry>, default_id: String, ready_timeout: Duration) -> Self {
        Self {
            models,
            hw: Hardware::detect(),
            default_id,
            current: RwLock::new(None),
            turn: Arc::new(Mutex::new(())),
            switching: Mutex::new(()),
            ready_timeout,
            monitor: crate::monitor::Monitor::new(),
        }
    }

    pub fn entry(&self, id: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.id == id).or_else(|| {
            // a part of a name is enough when only one name has it
            let hits: Vec<&ModelEntry> = self.models.iter().filter(|m| m.id.contains(id)).collect();
            (hits.len() == 1).then(|| hits[0])
        })
    }

    pub async fn current(&self) -> Option<Arc<Loaded>> {
        self.current.read().await.clone()
    }

    /// The loaded model, loading the default one when none is (a request after an unload starts it again).
    pub async fn ensure(&self) -> Result<Arc<Loaded>> {
        if let Some(l) = self.current().await {
            return Ok(l);
        }
        let id = self.default_id.clone();
        self.load(&id).await
    }

    /// The GPU's turn for one request; held until the request ends.
    pub async fn take_turn(&self) -> OwnedMutexGuard<()> {
        self.turn.clone().lock_owned().await
    }

    pub async fn unload(&self) {
        let _s = self.switching.lock().await;
        let _t = tokio::time::timeout(Duration::from_secs(120), self.turn.lock()).await;
        if let Some(l) = self.current.write().await.take() {
            stop(&l).await;
        }
    }

    pub async fn load(&self, id: &str) -> Result<Arc<Loaded>> {
        let entry = self.entry(id).cloned().ok_or_else(|| anyhow!("no installed model is named {id:?}"))?;
        let _s = self.switching.lock().await;
        if let Some(l) = self.current().await {
            if l.entry.id == entry.id {
                return Ok(l);
            }
        }
        // a running request finishes first (up to two minutes), then the model in memory is stopped
        let _t = tokio::time::timeout(Duration::from_secs(120), self.turn.lock())
            .await
            .map_err(|_| anyhow!("a request is still running after two minutes; try the switch again when it ends"))?;
        let before = self.current.write().await.take();
        let before_id = before.as_ref().map(|l| l.entry.id.clone());
        if let Some(l) = before {
            stop(&l).await;
        }
        match start(&entry, &self.hw, self.ready_timeout).await {
            Ok(l) => {
                let l = Arc::new(l);
                *self.current.write().await = Some(l.clone());
                Ok(l)
            }
            Err(e) => {
                if let Some(prev) = before_id.and_then(|id| self.entry(&id).cloned()) {
                    tracing::warn!("loading {} failed ({e:#}); starting {} again", entry.id, prev.id);
                    if let Ok(l) = start(&prev, &self.hw, self.ready_timeout).await {
                        *self.current.write().await = Some(Arc::new(l));
                    }
                }
                Err(e)
            }
        }
    }

    /// The models as GET /api/models lists them.
    pub async fn list(&self) -> Value {
        let cur = self.current().await.map(|l| l.entry.id.clone());
        let models: Vec<Value> = self
            .models
            .iter()
            .map(|m| {
                json!({
                    "id": m.id, "display_name": m.display_name(), "config": m.path.display().to_string(),
                    "context": m.cfg.max_context(), "multimodal": m.cfg.vision.is_some(),
                    "current": cur.as_deref() == Some(m.id.as_str()),
                    "status": if cur.as_deref() == Some(m.id.as_str()) { "loaded" } else { "available" },
                    "problem": Value::Null,
                })
            })
            .collect();
        json!({"models": models})
    }
}

async fn stop(l: &Loaded) {
    tracing::info!(model = %l.entry.id, "stopping");
    l.engine.shutdown().await;
    if let Some(v) = &l.vision {
        v.shutdown().await;
    }
}

async fn start(entry: &ModelEntry, hw: &Hardware, ready_timeout: Duration) -> Result<Loaded> {
    let (cfg, vision_gpu, adapted) = adapt(&entry.cfg, hw)?;
    for n in &adapted {
        tracing::info!(model = %entry.id, "{n}");
    }
    let tok = Arc::new(Tokenizer::from_dir(&cfg.tokenizer).with_context(|| format!("tokenizer of {}", entry.id))?);
    let template = Arc::new(
        ChatTemplate::from_file(cfg.tokenizer.join("chat_template.jinja")).with_context(|| format!("chat template of {}", entry.id))?,
    );
    let one = |t: &str| tok.encode(t, true).ok().filter(|v| v.len() == 1).map(|v| v[0]);
    let pad = one("<|image_pad|>");
    let vision_start = template.source().contains("<|vision_start|>").then(|| one("<|vision_start|>")).flatten();
    let pad_literal = tok.encode("<|image_pad|>", false).unwrap_or_default();
    let call_format = strata_frontend::formats::CallFormat::from_pre(tok.pre());
    let turn_ids: Vec<u32> = call_format.turn_openers().iter().filter_map(|t| one(t)).collect();
    let mut stop_ids: std::collections::HashSet<u32> = match one("<|im_end|>") {
        Some(im_end) => [Some(im_end), one("<|endoftext|>")].into_iter().flatten().collect(),
        // not a ChatML model (GLM): the ids its GGUF names for the end of the text, the turn and the message
        None => ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"]
            .iter()
            .filter_map(|k| tok.special_id(k))
            .collect(),
    };
    let mut think_open_ids = Vec::new();
    if call_format == strata_frontend::formats::CallFormat::Gemma {
        // after its calls Gemma opens the tools' answer, which is the client's to give; and a finetune's GGUF can
        // name only <eos>, so the end of its turn is a stop by its text too
        stop_ids.extend(one("<|tool_response>"));
        stop_ids.extend(one("<turn|>"));
        think_open_ids = tok.encode(strata_frontend::formats::GEMMA_THOUGHT, true).unwrap_or_default();
    }
    let family = cfg.family();
    // the image encoder first: the engine sizes its expert cache to the VRAM it finds free, so an encoder started
    // after it finds none (PolyStrata starts them in this order too)
    let vision = match &cfg.vision {
        Some(v) => {
            let env = child_env(&cfg);
            match Vision::start(v, vision_gpu, &env, cfg.log.as_deref()).await {
                Ok(vis) => Some(vis),
                Err(e) => {
                    tracing::warn!(model = %entry.id, "image encoder did not start ({e}); images are off for this model");
                    None
                }
            }
        }
        None => None,
    };
    tracing::info!(model = %entry.id, family = ?family, "starting the engine");
    let engine = match Engine::start(&cfg, ready_timeout).await {
        Ok(e) => e,
        Err(e) => {
            if let Some(v) = &vision {
                v.shutdown().await;
            }
            return Err(anyhow::Error::new(e).context(format!("engine of {}", entry.id)));
        }
    };
    let sampling_defaults = crate::settings::config_sampling(&cfg.rest).map_err(anyhow::Error::msg)?;
    let max_context = cfg.max_context();
    let prompts = PromptBuilder::new(template, tok.clone()).with_format(call_format);
    Ok(Loaded {
        entry: ModelEntry { cfg, ..entry.clone() },
        tok,
        prompts,
        engine,
        vision,
        turn_ids,
        call_format,
        stop_ids,
        think_open_ids,
        max_context,
        family,
        pad,
        vision_start,
        pad_literal,
        tool_orders: ToolOrders::default(),
        adapted,
        shared: std::sync::RwLock::new(crate::settings::read_shared(&entry.path)),
        sampling_defaults,
    })
}
