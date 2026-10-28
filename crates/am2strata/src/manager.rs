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
use tokio::sync::RwLock;

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
    /// #25: answers' thinking by their text, for clients that send an answer back without it (the last 512)
    pub reasoning_memory: std::sync::Mutex<std::collections::VecDeque<(String, String)>>,
    /// the tokens the engine holds after the last request (its prompt and reply), for the prompt snapshots
    pub held: std::sync::Mutex<Vec<u32>>,
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

    fn flag(&self, key: &str) -> bool {
        self.entry.cfg.rest.get(key) == Some(&serde_json::Value::Bool(true))
    }

    /// #25 (opt-in "preserve_thinking"): earlier turns keep their thinking in the prompt.
    pub fn preserve_thinking(&self) -> bool {
        self.flag("preserve_thinking")
    }

    /// #1053 (opt-in "reasoning_close_retry"): a reply that ends inside its thinking is continued once, closed.
    pub fn reasoning_close_retry(&self) -> bool {
        self.flag("reasoning_close_retry")
    }

    /// #728 "reasoning_loop_recovery": None (off), Some(false) = "stop", Some(true) = "recover" (or true).
    pub fn loop_recovery(&self) -> Option<bool> {
        match self.entry.cfg.rest.get("reasoning_loop_recovery") {
            Some(serde_json::Value::Bool(true)) => Some(true),
            Some(serde_json::Value::String(s)) if s == "recover" => Some(true),
            Some(serde_json::Value::String(s)) if s == "stop" => Some(false),
            _ => None,
        }
    }

    /// #25: remember an answer's thinking by its text.
    pub fn remember_reasoning(&self, text: &str, reasoning: &str) {
        let (key, reasoning) = (text.trim(), reasoning.trim());
        if key.is_empty() || reasoning.is_empty() {
            return;
        }
        let mut m = self.reasoning_memory.lock().unwrap();
        m.retain(|(k, _)| k != key);
        m.push_back((key.to_string(), reasoning.to_string()));
        while m.len() > 512 {
            m.pop_front();
        }
    }

    /// #25: earlier answers get back the thinking this server wrote for them, where the client left it out.
    pub fn restore_reasoning(&self, messages: &mut [serde_json::Value]) {
        let m = self.reasoning_memory.lock().unwrap();
        for msg in messages.iter_mut() {
            if msg.get("role").and_then(serde_json::Value::as_str) != Some("assistant") {
                continue;
            }
            let Some(text) = msg.get("content").and_then(serde_json::Value::as_str) else { continue };
            let has = msg.get("reasoning_content").is_some_and(|r| r.as_str().is_some_and(|s| !s.is_empty()));
            if has || text.contains("</think>") {
                continue;
            }
            if let Some((_, r)) = m.iter().find(|(k, _)| k == text.trim()) {
                msg["reasoning_content"] = serde_json::json!(r);
            }
        }
    }

    /// The thinking level of a request that names none: the shared Chat settings first, then the config's.
    pub fn config_effort(&self) -> Option<String> {
        self.entry.cfg.rest.get("reasoning_effort").and_then(serde_json::Value::as_str).map(str::to_owned)
    }
}

pub struct Manager {
    /// the installed models, scanned again when they are listed or one is loaded (a model pulled meanwhile shows up)
    models: std::sync::RwLock<Vec<ModelEntry>>,
    models_dir: PathBuf,
    pub hw: Hardware,
    pub default_id: String,
    current: RwLock<Option<Arc<Loaded>>>,
    /// one job at a time on the GPU - chat requests, model switches, media jobs - in the queue's order (queue.rs)
    pub queue: Arc<crate::queue::Queue>,
    pub ready_timeout: Duration,
    pub monitor: Arc<crate::monitor::Monitor>,
    /// facts and conversation archives (memory.rs), under <data>/memory
    pub memory: Arc<crate::memory::Memory>,
    /// prompt snapshots on the SSD (snapshots.rs), under <data>/cache/kv
    pub snapshots: Arc<crate::snapshots::Snapshots>,
    /// media jobs a chat's tools queue (media_jobs.rs; set once the server is built)
    pub media_jobs: std::sync::OnceLock<Arc<crate::media_jobs::Jobs>>,
    /// web_search / fetch_url (web_tools.rs; set from the run config at start)
    pub web: std::sync::OnceLock<Arc<crate::web_tools::WebTools>>,
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
    pub fn new(models: Vec<ModelEntry>, models_dir: PathBuf, default_id: String, ready_timeout: Duration, memory_dir: PathBuf,
               queue_max_wait: Option<Duration>) -> Self {
        Self {
            models: std::sync::RwLock::new(models),
            models_dir,
            hw: Hardware::detect(),
            default_id,
            current: RwLock::new(None),
            queue: crate::queue::Queue::new(queue_max_wait),
            ready_timeout,
            monitor: crate::monitor::Monitor::new(),
            memory: Arc::new(crate::memory::Memory::open(&memory_dir)),
            snapshots: Arc::new(crate::snapshots::Snapshots::open(&memory_dir)),
            media_jobs: std::sync::OnceLock::new(),
            web: std::sync::OnceLock::new(),
        }
    }

    /// The installed models, scanned again (a config pulled or edited meanwhile is seen without a restart).
    /// The folder of the run configs (PolyStrata's).
    pub fn models_dir(&self) -> PathBuf {
        self.models_dir.clone()
    }

    pub fn models(&self) -> Vec<ModelEntry> {
        let fresh = scan(&self.models_dir);
        if !fresh.is_empty() {
            *self.models.write().unwrap() = fresh;
        }
        self.models.read().unwrap().clone()
    }

    pub fn entry(&self, id: &str) -> Option<ModelEntry> {
        let models = self.models();
        models.iter().find(|m| m.id == id).cloned().or_else(|| {
            // a part of a name is enough when only one name has it
            let hits: Vec<&ModelEntry> = models.iter().filter(|m| m.id.contains(id)).collect();
            (hits.len() == 1).then(|| hits[0].clone())
        })
    }

    pub async fn current(&self) -> Option<Arc<Loaded>> {
        self.current.read().await.clone()
    }

    /// The loaded model, loading the default one when none is (a request after an unload starts it again).
    pub async fn ensure(&self) -> Result<Arc<Loaded>> {
        self.ensure_with_turn().await.map(|(l, _)| l)
    }

    /// The model a request runs on. When none is loaded (stopped for a media job, or never started) the request
    /// waits its place in the queue, the model loads in its turn, and the turn comes back with it - so the request
    /// does not queue a second time to run.
    pub async fn ensure_with_turn(&self) -> Result<(Arc<Loaded>, Option<crate::queue::Pass>)> {
        self.ensure_with_turn_told(|_| {}).await
    }

    /// ensure_with_turn, telling `on_position` the request's place while it waits.
    pub async fn ensure_with_turn_told(&self, on_position: impl FnMut(usize)) -> Result<(Arc<Loaded>, Option<crate::queue::Pass>)> {
        if let Some(l) = self.current().await {
            return Ok((l, None));
        }
        let id = self.default_id.clone();
        let w = self.monitor.waiting();
        let p = self.queue.enter(crate::queue::Need::Text(id.clone()), "chat", &format!("{id} (loads it)"), on_position).await
            .map_err(|t| anyhow!("{t}"))?;
        drop(w);
        let l = self.load_locked(&id).await?;
        Ok((l, Some(p)))
    }

    /// The GPU's turn for one request with `model` loaded; held until the request ends. It waits its place in the
    /// queue (no time limit unless queue_max_wait_s), and the caller makes sure the model is loaded once it has it.
    pub async fn take_turn(&self, model: &str, label: &str, on_position: impl FnMut(usize))
                           -> std::result::Result<crate::queue::Pass, crate::queue::TooLong> {
        let w = self.monitor.waiting();
        let p = self.queue.enter(crate::queue::Need::Text(model.to_string()), "chat", label, on_position).await;
        drop(w);
        p
    }

    /// The whole GPU for a media job: it waits its place in the queue; with `unload_text` the text model is stopped
    /// first (it loads again with the next chat), freeing its VRAM.
    pub async fn gpu_exclusive(&self, unload_text: bool) -> Result<crate::queue::Pass> {
        self.gpu_exclusive_as(unload_text, "media", "").await
    }

    pub async fn gpu_exclusive_as(&self, unload_text: bool, kind: &str, label: &str) -> Result<crate::queue::Pass> {
        let w = self.monitor.waiting();
        let need = if unload_text { crate::queue::Need::Gpu } else { crate::queue::Need::Any };
        let p = self.queue.enter(need, kind, label, |_| {}).await
            .map_err(|t| anyhow!("{t}"))?;
        drop(w);
        if unload_text {
            if let Some(l) = self.current.write().await.take() {
                tracing::info!(model = %l.entry.id, "stopping the text model for a media job (it loads again with the next request)");
                self.snapshots.save(&l, "before the model stops").await;
                stop(&l).await;
            }
            self.queue.set_resident(crate::queue::Resident::Gpu);
        }
        Ok(p)
    }

    pub async fn unload(&self) {
        // an unload always waits its turn, even behind a full queue
        let _p = loop {
            match self.queue.enter(crate::queue::Need::Any, "unload", "", |_| {}).await {
                Ok(p) => break p,
                Err(_) => tokio::time::sleep(Duration::from_secs(1)).await,
            }
        };
        if let Some(l) = self.current.write().await.take() {
            self.snapshots.save(&l, "before the model stops").await;
            stop(&l).await;
        }
        self.queue.set_resident(crate::queue::Resident::None);
    }

    /// Loads a model (a switch from the web app or the API): in its place in the queue, after the jobs before it.
    pub async fn load(&self, id: &str) -> Result<Arc<Loaded>> {
        let entry = self.entry(id).ok_or_else(|| anyhow!("no installed model is named {id:?}"))?;
        if let Some(l) = self.current().await {
            if l.entry.id == entry.id {
                return Ok(l);
            }
        }
        let _p = self.queue.enter(crate::queue::Need::Text(entry.id.clone()), "switch", &entry.id, |_| {}).await
            .map_err(|t| anyhow!("{t}"))?;
        self.load_locked(&entry.id).await
    }

    /// Loads `id` (stopping the model in memory first) for the job that holds the turn.
    pub async fn load_locked(&self, id: &str) -> Result<Arc<Loaded>> {
        let entry = self.entry(id).ok_or_else(|| anyhow!("no installed model is named {id:?}"))?;
        if let Some(l) = self.current().await {
            if l.entry.id == entry.id {
                self.queue.set_resident(crate::queue::Resident::Text(entry.id.clone()));
                return Ok(l);
            }
        }
        let before = self.current.write().await.take();
        let before_id = before.as_ref().map(|l| l.entry.id.clone());
        if let Some(l) = before {
            self.snapshots.save(&l, "before the model stops").await;
            stop(&l).await;
        }
        match start(&entry, &self.hw, self.ready_timeout).await {
            Ok(l) => {
                let l = Arc::new(l);
                *self.current.write().await = Some(l.clone());
                self.queue.set_resident(crate::queue::Resident::Text(entry.id.clone()));
                Ok(l)
            }
            Err(e) => {
                if let Some(prev) = before_id.and_then(|id| self.entry(&id)) {
                    tracing::warn!("loading {} failed ({e:#}); starting {} again", entry.id, prev.id);
                    if let Ok(l) = start(&prev, &self.hw, self.ready_timeout).await {
                        *self.current.write().await = Some(Arc::new(l));
                        self.queue.set_resident(crate::queue::Resident::Text(prev.id.clone()));
                    }
                } else {
                    self.queue.set_resident(crate::queue::Resident::None);
                }
                Err(e)
            }
        }
    }

    /// The models as GET /api/models lists them.
    pub async fn list(&self) -> Value {
        let cur = self.current().await.map(|l| l.entry.id.clone());
        let models: Vec<Value> = self
            .models()
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
    ensure_pack(&cfg, &entry.id).await?;
    if !cfg.tokenizer.join("vocab.json").exists() {
        ensure_tokenizer(&cfg).with_context(|| format!("tokenizer of {}", entry.id))?;
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
    if call_format == strata_frontend::formats::CallFormat::Glm {
        // #70 (PolyStrata a8d9d6b0): GLM ends its turn with <|user|> and a tool call with <|observation|>, which
        // llama.cpp's converter names eot and eom; a GGUF can name neither (REAP50: eos only) and the reply then ran
        // on. Flint, which writes GLM's calls, also opens a developer turn.
        for end in ["<|user|>", "<|observation|>", "<|developer|>"] {
            stop_ids.extend(one(end));
        }
        think_open_ids = tok.encode("<think>", true).unwrap_or_default();
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
        reasoning_memory: Default::default(),
        held: Default::default(),
    })
}

/// A config that names a pack (`--pack DIR`) not built yet - a model just downloaded, or one whose files were updated -
/// gets it built here from the config's model file, as `am2strata-run iq-pack` builds it, before its engine starts.
async fn ensure_pack(cfg: &strata_engine::RunConfig, id: &str) -> Result<()> {
    let Some(dir) = cfg.args.windows(2).find(|w| w[0] == "--pack").map(|w| std::path::PathBuf::from(&w[1])) else { return Ok(()) };
    if dir.join("index.txt").exists() {
        return Ok(());
    }
    let gguf = cfg.args.windows(2).find(|w| w[0] == "--model" || w[0] == "--native").map(|w| std::path::PathBuf::from(&w[1]))
        .with_context(|| format!("{id}: the config names a pack ({}) but no model file to build it from", dir.display()))?;
    tracing::info!(model = %id, "building the pack {} from {} (first load, or the model files changed)", dir.display(), gguf.display());
    let t0 = std::time::Instant::now();
    let o = strata_pack::iq_pack::Options { gguf, out: dir.clone(), base: None, skip_experts: false, compat_bf16: false, experts_bin: false };
    tokio::task::spawn_blocking(move || strata_pack::iq_pack::run(&o)).await?
        .map_err(|e| anyhow!("building the pack for {id}: {e}"))?;
    tracing::info!(model = %id, "pack built in {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}

/// A config whose tokenizer folder is missing (a shared config beside freshly downloaded files): write it from the
/// model's GGUF, as `am2strata-run pull` does. The folder must be named `tokenizer`, as packs lay it out.
fn ensure_tokenizer(cfg: &strata_engine::RunConfig) -> Result<()> {
    let dir = &cfg.tokenizer;
    anyhow::ensure!(dir.file_name().is_some_and(|n| n == "tokenizer"),
                    "{} does not exist (only a folder named tokenizer can be made from the model)", dir.display());
    let gguf = cfg.args.windows(2).find(|w| w[0] == "--model" || w[0] == "--native").map(|w| std::path::PathBuf::from(&w[1]))
        .or_else(|| cfg.vision.as_ref().map(|v| v.model.clone()))
        .with_context(|| format!("{} does not exist, and the config names no model file to make it from", dir.display()))?;
    let pack = dir.parent().unwrap_or(std::path::Path::new("."));
    tracing::info!("writing the tokenizer from {} into {}", gguf.display(), dir.display());
    let g = strata_gguf::Gguf::open(&gguf).with_context(|| format!("reading {}", gguf.display()))?;
    strata_gguf::extract_tokenizer(&g, pack).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}
