//! The Workshop on the server: convert, quantize and merge jobs (strata-workshop, all Rust), queued one at a time and
//! run as the `am2strata-workshop` command (a job that runs out of memory cannot take the server down), with their
//! progress for the web app's Workshop tab and `am2strata-run workshop`.
//!
//! - `POST /v1/workshop/jobs` `{"kind": "convert" | "quantize" | "merge", ...}` queues a job:
//!   - convert: `"input"` (a Hugging Face model folder), `"output"`? (a .gguf), `"outtype"`? (f16 | f32), `"pre_tokenizer"`?
//!   - quantize: `"input"` (a .gguf), `"output"`?, `"type"` (q8_0 | q6_k | q5_k | q4_k | q4_0 | f16)
//!   - merge: `"base"` (an HF model folder), `"adapter"` (a PEFT/Unsloth adapter folder), `"output"`? (a folder)
//!   - strata: `"input"` (a .gguf), `"name"`?, `"display_name"`?, `"overwrite"`? — make it a model the server runs: the
//!     same run config and tokenizer folder a downloaded model gets (strata-hub's config writer, which picks the engine
//!     for the architecture and refuses one no engine runs), so it appears in the model list
//!   - datagen: `"recipe"` (qa | chat | instruct), `"sources"` (documents: files or folders; qa and chat), `"seeds"`? (a
//!     JSONL of example chats; instruct), `"output"`? (a .jsonl), `"count"`?, `"per_chunk"`?, `"chunk_chars"`?,
//!     `"temperature"`?, `"min_score"`? (1-5: a judge pass), `"topic"`?, `"language"`? — training data written by the
//!     loaded model, through this server's own chat API (its address and key reach the command in its environment,
//!     never on its command line)
//! - `POST /v1/workshop/uploads?name=notes.pdf` (the file as the body) saves a document into `<data>/workshop/uploads`.
//! - `GET /v1/workshop/dataset?path=…&offset=&limit=` pages through a dataset's records (with its card);
//!   `POST /v1/workshop/dataset` `{"path", "remove": [i, …], "replace": {"i": record}}` edits it in place.
//! - `GET /v1/workshop/jobs` lists jobs, `GET /v1/workshop/jobs/{id}` one (with its log), `DELETE` cancels.
//! - `GET /v1/workshop/files` lists the models and adapters the Workshop may use, for its pickers.
//!
//! **Paths** must be inside the models folder, `<data>/workshop`, or a folder listed in the run config's
//! `"workshop": {"dirs": [...]}` — the API cannot read or write anywhere else on the PC. Only the server's own key may
//! use the Workshop (not client profiles).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::extract::{Path as UrlPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::{authorized, error, AppState};

const KINDS: [&str; 5] = ["convert", "quantize", "merge", "strata", "datagen"];

#[derive(Clone, Debug)]
pub struct Job {
    pub id: u64,
    pub kind: String,
    pub args: Vec<String>,
    pub request: Value,
    pub output: PathBuf,
    pub state: &'static str,
    pub queued_at: u64,
    pub started_at: Option<u64>,
    pub finished_at: Option<u64>,
    pub log: Vec<String>,
    pub error: Option<String>,
    pub result: Option<Value>,
    pub pid: Option<u32>,
    /// the last progress event (datagen): {"done", "total", "message"}
    pub progress: Option<Value>,
}

impl Job {
    pub fn view(&self, with_log: bool) -> Value {
        let mut v = json!({"id": self.id, "kind": self.kind, "state": self.state, "request": self.request,
                           "output": self.output.display().to_string(), "queued_at": self.queued_at,
                           "started_at": self.started_at, "finished_at": self.finished_at, "error": self.error,
                           "result": self.result, "progress": self.progress});
        if with_log {
            v["log"] = json!(self.log);
        } else {
            v["last"] = json!(self.log.last());
        }
        v
    }
}

pub struct Workshop {
    roots: Vec<PathBuf>,
    out_dir: PathBuf,
    models_dir: PathBuf,
    packs_dir: PathBuf,
    jobs: Mutex<Vec<Job>>,
    next: Mutex<u64>,
    /// jobs run one at a time, in order
    gate: tokio::sync::Mutex<()>,
    /// this server's own address and key, for jobs that call its API (datagen)
    server: Mutex<Option<(String, Option<String>)>>,
}

static WORKSHOP: std::sync::OnceLock<Arc<Workshop>> = std::sync::OnceLock::new();

pub fn get() -> Option<Arc<Workshop>> {
    WORKSHOP.get().cloned()
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// The `am2strata-workshop` command: AM2STRATA_WORKSHOP_EXE, else beside this server's binary.
pub fn exe() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("AM2STRATA_WORKSHOP_EXE") {
        return Some(PathBuf::from(p));
    }
    let me = std::env::current_exe().ok()?;
    let name = if cfg!(windows) { "am2strata-workshop.exe" } else { "am2strata-workshop" };
    let p = me.with_file_name(name);
    p.exists().then_some(p)
}

impl Workshop {
    pub fn init(models_dir: &Path, data_dir: &Path, rest: &std::collections::BTreeMap<String, Value>) -> Arc<Workshop> {
        let out_dir = data_dir.join("workshop");
        let mut roots = vec![models_dir.to_path_buf(), out_dir.clone()];
        if let Some(dirs) = rest.get("workshop").and_then(|w| w.get("dirs")).and_then(|d| d.as_array()) {
            roots.extend(dirs.iter().filter_map(|d| d.as_str()).map(|d| crate::expand_path(d)));
        }
        // every root resolved (symlinks followed) once, here; one that cannot be is left out, never compared unresolved
        std::fs::create_dir_all(&out_dir).ok();
        let roots = roots.into_iter().filter_map(|r| match r.canonicalize() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!("workshop: {} is not usable as a Workshop folder ({e}); left out", r.display());
                None
            }
        }).collect();
        let (models_dir, packs_dir) = (models_dir.to_path_buf(), data_dir.join("packs"));
        WORKSHOP.get_or_init(|| Arc::new(Workshop { roots, out_dir, models_dir, packs_dir, jobs: Mutex::new(vec![]), next: Mutex::new(1),
                                                    gate: tokio::sync::Mutex::new(()), server: Mutex::new(None) })).clone()
    }

    /// Where this server answers (bound to `host`:`port`), so a datagen job can use its chat API.
    pub fn set_server(&self, host: &str, port: u16, key: Option<String>) {
        let h = match host {
            "" | "0.0.0.0" | "::" | "[::]" => "127.0.0.1".to_string(),
            h if h.contains(':') && !h.starts_with('[') => format!("[{h}]"),
            h => h.to_string(),
        };
        *self.server.lock().unwrap() = Some((format!("http://{h}:{port}"), key));
    }

    pub fn uploads_dir(&self) -> PathBuf {
        self.out_dir.join("uploads")
    }

    /// A new file in the Workshop folder: `dir/name`, or `name-2`, `name-3`, … when it is taken.
    fn fresh(dir: &Path, stem: &str, ext: &str) -> PathBuf {
        std::fs::create_dir_all(dir).ok();
        let mut p = dir.join(format!("{stem}{ext}"));
        let mut n = 2;
        while p.exists() {
            p = dir.join(format!("{stem}-{n}{ext}"));
            n += 1;
        }
        p
    }

    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// `p` resolved, if it is inside one of the allowed folders (an output need not exist yet: its parent is checked).
    pub fn allowed(&self, p: &str, must_exist: bool) -> Result<PathBuf, String> {
        let path = crate::expand_path(p);
        if !path.is_absolute() {
            return Err(format!("{p}: give the full path (it is resolved on the server)"));
        }
        let resolved = if path.exists() {
            // symlinks followed: a link inside a Workshop folder that points outside it is outside
            path.canonicalize().map_err(|e| format!("{p}: {e}"))?
        } else if must_exist {
            return Err(format!("{p} does not exist"));
        } else {
            // a new output: the nearest folder that exists, resolved, plus the plain names after it (no "..")
            let mut base = path.as_path();
            let mut rest: Vec<&std::ffi::OsStr> = vec![];
            while !base.exists() {
                match (base.parent(), base.file_name()) {
                    (Some(parent), Some(name)) => { rest.push(name); base = parent; }
                    _ => return Err(format!("{p}: not a usable path")),
                }
            }
            if path.components().any(|c| matches!(c, std::path::Component::ParentDir | std::path::Component::CurDir)) {
                return Err(format!("{p}: \"..\" and \".\" are not allowed in a new path"));
            }
            let mut r = base.canonicalize().map_err(|e| format!("{p}: {e}"))?;
            for name in rest.iter().rev() {
                r.push(name);
            }
            r
        };
        if !self.roots.iter().any(|r| resolved.starts_with(r)) {
            return Err(format!("{p} is outside the Workshop's folders ({}); add its folder to \"workshop\": {{\"dirs\": [...]}} in the run config",
                               self.roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", ")));
        }
        // only now, inside a Workshop folder, are missing folders for a new output made
        if !must_exist {
            if let Some(parent) = resolved.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            }
        }
        Ok(resolved)
    }

    fn default_output(&self, input: &Path, suffix: &str) -> PathBuf {
        let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        self.out_dir.join(format!("{stem}{suffix}"))
    }

    /// Check a request and turn it into the command's arguments and its output path.
    pub fn plan(&self, req: &Value) -> Result<(String, Vec<String>, PathBuf), String> {
        let kind = req["kind"].as_str().ok_or_else(|| format!("\"kind\" is one of {}", KINDS.join(", ")))?;
        let s = |k: &str| req[k].as_str().filter(|v| !v.is_empty());
        std::fs::create_dir_all(&self.out_dir).ok();
        match kind {
            "convert" => {
                let input = self.allowed(s("input").ok_or("convert needs \"input\": a Hugging Face model folder")?, true)?;
                if !input.join("config.json").exists() {
                    return Err(format!("{} has no config.json: convert takes a Hugging Face model folder", input.display()));
                }
                let outtype = s("outtype").unwrap_or("f16");
                if !["f16", "f32"].contains(&outtype) {
                    return Err("\"outtype\" is f16 or f32".into());
                }
                let output = match s("output") {
                    Some(o) => self.allowed(o, false)?,
                    None => self.default_output(&input, &format!("-{outtype}.gguf")),
                };
                let mut args = vec!["convert".into(), input.display().to_string(), "--out".into(), output.display().to_string(),
                                    "--outtype".into(), outtype.into()];
                if let Some(p) = s("pre_tokenizer") {
                    args.extend(["--pre-tokenizer".into(), p.into()]);
                }
                Ok((kind.into(), args, output))
            }
            "quantize" => {
                let input = self.allowed(s("input").ok_or("quantize needs \"input\": a .gguf")?, true)?;
                let ty = s("type").ok_or("quantize needs \"type\": q8_0, q6_k, q5_k, q4_k, q4_0 or f16")?.to_lowercase();
                if !["q8_0", "q6_k", "q5_k", "q4_k", "q4_0", "f16"].contains(&ty.as_str()) {
                    return Err(format!("\"type\" {ty:?} is not one of q8_0, q6_k, q5_k, q4_k, q4_0, f16"));
                }
                let output = match s("output") {
                    Some(o) => self.allowed(o, false)?,
                    None => {
                        let stem = input.file_stem().and_then(|x| x.to_str()).unwrap_or("model");
                        let stem = stem.trim_end_matches("-f16").trim_end_matches("-f32");
                        self.out_dir.join(format!("{stem}-{}.gguf", ty.to_uppercase()))
                    }
                };
                Ok((kind.into(), vec!["quantize".into(), input.display().to_string(), "--out".into(), output.display().to_string(),
                                      "--type".into(), ty], output))
            }
            "merge" => {
                let base = self.allowed(s("base").ok_or("merge needs \"base\": the base model's Hugging Face folder")?, true)?;
                let adapter = self.allowed(s("adapter").ok_or("merge needs \"adapter\": the adapter folder")?, true)?;
                if !adapter.join("adapter_config.json").exists() {
                    return Err(format!("{} has no adapter_config.json: not a LoRA adapter folder", adapter.display()));
                }
                let output = match s("output") {
                    Some(o) => self.allowed(o, false)?,
                    None => self.default_output(&adapter, "-merged"),
                };
                Ok((kind.into(), vec!["merge".into(), base.display().to_string(), adapter.display().to_string(),
                                      "--out".into(), output.display().to_string()], output))
            }
            "strata" => {
                let input = self.allowed(s("input").ok_or("strata needs \"input\": a .gguf")?, true)?;
                let file = input.file_name().and_then(|n| n.to_str()).unwrap_or("model.gguf");
                let id = s("name").map(strata_hub::config::id_of).unwrap_or_else(|| strata_hub::config::id_of(file));
                if id.is_empty() {
                    return Err("\"name\" has no letters or digits".into());
                }
                let cfg = self.models_dir.join(format!("strata-{id}.json"));
                if cfg.exists() && req["overwrite"] != true {
                    return Err(format!("a model named {id} is already installed ({}); pick another \"name\" or set \"overwrite\"", cfg.display()));
                }
                Ok((kind.into(), vec![input.display().to_string(), id, s("display_name").unwrap_or("").to_string()], cfg))
            }
            "datagen" => {
                let recipe = s("recipe").unwrap_or("qa");
                if !["qa", "chat", "instruct"].contains(&recipe) {
                    return Err("\"recipe\" is qa, chat or instruct".into());
                }
                let mut sources = vec![];
                for v in req["sources"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                    let p = v.as_str().filter(|p| !p.is_empty()).ok_or("\"sources\" is a list of paths")?;
                    sources.push(self.allowed(p, true)?);
                }
                let seeds = s("seeds").map(|p| self.allowed(p, true)).transpose()?;
                if recipe != "instruct" && sources.is_empty() {
                    return Err(format!("{recipe} needs \"sources\": documents (files or folders) to write the data from"));
                }
                if recipe == "instruct" && seeds.is_none() {
                    return Err("instruct needs \"seeds\": a JSONL of example chats to write more like".into());
                }
                let num = |k: &str, lo: f64, hi: f64| -> Result<Option<String>, String> {
                    match &req[k] {
                        Value::Null => Ok(None),
                        v => v.as_f64().filter(|n| (lo..=hi).contains(n)).map(|n| Some(n.to_string()))
                            .ok_or_else(|| format!("\"{k}\" is a number from {lo} to {hi}")),
                    }
                };
                let output = match s("output") {
                    Some(o) if !o.ends_with(".jsonl") => return Err("\"output\" is a .jsonl file".into()),
                    Some(o) => self.allowed(o, false)?,
                    None => {
                        let from = sources.first().or(seeds.as_ref()).and_then(|p| p.file_stem()).and_then(|x| x.to_str()).unwrap_or("data");
                        Self::fresh(&self.out_dir.join("datasets"), &format!("{from}-{recipe}"), ".jsonl")
                    }
                };
                let mut args = vec!["datagen".into(), "--recipe".into(), recipe.into(), "--out".into(), output.display().to_string()];
                for p in &sources {
                    args.extend(["--source".into(), p.display().to_string()]);
                }
                if let Some(p) = &seeds {
                    args.extend(["--seeds".into(), p.display().to_string()]);
                }
                for (k, flag, lo, hi) in [("count", "--count", 1.0, 100_000.0), ("per_chunk", "--per-chunk", 1.0, 20.0),
                                          ("chunk_chars", "--chunk-chars", 500.0, 20_000.0), ("temperature", "--temperature", 0.0, 2.0),
                                          ("min_score", "--min-score", 0.0, 5.0)] {
                    if let Some(v) = num(k, lo, hi)? {
                        // integers for the integer flags (a JSON 100.0 is 100)
                        let v = if k == "temperature" { v } else { (v.parse::<f64>().unwrap_or(0.0) as u64).to_string() };
                        args.extend([flag.into(), v]);
                    }
                }
                for k in ["topic", "language"] {
                    if let Some(v) = s(k) {
                        args.extend([format!("--{k}"), v.chars().take(200).collect()]);
                    }
                }
                Ok((kind.into(), args, output))
            }
            other => Err(format!("unknown kind {other:?}: {}", KINDS.join(", "))),
        }
    }

    pub fn submit(self: &Arc<Self>, req: Value) -> Result<Job, String> {
        let (kind, args, output) = self.plan(&req)?;
        let exe = if kind == "strata" { PathBuf::new() } else {
            exe().ok_or("the am2strata-workshop command is not beside the server: build it (cargo build --release -p strata-workshop) or set AM2STRATA_WORKSHOP_EXE")?
        };
        let id = { let mut n = self.next.lock().unwrap(); let id = *n; *n += 1; id };
        let job = Job { id, kind, args, request: req, output, state: "queued", queued_at: now(), started_at: None,
                        finished_at: None, log: vec![], error: None, result: None, pid: None, progress: None };
        self.jobs.lock().unwrap().push(job.clone());
        let me = self.clone();
        tokio::spawn(async move { me.run(id, exe).await });
        Ok(job)
    }

    fn update(&self, id: u64, f: impl FnOnce(&mut Job)) {
        if let Some(j) = self.jobs.lock().unwrap().iter_mut().find(|j| j.id == id) {
            f(j);
        }
        crate::live_channel::poke();
    }

    async fn run(self: Arc<Self>, id: u64, exe: PathBuf) {
        let _turn = self.gate.lock().await;
        let Some(job) = self.job(id) else { return };
        if job.state == "cancelled" {
            return;
        }
        if job.kind == "strata" {
            self.update(id, |j| { j.state = "running"; j.started_at = Some(now()); });
            let (models, packs, args) = (self.models_dir.clone(), self.packs_dir.clone(), job.args.clone());
            let r = tokio::task::spawn_blocking(move || -> Result<Value, String> {
                let gguf = PathBuf::from(&args[0]);
                let g = strata_hub::config::inspect(&gguf)?;
                let arch = g.architecture().unwrap_or("").to_string();
                let display = if args[2].is_empty() {
                    g.metadata.get("general.name").and_then(|v| v.as_str()).map(str::to_string).unwrap_or_else(|| args[1].clone())
                } else { args[2].clone() };
                let w = strata_hub::config::write(&models, &packs, &gguf, &g, None, None, &args[1], &display)?;
                Ok(json!({"output": w.config.display().to_string(), "kind": "run-config", "model": args[1], "architecture": arch,
                          "tokenizer": w.tokenizer.display().to_string(), "display_name": display}))
            }).await.unwrap_or_else(|e| Err(e.to_string()));
            self.update(id, |j| {
                j.finished_at = Some(now());
                match r {
                    Ok(v) => {
                        j.log.push(format!("{} is installed ({}): pick it in the model list to load it",
                                           v["model"].as_str().unwrap_or(""), v["architecture"].as_str().unwrap_or("")));
                        j.result = Some(v);
                        j.state = "done";
                    }
                    Err(e) => { j.error = Some(e); j.state = "failed"; }
                }
            });
            return;
        }
        let mut cmd = tokio::process::Command::new(&exe);
        cmd.arg("--json").args(&job.args).stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).kill_on_drop(true);
        // the server's address and key for a job that uses its API: in the environment, never on the command line
        cmd.env_remove("AM2STRATA_URL").env_remove("AM2STRATA_API_KEY");
        if job.kind == "datagen" {
            if let Some((url, key)) = self.server.lock().unwrap().clone() {
                cmd.env("AM2STRATA_URL", url);
                if let Some(k) = key {
                    cmd.env("AM2STRATA_API_KEY", k);
                }
            }
        }
        // its own process group, so cancelling stops everything the job started (not only the command)
        #[cfg(unix)]
        cmd.process_group(0);
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                self.update(id, |j| { j.state = "failed"; j.error = Some(format!("{}: {e}", exe.display())); j.finished_at = Some(now()); });
                return;
            }
        };
        self.update(id, |j| { j.state = "running"; j.started_at = Some(now()); j.pid = child.id(); });
        tracing::info!("workshop: job {id} ({}) started", job.kind);
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let me = self.clone();
        let errs = tokio::spawn(async move {
            use tokio::io::AsyncBufReadExt;
            let mut lines = tokio::io::BufReader::new(stderr).lines();
            let mut tail = vec![];
            while let Ok(Some(l)) = lines.next_line().await {
                tail.push(l);
                if tail.len() > 20 { tail.remove(0); }
            }
            tail
        });
        {
            use tokio::io::AsyncBufReadExt;
            let mut lines = tokio::io::BufReader::new(stdout).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                let ev: Value = serde_json::from_str(&l).unwrap_or(json!({"event": "log", "message": l}));
                me.update(id, |j| match ev["event"].as_str() {
                    Some("start") => j.log.push(ev["what"].as_str().unwrap_or("").to_string()),
                    Some("log") => j.log.push(ev["message"].as_str().unwrap_or("").to_string()),
                    Some("done") => j.result = Some(ev.clone()),
                    Some("progress") => {
                        if let Some(m) = ev["message"].as_str().filter(|m| !m.is_empty()) {
                            j.log.push(format!("[{}/{}] {m}", ev["done"], ev["total"]));
                        }
                        j.progress = Some(json!({"done": ev["done"], "total": ev["total"], "message": ev["message"]}));
                    }
                    Some("error") => j.error = ev["message"].as_str().map(str::to_string),
                    _ => j.log.push(l.clone()),
                });
            }
        }
        let status = child.wait().await;
        let tail = errs.await.unwrap_or_default();
        self.update(id, |j| {
            j.finished_at = Some(now());
            j.pid = None;
            if j.state == "cancelled" {
                return;
            }
            match status {
                Ok(s) if s.success() && j.error.is_none() => j.state = "done",
                Ok(s) => {
                    j.state = "failed";
                    if j.error.is_none() {
                        // a Rust panic: its message, not the "run with RUST_BACKTRACE" note after it
                        let why = tail.iter().position(|l| l.contains("panicked at"))
                            .map(|i| format!("the job crashed: {}", tail.get(i + 1).map(|m| m.trim()).filter(|m| !m.is_empty()).unwrap_or(&tail[i])))
                            .or_else(|| tail.iter().rev().find(|l| !l.trim().is_empty() && !l.starts_with("note:")).cloned())
                            .unwrap_or_else(|| format!("the job ended with {s}"));
                        j.error = Some(if s.code().is_none() { format!("stopped by the system ({s}; out of memory?): {why}") } else { why });
                    }
                }
                Err(e) => { j.state = "failed"; j.error = Some(e.to_string()); }
            }
        });
        tracing::info!("workshop: job {id} finished: {}", self.job(id).map(|j| j.state).unwrap_or("?"));
    }

    pub fn job(&self, id: u64) -> Option<Job> {
        self.jobs.lock().unwrap().iter().find(|j| j.id == id).cloned()
    }

    pub fn list(&self) -> Vec<Job> {
        self.jobs.lock().unwrap().iter().rev().cloned().collect()
    }

    /// Cancel a queued or running job (the running command is stopped).
    pub fn cancel(&self, id: u64) -> Result<Job, String> {
        let mut jobs = self.jobs.lock().unwrap();
        let j = jobs.iter_mut().find(|j| j.id == id).ok_or("no such job")?;
        match j.state {
            "queued" | "running" => {
                if let Some(pid) = j.pid {
                    // the whole group: the command and whatever it started; SIGKILL if it is still there in 3 s
                    #[cfg(unix)]
                    {
                        unsafe { libc::kill(-(pid as i32), libc::SIGTERM); }
                        std::thread::spawn(move || {
                            std::thread::sleep(std::time::Duration::from_secs(3));
                            unsafe { libc::kill(-(pid as i32), libc::SIGKILL); }
                        });
                    }
                    #[cfg(not(unix))]
                    let _ = std::process::Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).status();
                }
                j.state = "cancelled";
                j.finished_at = Some(now());
                Ok(j.clone())
            }
            s => Err(format!("the job is already {s}")),
        }
    }

    /// The models and adapters inside the Workshop's folders (two levels deep), for its pickers.
    pub fn files(&self) -> Value {
        let mut gguf = vec![];
        let mut hf = vec![];
        let mut adapters = vec![];
        fn walk(d: &Path, depth: usize, gguf: &mut Vec<Value>, hf: &mut Vec<Value>, adapters: &mut Vec<Value>) {
            let Ok(rd) = std::fs::read_dir(d) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    if p.join("adapter_config.json").exists() {
                        adapters.push(json!(p.display().to_string()));
                    } else if p.join("config.json").exists() && std::fs::read_dir(&p).into_iter().flatten().flatten()
                        .any(|f| f.path().extension().and_then(|x| x.to_str()) == Some("safetensors")) {
                        hf.push(json!(p.display().to_string()));
                    } else if depth > 0 {
                        walk(&p, depth - 1, gguf, hf, adapters);
                    }
                } else if p.extension().and_then(|x| x.to_str()) == Some("gguf") {
                    let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    // shard 2+ of a split model is not a model of its own
                    if !n.contains("-of-") || n.contains("-00001-of-") {
                        gguf.push(json!({"path": p.display().to_string(), "bytes": e.metadata().map(|m| m.len()).unwrap_or(0)}));
                    }
                }
            }
        }
        for r in &self.roots {
            walk(r, 2, &mut gguf, &mut hf, &mut adapters);
        }
        let listing = |d: &Path, want: &dyn Fn(&Path) -> bool| -> Vec<Value> {
            let mut v: Vec<(u64, Value)> = std::fs::read_dir(d).into_iter().flatten().flatten().filter(|e| e.path().is_file() && want(&e.path()))
                .map(|e| {
                    let m = e.metadata().ok();
                    let t = m.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
                    (t, json!({"path": e.path().display().to_string(), "bytes": m.map(|m| m.len()).unwrap_or(0), "modified": t}))
                }).collect();
            v.sort_by(|a, b| b.0.cmp(&a.0));
            v.into_iter().map(|x| x.1).collect()
        };
        let datasets = listing(&self.out_dir.join("datasets"), &|p| p.extension().and_then(|x| x.to_str()) == Some("jsonl"));
        let uploads = listing(&self.uploads_dir(), &|_| true);
        json!({"gguf": gguf, "hf_models": hf, "adapters": adapters, "datasets": datasets, "uploads": uploads, "folders": self.roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>(),
               "output_folder": self.out_dir.display().to_string(), "installed": exe().is_some()})
    }
}

/// Only the server's own key: the Workshop reads and writes model files.
fn admin(s: &AppState, headers: &HeaderMap) -> Result<Arc<Workshop>, Response> {
    // a client profile's key also passes `authorized`, so it is turned away here by name
    let valid = authorized(s, headers);
    let client_profile = crate::clients::given_key(headers).is_some_and(|k| s.clients.by_key(k).is_some());
    if !valid || client_profile {
        return Err(error(StatusCode::UNAUTHORIZED, "authentication_error", "the Workshop needs the server's own API key"));
    }
    get().ok_or_else(|| error(StatusCode::SERVICE_UNAVAILABLE, "server_error", "the server is still starting"))
}

pub async fn jobs_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("Workshop jobs can be started");
    }
    match w.submit(req) {
        Ok(j) => (StatusCode::ACCEPTED, Json(j.view(false))).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    }
}

pub async fn jobs_get(State(s): State<AppState>, headers: HeaderMap) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    Json(json!({"jobs": w.list().iter().map(|j| j.view(false)).collect::<Vec<_>>()})).into_response()
}

/// A job id from the URL, read only after the key was checked (so a stranger learns nothing from the route).
fn job_id(id: &str) -> Result<u64, Response> {
    id.parse().map_err(|_| error(StatusCode::NOT_FOUND, "not_found", "no such job"))
}

pub async fn job_get(State(s): State<AppState>, headers: HeaderMap, UrlPath(id): UrlPath<String>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    let id = match job_id(&id) { Ok(i) => i, Err(r) => return r };
    match w.job(id) {
        Some(j) => Json(j.view(true)).into_response(),
        None => error(StatusCode::NOT_FOUND, "not_found", "no such job"),
    }
}

pub async fn job_delete(State(s): State<AppState>, headers: HeaderMap, UrlPath(id): UrlPath<String>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("Workshop jobs can be cancelled");
    }
    let id = match job_id(&id) { Ok(i) => i, Err(r) => return r };
    match w.cancel(id) {
        Ok(j) => Json(j.view(false)).into_response(),
        Err(e) => error(StatusCode::CONFLICT, "invalid_request_error", e),
    }
}

pub async fn files_get(State(s): State<AppState>, headers: HeaderMap) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    Json(w.files()).into_response()
}

/// A file name made safe: letters, digits, `.`, `-` and `_` only, no leading dot, at most 120 characters.
fn safe_name(n: &str) -> Option<String> {
    let base = n.rsplit(['/', '\\']).next().unwrap_or("");
    let s: String = base.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect();
    let s = s.trim_start_matches('.').chars().take(120).collect::<String>();
    (!s.is_empty() && s.chars().any(|c| c.is_ascii_alphanumeric())).then_some(s)
}

/// `POST /v1/workshop/uploads?name=…`: the body is the file; it is saved into the uploads folder (a new name when taken).
pub async fn upload_post(State(s): State<AppState>, headers: HeaderMap, axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
                         body: axum::body::Bytes) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("files can be uploaded to the Workshop");
    }
    let Some(name) = q.get("name").and_then(|n| safe_name(n)) else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "give the file's name: ?name=notes.pdf");
    };
    if body.is_empty() {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "the file is empty");
    }
    let (stem, ext) = match name.rfind('.') {
        Some(i) if i > 0 => (name[..i].to_string(), name[i..].to_string()),
        _ => (name.clone(), String::new()),
    };
    let path = Workshop::fresh(&w.uploads_dir(), &stem, &ext);
    match tokio::fs::write(&path, &body).await {
        Ok(()) => Json(json!({"path": path.display().to_string(), "bytes": body.len()})).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("{}: {e}", path.display())),
    }
}

impl Workshop {
    /// A dataset the Workshop may read: a .jsonl inside its folders.
    fn dataset_path(&self, p: &str) -> Result<PathBuf, String> {
        if !p.ends_with(".jsonl") {
            return Err("a dataset is a .jsonl file".into());
        }
        self.allowed(p, true)
    }

    /// True while a job is writing `p`.
    fn busy(&self, p: &Path) -> bool {
        self.jobs.lock().unwrap().iter().any(|j| matches!(j.state, "queued" | "running") && j.output == p)
    }
}

/// `GET /v1/workshop/dataset?path=…&offset=0&limit=50`: a page of records, the total, and the dataset's card.
pub async fn dataset_get(State(s): State<AppState>, headers: HeaderMap,
                         axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    let path = match w.dataset_path(q.get("path").map(String::as_str).unwrap_or("")) {
        Ok(p) => p,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    let offset: usize = q.get("offset").and_then(|v| v.parse().ok()).unwrap_or(0);
    let limit: usize = q.get("limit").and_then(|v| v.parse().ok()).unwrap_or(50).clamp(1, 500);
    let text = match tokio::fs::read_to_string(&path).await {
        Ok(t) => t,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("{}: {e}", path.display())),
    };
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let records: Vec<Value> = lines.iter().enumerate().skip(offset).take(limit)
        .map(|(i, l)| json!({"i": i, "record": serde_json::from_str::<Value>(l).unwrap_or_else(|_| json!({"unparsable": l}))}))
        .collect();
    // the card beside it, as strata-workshop's datagen names it (`x.jsonl` → `x.card.md`)
    let card_path = path.with_file_name(format!("{}.card.md", path.file_stem().and_then(|x| x.to_str()).unwrap_or("dataset")));
    let card = tokio::fs::read_to_string(&card_path).await.ok();
    Json(json!({"path": path.display().to_string(), "total": lines.len(), "offset": offset, "records": records, "card": card,
                "writing": w.busy(&path)})).into_response()
}

/// `POST /v1/workshop/dataset` `{"path", "remove": [i, …], "replace": {"i": record, …}}`: edits the dataset in place
/// (written to a new file, then moved over the old one).
pub async fn dataset_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("datasets can be edited");
    }
    let bad = |m: String| error(StatusCode::BAD_REQUEST, "invalid_request_error", m);
    let path = match w.dataset_path(req["path"].as_str().unwrap_or("")) { Ok(p) => p, Err(e) => return bad(e) };
    if w.busy(&path) {
        return error(StatusCode::CONFLICT, "invalid_request_error", "a job is still writing this dataset");
    }
    let text = match tokio::fs::read_to_string(&path).await { Ok(t) => t, Err(e) => return bad(format!("{}: {e}", path.display())) };
    let mut lines: Vec<Option<String>> = text.lines().filter(|l| !l.trim().is_empty()).map(|l| Some(l.to_string())).collect();
    if let Some(m) = req["replace"].as_object() {
        for (k, rec) in m {
            let Some(i) = k.parse::<usize>().ok().filter(|i| *i < lines.len()) else { return bad(format!("no record {k}")) };
            let ok = rec["messages"].as_array().is_some_and(|a| !a.is_empty() && a.iter().all(|m| m["role"].is_string() && m["content"].is_string()));
            if !ok {
                return bad(format!("record {k}: a record is {{\"messages\": [{{\"role\", \"content\"}}, …]}}"));
            }
            lines[i] = Some(rec.to_string());
        }
    }
    for v in req["remove"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        let Some(i) = v.as_u64().map(|i| i as usize).filter(|i| *i < lines.len()) else { return bad(format!("no record {v}")) };
        lines[i] = None;
    }
    let kept: Vec<String> = lines.into_iter().flatten().collect();
    let tmp = path.with_extension("jsonl.tmp");
    let body = if kept.is_empty() { String::new() } else { kept.join("\n") + "\n" };
    if let Err(e) = tokio::fs::write(&tmp, body).await {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string());
    }
    if let Err(e) = tokio::fs::rename(&tmp, &path).await {
        tokio::fs::remove_file(&tmp).await.ok();
        return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string());
    }
    Json(json!({"path": path.display().to_string(), "total": kept.len()})).into_response()
}

#[cfg(test)]
mod tests {
    use super::safe_name;

    #[test]
    fn upload_names_are_made_safe() {
        assert_eq!(safe_name("notes.pdf").as_deref(), Some("notes.pdf"));
        assert_eq!(safe_name("../../etc/passwd").as_deref(), Some("passwd"));
        assert_eq!(safe_name("C:\\x\\my file (1).md").as_deref(), Some("my_file__1_.md"));
        assert_eq!(safe_name(".bashrc").as_deref(), Some("bashrc"));
        assert_eq!(safe_name("..."), None);
        assert_eq!(safe_name(""), None);
    }
}
