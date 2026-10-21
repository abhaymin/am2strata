//! Starts a PolyStrata engine (`strata` or `strata-poly`) from a run config (`strata-<model>.json`) and drives it
//! over the line protocol, one request at a time, as serve/server.py's StrataEngine does.

pub mod hardware;
pub mod vision;

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use strata_protocol::{Done, Event, Gen, Images, Sampling};
pub use vision::{Vision, VisionConfig};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, Mutex};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reading the run config {path}: {source}")]
    Config { path: String, source: Box<dyn std::error::Error + Send + Sync> },
    #[error("starting the engine {exe}: {source}")]
    Spawn { exe: String, source: std::io::Error },
    #[error("the engine stopped unexpectedly")]
    Died,
    #[error("the engine did not become ready within {0:?}")]
    StartTimeout(Duration),
    #[error("the engine refused the request: {0}")]
    Refused(String),
    #[error("engine I/O: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// The fields of a PolyStrata run config that the server and the engine start need. Unknown keys are kept in
/// `rest` so nothing a config carries is lost.
#[derive(Debug, Clone, Deserialize)]
pub struct RunConfig {
    pub exe: PathBuf,
    #[serde(default)]
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub tokenizer: PathBuf,
    pub model_name: String,
    #[serde(default)]
    pub display_name: Option<String>,
    pub log: Option<PathBuf>,
    #[serde(default)]
    pub lib_dirs: Vec<PathBuf>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub gpu: Option<serde_json::Value>,
    pub port: Option<u16>,
    pub host: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub fit_max_tokens: bool,
    #[serde(default)]
    pub vision: Option<VisionConfig>,
    /// the model family for the model card's sampling ("qwen4exp", "qwen35moe"); read from the GGUF when absent
    #[serde(default)]
    pub arch: Option<String>,
    #[serde(flatten)]
    pub rest: BTreeMap<String, serde_json::Value>,
}

/// `general.architecture` from a GGUF file's header (it is among the first keys).
pub fn gguf_architecture(path: &std::path::Path) -> Option<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf = vec![0u8; 1 << 20];
    let n = f.read(&mut buf).ok()?;
    let b = &buf[..n];
    if b.get(..4)? != b"GGUF" {
        return None;
    }
    let rd_u64 = |o: usize| -> Option<u64> { Some(u64::from_le_bytes(b.get(o..o + 8)?.try_into().ok()?)) };
    let rd_u32 = |o: usize| -> Option<u32> { Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?)) };
    let n_kv = rd_u64(16)?;
    let mut o = 24;
    // value sizes of the fixed-width GGUF types, by type id
    let fixed = |t: u32| -> Option<usize> { Some(match t { 0 | 1 | 7 => 1, 2 | 3 => 2, 4 | 5 | 6 => 4, 10 | 11 | 12 => 8, _ => return None }) };
    for _ in 0..n_kv.min(4096) {
        let klen = rd_u64(o)? as usize;
        let key = std::str::from_utf8(b.get(o + 8..o + 8 + klen)?).ok()?;
        o += 8 + klen;
        let t = rd_u32(o)?;
        o += 4;
        if t == 8 {
            let vlen = rd_u64(o)? as usize;
            let v = b.get(o + 8..o + 8 + vlen)?;
            if key == "general.architecture" {
                return std::str::from_utf8(v).ok().map(str::to_owned);
            }
            o += 8 + vlen;
        } else if t == 9 {
            let et = rd_u32(o)?;
            let cnt = rd_u64(o + 4)? as usize;
            o += 12;
            if et == 8 {
                for _ in 0..cnt {
                    o += 8 + rd_u64(o)? as usize;
                }
            } else {
                o += cnt * fixed(et)?;
            }
        } else {
            o += fixed(t)?;
        }
    }
    None
}

/// The engine's environment: GPU selection, the config's `env`, and its library directories first on the path.
pub fn child_env(cfg: &RunConfig) -> Vec<(String, String)> {
    let mut env = Vec::new();
    let gpus = cfg.gpus();
    if !gpus.is_empty() {
        env.push(("CUDA_DEVICE_ORDER".into(), "PCI_BUS_ID".into()));
        env.push(("CUDA_VISIBLE_DEVICES".into(), gpus.iter().map(u64::to_string).collect::<Vec<_>>().join(",")));
    }
    for (k, v) in &cfg.env {
        env.push((k.clone(), v.clone()));
    }
    let dirs: Vec<String> = cfg.lib_dirs.iter().filter(|d| d.is_dir()).map(|d| d.display().to_string()).collect();
    if !dirs.is_empty() {
        let mut path = dirs.join(":");
        if let Ok(old) = std::env::var("LD_LIBRARY_PATH") {
            if !old.is_empty() {
                path = format!("{path}:{old}");
            }
        }
        env.push(("LD_LIBRARY_PATH".into(), path));
    }
    env
}

/// A shareable config: in every string (engine arguments too), a leading `~/` is the home folder and `${NAME}` an
/// environment variable. `${STRATA_HOME}` (PolyStrata's folder: engines, run configs), `${STRATA_DATA}` (models,
/// packs) and `${AM2STRATA_HOME}` (this checkout: the Rust engines) default to ~/Development/PolyStrata,
/// ~/Development/Strata-data and ~/Development/am2strata. Absolute paths are left as they are.
pub fn expand_str(s: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    let var = |name: &str| -> Option<String> {
        std::env::var(name).ok().filter(|v| !v.is_empty()).or_else(|| match name {
            "STRATA_HOME" => Some(format!("{home}/Development/PolyStrata")),
            "STRATA_DATA" => Some(format!("{home}/Development/Strata-data")),
            "AM2STRATA_HOME" => Some(format!("{home}/Development/am2strata")),
            "HOME" => Some(home.clone()),
            _ => None,
        })
    };
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find("${") {
        out.push_str(&rest[..i]);
        match rest[i + 2..].find('}') {
            Some(j) => {
                let name = &rest[i + 2..i + 2 + j];
                match var(name) {
                    Some(v) => out.push_str(&v),
                    None => out.push_str(&rest[i..i + 3 + j]),
                }
                rest = &rest[i + 3 + j..];
            }
            None => {
                out.push_str(&rest[i..]);
                rest = "";
            }
        }
    }
    out.push_str(rest);
    match out.strip_prefix("~/") {
        Some(r) if !home.is_empty() => format!("{home}/{r}"),
        _ => out,
    }
}

fn expand_paths(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::String(s) if s.contains("${") || s.starts_with("~/") => *s = expand_str(s),
        serde_json::Value::Array(a) => a.iter_mut().for_each(expand_paths),
        serde_json::Value::Object(o) => {
            for (k, x) in o.iter_mut() {
                if k != "api_key" {
                    expand_paths(x);
                }
            }
        }
        _ => {}
    }
}

impl RunConfig {
    pub fn load(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Config { path: path.display().to_string(), source: Box::new(e) })?;
        let mut v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| Error::Config { path: path.display().to_string(), source: Box::new(e) })?;
        expand_paths(&mut v);
        serde_json::from_value(v).map_err(|e| Error::Config { path: path.display().to_string(), source: Box::new(e) })
    }

    /// `--max-context N` from the engine arguments (the engine's own default is 8192).
    pub fn max_context(&self) -> usize {
        self.args
            .iter()
            .position(|a| a == "--max-context")
            .and_then(|i| self.args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192)
    }

    /// The model file: `--model` (strata-poly) or `--native` (strata).
    pub fn model_file(&self) -> Option<PathBuf> {
        ["--model", "--native"].iter().find_map(|flag| {
            self.args.iter().position(|a| a == flag).and_then(|i| self.args.get(i + 1)).map(PathBuf::from)
        })
    }

    /// The model family: the config's `arch`, else the GGUF's `general.architecture`.
    pub fn family(&self) -> Option<String> {
        self.arch.clone().or_else(|| self.model_file().and_then(|p| gguf_architecture(&p)))
    }

    /// The first GPU the config names (nvidia-smi numbering).
    pub fn first_gpu(&self) -> Option<u32> {
        self.gpus().first().map(|g| *g as u32)
    }

    /// The GPUs to show the engine, numbered as nvidia-smi numbers them.
    fn gpus(&self) -> Vec<u64> {
        match &self.gpu {
            Some(serde_json::Value::Number(n)) => n.as_u64().into_iter().collect(),
            Some(serde_json::Value::Array(a)) => a.iter().filter_map(|v| v.as_u64()).collect(),
            _ => Vec::new(),
        }
    }
}

/// What a running request yields.
#[derive(Debug, Clone)]
pub enum GenEvent {
    Token(u32),
    Progress { done: u64, total: u64 },
    Done(Done),
}

struct Inner {
    stdin: Mutex<ChildStdin>,
    events: Mutex<mpsc::UnboundedReceiver<Option<Event>>>,
    /// one request at a time
    turn: Mutex<()>,
}

pub struct Engine {
    inner: Arc<Inner>,
    child: Mutex<Child>,
    pub info: Vec<String>,
}

impl Engine {
    /// Start the engine and wait for READY. A resident-RAM load can take minutes, hence the long default.
    pub async fn start(cfg: &RunConfig, ready_timeout: Duration) -> Result<Self> {
        let mut cmd = Command::new(&cfg.exe);
        cmd.arg("--serve").args(&cfg.args).stdin(Stdio::piped()).stdout(Stdio::piped()).kill_on_drop(true);
        if let Some(cwd) = &cfg.cwd {
            cmd.current_dir(cwd);
        }
        for (k, v) in child_env(cfg) {
            cmd.env(k, v);
        }
        match &cfg.log {
            Some(log) => {
                let f = std::fs::OpenOptions::new().create(true).append(true).open(log)?;
                cmd.stderr(Stdio::from(f));
            }
            None => {
                cmd.stderr(Stdio::null());
            }
        }
        let mut child = cmd.spawn().map_err(|e| Error::Spawn { exe: cfg.exe.display().to_string(), source: e })?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");

        let (tx, mut rx) = mpsc::unbounded_channel::<Option<Event>>();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if tx.send(Some(Event::parse(&line))).is_err() {
                    return;
                }
            }
            let _ = tx.send(None); // the engine's output closed: it died or quit
        });

        let mut info = Vec::new();
        let ready = async {
            loop {
                match rx.recv().await {
                    Some(Some(Event::Ready(_))) => return Ok(()),
                    Some(Some(Event::Info(s))) => info.push(s),
                    Some(Some(other)) => tracing::debug!(?other, "engine start-up line"),
                    Some(None) | None => return Err(Error::Died),
                }
            }
        };
        tokio::time::timeout(ready_timeout, ready).await.map_err(|_| Error::StartTimeout(ready_timeout))??;
        tracing::info!(?info, "engine ready");
        Ok(Self {
            inner: Arc::new(Inner { stdin: Mutex::new(stdin), events: Mutex::new(rx), turn: Mutex::new(()) }),
            child: Mutex::new(child),
            info,
        })
    }

    /// Run one request. Events arrive on the returned channel; dropping the receiver stops the engine (STOP) and
    /// drains it to DONE, so the next request starts clean.
    pub async fn generate(&self, max_new: u32, sampling: Sampling, turn_at: Option<usize>, ids: Vec<u32>,
                          images: Option<Images>) -> mpsc::Receiver<Result<GenEvent>> {
        let (out, rx) = mpsc::channel(256);
        let inner = self.inner.clone();
        tokio::spawn(async move {
            let _turn = inner.turn.lock().await;
            let line = Gen { max_new, sampling: &sampling, turn_at, ids: &ids, images: images.as_ref() }.line();
            {
                let mut stdin = inner.stdin.lock().await;
                if stdin.write_all(line.as_bytes()).await.is_err() || stdin.flush().await.is_err() {
                    let _ = out.send(Err(Error::Died)).await;
                    return;
                }
            }
            let mut events = inner.events.lock().await;
            let mut stopped = false;
            loop {
                let ev = match events.recv().await {
                    Some(Some(ev)) => ev,
                    Some(None) | None => {
                        let _ = out.send(Err(Error::Died)).await;
                        return;
                    }
                };
                let forward = match ev {
                    Event::Token(t) => Some(GenEvent::Token(t)),
                    Event::Progress { done, total, .. } => Some(GenEvent::Progress { done, total }),
                    Event::Done(d) => {
                        if !stopped {
                            let _ = out.send(Ok(GenEvent::Done(d))).await;
                        }
                        return;
                    }
                    Event::Err(msg) => {
                        if !stopped {
                            let _ = out.send(Err(Error::Refused(msg))).await;
                        }
                        return;
                    }
                    _ => None,
                };
                if let Some(ev) = forward {
                    if !stopped && out.send(Ok(ev)).await.is_err() {
                        // the client went away: stop the engine and keep reading until DONE
                        stopped = true;
                        let mut stdin = inner.stdin.lock().await;
                        let _ = stdin.write_all(strata_protocol::STOP.as_bytes()).await;
                        let _ = stdin.flush().await;
                    }
                }
            }
        });
        rx
    }

    /// Ask the engine to exit, then make sure it has.
    pub async fn shutdown(&self) {
        {
            let mut stdin = self.inner.stdin.lock().await;
            let _ = stdin.write_all(strata_protocol::QUIT.as_bytes()).await;
            let _ = stdin.flush().await;
        }
        let mut child = self.child.lock().await;
        if tokio::time::timeout(Duration::from_secs(30), child.wait()).await.is_err() {
            let _ = child.kill().await;
        }
    }
}

#[cfg(test)]
mod expand_tests {
    use super::expand_str;

    #[test]
    fn shareable_paths_expand() {
        let home = std::env::var("HOME").unwrap();
        assert_eq!(expand_str("~/x/y"), format!("{home}/x/y"));
        assert_eq!(expand_str("/abs/path"), "/abs/path");
        assert_eq!(expand_str("${HOME}/a"), format!("{home}/a"));
        assert_eq!(expand_str("${NO_SUCH_VAR_AM2}/a"), "${NO_SUCH_VAR_AM2}/a", "an unknown variable stays visible");
        if std::env::var("STRATA_DATA").is_err() {
            assert_eq!(expand_str("${STRATA_DATA}/models"), format!("{home}/Development/Strata-data/models"));
        }
        assert_eq!(expand_str("--kv"), "--kv");
    }
}
