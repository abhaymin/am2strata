//! Starts a PolyStrata engine (`strata` or `strata-poly`) from a run config (`strata-<model>.json`) and drives it
//! over the line protocol, one request at a time, as serve/server.py's StrataEngine does.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use strata_protocol::{Done, Event, Gen, Sampling};
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
    #[serde(flatten)]
    pub rest: BTreeMap<String, serde_json::Value>,
}

impl RunConfig {
    pub fn load(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Config { path: path.display().to_string(), source: Box::new(e) })?;
        serde_json::from_str(&text).map_err(|e| Error::Config { path: path.display().to_string(), source: Box::new(e) })
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
        let gpus = cfg.gpus();
        if !gpus.is_empty() {
            cmd.env("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
            cmd.env("CUDA_VISIBLE_DEVICES", gpus.iter().map(u64::to_string).collect::<Vec<_>>().join(","));
        }
        for (k, v) in &cfg.env {
            cmd.env(k, v);
        }
        let dirs: Vec<String> = cfg.lib_dirs.iter().filter(|d| d.is_dir()).map(|d| d.display().to_string()).collect();
        if !dirs.is_empty() {
            let mut path = dirs.join(":");
            if let Ok(old) = std::env::var("LD_LIBRARY_PATH") {
                if !old.is_empty() {
                    path = format!("{path}:{old}");
                }
            }
            cmd.env("LD_LIBRARY_PATH", path);
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
    pub async fn generate(&self, max_new: u32, sampling: Sampling, turn_at: Option<usize>, ids: Vec<u32>)
        -> mpsc::Receiver<Result<GenEvent>> {
        let (out, rx) = mpsc::channel(256);
        let inner = self.inner.clone();
        tokio::spawn(async move {
            let _turn = inner.turn.lock().await;
            let line = Gen { max_new, sampling: &sampling, turn_at, ids: &ids }.line();
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
