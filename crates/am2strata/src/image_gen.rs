//! Image generation (v2.0): POST /v1/images/generations, OpenAI's images API, drawn by `am2strata-media` (Rust,
//! candle; crates/strata-media) over JSON lines. One GPU job at a time: the job holds the manager's GPU turn and, by
//! default, stops the text model first (it loads again with the next chat request); after the job the media engine
//! frees its VRAM. The run config's `"images": {"model": ..., "gpu_policy": "unload" | "keep"}` changes either.

use crate::{authorized, error, AppState};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;

pub const DEFAULT_MODEL: &str = "z-image-turbo:q8";

struct Proc {
    child: Child,
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
}

pub struct Media {
    pub exe: Option<PathBuf>,
    pub model: String,
    pub keep_text_model: bool,
    mold_home: PathBuf,
    proc: Mutex<Option<Proc>>,
}

/// The media engine: $AM2STRATA_MEDIA_EXE, else crates/strata-media's release build beside this binary's checkout.
pub fn find_exe() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("AM2STRATA_MEDIA_EXE").map(PathBuf::from) {
        return p.is_file().then_some(p);
    }
    let exe = std::env::current_exe().ok()?;
    exe.ancestors().skip(1).take(5).map(|d| d.join("crates/strata-media/target/release/am2strata-media")).find(|p| p.is_file())
}

impl Media {
    pub fn new(data_dir: &Path, rest: &std::collections::BTreeMap<String, Value>) -> Media {
        let cfg = rest.get("images").cloned().unwrap_or(Value::Null);
        Media {
            exe: find_exe(),
            model: cfg["model"].as_str().unwrap_or(DEFAULT_MODEL).to_string(),
            keep_text_model: cfg["gpu_policy"].as_str() == Some("keep"),
            mold_home: data_dir.join("media"),
            proc: Mutex::new(None),
        }
    }

    pub fn available(&self) -> bool {
        self.exe.is_some()
    }

    async fn spawn(&self) -> anyhow::Result<Proc> {
        let exe = self.exe.as_ref().ok_or_else(|| anyhow::anyhow!("the media engine is not built"))?;
        std::fs::create_dir_all(&self.mold_home)?;
        let mut child = tokio::process::Command::new(exe)
            .env("MOLD_HOME", &self.mold_home)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()?;
        let stdin = child.stdin.take().unwrap();
        let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
        let first = lines.next_line().await?.ok_or_else(|| anyhow::anyhow!("the media engine exited at start"))?;
        let v: Value = serde_json::from_str(&first)?;
        if v["event"] != "ready" {
            anyhow::bail!("the media engine said {first}");
        }
        tracing::info!("media engine started ({}): {first}", exe.display());
        Ok(Proc { child, stdin, lines })
    }

    /// One command; `on_event` sees every event until the one that ends it (done / loaded / pulled / unloaded / error).
    pub async fn call(&self, cmd: Value, mut on_event: impl FnMut(&Value)) -> anyhow::Result<Value> {
        let mut guard = self.proc.lock().await;
        if guard.as_mut().is_none_or(|p| p.child.try_wait().ok().flatten().is_some()) {
            *guard = Some(self.spawn().await?);
        }
        let p = guard.as_mut().unwrap();
        p.stdin.write_all(format!("{cmd}\n").as_bytes()).await?;
        p.stdin.flush().await?;
        loop {
            let Some(line) = p.lines.next_line().await? else {
                *guard = None;
                anyhow::bail!("the media engine stopped unexpectedly");
            };
            let v: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            match v["event"].as_str().unwrap_or("") {
                "error" => anyhow::bail!("{}", v["message"].as_str().unwrap_or("the media engine failed")),
                "done" | "loaded" | "pulled" | "unloaded" => return Ok(v),
                _ => on_event(&v),
            }
        }
    }
}

fn bad(msg: impl Into<String>) -> Response {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
}

pub async fn generations(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if !s.media.available() {
        return error(StatusCode::NOT_IMPLEMENTED, "not_available",
                     "image generation is not built: cargo build --release --manifest-path crates/strata-media/Cargo.toml --features cuda");
    }
    let Some(prompt) = req["prompt"].as_str().filter(|p| !p.trim().is_empty()) else {
        return bad("send {\"prompt\": \"...\"}");
    };
    let size = req["size"].as_str().filter(|z| *z != "auto").unwrap_or("1024x1024");
    let Some((w, h)) = size.split_once('x').and_then(|(a, b)| Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?))) else {
        return bad(format!("size must be WIDTHxHEIGHT, not {size:?}"));
    };
    if !(256..=2048).contains(&w) || !(256..=2048).contains(&h) || w % 16 != 0 || h % 16 != 0 {
        return bad("each side of size must be a multiple of 16 from 256 to 2048");
    }
    let n = req["n"].as_u64().unwrap_or(1);
    if !(1..=4).contains(&n) {
        return bad("n must be 1 to 4");
    }
    if req["response_format"].as_str().is_some_and(|f| f != "b64_json") {
        return bad("response_format: only b64_json for now");
    }
    let steps = req["steps"].as_u64().unwrap_or(8).clamp(1, 100);
    let seed = req["seed"].as_u64().unwrap_or_else(|| (crate::now() ^ (std::process::id() as u64) << 20) % 4_294_967_295);
    let model = req["model"].as_str().filter(|m| m.contains(':')).unwrap_or(&s.media.model).to_string();
    let t0 = Instant::now();
    let _gpu = match s.mgr.gpu_exclusive(!s.media.keep_text_model).await {
        Ok(g) => g,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, "busy", format!("{e:#}")),
    };
    let mut data = Vec::new();
    let mut failure = None;
    for i in 0..n {
        let r = json!({"model": model, "prompt": prompt, "width": w, "height": h, "steps": steps, "guidance": 0.0,
                       "seed": seed + i, "batch_size": 1});
        let mut png = None;
        let res = s.media.call(json!({"op": "generate", "request": r}), |ev| {
            if ev["event"] == "image" && png.is_none() {
                png = ev["png"].as_str().map(str::to_owned);
            }
        }).await;
        match (res, png) {
            (Ok(_), Some(b64)) => data.push(json!({"b64_json": b64, "seed": seed + i, "revised_prompt": prompt})),
            (Ok(_), None) => {
                failure = Some("the media engine returned no image".to_string());
                break;
            }
            (Err(e), _) => {
                failure = Some(format!("{e:#}"));
                break;
            }
        }
    }
    // the VRAM goes back to the text model (it loads again with the next chat request)
    if let Err(e) = s.media.call(json!({"op": "unload"}), |_| {}).await {
        tracing::warn!("the media engine did not unload: {e:#}");
    }
    if let Some(f) = failure {
        return error(StatusCode::BAD_GATEWAY, "image_error", f);
    }
    Json(json!({"created": crate::now(), "data": data, "size": format!("{w}x{h}"),
                "strata": {"seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0, "steps": steps, "model": model,
                           "engine": "am2strata-media (Rust, candle)"}})).into_response()
}
