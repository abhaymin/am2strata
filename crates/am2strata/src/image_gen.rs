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
    /// the image model jobs use: the run config's, or the one chosen in Settings > Models (kept in
    /// <data>/media/am2strata.json)
    model: std::sync::RwLock<String>,
    /// the media download in progress or last finished: {"state": "idle" | "pulling" | "done" | "error", ...}
    pub pull_job: std::sync::Mutex<Value>,
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
        let chosen = std::fs::read_to_string(data_dir.join("media/am2strata.json")).ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .and_then(|v| v["image_model"].as_str().map(str::to_string));
        Media {
            exe: find_exe(),
            model: std::sync::RwLock::new(chosen.or_else(|| cfg["model"].as_str().map(str::to_string)).unwrap_or_else(|| DEFAULT_MODEL.to_string())),
            pull_job: std::sync::Mutex::new(json!({"state": "idle"})),
            keep_text_model: cfg["gpu_policy"].as_str() == Some("keep"),
            mold_home: data_dir.join("media"),
            proc: Mutex::new(None),
        }
    }

    pub fn available(&self) -> bool {
        self.exe.is_some()
    }

    pub fn model(&self) -> String {
        self.model.read().unwrap().clone()
    }

    fn set_model(&self, m: &str) -> anyhow::Result<()> {
        std::fs::create_dir_all(&self.mold_home)?;
        std::fs::write(self.mold_home.join("am2strata.json"), serde_json::to_string_pretty(&json!({"image_model": m}))?)?;
        *self.model.write().unwrap() = m.to_string();
        Ok(())
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
                "done" | "loaded" | "pulled" | "unloaded" | "models" => return Ok(v),
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
    let model = req["model"].as_str().filter(|m| m.contains(':')).map(str::to_string).unwrap_or_else(|| s.media.model());
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

/// Speech models planned for v2.3 (no engine yet): listed so that every kind is visible in Settings > Models.
const AUDIO_PLANNED: &[(&str, &str, &str, f64)] = &[
    ("whisper-large-v3-turbo", "Whisper large-v3 turbo", "speech to text (99 languages)", 1.6),
    ("parakeet-tdt-0.6b-v3", "Parakeet TDT 0.6B v3", "speech to text (25 European languages)", 1.2),
    ("qwen3-tts-1.7b", "Qwen3-TTS 1.7B", "text to speech, voice cloning", 3.8),
    ("kokoro-82m", "Kokoro 82M", "text to speech, small and fast", 0.3),
];

/// GET /api/media/models: the media engine's catalog (image, video, 3D) with what is downloaded, the image model in
/// use, the download in progress, and the speech models still to come.
pub async fn models(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let audio: Vec<Value> = AUDIO_PLANNED.iter().map(|(n, d, what, gb)| json!({"name": n, "display_name": d, "kind": "audio",
        "description": what, "size_gb": gb, "downloaded": false, "planned": "v2.3"})).collect();
    let pull = s.media.pull_job.lock().unwrap().clone();
    let base = json!({"engine": s.media.available(), "image_model": s.media.model(), "pull": pull});
    if !s.media.available() {
        let mut v = base;
        v["models"] = json!(audio);
        v["note"] = json!("the media engine is not built: cargo build --release --manifest-path crates/strata-media/Cargo.toml --features cuda");
        return Json(v).into_response();
    }
    match s.media.call(json!({"op": "list"}), |_| {}).await {
        Ok(r) => {
            let mut all = r["models"].as_array().cloned().unwrap_or_default();
            all.extend(audio);
            let mut v = base;
            v["models"] = json!(all);
            Json(v).into_response()
        }
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "media_error", format!("{e:#}")),
    }
}

/// POST /api/media/models/pull {"model": "<catalog name>"}: downloads it in the background (GET .../pull for progress).
pub async fn pull(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if !s.media.available() {
        return error(StatusCode::NOT_IMPLEMENTED, "not_available", "the media engine is not built");
    }
    let Some(model) = req["model"].as_str().filter(|m| m.contains(':')).map(str::to_string) else {
        return bad("send {\"model\": \"<name>:<variant>\"} from GET /api/media/models");
    };
    {
        let mut job = s.media.pull_job.lock().unwrap();
        if job["state"] == "pulling" {
            return error(StatusCode::CONFLICT, "busy", format!("{} is downloading; one download at a time", job["model"].as_str().unwrap_or("")));
        }
        *job = json!({"state": "pulling", "model": model, "started": chrono_now()});
    }
    let media = s.media.clone();
    tokio::spawn(async move {
        let r = media.call(json!({"op": "pull", "model": model}), |_| {}).await;
        *media.pull_job.lock().unwrap() = match r {
            Ok(_) => json!({"state": "done", "model": model}),
            Err(e) => json!({"state": "error", "model": model, "error": format!("{e:#}")}),
        };
    });
    (StatusCode::ACCEPTED, Json(json!({"state": "pulling"}))).into_response()
}

pub async fn pull_status(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    Json(s.media.pull_job.lock().unwrap().clone()).into_response()
}

/// POST /api/media/models/default {"model": ...}: the image model for the next jobs (a downloaded image model).
pub async fn set_default(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let Some(model) = req["model"].as_str() else { return bad("send {\"model\": ...}") };
    if s.media.available() {
        let list = match s.media.call(json!({"op": "list"}), |_| {}).await {
            Ok(r) => r,
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "media_error", format!("{e:#}")),
        };
        let found = list["models"].as_array().and_then(|a| a.iter().find(|m| m["name"] == model).cloned());
        match found {
            None => return bad(format!("{model} is not in the media catalog")),
            Some(m) if m["kind"] != "image" => return bad(format!("{model} is a {} model; image jobs need an image model", m["kind"].as_str().unwrap_or("?"))),
            Some(m) if m["downloaded"] != true => return bad(format!("{model} is not downloaded yet")),
            _ => {}
        }
    }
    match s.media.set_model(model) {
        Ok(()) => Json(json!({"image_model": model})).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "io_error", format!("{e:#}")),
    }
}

fn chrono_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}
