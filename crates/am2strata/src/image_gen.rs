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
pub const DEFAULT_STT_MODEL: &str = "whisper-large-v3-turbo";
pub const DEFAULT_TTS_MODEL: &str = "parler-tts-mini-v1";
/// Wan 2.2 TI2V-5B (Apache-2.0): text or image to video, 16.9 GB at Q8
pub const DEFAULT_VIDEO_MODEL: &str = "wan22-ti2v-5b:q8";

struct Proc {
    child: Child,
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
}

pub struct Media {
    pub exe: Option<PathBuf>,
    /// the image and video models jobs use: the run config's, or the ones chosen in the web app (kept in
    /// <data>/media/am2strata.json)
    model: std::sync::RwLock<String>,
    video_model: std::sync::RwLock<String>,
    /// speech to text and text to speech (v2.3), and where they run: "auto" (the CPU when the RAM has room for the
    /// model, else the GPU with the text model stopped, as for an image), "cpu" or "cuda"
    stt_model: std::sync::RwLock<String>,
    tts_model: std::sync::RwLock<String>,
    pub audio_device: std::sync::RwLock<String>,
    /// the media download in progress or last finished: {"state": "idle" | "pulling" | "done" | "error", ...}
    pub pull_job: std::sync::Mutex<Value>,
    /// the video formats the running engine can encode (its ready event); empty before it started
    pub video_formats: std::sync::Mutex<Vec<String>>,
    /// the engine was built with CUDA (its ready event)
    pub cuda: std::sync::atomic::AtomicBool,
    /// the timer that frees the speech models after a while unused (audio_api.rs)
    pub audio_idle: std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
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
        let saved = std::fs::read_to_string(data_dir.join("media/am2strata.json")).ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok()).unwrap_or(Value::Null);
        let chosen = saved["image_model"].as_str().map(str::to_string);
        let video = saved["video_model"].as_str().map(str::to_string)
            .or_else(|| rest.get("videos").and_then(|v| v["model"].as_str()).map(str::to_string))
            .unwrap_or_else(|| DEFAULT_VIDEO_MODEL.to_string());
        Media {
            exe: find_exe(),
            model: std::sync::RwLock::new(chosen.or_else(|| cfg["model"].as_str().map(str::to_string)).unwrap_or_else(|| DEFAULT_MODEL.to_string())),
            video_model: std::sync::RwLock::new(video),
            stt_model: std::sync::RwLock::new(saved["stt_model"].as_str().unwrap_or(DEFAULT_STT_MODEL).to_string()),
            tts_model: std::sync::RwLock::new(saved["tts_model"].as_str().unwrap_or(DEFAULT_TTS_MODEL).to_string()),
            audio_device: std::sync::RwLock::new(saved["audio_device"].as_str().unwrap_or("auto").to_string()),
            pull_job: std::sync::Mutex::new(json!({"state": "idle"})),
            video_formats: std::sync::Mutex::new(Vec::new()),
            cuda: std::sync::atomic::AtomicBool::new(true),
            audio_idle: std::sync::Mutex::new(None),
            keep_text_model: cfg["gpu_policy"].as_str() == Some("keep"),
            mold_home: data_dir.join("media"),
            proc: Mutex::new(None),
        }
    }

    /// A command is running (a job): a quick look such as the capability check uses what it saw last instead.
    pub fn busy(&self) -> bool {
        self.proc.try_lock().is_err()
    }

    pub fn available(&self) -> bool {
        self.exe.is_some()
    }

    pub fn model(&self) -> String {
        self.model.read().unwrap().clone()
    }

    pub fn video_model(&self) -> String {
        self.video_model.read().unwrap().clone()
    }

    pub fn stt_model(&self) -> String {
        self.stt_model.read().unwrap().clone()
    }

    pub fn tts_model(&self) -> String {
        self.tts_model.read().unwrap().clone()
    }

    /// `kind`: "image", "video", "stt", "tts" (the model) or "audio_device" (where speech runs).
    pub fn set_model(&self, kind: &str, m: &str) -> anyhow::Result<()> {
        match kind {
            "video" => *self.video_model.write().unwrap() = m.to_string(),
            "stt" => *self.stt_model.write().unwrap() = m.to_string(),
            "tts" => *self.tts_model.write().unwrap() = m.to_string(),
            "audio_device" => *self.audio_device.write().unwrap() = m.to_string(),
            _ => *self.model.write().unwrap() = m.to_string(),
        }
        std::fs::create_dir_all(&self.mold_home)?;
        std::fs::write(self.mold_home.join("am2strata.json"), serde_json::to_string_pretty(&json!({
            "image_model": self.model(), "video_model": self.video_model(), "stt_model": self.stt_model(),
            "tts_model": self.tts_model(), "audio_device": self.audio_device.read().unwrap().clone()}))?)?;
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
        // an older engine names none: it was built without mp4 (APNG and GIF are always there)
        let formats: Vec<String> = v["video_formats"].as_array().map(|a| a.iter().filter_map(|f| f.as_str().map(str::to_string)).collect())
            .unwrap_or_else(|| vec!["apng".into(), "gif".into()]);
        *self.video_formats.lock().unwrap() = formats;
        self.cuda.store(v["cuda"].as_bool().unwrap_or(false), std::sync::atomic::Ordering::Relaxed);
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
                "done" | "loaded" | "pulled" | "unloaded" | "models" | "transcript" | "speech" => return Ok(v),
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
    let label = req["strata_label"].as_str().unwrap_or("").to_string();
    let _gpu = match s.mgr.gpu_exclusive_as(!s.media.keep_text_model, "media", &label).await {
        Ok(g) => g,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, "busy", format!("{e:#}")),
    };
    let mon = s.mgr.monitor.clone();
    let job = mon.media_begin("image", &model);
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
            if ev["event"] == "step" {
                mon.media_step(job, ev["step"].as_u64().unwrap_or(0) + i * steps, ev["total"].as_u64().unwrap_or(steps) * n);
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
        mon.media_end(job, "error", &f);
        return error(StatusCode::BAD_GATEWAY, "image_error", f);
    }
    mon.media_end(job, "stop", &format!("{} image{}, {w}x{h}", data.len(), if data.len() == 1 { "" } else { "s" }));
    Json(json!({"created": crate::now(), "data": data, "size": format!("{w}x{h}"),
                "strata": {"seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0, "steps": steps, "model": model,
                           "engine": "am2strata-media (Rust, candle)"}})).into_response()
}

/// GET /api/media/models: the media engine's catalog (image, video, 3D) with what is downloaded, the image model in
/// use, the download in progress, and the speech models still to come.
pub async fn models(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let pull = s.media.pull_job.lock().unwrap().clone();
    let base = json!({"engine": s.media.available(), "image_model": s.media.model(), "video_model": s.media.video_model(),
                      "stt_model": s.media.stt_model(), "tts_model": s.media.tts_model(),
                      "audio_device": s.media.audio_device.read().unwrap().clone(), "pull": pull});
    if !s.media.available() {
        let mut v = base;
        v["models"] = json!([]);
        v["note"] = json!("the media engine is not built: cargo build --release --manifest-path crates/strata-media/Cargo.toml --features cuda");
        return Json(v).into_response();
    }
    match s.media.call(json!({"op": "list"}), |_| {}).await {
        Ok(r) => {
            let mut v = base;
            v["models"] = r["models"].clone();
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

/// POST /api/media/models/default {"model": ...}: the image or video model for the next jobs (a downloaded one; its
/// kind decides which).
pub async fn set_default(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let Some(model) = req["model"].as_str() else { return bad("send {\"model\": ...}") };
    let mut kind = "image".to_string();
    if s.media.available() {
        let list = match s.media.call(json!({"op": "list"}), |_| {}).await {
            Ok(r) => r,
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "media_error", format!("{e:#}")),
        };
        let found = list["models"].as_array().and_then(|a| a.iter().find(|m| m["name"] == model).cloned());
        match found {
            None => return bad(format!("{model} is not in the media catalog")),
            Some(m) if !matches!(m["kind"].as_str(), Some("image" | "video" | "audio")) => return bad(format!("{model} is a {} model; image, video and audio models can be chosen", m["kind"].as_str().unwrap_or("?"))),
            Some(m) if m["downloaded"] != true => return bad(format!("{model} is not downloaded yet")),
            // an audio model is the speech-to-text or the text-to-speech one, by its task
            Some(m) if m["kind"] == "audio" => kind = m["task"].as_str().unwrap_or("stt").to_string(),
            Some(m) => kind = m["kind"].as_str().unwrap_or("image").to_string(),
        }
    }
    match s.media.set_model(&kind, model) {
        Ok(()) => Json(json!({"kind": kind, "model": model, "image_model": s.media.model(), "video_model": s.media.video_model(),
                              "stt_model": s.media.stt_model(), "tts_model": s.media.tts_model()})).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "io_error", format!("{e:#}")),
    }
}

fn chrono_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// POST /v1/videos/generations {"prompt", "model"?, "size"?: "WxH", "frames"?, "fps"?, "steps"?, "seed"?, "format"?}:
/// one clip, drawn by the media engine's video models (Wan, LTX), as one GPU job like an image. The answer is the clip
/// in base64 (MP4 by default; "webp" or "gif" on request) with its frame count and rate.
pub async fn video_generations(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if !s.media.available() {
        return error(StatusCode::NOT_IMPLEMENTED, "not_available",
                     "video generation is not built: cargo build --release --manifest-path crates/strata-media/Cargo.toml --features cuda");
    }
    let Some(prompt) = req["prompt"].as_str().filter(|p| !p.trim().is_empty()) else {
        return bad("send {\"prompt\": \"...\"}");
    };
    let model = req["model"].as_str().filter(|m| m.contains(':')).map(str::to_string).unwrap_or_else(|| s.media.video_model());
    // the model's own defaults (size, steps, guidance, frames, fps) fill what the request leaves out
    let list = match s.media.call(json!({"op": "list"}), |_| {}).await {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_GATEWAY, "media_error", format!("{e:#}")),
    };
    let Some(m) = list["models"].as_array().and_then(|a| a.iter().find(|x| x["name"] == model.as_str()).cloned()) else {
        return bad(format!("{model} is not in the media catalog"));
    };
    if m["kind"] != "video" {
        return bad(format!("{model} is not a video model"));
    }
    if m["downloaded"] != true {
        return bad(format!("{model} is not downloaded: download it in Settings > Models ({} GB)", m["size_gb"]));
    }
    let d = &m["defaults"];
    let (mut w, mut h) = (d["width"].as_u64().unwrap_or(832), d["height"].as_u64().unwrap_or(480));
    if let Some(size) = req["size"].as_str() {
        match size.split_once('x').and_then(|(a, b)| Some((a.parse::<u64>().ok()?, b.parse::<u64>().ok()?))) {
            Some((a, b)) if (128..=1920).contains(&a) && (128..=1920).contains(&b) && a % 16 == 0 && b % 16 == 0 => (w, h) = (a, b),
            _ => return bad("size must be WIDTHxHEIGHT, each a multiple of 16 from 128 to 1920"),
        }
    }
    let asked_frames = req["frames"].as_u64().or(d["frames"].as_u64()).unwrap_or(33);
    if !(1..=257).contains(&asked_frames) {
        return bad("frames must be 1 to 257");
    }
    // Wan takes 4n+1 frames, LTX 8n+1: a count between is rounded up (and said so)
    let step = video_frame_step(m["family"].as_str().unwrap_or(""));
    let frames = (asked_frames.saturating_sub(1)).div_ceil(step) * step + 1;
    let frames_note = (frames != asked_frames).then(|| format!("{asked_frames} frames rounded to {frames} ({} takes {step}n+1)", m["family"].as_str().unwrap_or("")));
    let fps = req["fps"].as_u64().or(d["fps"].as_u64()).unwrap_or(16).clamp(1, 60);
    let steps = req["steps"].as_u64().or(d["steps"].as_u64()).unwrap_or(30).clamp(1, 100);
    let guidance = req["guidance"].as_f64().or(d["guidance"].as_f64()).unwrap_or(5.0);
    let formats = s.media.video_formats.lock().unwrap().clone();
    let format = match req["format"].as_str() {
        // not asked: MP4 when this engine makes it, else APNG
        None => if formats.iter().any(|f| f == "mp4") { "mp4".to_string() } else { "apng".to_string() },
        Some(f) if formats.iter().any(|x| x == f) => f.to_string(),
        Some(f) => return bad(format!("format {f:?}: this media engine makes {}", formats.join(", "))),
    };
    let format = format.as_str();
    let seed = req["seed"].as_u64().unwrap_or_else(|| (crate::now() ^ (std::process::id() as u64) << 20) % 4_294_967_295);
    let t0 = Instant::now();
    let label = req["strata_label"].as_str().unwrap_or("").to_string();
    let _gpu = match s.mgr.gpu_exclusive_as(!s.media.keep_text_model, "media", &label).await {
        Ok(g) => g,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, "busy", format!("{e:#}")),
    };
    let r = json!({"model": model, "prompt": prompt, "width": w, "height": h, "steps": steps, "guidance": guidance, "seed": seed,
                   "batch_size": 1, "frames": frames, "fps": fps, "output_format": format});
    let mon = s.mgr.monitor.clone();
    let job = mon.media_begin("video", &model);
    let mut clip = None;
    let res = s.media.call(json!({"op": "generate", "request": r}), |ev| {
        if ev["event"] == "video" && clip.is_none() {
            clip = Some(ev.clone());
        }
        if ev["event"] == "step" {
            mon.media_step(job, ev["step"].as_u64().unwrap_or(0), ev["total"].as_u64().unwrap_or(steps));
        }
    }).await;
    if let Err(e) = s.media.call(json!({"op": "unload"}), |_| {}).await {
        tracing::warn!("the media engine did not unload: {e:#}");
    }
    let v = match (res, clip) {
        (Ok(_), Some(v)) => v,
        (Ok(_), None) => {
            mon.media_end(job, "error", "no video returned");
            return error(StatusCode::BAD_GATEWAY, "video_error", "the media engine returned no video");
        }
        (Err(e), _) => {
            mon.media_end(job, "error", &format!("{e:#}"));
            return error(StatusCode::BAD_GATEWAY, "video_error", format!("{e:#}"));
        }
    };
    mon.media_end(job, "stop", &format!("{}x{}, {} frames at {} fps", v["width"], v["height"], v["frames"], v["fps"]));
    Json(json!({"created": crate::now(), "data": [{"b64": v["data"], "format": v["format"], "seed": seed}],
                "size": format!("{}x{}", v["width"], v["height"]), "frames": v["frames"], "fps": v["fps"],
                "strata": {"seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0, "steps": steps, "model": model,
                           "engine": "am2strata-media (Rust, candle)", "frames_note": frames_note}})).into_response()
}

/// A video family's frame rule: n * step + 1 frames (Wan 4, LTX-Video and LTX-2 8; mold checks the same).
pub fn video_frame_step(family: &str) -> u64 {
    if family.starts_with("ltx") { 8 } else { 4 }
}
