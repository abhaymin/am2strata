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
    /// commands sent whose ending event was not read: a caller dropped while it waited (a client that went away, the
    /// idle unload's timer aborted). Their answers are still coming, and are read and dropped before the next command,
    /// or every later answer would go to the command after the one it belongs to.
    unread: usize,
    /// a caller was dropped in the middle of writing its command: the engine has half a line, and the only way back
    /// to a clean state is a new engine
    writing: bool,
}

/// The events that end a command.
fn ends(event: &str) -> bool {
    matches!(event, "error" | "done" | "loaded" | "pulled" | "unloaded" | "models" | "transcript" | "speech" | "song" | "removal" | "removed" | "accepted")
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
    /// the engine's last catalog (its "models" event), answered while it is busy with a job instead of waiting minutes
    catalog: std::sync::Mutex<Option<Value>>,
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
            catalog: std::sync::Mutex::new(None),
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
        Ok(Proc { child, stdin, lines, unread: 0, writing: false })
    }

    /// One command; `on_event` sees every event until the one that ends it (done / loaded / pulled / unloaded / error).
    /// The engine's catalog. While a job holds the engine (an image takes minutes), the last catalog is answered at once,
    /// marked `"busy": true`, instead of waiting behind the job; a free engine is asked again and the answer kept.
    pub async fn list(&self) -> anyhow::Result<Value> {
        if self.proc.try_lock().is_err() {
            if let Some(mut c) = self.catalog.lock().unwrap().clone() {
                c["busy"] = json!(true);
                return Ok(c);
            }
        }
        let r = self.call(json!({"op": "list"}), |_| {}).await?;
        *self.catalog.lock().unwrap() = Some(r.clone());
        Ok(r)
    }

    pub async fn call(&self, cmd: Value, mut on_event: impl FnMut(&Value)) -> anyhow::Result<Value> {
        let mut guard = self.proc.lock().await;
        if guard.as_ref().is_some_and(|p| p.writing) {
            tracing::warn!("a command to the media engine was cut off while it was written: starting the engine again");
            *guard = None;
        }
        if guard.as_mut().is_none_or(|p| p.child.try_wait().ok().flatten().is_some()) {
            *guard = Some(self.spawn().await?);
        }
        let p = guard.as_mut().unwrap();
        // an earlier caller's answer still on its way: wait for it, and drop it
        while p.unread > 0 {
            let Some(line) = p.lines.next_line().await? else {
                *guard = None;
                anyhow::bail!("the media engine stopped unexpectedly");
            };
            if serde_json::from_str::<Value>(&line).is_ok_and(|v| ends(v["event"].as_str().unwrap_or(""))) {
                p.unread -= 1;
            }
        }
        p.unread += 1;
        p.writing = true;
        p.stdin.write_all(format!("{cmd}\n").as_bytes()).await?;
        p.stdin.flush().await?;
        p.writing = false;
        loop {
            let Some(line) = p.lines.next_line().await? else {
                *guard = None;
                anyhow::bail!("the media engine stopped unexpectedly");
            };
            let v: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let event = v["event"].as_str().unwrap_or("");
            if !ends(event) {
                on_event(&v);
                continue;
            }
            p.unread -= 1;
            if event == "error" {
                anyhow::bail!("{}", v["message"].as_str().unwrap_or("the media engine failed"));
            }
            return Ok(v);
        }
    }
}

fn bad(msg: impl Into<String>) -> Response {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
}

pub async fn generations(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    images(s, headers, req, Vec::new()).await
}

/// The most images one edit takes: the one edited and its references (Qwen Image 2.1 takes up to 3 in all).
const MAX_EDIT_IMAGES: usize = 3;

/// The size an edit draws at when none is asked: the first image's shape at about a megapixel, in multiples of 16.
fn edit_size(png_or_jpeg: &[u8]) -> Option<(u32, u32)> {
    let (w, h) = image::ImageReader::new(std::io::Cursor::new(png_or_jpeg)).with_guessed_format().ok()?.into_dimensions().ok()?;
    let k = (1024.0 * 1024.0 / (w as f64 * h as f64)).sqrt();
    let side = |x: u32| (((x as f64 * k) / 16.0).round() as u32 * 16).clamp(256, 2048);
    Some((side(w), side(h)))
}

/// POST /v1/images/edits: OpenAI's images edit API. Multipart as OpenAI's clients send it (`image` or `image[]`, one
/// to three files, `prompt`, `size`, `n`, `model`), or JSON with `images` (or `image`) as base64 / data URLs. The first
/// image is edited, the others are references; drawn by an image model that takes edits (Qwen Image 2.1).
pub async fn edits(State(s): State<AppState>, headers: HeaderMap, req: axum::extract::Request) -> Response {
    use axum::extract::FromRequest;
    use base64::Engine as _;
    if !crate::api_authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let ct =req.headers().get(axum::http::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let (mut fields, mut files) = (json!({}), Vec::<Vec<u8>>::new());
    if ct.starts_with("multipart/form-data") {
        let mut mp = match axum::extract::Multipart::from_request(req, &s).await {
            Ok(m) => m,
            Err(e) => return bad(format!("multipart: {e}")),
        };
        loop {
            let field = match mp.next_field().await {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => return bad(format!("multipart: {e}")),
            };
            let name = field.name().unwrap_or("").trim_end_matches("[]").to_string();
            match name.as_str() {
                "image" => match field.bytes().await {
                    Ok(b) => files.push(b.to_vec()),
                    Err(e) => return bad(format!("the image: {e}")),
                },
                "mask" => return bad("mask is not supported: describe the change in the prompt (the whole image is edited)"),
                _ => {
                    let v = field.text().await.unwrap_or_default();
                    fields[name] = v.parse::<u64>().map(Value::from).unwrap_or(Value::String(v));
                }
            }
        }
    } else {
        let Json(v): Json<Value> = match Json::from_request(req, &s).await {
            Ok(j) => j,
            Err(e) => return bad(format!("send multipart/form-data, or JSON: {e}")),
        };
        if v.get("mask").is_some_and(|m| !m.is_null()) {
            return bad("mask is not supported: describe the change in the prompt (the whole image is edited)");
        }
        let list: Vec<Value> = match (&v["images"], &v["image"]) {
            (Value::Array(a), _) => a.clone(),
            (_, Value::Array(a)) => a.clone(),
            (_, Value::String(one)) => vec![Value::String(one.clone())],
            _ => Vec::new(),
        };
        for (i, item) in list.iter().enumerate() {
            // a string, or {"image_url": ...} / {"b64_json": ...} as OpenAI's newer clients send
            let raw = item.as_str().or_else(|| item["image_url"].as_str()).or_else(|| item["b64_json"].as_str()).unwrap_or("");
            let b64 = raw.split_once(";base64,").map(|(_, b)| b).unwrap_or(raw);
            match base64::engine::general_purpose::STANDARD.decode(b64.trim()) {
                Ok(b) if !b.is_empty() => files.push(b),
                _ => return bad(format!("images[{i}]: base64 or a data: URL")),
            }
        }
        fields = v;
    }
    if files.is_empty() {
        return bad("send the image to edit: multipart `image`, or JSON {\"images\": [\"data:image/png;base64,...\"]}");
    }
    if files.len() > MAX_EDIT_IMAGES {
        return bad(format!("at most {MAX_EDIT_IMAGES} images: the one edited and up to {} references", MAX_EDIT_IMAGES - 1));
    }
    if fields["size"].as_str().is_none_or(|z| z == "auto") {
        match edit_size(&files[0]) {
            Some((w, h)) => fields["size"] = json!(format!("{w}x{h}")),
            None => return bad("the image is not a PNG, JPEG or WebP"),
        }
    }
    let b64 = files.iter().map(|f| base64::engine::general_purpose::STANDARD.encode(f)).collect();
    images(s, headers, fields, b64).await
}

/// POST /api/media/licenses/accept {"id": "qwen-research"}: the user agreed to a model licence in the app (the
/// download it gates then goes ahead). This server's own page only.
pub async fn accept_license(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("a licence can be accepted");
    }
    let Some(id) = req["id"].as_str() else { return bad("send {\"id\": \"<licence id>\"} from GET /api/media/models") };
    match s.media.call(json!({"op": "accept_license", "id": id}), |_| {}).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => bad(format!("{e:#}")),
    }
}

/// Generations and edits: `edit_images` (base64) empty for a text-to-image request.
async fn images(s: AppState, headers: HeaderMap, req: Value, edit_images: Vec<String>) -> Response {
    if !crate::api_authorized(&s, &headers) {
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
    // steps, guidance and the negative prompt a request leaves out are the model's own (the engine fills them in)
    let steps = req["steps"].as_u64().map(|n| n.clamp(1, 100));
    let guidance = req["guidance"].as_f64().filter(|g| (0.0..=20.0).contains(g));
    let negative = req["negative_prompt"].as_str();
    let seed = req["seed"].as_u64().unwrap_or_else(|| (crate::now() ^ (std::process::id() as u64) << 20) % 4_294_967_295);
    let model = req["model"].as_str().filter(|m| m.contains(':')).map(str::to_string).unwrap_or_else(|| s.media.model());
    if !edit_images.is_empty() {
        // a model that cannot take edit images would draw from the prompt alone: say so instead
        let takes = match s.media.list().await {
            Ok(r) => r["models"].as_array().and_then(|a| a.iter().find(|m| m["name"] == model.as_str())).map(|m| m["edits"] == true),
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "media_error", format!("{e:#}")),
        };
        match takes {
            None => return bad(format!("{model} is not in the media catalog")),
            Some(false) => return bad(format!("{model} cannot edit images; choose one that can, such as chatr-omni-qwen-image-2.1:q8 or qwen-image-2.1:q8 (\"model\")")),
            Some(true) => {}
        }
    }
    let t0 = Instant::now();
    let label = s.media_jobs.label(&req["strata_label"]);
    let _gpu = match s.mgr.gpu_exclusive_as(!s.media.keep_text_model, "media", &label).await {
        Ok(g) => g,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, "busy", format!("{e:#}")),
    };
    let mon = s.mgr.monitor.clone();
    let job = mon.media_begin(if edit_images.is_empty() { "image" } else { "image edit" }, &model);
    let _row = crate::monitor::MediaRow::new(&mon, job);
    let mut data = Vec::new();
    let mut failure = None;
    let mut ran = steps.unwrap_or(0);
    for i in 0..n {
        let mut r = json!({"model": model, "prompt": prompt, "width": w, "height": h, "seed": seed + i, "batch_size": 1});
        if let Some(p) = placement(&s, &model) {
            r["placement"] = p;
        }
        if let Some(st) = steps {
            r["steps"] = json!(st);
        }
        if let Some(g) = guidance {
            r["guidance"] = json!(g);
        }
        if let Some(neg) = negative {
            r["negative_prompt"] = json!(neg);
        }
        if !edit_images.is_empty() {
            r["edit_images"] = json!(edit_images);
        }
        let mut png = None;
        let res = s.media.call(json!({"op": "generate", "request": r}), |ev| {
            if ev["event"] == "image" && png.is_none() {
                png = ev["png"].as_str().map(str::to_owned);
            }
            if ev["event"] == "step" {
                let total = ev["total"].as_u64().unwrap_or(1);
                ran = total;
                mon.media_step(job, ev["step"].as_u64().unwrap_or(0) + i * total, total * n);
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
    for d in &data {
        use base64::Engine as _;
        if let Ok(png) = base64::engine::general_purpose::STANDARD.decode(d["b64_json"].as_str().unwrap_or("")) {
            mon.media_file_add(job, "png", png);
        }
    }
    mon.media_end(job, "stop", &format!("{} image{}, {w}x{h}", data.len(), if data.len() == 1 { "" } else { "s" }));
    Json(json!({"created": crate::now(), "data": data, "size": format!("{w}x{h}"),
                "strata": {"seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0, "steps": ran, "model": model,
                           "edit_images": edit_images.len(),
                           "engine": "am2strata-media (Rust, candle)"}})).into_response()
}

/// Where a GPU job's parts go when the PC has a second card and the text model is stopped for the job: the main model
/// on GPU 0, its text encoder and VAE on GPU 1 (devices in sequence). Measured 2026-11-09: Qwen-Image 2.1 (Q8,
/// 1024x768) on GPU 0 alone peaked at 15.2 of 16.3 GB in the VAE decode while GPU 1 stayed empty, and the same job
/// failed with CUDA_ERROR_OUT_OF_MEMORY. For the families checked so far (Qwen-Image); others keep mold's own choice.
fn placement(s: &AppState, model: &str) -> Option<Value> {
    let qwen_image = model.contains("qwen-image");
    if !qwen_image || s.mgr.hw.gpus.len() < 2 || s.media.keep_text_model {
        return None;
    }
    let gpu = |n: usize| json!({"kind": "gpu", "ordinal": n});
    Some(json!({"text_encoders": gpu(1), "advanced": {"transformer": gpu(0), "vae": gpu(1), "qwen": gpu(1)}}))
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
    match s.media.list().await {
        Ok(r) => {
            let mut v = base;
            v["models"] = r["models"].clone();
            v["busy"] = json!(r["busy"] == true);
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
    // image and video models are "<name>:<variant>"; speech models are plain names from the catalog
    let asked = req["model"].as_str().unwrap_or("").to_string();
    let listed = !asked.is_empty() && !asked.contains(':') && s.media.list().await.ok()
        .and_then(|l| l["models"].as_array().map(|a| a.iter().any(|x| x["name"] == asked.as_str()))).unwrap_or(false);
    if !(asked.contains(':') || listed) {
        return bad("send {\"model\": \"<name>\"} from GET /api/media/models");
    }
    let model = asked;
    {
        let mut job = s.media.pull_job.lock().unwrap();
        if job["state"] == "pulling" {
            return error(StatusCode::CONFLICT, "busy", format!("{} is downloading; one download at a time", job["model"].as_str().unwrap_or("")));
        }
        *job = json!({"state": "pulling", "model": model, "started": chrono_now(), "done": 0, "total": 0, "left": 0,
                      "percent": 0.0, "speed": 0, "eta_s": null, "file": null, "file_index": 0, "files": 0});
    }
    let media = s.media.clone();
    tokio::spawn(async move {
        let mut rate = Rate::default();
        let job = &media.pull_job;
        let r = media.call(json!({"op": "pull", "model": model}), |v| {
            if v["event"] == "progress" {
                let mut j = job.lock().unwrap();
                rate.progress(&mut j, v);
            }
        }).await;
        // the last progress stays, so the page can say how much came down
        let mut j = media.pull_job.lock().unwrap();
        j["model"] = json!(model);
        j["eta_s"] = Value::Null;
        match r {
            Ok(_) => {
                j["state"] = json!("done");
                j["left"] = json!(0);
                j["percent"] = json!(100.0);
                j["done"] = j["total"].clone();
            }
            Err(e) => {
                j["state"] = json!("error");
                j["error"] = json!(format!("{e:#}"));
            }
        }
    });
    (StatusCode::ACCEPTED, Json(json!({"state": "pulling"}))).into_response()
}

/// A download's speed over the last 10 s (a resumed download starts from its bytes on disk, so the average since the
/// start would mislead), and from it the bytes and the time left.
#[derive(Default)]
pub struct Rate {
    samples: std::collections::VecDeque<(std::time::Instant, u64)>,
}

impl Rate {
    pub fn progress(&mut self, j: &mut Value, v: &Value) {
        let now = std::time::Instant::now();
        let done = v["done"].as_u64().unwrap_or(0);
        let total = v["total"].as_u64().unwrap_or(0).max(done);
        // a new batch (bytes went back): start the window again
        if self.samples.back().is_some_and(|s| s.1 > done) {
            self.samples.clear();
        }
        self.samples.push_back((now, done));
        while self.samples.len() > 2 && now.duration_since(self.samples[0].0).as_secs_f64() > 10.0 {
            self.samples.pop_front();
        }
        let (t0, d0) = self.samples[0];
        let secs = now.duration_since(t0).as_secs_f64();
        let speed = if secs >= 0.1 { ((done - d0) as f64 / secs) as u64 } else { j["speed"].as_u64().unwrap_or(0) };
        let left = total - done;
        j["done"] = json!(done);
        j["total"] = json!(total);
        j["left"] = json!(left);
        j["percent"] = json!(if total > 0 { (1000.0 * done as f64 / total as f64).round() / 10.0 } else { 0.0 });
        j["speed"] = json!(speed);
        j["eta_s"] = if speed > 0 { json!((left as f64 / speed as f64).round() as u64) } else { Value::Null };
        if let Some(f) = v["file"].as_str() {
            j["file"] = json!(f);
        }
        if let Some(i) = v["file_index"].as_u64().filter(|i| *i > 0) {
            j["file_index"] = json!(i);
        }
        if let Some(n) = v["files"].as_u64().filter(|n| *n > 0) {
            j["files"] = json!(n);
        }
    }
}

#[cfg(test)]
mod rate_tests {
    use super::*;

    #[test]
    fn speed_left_and_time_left() {
        let mut r = Rate::default();
        let mut j = json!({});
        let t = std::time::Instant::now() - std::time::Duration::from_secs(4);
        r.samples.push_back((t, 0));
        r.progress(&mut j, &json!({"done": 400_000_000u64, "total": 1_000_000_000u64, "file": "a.gguf", "file_index": 1, "files": 2}));
        assert_eq!(j["left"], 600_000_000u64);
        assert_eq!(j["percent"], 40.0);
        let speed = j["speed"].as_u64().unwrap();
        assert!((95_000_000..=101_000_000).contains(&speed), "{speed}");
        let eta = j["eta_s"].as_u64().unwrap();
        assert!((5..=7).contains(&eta), "{eta}");
        // a later event without the file count keeps the one known
        r.progress(&mut j, &json!({"done": 500_000_000u64, "total": 1_000_000_000u64, "file": "a.gguf", "file_index": 1, "files": 0}));
        assert_eq!(j["files"], 2);
    }
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
        let list = match s.media.list().await {
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

/// POST /v1/videos/generations {"prompt", "model"?, "size"?: "WxH", "frames"?, "fps"?, "steps"?, "seed"?, "format"?,
/// "seconds"? / "total_frames"?}: one clip, drawn by the media engine's video models (Wan, LTX), as one GPU job like an
/// image. Longer than one clip ("seconds" or "total_frames" past `frames`): a chain of clips (mold's chain, each
/// continuing from the one before on a model that takes a source image), stitched into one video. The answer is the clip
/// in base64 (MP4 by default; "webp" or "gif" on request) with its frame count and rate.
pub async fn video_generations(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !crate::api_authorized(&s, &headers) {
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
    let list = match s.media.list().await {
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
    let label = s.media_jobs.label(&req["strata_label"]);
    let _gpu = match s.mgr.gpu_exclusive_needing(!s.media.keep_text_model, "media", &label, crate::manager::VIDEO_JOB_GB).await {
        Ok(g) => g,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, "busy", format!("{e:#}")),
    };
    let r = json!({"model": model, "prompt": prompt, "width": w, "height": h, "steps": steps, "guidance": guidance, "seed": seed,
                   "batch_size": 1, "frames": frames, "fps": fps, "output_format": format});
    // longer than one clip: a chain of clips. The carry between clips is the family's: Wan continues from the last
    // frame alone (mold's WAN_HANDOFF_DUPLICATED_FRAMES), LTX from a 17-frame motion tail
    let tail = if m["family"].as_str().unwrap_or("").starts_with("wan") { 1 } else { 17 };
    let total = req["total_frames"].as_u64().or(req["seconds"].as_f64().map(|x| (x * fps as f64).round() as u64))
        .map(|t| t.min(MAX_CHAIN_FRAMES));
    let (op, r) = match total.filter(|t| *t > frames) {
        Some(t) => {
            let mut c = json!({"model": model, "prompt": prompt, "width": w, "height": h, "steps": steps, "guidance": guidance,
                               "seed": seed, "fps": fps, "output_format": format, "total_frames": t, "clip_frames": frames,
                               "motion_tail_frames": tail});
            // Wan chains: "overlap_frames" (crossfaded frames at each join, default 8) and "hold_colour" (default on)
            if let Some(o) = req["overlap_frames"].as_u64() {
                c["overlap_frames"] = json!(o.min(32));
            }
            if let Some(h) = req["hold_colour"].as_bool() {
                c["hold_colour"] = json!(h);
            }
            ("chain", c)
        }
        None => ("generate", r),
    };
    let mon = s.mgr.monitor.clone();
    let job = mon.media_begin("video", &model);
    let _row = crate::monitor::MediaRow::new(&mon, job);
    let mut clip = None;
    let res = s.media.call(json!({"op": op, "request": r}), |ev| {
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
    if let Ok(clip) = {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD.decode(v["data"].as_str().unwrap_or(""))
    } {
        mon.media_file_add(job, v["format"].as_str().unwrap_or("mp4"), clip);
    }
    mon.media_end(job, "stop", &format!("{}x{}, {} frames at {} fps", v["width"], v["height"], v["frames"], v["fps"]));
    Json(json!({"created": crate::now(), "data": [{"b64": v["data"], "format": v["format"], "seed": seed}],
                "size": format!("{}x{}", v["width"], v["height"]), "frames": v["frames"], "fps": v["fps"], "clips": v["clips"].as_u64().unwrap_or(1),
                "strata": {"seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0, "steps": steps, "model": model,
                           "engine": "am2strata-media (Rust, candle)", "frames_note": frames_note}})).into_response()
}

/// The longest chained video: 10 minutes at 24 fps.
pub const MAX_CHAIN_FRAMES: u64 = 14_400;

/// A video family's frame rule: n * step + 1 frames (Wan 4, LTX-Video and LTX-2 8; mold checks the same).
pub fn video_frame_step(family: &str) -> u64 {
    if family.starts_with("ltx") { 8 } else { 4 }
}
