//! Speech (v2.3), OpenAI's audio API in front of am2strata-media's Whisper and Parler-TTS:
//! - POST /v1/audio/transcriptions and /v1/audio/translations: a file (multipart, as OpenAI's clients send it, or JSON
//!   with "file" as base64 / a data URL, as the web app may) -> its text; response_format json, text, verbose_json, srt
//!   or vtt; `language` (ISO 639-1) skips the detection.
//! - POST /v1/audio/speech {"input", "voice"?, "model"?, "response_format"?: "wav"} -> audio/wav.
//! - POST /api/media/audio/device {"device": "auto" | "cpu" | "cuda"}: where speech runs. "auto": on the CPU when the
//!   free RAM has room for the model, else on the GPU with the text model stopped for the job (as for an image).
//! OpenAI's model names (whisper-1, tts-1, tts-1-hd, gpt-4o-*) mean the models chosen in Settings > Models.

use crate::{authorized, error, AppState};
use axum::body::Body;
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine as _;
use serde_json::{json, Value};
use std::time::Instant;

fn bad(msg: impl Into<String>) -> Response {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
}

/// MemAvailable in GB (0 when unknown). AM2_TEST_MEM_GB stands in for it in the tests.
fn mem_available_gb() -> f64 {
    if let Some(v) = std::env::var("AM2_TEST_MEM_GB").ok().and_then(|v| v.parse().ok()) {
        return v;
    }
    std::fs::read_to_string("/proc/meminfo").ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()))
        .map(|kb| kb / 1048576.0).unwrap_or(0.0)
}

/// The catalog entry of an audio model (from the engine's list), checked for its task and download.
async fn audio_model(s: &AppState, name: &str, task: &str) -> Result<Value, Response> {
    let list = s.media.call(json!({"op": "list"}), |_| {}).await
        .map_err(|e| error(StatusCode::BAD_GATEWAY, "media_error", format!("{e:#}")))?;
    let Some(m) = list["models"].as_array().and_then(|a| a.iter().find(|m| m["name"] == name).cloned()) else {
        return Err(bad(format!("{name} is not an audio model here (Settings > Models > Media models > Audio)")));
    };
    if m["task"] != task {
        return Err(bad(format!("{name} is a {} model, not {}", m["task"].as_str().unwrap_or("?"),
                               if task == "stt" { "speech to text" } else { "text to speech" })));
    }
    if m["downloaded"] != true {
        return Err(bad(format!("{name} is not downloaded: download it in Settings > Models ({} GB)", m["size_gb"])));
    }
    Ok(m)
}

/// Where this job runs, and whether it holds the GPU (the text model stopped meanwhile).
fn placement(s: &AppState, size_gb: f64) -> (&'static str, bool) {
    // an engine without CUDA computes on the CPU whatever the setting
    if !s.media.cuda.load(std::sync::atomic::Ordering::Relaxed) {
        return ("cpu", false);
    }
    match s.media.audio_device.read().unwrap().as_str() {
        "cpu" => ("cpu", false),
        "cuda" => ("cuda", true),
        // the weights in f32 and the activations beside them; Orca's 46 GB leave ~5 GB, which is not room for Parler
        _ if mem_available_gb() >= size_gb * 1.6 + 1.5 => ("cpu", false),
        _ => ("cuda", true),
    }
}

/// One audio job on the media engine: the GPU taken when it runs there, the Monitor told, the engine's models freed
/// after a GPU job. `on_event` sees the engine's progress events.
async fn job(s: &AppState, kind: &'static str, model: &str, size_gb: f64, mut cmd: Value, mut on_event: impl FnMut(&Value))
             -> Result<(Value, &'static str, u64), Response> {
    let (device, gpu) = placement(s, size_gb);
    cmd["device"] = json!(device);
    let _turn = if gpu {
        Some(s.mgr.gpu_exclusive(!s.media.keep_text_model).await
            .map_err(|e| error(StatusCode::SERVICE_UNAVAILABLE, "busy", format!("{e:#}")))?)
    } else {
        None
    };
    let mon = s.mgr.monitor.clone();
    let id = mon.media_begin(kind, model);
    let r = s.media.call(cmd, |ev| {
        if ev["event"] == "part" {
            mon.media_step(id, ev["done"].as_u64().unwrap_or(0), ev["total"].as_u64().unwrap_or(1));
        }
        on_event(ev)
    }).await;
    if gpu {
        if let Err(e) = s.media.call(json!({"op": "unload"}), |_| {}).await {
            tracing::warn!("the media engine did not unload: {e:#}");
        }
    } else if mem_available_gb() < size_gb * 2.0 + 2.0 {
        // the RAM is short (a large text model beside it): the speech model goes at once
        let _ = s.media.call(json!({"op": "unload_audio"}), |_| {}).await;
    } else {
        // else it stays warm for the next request, and goes after 120 s without one
        let media = s.media.clone();
        let mut idle = s.media.audio_idle.lock().unwrap();
        if let Some(h) = idle.take() {
            h.abort();
        }
        *idle = Some(tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(120)).await;
            let _ = media.call(json!({"op": "unload_audio"}), |_| {}).await;
        }));
    }
    match r {
        Ok(v) => Ok((v, device, id)),
        Err(e) => {
            mon.media_end(id, "error", &format!("{e:#}"));
            Err(error(StatusCode::BAD_GATEWAY, "audio_error", format!("{e:#}")))
        }
    }
}

fn openai_name(asked: Option<&str>, default: String, openai: &[&str]) -> String {
    match asked {
        Some(m) if !m.is_empty() && !openai.contains(&m) && !m.starts_with("gpt-4o") => m.to_string(),
        _ => default,
    }
}

/// The fields of a transcription request: multipart (OpenAI's form) or JSON with the file in base64 / a data URL.
async fn fields(s: &AppState, req: Request) -> Result<(Vec<u8>, Value), Response> {
    let ct = req.headers().get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    if ct.starts_with("multipart/form-data") {
        let mut mp = Multipart::from_request(req, s).await.map_err(|e| bad(format!("multipart: {e}")))?;
        let (mut file, mut f) = (Vec::new(), json!({}));
        while let Some(field) = mp.next_field().await.map_err(|e| bad(format!("multipart: {e}")))? {
            let name = field.name().unwrap_or("").to_string();
            if name == "file" {
                file = field.bytes().await.map_err(|e| bad(format!("the file: {e}")))?.to_vec();
            } else {
                let v = field.text().await.unwrap_or_default();
                f[name.trim_end_matches("[]")] = json!(v);
            }
        }
        return Ok((file, f));
    }
    let Json(v): Json<Value> = Json::from_request(req, s).await.map_err(|e| bad(format!("send multipart/form-data, or JSON: {e}")))?;
    let raw = v["file"].as_str().unwrap_or("");
    let b64 = raw.split_once(";base64,").map(|(_, b)| b).unwrap_or(raw);
    let file = base64::engine::general_purpose::STANDARD.decode(b64.trim()).map_err(|_| bad("\"file\": base64 or a data: URL"))?;
    Ok((file, v))
}

fn stamp(t: f64, sep: char) -> String {
    let ms = (t * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02}{sep}{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

async fn transcribe(s: AppState, headers: HeaderMap, req: Request, translate: bool) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if !s.media.available() {
        return error(StatusCode::NOT_IMPLEMENTED, "not_available", "the media engine is not built");
    }
    let (file, f) = match fields(&s, req).await {
        Ok(x) => x,
        Err(r) => return r,
    };
    if file.is_empty() {
        return bad("send the audio as \"file\"");
    }
    let format = f["response_format"].as_str().unwrap_or("json").to_string();
    if !matches!(format.as_str(), "json" | "text" | "verbose_json" | "srt" | "vtt") {
        return bad("response_format: json, text, verbose_json, srt or vtt");
    }
    let model = openai_name(f["model"].as_str(), s.media.stt_model(), &["whisper-1"]);
    let m = match audio_model(&s, &model, "stt").await {
        Ok(m) => m,
        Err(r) => return r,
    };
    let t0 = Instant::now();
    let mut cmd = json!({"op": "transcribe", "model": model, "audio": base64::engine::general_purpose::STANDARD.encode(&file),
                         "translate": translate});
    if let Some(l) = f["language"].as_str().filter(|l| !l.is_empty()) {
        cmd["language"] = json!(l.to_lowercase());
    }
    let kind = if translate { "translation" } else { "speech-to-text" };
    let (r, device, id) = match job(&s, kind, &model, m["size_gb"].as_f64().unwrap_or(1.0), cmd, |_| {}).await {
        Ok(x) => x,
        Err(r) => return r,
    };
    let text = r["text"].as_str().unwrap_or("").to_string();
    let secs = r["duration"].as_f64().unwrap_or(0.0);
    let words: usize = text.split_whitespace().count();
    s.mgr.monitor.media_end(id, "stop", &format!("{secs:.1} s of audio, {words} words, on the {device}"));
    let segments = r["segments"].as_array().cloned().unwrap_or_default();
    match format.as_str() {
        "text" => text.into_response(),
        "srt" | "vtt" => {
            let mut out = if format == "vtt" { "WEBVTT\n\n".to_string() } else { String::new() };
            for (i, sg) in segments.iter().enumerate() {
                let sep = if format == "vtt" { '.' } else { ',' };
                if format == "srt" {
                    out.push_str(&format!("{}\n", i + 1));
                }
                out.push_str(&format!("{} --> {}\n{}\n\n", stamp(sg["start"].as_f64().unwrap_or(0.0), sep),
                                      stamp(sg["end"].as_f64().unwrap_or(0.0), sep), sg["text"].as_str().unwrap_or("")));
            }
            ([(header::CONTENT_TYPE, if format == "vtt" { "text/vtt" } else { "application/x-subrip" })], out).into_response()
        }
        "verbose_json" => Json(json!({"task": r["task"], "language": r["language"], "duration": secs, "text": text,
                                      "segments": segments, "strata": {"model": model, "device": device,
                                      "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0}})).into_response(),
        _ => Json(json!({"text": text})).into_response(),
    }
}

pub async fn transcriptions(State(s): State<AppState>, headers: HeaderMap, req: Request) -> Response {
    transcribe(s, headers, req, false).await
}

pub async fn translations(State(s): State<AppState>, headers: HeaderMap, req: Request) -> Response {
    transcribe(s, headers, req, true).await
}

/// POST /v1/audio/speech
pub async fn speech(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if !s.media.available() {
        return error(StatusCode::NOT_IMPLEMENTED, "not_available", "the media engine is not built");
    }
    let Some(text) = req["input"].as_str().filter(|t| !t.trim().is_empty()) else {
        return bad("send {\"input\": \"the text to speak\"}");
    };
    if text.chars().count() > 4096 {
        return bad("input: at most 4096 characters");
    }
    if let Some(f) = req["response_format"].as_str().filter(|f| *f != "wav") {
        return bad(format!("response_format {f:?}: this server makes wav (WAVE, 16-bit PCM) only"));
    }
    let model = openai_name(req["model"].as_str(), s.media.tts_model(), &["tts-1", "tts-1-hd"]);
    let m = match audio_model(&s, &model, "tts").await {
        Ok(m) => m,
        Err(r) => return r,
    };
    // OpenAI's voice names map to this model's voices; a sentence describing a voice is taken as it is
    let voice = req["voice"].as_str().unwrap_or("laura");
    let voice = match voice {
        "alloy" | "nova" | "shimmer" | "coral" | "sage" => "laura",
        "echo" | "onyx" | "ash" => "jon",
        "fable" | "ballad" => "lea",
        v => v,
    };
    let mut cmd = json!({"op": "speak", "model": model, "text": text, "voice": voice, "seed": req["seed"].as_u64().unwrap_or(0)});
    if let Some(t) = req["temperature"].as_f64() {
        cmd["temperature"] = json!(t.clamp(0.0, 2.0)); // Parler samples at 1.0 unless told (0: greedy)
    }
    let (r, device, id) = match job(&s, "text-to-speech", &model, m["size_gb"].as_f64().unwrap_or(3.5), cmd, |_| {}).await {
        Ok(x) => x,
        Err(r) => return r,
    };
    let wav = match base64::engine::general_purpose::STANDARD.decode(r["data"].as_str().unwrap_or("")) {
        Ok(b) => b,
        Err(_) => return error(StatusCode::BAD_GATEWAY, "audio_error", "the media engine returned no audio"),
    };
    s.mgr.monitor.media_end(id, "stop", &format!("{:.1} s of speech, {} characters, on the {device}", r["seconds"].as_f64().unwrap_or(0.0),
                                          text.chars().count()));
    Response::builder().header(header::CONTENT_TYPE, "audio/wav")
        .header("x-strata-seconds", r["seconds"].to_string()).header("x-strata-model", model)
        .body(Body::from(wav)).unwrap()
}

/// POST /api/media/audio/device {"device": "auto" | "cpu" | "cuda"}
pub async fn set_device(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("settings can be changed");
    }
    let Some(d) = req["device"].as_str().filter(|d| matches!(*d, "auto" | "cpu" | "cuda")) else {
        return bad("device: auto, cpu or cuda");
    };
    match s.media.set_model("audio_device", d) {
        Ok(()) => Json(json!({"audio_device": d})).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "io_error", format!("{e:#}")),
    }
}
