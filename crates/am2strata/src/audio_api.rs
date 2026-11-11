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

fn mem_available_gb() -> f64 {
    crate::mem::available_gb()
}

/// The catalog entry of an audio model (from the engine's list), checked for its task and download.
async fn audio_model(s: &AppState, name: &str, task: &str) -> Result<Value, Response> {
    let list = s.media.list().await
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

pub const MAX_SPEECH_CHARS: usize = 20_000;

/// How long a CPU speech job waits for RAM to come free before it gives up.
const RAM_WAIT_S: u64 = 600;

/// A language's English name, for messages ("Chinese" for "zh"); the code itself when it is not known here.
fn language_name(code: &str) -> String {
    let names = [("zh", "Chinese"), ("ja", "Japanese"), ("he", "Hebrew"), ("th", "Thai"), ("en", "English"), ("bn", "Bengali"),
                 ("ta", "Tamil"), ("te", "Telugu"), ("mr", "Marathi"), ("gu", "Gujarati"), ("kn", "Kannada"), ("ml", "Malayalam"),
                 ("or", "Odia"), ("pa", "Punjabi"), ("as", "Assamese"), ("ne", "Nepali")];
    strata_names(code).or_else(|| names.iter().find(|n| n.0 == code).map(|n| n.1.to_string())).unwrap_or_else(|| code.to_string())
}

fn strata_names(code: &str) -> Option<String> {
    const L: &[(&str, &str)] = &[("ar", "Arabic"), ("da", "Danish"), ("de", "German"), ("el", "Greek"), ("es", "Spanish"),
        ("fi", "Finnish"), ("fr", "French"), ("hi", "Hindi"), ("it", "Italian"), ("ko", "Korean"), ("ms", "Malay"), ("nl", "Dutch"),
        ("no", "Norwegian"), ("pl", "Polish"), ("pt", "Portuguese"), ("ru", "Russian"), ("sv", "Swedish"), ("sw", "Swahili"), ("tr", "Turkish")];
    L.iter().find(|l| l.0 == code).map(|l| l.1.to_string())
}

/// Where this job runs, and whether it holds the GPU (the text model stopped meanwhile).
fn placement(s: &AppState, size_gb: f64, tts: bool) -> (&'static str, bool) {
    // an engine without CUDA computes on the CPU whatever the setting
    if !s.media.cuda.load(std::sync::atomic::Ordering::Relaxed) {
        return ("cpu", false);
    }
    match s.media.audio_device.read().unwrap().as_str() {
        "cpu" => ("cpu", false),
        "cuda" => ("cuda", true),
        // no text model loaded: the GPU is free, and 10 to 30 times faster
        _ if !matches!(s.mgr.queue.resident(), crate::queue::Resident::Text(_)) => ("cuda", true),
        // Parler on the CPU runs at about 20x real time (a paragraph: 140 s, on the GPU 6 s): a GPU job, the text model
        // stopped meanwhile and loaded again with the next chat
        _ if tts => ("cuda", true),
        // measured peak RSS on the CPU (2026-10-28): Whisper large-v3-turbo 5.0 GB for 1.6 GB of weights, Parler mini
        // 7.1 GB for 3.5 GB; Orca's 46 GB leave ~5 GB, room for neither
        _ if mem_available_gb() >= size_gb * 2.5 + 1.5 => ("cpu", false),
        _ => ("cuda", true),
    }
}

/// One audio job on the media engine: the GPU taken when it runs there, the Monitor told, the engine's models freed
/// after a GPU job. `on_event` sees the engine's progress events.
async fn job(s: &AppState, kind: &'static str, model: &str, size_gb: f64, mut cmd: Value, mut on_event: impl FnMut(&Value))
             -> Result<(Value, &'static str, crate::monitor::MediaRow), Response> {
    // an ONNX model (Chatterbox) runs on the CPU, beside the text model: no GPU turn, but it must fit in the RAM left
    let onnx = cmd["runtime"] == "onnx";
    let (device, gpu) = if onnx { ("cpu", false) } else { placement(s, size_gb, kind == "text-to-speech") };
    if onnx {
        // measured (2026-11-09): Chatterbox Turbo's 3.3 GB of fp32 weights peak at 5.0 GB while loading
        // the take check's Whisper too (measured: Whisper large-v3-turbo peaks at 5.0 GB for 1.6 GB of weights)
        let need = size_gb * 1.5 + 0.5 + cmd["check_gb"].as_f64().unwrap_or(0.0) * 3.0;
        // in sequence when the RAM is short: the job waits (up to 10 minutes) for what runs beside it to finish and
        // free its memory, instead of starting and pushing the PC out of RAM, or failing at once
        let t0 = std::time::Instant::now();
        let mut free = mem_available_gb();
        // (with the tests' fixed RAM figure there is nothing to wait for)
        let wait = if std::env::var("AM2_TEST_MEM_GB").is_ok() { 0 } else { RAM_WAIT_S };
        while free < need && t0.elapsed() < std::time::Duration::from_secs(wait) {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            free = mem_available_gb();
        }
        if free < need {
            return Err(error(StatusCode::SERVICE_UNAVAILABLE, "busy",
                format!("{model} needs about {need:.1} GB of free RAM on the CPU and {free:.1} GB stayed free for {} minutes: unload the text model or use a smaller speech model", RAM_WAIT_S / 60)));
        }
    }
    cmd.as_object_mut().map(|o| { o.remove("runtime"); o.remove("check_gb") });
    cmd["device"] = json!(device);
    let _turn = if gpu {
        Some(s.mgr.gpu_exclusive(!s.media.keep_text_model).await
            .map_err(|e| error(StatusCode::SERVICE_UNAVAILABLE, "busy", format!("{e:#}")))?)
    } else {
        None
    };
    let mon = s.mgr.monitor.clone();
    let id = mon.media_begin(kind, model);
    let row = crate::monitor::MediaRow::new(&mon, id);
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
        Ok(v) => Ok((v, device, row)),
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
    if !crate::api_authorized(&s, &headers) {
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
    let (r, device, row) = match job(&s, kind, &model, m["size_gb"].as_f64().unwrap_or(1.0), cmd, |_| {}).await {
        Ok(x) => x,
        Err(r) => return r,
    };
    let id = row.id;
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

/// A chat's audio attachment as text: the speech-to-text model on the CPU (no GPU turn, so the answer's text model
/// stays loaded). Returns the text and the audio's length in seconds.
pub async fn transcribe_for_chat(s: &AppState, b64: &str) -> Result<(String, f64), String> {
    if !s.media.available() {
        return Err("the media engine is not built".into());
    }
    let model = s.media.stt_model();
    let m = audio_model(s, &model, "stt").await.map_err(|_| format!("{model} is not downloaded (Settings > Models > Media models > Audio)"))?;
    let mon = s.mgr.monitor.clone();
    let id = mon.media_begin("speech-to-text", &model);
    let _row = crate::monitor::MediaRow::new(&mon, id);
    let r = s.media.call(json!({"op": "transcribe", "model": model, "audio": b64, "translate": false, "device": "cpu"}), |_| {}).await;
    if mem_available_gb() < m["size_gb"].as_f64().unwrap_or(1.0) * 2.0 + 2.0 {
        let _ = s.media.call(json!({"op": "unload_audio"}), |_| {}).await;
    }
    match r {
        Ok(v) => {
            let text = v["text"].as_str().unwrap_or("").to_string();
            let secs = v["duration"].as_f64().unwrap_or(0.0);
            mon.media_end(id, "stop", &format!("{secs:.1} s of audio from a chat, {} words, on the cpu", text.split_whitespace().count()));
            Ok((text, secs))
        }
        Err(e) => {
            mon.media_end(id, "error", &format!("{e:#}"));
            Err(format!("{e:#}"))
        }
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
    if !crate::api_authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if !s.media.available() {
        return error(StatusCode::NOT_IMPLEMENTED, "not_available", "the media engine is not built");
    }
    let Some(text) = req["input"].as_str().filter(|t| !t.trim().is_empty()) else {
        return bad("send {\"input\": \"the text to speak\"}");
    };
    // OpenAI stops at 4,096 characters (about 4.5 minutes); speech is made part by part here, so a longer text only
    // takes longer: 20,000 characters is about 20 minutes of speech
    if text.chars().count() > MAX_SPEECH_CHARS {
        return bad(format!("input: at most {MAX_SPEECH_CHARS} characters (about 20 minutes of speech)"));
    }
    if let Some(f) = req["response_format"].as_str().filter(|f| *f != "wav") {
        return bad(format!("response_format {f:?}: this server makes wav (WAVE, 16-bit PCM) only"));
    }
    let mut model = openai_name(req["model"].as_str(), s.media.tts_model(), &["tts-1", "tts-1-hd"]);
    let mut m = match audio_model(&s, &model, "tts").await {
        Ok(m) => m,
        Err(r) => return r,
    };
    // The language: the one asked for, or found from the text ("auto", or none given). A model that does not speak
    // it is swapped for a downloaded one that does, unless the caller named the model.
    // the web app names its picker's model and allows the swap ("switch_for_language": true)
    let named = req["model"].as_str().is_some_and(|n| n == model) && req["switch_for_language"] != true;
    let asked = req["language"].as_str().map(|l| l.trim().to_lowercase()).filter(|l| !l.is_empty() && l != "auto");
    let language = asked.clone().unwrap_or_else(|| crate::langdetect::detect(text).to_string());
    let speaks = |m: &Value, l: &str| match m["languages"].as_array() {
        Some(a) => a.iter().any(|x| x["code"] == l),
        None => l == "en",   // Chatterbox Turbo and Parler speak English
    };
    let detected = language.clone();
    let mut language = language;
    if !speaks(&m, &language) {
        let list = s.media.list().await.unwrap_or_default();
        let other = list["models"].as_array().and_then(|a| a.iter().find(|x| x["kind"] == "audio" && x["task"] == "tts"
            && x["downloaded"] == true && speaks(x, &language)).cloned());
        let name = language_name(&language);
        match other {
            Some(o) if !named => {
                model = o["name"].as_str().unwrap_or_default().to_string();
                m = o;
            }
            // only found, not asked for: read as English, as before, and the header says what was found
            _ if asked.is_none() => language = "en".into(),
            Some(_) => return bad(format!("{model} does not speak {name}; leave out \"model\", or name {}", "chatterbox-multilingual")),
            None => {
                let any = list["models"].as_array().and_then(|a| a.iter().find(|x| x["kind"] == "audio" && speaks(x, &language)).cloned());
                return bad(match any {
                    Some(x) => format!("{} needs {} ({} GB): download it in Settings > Models > Media models > Audio",
                                       name, x["name"].as_str().unwrap_or("?"), x["size_gb"]),
                    None => format!("no speech model here speaks {name} yet"),
                });
            }
        }
    }
    // OpenAI's voice names map to this model's voices; a sentence describing a voice is taken as it is
    let voice = req["voice"].as_str().unwrap_or("laura");
    let voice = match voice {
        "alloy" | "nova" | "shimmer" | "coral" | "sage" => "laura",
        "echo" | "onyx" | "ash" => "jon",
        "fable" | "ballad" => "lea",
        v => v,
    };
    let mut cmd = json!({"op": "speak", "model": model, "text": text, "voice": voice, "seed": req["seed"].as_u64().unwrap_or(0),
                         "runtime": m["runtime"]});
    if let Some(t) = req["temperature"].as_f64() {
        cmd["temperature"] = json!(t.clamp(0.0, 2.0)); // Parler samples at 0.7 unless told (0: greedy)
    }
    if m["family"] == "chatterbox" {
        // the voice of a clip (base64, or a data URL) that the caller has the speaker's consent to use
        if let Some(clip) = req["voice_audio"].as_str().filter(|c| !c.is_empty()) {
            let b64 = clip.split_once(";base64,").map(|(_, b)| b).unwrap_or(clip);
            if b64.len() > 8 * 1024 * 1024 {
                return bad("voice_audio: at most 6 MB (5 to 10 seconds of one person speaking is enough)");
            }
            cmd["voice_audio"] = json!(b64);
        }
        if m["languages"].is_array() {
            cmd["language"] = json!(language);
        }
        if let Some(x) = req["exaggeration"].as_f64() {
            cmd["exaggeration"] = json!(x.clamp(0.0, 2.0));
        }
        cmd["threads"] = json!(std::thread::available_parallelism().map(|n| (n.get() / 2).max(1)).unwrap_or(4));
        // each part is heard by the speech-to-text model and made again when it does not match its text (some takes
        // add or drop words); "check": false skips it
        if req["check"] != false {
            let stt = s.media.stt_model();
            if let Ok(w) = audio_model(&s, &stt, "stt").await {
                cmd["check_model"] = json!(stt);
                cmd["check_gb"] = w["size_gb"].clone();
            }
        }
    } else if req["voice_audio"].is_string() {
        return bad(format!("voice_audio: {model} does not clone a voice; Chatterbox does"));
    } else if m["languages"].is_array() {
        // a multilingual Parler (Indic Parler: its named speaker for the language and the voice's gender)
        cmd["language"] = json!(language);
    }
    let (r, device, row) = match job(&s, "text-to-speech", &model, m["size_gb"].as_f64().unwrap_or(3.5), cmd, |_| {}).await {
        Ok(x) => x,
        Err(r) => return r,
    };
    let id = row.id;
    let wav = match base64::engine::general_purpose::STANDARD.decode(r["data"].as_str().unwrap_or("")) {
        Ok(b) => b,
        Err(_) => return error(StatusCode::BAD_GATEWAY, "audio_error", "the media engine returned no audio"),
    };
    s.mgr.monitor.media_file_add(id, "wav", wav.clone());
    s.mgr.monitor.media_end(id, "stop", &format!("{:.1} s of speech, {} characters, on the {device}", r["seconds"].as_f64().unwrap_or(0.0),
                                          text.chars().count()));
    Response::builder().header(header::CONTENT_TYPE, "audio/wav")
        .header("x-strata-seconds", r["seconds"].to_string()).header("x-strata-model", model).header("x-strata-language", language).header("x-strata-language-found", detected)
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
