//! POST /v1/audio/music: a song with vocals from a style and lyrics (ACE-Step 1.5, in the media engine, in Rust).
//! OpenAI has no such endpoint; the fields follow /v1/audio/speech's spirit:
//! {"style" (or "caption" / "prompt"), "lyrics", "language", "duration" (s, 10 to 600), "seed", "bpm", "key",
//!  "time_signature", "steps", "response_format": "wav"} -> audio/wav (48 kHz stereo).
//! A GPU job like an image: the text model stops for it and loads again with the next chat.

use crate::{authorized, error, AppState};
use axum::{body::Body, extract::State, http::{header, HeaderMap, StatusCode}, response::Response, Json};
use serde_json::{json, Value};

pub const MODEL: &str = "ace-step-1.5-turbo";

fn bad(msg: impl Into<String>) -> Response {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
}

pub async fn music(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if !s.media.available() {
        return error(StatusCode::NOT_IMPLEMENTED, "not_available", "the media engine is not built");
    }
    let text = |k: &str| req[k].as_str().map(str::trim).filter(|t| !t.is_empty()).map(str::to_string);
    let Some(style) = text("style").or_else(|| text("caption")).or_else(|| text("prompt")) else {
        return bad("style: describe the song (genre, instruments, voice, mood), e.g. \"warm indie pop, female vocal, acoustic guitar\"");
    };
    if style.chars().count() > 512 {
        return bad("style: at most 512 characters");
    }
    let lyrics = text("lyrics").unwrap_or_else(|| "[Instrumental]".into());
    if lyrics.chars().count() > 4096 {
        return bad("lyrics: at most 4096 characters");
    }
    let duration = req["duration"].as_u64().unwrap_or(60);
    if !(10..=600).contains(&duration) {
        return bad("duration: 10 to 600 seconds (10 minutes)");
    }
    if let Some(f) = req["response_format"].as_str().filter(|f| *f != "wav") {
        return bad(format!("response_format {f:?}: this server makes wav only"));
    }
    let list = match s.media.list().await {
        Ok(l) => l,
        Err(e) => return error(StatusCode::BAD_GATEWAY, "media_error", format!("{e:#}")),
    };
    let Some(m) = list["models"].as_array().and_then(|a| a.iter().find(|m| m["name"] == MODEL).cloned()) else {
        return bad(format!("{MODEL} is not in this media engine's catalog: update the media engine"));
    };
    if m["downloaded"] != true {
        return bad(format!("{MODEL} is not downloaded: download it in Settings > Models > Media models > Music ({} GB)", m["size_gb"]));
    }
    let language = text("language").map(|l| l.to_lowercase())
        .unwrap_or_else(|| crate::langdetect::detect(&lyrics).to_string());
    let seed = req["seed"].as_u64().unwrap_or_else(|| (crate::now() ^ (std::process::id() as u64) << 20) % 4_294_967_295);
    let steps = req["steps"].as_u64().unwrap_or(8).clamp(1, 20);
    let mut cmd = json!({"op": "song", "caption": style, "lyrics": lyrics, "language": language, "duration": duration,
                         "seed": seed, "steps": steps, "device": "cuda"});
    for k in ["bpm", "key", "time_signature"] {
        if !req[k].is_null() {
            cmd[k] = req[k].clone();
        }
    }
    let t0 = std::time::Instant::now();
    let label = s.media_jobs.label(&req["strata_label"]);
    let _gpu = match s.mgr.gpu_exclusive_as(!s.media.keep_text_model, "media", &label).await {
        Ok(g) => g,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, "busy", format!("{e:#}")),
    };
    let mon = s.mgr.monitor.clone();
    let job = mon.media_begin("song", MODEL);
    let _row = crate::monitor::MediaRow::new(&mon, job);
    let res = s.media.call(cmd, |ev| {
        if ev["event"] == "step" {
            mon.media_step(job, ev["step"].as_u64().unwrap_or(0), ev["total"].as_u64().unwrap_or(steps));
        }
    }).await;
    if let Err(e) = s.media.call(json!({"op": "unload"}), |_| {}).await {
        tracing::warn!("the media engine did not unload: {e:#}");
    }
    let v = match res {
        Ok(v) => v,
        Err(e) => {
            mon.media_end(job, "error", &format!("{e:#}"));
            return error(StatusCode::BAD_GATEWAY, "music_error", format!("{e:#}"));
        }
    };
    let wav = {
        use base64::Engine as _;
        match base64::engine::general_purpose::STANDARD.decode(v["data"].as_str().unwrap_or("")) {
            Ok(b) => b,
            Err(_) => return error(StatusCode::BAD_GATEWAY, "music_error", "the media engine returned no audio"),
        }
    };
    mon.media_file_add(job, "wav", wav.clone());
    mon.media_end(job, "stop", &format!("{duration} s song, {language}, seed {seed}, {steps} steps"));
    Response::builder().header(header::CONTENT_TYPE, "audio/wav")
        .header("x-strata-seconds", duration.to_string()).header("x-strata-model", MODEL)
        .header("x-strata-language", language).header("x-strata-seed", seed.to_string())
        .header("x-strata-ms", t0.elapsed().as_millis().to_string())
        .body(Body::from(wav)).unwrap()
}
