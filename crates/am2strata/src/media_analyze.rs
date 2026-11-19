//! Music facts of a recording that a producer needs to learn before making a matching take — tempo, the drone's key and
//! home note (the tonic), the notes the voice sits on (the swaras) with their share, how long each phrase holds. This is
//! not the analysis itself: the endpoint only forwards it to the media engine's `analyze` op (a CPU job, no model, exactly
//! like `/v1/audio/transcriptions` forwards `transcribe`) and returns what the engine reports (`tempo`, `repeat`,
//! `pitch.tonic` + `range` + `swaras`, `phrases`). It reads a file in any format the engine decodes (MP3, WAV, FLAC, AAC,
//! OGG) from a multipart `file` or a base64 / data URL in JSON `"audio"`, and takes an optional window (`from_s`, `seconds`).

use crate::{api_authorized, error, AppState};
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine as _;
use serde_json::json;

/// The largest audio this endpoint forwards to the engine, in bytes (200 MiB).
const MAX_BYTES: usize = 200 * 1024 * 1024;

fn bad(msg: impl Into<String>) -> Response { error(StatusCode::BAD_REQUEST, "invalid_request_error", msg) }

/// POST /v1/media/analyze: the music facts of an audio file (a multipart `file`, or a base64 / data URL in JSON `"audio"`),
/// in any format the engine reads. It forwards the raw bytes to the media engine's `analyze` op with the optional `from_s`/
/// `seconds` window and returns the engine's analysis event, without its `event` field; at most 200 MiB are forwarded.
pub async fn analyze_endpoint(State(s): State<AppState>, headers: HeaderMap, req: Request) -> Response {
    if !api_authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if !s.media.available() {
        return error(StatusCode::NOT_IMPLEMENTED, "not_available", "the media engine is not built");
    }
    let ctype = req.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let (audio, from_s, seconds) = if ctype.starts_with("multipart/form-data") {
        // multipart: the file plus optional from_s / seconds as form fields (200 MiB audio read straight into memory)
        let Ok(mut mp) = Multipart::from_request(req, &State(s.clone())).await else { return bad("multipart/form-data could not be read") };
        let mut bytes: Option<Vec<u8>> = None;
        let (mut from_s, mut seconds) = (0.0f64, 0.0f64);
        while let Ok(Some(f)) = mp.next_field().await {
            match f.name() {
                Some("file") => match f.bytes().await {
                    Ok(b) => { bytes = Some(b.to_vec()); }
                    Err(e) => return bad(&format!("the file could not be read: {e}")),
                },
                Some("from_s") => if let Some(v) = f.text().await.unwrap_or_default().parse::<f64>().ok() { from_s = v; },
                Some("seconds") => if let Some(v) = f.text().await.unwrap_or_default().parse::<f64>().ok() { seconds = v; },
                _ => {}
            }
        }
        let bytes = match bytes { Some(b) => b, None => return bad("send the audio as the multipart field `file`") };
        (bytes, from_s, seconds)
    } else {
        // JSON: {"audio": <base64|data URL>, "from_s"?, "seconds"?} — the same base64 transport /v1/media/check uses.
        let Json(body) = match Json::<serde_json::Value>::from_request(req, &State(s.clone())).await {
            Ok(b) => b,
            Err(_) => return bad("send JSON with a base64 \"audio\", or multipart `file`"),
        };
        let src = body["audio"].as_str().filter(|s| !s.is_empty()).unwrap_or("");
        let raw = src.split_once(";base64,").map(|(_, b)| b).unwrap_or(src);
        let bytes = match base64::engine::general_purpose::STANDARD.decode(raw.trim()) {
            Ok(b) => b,
            Err(_) => return bad("{\"audio\": is required (base64 or a data: URL, or multipart `file`)"),
        };
        (bytes, body["from_s"].as_f64().unwrap_or(0.0), body["seconds"].as_f64().unwrap_or(0.0))
    };

    if audio.is_empty() { return bad("send the audio as \"file\""); }
    if audio.len() > MAX_BYTES { return bad("an audio of at most 200 MiB"); }

    let cmd = json!({"op": "analyze", "audio": base64::engine::general_purpose::STANDARD.encode(&audio),
                     "from_s": from_s, "seconds": seconds});
    match s.media.call(cmd, |_| {}).await {
        Ok(mut v) => {
            // the engine's `analysis` event, returned without its `event` field
            if let Some(o) = v.as_object_mut() { o.remove("event"); }
            Json(v).into_response()
        }
        Err(e) => error(StatusCode::BAD_GATEWAY, "media_error", format!("{e:#}")),
    }
}
