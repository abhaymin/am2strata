//! The API monitor (/api-monitor, GET /api/requests, POST /load and /unload): the last 100 API calls with their
//! request bodies, the model's output and reasoning, usage, timings and outcome, kept in memory only (no headers or
//! API keys). Off unless the config's "api_monitor" (or --api-monitor) turns it on. Port of serve/server.py #332.
//!
//! One layer around /v1/chat/completions, /v1/messages and /v1/responses records each call as it passes: the
//! request body before the handler, then the response as it streams out (SSE events of any of the three APIs, or
//! the JSON answer), so the handlers know nothing of it.

use crate::{authorized, error, AppState};
use axum::body::{Body, Bytes};
use axum::extract::{Query, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const KEEP: usize = 100;
const CAP: usize = 262144;

struct Record {
    v: Map<String, Value>,
    clock: Instant,
}

#[derive(Default)]
pub struct ApiMonitor {
    pub enabled: bool,
    records: Mutex<VecDeque<Record>>,
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

fn capped(s: &str) -> (String, bool) {
    if s.len() <= CAP {
        return (s.to_string(), false);
    }
    let mut end = CAP;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    (s[..end].to_string(), true)
}

impl ApiMonitor {
    pub fn new(enabled: bool) -> Arc<Self> {
        Arc::new(ApiMonitor { enabled, records: Mutex::new(VecDeque::new()) })
    }

    fn begin(&self, path: &str, body: &[u8], default_model: &str) -> String {
        let req: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
        let raw = serde_json::to_string_pretty(&req).unwrap_or_else(|_| String::from_utf8_lossy(body).into_owned());
        let (input, input_truncated) = capped(&raw);
        let id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
        let model = req.get("model").and_then(Value::as_str).filter(|m| !m.is_empty()).unwrap_or(default_model);
        let rf = req.get("response_format").and_then(|r| r.get("type")).cloned().unwrap_or(Value::Null);
        let v = json!({"id": id, "path": path, "model": model, "started_at": now(), "state": "queued",
                       "stream": req.get("stream").and_then(Value::as_bool).unwrap_or(false), "response_format": rf,
                       "input": input, "input_truncated": input_truncated, "output": "", "reasoning": "",
                       "output_truncated": false, "reasoning_truncated": false,
                       "queue_s": 0.0, "load_s": 0.0, "first_token_s": null});
        let mut g = self.records.lock().unwrap();
        g.push_back(Record { v: v.as_object().cloned().unwrap_or_default(), clock: Instant::now() });
        while g.len() > KEEP {
            g.pop_front();
        }
        id
    }

    fn update(&self, id: &str, f: impl FnOnce(&mut Map<String, Value>, Instant)) {
        let mut g = self.records.lock().unwrap();
        if let Some(r) = g.iter_mut().find(|r| r.v["id"] == id) {
            f(&mut r.v, r.clock);
        }
    }

    fn append(&self, id: &str, content: &str, reasoning: &str) {
        if content.is_empty() && reasoning.is_empty() {
            return;
        }
        self.update(id, |r, clock| {
            if r["first_token_s"].is_null() {
                r.insert("first_token_s".into(), json!((clock.elapsed().as_secs_f64() * 1000.0).round() / 1000.0));
            }
            for (name, add) in [("output", content), ("reasoning", reasoning)] {
                if add.is_empty() || r[&format!("{name}_truncated")] == true {
                    continue;
                }
                let joined = format!("{}{add}", r[name].as_str().unwrap_or(""));
                let (text, cut) = capped(&joined);
                r.insert(name.into(), json!(text));
                r.insert(format!("{name}_truncated"), json!(cut));
            }
        });
    }

    fn finish(&self, id: &str, outcome: &str) {
        self.update(id, |r, clock| {
            r.insert("wallclock_s".into(), json!((clock.elapsed().as_secs_f64() * 1000.0).round() / 1000.0));
            r.insert("finished_at".into(), json!(now()));
            let state = if r.get("error").is_some_and(|e| !e.is_null()) { "error" } else { outcome };
            r.insert("state".into(), json!(state));
        });
    }

    fn list(&self) -> Vec<Value> {
        let g = self.records.lock().unwrap();
        g.iter()
            .rev()
            .map(|r| {
                let mut v: Map<String, Value> = r.v.iter()
                    .filter(|(k, _)| !matches!(k.as_str(), "input" | "output" | "reasoning" | "response"))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                v.entry("wallclock_s").or_insert(json!((r.clock.elapsed().as_secs_f64() * 1000.0).round() / 1000.0));
                Value::Object(v)
            })
            .collect()
    }

    fn get(&self, id: &str) -> Option<Value> {
        self.records.lock().unwrap().iter().find(|r| r.v["id"] == id).map(|r| Value::Object(r.v.clone()))
    }
}

/// What one streamed event (or a whole JSON answer) adds: (content, reasoning) and the usage / timings / error it
/// carries, for the OpenAI, Anthropic and Responses shapes.
fn take(id: &str, mon: &ApiMonitor, v: &Value) {
    let (mut content, mut reasoning) = (String::new(), String::new());
    if let Some(c) = v.get("choices").and_then(|c| c.get(0)) {
        for part in ["delta", "message"] {
            if let Some(d) = c.get(part) {
                content.push_str(d.get("content").and_then(Value::as_str).unwrap_or(""));
                reasoning.push_str(d.get("reasoning_content").and_then(Value::as_str).unwrap_or(""));
            }
        }
    }
    match v.get("type").and_then(Value::as_str) {
        Some("content_block_delta") => {
            let d = &v["delta"];
            content.push_str(d.get("text").and_then(Value::as_str).unwrap_or(""));
            reasoning.push_str(d.get("thinking").and_then(Value::as_str).unwrap_or(""));
        }
        Some("response.output_text.delta") => content.push_str(v["delta"].as_str().unwrap_or("")),
        Some("response.reasoning_text.delta") => reasoning.push_str(v["delta"].as_str().unwrap_or("")),
        Some("message") if v.get("content").is_some() => {
            // an Anthropic answer that was not streamed
            for b in v["content"].as_array().into_iter().flatten() {
                content.push_str(b.get("text").and_then(Value::as_str).unwrap_or(""));
                reasoning.push_str(b.get("thinking").and_then(Value::as_str).unwrap_or(""));
            }
        }
        Some("response") => {
            for item in v["output"].as_array().into_iter().flatten() {
                for p in item["content"].as_array().into_iter().flatten() {
                    let t = p.get("text").and_then(Value::as_str).unwrap_or("");
                    if item["type"] == "reasoning" { reasoning.push_str(t) } else { content.push_str(t) }
                }
            }
        }
        _ => {}
    }
    mon.append(id, &content, &reasoning);
    let usage = v.get("usage").or_else(|| v.get("message").and_then(|m| m.get("usage")))
        .or_else(|| v.get("response").and_then(|r| r.get("usage"))).filter(|u| !u.is_null()).cloned();
    let timings = v.get("timings").filter(|t| !t.is_null()).cloned();
    let err = v.get("error").filter(|e| !e.is_null()).cloned()
        .or_else(|| v.get("response").and_then(|r| r.get("error")).filter(|e| !e.is_null()).cloned());
    if usage.is_some() || timings.is_some() || err.is_some() {
        mon.update(id, |r, _| {
            if let Some(u) = usage {
                // an Anthropic stream says input_tokens at the start and output_tokens at the end
                let merged = match (r.get("usage").and_then(Value::as_object), u.as_object()) {
                    (Some(old), Some(new)) => { let mut m = old.clone(); m.extend(new.clone()); Value::Object(m) }
                    _ => u,
                };
                r.insert("usage".into(), merged);
            }
            if let Some(t) = timings {
                r.insert("timings".into(), t);
            }
            if let Some(e) = err {
                r.insert("error".into(), e);
            }
        });
    }
}

/// The layer: records the calls of the three chat APIs while the monitor is on.
pub async fn layer(State(s): State<AppState>, req: Request, next: Next) -> Response {
    let mon = s.api_monitor.clone();
    if !mon.enabled || req.method() != axum::http::Method::POST {
        return next.run(req).await;
    }
    let path = req.uri().path().to_string();
    if !matches!(path.as_str(), "/v1/chat/completions" | "/v1/messages" | "/v1/responses" | "/v1/images/generations" | "/v1/videos/generations"
                 | "/v1/audio/transcriptions" | "/v1/audio/translations" | "/v1/audio/speech") {
        return next.run(req).await;
    }
    let (parts, body) = req.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, 256 << 20).await else {
        return error(StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error", "the request body is too large");
    };
    let multipart = parts.headers.get(axum::http::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).is_some_and(|c| c.starts_with("multipart/"));
    let default_model = if path.starts_with("/v1/audio/speech") {
        s.media.tts_model()
    } else if path.starts_with("/v1/audio") {
        s.media.stt_model()
    } else if path.starts_with("/v1/images") {
        s.media.model()
    } else if path.starts_with("/v1/videos") {
        s.media.video_model()
    } else {
        s.mgr.current().await.map(|l| l.entry.id.clone()).unwrap_or_else(|| s.mgr.default_id.clone())
    };
    // an upload is kept as a note of its size, not its bytes
    let note = multipart.then(|| serde_json::to_vec(&json!({"upload": format!("<multipart upload, {} KB>", bytes.len() / 1024)})).unwrap());
    let id = mon.begin(&path, note.as_deref().unwrap_or(&bytes), &default_model);
    let resp = next.run(Request::from_parts(parts, Body::from(bytes))).await;
    let status = resp.status().as_u16();
    mon.update(&id, |r, clock| {
        r.insert("http_status".into(), json!(status));
        r.insert("queue_s".into(), json!((clock.elapsed().as_secs_f64() * 1000.0).round() / 1000.0));
        r.insert("state".into(), json!("generating"));
    });
    let sse = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|c| c.starts_with("text/event-stream"));
    let (parts, body) = resp.into_parts();
    let audio = parts.headers.get(axum::http::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).is_some_and(|c| c.starts_with("audio/"));
    if !sse {
        // a JSON answer (or an error): kept whole; an audio answer as a note of its size
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap_or_default();
        if audio {
            mon.update(&id, |r, _| {
                r.insert("response".into(), json!(format!("<WAV audio, {} KB>", bytes.len() / 1024)));
            });
            mon.finish(&id, "completed");
            return Response::from_parts(parts, Body::from(bytes));
        }
        if let Ok(mut v) = serde_json::from_slice::<Value>(&bytes) {
            take(&id, &mon, &v);
            // an image or video answer: the record keeps its details, not the megabytes of base64
            if let Some(items) = v.get_mut("data").and_then(Value::as_array_mut) {
                for it in items {
                    for k in ["b64_json", "b64"] {
                        if let Some(b) = it.get(k).and_then(Value::as_str) {
                            let note = format!("<{}, {} KB>", if k == "b64_json" { "PNG" } else { "video" }, b.len() * 3 / 4 / 1024);
                            it[k] = json!(note);
                        }
                    }
                }
            }
            let (raw, cut) = capped(&serde_json::to_string_pretty(&v).unwrap_or_default());
            mon.update(&id, |r, _| {
                r.insert("response".into(), json!(raw));
                r.insert("response_truncated".into(), json!(cut));
            });
        }
        mon.finish(&id, "completed");
        return Response::from_parts(parts, Body::from(bytes));
    }
    // a stream: each event is read as it goes out; the record ends with the stream, or as "disconnected" when the
    // client left first (the guard is dropped with the body)
    struct Guard {
        mon: Arc<ApiMonitor>,
        id: String,
        done: bool,
        buf: String,
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            self.mon.finish(&self.id, if self.done { "completed" } else { "disconnected" });
        }
    }
    let mut guard = Guard { mon: mon.clone(), id: id.clone(), done: false, buf: String::new() };
    let stream = body.into_data_stream().map(move |chunk| {
        if let Ok(b) = &chunk {
            guard.buf.push_str(&String::from_utf8_lossy(b));
            while let Some(nl) = guard.buf.find('\n') {
                let line = guard.buf[..nl].trim_end_matches('\r').to_string();
                guard.buf.drain(..=nl);
                if let Some(data) = line.strip_prefix("data:") {
                    let data = data.trim();
                    if data == "[DONE]" {
                        guard.done = true;
                    } else if let Ok(v) = serde_json::from_str::<Value>(data) {
                        let t = v.get("type").and_then(Value::as_str).unwrap_or("");
                        if matches!(t, "message_stop" | "response.completed" | "response.incomplete" | "response.failed") {
                            guard.done = true;
                        }
                        take(&guard.id, &guard.mon, &v);
                    }
                }
            }
        }
        chunk.map(|b: Bytes| b)
    });
    Response::from_parts(parts, Body::from_stream(stream))
}

/// GET /api/requests (and ?id=): the calls kept, newest first, or one with its bodies.
pub async fn requests(State(s): State<AppState>, headers: HeaderMap, Query(q): Query<HashMap<String, String>>) -> Response {
    if !s.api_monitor.enabled {
        return error(StatusCode::NOT_FOUND, "not_found", "the API monitor is off (\"api_monitor\": true in the config, or --api-monitor)");
    }
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if let Some(id) = q.get("id") {
        return match s.api_monitor.get(id) {
            Some(v) => Json(v).into_response(),
            None => (StatusCode::NOT_FOUND, Json(json!({"error": {"message": "request no longer retained"}}))).into_response(),
        };
    }
    let loaded = s.mgr.current().await.is_some();
    Json(json!({"requests": s.api_monitor.list(), "retention": KEEP, "persistent": false, "loaded": loaded, "auto_load": true}))
        .into_response()
}

/// POST /load and /unload: the page's buttons (the default model; JSON from this server's own page only).
pub async fn load(State(s): State<AppState>, headers: HeaderMap) -> Response {
    control(s, headers, true).await
}

pub async fn unload(State(s): State<AppState>, headers: HeaderMap) -> Response {
    control(s, headers, false).await
}

async fn control(s: AppState, headers: HeaderMap, load: bool) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("the model can be loaded or unloaded");
    }
    if load {
        match s.mgr.ensure().await {
            Ok(_) => Json(json!({"status": "loaded"})).into_response(),
            Err(e) => error(StatusCode::SERVICE_UNAVAILABLE, "server_error", format!("{e:#}")),
        }
    } else {
        s.mgr.unload().await;
        Json(json!({"status": "unloaded"})).into_response()
    }
}
