//! The web app (PolyStrata's serve/web files, served as they are) and the endpoints it reads: health, the installed
//! models and switching between them, shared settings, the run-config view, MCP and metrics.

use crate::{authorized, error, AppState};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::collections::HashMap;

pub async fn health(State(s): State<AppState>) -> Json<Value> {
    let cur = s.mgr.current().await;
    let model = cur.as_ref().map(|l| l.entry.display_name()).unwrap_or_else(|| {
        s.mgr.entry(&s.mgr.default_id).map(|e| e.display_name()).unwrap_or_default()
    });
    Json(json!({
        "status": "ok", "model": model, "loaded": cur.is_some(),
        "images": cur.as_ref().is_some_and(|l| l.vision.is_some()),
        "video": crate::media::ffmpeg_available(),
        "image_generation": false,
        "hardware": s.mgr.hw,
        "adapted": cur.as_ref().map(|l| l.adapted.clone()).unwrap_or_default(),
        "server": "strata-serve (Rust)",
    }))
}

pub async fn list_models(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    Json(s.mgr.list().await).into_response()
}

pub async fn current_model(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let list = s.mgr.list().await;
    let cur = list["models"].as_array().and_then(|a| a.iter().find(|m| m["current"] == true).cloned());
    Json(json!({"model": cur, "loaded": cur.is_some()})).into_response()
}

pub async fn load_model(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let Some(name) = req.get("model").and_then(Value::as_str) else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "send {\"model\": \"<name>\"}");
    };
    match s.mgr.load(name).await {
        Ok(l) => {
            let list = s.mgr.list().await;
            let m = list["models"].as_array().and_then(|a| a.iter().find(|m| m["id"] == l.entry.id.as_str()).cloned());
            Json(json!({"status": "loaded", "model": m, "adapted": l.adapted})).into_response()
        }
        Err(e) => error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("{e:#}")),
    }
}

pub async fn unload_model(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    s.mgr.unload().await;
    Json(json!({"status": "unloaded"})).into_response()
}

/// Shared chat settings: not kept by this server yet (the web app keeps them per browser).
pub async fn settings_get() -> Json<Value> {
    Json(json!({"shared": false, "defaults": null}))
}

pub async fn settings_post() -> Json<Value> {
    Json(json!({"shared": false}))
}

/// The run-config view: not offered (the web app hides its card on a 404).
pub async fn config_get() -> Response {
    error(StatusCode::NOT_FOUND, "not_found", "the run-config view is not available on this server")
}

pub async fn mcp() -> Json<Value> {
    Json(json!({"servers": [], "tools": 0}))
}

/// The engine facts the monitor and the About tab show.
async fn engine_facts(s: &AppState) -> (Value, bool) {
    let cur = s.mgr.current().await;
    let facts = match cur.as_ref() {
        Some(l) => {
            let args = &l.entry.cfg.args;
            let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
            json!({"model": l.entry.display_name(), "id": l.entry.id, "max_context": l.max_context,
                   "images": l.vision.is_some(), "kv": arg("--kv"), "kv_resident": arg("--kv-resident").and_then(|v| v.parse::<u64>().ok()),
                   "family": l.family, "adapted": l.adapted})
        }
        None => json!({"model": s.mgr.entry(&s.mgr.default_id).map(|e| e.display_name()), "max_context": null, "images": false}),
    };
    (facts, cur.is_some())
}

/// GET /metrics: the Monitor tab's figures (the running request, the last ones, totals and the machine).
pub async fn metrics(State(s): State<AppState>, headers: HeaderMap, Query(q): Query<HashMap<String, String>>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let (engine, loaded) = engine_facts(&s).await;
    let all = q.get("requests").is_some_and(|v| v == "all");
    Json(s.mgr.monitor.metrics(engine, loaded, all)).into_response()
}

/// GET /metrics/prometheus.
pub async fn prometheus(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let loaded = s.mgr.current().await.is_some();
    ([("content-type", "text/plain; version=0.0.4")], s.mgr.monitor.prometheus(loaded)).into_response()
}

/// GET /v1/status.
pub async fn status(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let (engine, loaded) = engine_facts(&s).await;
    let ctx = engine["max_context"].as_u64().map(|c| c as usize);
    let images = engine["images"].as_bool().unwrap_or(false);
    Json(s.mgr.monitor.status(engine["model"].clone(), loaded, ctx, images)).into_response()
}
