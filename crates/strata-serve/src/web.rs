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

/// The monitor's figures: the current model, the machine, and empty history until request tracking is ported.
pub async fn metrics(State(s): State<AppState>, Query(_q): Query<HashMap<String, String>>) -> Json<Value> {
    let cur = s.mgr.current().await;
    Json(json!({
        "engine": {"model": cur.as_ref().map(|l| l.entry.id.clone()), "loaded": cur.is_some(),
                   "context": cur.as_ref().map(|l| l.max_context)},
        "hardware": s.mgr.hw, "hardware_static": s.mgr.hw,
        "history": {}, "live": {"running": 0}, "requests": [],
    }))
}
