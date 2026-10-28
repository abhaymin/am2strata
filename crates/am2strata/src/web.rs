//! The web app (am2Strata's web/, derived from PolyStrata's serve/web) and the endpoints it reads: health, the installed
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
        "image_generation": s.media.available(),
        "hardware": s.mgr.hw,
        "adapted": cur.as_ref().map(|l| l.adapted.clone()).unwrap_or_default(),
        "server": "am2Strata",
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
    // no body: a plain form post from another site could send it
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("the model can be unloaded");
    }
    s.mgr.unload().await;
    Json(json!({"status": "unloaded"})).into_response()
}

/// The run config the settings belong to: the loaded model's, else the default one's.
async fn config_path(s: &AppState) -> Option<std::path::PathBuf> {
    match s.mgr.current().await {
        Some(l) => Some(l.entry.path.clone()),
        None => s.mgr.entry(&s.mgr.default_id).map(|e| e.path.clone()),
    }
}

/// GET /settings: the Chat settings shared with other apps (every client gets them for what it leaves out).
pub async fn settings_get(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let d = match s.mgr.current().await {
        Some(l) => l.shared.read().unwrap().clone(),
        None => config_path(&s).await.map(|p| crate::settings::read_shared(&p)).unwrap_or_default(),
    };
    Json(json!({"shared": !d.is_empty(), "defaults": d})).into_response()
}

/// POST /settings {"defaults": {...} | null}: set (or, null, clear) the shared Chat settings.
pub async fn settings_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("settings can be changed");
    }
    let d = match crate::settings::clean_shared_defaults(req.get("defaults").unwrap_or(&Value::Null)) {
        Ok(d) => d,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    if let Some(l) = s.mgr.current().await {
        *l.shared.write().unwrap() = d.clone();
    }
    if let Some(p) = config_path(&s).await {
        crate::settings::write_shared(&p, &d);
    }
    Json(json!({"shared": !d.is_empty(), "defaults": d})).into_response()
}

fn read_config(p: &std::path::Path) -> Result<Value, Response> {
    let text = std::fs::read_to_string(p).map_err(|e| {
        error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("the run config cannot be read: {e}"))
    })?;
    match serde_json::from_str::<Value>(text.trim_start_matches('\u{feff}')) {
        Ok(v @ Value::Object(_)) => Ok(v),
        _ => Err(error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "the run config is not a JSON object")),
    }
}

/// GET /config: a few documented keys of the run config, for the Settings view.
pub async fn config_get(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let Some(p) = config_path(&s).await else {
        return error(StatusCode::NOT_FOUND, "not_found", "this server was started without a run config");
    };
    match read_config(&p) {
        Ok(cfg) => Json(crate::settings::view(&cfg, &p)).into_response(),
        Err(r) => r,
    }
}

/// POST /config {"set": {key: value | null}}: change those keys only; the earlier file is kept as .bak. Used from
/// the next start of, or switch to, the model.
pub async fn config_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("the run config can be changed");
    }
    let Some(p) = config_path(&s).await else {
        return error(StatusCode::NOT_FOUND, "not_found", "this server was started without a run config");
    };
    let cfg = match read_config(&p) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let (new, changed) = match crate::settings::apply(&cfg, req.get("set")) {
        Ok(x) => x,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    if !changed.is_empty() {
        match crate::settings::save(&p, &new) {
            Ok(bak) => tracing::info!("the Settings view changed {} in {} (the earlier file: {}); used from the next start",
                                      changed.join(", "), p.display(), bak.display()),
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("the run config cannot be written: {e}")),
        }
    }
    let mut v = crate::settings::view(&new, &p);
    v["changed"] = json!(changed);
    Json(v).into_response()
}

/// GET /mcp: the MCP servers, their state and tools (the web app's switch and Monitor card).
pub async fn mcp(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    Json(s.mcp.as_ref().map(|h| h.status()).unwrap_or(json!({"servers": [], "tools": 0}))).into_response()
}

/// The engine facts the monitor and the About tab show.
async fn engine_facts(s: &AppState) -> (Value, bool) {
    let cur = s.mgr.current().await;
    let facts = match cur.as_ref() {
        Some(l) => {
            let args = &l.entry.cfg.args;
            let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
            let mut f = json!({"model": l.entry.display_name(), "id": l.entry.id, "max_context": l.max_context,
                   "images": l.vision.is_some(), "kv": arg("--kv"), "kv_resident": arg("--kv-resident").and_then(|v| v.parse::<u64>().ok()),
                   "family": l.family, "adapted": l.adapted});
            // the engine's own start-up report (INFO k=v ...): version, expert cache, speculation, as PolyStrata shows it
            for item in l.engine.info.iter().flat_map(|line| line.split_whitespace()) {
                if let Some((k, v)) = item.split_once('=') {
                    let key = if k == "engine" { "version" } else { k };
                    let val = v.parse::<i64>().map(|n| json!(n)).or_else(|_| v.parse::<f64>().map(|x| json!(x))).unwrap_or(json!(v));
                    if f.get(key).is_none_or(|x| x.is_null()) {
                        f[key] = val;
                    }
                }
            }
            f
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
    let mut m = s.mgr.monitor.metrics(engine, loaded, all);
    m["queue"] = s.mgr.queue.snapshot();
    Json(m).into_response()
}

/// GET /api/queue: the job that has the GPU's turn and the ones waiting, in the order they will run.
pub async fn queue(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    Json(s.mgr.queue.snapshot()).into_response()
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
