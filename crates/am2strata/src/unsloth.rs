//! What `unsloth start` asks a server before it points a coding agent at it. With UNSLOTH_STUDIO_URL and
//! UNSLOTH_API_KEY set, the agents then use /v1 here as any client does. Unsloth Studio's Connections page needs none
//! of this: it only uses /v1/models and /v1/chat/completions.
//!
//! Unsloth's automatic key is left out on purpose: its /api/auth/identity proof takes the secret of the user's own
//! Unsloth install, so an agent gets this server's key from UNSLOTH_API_KEY or --api-key.

use crate::manager::ModelEntry;
use crate::{authorized, error, AppState};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

/// The `service` unsloth_cli's find_studio_server looks for: any other value and the CLI says no server is running.
const SERVICE: &str = "Unsloth UI Backend";

/// GET /api/health. It answers to no key, as Unsloth's does: the CLI asks before it has one.
pub async fn health() -> Json<Value> {
    Json(json!({"status": "healthy", "service": SERVICE, "server": "am2Strata"}))
}

/// The model a request runs on: the loaded one, else the default one, which the first request loads again.
async fn resident(s: &AppState) -> Option<ModelEntry> {
    match s.mgr.current().await {
        Some(l) => Some(l.entry.clone()),
        None => s.mgr.entry(&s.mgr.default_id),
    }
}

/// GET /api/inference/loaded-models. `context_length` becomes the agent's context window.
pub async fn loaded_models(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let data: Vec<Value> = resident(&s).await.iter()
        .map(|m| json!({"id": m.id, "object": "model", "owned_by": "am2strata", "loaded": true,
                        "context_length": m.cfg.max_context()}))
        .collect();
    Json(json!({"object": "list", "data": data})).into_response()
}

/// GET /api/inference/status. `is_gguf` opens Codex, which Unsloth keeps to llama-server's GGUF models because its
/// other back end has no Responses API; this server has one for every model.
pub async fn status(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let m = resident(&s).await;
    let id = m.as_ref().map(|m| m.id.clone());
    Json(json!({"active_model": id, "model_identifier": id, "is_gguf": true,
                "context_length": m.as_ref().map(|m| m.cfg.max_context())})).into_response()
}

/// POST /api/inference/load, for `unsloth start --model`. The model's run config decides how it loads, so a setting
/// this load would not apply is refused rather than dropped.
pub async fn load(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let Some(name) = req.get("model_path").and_then(Value::as_str) else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "send {\"model_path\": \"<name>\"}");
    };
    let knobs = unknown_settings(&req);
    if !knobs.is_empty() {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error",
                     format!("am2Strata takes these from the model's run config, so leave them out or change them in Settings: {}", knobs.join(", ")));
    }
    let before = s.mgr.current().await.map(|l| l.entry.id.clone());
    match s.mgr.load(name).await {
        Ok(l) => {
            let done = if before.as_deref() == Some(l.entry.id.as_str()) { "already_loaded" } else { "loaded" };
            Json(json!({"status": done, "model": l.entry.id, "display_name": l.entry.display_name()})).into_response()
        }
        Err(e) => error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("{e:#}")),
    }
}

/// The load settings a request really sets. load_in_4bit is about models this server does not run.
fn unknown_settings(req: &Value) -> Vec<&str> {
    let unset = |v: &Value| v.is_null() || *v == json!(false) || *v == json!(0) || *v == json!("");
    let mut knobs: Vec<&str> = req.as_object().into_iter().flatten()
        .filter(|(k, v)| !matches!(k.as_str(), "model_path" | "load_in_4bit") && !unset(v))
        .map(|(k, _)| k.as_str())
        .collect();
    knobs.sort_unstable();
    knobs
}
