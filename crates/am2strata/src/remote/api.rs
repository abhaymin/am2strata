//! The owner's page and API for remote models: `GET /rtenai` (the page, linked from nowhere) and `/api/rtenai/...`
//! (the server's own key only; writes from this server's own page only). The page is served while the feature is off
//! too, so the owner can turn it on; its API then says it is off and lets the owner write the block.

use super::config::{self, Config};
use super::get;
use crate::AppState;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

const PAGE: &str = include_str!("page.html");

/// GET /rtenai: the page. It shows nothing of the providers until the owner's key is accepted by its API.
pub async fn page() -> Response {
    ([(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8"), (axum::http::header::CACHE_CONTROL, "no-store")], PAGE).into_response()
}

/// Only the server's own key (a client profile's or a generated key gets 401, as for Settings).
fn owner(s: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    let other = crate::clients::given_key(headers).is_some_and(|k| s.clients.by_key(k).is_some());
    if !crate::authorized(s, headers) || other {
        return Err(crate::error(StatusCode::UNAUTHORIZED, "authentication_error", "the server's own API key is needed"));
    }
    Ok(())
}

fn owner_write(s: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    owner(s, headers)?;
    if crate::foreign_origin(s, headers) {
        return Err(crate::not_own_page("remote models can be changed"));
    }
    Ok(())
}

fn off() -> Response {
    crate::error(StatusCode::NOT_FOUND, "not_found", "remote models are off (no \"remote\" block in the run config)")
}

fn view(c: Option<&Config>) -> Value {
    let Some(r) = get() else { return json!({"enabled": false}) };
    let Some(c) = c else { return json!({"enabled": false, "config": r.config_path.as_ref().map(|p| p.display().to_string())}) };
    let providers: Vec<Value> = c.providers.iter().map(|p| json!({
        "name": p.name, "kind": p.kind.name(), "base_url": p.base_url, "models": p.models, "max_concurrent": p.max_concurrent,
        "kinds": p.kinds, "key": config::key_view(p, &r.keys)})).collect();
    let mut block = c.to_value();
    // a key written into the run config as it is never leaves the server: only env:/file: references are shown
    for (_, p) in block["providers"].as_object_mut().into_iter().flatten() {
        if p["key"].as_str().is_some_and(|k| !k.starts_with("env:") && !k.starts_with("file:")) {
            p["key"] = json!("(in the run config)");
        }
    }
    json!({"enabled": true, "config": r.config_path.as_ref().map(|p| p.display().to_string()), "providers": providers,
           "mix": block["mix"], "mixes": block.get("mixes").cloned().unwrap_or(json!({})), "block": block})
}

/// GET /api/rtenai: on or off, the providers (keys as set / not set and their last 4 characters), the mixes.
pub async fn status(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(r) = owner(&s, &headers) {
        return r;
    }
    Json(view(get().and_then(|r| r.config()).as_ref())).into_response()
}

/// POST /api/rtenai {"remote": {...}} writes the run config's block (and turns the feature on); {"remote": null}
/// removes it (off). A provider's "key" may only be "env:NAME" or "file:PATH" here; left out, the one it had stays.
pub async fn save(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if let Err(r) = owner_write(&s, &headers) {
        return r;
    }
    let Some(r) = get() else { return off() };
    let bad = |m: String| crate::error(StatusCode::BAD_REQUEST, "invalid_request_error", m);
    let Some(block) = req.as_object().and_then(|o| o.get("remote")).cloned() else {
        return bad("send {\"remote\": {\"providers\": {...}, \"mix\": {...}}} (or {\"remote\": null} to turn remote models off)".into());
    };
    let new = if block.is_null() {
        None
    } else {
        if !block.is_object() {
            return bad("remote: expected an object, or null".into());
        }
        let mut block = block;
        let old = r.config();
        for (name, p) in block.get_mut("providers").and_then(Value::as_object_mut).into_iter().flatten() {
            match p.get("key") {
                None => {
                    if let Some(k) = old.as_ref().and_then(|c| c.provider(name)).and_then(|o| o.key.clone()) {
                        p["key"] = json!(k);
                    }
                }
                Some(Value::String(k)) if k.is_empty() || k.starts_with("env:") || k.starts_with("file:") => {}
                Some(Value::String(k)) if k == "(in the run config)" => {
                    if let Some(k) = old.as_ref().and_then(|c| c.provider(name)).and_then(|o| o.key.clone()) {
                        p["key"] = json!(k);
                    }
                }
                Some(Value::Null) => {}
                Some(_) => return bad(format!("remote.providers.{name}.key: give \"env:NAME\" or \"file:PATH\" here; a key itself is set with \
                                               POST /api/rtenai/keys/{name} (kept in remote-keys.json, never in the run config)")),
            }
        }
        match Config::parse(&block) {
            Ok(c) => Some(c),
            Err(e) => return bad(e),
        }
    };
    let Some(path) = r.config_path.clone() else {
        return crate::error(StatusCode::NOT_FOUND, "not_found", "this server was started without a run config");
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => return crate::error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("the run config cannot be read: {e}")),
    };
    let mut cfg = match serde_json::from_str::<Value>(text.trim_start_matches('\u{feff}')) {
        Ok(v @ Value::Object(_)) => v,
        _ => return crate::error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "the run config is not a JSON object"),
    };
    match &new {
        Some(c) => cfg["remote"] = c.to_value(),
        None => {
            cfg.as_object_mut().map(|o| o.remove("remote"));
        }
    }
    if let Err(e) = crate::settings::save(&path, &cfg) {
        return crate::error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("the run config cannot be written: {e}"));
    }
    tracing::info!("remote models: {} in {}", if new.is_some() { "settings saved" } else { "turned off" }, path.display());
    r.set_config(new);
    Json(view(r.config().as_ref())).into_response()
}

fn provider(name: &str) -> Result<(&'static std::sync::Arc<super::Remote>, config::Provider), Response> {
    let Some((r, c)) = super::on() else { return Err(off()) };
    match c.provider(name) {
        Some(p) => Ok((r, p.clone())),
        None => Err(crate::error(StatusCode::NOT_FOUND, "not_found", format!("no provider {name:?} in remote.providers"))),
    }
}

/// POST /api/rtenai/keys/{provider} {"key": "..."}: kept in <data>/remote-keys.json (0600); answered with set / not set.
pub async fn set_key(State(s): State<AppState>, headers: HeaderMap, UrlPath(name): UrlPath<String>, Json(req): Json<Value>) -> Response {
    if let Err(r) = owner_write(&s, &headers) {
        return r;
    }
    let (r, p) = match provider(&name) {
        Ok(x) => x,
        Err(e) => return e,
    };
    let Some(key) = req.get("key").and_then(Value::as_str).map(str::trim).filter(|k| !k.is_empty() && k.len() <= 4096 && !k.contains(char::is_whitespace)) else {
        return crate::error(StatusCode::BAD_REQUEST, "invalid_request_error", "send {\"key\": \"<the provider's API key>\"}");
    };
    if let Err(e) = r.keys.set(&p.name, key) {
        return crate::error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e);
    }
    tracing::info!("remote models: a key was set for {}", p.name);
    Json(json!({"provider": p.name, "key": config::key_view(&p, &r.keys)})).into_response()
}

/// DELETE /api/rtenai/keys/{provider}: the key set here is forgotten (one in the run config stays).
pub async fn remove_key(State(s): State<AppState>, headers: HeaderMap, UrlPath(name): UrlPath<String>) -> Response {
    if let Err(r) = owner_write(&s, &headers) {
        return r;
    }
    let (r, p) = match provider(&name) {
        Ok(x) => x,
        Err(e) => return e,
    };
    match r.keys.remove(&p.name) {
        Ok(true) => Json(json!({"provider": p.name, "key": config::key_view(&p, &r.keys)})).into_response(),
        Ok(false) => crate::error(StatusCode::NOT_FOUND, "not_found", format!("{}: no key was set here", p.name)),
        Err(e) => crate::error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e),
    }
}

/// POST /api/rtenai/test/{provider}: the server asks the provider for its models now (the key and the address work).
pub async fn test(State(s): State<AppState>, headers: HeaderMap, UrlPath(name): UrlPath<String>) -> Response {
    if let Err(r) = owner_write(&s, &headers) {
        return r;
    }
    let (r, p) = match provider(&name) {
        Ok(x) => x,
        Err(e) => return e,
    };
    match r.fetch_models(&p).await {
        Ok(m) => Json(json!({"provider": p.name, "ok": true, "models": m})).into_response(),
        Err(e) => Json(json!({"provider": p.name, "ok": false, "error": e})).into_response(),
    }
}

/// GET /api/rtenai/models/{provider}: the models offered (the run config's, or the provider's list kept 10 minutes).
pub async fn models(State(s): State<AppState>, headers: HeaderMap, UrlPath(name): UrlPath<String>) -> Response {
    if let Err(r) = owner(&s, &headers) {
        return r;
    }
    let (r, p) = match provider(&name) {
        Ok(x) => x,
        Err(e) => return e,
    };
    Json(json!({"provider": p.name, "models": r.models_of(&p).await, "listed": !p.models.is_empty()})).into_response()
}

/// GET /api/rtenai/usage?days=30: each provider's tokens per day and model, from the usage counts (no prices).
pub async fn usage(State(s): State<AppState>, headers: HeaderMap, Query(q): Query<HashMap<String, String>>) -> Response {
    if let Err(r) = owner(&s, &headers) {
        return r;
    }
    let Some((_, c)) = super::on() else { return Json(json!({"enabled": false, "providers": {}})).into_response() };
    let days = q.get("days").and_then(|d| d.parse::<u64>().ok()).unwrap_or(30).clamp(1, 366);
    let now = crate::now();
    let mut out: BTreeMap<String, Value> = c.providers.iter()
        .map(|p| (p.name.clone(), json!({"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "errors": 0, "by_day": []})))
        .collect();
    for i in (0..days).rev() {
        let at = now.saturating_sub(i * 86400);
        let rep = s.keys.usage.report(1, None, at);
        let day = crate::api_keys::day_of(at);
        let mut of_day: BTreeMap<String, Vec<Value>> = BTreeMap::new();
        for (model, n) in &rep.by_model {
            let Some((p, _)) = model.split_once('/') else { continue };
            let Some(row) = out.get_mut(p) else { continue };
            for (k, x) in [("requests", n.requests), ("prompt_tokens", n.prompt_tokens), ("completion_tokens", n.completion_tokens),
                           ("cached_tokens", n.cached_tokens), ("errors", n.errors)] {
                row[k] = json!(row[k].as_u64().unwrap_or(0) + x);
            }
            of_day.entry(p.to_string()).or_default().push(json!({"model": model, "requests": n.requests, "prompt_tokens": n.prompt_tokens,
                "completion_tokens": n.completion_tokens, "cached_tokens": n.cached_tokens}));
        }
        for (p, models) in of_day {
            if let Some(a) = out.get_mut(&p).and_then(|r| r["by_day"].as_array_mut()) {
                a.push(json!({"day": day, "models": models}));
            }
        }
    }
    Json(json!({"enabled": true, "days": days, "providers": out})).into_response()
}
