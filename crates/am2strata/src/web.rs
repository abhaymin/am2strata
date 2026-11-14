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
        "status": "ok", "alerts": s.mgr.monitor.alerts().len(), "model": model, "loaded": cur.is_some(),
        "images": cur.as_ref().is_some_and(|l| l.vision.is_some()),
        "video": crate::media::ffmpeg_available(),
        "image_generation": s.media.available(),
        "hardware": s.mgr.hw,
        "adapted": cur.as_ref().map(|l| l.adapted.clone()).unwrap_or_default(),
        "server": "am2Strata",
    }))
}

pub async fn list_models(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !crate::authorized_or_master(&s, &headers) {
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
    // the request history: a page of it (newest first), or all of it with ?requests=all
    let num = |k: &str| q.get(k).and_then(|v| v.parse::<usize>().ok());
    let page = if q.get("requests").is_some_and(|v| v == "all") { None }
               else { Some((num("page").unwrap_or(1), num("per").unwrap_or(crate::monitor::PAGE_DEFAULT))) };
    let mut m = s.mgr.monitor.metrics(engine, loaded, page);
    m["queue"] = s.mgr.queue.snapshot();
    m["prompt_snapshots"] = s.mgr.snapshots.stats();
    m["version"] = json!(env!("CARGO_PKG_VERSION"));
    Json(m).into_response()
}

/// POST /api/monitor/clear: the Monitor's request history, its file and the media it kept. Only from this server's
/// own page (or a trusted origin), as Settings.
pub async fn monitor_clear(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("the Monitor's history can be cleared");
    }
    Json(json!({"cleared": s.mgr.monitor.clear()})).into_response()
}

/// GET /api/monitor/media/<name>: an image, video or speech a media job made, as the Monitor's history lists it.
pub async fn monitor_media(State(s): State<AppState>, headers: HeaderMap, axum::extract::Path(name): axum::extract::Path<String>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    match s.mgr.monitor.media_file(&name) {
        Some((kind, bytes)) => ([(axum::http::header::CONTENT_TYPE, kind), (axum::http::header::CACHE_CONTROL, "private, max-age=86400")],
                                bytes).into_response(),
        None => error(StatusCode::NOT_FOUND, "not_found_error", "no such media in the Monitor's history"),
    }
}

/// GET /api/agents[?url=&model=&small=]: the coding agents, what each supports and what to enter, pointed at this
/// server (the address the page was opened at) or at `url`; the model is the loaded one unless `model` names one.
/// The guides name the key's variable, never the key.
pub async fn agents(State(s): State<AppState>, headers: HeaderMap, Query(q): Query<HashMap<String, String>>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let host = headers.get("host").and_then(|v| v.to_str().ok()).unwrap_or("127.0.0.1:8000");
    let url = q.get("url").filter(|u| u.starts_with("http://") || u.starts_with("https://")).cloned()
        .unwrap_or_else(|| format!("http://{host}"));
    let list = s.mgr.list().await;
    let current = list["models"].as_array().and_then(|a| a.iter().find(|m| m["current"] == true)).and_then(|m| m["id"].as_str())
        .map(str::to_string).unwrap_or_else(|| s.mgr.default_id.clone());
    let model = q.get("model").filter(|m| !m.is_empty()).cloned().unwrap_or(current);
    let remote = q.contains_key("url");
    // a remote server's key is the user's to know; this one's: whether it has one
    let key = (remote || s.api_key.as_deref().is_some_and(|k| !k.is_empty())).then(|| "set".to_string());
    let t = crate::agents::Target { url: url.trim_end_matches('/').to_string(), key, model, small: q.get("small").filter(|m| !m.is_empty()).cloned(),
                                    tools: q.get("tools").filter(|t| !t.is_empty()).cloned() };
    Json(json!({"url": t.url, "model": t.model, "key_var": crate::agents::KEY_VAR, "agents": crate::agents::list(&t)})).into_response()
}

/// GET /api/queue: the job that has the GPU's turn and the ones waiting, in the order they will run.
pub async fn queue(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !crate::authorized_or_master(&s, &headers) {
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

/// POST /api/cache/snapshots/clear: every prompt snapshot on the SSD goes (this server's own page only).
pub async fn clear_snapshots(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("the caches can be cleared");
    }
    let _p = s.mgr.queue.enter(crate::queue::Need::Any, "clear", "snapshots", |_| {}).await;
    Json(json!({"freed_bytes": s.mgr.snapshots.clear()})).into_response()
}

/// GET /api/system: this machine for any client: every GPU, the CPU, the memory (each with its load now) and any NPU or TPU.
pub async fn system(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let mon = s.mgr.monitor.clone();
    // a load that ran without a card of its config is a warning too (adapt() dropped it)
    let dropped: Vec<String> = s.mgr.current().await.map(|l| l.adapted.iter().filter(|n| n.contains("are not available")).cloned().collect()).unwrap_or_default();
    match tokio::task::spawn_blocking(move || mon.system()).await {
        Ok(mut v) => {
            if let Some(a) = v["alerts"].as_array_mut() {
                a.extend(dropped.into_iter().map(Value::String));
            }
            Json(v).into_response()
        }
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string()),
    }
}

/// The machine's facts for the optimizer (OPTIMIZER-PLAN step 1): CPU, RAM, GPUs and their links, the models' disk,
/// and what was measured on this hardware (GET); POST /api/machine/measure measures the disk's read speed.
pub async fn machine(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let (d, m) = (s.data_dir.clone(), s.models_dir.clone());
    match tokio::task::spawn_blocking(move || crate::machine::view(&d, &m)).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string()),
    }
}

pub async fn machine_measure(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let (d, m) = (s.data_dir.clone(), s.models_dir.clone());
    // the models' files are where their run configs' --model says, often not beside the configs
    let mut weights: Vec<std::path::PathBuf> = s.mgr.models().iter()
        .filter_map(|e| crate::optimizer::arg(&e.cfg.args, "--model"))
        .filter_map(|p| std::path::Path::new(&p).parent().and_then(|d| d.parent()).map(|d| d.to_path_buf()))
        .collect();
    weights.sort();
    weights.dedup();
    match tokio::task::spawn_blocking(move || crate::machine::measure(&d, &m, &weights)).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string()),
    }
}

/// The optimizer (OPTIMIZER-PLAN step 2): POST {"objective": "decode"|"prefill", "budget_s": 1200} tunes a model in
/// the background (it holds the turn, so chats wait for it, told in the queue); GET /api/optimise follows it.
pub async fn optimise(State(s): State<AppState>, headers: HeaderMap, axum::extract::Path(id): axum::extract::Path<String>,
                      body: Option<Json<Value>>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if s.mgr.entry(&id).is_none() {
        return error(StatusCode::NOT_FOUND, "not_found", format!("no installed model is named {id:?}"));
    }
    if matches!(crate::optimizer::JOB.lock().unwrap()["state"].as_str(), Some("waiting" | "running")) {
        return error(StatusCode::CONFLICT, "busy", "a model is being tuned already: see GET /api/optimise");
    }
    let b = body.map(|b| b.0).unwrap_or(Value::Null);
    let objective = match b["objective"].as_str().unwrap_or("decode") {
        o @ ("decode" | "prefill") => o.to_string(),
        o => return error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("objective {o:?}: decode or prefill")),
    };
    let budget = std::time::Duration::from_secs(b["budget_s"].as_u64().unwrap_or(1200).clamp(60, 4 * 3600));
    *crate::optimizer::JOB.lock().unwrap() = json!({"state": "waiting", "model": id});
    tokio::spawn(crate::optimizer::run(s.mgr.clone(), id.clone(), objective, budget));
    (StatusCode::ACCEPTED, Json(json!({"state": "waiting", "model": id, "status": "/api/optimise"}))).into_response()
}

pub async fn optimise_status(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let j = crate::optimizer::JOB.lock().unwrap().clone();
    Json(if j.is_null() { json!({"state": "idle"}) } else { j }).into_response()
}

/// {"enabled": bool}: use a model's tuned settings from its next load, or not (the profile is kept).
pub async fn set_tuned(State(s): State<AppState>, headers: HeaderMap, axum::extract::Path(id): axum::extract::Path<String>,
                       Json(b): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let Some(e) = s.mgr.entry(&id) else {
        return error(StatusCode::NOT_FOUND, "not_found", format!("no installed model is named {id:?}"));
    };
    let Some(on) = b["enabled"].as_bool() else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "send {\"enabled\": true|false}");
    };
    match crate::optimizer::set_enabled(&e.path, on) {
        Ok(()) => Json(json!({"model": id, "tuned": crate::optimizer::view(&e.path), "note": "used from the model's next load"})).into_response(),
        Err(err) => error(StatusCode::NOT_FOUND, "not_found", format!("{err:#}")),
    }
}

/// The GPUs am2strata-hybrid can put weights on here (its --list-devices, run once): their names are what
/// "--devices" takes in a run config (e.g. "cuda:0=12,hip:0=6").
pub async fn gpus(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    // a list is kept; a failure is not (the engine may be built after the server started)
    static LIST: std::sync::Mutex<Option<Value>> = std::sync::Mutex::new(None);
    if let Some(v) = LIST.lock().unwrap().clone() {
        return Json(v).into_response();
    }
    let v = (|| async {
        let exe = std::env::current_exe().ok().and_then(|e| e.ancestors().skip(1).take(5)
            .map(|d| d.join("crates/strata-hybrid/target/release/am2strata-hybrid")).find(|p| p.is_file()));
        let Some(exe) = exe else { return json!({"devices": [], "note": "am2strata-hybrid is not built (tools/build-hybrid.sh)"}) };
        let out = tokio::time::timeout(std::time::Duration::from_secs(60),
                                       tokio::process::Command::new(&exe).arg("--list-devices").output()).await;
        match out {
            Ok(Ok(o)) => match serde_json::from_slice::<Value>(&o.stdout) {
                Ok(list) => json!({"devices": list, "engine": exe}),
                Err(_) => json!({"devices": [], "note": "am2strata-hybrid did not list its devices"}),
            },
            _ => json!({"devices": [], "note": "am2strata-hybrid did not answer in 60 s"}),
        }
    })().await;
    if v["devices"].as_array().is_some_and(|a| !a.is_empty()) {
        *LIST.lock().unwrap() = Some(v.clone());
    }
    Json(v).into_response()
}

/// GET /api/clients: the run config's client profiles (keys masked), for Settings › Clients.
pub async fn clients(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    Json(s.clients.view()).into_response()
}

/// GET /api/tls: HTTPS as it is now — on or off, mode, port, the certificate (names, issuer, expiry, SHA-256), the
/// addresses to open, Tailscale's view of this PC, and the last error. Settings › Secure connection and
/// `am2strata-run tls status` show this.
pub async fn tls_get(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    match crate::server_tls::runtime() {
        Some(rt) => Json(rt.status().await).into_response(),
        None => error(StatusCode::SERVICE_UNAVAILABLE, "server_error", "the server is still starting"),
    }
}

/// POST /api/tls {"action": "self-signed" | "load" | "tailscale" | "renew" | "off", "https_port"?, "http"?,
/// "cert_pem"?, "key_pem"? (for load)}: change HTTPS now, without a restart, and keep it in the run config's "tls"
/// block for the next start. Nothing changes when it fails (the error says why).
pub async fn tls_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("HTTPS can be changed");
    }
    let Some(rt) = crate::server_tls::runtime() else {
        return error(StatusCode::SERVICE_UNAVAILABLE, "server_error", "the server is still starting");
    };
    let p = config_path(&s).await;
    let mut cfg = match &p {
        Some(p) => match read_config(p) {
            Ok(c) => c,
            Err(r) => return r,
        },
        None => json!({}),
    };
    let current = cfg.get("tls").cloned().filter(|t| t.is_object()).unwrap_or_else(|| json!({}));
    let mut block = current.clone();
    for k in ["https_port", "http", "hsts", "hsts_max_age", "hsts_include_subdomains"] {
        if let Some(v) = req.get(k) {
            block[k] = v.clone();
        }
    }
    let action = req["action"].as_str().unwrap_or("");
    match action {
        "off" => {
            let _ = rt.apply(None).await;
            if let Some(o) = cfg.as_object_mut() {
                o.remove("tls");
            }
        }
        "self-signed" | "tailscale" | "renew" | "load" => {
            match action {
                "self-signed" => { block["mode"] = json!("self-signed"); block.as_object_mut().unwrap().retain(|k, _| k != "cert" && k != "key"); }
                "tailscale" => { block["mode"] = json!("tailscale"); block.as_object_mut().unwrap().retain(|k, _| k != "cert" && k != "key"); }
                "load" => {
                    let (Some(cert), Some(key)) = (req["cert_pem"].as_str(), req["key_pem"].as_str()) else {
                        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "load needs \"cert_pem\" and \"key_pem\" (PEM text)");
                    };
                    match crate::server_tls::save_uploaded(&s.data_dir, cert, key) {
                        Ok((c, k)) => {
                            block["mode"] = json!("files");
                            block["cert"] = json!(c.display().to_string());
                            block["key"] = json!(k.display().to_string());
                        }
                        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
                    }
                }
                _ => {
                    if current.as_object().is_none_or(|o| o.is_empty()) {
                        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "HTTPS is off; nothing to renew");
                    }
                }
            }
            let mut probe = std::collections::BTreeMap::new();
            probe.insert("tls".to_string(), block.clone());
            let tcfg = match crate::server_tls::Config::from_config(&probe) {
                Ok(c) => c,
                Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
            };
            if let Err(e) = rt.apply(tcfg).await {
                return error(StatusCode::BAD_REQUEST, "invalid_request_error", e);
            }
            cfg["tls"] = block;
        }
        _ => return error(StatusCode::BAD_REQUEST, "invalid_request_error",
                          "\"action\" must be self-signed, load, tailscale, renew or off"),
    }
    let mut out = rt.status().await;
    if let Some(p) = &p {
        match crate::settings::save(p, &cfg) {
            Ok(_) => out["saved"] = json!(p.display().to_string()),
            Err(e) => out["saved_error"] = json!(format!("applied, but the run config could not be written: {e}")),
        }
    }
    Json(out).into_response()
}
