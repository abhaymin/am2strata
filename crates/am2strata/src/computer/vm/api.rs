//! The sandbox VM's API, for the Computer tab and `am2strata-run vm`. Only the server's own key (not client profiles).
//!
//! - `GET /api/vm`: its state, settings, host use (RAM, CPU), the current or last task.
//! - `POST /api/vm {"action": "start"|"stop"|"pause"|"shutdown"|"reset"|"download"|"setup"}`: a task (follow it at
//!   `GET /api/vm/tasks/{id}`); `reset` needs `"confirm": true` (it deletes the VM's disk).
//! - `POST /api/vm/settings {cpus, memory_mb, disk_gb, width, height, idle_pause_s, idle_stop_s, host_reserve_mb}`.
//! - `GET /api/vm/apps`: the catalog with what is installed; `POST /api/vm/apps {"app"} or {"kind", "target"}` with
//!   `"action": "install"|"remove"|"open"`.
//! - `POST /api/vm/exec {"command", "root"?, "timeout_s"?}`: a shell command in the VM; its exit code and output.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path as UrlPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::{authorized, error, AppState};

use super::Vm;

fn guard(s: &AppState, headers: &HeaderMap, write: bool) -> Result<Arc<Vm>, Response> {
    let own = authorized(s, headers) && !crate::clients::given_key(headers).is_some_and(|k| s.clients.by_key(k).is_some());
    if !own {
        return Err(error(StatusCode::UNAUTHORIZED, "authentication_error", "the sandbox VM needs the server's own API key"));
    }
    if write && crate::foreign_origin(s, headers) {
        return Err(crate::not_own_page("the sandbox VM can be controlled"));
    }
    super::get().ok_or_else(|| error(StatusCode::SERVICE_UNAVAILABLE, "server_error", "the server is still starting"))
}

fn bad(e: String) -> Response {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", e)
}

fn started(r: Result<u64, String>) -> Response {
    match r {
        Ok(id) => (StatusCode::ACCEPTED, Json(json!({"task": id}))).into_response(),
        Err(e) => error(StatusCode::CONFLICT, "invalid_request_error", e),
    }
}

pub async fn status_get(State(s): State<AppState>, headers: HeaderMap) -> Response {
    let vm = match guard(&s, &headers, false) { Ok(v) => v, Err(r) => return r };
    Json(tokio::task::spawn_blocking(move || vm.status()).await.unwrap_or_default()).into_response()
}

pub async fn action_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let vm = match guard(&s, &headers, true) { Ok(v) => v, Err(r) => return r };
    let action = req["action"].as_str().unwrap_or("");
    match action {
        "start" | "resume" => started(vm.task("start", |vm, log| vm.start(log).map(|_| "running".into()))),
        "stop" => started(vm.task("save and stop", |vm, log| vm.stop(log).map(|_| "saved".into()))),
        "shutdown" => started(vm.task("shut down", |vm, log| vm.shutdown(log).map(|_| "off".into()))),
        "pause" => match tokio::task::spawn_blocking(move || vm.pause()).await.unwrap_or_else(|e| Err(e.to_string())) {
            Ok(()) => Json(json!({"ok": true})).into_response(),
            Err(e) => bad(e),
        },
        "setup" => started(vm.task("desktop setup", |vm, log| {
            let _ = std::fs::remove_file(vm.vm_dir().join("provisioned"));
            vm.start(log).map(|_| "set up".into())
        })),
        "download" => started(vm.task("download the Debian image", |vm, log| vm.download_image(log))),
        "reset" => {
            if req["confirm"] != true {
                return bad("reset deletes the VM's disk (its apps and files): send \"confirm\": true".into());
            }
            started(vm.task("reset", |vm, _| vm.reset().map(|_| "the VM was deleted; the next start makes a new one".into())))
        }
        other => bad(format!("unknown action {other:?}: start, stop, pause, shutdown, setup, download, reset")),
    }
}

pub async fn settings_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let vm = match guard(&s, &headers, true) { Ok(v) => v, Err(r) => return r };
    match vm.set_config(&req) {
        Ok((c, restart)) => Json(json!({"config": c, "applies": if restart { "at the next start (stop the VM first)" } else { "now" }})).into_response(),
        Err(e) => bad(e),
    }
}

pub async fn task_get(State(s): State<AppState>, headers: HeaderMap, UrlPath(id): UrlPath<String>) -> Response {
    let vm = match guard(&s, &headers, false) { Ok(v) => v, Err(r) => return r };
    match id.parse().ok().and_then(|id| vm.task_view(id)) {
        Some(t) => Json(t).into_response(),
        None => error(StatusCode::NOT_FOUND, "not_found", "no such task"),
    }
}

pub async fn apps_get(State(s): State<AppState>, headers: HeaderMap) -> Response {
    let vm = match guard(&s, &headers, false) { Ok(v) => v, Err(r) => return r };
    let r = tokio::task::spawn_blocking(move || vm.catalog().map(|c| (c, vm.ads_state()))).await.unwrap_or_else(|e| Err(e.to_string()));
    match r {
        Ok((c, ads)) => Json(json!({"apps": c, "ads": ads.map(|(block, lists, allow)| json!({"block": block, "lists": lists, "allow": allow}))})).into_response(),
        Err(e) => bad(e),
    }
}

pub async fn apps_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let vm = match guard(&s, &headers, true) { Ok(v) => v, Err(r) => return r };
    let action = req["action"].as_str().unwrap_or("install").to_string();
    let (id, target) = match (req["app"].as_str(), req["kind"].as_str()) {
        (Some(a), _) if super::guest::app(a).is_some() => (a.to_string(), String::new()),
        (Some(a), _) => return bad(format!("no app {a:?} in the catalog: give \"kind\" (apt, flatpak, deb, appimage) and \"target\" instead")),
        (None, Some(k)) => (k.to_string(), req["target"].as_str().unwrap_or("").to_string()),
        (None, None) => return bad("give \"app\" (a catalog id) or \"kind\" and \"target\"".into()),
    };
    match action.as_str() {
        "open" => match tokio::task::spawn_blocking(move || vm.open_app(&id)).await.unwrap_or_else(|e| Err(e.to_string())) {
            Ok(()) => Json(json!({"ok": true})).into_response(),
            Err(e) => bad(e),
        },
        "install" | "remove" => {
            let install = action == "install";
            if super::guest::app(&id).is_none() {
                if let Err(e) = super::guest::package_script(&id, &target, install) {
                    return bad(e);
                }
            }
            let what = format!("{action} {}", if target.is_empty() { &id } else { &target });
            started(vm.task(&what, move |vm, log| vm.install(&id, &target, install, log)))
        }
        other => bad(format!("unknown action {other:?}: install, remove, open")),
    }
}

/// `POST /api/vm/ads {"block": true|false, "lists": ["pihole", "adguard"], "allow": ["example.com", …]}`
pub async fn ads_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let vm = match guard(&s, &headers, true) { Ok(v) => v, Err(r) => return r };
    let Some(block) = req["block"].as_bool() else { return bad("give \"block\": true or false".into()) };
    let allow: Vec<String> = req["allow"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default();
    let lists: Vec<String> = req["lists"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default();
    if let Err(e) = super::guest::ads_script(block, &lists, &allow) {
        return bad(e);
    }
    match tokio::task::spawn_blocking(move || vm.set_ads(block, &lists, &allow)).await.unwrap_or_else(|e| Err(e.to_string())) {
        Ok(msg) => Json(json!({"ok": true, "block": block, "message": msg})).into_response(),
        Err(e) => bad(e),
    }
}

pub async fn exec_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let vm = match guard(&s, &headers, true) { Ok(v) => v, Err(r) => return r };
    let Some(cmd) = req["command"].as_str().filter(|c| !c.trim().is_empty()).map(str::to_string) else {
        return bad("give \"command\"".into());
    };
    let root = req["root"].as_bool().unwrap_or(false);
    let timeout = Duration::from_secs(req["timeout_s"].as_u64().unwrap_or(120).clamp(1, 3600));
    let r = tokio::task::spawn_blocking(move || {
        vm.ensure()?;
        vm.sh(&cmd, root, None, timeout, &|_| {})
    }).await.unwrap_or_else(|e| Err(e.to_string()));
    match r {
        Ok((code, output)) => Json(json!({"code": code, "output": output})).into_response(),
        Err(e) => bad(e),
    }
}
