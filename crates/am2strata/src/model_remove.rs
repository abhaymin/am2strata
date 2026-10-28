//! Removing a downloaded model from the disk (Settings › Models › Remove). A text model is its run config
//! (strata-<id>.json) and what it points at; a media model is the media engine's (its own files, mold keeping the
//! parts another model shares).
//!
//! What a text model's removal deletes: its run config and log, and every download folder (`<data>/models/<dir>`) and
//! pack (`<data>/packs/<id>`) it refers to that no other run config refers to. A folder that is a symbolic link loses
//! the link only. Files outside those two folders (PolyStrata's own data, the shared MTP drafts) are never deleted.
//! The model in use and the server's start-up model cannot be removed. GET .../removal shows the plan first.

use crate::{authorized, error, AppState};
use axum::extract::{Path as UrlPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Every absolute path a run config names (args, the vision block, the tokenizer...).
fn paths_in(v: &Value, out: &mut Vec<PathBuf>) {
    match v {
        Value::String(s) if s.starts_with('/') => out.push(PathBuf::from(s)),
        Value::Array(a) => a.iter().for_each(|x| paths_in(x, out)),
        Value::Object(o) => o.values().for_each(|x| paths_in(x, out)),
        _ => {}
    }
}

/// The deletable folder a path is in: `<root>/<first part>` for root = <data>/models or <data>/packs.
fn unit_of(p: &Path, roots: &[PathBuf]) -> Option<PathBuf> {
    roots.iter().find_map(|r| {
        let rest = p.strip_prefix(r).ok()?;
        let first = rest.components().next()?;
        Some(r.join(first.as_os_str()))
    })
}

/// Bytes on disk under `p`, a symbolic link counted as itself.
fn disk_bytes(p: &Path) -> u64 {
    let Ok(m) = std::fs::symlink_metadata(p) else { return 0 };
    if m.file_type().is_symlink() {
        0
    } else if m.is_dir() {
        std::fs::read_dir(p).map(|rd| rd.flatten().map(|e| disk_bytes(&e.path())).sum()).unwrap_or(0)
    } else {
        m.len()
    }
}

fn configs(models_dir: &Path) -> Vec<(PathBuf, Value)> {
    std::fs::read_dir(models_dir).map(|rd| rd.flatten().map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("strata-") && n.ends_with(".json")))
        .filter_map(|p| Some((p.clone(), serde_json::from_slice::<Value>(&std::fs::read(&p).ok()?).ok()?))).collect())
        .unwrap_or_default()
}

/// The plan for removing text model `id`: {"model", "files": [{path, bytes, link?}], "kept": [{path, why}], "bytes",
/// "blocked"?: why it cannot be removed now}.
pub async fn plan(s: &AppState, id: &str) -> Result<Value, (StatusCode, String)> {
    let entry = s.mgr.models().into_iter().find(|m| m.id == id)
        .ok_or((StatusCode::NOT_FOUND, format!("no installed model {id}")))?;
    let roots = [s.data_dir.join("models"), s.data_dir.join("packs")];
    let all = configs(&s.mgr.models_dir());
    let Some(own) = all.iter().find(|(p, _)| *p == entry.path).map(|(_, v)| v.clone()) else {
        return Err((StatusCode::NOT_FOUND, format!("the run config of {id} is gone")));
    };
    let mut refs = Vec::new();
    paths_in(&own, &mut refs);
    let mut units = BTreeSet::new();
    let mut kept = Vec::new();
    for r in &refs {
        match unit_of(r, &roots) {
            Some(u) => {
                units.insert(u);
            }
            // data kept elsewhere (shared drafts, PolyStrata's profiles); not the engines, tokenizers and folders
            None if r.is_file() && r.starts_with(&s.data_dir) && own["log"].as_str() != r.to_str()
                && !kept.iter().any(|k: &Value| k["path"] == json!(r)) =>
                kept.push(json!({"path": r, "why": "outside the download folders (shared)"})),
            None => {}
        }
    }
    let mut files = vec![json!({"path": entry.path, "bytes": disk_bytes(&entry.path)})];
    if let Some(log) = own["log"].as_str().map(PathBuf::from).filter(|l| l.is_file()) {
        files.push(json!({"path": log, "bytes": disk_bytes(&log)}));
    }
    for u in units {
        if !std::fs::symlink_metadata(&u).is_ok() {
            continue;
        }
        let users: Vec<String> = all.iter().filter(|(p, _)| *p != entry.path).filter(|(_, v)| {
            let mut r = Vec::new();
            paths_in(v, &mut r);
            r.iter().any(|x| x.starts_with(&u))
        }).filter_map(|(p, _)| p.file_stem().and_then(|n| n.to_str()).map(|n| n.trim_start_matches("strata-").to_string())).collect();
        if !users.is_empty() {
            kept.push(json!({"path": u, "why": format!("also used by {}", users.join(", "))}));
            continue;
        }
        let link = std::fs::symlink_metadata(&u).map(|m| m.file_type().is_symlink()).unwrap_or(false);
        let mut f = json!({"path": u, "bytes": disk_bytes(&u)});
        if link {
            f["link"] = json!(std::fs::read_link(&u).ok());
        }
        files.push(f);
    }
    let bytes: u64 = files.iter().map(|f| f["bytes"].as_u64().unwrap_or(0)).sum();
    let mut v = json!({"model": id, "display_name": entry.display_name(), "files": files, "kept": kept, "bytes": bytes});
    let current = s.mgr.current().await.map(|l| l.entry.id.clone());
    if current.as_deref() == Some(id) {
        v["blocked"] = json!("it is loaded: switch to another model or unload it first");
    } else if s.mgr.default_id == id {
        v["blocked"] = json!("it is the model the server starts with (--model): start it with another one first");
    }
    Ok(v)
}

fn denied(s: &AppState, headers: &HeaderMap, write: bool) -> Option<Response> {
    if !authorized(s, headers) {
        return Some(error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required"));
    }
    if write && crate::foreign_origin(s, headers) {
        return Some(crate::not_own_page("models can be removed"));
    }
    None
}

/// GET /api/models/{id}/removal
pub async fn removal(State(s): State<AppState>, headers: HeaderMap, UrlPath(id): UrlPath<String>) -> Response {
    if let Some(r) = denied(&s, &headers, false) {
        return r;
    }
    match plan(&s, &id).await {
        Ok(v) => Json(v).into_response(),
        Err((code, m)) => error(code, "not_found", m),
    }
}

/// DELETE /api/models/{id}: the plan carried out, in the queue's turn (no load starts meanwhile).
pub async fn remove(State(s): State<AppState>, headers: HeaderMap, UrlPath(id): UrlPath<String>) -> Response {
    if let Some(r) = denied(&s, &headers, true) {
        return r;
    }
    let _turn = loop {
        match s.mgr.queue.enter(crate::queue::Need::Any, "remove", &id, |_| {}).await {
            Ok(p) => break p,
            Err(_) => tokio::time::sleep(std::time::Duration::from_secs(1)).await,
        }
    };
    let p = match plan(&s, &id).await {
        Ok(v) => v,
        Err((code, m)) => return error(code, "not_found", m),
    };
    if let Some(why) = p["blocked"].as_str() {
        return error(StatusCode::CONFLICT, "model_in_use", format!("{id} cannot be removed now: {why}"));
    }
    let roots = [s.data_dir.join("models"), s.data_dir.join("packs")];
    let (mut freed, mut removed, mut warnings) = (0u64, Vec::new(), Vec::new());
    for f in p["files"].as_array().into_iter().flatten() {
        let path = PathBuf::from(f["path"].as_str().unwrap_or(""));
        // the plan only names the config, its log and folders under the two roots; checked again here
        let allowed = path.parent() == Some(s.mgr.models_dir().as_path()) || unit_of(&path, &roots).as_deref() == Some(path.as_path());
        if !allowed {
            warnings.push(format!("{} skipped: not a file this removal may delete", path.display()));
            continue;
        }
        let meta = match std::fs::symlink_metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let r = if meta.is_dir() && !meta.file_type().is_symlink() { std::fs::remove_dir_all(&path) } else { std::fs::remove_file(&path) };
        match r {
            Ok(()) => {
                freed += f["bytes"].as_u64().unwrap_or(0);
                removed.push(path);
            }
            Err(e) => warnings.push(format!("{}: {e}", path.display())),
        }
    }
    tracing::info!(model = %id, freed_gb = freed as f64 / 1e9, "model removed from the disk");
    Json(json!({"model": id, "removed": removed, "freed_bytes": freed, "kept": p["kept"], "warnings": warnings})).into_response()
}

/// GET /api/media/models/{name}/removal and DELETE /api/media/models/{name}: through the media engine.
pub async fn media_removal(State(s): State<AppState>, headers: HeaderMap, UrlPath(name): UrlPath<String>) -> Response {
    if let Some(r) = denied(&s, &headers, false) {
        return r;
    }
    match s.media.call(json!({"op": "remove_plan", "model": name}), |_| {}).await {
        Ok(mut v) => {
            v.as_object_mut().map(|o| o.remove("event"));
            Json(v).into_response()
        }
        Err(e) => error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("{e:#}")),
    }
}

pub async fn media_remove(State(s): State<AppState>, headers: HeaderMap, UrlPath(name): UrlPath<String>) -> Response {
    if let Some(r) = denied(&s, &headers, true) {
        return r;
    }
    // in the queue's turn: no media job uses the model while it goes
    let _turn = loop {
        match s.mgr.queue.enter(crate::queue::Need::Any, "remove", &name, |_| {}).await {
            Ok(p) => break p,
            Err(_) => tokio::time::sleep(std::time::Duration::from_secs(1)).await,
        }
    };
    match s.media.call(json!({"op": "remove", "model": name}), |_| {}).await {
        Ok(mut v) => {
            v.as_object_mut().map(|o| o.remove("event"));
            tracing::info!(model = %name, freed_gb = v["freed_bytes"].as_f64().unwrap_or(0.0) / 1e9, "media model removed from the disk");
            Json(v).into_response()
        }
        Err(e) => error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("{e:#}")),
    }
}
