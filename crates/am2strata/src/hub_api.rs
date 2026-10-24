//! Hugging Face downloads from the web app (Settings › Models): POST /api/hub/quants lists a repository's
//! quantizations, POST /api/hub/pull starts one download (one at a time), GET /api/hub/pull reports its progress.
//! They write files on this PC, so only this server's own page (or a trusted origin) may start one.

use crate::{authorized, error, foreign_origin, not_own_page, AppState};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::time::Instant;
use strata_hub::pull::{Event, Outcome, Spec};

fn token_of(req: &Value) -> Option<String> {
    strata_hub::token(req.get("token").and_then(Value::as_str).map(str::trim).filter(|t| !t.is_empty()))
}

pub async fn quants(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if foreign_origin(&s, &headers) {
        return not_own_page("models can be listed from Hugging Face");
    }
    let Some(repo) = req.get("repo").and_then(Value::as_str).filter(|r| !r.trim().is_empty()) else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "send {\"repo\": \"owner/name\"}");
    };
    match strata_hub::pull::list(repo, token_of(&req)).await {
        Ok(qs) if qs.is_empty() => error(StatusCode::BAD_REQUEST, "invalid_request_error",
                                         format!("{} has no GGUF files (am2Strata runs GGUF models)", strata_hub::pull::repo_name(repo))),
        Ok(qs) => Json(json!({"repo": strata_hub::pull::repo_name(repo), "quants": qs.into_iter()
            .map(|(q, size, files)| json!({"quant": q, "size": size, "files": files})).collect::<Vec<_>>()})).into_response(),
        Err(e) => error(StatusCode::BAD_GATEWAY, "hub_error", e),
    }
}

pub async fn status(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    Json(s.hub_job.lock().unwrap().clone()).into_response()
}

pub async fn pull(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if foreign_origin(&s, &headers) {
        return not_own_page("models can be downloaded");
    }
    let (Some(repo), Some(quant)) = (req.get("repo").and_then(Value::as_str), req.get("quant").and_then(Value::as_str)) else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "send {\"repo\": \"owner/name\", \"quant\": \"Q4_K_M\"}");
    };
    {
        let mut job = s.hub_job.lock().unwrap();
        if job["state"] == "running" {
            return error(StatusCode::CONFLICT, "busy", format!("{} {} is still downloading", job["repo"].as_str().unwrap_or(""), job["quant"].as_str().unwrap_or("")));
        }
        *job = json!({"state": "running", "repo": strata_hub::pull::repo_name(repo), "quant": quant, "file": null, "done": 0, "total": 0,
                      "speed": 0, "notes": [], "message": null, "model": null, "started": crate::now()});
    }
    let spec = Spec { repo: repo.to_string(), revision: "main".into(), quant: Some(quant.to_string()), files: Vec::new(), token: token_of(&req),
                      mmproj: req.get("mmproj").and_then(Value::as_bool).unwrap_or(true), data_dir: s.data_dir.clone(),
                      models_dir: s.models_dir.clone(), context: None, write_config: true, dir: None };
    run_job(s.hub_job.clone(), spec, None);
    (StatusCode::ACCEPTED, Json(json!({"state": "running"}))).into_response()
}

/// Runs a download as the server's one hub job (GET /api/hub/pull shows it). `after`: run when it succeeded (an update
/// marks the model's pack and tokenizer for rebuilding).
fn run_job(job: std::sync::Arc<std::sync::Mutex<Value>>, spec: Spec, after: Option<Box<dyn FnOnce() -> Option<String> + Send>>) {
    tokio::spawn(async move {
        let t0 = Instant::now();
        let mut on = |e: Event| {
            let mut j = job.lock().unwrap();
            match e {
                Event::Note(n) => j["notes"].as_array_mut().unwrap().push(json!(n)),
                Event::Progress { file, done, total } => {
                    j["file"] = json!(file);
                    j["done"] = json!(done);
                    j["total"] = json!(total);
                    j["speed"] = json!((done as f64 / t0.elapsed().as_secs_f64().max(0.1)) as u64);
                }
                Event::FileDone { file, checked } => j["notes"].as_array_mut().unwrap()
                    .push(json!(format!("{file}: done, SHA-256 {}", if checked { "checked" } else { "not published" }))),
            }
        };
        let result = strata_hub::pull::run(&spec, &mut on).await;
        let note = if result.is_ok() { after.and_then(|f| f()) } else { None };
        let mut j = job.lock().unwrap();
        match result {
            Ok(Outcome::Configured { id, .. }) => {
                j["state"] = json!("done");
                j["model"] = json!(id);
                j["message"] = json!(format!("{id} is installed: pick it in the model list to load it"));
            }
            Ok(Outcome::Downloaded { dir, why }) => {
                j["state"] = json!("done");
                j["message"] = json!(match (why, note) {
                    (Some(w), _) => format!("downloaded to {}, but it cannot run here: {w}", dir.display()),
                    (None, Some(n)) => n,
                    (None, None) => format!("downloaded to {}", dir.display()),
                });
            }
            Err(e) => {
                j["state"] = json!("error");
                j["message"] = json!(e);
            }
        }
        tracing::info!("hub pull {}: {}", j["repo"], j["message"]);
    });
}

/// The model folder and pack of an installed model, from its run config's --model (or --native) and --pack.
fn model_paths(s: &AppState, id: &str) -> Result<(std::path::PathBuf, Option<std::path::PathBuf>), String> {
    let e = s.mgr.entry(id).ok_or_else(|| format!("no installed model is named {id:?}"))?;
    let arg = |k: &[&str]| e.cfg.args.windows(2).find(|w| k.contains(&w[0].as_str())).map(|w| std::path::PathBuf::from(&w[1]));
    let model = arg(&["--model", "--native"]).ok_or("its run config names no model file")?;
    let dir = model.parent().ok_or("its model file has no folder")?.to_path_buf();
    Ok((dir, arg(&["--pack"])))
}

/// GET /api/models/updates?id=<model>: whether Hugging Face has newer files than the ones this model was downloaded with.
pub async fn updates(State(s): State<AppState>, headers: HeaderMap,
                     axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let Some(id) = q.get("id") else { return error(StatusCode::BAD_REQUEST, "invalid_request_error", "add ?id=<model>") };
    let (dir, _) = match model_paths(&s, id) {
        Ok(p) => p,
        Err(e) => return error(StatusCode::NOT_FOUND, "not_found", e),
    };
    match strata_hub::pull::check_update(&dir, strata_hub::token(None)).await {
        Ok(u) => Json(json!({"id": id, "known": true, "repo": u.repo, "update": !u.changed.is_empty(), "changed": u.changed,
                             "removed": u.removed, "bytes": u.bytes})).into_response(),
        Err(e) => Json(json!({"id": id, "known": false, "update": false, "reason": e})).into_response(),
    }
}

/// POST /api/models/update {"id"}: downloads the changed files (each replaces its old copy only once it is whole and its
/// SHA-256 checked), then marks the pack and tokenizer to be built again on the next load. Not while the model is loaded.
pub async fn update(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if foreign_origin(&s, &headers) {
        return not_own_page("models can be updated");
    }
    let Some(id) = req.get("id").and_then(Value::as_str) else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "send {\"id\": \"<model>\"}");
    };
    if s.mgr.current().await.is_some_and(|l| l.entry.id == id) {
        return error(StatusCode::CONFLICT, "busy", format!("{id} is loaded: unload it (or load another model) before updating its files"));
    }
    let (dir, pack) = match model_paths(&s, id) {
        Ok(p) => p,
        Err(e) => return error(StatusCode::NOT_FOUND, "not_found", e),
    };
    let u = match strata_hub::pull::check_update(&dir, token_of(&req)).await {
        Ok(u) => u,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    if u.changed.is_empty() {
        return Json(json!({"state": "current", "message": format!("{id} is up to date with {}", u.repo)})).into_response();
    }
    {
        let mut job = s.hub_job.lock().unwrap();
        if job["state"] == "running" {
            return error(StatusCode::CONFLICT, "busy", "a download is running; one at a time");
        }
        *job = json!({"state": "running", "repo": u.repo, "quant": "update", "file": null, "done": 0, "total": 0, "speed": 0,
                      "notes": [format!("updating {id}: {}", u.changed.join(", "))], "message": null, "model": id, "started": crate::now()});
    }
    let spec = Spec { repo: u.repo.clone(), revision: u.revision.clone(), quant: None, files: u.changed.clone(), token: token_of(&req),
                      mmproj: false, data_dir: s.data_dir.clone(), models_dir: s.models_dir.clone(), context: None,
                      write_config: false, dir: dir.file_name().and_then(|n| n.to_str()).map(str::to_string) };
    let id2 = id.to_string();
    run_job(s.hub_job.clone(), spec, Some(Box::new(move || {
        // the pack and tokenizer describe the old files: built again from the new ones on the next load
        if let Some(p) = pack {
            let _ = std::fs::remove_file(p.join("index.txt"));
            let _ = std::fs::remove_dir_all(p.join("tokenizer"));
        }
        Some(format!("{id2} is updated; its pack is built again on the next load"))
    })));
    (StatusCode::ACCEPTED, Json(json!({"state": "running", "changed": u.changed, "bytes": u.bytes}))).into_response()
}
