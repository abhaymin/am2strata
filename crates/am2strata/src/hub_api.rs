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
    let job = s.hub_job.clone();
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
        let mut j = job.lock().unwrap();
        match result {
            Ok(Outcome::Configured { id, .. }) => {
                j["state"] = json!("done");
                j["model"] = json!(id);
                j["message"] = json!(format!("{id} is installed: pick it in the model list to load it"));
            }
            Ok(Outcome::Downloaded { dir, why }) => {
                j["state"] = json!("done");
                j["message"] = json!(match why {
                    Some(w) => format!("downloaded to {}, but it cannot run here: {w}", dir.display()),
                    None => format!("downloaded to {}", dir.display()),
                });
            }
            Err(e) => {
                j["state"] = json!("error");
                j["message"] = json!(e);
            }
        }
        tracing::info!("hub pull {}: {}", j["repo"], j["message"]);
    });
    (StatusCode::ACCEPTED, Json(json!({"state": "running"}))).into_response()
}
