//! strata-serve: the PolyStrata server in Rust.
//!
//! Serves every installed model (the `strata-*.json` run configs of a PolyStrata folder) through one address,
//! switching between them on request, with the OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) APIs,
//! pictures and video frames, tool calls, and PolyStrata's web app. It drives the existing C++/CUDA engines over
//! their line protocol; prompts and token ids match PolyStrata's Python server exactly (golden tests).

mod anthropic;
mod manager;
mod media;
mod monitor;
mod openai;
mod responses;
mod run;
mod web;

use anyhow::{bail, Context, Result};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use manager::Manager;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tower_http::services::{ServeDir, ServeFile};

#[derive(Parser)]
#[command(about = "PolyStrata server in Rust: every installed model, OpenAI and Anthropic APIs, the web app")]
struct Args {
    /// folder with the strata-<model>.json run configs (a PolyStrata folder)
    #[arg(long, default_value = "~/Development/PolyStrata")]
    models_dir: String,
    /// the model loaded at start and after an unload (id or a unique part of it; default: the first config's)
    #[arg(long)]
    model: Option<String>,
    /// port (default: the default model's config, else 8080)
    #[arg(long)]
    port: Option<u16>,
    /// address (default: the default model's config host, else 127.0.0.1)
    #[arg(long)]
    host: Option<String>,
    /// API key (default: the default model's config api_key)
    #[arg(long)]
    api_key: Option<String>,
    /// the web app's files (default: <models-dir>/serve/web)
    #[arg(long)]
    web_dir: Option<String>,
    /// do not load a model until the first request
    #[arg(long)]
    lazy: bool,
    /// seconds to wait for an engine to load
    #[arg(long, default_value_t = 900)]
    ready_timeout: u64,
}

#[derive(Clone)]
pub struct AppState {
    pub mgr: Arc<Manager>,
    pub api_key: Option<String>,
}

pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn error(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    (status, Json(json!({"error": {"type": kind, "message": message.into()}}))).into_response()
}

pub fn authorized(state: &AppState, headers: &HeaderMap) -> bool {
    let Some(key) = state.api_key.as_deref().filter(|k| !k.is_empty()) else { return true };
    let bearer = headers.get("authorization").and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer "));
    let x_key = headers.get("x-api-key").and_then(|v| v.to_str().ok());
    bearer == Some(key) || x_key == Some(key)
}

fn expand(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(rest) => std::env::var("HOME").map(|h| PathBuf::from(h).join(rest)).unwrap_or_else(|_| PathBuf::from(p)),
        None => PathBuf::from(p),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let args = Args::parse();
    let dir = expand(&args.models_dir);
    let models = manager::scan(&dir);
    if models.is_empty() {
        bail!("no strata-*.json run configs in {}", dir.display());
    }
    let default = match &args.model {
        Some(m) => {
            let hits: Vec<_> = models.iter().filter(|e| e.id == *m || e.id.contains(m.as_str())).collect();
            match hits.as_slice() {
                [one] => one.id.clone(),
                [] => bail!("no installed model matches {m:?}; installed: {}", models.iter().map(|e| e.id.as_str()).collect::<Vec<_>>().join(", ")),
                _ => hits.iter().find(|e| e.id == *m).map(|e| e.id.clone()).with_context(|| format!("{m:?} matches several models"))?,
            }
        }
        None => models[0].id.clone(),
    };
    let first = models.iter().find(|e| e.id == default).cloned().expect("default exists");
    let host = args.host.or(first.cfg.host.clone()).unwrap_or_else(|| "127.0.0.1".into());
    let port = args.port.or(first.cfg.port).unwrap_or(8080);
    let api_key = args.api_key.or(first.cfg.api_key.clone());
    for m in &models {
        tracing::info!(id = %m.id, context = m.cfg.max_context(), images = m.cfg.vision.is_some(), "installed model");
    }
    let mgr = Arc::new(Manager::new(models, default.clone(), Duration::from_secs(args.ready_timeout)));
    for g in &mgr.hw.gpus {
        tracing::info!("GPU {}: {} ({} MiB, sm_{})", g.index, g.name, g.vram_mib, g.arch);
    }
    tracing::info!("RAM {} MiB, {} CPUs; video frames {}", mgr.hw.ram_mib, mgr.hw.cpus,
                   if media::ffmpeg_available() { "with ffmpeg" } else { "need ffmpeg (sudo apt install ffmpeg)" });
    if !args.lazy {
        mgr.load(&default).await.with_context(|| format!("loading {default}"))?;
    }
    let state = AppState { mgr: mgr.clone(), api_key };
    let web_dir = args.web_dir.map(|p| expand(&p)).unwrap_or_else(|| dir.join("serve/web"));
    let app = Router::new()
        .route("/health", get(web::health))
        .route("/v1/models", get(openai::models))
        .route("/v1/chat/completions", post(openai::chat))
        .route("/v1/responses", post(responses::responses))
        .route("/v1/responses/{*rest}", get(responses::not_stored).post(responses::not_stored).delete(responses::not_stored))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/api/models", get(web::list_models))
        .route("/api/models/current", get(web::current_model))
        .route("/api/models/load", post(web::load_model))
        .route("/api/models/unload", post(web::unload_model))
        .route("/settings", get(web::settings_get).post(web::settings_post))
        .route("/config", get(web::config_get))
        .route("/mcp", get(web::mcp))
        .route("/metrics", get(web::metrics))
        .route("/metrics/prometheus", get(web::prometheus))
        .route("/v1/status", get(web::status))
        // the web app: the page at /, its files under /web/ (as PolyStrata serves them), the monitor at /api-monitor
        .route_service("/", ServeFile::new(web_dir.join("index.html")))
        .route_service("/api-monitor", ServeFile::new(web_dir.join("monitor.html")))
        .nest_service("/web", ServeDir::new(&web_dir))
        .nest_service("/fonts", ServeDir::new(web_dir.join("fonts")))
        .layer(axum::extract::DefaultBodyLimit::max(256 << 20))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await.with_context(|| format!("bind {host}:{port}"))?;
    tracing::info!("listening on http://{host}:{port}/ (web app from {}), API at /v1", web_dir.display());
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    mgr.unload().await;
    Ok(())
}
