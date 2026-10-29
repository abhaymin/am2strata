//! am2Strata: a new edition of the PolyStrata server, written in Rust.
//!
//! Serves every installed model (the `strata-*.json` run configs of a PolyStrata folder) through one address,
//! switching between them on request, with the OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) APIs,
//! pictures and video frames, tool calls, and PolyStrata's web app. It drives the existing C++/CUDA engines over
//! their line protocol; prompts and token ids match PolyStrata's Python server exactly (golden tests).

mod anthropic;
mod browser;
mod api_monitor;
mod audio_api;
mod code_run;
mod image_gen;
mod preview;
mod queue;
mod manager;
mod hub_api;
mod mcp;
mod memory;
mod media;
mod media_jobs;
mod model_remove;
mod monitor;
mod openai;
mod responses;
mod run;
mod server_tools;
mod settings;
mod snapshots;
mod structured;
mod web;
mod web_tools;

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
#[command(name = "am2strata", about = "am2Strata: every installed PolyStrata model behind one address - OpenAI, Anthropic and Responses APIs, MCP tools, the web app")]
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
    /// the web app's files (default: the am2Strata app, web/ in the folder the binary was built in)
    #[arg(long)]
    web_dir: Option<String>,
    /// keep the last 100 API calls (bodies, output, timings) in memory for /api-monitor (also "api_monitor": true
    /// in the config)
    #[arg(long)]
    api_monitor: bool,
    /// where models downloaded from Hugging Face go (their files under models/, tokenizers under packs/)
    #[arg(long, default_value = "~/Development/Strata-data")]
    data_dir: String,
    /// do not load a model until the first request
    #[arg(long)]
    lazy: bool,
    /// seconds to wait for an engine to load
    #[arg(long, default_value_t = 900)]
    ready_timeout: u64,
    /// only check that the configs are readable and --model names one installed model, print its id, and exit
    /// (am2strata-run does this before it stops a running server)
    #[arg(long)]
    check: bool,
}

#[derive(Clone)]
pub struct AppState {
    pub mgr: Arc<Manager>,
    pub api_key: Option<String>,
    /// the MCP servers of the default model's run config (the web app's chats can use their tools)
    pub mcp: Option<Arc<mcp::McpHub>>,
    /// web origins besides this server's own that may change settings and run tools (the config's trusted_origins)
    pub trusted_origins: Vec<String>,
    pub api_monitor: Arc<api_monitor::ApiMonitor>,
    /// the Hugging Face download running or last run (Settings › Models)
    pub hub_job: Arc<std::sync::Mutex<serde_json::Value>>,
    pub models_dir: PathBuf,
    pub data_dir: PathBuf,
    /// running a chat's Python code for the canvas (POST /api/run)
    pub code_run: Arc<code_run::CodeRun>,
    pub code_run_config_on: bool,
    /// the canvas's pages, served sandboxed at /preview/<id>
    pub previews: Arc<preview::Previews>,
    /// image generation (v2): the am2strata-media engine process
    pub media: Arc<image_gen::Media>,
    /// media jobs a chat's tools queued (media_jobs.rs)
    pub media_jobs: Arc<media_jobs::Jobs>,
}

/// A web page of another origin sent this: its Origin is neither this server's address nor a trusted one (no Origin
/// is a client that is no browser). Settings, the run config and MCP tools refuse it (serve/server.py).
pub fn foreign_origin(state: &AppState, headers: &HeaderMap) -> bool {
    let origin = headers.get("origin").and_then(|v| v.to_str().ok()).unwrap_or("").trim_end_matches('/');
    let host = headers.get("host").and_then(|v| v.to_str().ok()).unwrap_or("");
    !origin.is_empty() && origin.split("://").last() != Some(host) && !state.trusted_origins.iter().any(|t| t == origin)
}

pub fn not_own_page(what: &str) -> Response {
    error(StatusCode::FORBIDDEN, "permission_error",
          format!("{what} only from PolyStrata's own page (or an origin in the config's trusted_origins)"))
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
    // compared in constant time (the key is not guessed byte by byte from response times)
    let same = |given: Option<&str>| given.is_some_and(|g| {
        let (a, b) = (g.as_bytes(), key.as_bytes());
        a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
    });
    same(bearer) || same(x_key)
}

/// The names this server answers to: IP addresses, localhost, this PC's name (and name.local), the hosts of the
/// trusted origins and the config's `allowed_hosts`. A request for any other Host is refused: a web page that points
/// its own domain at this PC (DNS rebinding) would otherwise pass for this server's own page.
#[derive(Clone)]
pub struct Hosts(Arc<Vec<String>>);

impl Hosts {
    pub fn new(trusted_origins: &[String], extra: &[String]) -> Hosts {
        let mut v: Vec<String> = vec!["localhost".into()];
        if let Ok(h) = std::fs::read_to_string("/etc/hostname") {
            let h = h.trim().to_lowercase();
            if !h.is_empty() {
                // this PC's name under the usual LAN suffixes (mDNS, routers)
                for suffix in ["local", "lan", "home", "localdomain", "internal", "home.arpa"] {
                    v.push(format!("{h}.{suffix}"));
                }
                v.push(h);
            }
        }
        // its Tailscale name (MagicDNS)
        if let Ok(o) = std::process::Command::new("tailscale").args(["status", "--json"]).output() {
            if let Some(name) = serde_json::from_slice::<serde_json::Value>(&o.stdout).ok()
                .and_then(|j| j["Self"]["DNSName"].as_str().map(|d| d.trim_end_matches('.').to_lowercase())).filter(|d| !d.is_empty()) {
                v.push(name);
            }
        }
        v.extend(trusted_origins.iter().filter_map(|o| o.split("://").nth(1)).map(|h| strip_port(h).to_lowercase()));
        v.extend(extra.iter().map(|h| h.to_lowercase()));
        Hosts(Arc::new(v))
    }

    pub fn allows(&self, host: &str) -> bool {
        let name = strip_port(host).trim_start_matches('[').trim_end_matches(']').to_lowercase();
        name.parse::<std::net::IpAddr>().is_ok() || name.ends_with(".localhost") || self.0.iter().any(|h| *h == name)
    }
}

fn strip_port(h: &str) -> &str {
    if h.starts_with('[') {
        return h.split(']').next().map(|x| &h[..x.len() + 1]).unwrap_or(h);
    }
    match h.rsplit_once(':') {
        Some((n, p)) if p.chars().all(|c| c.is_ascii_digit()) && !n.contains(':') => n,
        _ => h,
    }
}

async fn check_host(axum::extract::State(hosts): axum::extract::State<Hosts>, req: axum::extract::Request, next: axum::middleware::Next) -> Response {
    let host = req.headers().get("host").and_then(|v| v.to_str().ok()).unwrap_or("");
    if !host.is_empty() && !hosts.allows(host) {
        return error(StatusCode::MISDIRECTED_REQUEST, "permission_error",
                     format!("this server does not answer to the name {host:?}: add it to \"allowed_hosts\" in the run config"));
    }
    next.run(req).await
}

fn expand(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(rest) => std::env::var("HOME").map(|h| PathBuf::from(h).join(rest)).unwrap_or_else(|_| PathBuf::from(p)),
        None => PathBuf::from(p),
    }
}

/// am2Strata's own web app: `web/` beside the binary or in a folder above it (target/release/am2strata ->
/// <am2strata folder>/web), so the folder can be moved or renamed.
fn own_web_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?.canonicalize().ok()?;
    exe.ancestors().skip(1).take(4).map(|d| d.join("web")).find(|w| w.join("index.html").exists())
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
        let found: Vec<String> = std::fs::read_dir(&dir).map(|rd| rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("strata-") && n.ends_with(".json")).collect()).unwrap_or_default();
        if found.is_empty() {
            bail!("no strata-*.json run configs in {}", dir.display());
        }
        let why: Vec<String> = found.iter().map(|n| match strata_engine::RunConfig::load(dir.join(n)) {
            Ok(c) => format!("{n}: its engine {} does not exist (deploy/build-engines.sh builds it)", c.exe.display()),
            Err(e) => format!("{n}: {e}"),
        }).collect();
        bail!("no usable run config in {}:\n  {}", dir.display(), why.join("\n  "));
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
    if args.check {
        println!("{default}");
        return Ok(());
    }
    let host = args.host.or(first.cfg.host.clone()).unwrap_or_else(|| "127.0.0.1".into());
    let port = args.port.or(first.cfg.port).unwrap_or(8080);
    let api_key = args.api_key.or(first.cfg.api_key.clone());
    for m in &models {
        tracing::info!(id = %m.id, context = m.cfg.max_context(), images = m.cfg.vision.is_some(), "installed model");
    }
    // queue_max_wait_s (the default model's run config): how long a job may wait for its turn before it is refused
    let max_wait = first.cfg.rest.get("queue_max_wait_s").and_then(|v| v.as_f64()).filter(|v| *v > 0.0).map(Duration::from_secs_f64);
    let mgr = Arc::new(Manager::new(models, dir.clone(), default.clone(), Duration::from_secs(args.ready_timeout), expand(&args.data_dir), max_wait));
    for g in &mgr.hw.gpus {
        tracing::info!("GPU {}: {} ({} MiB, sm_{})", g.index, g.name, g.vram_mib, g.arch);
    }
    tracing::info!("RAM {} MiB, {} CPUs; video frames {}", mgr.hw.ram_mib, mgr.hw.cpus,
                   if media::ffmpeg_available() { "with ffmpeg" } else { "need ffmpeg (sudo apt install ffmpeg)" });
    if !args.lazy {
        mgr.load(&default).await.with_context(|| format!("loading {default}"))?;
    }
    let hub = mcp::McpHub::from_config(&first.cfg.rest).map_err(anyhow::Error::msg)?;
    if let Some(h) = &hub {
        h.start();
    }
    let trusted_origins: Vec<String> = first.cfg.rest.get("trusted_origins").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.trim_end_matches('/').to_string())).collect()).unwrap_or_default();
    let monitor_on = args.api_monitor || first.cfg.rest.get("api_monitor") == Some(&serde_json::Value::Bool(true));
    if monitor_on {
        tracing::info!("API monitor on: the last 100 API calls are kept in memory, at /api-monitor");
    }
    let extra_hosts: Vec<String> = first.cfg.rest.get("allowed_hosts").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
    let hosts = Hosts::new(&trusted_origins, &extra_hosts);
    tracing::info!("answering to IP addresses, localhost and {}", hosts.0.join(", "));
    let loopback = matches!(host.as_str(), "127.0.0.1" | "::1" | "localhost");
    let flag = |k: &str| first.cfg.rest.get(k).and_then(|v| v.as_bool());
    let _ = media::POLICY.set(media::SourcePolicy { local_files: flag("allow_local_files").unwrap_or(loopback),
                                                     private_urls: flag("allow_private_urls").unwrap_or(false) });
    let mut code_run = code_run::CodeRun::from_config(&first.cfg.rest);
    if api_key.as_deref().unwrap_or("").is_empty() && !loopback {
        tracing::warn!("no API key and listening on {host}: anyone on the network can use this server; set \"api_key\" in the run config");
        if code_run.enabled {
            tracing::warn!("running code (the canvas) is off: it needs an API key when the server listens beyond this PC");
            code_run.enabled = false;
            code_run.needs_key = true;
        }
    }
    let state = AppState { mgr: mgr.clone(), api_key, mcp: hub.clone(), trusted_origins,
                           api_monitor: api_monitor::ApiMonitor::new(monitor_on),
                           hub_job: Arc::new(std::sync::Mutex::new(json!({"state": "idle"}))),
                           models_dir: dir.clone(), data_dir: expand(&args.data_dir),
                           code_run: Arc::new(code_run),
                           code_run_config_on: first.cfg.rest.get("code_run") != Some(&serde_json::Value::Bool(false)),
                           previews: Arc::new(preview::Previews::default()),
                           media: Arc::new(image_gen::Media::new(&expand(&args.data_dir), &first.cfg.rest)),
                           media_jobs: Arc::new(media_jobs::Jobs::new()) };
    state.media_jobs.bind(state.clone());
    let _ = mgr.media_jobs.set(state.media_jobs.clone());
    let _ = mgr.web.set(Arc::new(web_tools::WebTools::from_config(&first.cfg.rest)));
    let _ = mgr.browser.set(Arc::new(browser::Browsers::new(browser::Settings::from_config(&first.cfg.rest, &expand(&args.data_dir)))));
    let web_dir = match args.web_dir.map(|p| expand(&p)) {
        Some(d) => d,
        None => own_web_dir().context("the am2Strata web app (web/ in the am2strata folder) was not found; pass --web-dir")?,
    };
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
        .route("/api/queue", get(web::queue))
        .route("/api/gpus", get(web::gpus))
        .route("/api/tools/approve", post(browser::approve))
        .route("/api/cache/snapshots/clear", post(web::clear_snapshots))
        .route("/api/models/{id}", axum::routing::delete(model_remove::remove))
        .route("/api/models/{id}/removal", get(model_remove::removal))
        .route("/api/media/models/{name}", axum::routing::delete(model_remove::media_remove))
        .route("/api/media/models/{name}/removal", get(model_remove::media_removal))
        .route("/api/media/capabilities", get(media_jobs::capabilities))
        .route("/api/jobs/{id}", get(media_jobs::status))
        .route("/v1/media/jobs/{id}", get(media_jobs::status))
        .route("/v1/media/jobs/{id}/content", get(media_jobs::content))
        .route("/api/models/updates", get(hub_api::updates))
        .route("/api/models/update", post(hub_api::update))
        .route("/api/requests", get(api_monitor::requests))
        .route("/api/hub/quants", post(hub_api::quants))
        .route("/api/hub/pull", get(hub_api::status).post(hub_api::pull))
        .route("/api/run", get(code_run::info).post(code_run::run))
        .route("/v1/images/generations", post(image_gen::generations))
        .route("/v1/videos/generations", post(image_gen::video_generations))
        .route("/v1/audio/transcriptions", post(audio_api::transcriptions))
        .route("/v1/audio/translations", post(audio_api::translations))
        .route("/v1/audio/speech", post(audio_api::speech))
        .route("/api/media/audio/device", post(audio_api::set_device))
        .route("/api/memory", get(memory::list).post(memory::add))
        .route("/api/memory/settings", post(memory::set_settings))
        .route("/api/memory/{id}", axum::routing::delete(memory::delete))
        .route("/api/media/models", get(image_gen::models))
        .route("/api/media/models/pull", get(image_gen::pull_status).post(image_gen::pull))
        .route("/api/media/models/default", post(image_gen::set_default))
        .route("/api/preview", post(preview::create))
        .route("/preview/{id}", get(preview::show))
        .route("/load", post(api_monitor::load))
        .route("/unload", post(api_monitor::unload))
        .route("/settings", get(web::settings_get).post(web::settings_post))
        .route("/config", get(web::config_get).post(web::config_post))
        .route("/mcp", get(web::mcp))
        .route("/metrics", get(web::metrics))
        .route("/metrics/prometheus", get(web::prometheus))
        .route("/v1/status", get(web::status))
        // the web app: the page at /, its files under /web/ (as PolyStrata serves them), the monitor at /api-monitor
        .route_service("/", ServeFile::new(web_dir.join("index.html")))
        .route_service("/api-monitor", ServeFile::new(web_dir.join("monitor.html")))
        .nest_service("/web", ServeDir::new(&web_dir))
        .nest_service("/fonts", ServeDir::new(web_dir.join("fonts")))
        // browsers check the web app's files for a newer version every time (an update is never stuck behind a cache)
        .layer(tower_http::set_header::SetResponseHeaderLayer::if_not_present(
            axum::http::header::CACHE_CONTROL, axum::http::HeaderValue::from_static("no-cache")))
        // no other site may frame the app (clickjacking), nothing is sniffed as another type, no address leaks out
        .layer(tower_http::set_header::SetResponseHeaderLayer::if_not_present(
            axum::http::header::X_FRAME_OPTIONS, axum::http::HeaderValue::from_static("SAMEORIGIN")))
        .layer(tower_http::set_header::SetResponseHeaderLayer::if_not_present(
            axum::http::header::X_CONTENT_TYPE_OPTIONS, axum::http::HeaderValue::from_static("nosniff")))
        .layer(tower_http::set_header::SetResponseHeaderLayer::if_not_present(
            axum::http::header::REFERRER_POLICY, axum::http::HeaderValue::from_static("no-referrer")))
        .layer(axum::middleware::from_fn_with_state(state.clone(), api_monitor::layer))
        .layer(axum::extract::DefaultBodyLimit::max(256 << 20))
        .layer(axum::middleware::from_fn_with_state(hosts, check_host))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await.with_context(|| format!("bind {host}:{port}"))?;
    tracing::info!("listening on http://{host}:{port}/ (web app from {}), API at /v1", web_dir.display());
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    mgr.unload().await;
    if let Some(b) = mgr.browser.get() {
        b.close().await;
    }
    if let Some(h) = hub {
        h.close().await;
    }
    Ok(())
}
