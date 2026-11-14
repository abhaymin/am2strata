//! am2Strata: a new edition of the PolyStrata server, written in Rust.
//!
//! Serves every installed model (the `strata-*.json` run configs of a PolyStrata folder) through one address,
//! switching between them on request, with the OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) APIs,
//! pictures and video frames, tool calls, and PolyStrata's web app. It drives the existing C++/CUDA engines over
//! their line protocol; prompts and token ids match PolyStrata's Python server exactly (golden tests).

mod agents;
mod cluster;
mod cluster_tls;
mod workshop;
mod server_tls;
mod machine;
mod optimizer;
mod clients;
mod gemini;
mod anthropic;
mod browser;
mod computer;
mod api_monitor;
mod api_keys;
mod audio_api;
mod langdetect;
mod live_channel;
mod music;
mod code_run;
mod image_gen;
mod preview;
mod production;
mod queue;
mod manager;
mod hub_api;
mod mcp_api;
mod mem;
mod mcp;
mod memory;
mod media;
mod compose;
mod media_check;
mod media_dub;
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
mod unsloth;
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
#[command(name = "am2strata", version, about = "am2Strata: every installed PolyStrata model behind one address - OpenAI, Anthropic and Responses APIs, MCP tools, the web app")]
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
    /// the run config's client profiles: other API keys, each with its tools (clients.rs)
    pub clients: Arc<clients::Clients>,
    /// the keys made while the server runs, and everyone's usage counts (api_keys.rs)
    pub keys: Arc<api_keys::ApiKeys>,
    /// the MCP servers: the default model's run config's and the app's (always there; it may have none)
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
    /// the workers this server routes requests to (cluster.rs; none: a single machine)
    pub cluster: Arc<cluster::Cluster>,
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

/// The priority and client of a request, for the queue: high for this server's own page (a person waiting), normal
/// for API calls, or what `x-strata-priority` asks (a client profile's key may ask normal or low, not high).
async fn priority_scope(axum::extract::State(s): axum::extract::State<AppState>, req: axum::extract::Request,
                        next: axum::middleware::Next) -> axum::response::Response {
    use crate::queue::Priority;
    let h = req.headers();
    let own_page = h.contains_key("origin") && !foreign_origin(&s, h);
    let client_profile = if authorized(&s, h) { None } else { clients::given_key(h).and_then(|k| s.clients.by_key(k)).map(|p| p.name.clone()) };
    let asked = h.get("x-strata-priority").and_then(|v| v.to_str().ok()).and_then(Priority::parse);
    let mut p = asked.unwrap_or(if own_page { Priority::High } else { Priority::Normal });
    if client_profile.is_some() && p == Priority::High {
        p = Priority::Normal;
    }
    let client = client_profile.unwrap_or_else(|| if own_page { "web".into() } else { "api".into() });
    crate::queue::PRIORITY.scope(p, crate::queue::CLIENT.scope(client, next.run(req))).await
}

/// Requests waiting for the GPU's turn from which an answer carries a "busy" warning.
const BUSY_WAITING: usize = 4;

/// Every 503 and 429 gets a Retry-After (a client backs off instead of hammering a PC that is busy or short of
/// memory); every API answer made while the PC is short of memory or the queue is long carries `x-strata-warning`.
async fn overload_headers(axum::extract::State(s): axum::extract::State<AppState>, req: axum::extract::Request, next: axum::middleware::Next) -> axum::response::Response {
    let api = { let p = req.uri().path(); p.starts_with("/v1") || p.starts_with("/api/inference") };
    let mut r = next.run(req).await;
    let st = r.status();
    if (st == StatusCode::SERVICE_UNAVAILABLE || st == StatusCode::TOO_MANY_REQUESTS) && !r.headers().contains_key("retry-after") {
        r.headers_mut().insert("retry-after", axum::http::HeaderValue::from_static("30"));
    }
    if api {
        let (avail, total) = (mem::available_gb(), mem::total_gb());
        let waiting = s.mgr.queue.waiting();
        let mut w = Vec::new();
        if mem::low(avail, total) {
            w.push(format!("low memory: {avail:.1} GB free"));
            s.mgr.monitor.notice("this PC is low on free memory: new models and media jobs may wait, or be refused");
        }
        if waiting >= BUSY_WAITING {
            w.push(format!("busy: {waiting} requests waiting for their turn"));
            s.mgr.monitor.notice("the server is busy: requests are waiting for their turn");
        }
        if let (false, Ok(v)) = (w.is_empty(), axum::http::HeaderValue::from_str(&w.join("; "))) {
            r.headers_mut().insert("x-strata-warning", v);
        }
    }
    r
}

/// A panic in a handler, as a 500 in the APIs' error shape (the panic itself is logged).
fn panic_answer(err: Box<dyn std::any::Any + Send + 'static>) -> axum::http::Response<axum::body::Body> {
    let what = err.downcast_ref::<String>().cloned().or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
    tracing::error!("a request handler panicked: {what}");
    error(StatusCode::INTERNAL_SERVER_ERROR, "server_error",
          "the server hit an internal error on this request (its log has the details); it keeps running: send the request again")
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

/// Who may use the model APIs (chat, messages, responses, Gemini, media): the main key (Some(the "default" profile,
/// if the run config has one)), or a client profile's key (Some(Some(that profile))). None: neither. The server's
/// settings stay the main key's (`authorized`).
pub fn api_access(state: &AppState, headers: &HeaderMap) -> Option<Option<clients::Profile>> {
    if authorized(state, headers) {
        return Some(state.clients.default.clone());
    }
    let k = clients::given_key(headers)?;
    // a key this worker minted for its master (cluster.rs): the model APIs, no tools of a profile
    if state.cluster.is_master_key(k, cluster_tls::peer_of(headers)) {
        return Some(None);
    }
    state.clients.by_key(k).map(Some)
}

/// The main key, or (on a worker) a key it gave its master: what the master polls, the model list and the queue.
pub fn authorized_or_master(state: &AppState, headers: &HeaderMap) -> bool {
    authorized(state, headers) || clients::given_key(headers).is_some_and(|k| state.cluster.is_master_key(k, cluster_tls::peer_of(headers)))
}

pub fn api_authorized(state: &AppState, headers: &HeaderMap) -> bool {
    api_access(state, headers).is_some()
}

/// The key in use when the default model's run config has none: the newest other run config's `api_key` (the key the
/// server ran with before its default model changed), so a new default never leaves the server open by accident.
fn other_key(models: &[manager::ModelEntry], default: &str) -> Option<String> {
    let mut with_key: Vec<&manager::ModelEntry> = models.iter()
        .filter(|m| m.id != default && m.cfg.api_key.as_deref().is_some_and(|k| !k.is_empty())).collect();
    with_key.sort_by_key(|m| std::cmp::Reverse(std::fs::metadata(&m.path).and_then(|md| md.modified()).ok()));
    let m = with_key.first()?;
    let distinct: std::collections::HashSet<_> = with_key.iter().filter_map(|m| m.cfg.api_key.as_deref()).collect();
    tracing::warn!("{default}'s run config has no api_key: using the one in {} (set \"api_key\" there to choose){}",
                   m.path.display(), if distinct.len() > 1 { "; the run configs hold different keys, the newest was taken" } else { "" });
    m.cfg.api_key.clone()
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

/// `~/` expanded (for the modules that take paths from requests or the run config).
pub fn expand_path(p: &str) -> PathBuf {
    expand(p)
}

fn expand(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(rest) => std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).map(|h| PathBuf::from(h).join(rest)).unwrap_or_else(|_| PathBuf::from(p)),
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
    // `am2strata agent ...`: the coding agents' command lives in am2strata-run, beside this binary
    if matches!(std::env::args().nth(1).as_deref(), Some("agent" | "agentrun")) {
        let exe = std::env::current_exe()?.with_file_name(if cfg!(windows) { "am2strata-run.exe" } else { "am2strata-run" });
        let status = std::process::Command::new(&exe).args(std::env::args_os().skip(1)).status()
            .map_err(|e| anyhow::anyhow!("{}: {e}", exe.display()))?;
        std::process::exit(status.code().unwrap_or(1));
    }
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
    let api_key = args.api_key.or(first.cfg.api_key.clone()).filter(|k| !k.is_empty())
        .or_else(|| other_key(&models, &default));
    for m in &models {
        tracing::info!(id = %m.id, context = m.cfg.max_context(), images = m.cfg.vision.is_some(), "installed model");
    }
    // queue_max_wait_s (the default model's run config): how long a job may wait for its turn before it is refused
    let max_wait = first.cfg.rest.get("queue_max_wait_s").and_then(|v| v.as_f64()).filter(|v| *v > 0.0).map(Duration::from_secs_f64);
    let mgr = Arc::new(Manager::new(models, dir.clone(), default.clone(), Duration::from_secs(args.ready_timeout), expand(&args.data_dir), max_wait));
    let switch = first.cfg.rest.get("switch_on_request") == Some(&serde_json::Value::Bool(true));
    mgr.switch_on_request.store(switch, std::sync::atomic::Ordering::Relaxed);
    for g in &mgr.hw.gpus {
        tracing::info!("GPU {}: {} ({} MiB, sm_{})", g.index, g.name, g.vram_mib, g.arch);
    }
    tracing::info!("RAM {} MiB, {} CPUs; video frames {}", mgr.hw.ram_mib, mgr.hw.cpus,
                   if media::ffmpeg_available() { "with ffmpeg" } else { "need ffmpeg (sudo apt install ffmpeg)" });
    if !args.lazy {
        mgr.load(&default).await.with_context(|| format!("loading {default}"))?;
    }
    // the run config's MCP servers and the app's (Settings › Tools)
    let hub = mcp::McpHub::from_config(&first.cfg.rest, Some(expand(&args.data_dir).join("mcp-servers.json"))).map_err(anyhow::Error::msg)?;
    hub.start();
    let trusted_origins: Vec<String> = first.cfg.rest.get("trusted_origins").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.trim_end_matches('/').to_string())).collect()).unwrap_or_default();
    let monitor_on = args.api_monitor || first.cfg.rest.get("api_monitor") == Some(&serde_json::Value::Bool(true));
    if monitor_on {
        tracing::info!("API monitor on: the last 100 API calls are kept in memory, at /api-monitor");
    }
    let extra_hosts: Vec<String> = first.cfg.rest.get("allowed_hosts").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
    let hosts = Hosts::new(&trusted_origins, &extra_hosts);
    // HTTPS (server_tls.rs): checked now so a wrong "tls" block stops the start; turned on once the app exists
    let tls_cfg = server_tls::Config::from_config(&first.cfg.rest).map_err(anyhow::Error::msg)?;
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
    // client profiles: an API key per client, with its tools (clients.rs)
    // keys made while the server runs (api_keys.rs): each is a client profile too
    let keys = api_keys::ApiKeys::open(Some(&expand(&args.data_dir)));
    keys.start_flush();
    let clients = clients::Clients::from_config(&first.cfg.rest, api_key.as_deref()).map_err(anyhow::Error::msg)?.with_generated(keys.clone());
    if !clients.is_empty() {
        tracing::info!("client profiles: {}", clients.names().join(", "));
    }
    // the models to try, in order, when the one asked for cannot load
    *mgr.preferences.write().unwrap() = first.cfg.rest.get("model_preferences").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
    // quick switching: the switch costs learnt before, and the files of a model a job waits for warmed up
    mgr.seed_switch_costs();
    {
        let mgr = mgr.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(3)).await;
                mgr.warm_wanted();
            }
        });
    }
    let cluster = Arc::new(match cluster::Cluster::from_config(&first.cfg.rest, Some(&expand(&args.data_dir))) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("am2strata: {e}");
            std::process::exit(2);
        }
    });
    if !cluster.is_empty() {
        tracing::info!("cluster: master of {}", cluster.workers().iter().map(|w| format!("{} ({})", w.name, w.url)).collect::<Vec<_>>().join(", "));
    }
    cluster.start(port);
    let clients = Arc::new(clients);
    let _ = mgr.clients.set(clients.clone());
    let state = AppState { mgr: mgr.clone(), api_key, clients: clients.clone(), keys: keys.clone(), mcp: Some(hub.clone()), trusted_origins,
                           api_monitor: api_monitor::ApiMonitor::new(monitor_on, Some(expand(&args.data_dir).join("api-calls"))),
                           hub_job: Arc::new(std::sync::Mutex::new(json!({"state": "idle"}))),
                           models_dir: dir.clone(), data_dir: expand(&args.data_dir),
                           code_run: Arc::new(code_run),
                           code_run_config_on: first.cfg.rest.get("code_run") != Some(&serde_json::Value::Bool(false)),
                           previews: Arc::new(preview::Previews::default()),
                           cluster: cluster.clone(),
                           media: Arc::new(image_gen::Media::new(&expand(&args.data_dir), &first.cfg.rest)),
                           media_jobs: Arc::new(media_jobs::Jobs::new()) };
    // media jobs kept across restarts; the ones the server stopped in run again
    let unfinished = state.media_jobs.open(expand(&args.data_dir).join("media-jobs"));
    state.media_jobs.bind(state.clone());
    let _ = mgr.code.set(state.code_run.clone());
    let _ = mgr.media_check.set(Arc::new(media_check::Checker::new(Arc::new(state.clone()))));
    let _ = mgr.media_dub.set(Arc::new(media_dub::Dubbing::new(Arc::new(state.clone()))));
    state.media_jobs.resume(unfinished);
    let _ = mgr.media_jobs.set(state.media_jobs.clone());
    let _ = mgr.web.set(Arc::new(web_tools::WebTools::from_config(&first.cfg.rest)));
    let _ = mgr.browser.set(Arc::new(browser::Browsers::new(browser::Settings::from_config(&first.cfg.rest, &expand(&args.data_dir)))));
    // the sandbox VM (computer/vm): made before computer use, whose "vm" mode drives it; it starts only when used
    #[cfg(target_os = "linux")]
    computer::vm::Vm::init(&expand(&args.data_dir));
    // desktop computer use: the mode and preset from the run config (off by default; the backend is this platform's)
    let _ = mgr.computer.set(Arc::new(computer::Computers::new(computer::settings::Config::from_config(&first.cfg.rest, &expand(&args.data_dir)))));
    // the Workshop: convert, quantize and merge jobs (workshop.rs)
    workshop::Workshop::init(&dir, &expand(&args.data_dir), &first.cfg.rest).set_server(&host, port, state.api_key.clone());
    let web_dir = match args.web_dir.map(|p| expand(&p)) {
        Some(d) => d,
        None => own_web_dir().context("the am2Strata web app (web/ in the am2strata folder) was not found; pass --web-dir")?,
    };
    let app = Router::new()
        .route("/health", get(web::health))
        .route("/api/health", get(unsloth::health))
        .route("/api/inference/loaded-models", get(unsloth::loaded_models))
        .route("/api/inference/status", get(unsloth::status))
        .route("/api/inference/load", post(unsloth::load))
        .route("/v1/models", get(openai::models))
        .route("/v1/chat/completions", post(openai::chat))
        .route("/v1/responses", post(responses::responses))
        .route("/v1/responses/{*rest}", get(responses::not_stored).post(responses::not_stored).delete(responses::not_stored))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/api/system", get(web::system))
        .route("/api/models", get(web::list_models))
        .route("/api/models/current", get(web::current_model))
        .route("/api/models/load", post(web::load_model))
        .route("/api/models/unload", post(web::unload_model))
        .route("/api/queue", get(web::queue))
        .route("/api/gpus", get(web::gpus))
        .route("/api/machine", get(web::machine))
        .route("/api/cluster", get(cluster::status))
        .route("/api/cluster/pair", post(cluster::pair))
        .route("/api/cluster/join", post(cluster::join))
        .route("/api/cluster/forget", post(cluster::forget))
        .route("/api/cluster/workers/{name}", axum::routing::delete(cluster::remove))
        .route("/api/machine/measure", post(web::machine_measure))
        .route("/api/optimise", get(web::optimise_status))
        .route("/api/models/{id}/optimise", post(web::optimise))
        .route("/api/models/{id}/tuned", post(web::set_tuned))
        .route("/api/tools/approve", post(browser::approve))
        .route("/api/cache/snapshots/clear", post(web::clear_snapshots))
        .route("/api/models/{id}", axum::routing::delete(model_remove::remove))
        .route("/api/models/{id}/removal", get(model_remove::removal))
        .route("/api/media/models/{name}", axum::routing::delete(model_remove::media_remove))
        .route("/api/media/models/{name}/removal", get(model_remove::media_removal))
        .route("/api/media/capabilities", get(media_jobs::capabilities))
        .route("/api/media/genres", get(music::genres))
        .route("/api/jobs", post(media_jobs::again))
        .route("/v1/media/productions", post(media_jobs::productions))
        .route("/v1/media/videos/compose", post(compose::compose_videos))
        .route("/v1/media/thumbnails", post(compose::thumbnails))
        .route("/v1/media/check", post(media_check::check_endpoint))
        .route("/v1/media/dub", post(media_dub::dub_endpoint))
        .route("/api/jobs/{id}", get(media_jobs::status))
        .route("/v1/media/jobs/{id}", get(media_jobs::status))
        .route("/v1/media/jobs/{id}/content", get(media_jobs::content))
        .route("/v1/media/jobs/{id}/captions.srt", get(media_jobs::captions))
        .route("/api/models/updates", get(hub_api::updates))
        .route("/api/models/update", post(hub_api::update))
        .route("/api/requests", get(api_monitor::requests))
        .route("/api/requests/clear", post(api_monitor::clear))
        .route("/api/hub/quants", post(hub_api::quants))
        .route("/api/hub/pull", get(hub_api::status).post(hub_api::pull))
        .route("/api/run", get(code_run::info).post(code_run::run))
        .route("/v1/images/generations", post(image_gen::generations))
        .route("/v1/images/edits", post(image_gen::edits))
        .route("/v1/videos/generations", post(image_gen::video_generations))
        .route("/v1/audio/transcriptions", post(audio_api::transcriptions))
        .route("/v1/audio/translations", post(audio_api::translations))
        .route("/v1/audio/speech", post(audio_api::speech))
        .route("/v1/audio/music", post(music::music))
        .route("/api/media/audio/device", post(audio_api::set_device))
        .route("/api/memory", get(memory::list).post(memory::add))
        .route("/v1/memory", get(memory::v1_list).post(memory::v1_add))
        .route("/v1/memory/{id}", axum::routing::delete(memory::v1_delete))
        .route("/api/memory/settings", post(memory::set_settings))
        .route("/api/memory/{id}", axum::routing::delete(memory::delete))
        .route("/api/media/models", get(image_gen::models))
        .route("/api/media/models/pull", get(image_gen::pull_status).post(image_gen::pull))
        .route("/api/media/models/default", post(image_gen::set_default))
        .route("/api/media/licenses/accept", post(image_gen::accept_license))
        .route("/api/preview", post(preview::create))
        .route("/preview/{id}", get(preview::show))
        .route("/load", post(api_monitor::load))
        .route("/unload", post(api_monitor::unload))
        .route("/settings", get(web::settings_get).post(web::settings_post))
        .route("/config", get(web::config_get).post(web::config_post))
        .route("/v1/workshop/jobs", get(workshop::jobs_get).post(workshop::jobs_post))
        .route("/v1/workshop/jobs/{id}", get(workshop::job_get).delete(workshop::job_delete))
        .route("/v1/workshop/files", get(workshop::files_get))
        .route("/v1/workshop/uploads", axum::routing::post(workshop::upload_post))
        .route("/v1/workshop/dataset", get(workshop::dataset_get).post(workshop::dataset_post))
        .route("/api/tls", get(web::tls_get).post(web::tls_post))
        // the Computer tab: the sandbox desktop live, the operator's control and clipboard (computer/live.rs)
        .route("/api/computer/status", get(computer::live::status_get))
        .route("/api/computer/stream", get(computer::live::stream))
        .route("/api/computer/screenshot", get(computer::live::screenshot_get))
        .route("/api/computer/control", post(computer::live::control_post))
        .route("/api/computer/clipboard", get(computer::live::clipboard_get_h).post(computer::live::clipboard_post))
        .route("/api/computer/type", post(computer::live::type_post))
        .route("/api/computer/open", post(computer::live::open_post))
        .merge(computer::vm_routes())
        .route("/mcp", get(web::mcp))
        .route("/api/mcp/servers", get(mcp_api::list).post(mcp_api::add))
        .route("/api/mcp/servers/test", post(mcp_api::test))
        .route("/api/mcp/servers/{name}", axum::routing::put(mcp_api::update).delete(mcp_api::remove))
        .route("/api/mcp/servers/{name}/tools", post(mcp_api::tools))
        .route("/metrics", get(web::metrics))
        .route("/api/monitor/clear", post(web::monitor_clear))
        .route("/api/agents", get(web::agents))
        .route("/api/clients", get(web::clients))
        .route("/api/keys", get(api_keys::list).post(api_keys::create))
        .route("/api/keys/{id}", post(api_keys::update).delete(api_keys::revoke))
        .route("/api/usage", get(api_keys::usage))
        .route("/v1/usage", get(api_keys::own_usage))
        .route("/v1beta/models", get(gemini::models))
        .route("/v1beta/models/{target}", get(gemini::model).post(gemini::call))
        .route("/api/monitor/media/{name}", get(web::monitor_media))
        .route("/metrics/prometheus", get(web::prometheus))
        .route("/v1/status", get(web::status))
        // the live channel: one WebSocket for pushed updates and requests (live_channel.rs)
        .route("/v1/live", get(live_channel::live))
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
        // a chat whose model a worker should answer goes there (cluster.rs); inside the records, so they show it
        .layer(axum::middleware::from_fn_with_state(state.clone(), cluster::route))
        .layer(axum::middleware::from_fn_with_state(state.clone(), api_monitor::layer))
        // generated keys: a revoked or expired one refused, limits, and everyone's usage counted (api_keys.rs)
        .layer(axum::middleware::from_fn_with_state(state.clone(), api_keys::layer))
        // overload told plainly: Retry-After on every refusal for load, a warning header while the PC is short of memory
        // or the queue is long
        .layer(axum::middleware::from_fn_with_state(state.clone(), overload_headers))
        // each request's priority and client, for the queue (queue.rs)
        .layer(axum::middleware::from_fn_with_state(state.clone(), priority_scope))
        .layer(axum::extract::DefaultBodyLimit::max(256 << 20))
        // a handler that panics is answered with a JSON error, never a dropped connection
        .layer(tower_http::catch_panic::CatchPanicLayer::custom(panic_answer))
        // HTTPS: HSTS on TLS responses; plain HTTP from elsewhere than this PC sent to HTTPS (server_tls.rs). Inside
        // check_host, so a redirect only ever names a host this server answers to (no open redirect via a forged Host)
        .layer(axum::middleware::from_fn(server_tls::security_headers))
        .layer(axum::middleware::from_fn(server_tls::http_policy))
        .layer(axum::middleware::from_fn_with_state(hosts, check_host))
        // a TLS peer's pin (cluster_tls.rs): set only for the cluster listener's requests, removed from all others
        .layer(axum::middleware::from_fn(cluster_tls::peer_header))
        .with_state(state);
    // a worker: the encrypted listener for its paired masters (the same app)
    cluster.serve_tls(app.clone(), host.clone(), port);
    // HTTPS: the same app over TLS on its own port (8001 by default), changeable later from Settings or the CLI
    let worker_tls = if cluster.role == cluster::Role::Worker { cluster.tls_port(port) } else { None };
    let tls_rt = server_tls::Runtime::init(expand(&args.data_dir), host.clone(), port, worker_tls);
    tls_rt.set_app(app.clone());
    live_channel::set_app(app.clone());
    if tls_cfg.is_some() {
        if let Err(e) = tls_rt.apply(tls_cfg).await {
            tracing::error!("https: not started: {e} (plain HTTP keeps working as before)");
        }
    }
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await.with_context(|| format!("bind {host}:{port}"))?;
    tracing::info!("listening on http://{host}:{port}/ (web app from {}), API at /v1", web_dir.display());
    // each request's address is known (adding a command MCP server is allowed only from this PC)
    axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>())
        .with_graceful_shutdown(async {
            // Ctrl+C, or SIGTERM (how systemd stops the service): either way the clean shutdown below runs, so the
            // engine and the sandbox desktop's Xvfb stop with the server instead of being left running
            #[cfg(unix)]
            {
                let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok();
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = async { match term.as_mut() { Some(t) => { t.recv().await; } None => std::future::pending::<()>().await } } => {}
                }
            }
            #[cfg(not(unix))]
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    keys.usage.flush();
    mgr.unload().await;
    if let Some(c) = mgr.computer.get() {
        c.shutdown();
    }
    // the sandbox VM is saved, so it comes back as it was at the next start
    #[cfg(target_os = "linux")]
    if let Some(vm) = computer::vm::get() {
        if let Err(e) = tokio::task::spawn_blocking(move || vm.shutdown_for_exit()).await {
            tracing::warn!("sandbox VM: saving it at exit failed: {e}");
        }
    }
    if let Some(b) = mgr.browser.get() {
        b.close().await;
    }
    hub.close().await;
    Ok(())
}
