//! Remote models (docs/internal/REMOTE-MODELS.md): OpenAI-compatible and Anthropic-compatible providers beside the
//! local models, and the "mix" that picks one per request. Off unless the owner's run config has a `"remote"` block
//! (or the owner turns it on at /rtenai): then nothing here changes a request, a model list or a page.
//!
//! - A request names `<provider>/<model>` (or `mix`, `mix/<name>`). The layer here checks who may use it (the owner's
//!   key; a client profile or generated key only when its model list names it, `<provider>/*`, `remote/*` or the mix),
//!   picks the model of a mix, and adds `x-strata-routed` (the model that answered) and `x-strata-note`.
//! - The API handlers build the request as for a local model (the same front end), then `Target::Remote` sends it to
//!   the provider in its own API (translate.rs). The server's tool loop (run::start_loop) runs here as for a local
//!   model: memory, web, MCP, media and computer tools; the provider only writes text and calls.
//! - Remote requests never take the local GPU queue's turn; each provider has its own cap (`max_concurrent`).

pub mod api;
pub mod config;
pub mod translate;

use crate::manager::{Loaded, Manager};
use crate::run::{Refusal, RunEvent, RunOpts, Started};
use crate::AppState;
use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use config::{Config, KeyStore, Kind, Mix, Provider, When};
use futures::StreamExt;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};
use strata_frontend::Prepared;
use tokio::sync::{mpsc, Semaphore};

/// How long a provider's own model list is kept.
const LIST_FOR: Duration = Duration::from_secs(600);
/// How long a failed model list is kept before it is asked again.
const FAILED_LIST_FOR: Duration = Duration::from_secs(60);
/// A provider silent this long: the client is sent a keep-alive (one that left ends the call).
const KEEP_ALIVE: Duration = Duration::from_secs(3);

pub struct Remote {
    cfg: RwLock<Option<Config>>,
    pub keys: KeyStore,
    lists: Mutex<HashMap<String, (Instant, Result<Vec<String>, String>)>>,
    sems: Mutex<HashMap<String, (usize, Arc<Semaphore>)>>,
    http: reqwest::Client,
    /// the run config's client profiles' "models" lists (read for remote models only)
    profile_models: HashMap<String, Vec<String>>,
    /// the run config the block is saved to (the default model's)
    pub config_path: Option<PathBuf>,
}

static REMOTE: OnceLock<Arc<Remote>> = OnceLock::new();

/// At start: the default model's run config (`rest`), the data folder (remote-keys.json) and that config's path.
pub fn init(rest: &BTreeMap<String, Value>, data_dir: &Path, config_path: Option<PathBuf>) -> Result<(), String> {
    let cfg = match rest.get("remote").filter(|v| !v.is_null()) {
        Some(v) => Some(Config::parse(v)?),
        None => None,
    };
    let profile_models = rest.get("clients").and_then(Value::as_object).into_iter().flatten()
        .filter_map(|(name, p)| p.get("models").and_then(Value::as_array)
            .map(|a| (name.clone(), a.iter().filter_map(|m| m.as_str().map(str::to_string)).collect())))
        .collect();
    let r = Remote {
        cfg: RwLock::new(cfg), keys: KeyStore::open(Some(data_dir.join("remote-keys.json"))), lists: Mutex::new(HashMap::new()),
        sems: Mutex::new(HashMap::new()), profile_models, config_path,
        http: reqwest::Client::builder().connect_timeout(Duration::from_secs(15)).build().map_err(|e| e.to_string())?,
    };
    if let Some(c) = r.config() {
        tracing::info!("remote models: {} provider(s)", c.providers.len());
        for p in &c.providers {
            if let Err(e) = config::resolve_key(p, &r.keys) {
                tracing::warn!("remote models: {e}");
            }
        }
    }
    let _ = REMOTE.set(Arc::new(r));
    Ok(())
}

pub fn get() -> Option<&'static Arc<Remote>> {
    REMOTE.get()
}

/// The feature with its settings, when it is on.
fn on() -> Option<(&'static Arc<Remote>, Config)> {
    let r = get()?;
    let c = r.config()?;
    Some((r, c))
}

impl Remote {
    pub fn config(&self) -> Option<Config> {
        self.cfg.read().unwrap().clone()
    }

    pub fn set_config(&self, c: Option<Config>) {
        *self.cfg.write().unwrap() = c;
        self.lists.lock().unwrap().clear();
    }

    fn sem(&self, p: &Provider) -> Arc<Semaphore> {
        let mut g = self.sems.lock().unwrap();
        match g.get(&p.name) {
            Some((n, s)) if *n == p.max_concurrent => s.clone(),
            _ => {
                let s = Arc::new(Semaphore::new(p.max_concurrent));
                g.insert(p.name.clone(), (p.max_concurrent, s.clone()));
                s
            }
        }
    }

    fn auth(&self, rb: reqwest::RequestBuilder, p: &Provider, key: &str) -> reqwest::RequestBuilder {
        match p.kind {
            Kind::OpenAi => rb.bearer_auth(key),
            Kind::Anthropic => rb.header("x-api-key", key).header("anthropic-version", "2023-06-01"),
        }
    }

    /// The provider's own model list, asked now (GET /models; Anthropic GET /v1/models).
    pub async fn fetch_models(&self, p: &Provider) -> Result<Vec<String>, String> {
        let (_, key) = config::resolve_key(p, &self.keys)?;
        let r = self.auth(self.http.get(p.url("/models")), p, &key).timeout(Duration::from_secs(15)).send().await
            .map_err(|e| format!("{}: not reachable ({})", p.name, scrub(&error_chain(&e), &key)))?;
        let status = r.status().as_u16();
        let v: Value = r.json().await.unwrap_or(Value::Null);
        if status == 401 || status == 403 {
            return Err(format!("{}: the provider refused the server's key (HTTP {status})", p.name));
        }
        if !(200..300).contains(&status) {
            return Err(format!("{}: HTTP {status}: {}", p.name, scrub(v["error"]["message"].as_str().unwrap_or(""), &key)));
        }
        let mut ids: Vec<String> = v["data"].as_array().into_iter().flatten().filter_map(|m| m["id"].as_str().map(str::to_string)).collect();
        ids.sort();
        ids.dedup();
        self.lists.lock().unwrap().insert(p.name.clone(), (Instant::now(), Ok(ids.clone())));
        Ok(ids)
    }

    /// The models a provider offers: the run config's, or its own list (kept 10 minutes).
    pub async fn models_of(&self, p: &Provider) -> Vec<String> {
        if !p.models.is_empty() {
            return p.models.clone();
        }
        if let Some((at, got)) = self.lists.lock().unwrap().get(&p.name) {
            let fresh = if got.is_ok() { LIST_FOR } else { FAILED_LIST_FOR };
            if at.elapsed() < fresh {
                return got.clone().unwrap_or_default();
            }
        }
        match self.fetch_models(p).await {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!("remote models: the list of {} could not be read: {e}", p.name);
                self.lists.lock().unwrap().insert(p.name.clone(), (Instant::now(), Err(e)));
                Vec::new()
            }
        }
    }
}

/// An error's text with its causes.
fn error_chain(e: &dyn std::error::Error) -> String {
    let mut s = e.to_string();
    let mut src = e.source();
    while let Some(c) = src {
        s.push_str(": ");
        s.push_str(&c.to_string());
        src = c.source();
    }
    s
}

/// `text` without the key (a provider's message may quote it).
fn scrub(text: &str, key: &str) -> String {
    let mut t = text.to_string();
    if key.len() >= 6 {
        t = t.replace(key, "[the key]");
    }
    // OpenAI quotes a key it refuses masked ("sk-ab***cd"): no part of a key goes out
    t.split(' ').map(|w| if w.starts_with("sk-") || w.contains("***") { "[a key]" } else { w }).collect::<Vec<_>>().join(" ")
}

// ------------------------------------------------------------------ names and who may use them

/// `model` names a configured provider: `<provider>/<model>`.
fn provider_of<'a>(c: &'a Config, model: &str) -> Option<(&'a Provider, String)> {
    let (p, m) = model.split_once('/')?;
    let p = c.provider(p)?;
    (!m.is_empty()).then(|| (p, m.to_string()))
}

fn is_mix_name(model: &str) -> bool {
    model == "mix" || model.starts_with("mix/")
}

/// Whether a key's model list entry `grant` covers `model` (beyond the same name): `remote/*`, `<provider>/*`, `mix/*`.
/// For api_keys.rs's check of a generated key's list; false while the feature is off.
pub fn grant_covers(grant: &str, model: &str) -> bool {
    let Some((_, c)) = on() else { return false };
    // a Gemini path's model comes percent-encoded (openai%2Fgpt-5.5)
    covers(&c, grant, &percent_decoded(model))
}

fn covers(c: &Config, grant: &str, model: &str) -> bool {
    let g = grant.trim().to_ascii_lowercase();
    let m = model.to_ascii_lowercase();
    if g == m {
        return true;
    }
    if is_mix_name(model) {
        return g == "mix/*" && m.starts_with("mix/");
    }
    if provider_of(c, model).is_none() {
        return false;
    }
    g == "remote/*" || g.strip_suffix("/*").is_some_and(|p| m.split('/').next() == Some(p))
}

/// Who sends a request, for remote models.
struct Caller {
    owner: bool,
    /// a client profile's or generated key's model list
    grants: Vec<String>,
    /// the names a mix rule's "key" may give: the key's id and name, the profile's name, or "owner"
    names: Vec<String>,
}

impl Caller {
    fn may(&self, c: &Config, model: &str) -> bool {
        self.owner || self.grants.iter().any(|g| covers(c, g, model))
    }
}

fn caller(s: &AppState, headers: &HeaderMap, query_key: Option<&str>) -> Option<Caller> {
    let given = crate::clients::given_key(headers).or(query_key);
    if let Some(k) = given {
        if let crate::api_keys::Lookup::Active(rec) = s.keys.lookup(k, crate::now()) {
            return Some(Caller { owner: false, grants: rec.models.clone().unwrap_or_default(), names: vec![rec.id.clone(), rec.name.clone()] });
        }
        if let Some(p) = s.clients.config_by_key(k) {
            let grants = get().and_then(|r| r.profile_models.get(&p.name).cloned()).unwrap_or_default();
            return Some(Caller { owner: false, grants, names: vec![p.name.clone()] });
        }
        if s.cluster.is_master_key(k, crate::cluster_tls::peer_of(headers)) {
            return None; // a master spends its own keys, not this server's
        }
    }
    let mut h = headers.clone();
    if let (None, Some(k)) = (crate::clients::given_key(headers), query_key) {
        if let Ok(v) = HeaderValue::from_str(&format!("Bearer {k}")) {
            h.insert("authorization", v);
        }
    }
    crate::authorized(s, &h).then(|| Caller { owner: true, grants: Vec::new(), names: vec!["owner".into()] })
}

/// The remote models and mixes this caller may use, as (id, provider or None for a mix).
async fn visible(s: &AppState, headers: &HeaderMap) -> Vec<(String, Option<Provider>)> {
    let Some((r, c)) = on() else { return Vec::new() };
    let Some(who) = caller(s, headers, None) else { return Vec::new() };
    let mut out = Vec::new();
    for name in std::iter::once("mix".to_string()).chain(c.mixes.keys().map(|n| format!("mix/{n}"))) {
        if who.may(&c, &name) {
            out.push((name, None));
        }
    }
    for p in c.providers.iter().filter(|p| p.has_kind("chat") || p.has_kind("image")) {
        // a caller with no grant for this provider never makes it ask its list
        if !who.owner && !who.grants.iter().any(|g| g == "remote/*" || g.starts_with(&format!("{}/", p.name))) {
            continue;
        }
        for m in r.models_of(p).await {
            let id = format!("{}/{m}", p.name);
            if who.may(&c, &id) {
                out.push((id, Some(p.clone())));
            }
        }
    }
    out
}

/// /v1/models: the remote models and mixes this caller may use (none while the feature is off).
pub async fn v1_models(s: &AppState, headers: &HeaderMap) -> Vec<Value> {
    visible(s, headers).await.into_iter().map(|(id, p)| match p {
        Some(p) => json!({"id": id, "object": "model", "created": crate::now(), "owned_by": p.name, "remote": true, "provider": p.name,
                          "kinds": p.kinds, "context_length": p.context}),
        None => json!({"id": id, "object": "model", "created": crate::now(), "owned_by": "am2strata", "mix": true}),
    }).collect()
}

/// /api/models (the web app's picker): the same for the owner only.
pub async fn app_models(s: &AppState, headers: &HeaderMap) -> Vec<Value> {
    if !caller(s, headers, None).is_some_and(|c| c.owner) {
        return Vec::new();
    }
    visible(s, headers).await.into_iter().map(|(id, p)| match p {
        Some(p) => json!({"id": id, "display_name": id.split_once('/').map(|x| x.1).unwrap_or(&id), "remote": true, "provider": p.name,
                          "kinds": p.kinds, "context": p.context, "status": "remote", "current": false, "default": false}),
        None => json!({"id": id, "display_name": id, "mix": true, "status": "remote", "current": false, "default": false}),
    }).collect()
}

/// A Gemini path's model (`/v1beta/models/<model>:<method>`), names a remote model or a mix.
pub fn knows(model: &str) -> bool {
    on().is_some_and(|(_, c)| is_mix_name(model) || provider_of(&c, model).is_some())
}

// ------------------------------------------------------------------ the mix

/// What a mix rule looks at.
#[derive(Default, Debug)]
pub struct Facts {
    pub tokens: u64,
    pub images: bool,
    pub tools: bool,
    pub local_busy: bool,
    pub names: Vec<String>,
}

fn holds(w: &When, f: &Facts) -> bool {
    w.tokens_over.is_none_or(|n| f.tokens > n)
        && w.has_images.is_none_or(|b| b == f.images)
        && w.tools.is_none_or(|b| b == f.tools)
        && w.local_busy.is_none_or(|b| b == f.local_busy)
        && w.key.as_ref().is_none_or(|k| f.names.iter().any(|n| n.eq_ignore_ascii_case(k)))
}

/// The models to try, in order: the first rule that holds (else the default), and once more the next rule that
/// holds or the default.
pub fn plan(m: &Mix, f: &Facts) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for t in m.routes.iter().filter(|r| holds(&r.when, f)).map(|r| r.target.clone()).chain(std::iter::once(m.default.clone())) {
        if !out.contains(&t) {
            out.push(t);
        }
        if out.len() == 2 {
            break;
        }
    }
    out
}

fn has_images(v: &Value) -> bool {
    match v {
        Value::String(s) => s.starts_with("data:image/"),
        Value::Array(a) => a.iter().any(has_images),
        Value::Object(o) => {
            matches!(o.get("type").and_then(Value::as_str), Some("image_url" | "image" | "input_image"))
                || o.get("inlineData").or_else(|| o.get("inline_data")).and_then(|d| d.get("mimeType").or_else(|| d.get("mime_type")))
                    .and_then(Value::as_str).is_some_and(|m| m.starts_with("image/"))
                || o.iter().filter(|(k, _)| k.as_str() != "tools").any(|(_, v)| has_images(v))
        }
        _ => false,
    }
}

fn facts(s: &AppState, body: &Value, who: &Caller) -> Facts {
    let flags = ["strata_mcp", "strata_web", "strata_media", "strata_browser"].iter().any(|f| body.get(*f) == Some(&Value::Bool(true)));
    let queue = &s.mgr.queue;
    Facts {
        tokens: translate::estimate_tokens(body),
        images: has_images(body),
        tools: flags || body.get("tools").and_then(Value::as_array).is_some_and(|t| !t.is_empty()),
        local_busy: queue.waiting() > 0 || !matches!(queue.resident(), crate::queue::Resident::Text(_)),
        names: who.names.clone(),
    }
}

// ------------------------------------------------------------------ the request's route

/// What the layer and the handler know of one request's route.
#[derive(Default)]
struct RouteInfo {
    /// the models to try, in order ("local", a local model id, or <provider>/<model>)
    candidates: Vec<String>,
    routed: Option<String>,
    notes: Vec<String>,
    retry_after: Option<String>,
    error_kind: Option<&'static str>,
}

tokio::task_local! {
    static ROUTE: Arc<Mutex<RouteInfo>>;
}

fn with_route<R>(f: impl FnOnce(&mut RouteInfo) -> R) -> Option<R> {
    ROUTE.try_with(|r| f(&mut r.lock().unwrap())).ok()
}

fn is_remote_target(t: &str) -> bool {
    t.contains('/')
}

fn local_name(t: &str) -> Option<&str> {
    (t != "local").then_some(t)
}

/// This request goes to a remote model first (it takes no turn of the local queue).
pub fn planned() -> bool {
    with_route(|r| r.candidates.first().is_some_and(|c| is_remote_target(c))).unwrap_or(false)
}

/// The model a request runs on: the local one (as Manager::ensure_for), or the remote model its route names.
pub enum Target {
    Local(Arc<Loaded>),
    Remote(Vec<String>),
}

fn no_orders() -> &'static strata_frontend::ToolOrders {
    static O: OnceLock<strata_frontend::ToolOrders> = OnceLock::new();
    O.get_or_init(Default::default)
}

impl Target {
    pub fn vision(&self) -> bool {
        match self {
            Target::Local(l) => l.vision.is_some(),
            Target::Remote(_) => true,
        }
    }

    pub fn tool_orders(&self) -> &strata_frontend::ToolOrders {
        match self {
            Target::Local(l) => &l.tool_orders,
            Target::Remote(_) => no_orders(),
        }
    }

    /// The Chat settings shared with other apps (a local model's), and its config's effort.
    pub fn shared(&self) -> (crate::settings::Defaults, Option<String>) {
        match self {
            Target::Local(l) => (l.shared.read().unwrap().clone(), l.config_effort()),
            Target::Remote(_) => (Default::default(), None),
        }
    }

    pub fn id(&self) -> String {
        match self {
            Target::Local(l) => l.entry.id.clone(),
            Target::Remote(c) => c.first().cloned().unwrap_or_default(),
        }
    }

    pub fn is_local(&self) -> bool {
        matches!(self, Target::Local(_))
    }
}

/// Manager::ensure_for, or the remote model this request's route names first.
pub async fn ensure_for(s: &AppState, name: Option<&str>) -> anyhow::Result<(Target, Option<crate::queue::Pass>)> {
    let cands = with_route(|r| r.candidates.clone()).unwrap_or_default();
    match cands.first() {
        Some(c) if is_remote_target(c) => Ok((Target::Remote(cands), None)),
        Some(c) => {
            let (l, p) = s.mgr.ensure_for(local_name(c)).await?;
            with_route(|r| r.routed = Some(l.entry.id.clone()));
            Ok((Target::Local(l), p))
        }
        None => {
            let (l, p) = s.mgr.ensure_for(name).await?;
            Ok((Target::Local(l), p))
        }
    }
}

// ------------------------------------------------------------------ talking to a provider

/// A provider's refusal or silence, before anything was answered.
#[derive(Debug)]
pub struct ProviderError {
    pub status: u16,
    pub kind: &'static str,
    pub message: String,
    /// worth trying the next model of a mix (network, 5xx, 429)
    pub retryable: bool,
    pub retry_after: Option<String>,
    /// for x-strata-note
    pub short: String,
}

impl ProviderError {
    fn new(status: u16, kind: &'static str, message: String, retryable: bool, short: String) -> Self {
        ProviderError { status, kind, message, retryable, retry_after: None, short }
    }

    async fn of(p: &Provider, r: reqwest::Response, key: &str) -> ProviderError {
        let status = r.status().as_u16();
        let retry_after = r.headers().get("retry-after").and_then(|v| v.to_str().ok()).map(str::to_string);
        let text = r.text().await.unwrap_or_default();
        let said = serde_json::from_str::<Value>(&text).ok()
            .and_then(|v| v["error"]["message"].as_str().or_else(|| v["message"].as_str()).map(str::to_string))
            .unwrap_or_else(|| text.chars().take(300).collect());
        let said = scrub(&said, key);
        let mut e = match status {
            401 => ProviderError::new(502, "api_error", format!("{}: the provider refused the server's key (HTTP 401); the server's owner \
                                                                 needs to check it", p.name), false, "HTTP 401".into()),
            403 => ProviderError::new(502, "api_error", format!("{}: the provider refused this request for the server's key (HTTP 403)", p.name),
                                      false, "HTTP 403".into()),
            429 => ProviderError::new(429, "rate_limit_error", format!("{}: the provider is rate limiting (HTTP 429): {said}", p.name), true,
                                      "HTTP 429".into()),
            500..=599 => ProviderError::new(status, "api_error", format!("{}: the provider failed (HTTP {status}): {said}", p.name), true,
                                            format!("HTTP {status}")),
            404 => ProviderError::new(404, "not_found_error", format!("{}: {said}", p.name), false, "HTTP 404".into()),
            _ => ProviderError::new(if (400..500).contains(&status) { status } else { 502 }, "invalid_request_error",
                                    format!("{}: the provider refused the request (HTTP {status}): {said}", p.name), false, format!("HTTP {status}")),
        };
        e.retry_after = retry_after;
        e
    }
}

/// One remote model for one request: each round of its tool loop is a call to the provider.
pub struct RemoteRound {
    mgr: Arc<Manager>,
    p: Provider,
    model: String,
    full: String,
    key: String,
    /// Anthropic's signed thinking blocks of this request's earlier rounds
    signed: Mutex<translate::Signatures>,
}

impl RemoteRound {
    fn new(mgr: Arc<Manager>, full: &str) -> Result<RemoteRound, ProviderError> {
        let Some((r, c)) = on() else {
            return Err(ProviderError::new(404, "not_found_error", format!("{full}: remote models are off"), false, "off".into()));
        };
        let Some((p, model)) = provider_of(&c, full) else {
            return Err(ProviderError::new(404, "not_found_error", format!("{full}: no such remote model"), false, "unknown".into()));
        };
        let (_, key) = config::resolve_key(p, &r.keys).map_err(|e| ProviderError::new(502, "api_error", e, false, "no key".into()))?;
        Ok(RemoteRound { mgr, p: p.clone(), model, full: full.to_string(), key, signed: Mutex::new(Vec::new()) })
    }

    /// The call, up to the provider's answer's start; the answer itself is read by a task of its own.
    async fn open(self: Arc<Self>, mut prep: Prepared, req: Value, opts: RunOpts) -> Result<Started, ProviderError> {
        let r = get().expect("remote models are set up");
        // the memory's block of facts and archived turns (the tools are offered as for a local model)
        if let Some(scope) = &opts.memory {
            let conv = crate::memory::conversation_id(&prep.messages);
            let query = prep.messages.iter().rev().find(|m| m["role"] == "user").map(|m| crate::memory::content_text(&m["content"])).unwrap_or_default();
            if let Some(b) = self.mgr.memory.block(&query, scope, conv.as_deref()) {
                let at = prep.messages.iter().take_while(|m| m["role"] == "system").count();
                prep.messages.insert(at, json!({"role": "system", "content": b}));
            }
        }
        let mut notes = Vec::new();
        let signed = self.signed.lock().unwrap().clone();
        let body = translate::body(self.p.kind, &self.model, &self.p, &prep, &req, &opts, &signed, &mut notes);
        if !notes.is_empty() {
            with_route(|r| r.notes.extend(notes));
        }
        let permit = r.sem(&self.p).acquire_owned().await.map_err(|_| ProviderError::new(503, "api_error", "closed".into(), true, "closed".into()))?;
        let path = match self.p.kind { Kind::OpenAi => "/chat/completions", Kind::Anthropic => "/messages" };
        let rq = r.auth(r.http.post(self.p.url(path)), &self.p, &self.key).timeout(Duration::from_secs(self.p.timeout_s)).json(&body);
        let resp = rq.send().await.map_err(|e| ProviderError::new(502, "api_error",
            format!("{}: the provider could not be reached ({})", self.p.name, scrub(&error_chain(&e), &self.key)), true, "not reachable".into()))?;
        if !resp.status().is_success() {
            return Err(ProviderError::of(&self.p, resp, &self.key).await);
        }
        let json_answer = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|c| c.starts_with("application/json"));
        let (tx, events) = mpsc::channel(256);
        let me = self.clone();
        let stream_tools = opts.stream_tools;
        tokio::spawn(async move {
            let _permit = permit; // the provider's place is held until the answer ends (or the client leaves)
            me.read(resp, json_answer, stream_tools, tx).await;
        });
        let prompt_tokens = translate::estimate_tokens(&Value::Array(prep.messages.clone())) as usize;
        Ok(Started { events, prompt_tokens, model: self.full.clone() })
    }

    /// The provider's answer as events, until it ends or the client goes (dropping the answer closes the connection,
    /// which ends it at the provider).
    async fn read(&self, resp: reqwest::Response, json_answer: bool, stream_tools: bool, tx: mpsc::Sender<RunEvent>) {
        if json_answer {
            let v: Value = tokio::select! {
                b = resp.bytes() => b.ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null),
                _ = tx.closed() => return,
            };
            let evs = match self.p.kind {
                Kind::OpenAi => translate::openai_json(&v),
                Kind::Anthropic => translate::anthropic_json(&v, &mut self.signed.lock().unwrap()),
            };
            for e in evs {
                if tx.send(e).await.is_err() {
                    return;
                }
            }
            return;
        }
        let (mut oa, mut an) = (translate::OpenAiStream::new(stream_tools), translate::AnthropicStream::new(stream_tools));
        let mut stream = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut complete = false;
        // something goes to the client at least every KEEP_ALIVE (a Ping when the provider sends no data, only its own
        // keep-alives or nothing), so a client that left is noticed and the call ends
        let mut last_sent = Instant::now();
        loop {
            let chunk = tokio::select! {
                c = stream.next() => c,
                _ = tx.closed() => return,
                _ = tokio::time::sleep_until((last_sent + KEEP_ALIVE).into()) => {
                    if tx.send(RunEvent::Ping).await.is_err() {
                        return;
                    }
                    last_sent = Instant::now();
                    continue;
                }
            };
            match chunk {
                None => break,
                Some(Err(e)) => {
                    let _ = tx.send(RunEvent::Error(format!("{}: the answer broke off ({})", self.p.name, scrub(&error_chain(&e), &self.key)))).await;
                    return;
                }
                Some(Ok(b)) => buf.extend_from_slice(&b),
            }
            while let Some(nl) = buf.iter().position(|c| *c == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                let line = String::from_utf8_lossy(&line);
                let Some(data) = line.trim_end().strip_prefix("data:") else { continue };
                let data = data.trim();
                if data == "[DONE]" {
                    complete = true;
                    continue;
                }
                let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
                let evs = match self.p.kind {
                    Kind::OpenAi => oa.feed(&v),
                    Kind::Anthropic => an.feed(&v),
                };
                for e in evs {
                    let stop = matches!(e, RunEvent::Error(_));
                    if tx.send(e).await.is_err() || stop {
                        return;
                    }
                    last_sent = Instant::now();
                }
            }
            if last_sent.elapsed() >= KEEP_ALIVE {
                if tx.send(RunEvent::Ping).await.is_err() {
                    return;
                }
                last_sent = Instant::now();
            }
        }
        let evs = match self.p.kind {
            Kind::OpenAi => oa.finish(complete),
            Kind::Anthropic => {
                self.signed.lock().unwrap().append(&mut an.signed);
                an.finish()
            }
        };
        for e in evs {
            if tx.send(e).await.is_err() {
                return;
            }
        }
    }
}

impl crate::run::Round for RemoteRound {
    fn round(self: Arc<Self>, prep: Prepared, req: Value, opts: RunOpts)
             -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Started, Refusal>> + Send>> {
        Box::pin(async move { self.open(prep, req, opts).await.map_err(|e| Refusal(e.status, e.message)) })
    }
    fn vision(&self) -> bool {
        true
    }
    fn context(&self) -> usize {
        self.p.context
    }
}

/// Start a request on a remote target: the first model of its route that answers (a mix tries the next once when one
/// fails before answering), with the server's tools in the tool loop when it was offered them.
pub async fn start(mgr: Arc<Manager>, cands: Vec<String>, mut prep: Prepared, req: Value, opts: RunOpts,
                   offered: crate::server_tools::Offered) -> Result<Started, Refusal> {
    translate::inline_images(&mut prep).await.map_err(|e| Refusal(400, e))?;
    for (i, c) in cands.iter().enumerate() {
        if !is_remote_target(c) {
            // a mix's fallback to this PC's model
            let (loaded, pass) = mgr.ensure_for(local_name(c)).await.map_err(|e| Refusal(503, format!("{e:#}")))?;
            *opts.pass.lock().unwrap() = pass;
            with_route(|r| r.routed = Some(loaded.entry.id.clone()));
            return match offered {
                Some((hub, names)) => crate::run::start_mcp(mgr, loaded, prep, req, opts, hub, names).await,
                None => crate::run::start(mgr, loaded, prep, req, opts).await,
            };
        }
        let got = match RemoteRound::new(mgr.clone(), c) {
            Ok(round) => {
                let round = Arc::new(round);
                round.clone().open(prep.clone(), req.clone(), opts.clone()).await.map(|first| (round, first))
            }
            Err(e) => Err(e),
        };
        match got {
            Ok((round, first)) => {
                with_route(|r| r.routed = Some(c.clone()));
                tracing::info!("remote: {c} answers");
                return match offered {
                    Some((hub, names)) => crate::run::start_loop(mgr, round, first, prep, req, opts, hub, names),
                    None => Ok(first),
                };
            }
            Err(e) if e.retryable && i + 1 < cands.len() => {
                tracing::warn!("remote: {c} did not answer ({}): {} is tried", e.short, cands[i + 1]);
                with_route(|r| r.notes.push(format!("{c} did not answer ({}): {} was tried instead", e.short, cands[i + 1])));
            }
            Err(e) => {
                tracing::warn!("remote: {c}: {}", e.message);
                with_route(|r| {
                    r.retry_after = e.retry_after.clone();
                    r.error_kind = Some(e.kind);
                    r.routed = Some(c.clone());
                });
                return Err(Refusal(e.status, e.message));
            }
        }
    }
    Err(Refusal(502, "no model to answer".into()))
}

// ------------------------------------------------------------------ the layer

#[derive(Clone, Copy, PartialEq)]
enum Family {
    OpenAi,
    Anthropic,
    Responses,
    Gemini,
    Images,
}

fn family_of(path: &str) -> Option<Family> {
    match path {
        "/v1/chat/completions" => Some(Family::OpenAi),
        "/v1/messages" => Some(Family::Anthropic),
        "/v1/responses" => Some(Family::Responses),
        "/v1/images/generations" => Some(Family::Images),
        p if p.starts_with("/v1beta/models/") && (p.ends_with(":generateContent") || p.ends_with(":streamGenerateContent")) => Some(Family::Gemini),
        _ => None,
    }
}

fn refuse(f: Family, status: StatusCode, kind: &str, msg: String) -> Response {
    match f {
        Family::Anthropic => (status, axum::Json(json!({"type": "error", "error": {"type": kind, "message": msg}}))).into_response(),
        Family::Gemini => (status, axum::Json(json!({"error": {"code": status.as_u16(), "message": msg, "status": kind}}))).into_response(),
        _ => crate::error(status, kind, msg),
    }
}

fn percent_decoded(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() && s.is_char_boundary(i + 3) {
            if let Ok(n) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(n);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The layer around the model APIs (inside the API monitor and the usage counts, so both see a remote answer as any
/// other): who may use a remote model or a mix, the mix's choice, the answer's x-strata-routed / x-strata-note.
pub async fn layer(axum::extract::State(s): axum::extract::State<AppState>, req: axum::extract::Request,
                   next: axum::middleware::Next) -> Response {
    let Some((r, c)) = on() else { return next.run(req).await };
    let path = req.uri().path().to_string();
    let Some(family) = family_of(&path).filter(|_| req.method() == axum::http::Method::POST) else { return next.run(req).await };
    let (mut parts, body) = req.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, 256 << 20).await else {
        return refuse(family, StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error", "the request is too large".into());
    };
    let mut v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let model = match family {
        Family::Gemini => path.strip_prefix("/v1beta/models/").and_then(|t| t.rsplit_once(':')).map(|(m, _)| percent_decoded(m))
            .map(|m| m.trim_start_matches("models/").to_string()),
        _ => v.get("model").and_then(Value::as_str).map(str::to_string),
    };
    let pass = |parts: axum::http::request::Parts, bytes: axum::body::Bytes| axum::extract::Request::from_parts(parts, Body::from(bytes));
    let Some(model) = model.filter(|m| is_mix_name(m) || provider_of(&c, m).is_some()) else {
        return next.run(pass(parts, bytes)).await;
    };
    let query_key = parts.uri.query().and_then(|q| q.split('&').find_map(|kv| kv.strip_prefix("key="))).map(percent_decoded);
    let Some(who) = caller(&s, &parts.headers, query_key.as_deref()) else {
        if crate::clients::given_key(&parts.headers).is_some_and(|k| s.cluster.is_master_key(k, crate::cluster_tls::peer_of(&parts.headers))) {
            return refuse(family, StatusCode::FORBIDDEN, "permission_error", format!("{model}: a cluster master may not use this server's remote models"));
        }
        // no key: the handler says so
        return next.run(pass(parts, bytes)).await;
    };
    if !who.may(&c, &model) {
        return refuse(family, StatusCode::FORBIDDEN, "permission_error", format!("this key may not use the model {model:?}"));
    }
    if family == Family::Images {
        return match provider_of(&c, &model) {
            Some((p, m)) if p.has_kind("image") && p.kind == Kind::OpenAi => images(r, p, &m, &model, v).await,
            Some((p, _)) => refuse(family, StatusCode::BAD_REQUEST, "invalid_request_error",
                                   format!("{model}: {} is not an image provider (\"kinds\": [\"image\"] on an OpenAI-compatible one)", p.name)),
            None => refuse(family, StatusCode::BAD_REQUEST, "invalid_request_error", format!("{model}: a mix routes chats, not images")),
        };
    }
    let cands = if is_mix_name(&model) {
        let Some(m) = c.mix_of(&model) else {
            return refuse(family, StatusCode::NOT_FOUND, "not_found_error", format!("{model}: no such mix in the run config's remote.mixes"));
        };
        plan(m, &facts(&s, &v, &who))
    } else {
        let (p, m) = provider_of(&c, &model).expect("checked");
        if !p.has_kind("chat") {
            return refuse(family, StatusCode::BAD_REQUEST, "invalid_request_error", format!("{model}: {} is not a chat provider", p.name));
        }
        if !p.models.is_empty() && !p.models.contains(&m) {
            return refuse(family, StatusCode::NOT_FOUND, "not_found_error",
                          format!("{model}: not one of {}'s models ({})", p.name, p.models.join(", ")));
        }
        vec![model.clone()]
    };
    // a mix's local model by name: the request names it (the handlers load the model a request names)
    let mut bytes = bytes;
    if let Some(id) = cands.first().filter(|c| !is_remote_target(c) && *c != "local") {
        match family {
            Family::Gemini => {
                let method = path.rsplit_once(':').map(|x| x.1).unwrap_or("generateContent");
                let pq = format!("/v1beta/models/{id}:{method}{}", parts.uri.query().map(|q| format!("?{q}")).unwrap_or_default());
                if let Ok(u) = pq.parse() {
                    parts.uri = u;
                }
            }
            _ => {
                v["model"] = json!(id);
                bytes = serde_json::to_vec(&v).unwrap_or_default().into();
            }
        }
        parts.headers.remove(axum::http::header::CONTENT_LENGTH);
    }
    let info = Arc::new(Mutex::new(RouteInfo { candidates: cands, ..Default::default() }));
    let mut resp = ROUTE.scope(info.clone(), next.run(pass(parts, bytes))).await;
    let (routed, notes, retry_after, kind) = {
        let g = info.lock().unwrap();
        (g.routed.clone(), g.notes.clone(), g.retry_after.clone(), g.error_kind)
    };
    if let Some(k) = kind.filter(|_| !resp.status().is_success()) {
        // the refusal in the client's shape, with the provider's kind of error
        let (p, body) = resp.into_parts();
        let b = axum::body::to_bytes(body, 1 << 20).await.unwrap_or_default();
        let mut e: Value = serde_json::from_slice(&b).unwrap_or(Value::Null);
        if e["error"]["type"].is_string() {
            e["error"]["type"] = json!(k);
        } else if e["error"]["status"].is_string() {
            e["error"]["status"] = json!(k);
        }
        resp = Response::from_parts(p, Body::from(if e.is_null() { b.to_vec() } else { serde_json::to_vec(&e).unwrap_or_default() }));
        resp.headers_mut().remove(axum::http::header::CONTENT_LENGTH);
    }
    let h = resp.headers_mut();
    if let Some(v) = routed.and_then(|m| HeaderValue::from_str(&m).ok()) {
        h.insert("x-strata-routed", v);
    }
    let ascii: String = notes.join("; ").chars().map(|c| if c.is_ascii() && !c.is_ascii_control() { c } else { ' ' }).collect();
    if let (false, Ok(v)) = (notes.is_empty(), HeaderValue::from_str(&ascii)) {
        h.insert("x-strata-note", v);
    }
    if let Some(v) = retry_after.and_then(|r| HeaderValue::from_str(&r).ok()) {
        h.insert("retry-after", v);
    }
    resp
}

/// POST /v1/images/generations for a remote image model: the request sent on, the provider's answer back.
async fn images(r: &Arc<Remote>, p: &Provider, model: &str, full: &str, mut v: Value) -> Response {
    let fail = |e: ProviderError| {
        let mut resp = crate::error(StatusCode::from_u16(e.status).unwrap_or(StatusCode::BAD_GATEWAY), e.kind, e.message);
        if let Some(v) = e.retry_after.and_then(|x| HeaderValue::from_str(&x).ok()) {
            resp.headers_mut().insert("retry-after", v);
        }
        resp
    };
    let key = match config::resolve_key(p, &r.keys) {
        Ok((_, k)) => k,
        Err(e) => return fail(ProviderError::new(502, "api_error", e, false, "no key".into())),
    };
    v["model"] = json!(model);
    let _permit = r.sem(p).acquire_owned().await;
    let resp = match r.auth(r.http.post(p.url("/images/generations")), p, &key).timeout(Duration::from_secs(p.timeout_s)).json(&v).send().await {
        Ok(x) => x,
        Err(e) => return fail(ProviderError::new(502, "api_error", format!("{}: the provider could not be reached ({})", p.name,
                                                                          scrub(&error_chain(&e), &key)), true, "not reachable".into())),
    };
    if !resp.status().is_success() {
        return fail(ProviderError::of(p, resp, &key).await);
    }
    let body = resp.bytes().await.unwrap_or_default();
    let mut out = Response::new(Body::from(body));
    out.headers_mut().insert("content-type", HeaderValue::from_static("application/json"));
    if let Ok(h) = HeaderValue::from_str(full) {
        out.headers_mut().insert("x-strata-routed", h);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config::parse(&json!({"providers": {
            "openai": {"kind": "openai", "base_url": "https://api.openai.com/v1", "models": ["gpt-5.5"]},
            "groq": {"kind": "openai", "base_url": "https://api.groq.com/openai/v1"}},
            "mix": {"default": "local", "routes": [
                {"when": {"tokens_over": 100}, "use": "openai/gpt-5.5"},
                {"when": {"has_images": true}, "use": "openai/gpt-5.5"},
                {"when": {"local_busy": true}, "use": "groq/llama"},
                {"when": {"key": "laptop"}, "use": "groq/llama"}]}})).unwrap()
    }

    #[test]
    fn a_mix_picks_the_first_rule_that_holds_and_one_fallback() {
        let c = cfg();
        let f = |tokens, images, busy, names: &[&str]| Facts { tokens, images, tools: false, local_busy: busy,
                                                               names: names.iter().map(|s| s.to_string()).collect() };
        assert_eq!(plan(&c.mix, &f(10, false, false, &["owner"])), ["local"]);
        assert_eq!(plan(&c.mix, &f(500, false, false, &[])), ["openai/gpt-5.5", "local"]);
        assert_eq!(plan(&c.mix, &f(500, true, true, &[])), ["openai/gpt-5.5", "groq/llama"], "the next rule that holds, once");
        assert_eq!(plan(&c.mix, &f(10, false, true, &[])), ["groq/llama", "local"]);
        assert_eq!(plan(&c.mix, &f(10, false, false, &["k1", "Laptop"])), ["groq/llama", "local"]);
    }

    #[test]
    fn grants_cover_remote_models_only_as_written() {
        let c = cfg();
        assert!(covers(&c, "remote/*", "openai/gpt-5.5") && covers(&c, "groq/*", "groq/x") && covers(&c, "openai/gpt-5.5", "openai/gpt-5.5"));
        assert!(!covers(&c, "groq/*", "openai/gpt-5.5") && !covers(&c, "remote/*", "mix") && !covers(&c, "remote/*", "mock-a"));
        assert!(covers(&c, "mix", "mix") && covers(&c, "mix/*", "mix/code") && !covers(&c, "mix/*", "mix") && !covers(&c, "remote/*", "nope/x"));
    }

    #[test]
    fn pictures_are_found_in_every_api_shape() {
        assert!(has_images(&json!({"messages": [{"content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})));
        assert!(has_images(&json!({"messages": [{"content": [{"type": "image", "source": {"type": "base64"}}]}]})));
        assert!(has_images(&json!({"contents": [{"parts": [{"inlineData": {"mimeType": "image/png", "data": "x"}}]}]})));
        assert!(!has_images(&json!({"messages": [{"content": "an image_url"}], "tools": [{"type": "image"}]})));
    }

    #[test]
    fn a_key_never_leaves_in_a_message() {
        assert_eq!(scrub("Incorrect API key provided: sk-ab***cd. See", "sk-abcdefcd"), "Incorrect API key provided: [a key] See");
        assert_eq!(scrub("bad key gsk_123456789", "gsk_123456789"), "bad key [the key]");
        assert_eq!(percent_decoded("openai%2Fgpt-5.5"), "openai/gpt-5.5");
    }
}
