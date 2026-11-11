//! Clusters, phase 1: request routing (docs/CLUSTER-PLAN.md, "Master and workers"). This server is the master: the
//! one people and apps talk to, holding the keys, profiles, memory, MCP servers and the queue. Its workers are other
//! am2Strata servers that run engines only. A chat request goes to a worker when this PC does not have its model and
//! a worker does, or when this PC is busy and a worker holds the same model loaded and idle; the answer is streamed
//! back through the master, which keeps its own records of it.
//!
//! The run config:
//! ```json
//! "cluster": {"workers": {"pc2": {"url": "http://192.168.8.50:8000", "key": "env:PC2_KEY"}}}
//! ```
//! - `key`: the worker's own API key (or `env:NAME`, read from the environment), so the worker answers only its master
//!   and whoever else holds its key. Give a worker a key of its own and do not share it.
//! - Traffic is plain HTTP with the key for now: across a network you do not trust, connect the machines through your
//!   own VPN (Tailscale, WireGuard), as CLUSTER-PLAN says. Paired ed25519 keys and encryption are the next step.
//! - Each worker is asked every 5 s for its models and queue. One that has not answered for 15 s gets no new work;
//!   a request it fails to take is answered here when this PC has the model, else refused plainly.

use axum::body::Body;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const POLL: Duration = Duration::from_secs(5);
const STALE: Duration = Duration::from_secs(15);

/// The APIs a worker can answer (a model's chat), routed by the request's "model".
pub const ROUTED: [&str; 4] = ["/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/completions"];

#[derive(Default, Clone)]
struct Seen {
    models: Vec<String>,
    current: Option<String>,
    waiting: usize,
    running: bool,
    at: Option<Instant>,
    error: Option<String>,
    routed: u64,
}

pub struct Worker {
    pub name: String,
    pub url: String,
    key: String,
    seen: Mutex<Seen>,
}

impl Worker {
    fn ready(&self) -> bool {
        self.seen.lock().unwrap().at.is_some_and(|t| t.elapsed() < STALE)
    }

    fn has(&self, model: &str) -> bool {
        self.seen.lock().unwrap().models.iter().any(|m| m == model)
    }

    fn view(&self) -> Value {
        let s = self.seen.lock().unwrap().clone();
        json!({"name": self.name, "url": self.url, "ready": self.ready(), "models": s.models, "loaded": s.current,
               "waiting": s.waiting, "running": s.running, "seen_s_ago": s.at.map(|t| t.elapsed().as_secs()),
               "error": s.error, "requests_routed": s.routed})
    }
}

#[derive(Default)]
pub struct Cluster {
    pub workers: Vec<Arc<Worker>>,
}

impl Cluster {
    pub fn from_config(rest: &std::collections::BTreeMap<String, Value>) -> Result<Cluster, String> {
        let mut c = Cluster::default();
        let Some(w) = rest.get("cluster").and_then(|c| c.get("workers")) else { return Ok(c) };
        let obj = w.as_object().ok_or("config cluster.workers: expected {\"name\": {\"url\": ..., \"key\": ...}}")?;
        for (name, v) in obj {
            let url = v["url"].as_str().map(|u| u.trim_end_matches('/').to_string())
                .filter(|u| u.starts_with("http://") || u.starts_with("https://"))
                .ok_or(format!("config cluster.workers.{name}.url: expected http://host:port"))?;
            let key = match v["key"].as_str() {
                Some(k) if k.starts_with("env:") => std::env::var(&k[4..]).map_err(|_| format!("config cluster.workers.{name}.key: {k} is not set"))?,
                Some(k) if !k.is_empty() => k.to_string(),
                _ => return Err(format!("config cluster.workers.{name}.key: the worker's API key (or env:NAME)")),
            };
            c.workers.push(Arc::new(Worker { name: name.clone(), url, key, seen: Mutex::new(Seen::default()) }));
        }
        Ok(c)
    }

    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// Ask each worker for its models and queue, every 5 s.
    pub fn start_polling(self: &Arc<Self>) {
        for w in self.workers.clone() {
            tokio::spawn(async move {
                let http = reqwest::Client::builder().timeout(Duration::from_secs(4)).build().expect("http client");
                loop {
                    let get = |p: &str| http.get(format!("{}{p}", w.url)).bearer_auth(&w.key).send();
                    let r = async {
                        let m: Value = get("/api/models").await?.error_for_status()?.json().await?;
                        let q: Value = get("/api/queue").await?.error_for_status()?.json().await?;
                        Ok::<_, reqwest::Error>((m, q))
                    }.await;
                    {
                        let mut s = w.seen.lock().unwrap();
                        match r {
                            Ok((m, q)) => {
                                let list = m["models"].as_array().cloned().unwrap_or_default();
                                s.models = list.iter().filter_map(|x| x["id"].as_str().map(str::to_string)).collect();
                                s.current = list.iter().find(|x| x["current"] == true).and_then(|x| x["id"].as_str().map(str::to_string));
                                s.waiting = q["waiting"].as_array().map(Vec::len).unwrap_or(0);
                                s.running = q["running"].is_object();
                                s.at = Some(Instant::now());
                                s.error = None;
                            }
                            Err(e) => s.error = Some(e.to_string()),
                        }
                    }
                    tokio::time::sleep(POLL).await;
                }
            });
        }
    }

    /// The worker for a request for `model`: `local` says whether this PC has the model, `local_busy` whether its
    /// queue has work waiting. None: answer here.
    pub fn pick(&self, model: &str, local: bool, local_busy: bool) -> Option<Arc<Worker>> {
        let mut c: Vec<&Arc<Worker>> = self.workers.iter().filter(|w| w.ready() && w.has(model)).collect();
        if local {
            // only to take load off this PC: a worker with the model loaded and nothing to do
            if !local_busy {
                return None;
            }
            c.retain(|w| { let s = w.seen.lock().unwrap(); s.current.as_deref() == Some(model) && !s.running && s.waiting == 0 });
        }
        // the one that will answer soonest: the model loaded first, then the shortest queue
        c.sort_by_key(|w| { let s = w.seen.lock().unwrap(); (s.current.as_deref() != Some(model), s.waiting + s.running as usize) });
        c.first().map(|w| (*w).clone())
    }

    pub fn view(&self) -> Value {
        json!({"role": if self.workers.is_empty() { "single" } else { "master" },
               "workers": self.workers.iter().map(|w| w.view()).collect::<Vec<_>>()})
    }
}

/// The request sent on to `w`, its answer streamed back. Err: the worker could not be reached (the caller answers
/// here when it can).
pub async fn forward(w: &Worker, path_and_query: &str, headers: &HeaderMap, body: axum::body::Bytes) -> Result<Response, String> {
    let http = reqwest::Client::builder().connect_timeout(Duration::from_secs(5)).build().map_err(|e| e.to_string())?;
    let mut rq = http.post(format!("{}{path_and_query}", w.url)).bearer_auth(&w.key).body(body);
    // what shapes the answer; never the client's own key
    for h in ["content-type", "accept", "anthropic-version", "anthropic-beta", "openai-beta", "x-strata-priority"] {
        if let Some(v) = headers.get(h) {
            rq = rq.header(h, v.as_bytes());
        }
    }
    let r = rq.send().await.map_err(|e| e.to_string())?;
    w.seen.lock().unwrap().routed += 1;
    let status = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let ctype = r.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("application/json").to_string();
    let stream = futures::StreamExt::map(r.bytes_stream(), |c| c.map_err(std::io::Error::other));
    let mut resp = Response::new(Body::from_stream(stream));
    *resp.status_mut() = status;
    resp.headers_mut().insert("content-type", ctype.parse().unwrap_or(axum::http::HeaderValue::from_static("application/json")));
    resp.headers_mut().insert("x-strata-worker", w.name.parse().unwrap_or(axum::http::HeaderValue::from_static("worker")));
    Ok(resp)
}

/// The layer: a chat request whose model a worker should answer goes there.
pub async fn route(axum::extract::State(s): axum::extract::State<crate::AppState>, req: axum::extract::Request,
                   next: axum::middleware::Next) -> Response {
    let path = req.uri().path().to_string();
    if s.cluster.is_empty() || req.method() != axum::http::Method::POST || !ROUTED.contains(&path.as_str())
        || !crate::api_authorized(&s, req.headers()) {
        return next.run(req).await;
    }
    let (parts, body) = req.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, 256 << 20).await else {
        return crate::error(StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error", "the request is too large");
    };
    let model = serde_json::from_slice::<Value>(&bytes).ok().and_then(|v| v["model"].as_str().map(str::to_string));
    if let Some(model) = model {
        let local = s.mgr.entry(&model).is_some();
        let busy = s.mgr.queue.snapshot()["waiting"].as_array().is_some_and(|a| !a.is_empty());
        if let Some(w) = s.cluster.pick(&model, local, busy) {
            let pq = parts.uri.path_and_query().map(|p| p.as_str().to_string()).unwrap_or(path.clone());
            match forward(&w, &pq, &parts.headers, bytes.clone()).await {
                Ok(r) => {
                    tracing::info!("{path}: {model} answered by worker {}", w.name);
                    return r;
                }
                Err(e) => {
                    w.seen.lock().unwrap().error = Some(e.clone());
                    tracing::warn!("worker {} did not take the request ({e})", w.name);
                    if !local {
                        return crate::error(StatusCode::SERVICE_UNAVAILABLE, "server_error",
                                            format!("{model} is on worker {}, which did not answer ({e}); no other machine has it", w.name));
                    }
                }
            }
        }
    }
    next.run(axum::extract::Request::from_parts(parts, Body::from(bytes))).await
}

/// GET /api/cluster: the workers, what they hold and whether they answer.
pub async fn status(axum::extract::State(s): axum::extract::State<crate::AppState>, headers: HeaderMap) -> Response {
    if !crate::authorized(&s, &headers) {
        return crate::error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    axum::Json(s.cluster.view()).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(name: &str, models: &[&str], current: Option<&str>, waiting: usize, seen: bool) -> Arc<Worker> {
        Arc::new(Worker { name: name.into(), url: "http://x".into(), key: "k".into(), seen: Mutex::new(Seen {
            models: models.iter().map(|m| m.to_string()).collect(), current: current.map(str::to_string), waiting,
            running: false, at: seen.then(Instant::now), error: None, routed: 0 }) })
    }

    #[test]
    fn a_request_goes_where_its_model_is_and_answers_soonest() {
        let c = Cluster { workers: vec![worker("a", &["big"], None, 0, true), worker("b", &["big", "tiel"], Some("big"), 1, true),
                                        worker("gone", &["big"], Some("big"), 0, false)] };
        assert_eq!(c.pick("big", false, false).unwrap().name, "b", "the model loaded beats an empty queue; a silent one is left out");
        assert!(c.pick("nowhere", false, false).is_none());
        // a model this PC has: here, unless this PC is busy and a worker has it loaded and idle
        assert!(c.pick("tiel", true, true).is_none(), "loaded nowhere idle");
        let c = Cluster { workers: vec![worker("idle", &["tiel"], Some("tiel"), 0, true)] };
        assert!(c.pick("tiel", true, false).is_none(), "this PC is free: here");
        assert_eq!(c.pick("tiel", true, true).unwrap().name, "idle");
    }

    #[test]
    fn workers_are_read_from_the_config() {
        let mut rest = std::collections::BTreeMap::new();
        rest.insert("cluster".to_string(), json!({"workers": {"pc2": {"url": "http://10.0.0.2:8000/", "key": "k2-0123456789"}}}));
        let c = Cluster::from_config(&rest).unwrap();
        assert_eq!((c.workers[0].name.as_str(), c.workers[0].url.as_str()), ("pc2", "http://10.0.0.2:8000"));
        rest.insert("cluster".to_string(), json!({"workers": {"pc2": {"url": "10.0.0.2"}}}));
        assert!(Cluster::from_config(&rest).is_err());
    }
}
