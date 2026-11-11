//! Clusters (docs/CLUSTER-PLAN.md, "Master and workers"). A master is the am2Strata people and apps talk to, holding
//! the keys, profiles, memory, MCP servers and the queue. Its workers are other am2Strata servers that run engines
//! only. A chat request goes to a worker when this PC does not have its model and a worker does, or when this PC is
//! busy and a worker holds the same model loaded and idle; the answer is streamed back through the master, which
//! keeps its own records of it (phase 1, request routing).
//!
//! Joining (part 1): nothing joins by itself.
//! - A worker (`"cluster": {"role": "worker"}`) has an ed25519 identity (`<data>/node-key`, 0600; its node id is the
//!   key's fingerprint), announces itself on the LAN by mDNS (`_am2strata._tcp`), and shows a 6-digit pairing code
//!   that changes every 10 minutes (in its log and its own Settings › Cluster, never to a caller without its key).
//! - The master lists the workers it finds as "wants to join". The user types the worker's code on the master
//!   (`POST /api/cluster/pair`); the master sends it to the worker (`POST /api/cluster/join`), which, for the right
//!   code, mints an API key for that master alone and keeps only the key's hash (`<data>/cluster-masters.json`,
//!   0600). Five wrong codes lock pairing for 10 minutes. The master keeps the worker and its key in
//!   `<data>/cluster-workers.json` (0600) and starts using it at once.
//! - A master's key opens the worker's model APIs and its model list and queue (what the master polls), never its
//!   settings; the worker's own main key never leaves it. Removing a worker on the master
//!   (`DELETE /api/cluster/workers/{name}`) also tells the worker to forget that master.
//! - Workers can also be given in the run config:
//!   `"cluster": {"workers": {"pc2": {"url": "http://192.168.8.50:8000", "key": "env:PC2_KEY"}}}`.
//! - Traffic is plain HTTP with the key for now: across a network you do not trust, connect the machines through your
//!   own VPN (Tailscale, WireGuard). Encryption with the paired keys is the next step.
//! - Each worker is asked every 5 s for its models and queue. One that has not answered for 15 s gets no new work;
//!   a request it fails to take is answered here when this PC has the model, else refused plainly.
//! - `AM2STRATA_NO_MDNS=1` turns mDNS off (tests, or a network where it is not wanted).

use axum::body::Body;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

const POLL: Duration = Duration::from_secs(5);
const STALE: Duration = Duration::from_secs(15);
const CODE_LIFE: Duration = Duration::from_secs(600);
const LOCK_AFTER: u32 = 5;
const LOCK_FOR: Duration = Duration::from_secs(600);
pub const SERVICE: &str = "_am2strata._tcp.local.";

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
    /// the worker's node id, when it was paired (not for run-config workers)
    pub node_id: Option<String>,
    /// paired from the app (kept in cluster-workers.json), not from the run config
    pub paired: bool,
    seen: Mutex<Seen>,
    stop: AtomicBool,
}

impl Worker {
    fn new(name: String, url: String, key: String, node_id: Option<String>, paired: bool) -> Arc<Worker> {
        Arc::new(Worker { name, url, key, node_id, paired, seen: Mutex::new(Seen::default()), stop: AtomicBool::new(false) })
    }

    fn ready(&self) -> bool {
        self.seen.lock().unwrap().at.is_some_and(|t| t.elapsed() < STALE)
    }

    fn has(&self, model: &str) -> bool {
        self.seen.lock().unwrap().models.iter().any(|m| m == model)
    }

    fn view(&self) -> Value {
        let s = self.seen.lock().unwrap().clone();
        json!({"name": self.name, "url": self.url, "node_id": self.node_id, "paired": self.paired, "ready": self.ready(),
               "models": s.models, "loaded": s.current, "waiting": s.waiting, "running": s.running,
               "seen_s_ago": s.at.map(|t| t.elapsed().as_secs()), "error": s.error, "requests_routed": s.routed})
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Role {
    /// a server that may have workers (none: a single machine)
    Master,
    Worker,
}

/// This node: its id (the fingerprint of its ed25519 public key) and name.
#[derive(Clone, Debug)]
pub struct Node {
    pub id: String,
    pub name: String,
}

/// A found server on the LAN, not paired yet.
#[derive(Clone, Debug)]
struct Found {
    name: String,
    url: String,
    version: String,
    seen: Instant,
}

pub struct Cluster {
    pub role: Role,
    pub node: Node,
    data_dir: Option<PathBuf>,
    workers: RwLock<Vec<Arc<Worker>>>,
    // worker side
    code: Mutex<(String, Instant)>,
    fails: Mutex<(u32, Option<Instant>)>,
    masters: RwLock<Vec<Value>>,
    // master side
    found: Mutex<HashMap<String, Found>>,
    polling: AtomicBool,
}

fn random_bytes<const N: usize>() -> [u8; N] {
    use rand::RngCore;
    let mut b = [0u8; N];
    rand::rngs::OsRng.fill_bytes(&mut b);
    b
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn sha_hex(s: &str) -> String {
    use sha2::Digest;
    hex(&sha2::Sha256::digest(s.as_bytes()))
}

fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

fn new_code() -> String {
    let n = u32::from_le_bytes(random_bytes::<4>()) % 1_000_000;
    format!("{n:06}")
}

/// Write `v` to `path` readable by its owner only.
fn write_private(path: &Path, v: &Value) -> std::io::Result<()> {
    let text = serde_json::to_string_pretty(v).unwrap_or_default();
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(path)?;
        f.write_all(text.as_bytes())?;
        std::fs::set_permissions(path, std::os::unix::fs::PermissionsExt::from_mode(0o600))
    }
    #[cfg(not(unix))]
    {
        std::fs::write(path, text)
    }
}

/// The node's ed25519 key in `<data>/node-key` (made on first start, 0600) -> its node id: the first 16 bytes of the
/// SHA-256 of the public key, in hex.
pub fn node_identity(data_dir: &Path) -> String {
    let path = data_dir.join("node-key");
    let seed: [u8; 32] = match std::fs::read(&path) {
        Ok(b) if b.len() == 32 => b.try_into().unwrap(),
        _ => {
            let seed = random_bytes::<32>();
            #[cfg(unix)]
            {
                use std::io::Write;
                use std::os::unix::fs::OpenOptionsExt;
                if let Ok(mut f) = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&path) {
                    let _ = f.write_all(&seed);
                }
            }
            #[cfg(not(unix))]
            let _ = std::fs::write(&path, seed);
            seed
        }
    };
    let public = ed25519_dalek::SigningKey::from_bytes(&seed).verifying_key();
    use sha2::Digest;
    hex(&sha2::Sha256::digest(public.as_bytes())[..16])
}

fn host_name() -> String {
    std::fs::read_to_string("/etc/hostname").ok().map(|h| h.trim().to_string()).filter(|h| !h.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok().or_else(|| std::env::var("COMPUTERNAME").ok()))
        .unwrap_or_else(|| "am2strata".into())
}

impl Cluster {
    /// A cluster with no workers (tests).
    pub fn empty() -> Cluster {
        Cluster { role: Role::Master, node: Node { id: String::new(), name: String::new() }, data_dir: None,
                  workers: RwLock::new(Vec::new()), code: Mutex::new((new_code(), Instant::now())), fails: Mutex::new((0, None)),
                  masters: RwLock::new(Vec::new()), found: Mutex::new(HashMap::new()), polling: AtomicBool::new(false) }
    }

    /// The run config's `"cluster"` and what was paired before (`<data>`).
    pub fn from_config(rest: &std::collections::BTreeMap<String, Value>, data_dir: Option<&Path>) -> Result<Cluster, String> {
        let mut c = Cluster::empty();
        let cfg = rest.get("cluster").cloned().unwrap_or(Value::Null);
        c.role = match cfg["role"].as_str() {
            None | Some("master") => Role::Master,
            Some("worker") => Role::Worker,
            Some(r) => return Err(format!("config cluster.role: {r:?} is not \"master\" or \"worker\"")),
        };
        c.data_dir = data_dir.map(Path::to_path_buf);
        if let Some(d) = data_dir {
            let _ = std::fs::create_dir_all(d);
            c.node = Node { id: node_identity(d), name: cfg["name"].as_str().map(str::to_string).unwrap_or_else(host_name) };
        }
        let mut list = Vec::new();
        if let Some(w) = cfg.get("workers") {
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
                list.push(Worker::new(name.clone(), url, key, None, false));
            }
        }
        // the workers paired from the app
        if let Some(d) = data_dir {
            let saved: Value = std::fs::read_to_string(d.join("cluster-workers.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
            for w in saved["workers"].as_array().into_iter().flatten() {
                let (Some(name), Some(url), Some(key)) = (w["name"].as_str(), w["url"].as_str(), w["key"].as_str()) else { continue };
                if !list.iter().any(|x: &Arc<Worker>| x.name == name) {
                    list.push(Worker::new(name.into(), url.into(), key.into(), w["node_id"].as_str().map(str::to_string), true));
                }
            }
            let masters: Value = std::fs::read_to_string(d.join("cluster-masters.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
            *c.masters.write().unwrap() = masters["masters"].as_array().cloned().unwrap_or_default();
        }
        *c.workers.write().unwrap() = list;
        Ok(c)
    }

    pub fn is_empty(&self) -> bool {
        self.workers.read().unwrap().is_empty()
    }

    pub fn workers(&self) -> Vec<Arc<Worker>> {
        self.workers.read().unwrap().clone()
    }

    /// Start: poll the workers; announce (a worker) or browse (a master) by mDNS; a worker logs its pairing code.
    pub fn start(self: &Arc<Self>, port: u16) {
        self.polling.store(true, Ordering::Relaxed);
        for w in self.workers() {
            spawn_poller(w);
        }
        if self.role == Role::Worker {
            tracing::info!("cluster: worker {} ({}); pairing code {} (changes every 10 minutes)", self.node.name, self.node.id, self.code());
        }
        if std::env::var("AM2STRATA_NO_MDNS").is_ok_and(|v| v == "1") {
            return;
        }
        let me = self.clone();
        std::thread::spawn(move || {
            if let Err(e) = me.mdns(port) {
                tracing::warn!("cluster: mDNS is off ({e})");
            }
        });
    }

    fn mdns(self: &Arc<Self>, port: u16) -> Result<(), String> {
        let daemon = mdns_sd::ServiceDaemon::new().map_err(|e| e.to_string())?;
        if self.role == Role::Worker {
            let props = [("name", self.node.name.as_str()), ("version", env!("CARGO_PKG_VERSION")), ("node_id", self.node.id.as_str())];
            let instance = format!("{}-{}", self.node.name, &self.node.id[..8.min(self.node.id.len())]);
            let host = format!("{}.local.", self.node.name);
            let info = mdns_sd::ServiceInfo::new(SERVICE, &instance, &host, "", port, &props[..]).map_err(|e| e.to_string())?
                .enable_addr_auto();
            daemon.register(info).map_err(|e| e.to_string())?;
            tracing::info!("cluster: announced on the LAN as {instance}");
            // the daemon runs in its own thread; keep it alive with this one
            loop {
                std::thread::sleep(Duration::from_secs(3600));
            }
        }
        let rx = daemon.browse(SERVICE).map_err(|e| e.to_string())?;
        while let Ok(ev) = rx.recv() {
            if let mdns_sd::ServiceEvent::ServiceResolved(info) = ev {
                if let Some(f) = found_of(&info) {
                    self.found.lock().unwrap().insert(f.0, f.1);
                }
            }
        }
        Ok(())
    }

    /// The pairing code now (a worker), changed every 10 minutes.
    pub fn code(&self) -> String {
        let mut c = self.code.lock().unwrap();
        if c.1.elapsed() >= CODE_LIFE {
            *c = (new_code(), Instant::now());
            tracing::info!("cluster: new pairing code {}", c.0);
        }
        c.0.clone()
    }

    /// A master's key, given to this worker: true when it is one this worker minted.
    pub fn is_master_key(&self, key: &str) -> bool {
        if self.role != Role::Worker {
            return false;
        }
        let h = sha_hex(key);
        self.masters.read().unwrap().iter().any(|m| m["key_sha256"].as_str().is_some_and(|k| same(k, &h)))
    }

    fn save_masters(&self) {
        if let Some(d) = &self.data_dir {
            let _ = write_private(&d.join("cluster-masters.json"), &json!({"masters": *self.masters.read().unwrap()}));
        }
    }

    fn save_workers(&self) {
        if let Some(d) = &self.data_dir {
            let list: Vec<Value> = self.workers().iter().filter(|w| w.paired)
                .map(|w| json!({"name": w.name, "url": w.url, "node_id": w.node_id, "key": w.key})).collect();
            let _ = write_private(&d.join("cluster-workers.json"), &json!({"workers": list}));
        }
    }

    /// A worker's answer to a master's join: Ok(the key minted for it) or Err((status, why)).
    pub fn join(&self, code: &str, master_id: &str, master_name: &str) -> Result<String, (StatusCode, String)> {
        if self.role != Role::Worker {
            return Err((StatusCode::NOT_FOUND, "this server is not a worker (its run config's \"cluster\": {\"role\": \"worker\"})".into()));
        }
        let mut fails = self.fails.lock().unwrap();
        if let Some(t) = fails.1 {
            if t.elapsed() < LOCK_FOR {
                return Err((StatusCode::TOO_MANY_REQUESTS, format!("pairing is locked for {} more seconds after {LOCK_AFTER} wrong codes",
                                                                  (LOCK_FOR - t.elapsed()).as_secs())));
            }
            *fails = (0, None);
        }
        if !same(code.trim(), &self.code()) {
            fails.0 += 1;
            if fails.0 >= LOCK_AFTER {
                fails.1 = Some(Instant::now());
                tracing::warn!("cluster: {LOCK_AFTER} wrong pairing codes; pairing locked for 10 minutes");
            }
            return Err((StatusCode::FORBIDDEN, "wrong pairing code".into()));
        }
        *fails = (0, None);
        drop(fails);
        let key = format!("am2w-{}", hex(&random_bytes::<32>()));
        let id = master_id.chars().filter(|c| c.is_ascii_alphanumeric()).take(64).collect::<String>();
        let name = master_name.chars().filter(|c| !c.is_control()).take(80).collect::<String>();
        {
            let mut m = self.masters.write().unwrap();
            m.retain(|x| x["id"] != id.as_str());
            m.push(json!({"id": id, "name": name, "key_sha256": sha_hex(&key), "paired_at": crate::now()}));
        }
        self.save_masters();
        // a used code is not used again
        *self.code.lock().unwrap() = (new_code(), Instant::now());
        tracing::info!("cluster: paired with master {name} ({id})");
        Ok(key)
    }

    /// Forget the master whose key this is (it removed this worker).
    pub fn forget(&self, key: &str) -> bool {
        let h = sha_hex(key);
        let before = self.masters.read().unwrap().len();
        self.masters.write().unwrap().retain(|m| m["key_sha256"].as_str().is_none_or(|k| !same(k, &h)));
        let gone = self.masters.read().unwrap().len() < before;
        if gone {
            self.save_masters();
        }
        gone
    }

    /// Add a paired worker (the master), saved and polled at once.
    fn add_worker(&self, w: Arc<Worker>) {
        self.workers.write().unwrap().retain(|x| x.name != w.name);
        self.workers.write().unwrap().push(w.clone());
        self.save_workers();
        if self.polling.load(Ordering::Relaxed) {
            spawn_poller(w);
        }
    }

    fn remove_worker(&self, name: &str) -> Option<Arc<Worker>> {
        let w = self.workers().into_iter().find(|w| w.name == name)?;
        w.stop.store(true, Ordering::Relaxed);
        self.workers.write().unwrap().retain(|x| x.name != name);
        self.save_workers();
        Some(w)
    }

    /// The worker for a request for `model`: `local` says whether this PC has the model, `local_busy` whether its
    /// queue has work waiting. None: answer here.
    pub fn pick(&self, model: &str, local: bool, local_busy: bool) -> Option<Arc<Worker>> {
        let all = self.workers();
        let mut c: Vec<&Arc<Worker>> = all.iter().filter(|w| w.ready() && w.has(model)).collect();
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

    /// GET /api/cluster (the main key only).
    pub fn view(&self) -> Value {
        let workers = self.workers();
        let paired: Vec<String> = workers.iter().filter_map(|w| w.node_id.clone()).collect();
        let found: Vec<Value> = self.found.lock().unwrap().iter()
            .filter(|(id, f)| *id != &self.node.id && !paired.contains(id) && f.seen.elapsed() < Duration::from_secs(600))
            .map(|(id, f)| json!({"node_id": id, "name": f.name, "url": f.url, "version": f.version})).collect();
        let mut v = json!({
            "role": match self.role { Role::Worker => "worker", Role::Master if workers.is_empty() => "single", Role::Master => "master" },
            "node": {"id": self.node.id, "name": self.node.name},
            "workers": workers.iter().map(|w| w.view()).collect::<Vec<_>>(),
            "found": found,
        });
        if self.role == Role::Worker {
            v["pairing_code"] = json!(self.code());
            v["masters"] = json!(self.masters.read().unwrap().iter()
                .map(|m| json!({"id": m["id"], "name": m["name"], "paired_at": m["paired_at"]})).collect::<Vec<_>>());
        }
        v
    }
}

/// A resolved mDNS service as (node id, Found).
fn found_of(info: &mdns_sd::ServiceInfo) -> Option<(String, Found)> {
    let prop = |k: &str| info.get_property_val_str(k).map(str::to_string);
    let id = prop("node_id")?;
    // a private LAN address first (192.168/16, 10/8, 172.16/12), then any IPv4 (a VPN's), then IPv6
    let rank = |a: &std::net::IpAddr| match a {
        std::net::IpAddr::V4(v) if v.is_private() => 0,
        std::net::IpAddr::V4(v) if !v.is_loopback() && !v.is_link_local() => 1,
        std::net::IpAddr::V4(_) => 3,
        std::net::IpAddr::V6(_) => 2,
    };
    let addr = info.get_addresses().iter().min_by_key(|a| rank(a))?.to_string();
    let host = if addr.contains(':') { format!("[{addr}]") } else { addr };
    Some((id, Found { name: prop("name").unwrap_or_default(), url: format!("http://{host}:{}", info.get_port()),
                      version: prop("version").unwrap_or_default(), seen: Instant::now() }))
}

/// Ask a worker for its models and queue every 5 s, until it is removed.
fn spawn_poller(w: Arc<Worker>) {
    tokio::spawn(async move {
        let http = reqwest::Client::builder().timeout(Duration::from_secs(4)).build().expect("http client");
        while !w.stop.load(Ordering::Relaxed) {
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

/// GET /api/cluster: this node, its workers (what they hold and whether they answer), the servers found on the LAN
/// that are not paired, and, on a worker, its pairing code and masters. The main key only.
pub async fn status(axum::extract::State(s): axum::extract::State<crate::AppState>, headers: HeaderMap) -> Response {
    if !crate::authorized(&s, &headers) {
        return crate::error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    axum::Json(s.cluster.view()).into_response()
}

/// POST /api/cluster/pair {"url" or "node_id", "code"} (the master; its own page or main key): sends the code to the
/// worker, and on its yes keeps the worker and its key and starts using it.
pub async fn pair(axum::extract::State(s): axum::extract::State<crate::AppState>, headers: HeaderMap,
                  axum::Json(req): axum::Json<Value>) -> Response {
    if !crate::authorized(&s, &headers) {
        return crate::error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("workers can be paired");
    }
    if s.cluster.role == Role::Worker {
        return crate::error(StatusCode::BAD_REQUEST, "invalid_request_error", "this server is a worker; pair it from its master");
    }
    let Some(code) = req["code"].as_str().map(|c| c.trim().to_string()).filter(|c| c.len() == 6 && c.chars().all(|x| x.is_ascii_digit())) else {
        return crate::error(StatusCode::BAD_REQUEST, "invalid_request_error", "send the worker's 6-digit \"code\"");
    };
    let url = match (req["url"].as_str(), req["node_id"].as_str()) {
        (Some(u), _) => u.trim().trim_end_matches('/').to_string(),
        (None, Some(id)) => match s.cluster.found.lock().unwrap().get(id) {
            Some(f) => f.url.clone(),
            None => return crate::error(StatusCode::NOT_FOUND, "not_found", format!("no server with node id {id} was found on the network")),
        },
        _ => return crate::error(StatusCode::BAD_REQUEST, "invalid_request_error", "send the worker's \"url\" (http://host:port) or its \"node_id\""),
    };
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return crate::error(StatusCode::BAD_REQUEST, "invalid_request_error", "url: expected http://host:port");
    }
    let http = match reqwest::Client::builder().timeout(Duration::from_secs(10)).build() {
        Ok(h) => h,
        Err(e) => return crate::error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string()),
    };
    let r = http.post(format!("{url}/api/cluster/join"))
        .json(&json!({"code": code, "master_id": s.cluster.node.id, "master_name": s.cluster.node.name})).send().await;
    let r = match r {
        Ok(r) => r,
        Err(e) => return crate::error(StatusCode::BAD_GATEWAY, "worker_unreachable", format!("{url} did not answer: {e}")),
    };
    let status = r.status().as_u16();
    let v: Value = r.json().await.unwrap_or(Value::Null);
    if status != 200 {
        let code = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
        return crate::error(code, "pairing_refused", format!("the worker said: {}", v["error"]["message"].as_str().unwrap_or("no")));
    }
    let (Some(key), Some(node_id)) = (v["key"].as_str(), v["node_id"].as_str()) else {
        return crate::error(StatusCode::BAD_GATEWAY, "pairing_refused", "the worker's answer had no key");
    };
    let mut name: String = v["name"].as_str().unwrap_or("worker").chars().filter(|c| c.is_ascii_alphanumeric() || "-_.".contains(*c)).take(40).collect();
    if name.is_empty() {
        name = "worker".into();
    }
    // a name already taken by another worker gets the node id's start
    if s.cluster.workers().iter().any(|w| w.name == name && w.node_id.as_deref() != Some(node_id)) {
        name = format!("{name}-{}", &node_id[..6.min(node_id.len())]);
    }
    s.cluster.add_worker(Worker::new(name.clone(), url.clone(), key.to_string(), Some(node_id.to_string()), true));
    tracing::info!("cluster: paired with worker {name} ({url})");
    axum::Json(json!({"paired": name, "url": url, "node_id": node_id})).into_response()
}

/// POST /api/cluster/join {"code", "master_id", "master_name"} (a worker; the code is the only proof): for the right
/// code, a key for that master alone.
pub async fn join(axum::extract::State(s): axum::extract::State<crate::AppState>, axum::Json(req): axum::Json<Value>) -> Response {
    let (code, id, name) = (req["code"].as_str().unwrap_or(""), req["master_id"].as_str().unwrap_or(""), req["master_name"].as_str().unwrap_or("master"));
    if id.is_empty() {
        return crate::error(StatusCode::BAD_REQUEST, "invalid_request_error", "send \"master_id\"");
    }
    match s.cluster.join(code, id, name) {
        Ok(key) => axum::Json(json!({"key": key, "node_id": s.cluster.node.id, "name": s.cluster.node.name})).into_response(),
        Err((status, why)) => crate::error(status, "pairing_refused", why),
    }
}

/// POST /api/cluster/forget (a worker, with the key its master was given): the master removed this worker.
pub async fn forget(axum::extract::State(s): axum::extract::State<crate::AppState>, headers: HeaderMap) -> Response {
    let Some(k) = crate::clients::given_key(&headers) else {
        return crate::error(StatusCode::UNAUTHORIZED, "authentication_error", "a master's key is required");
    };
    if s.cluster.forget(k) {
        axum::Json(json!({"forgotten": true})).into_response()
    } else {
        crate::error(StatusCode::UNAUTHORIZED, "authentication_error", "not a key this worker gave a master")
    }
}

/// DELETE /api/cluster/workers/{name} (the master): stop using a worker, and tell it to forget this master.
pub async fn remove(axum::extract::State(s): axum::extract::State<crate::AppState>, headers: HeaderMap,
                    axum::extract::Path(name): axum::extract::Path<String>) -> Response {
    if !crate::authorized(&s, &headers) {
        return crate::error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("workers can be removed");
    }
    let Some(w) = s.cluster.remove_worker(&name) else {
        return crate::error(StatusCode::NOT_FOUND, "not_found", format!("no worker is named {name:?}"));
    };
    let told = if w.paired {
        let http = reqwest::Client::builder().timeout(Duration::from_secs(5)).build();
        match http {
            Ok(h) => h.post(format!("{}/api/cluster/forget", w.url)).bearer_auth(&w.key).send().await.is_ok_and(|r| r.status().is_success()),
            Err(_) => false,
        }
    } else {
        false
    };
    let note = if w.paired { None } else { Some("this worker is in the run config: remove it there too, or it comes back at the next start") };
    axum::Json(json!({"removed": name, "worker_told": told, "note": note})).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(name: &str, models: &[&str], current: Option<&str>, waiting: usize, seen: bool) -> Arc<Worker> {
        let w = Worker::new(name.into(), "http://x".into(), "k".into(), None, false);
        *w.seen.lock().unwrap() = Seen { models: models.iter().map(|m| m.to_string()).collect(), current: current.map(str::to_string),
                                         waiting, running: false, at: seen.then(Instant::now), error: None, routed: 0 };
        w
    }

    fn with(ws: Vec<Arc<Worker>>) -> Cluster {
        let c = Cluster::empty();
        *c.workers.write().unwrap() = ws;
        c
    }

    #[test]
    fn a_request_goes_where_its_model_is_and_answers_soonest() {
        let c = with(vec![worker("a", &["big"], None, 0, true), worker("b", &["big", "tiel"], Some("big"), 1, true),
                          worker("gone", &["big"], Some("big"), 0, false)]);
        assert_eq!(c.pick("big", false, false).unwrap().name, "b", "the model loaded beats an empty queue; a silent one is left out");
        assert!(c.pick("nowhere", false, false).is_none());
        // a model this PC has: here, unless this PC is busy and a worker has it loaded and idle
        assert!(c.pick("tiel", true, true).is_none(), "loaded nowhere idle");
        let c = with(vec![worker("idle", &["tiel"], Some("tiel"), 0, true)]);
        assert!(c.pick("tiel", true, false).is_none(), "this PC is free: here");
        assert_eq!(c.pick("tiel", true, true).unwrap().name, "idle");
    }

    #[test]
    fn workers_are_read_from_the_config() {
        let mut rest = std::collections::BTreeMap::new();
        rest.insert("cluster".to_string(), json!({"workers": {"pc2": {"url": "http://10.0.0.2:8000/", "key": "k2-0123456789"}}}));
        let c = Cluster::from_config(&rest, None).unwrap();
        let w = c.workers();
        assert_eq!((w[0].name.as_str(), w[0].url.as_str()), ("pc2", "http://10.0.0.2:8000"));
        rest.insert("cluster".to_string(), json!({"workers": {"pc2": {"url": "10.0.0.2"}}}));
        assert!(Cluster::from_config(&rest, None).is_err());
        rest.insert("cluster".to_string(), json!({"role": "boss"}));
        assert!(Cluster::from_config(&rest, None).is_err());
    }

    #[test]
    fn the_node_identity_is_kept_and_private() {
        let d = std::env::temp_dir().join(format!("cluster-id-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let a = node_identity(&d);
        assert_eq!(a.len(), 32);
        assert_eq!(node_identity(&d), a, "the same key next start");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(d.join("node-key")).unwrap().permissions().mode() & 0o777, 0o600);
        }
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_worker_pairs_only_with_its_code_and_locks_after_five_wrong_ones() {
        let mut c = Cluster::empty();
        c.role = Role::Worker;
        let code = c.code();
        let wrong = if code == "000000" { "000001" } else { "000000" };
        assert_eq!(c.join(wrong, "m1", "master").unwrap_err().0, StatusCode::FORBIDDEN);
        let key = c.join(&code, "m1", "master").unwrap();
        assert!(c.is_master_key(&key) && !c.is_master_key("am2w-other"));
        assert_ne!(c.code(), code, "a used code is not used again");
        for _ in 0..LOCK_AFTER {
            let _ = c.join(wrong, "m2", "x");
        }
        assert_eq!(c.join(&c.code(), "m2", "x").unwrap_err().0, StatusCode::TOO_MANY_REQUESTS, "locked, even for the right code");
        assert!(c.forget(&key) && !c.is_master_key(&key));
    }

    #[test]
    fn a_found_service_becomes_a_worker_candidate() {
        let props = [("name", "pc2"), ("version", "0.1.4"), ("node_id", "abc123")];
        let info = mdns_sd::ServiceInfo::new(SERVICE, "pc2-abc123", "pc2.local.", "100.72.13.89,192.168.8.50", 8000, &props[..]).unwrap();
        let (id, f) = found_of(&info).unwrap();
        assert_eq!((id.as_str(), f.name.as_str(), f.url.as_str(), f.version.as_str()), ("abc123", "pc2", "http://192.168.8.50:8000", "0.1.4"), "the LAN address before the VPN's");
    }
}
