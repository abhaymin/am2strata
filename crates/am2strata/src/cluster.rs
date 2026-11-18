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
//! - **Encrypted** (cluster_tls.rs): pairing swaps the nodes' certificate pins, and a paired worker is then reached
//!   only over its TLS 1.3 listener (`"tls_port"`, default the API port + 443), each side checking the other's pinned
//!   certificate; a master's key works only with that master's own certificate, and no longer over plain HTTP unless
//!   the worker's `"cluster": {"allow_plain": true}`. Workers given in the run config (url and key) stay plain HTTP,
//!   shown as such: use a VPN for those.
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

/// The media requests a worker can make too, and the kind each needs (its /api/media/capabilities). The first five
/// answer with the file; compose and thumbnails answer with a job, followed here (see `remote_job`).
pub const MEDIA_ROUTED: [(&str, &str); 7] = [
    ("/v1/images/generations", "image"), ("/v1/videos/generations", "video"), ("/v1/audio/music", "music"),
    ("/v1/audio/speech", "tts"), ("/v1/audio/transcriptions", "stt"),
    ("/v1/media/videos/compose", "compose"), ("/v1/media/thumbnails", "compose"),
];

/// The kinds a worker's capabilities say it can make now.
pub fn media_kinds(caps: &Value) -> Vec<String> {
    let mut out: Vec<String> = caps["kinds"].as_object().map(|k| k.iter().filter(|(_, v)| v["usable"] == true)
        .map(|(n, _)| n.clone()).collect()).unwrap_or_default();
    if caps["compose"] == true {
        out.push("compose".into());
    }
    out
}

/// A media request another machine can make as sent: every file in it is in the request itself (a data: URL) or on
/// the web, none a job id, library id or path of this PC.
pub fn portable(v: &Value) -> bool {
    match v {
        Value::String(t) => {
            let t = t.trim();
            let id_like = |p: &str| t.strip_prefix(p).is_some_and(|r| !r.is_empty() && r.len() <= 80 && !r.contains(' '));
            !(id_like("mj-") || id_like("mf_") || t.starts_with('/') || t.starts_with("file:") || t.starts_with("~/")
              || t.starts_with("\\\\") || t.get(1..3) == Some(":\\"))
        }
        Value::Array(a) => a.iter().all(portable),
        Value::Object(o) => o.iter().all(|(k, x)| k == "prompt" || k == "input" || k == "lyrics" || k == "captions" || portable(x)),
        _ => true,
    }
}

#[derive(Default, Clone)]
struct Seen {
    models: Vec<String>,
    current: Option<String>,
    waiting: usize,
    running: bool,
    at: Option<Instant>,
    error: Option<String>,
    routed: u64,
    /// the media kinds it can make now (its /api/media/capabilities: image, video, tts, stt, music, compose), read every 30 s
    media: Vec<String>,
    /// media requests this master sent it that have not finished yet (a worker's compose does not show in its queue)
    inflight: usize,
}

pub struct Worker {
    pub name: String,
    pub url: String,
    key: String,
    /// paired with TLS: the worker's listener (https://host:tls_port) and its certificate's pin
    pub tls_url: Option<String>,
    pub pin: Option<String>,
    /// the HTTP client for it: pinned mutual TLS, or plain HTTP for a run-config worker
    pub(crate) client: reqwest::Client,
    /// the worker's node id, when it was paired (not for run-config workers)
    pub node_id: Option<String>,
    /// paired from the app (kept in cluster-workers.json), not from the run config
    pub paired: bool,
    seen: Mutex<Seen>,
    stop: AtomicBool,
}

impl Worker {
    fn new(name: String, url: String, key: String, node_id: Option<String>, paired: bool) -> Arc<Worker> {
        Self::with_tls(name, url, key, node_id, paired, None, None)
    }

    /// `tls`: (the worker's TLS url, its pin), and this node's identity to show it.
    fn with_tls(name: String, url: String, key: String, node_id: Option<String>, paired: bool, tls: Option<(String, String)>,
                me: Option<&crate::cluster_tls::Identity>) -> Arc<Worker> {
        let (client, error) = match (&tls, me) {
            (Some((_, pin)), Some(me)) => match crate::cluster_tls::client(me, pin) {
                Ok(c) => (c, None),
                // never a silent fall back to plain HTTP: requests to the https url then fail, and say why
                Err(e) => (reqwest::Client::new(), Some(format!("its TLS client could not be made: {e}"))),
            },
            _ => (reqwest::Client::builder().connect_timeout(Duration::from_secs(5)).build().unwrap_or_default(), None),
        };
        let (tls_url, pin) = tls.map(|(u, p)| (Some(u), Some(p))).unwrap_or((None, None));
        let seen = Seen { error, ..Seen::default() };
        Arc::new(Worker { name, url, key, tls_url, pin, client, node_id, paired, seen: Mutex::new(seen), stop: AtomicBool::new(false) })
    }

    /// Where its requests go: the TLS listener when paired with one, else its url. Exposed so an ask_expert worker-form
    /// expert can POST to {base}/v1/chat/completions over this worker's own (pinned mutual-TLS) client.
    pub fn base(&self) -> &str {
        self.tls_url.as_deref().unwrap_or(&self.url)
    }

    /// The key the pairing gave for this master: the request to a worker-form expert is authorized with it, never a config key.
    pub fn auth_key(&self) -> &str { &self.key }

    /// Last time the poller heard from it (within STALE): an ask_expert worker-form expert is offered only while ready.
    pub fn is_ready(&self) -> bool { self.ready() }

    /// Whether the worker's model list carries `model`: the expert is offered only while it does.
    pub fn has_model(&self, model: &str) -> bool { self.has(model) }

    fn ready(&self) -> bool {
        self.seen.lock().unwrap().at.is_some_and(|t| t.elapsed() < STALE)
    }

    fn has(&self, model: &str) -> bool {
        self.seen.lock().unwrap().models.iter().any(|m| m == model)
    }

    /// It can make this media kind (image, video, tts, stt, music, compose).
    fn can(&self, kind: &str) -> bool {
        self.seen.lock().unwrap().media.iter().any(|k| k == kind)
    }

    /// Nothing runs or waits there, and nothing this master sent is still being made.
    fn idle(&self) -> bool {
        let s = self.seen.lock().unwrap();
        !s.running && s.waiting == 0 && s.inflight == 0
    }

    fn view(&self) -> Value {
        let s = self.seen.lock().unwrap().clone();
        json!({"name": self.name, "url": self.url, "node_id": self.node_id, "paired": self.paired, "ready": self.ready(),
               "encrypted": self.tls_url.is_some(), "tls_url": self.tls_url,
               "transport": if self.tls_url.is_some() { "TLS 1.3, pinned certificates" } else { "plain HTTP: use a VPN" },
               "models": s.models, "loaded": s.current, "waiting": s.waiting, "running": s.running,
               "seen_s_ago": s.at.map(|t| t.elapsed().as_secs()), "error": s.error, "requests_routed": s.routed,
               "media": s.media, "media_in_flight": s.inflight})
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
    /// this node's TLS certificate (cluster_tls.rs), made with the node's data
    pub identity: Option<crate::cluster_tls::Identity>,
    /// a worker's TLS port for its masters (None: the API port + 443)
    tls_port: Option<u16>,
    /// a worker still takes a TLS-paired master's key over plain HTTP
    allow_plain: bool,
    /// the API's port (set at start), for the default TLS port
    api_port: std::sync::atomic::AtomicU16,
    workers: RwLock<Vec<Arc<Worker>>>,
    // worker side
    code: Mutex<(String, Instant)>,
    fails: Mutex<(u32, Option<Instant>)>,
    masters: RwLock<Vec<Value>>,
    /// while this PC is busy, a chat may go to an idle worker's loaded model when no worker has the model asked for
    /// (run config `cluster.overflow`, on by default): the cluster answers in parallel instead of queueing
    pub overflow: bool,
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
pub(crate) fn write_private(path: &Path, v: &Value) -> std::io::Result<()> {
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
        Cluster { role: Role::Master, node: Node { id: String::new(), name: String::new() }, data_dir: None, identity: None,
                  tls_port: None, allow_plain: false, api_port: std::sync::atomic::AtomicU16::new(0),
                  workers: RwLock::new(Vec::new()), code: Mutex::new((new_code(), Instant::now())), fails: Mutex::new((0, None)),
                  masters: RwLock::new(Vec::new()), found: Mutex::new(HashMap::new()), polling: AtomicBool::new(false), overflow: true }
    }

    /// The run config's `"cluster"` and what was paired before (`<data>`).
    pub fn from_config(rest: &std::collections::BTreeMap<String, Value>, data_dir: Option<&Path>) -> Result<Cluster, String> {
        let mut c = Cluster::empty();
        let cfg = rest.get("cluster").cloned().unwrap_or(Value::Null);
        c.overflow = cfg["overflow"].as_bool().unwrap_or(true);
        c.role = match cfg["role"].as_str() {
            None | Some("master") => Role::Master,
            Some("worker") => Role::Worker,
            Some(r) => return Err(format!("config cluster.role: {r:?} is not \"master\" or \"worker\"")),
        };
        c.data_dir = data_dir.map(Path::to_path_buf);
        if let Some(d) = data_dir {
            let _ = std::fs::create_dir_all(d);
            c.node = Node { id: node_identity(d), name: cfg["name"].as_str().map(str::to_string).unwrap_or_else(host_name) };
            match crate::cluster_tls::Identity::load_or_make(d, &c.node.name) {
                Ok(i) => c.identity = Some(i),
                Err(e) => tracing::warn!("cluster: no TLS certificate ({e}); cluster traffic stays plain HTTP"),
            }
        }
        c.tls_port = cfg["tls_port"].as_u64().and_then(|p| u16::try_from(p).ok());
        c.allow_plain = cfg["allow_plain"] == true;
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
                    let tls = w["tls_url"].as_str().zip(w["pin"].as_str()).map(|(u, p)| (u.to_string(), p.to_string()));
                    list.push(Worker::with_tls(name.into(), url.into(), key.into(), w["node_id"].as_str().map(str::to_string), true, tls,
                                               c.identity.as_ref()));
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

    /// This machine's name (the node's, else the host's): what API calls and Monitor show as the one that answered.
    pub fn node_name(&self) -> String {
        if self.node.name.is_empty() { host_name() } else { self.node.name.clone() }
    }

    pub fn workers(&self) -> Vec<Arc<Worker>> {
        self.workers.read().unwrap().clone()
    }

    /// The paired worker of a given name, if any: an ask_expert worker-form expert looks one up at call time.
    pub fn worker_by_name(&self, name: &str) -> Option<Arc<Worker>> {
        self.workers.read().unwrap().iter().find(|w| w.name == name).cloned()
    }

    /// A worker added by tests (an ask_expert worker-form expert's paired worker): `models` is what it lists, `ready` whether
    /// the poller has just heard from it. Its client and pairing key are plain; only readiness and models matter here.
    #[cfg(test)]
    pub(crate) fn add_test_worker(&self, name: &str, models: &[&str], ready: bool) {
        let w = Worker::new(name.into(), "http://x".into(), "k".into(), None, false);
        *w.seen.lock().unwrap() = Seen { models: models.iter().map(|m| m.to_string()).collect(), current: None, waiting: 0,
                                         running: false, at: ready.then(Instant::now), error: None, routed: 0, ..Default::default() };
        self.workers.write().unwrap().push(w);
    }

    /// Start: poll the workers; announce (a worker) or browse (a master) by mDNS; a worker logs its pairing code.
    pub fn start(self: &Arc<Self>, port: u16) {
        self.api_port.store(port, Ordering::Relaxed);
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

    /// A master's key, given to this worker: true when it is one this worker minted, and it came the way that master
    /// paired: over the TLS listener with that master's own certificate (`peer`: the pin of the certificate the
    /// request's connection showed), or, for a master paired without TLS (or with `allow_plain`), over plain HTTP too.
    pub fn is_master_key(&self, key: &str, peer: Option<&str>) -> bool {
        if self.role != Role::Worker {
            return false;
        }
        let h = sha_hex(key);
        self.masters.read().unwrap().iter().any(|m| {
            m["key_sha256"].as_str().is_some_and(|k| same(k, &h)) && match (m["cert_sha256"].as_str(), peer) {
                (Some(pin), Some(p)) => same(pin, p),
                (Some(_), None) => self.allow_plain,
                (None, _) => true,
            }
        })
    }

    /// The pins of the masters this worker paired with TLS (its listener lets only these in).
    pub fn master_pins(&self) -> Vec<String> {
        self.masters.read().unwrap().iter().filter_map(|m| m["cert_sha256"].as_str().map(str::to_string)).collect()
    }

    /// A worker's TLS port for its masters (`api_port`: the API's).
    pub fn tls_port(&self, api_port: u16) -> Option<u16> {
        self.tls_port.or_else(|| api_port.checked_add(443))
    }

    /// A worker: its encrypted listener for paired masters, serving `app`.
    pub fn serve_tls(self: &Arc<Self>, app: axum::Router, host: String, api_port: u16) {
        let (Some(me), Some(port), Role::Worker) = (self.identity.clone(), self.tls_port(api_port), &self.role) else { return };
        let c = self.clone();
        tokio::spawn(async move {
            if let Err(e) = crate::cluster_tls::serve(me, Box::new(move || c.master_pins()), app, host, port).await {
                tracing::warn!("cluster: the encrypted listener did not start ({e}); masters paired with TLS cannot reach this worker");
            }
        });
    }

    fn save_masters(&self) {
        if let Some(d) = &self.data_dir {
            let _ = write_private(&d.join("cluster-masters.json"), &json!({"masters": *self.masters.read().unwrap()}));
        }
    }

    fn save_workers(&self) {
        if let Some(d) = &self.data_dir {
            let list: Vec<Value> = self.workers().iter().filter(|w| w.paired)
                .map(|w| json!({"name": w.name, "url": w.url, "node_id": w.node_id, "key": w.key, "tls_url": w.tls_url, "pin": w.pin})).collect();
            let _ = write_private(&d.join("cluster-workers.json"), &json!({"workers": list}));
        }
    }

    /// A worker's answer to a master's join: Ok(the key minted for it) or Err((status, why)).
    pub fn join(&self, code: &str, master_id: &str, master_name: &str, master_pin: Option<&str>) -> Result<String, (StatusCode, String)> {
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
            m.push(json!({"id": id, "name": name, "key_sha256": sha_hex(&key), "paired_at": crate::now(),
                          "cert_sha256": master_pin.filter(|p| crate::cluster_tls::valid_pin(p)).map(|p| p.to_ascii_lowercase())}));
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
            if !local_busy {
                return None;
            }
            // only to take load off this PC: an idle worker that has the model (loaded beats installed, below)
            c.retain(|w| { let s = w.seen.lock().unwrap(); !s.running && s.waiting == 0 });
        }
        // the one that will answer soonest: the model loaded first, then the shortest queue
        c.sort_by_key(|w| { let s = w.seen.lock().unwrap(); (s.current.as_deref() != Some(model), s.waiting + s.running as usize) });
        c.first().map(|w| (*w).clone())
    }

    /// A worker to make a media job of `kind`: a ready one that can make it, idle when `need_idle` (this PC is only
    /// busy), else the least loaded one (this PC cannot make it at all).
    pub fn pick_media(&self, kind: &str, need_idle: bool) -> Option<Arc<Worker>> {
        let mut c: Vec<Arc<Worker>> = self.workers().into_iter().filter(|w| w.ready() && w.can(kind)).collect();
        if need_idle {
            c.retain(|w| w.idle());
        }
        c.sort_by_key(|w| { let s = w.seen.lock().unwrap(); s.inflight + s.waiting + s.running as usize });
        c.into_iter().next()
    }

    /// While this PC is busy and no worker has the model asked for: an idle worker and the model it has loaded, so
    /// a chat is answered now, in parallel, instead of waiting here (off with `cluster.overflow: false`).
    pub fn overflow_pick(&self) -> Option<(Arc<Worker>, String)> {
        if !self.overflow { return None; }
        self.workers().into_iter().filter(|w| w.ready()).find_map(|w| {
            let m = { let s = w.seen.lock().unwrap(); if s.running || s.waiting > 0 { None } else { s.current.clone() } };
            m.map(|m| (w, m))
        })
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
        let mut round = 0u64;
        while !w.stop.load(Ordering::Relaxed) {
            let get = |p: &str| w.client.get(format!("{}{p}", w.base())).bearer_auth(&w.key).timeout(Duration::from_secs(4)).send();
            // the media kinds it can make, every 30 s (the worker keeps its own answer for a minute)
            if round % 6 == 0 {
                let caps = async { get("/api/media/capabilities").await?.error_for_status()?.json::<Value>().await }.await;
                if let Ok(c) = caps {
                    w.seen.lock().unwrap().media = media_kinds(&c);
                }
            }
            round += 1;
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
                    Err(e) => s.error = Some(error_chain(&e)),
                }
            }
            tokio::time::sleep(POLL).await;
        }
    });
}

/// The request sent on to `w`, its answer streamed back. Err: the worker could not be reached (the caller answers
/// here when it can).
/// An error with its causes (a TLS refusal says why: "not the one paired").
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

pub async fn forward(w: &Worker, path_and_query: &str, headers: &HeaderMap, body: axum::body::Bytes) -> Result<Response, String> {
    forward_with(w, path_and_query, headers, body, &[]).await
}

/// `forward` with headers of this master's own added (the routed request's tools, on the pinned master link).
pub async fn forward_with(w: &Worker, path_and_query: &str, headers: &HeaderMap, body: axum::body::Bytes,
                          extra: &[(&'static str, String)]) -> Result<Response, String> {
    let mut rq = w.client.post(format!("{}{path_and_query}", w.base())).bearer_auth(&w.key).body(body);
    for (h, v) in extra {
        rq = rq.header(*h, v.as_str());
    }
    // what shapes the answer; never the client's own key
    for h in ["content-type", "accept", "anthropic-version", "anthropic-beta", "openai-beta", "x-strata-priority"] {
        if let Some(v) = headers.get(h) {
            rq = rq.header(h, v.as_bytes());
        }
    }
    let r = rq.send().await.map_err(|e| error_chain(&e))?;
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

/// Closes a forwarded request's Monitor row when its answer's body is dropped: after its last byte, or when the client
/// went away first.
struct RemoteRow {
    mon: std::sync::Arc<crate::monitor::Monitor>,
    id: u64,
    finish: &'static str,
}

impl Drop for RemoteRow {
    fn drop(&mut self) {
        self.mon.remote_end(self.id, self.finish);
    }
}

/// The layer: a chat request whose model a worker should answer goes there.
pub async fn route(axum::extract::State(s): axum::extract::State<crate::AppState>, req: axum::extract::Request,
                   next: axum::middleware::Next) -> Response {
    let path = req.uri().path().to_string();
    let media = MEDIA_ROUTED.iter().find(|(p, _)| *p == path).map(|(_, k)| *k);
    if s.cluster.is_empty() || req.method() != axum::http::Method::POST || (!ROUTED.contains(&path.as_str()) && media.is_none())
        || !crate::api_authorized(&s, req.headers()) {
        return next.run(req).await;
    }
    if let Some(kind) = media {
        return route_media(s, req, next, kind).await;
    }
    let (parts, body) = req.into_parts();
    let Ok(mut bytes) = axum::body::to_bytes(body, 256 << 20).await else {
        return crate::error(StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error", "the request is too large");
    };
    let parsed = serde_json::from_slice::<Value>(&bytes).ok();
    let model = parsed.as_ref().and_then(|v| v["model"].as_str().map(str::to_string));
    if let Some(model) = model {
        let local = s.mgr.entry(&model).is_some();
        let req_v = parsed.unwrap_or(Value::Null);
        let how = Routing::of(&s, &parts.headers, &req_v);
        // busy: a job runs here (a chat, or a media job holding the GPU) or waits; a new chat would wait behind it
        let busy = s.mgr.queue.busy();
        let original = bytes.clone();   // a failed forward is answered here, as asked
        let mut model = model;
        // a request that needs this PC's memory or MCP servers stays here while this PC has its model
        let mut picked = if local && how.needs_master { None } else { s.cluster.pick(&model, local, busy) };
        // another model instead of the one asked for only for a plain chat: never for an agent's or a tool request
        // (a coding agent would silently get a much weaker model), unless it opts in with x-strata-overflow: any
        if picked.is_none() && busy && path != "/v1/completions" && (how.plain_chat() || (how.overflow_any && !how.media_in)) {
            if let Some((w, other)) = s.cluster.overflow_pick() {
                // the worker answers with the model it has loaded; the answer's "model" says which
                if let Ok(mut v) = serde_json::from_slice::<Value>(&bytes) {
                    tracing::info!("{path}: this PC is busy; {model} goes to worker {} as {other}", w.name);
                    v["model"] = json!(other);
                    bytes = axum::body::Bytes::from(serde_json::to_vec(&v).unwrap_or_default());
                    model = other;
                    picked = Some(w);
                }
            }
        }
        if let Some(w) = picked {
            let pq = parts.uri.path_and_query().map(|p| p.as_str().to_string()).unwrap_or(path.clone());
            let mon = s.mgr.monitor.clone();
            let rid = mon.remote_begin(&w.name, &model, &path);
            let sent = how.body_for_worker(&bytes);
            match forward_with(&w, &pq, &parts.headers, sent, &how.headers()).await {
                Ok(r) => {
                    tracing::info!("{path}: {model} answered by worker {}", w.name);
                    // the answer's end (its last byte, or the client gone) closes Monitor's row
                    let ok = r.status().is_success();
                    let (rp, body) = r.into_parts();
                    let guard = RemoteRow { mon, id: rid, finish: if ok { "stop" } else { "error" } };
                    let stream = futures::StreamExt::map(body.into_data_stream(), move |c| { let _ = &guard; c });
                    return Response::from_parts(rp, Body::from_stream(stream));
                }
                Err(e) => {
                    mon.remote_end(rid, "error");
                    w.seen.lock().unwrap().error = Some(e.clone());
                    tracing::warn!("worker {} did not take the request ({e})", w.name);
                    bytes = original;
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

/// What a chat request carries that decides where it may run.
pub struct Routing {
    /// the caller's client profile (narrowed by x-strata-tools), when its key has one other than the main key's
    profile: Option<crate::clients::Profile>,
    /// the request uses this PC's memory or MCP servers, which a worker does not have
    pub needs_master: bool,
    /// it brings its own tools (a coding agent's), or names tool sets
    tools: bool,
    /// x-strata-overflow: any (another loaded model is fine)
    pub overflow_any: bool,
    /// it carries pictures, video or audio: only the model asked for (one that reads them) may answer
    media_in: bool,
}

/// A chat request with an image, video or audio part in any message (OpenAI's and Anthropic's part types).
pub fn has_media_parts(req: &Value) -> bool {
    let kinds = ["image_url", "image", "input_image", "video_url", "video", "input_video", "input_audio", "audio"];
    ["messages", "input"].iter().filter_map(|k| req[*k].as_array()).flatten().any(|m| {
        m["content"].as_array().is_some_and(|parts| parts.iter().any(|p| p["type"].as_str().is_some_and(|t| kinds.contains(&t))))
    })
}

impl Routing {
    pub fn of(s: &crate::AppState, headers: &HeaderMap, req: &Value) -> Routing {
        let mut profile = crate::api_access(s, headers).flatten();
        if let (Some(p), Some(h)) = (profile.as_ref(), headers.get("x-strata-tools").and_then(|v| v.to_str().ok())) {
            profile = p.narrowed(h).ok();
        }
        let client = profile.clone().filter(|p| p.name != "default");
        let mcp = profile.as_ref().is_some_and(|p| { let (n, all) = p.mcp(); all || !n.is_empty() }) || req["strata_mcp"] == true;
        let memory = client.as_ref().is_some_and(|p| p.has("memory")) || req["strata_memory"] == true;
        Routing {
            needs_master: mcp || memory,
            tools: req["tools"].as_array().is_some_and(|t| !t.is_empty()) || headers.contains_key("x-strata-tools") || client.is_some(),
            overflow_any: headers.get("x-strata-overflow").and_then(|v| v.to_str().ok()) == Some("any"),
            media_in: has_media_parts(req),
            profile: client,
        }
    }

    /// A chat with no tools of its own from the main key (the web app's), which any loaded model may answer.
    pub fn plain_chat(&self) -> bool {
        !self.tools && !self.needs_master && !self.media_in
    }

    /// The headers that carry the caller's tools to the worker: the worker runs the server tools it has for these
    /// sets (web, image, browser, ...) with no memory and no MCP servers of its own (cluster: the worker trusts only
    /// this master's key on the pinned link for them).
    pub fn headers(&self) -> Vec<(&'static str, String)> {
        let Some(p) = &self.profile else { return vec![] };
        let sets: Vec<&str> = crate::clients::SETS.iter().copied().filter(|t| *t != "memory" && p.has(t)).collect();
        vec![("x-strata-routed-tools", if sets.is_empty() { "none".into() } else { sets.join(",") }),
             ("x-strata-routed-for", p.name.chars().filter(|c| c.is_ascii_alphanumeric() || "-_.".contains(*c)).take(64).collect())]
    }

    /// The body sent on: this PC's memory and MCP servers turned off there (the worker would use its own).
    pub fn body_for_worker(&self, bytes: &axum::body::Bytes) -> axum::body::Bytes {
        match serde_json::from_slice::<Value>(bytes) {
            Ok(Value::Object(mut o)) => {
                o.remove("strata_mcp");
                o.insert("strata_memory".into(), Value::Bool(false));
                axum::body::Bytes::from(serde_json::to_vec(&Value::Object(o)).unwrap_or_default())
            }
            _ => bytes.clone(),
        }
    }
}

/// Decrements a worker's in-flight media count when dropped.
struct Inflight(Arc<Worker>);

impl Inflight {
    fn new(w: Arc<Worker>) -> Inflight {
        w.seen.lock().unwrap().inflight += 1;
        Inflight(w)
    }
}

impl Drop for Inflight {
    fn drop(&mut self) {
        let mut s = self.0.seen.lock().unwrap();
        s.inflight = s.inflight.saturating_sub(1);
    }
}

/// A media request: made by a worker that can make it when this PC cannot, or when this PC is busy and the worker is
/// idle; else here. Its files must travel in the request (`portable`). A compose or thumbnail answers with a job of
/// this server's own, followed on the worker until its file is here.
async fn route_media(s: crate::AppState, req: axum::extract::Request, next: axum::middleware::Next, kind: &'static str) -> Response {
    let (parts, body) = req.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, 512 << 20).await else {
        return crate::error(StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error", "the request is too large");
    };
    let is_json = parts.headers.get("content-type").and_then(|v| v.to_str().ok()).is_none_or(|c| c.contains("json"));
    let v: Value = if is_json { serde_json::from_slice(&bytes).unwrap_or(Value::Null) } else { Value::Null };
    let here = || next.run(axum::extract::Request::from_parts(parts.clone(), Body::from(bytes.clone())));
    // a multipart upload (a transcription's file) is the file itself: portable as it is
    if is_json && (v.is_null() || !portable(&v)) {
        return here().await;
    }
    let local_can = match kind {
        "compose" => crate::media::compose_missing().is_empty(),
        k => s.media_jobs.capabilities().await["kinds"][k]["usable"] == true,
    };
    let busy = s.mgr.queue.busy() || (kind == "compose" && s.media_jobs.active(&["compose", "thumbnail"]) > 0);
    if local_can && !busy {
        return here().await;
    }
    let Some(w) = s.cluster.pick_media(kind, local_can) else { return here().await };
    let pq = parts.uri.path_and_query().map(|p| p.as_str().to_string()).unwrap_or_default();
    let mon = s.mgr.monitor.clone();
    let rid = mon.remote_begin(&w.name, kind, &pq);
    let held = Inflight::new(w.clone());
    tracing::info!("{pq}: {} goes to worker {} ({})", kind, w.name, if local_can { "this PC is busy" } else { "this PC cannot make it" });
    let r = match forward(&w, &pq, &parts.headers, bytes.clone()).await {
        Ok(r) => r,
        Err(e) => {
            mon.remote_end(rid, "error");
            drop(held);
            w.seen.lock().unwrap().error = Some(e.clone());
            tracing::warn!("worker {} did not take the {kind} request ({e}); made here", w.name);
            return here().await;
        }
    };
    if kind != "compose" {
        // the file itself: streamed back, the row closed and the worker freed once it is all sent
        let ok = r.status().is_success();
        let (rp, body) = r.into_parts();
        let guard = (RemoteRow { mon, id: rid, finish: if ok { "stop" } else { "error" } }, held);
        let stream = futures::StreamExt::map(body.into_data_stream(), move |c| { let _ = &guard; c });
        return Response::from_parts(rp, Body::from_stream(stream));
    }
    // a job there: one here that follows it, so the caller polls this server as usual
    let status = r.status();
    let raw = axum::body::to_bytes(r.into_body(), 4 << 20).await.unwrap_or_default();
    let job: Value = serde_json::from_slice(&raw).unwrap_or(Value::Null);
    let Some(remote_id) = job["id"].as_str().filter(|_| status.is_success()).map(str::to_string) else {
        mon.remote_end(rid, "error");
        let mut resp = Response::new(Body::from(raw));
        *resp.status_mut() = status;
        resp.headers_mut().insert("content-type", axum::http::HeaderValue::from_static("application/json"));
        return resp;
    };
    let thumb = parts.uri.path().ends_with("/thumbnails");
    let (n, mut answer) = s.media_jobs.submit_remote(if thumb { "thumbnail" } else { "compose" }, &v, &w.name, &remote_id);
    answer["worker"] = json!(w.name);
    let s2 = s.clone();
    tokio::spawn(async move {
        let out = follow(&w, &remote_id).await;
        mon.remote_end(rid, if out.is_ok() { "stop" } else { "error" });
        drop(held);
        s2.media_jobs.finish_remote(n, out);
    });
    axum::Json(answer).into_response()
}

/// A worker's media job followed until it is done, then its file read: (mime, bytes, meta).
async fn follow(w: &Worker, id: &str) -> Result<(String, Vec<u8>, Value), String> {
    let start = Instant::now();
    let mut misses = 0;
    loop {
        tokio::time::sleep(Duration::from_secs(5)).await;
        if start.elapsed() > Duration::from_secs(12 * 3600) {
            return Err(format!("worker {} did not finish {id} in 12 hours", w.name));
        }
        let r = w.client.get(format!("{}/v1/media/jobs/{id}", w.base())).bearer_auth(&w.key).timeout(Duration::from_secs(10)).send().await;
        let v: Value = match r {
            Ok(r) if r.status().is_success() => r.json().await.unwrap_or(Value::Null),
            Ok(r) => return Err(format!("worker {} answered HTTP {} for {id}", w.name, r.status().as_u16())),
            Err(e) => {
                misses += 1;
                if misses > 60 { return Err(format!("worker {} stopped answering about {id}: {}", w.name, error_chain(&e))); }
                continue;
            }
        };
        misses = 0;
        match v["state"].as_str() {
            Some("done") => break,
            Some("error") => return Err(format!("on worker {}: {}", w.name, v["error"].as_str().unwrap_or("failed"))),
            _ => {}
        }
    }
    let r = w.client.get(format!("{}/v1/media/jobs/{id}/content", w.base())).bearer_auth(&w.key).timeout(Duration::from_secs(600))
        .send().await.map_err(|e| error_chain(&e))?;
    let mime = r.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("application/octet-stream").to_string();
    let st = w.client.get(format!("{}/v1/media/jobs/{id}", w.base())).bearer_auth(&w.key).timeout(Duration::from_secs(10)).send().await
        .ok();
    let bytes = r.bytes().await.map_err(|e| error_chain(&e))?.to_vec();
    let mut meta = match st { Some(r) => r.json::<Value>().await.map(|v| v["meta"].clone()).unwrap_or(Value::Null), None => Value::Null };
    if !meta.is_object() { meta = json!({}); }
    meta["worker"] = json!(w.name);
    Ok((mime, bytes, meta))
}

/// GET /api/cluster: this node, its workers (what they hold and whether they answer), the servers found on the LAN
/// that are not paired, and, on a worker, its pairing code and masters. The main key only.
pub async fn status(axum::extract::State(s): axum::extract::State<crate::AppState>, headers: HeaderMap) -> Response {
    if !crate::authorized(&s, &headers) {
        return crate::error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    axum::Json(s.cluster.view()).into_response()
}

/// GET /api/cluster/system (the main key): each worker's own GET /api/system, read with the key it gave this master,
/// for Settings' machine list. A worker that does not answer within 5 s is listed with why.
pub async fn systems(axum::extract::State(s): axum::extract::State<crate::AppState>, headers: HeaderMap) -> Response {
    if !crate::authorized(&s, &headers) {
        return crate::error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let reads = s.cluster.workers().into_iter().map(|w| async move {
        let r = w.client.get(format!("{}/api/system", w.base())).bearer_auth(&w.key).timeout(Duration::from_secs(5)).send().await;
        let (system, error) = match r {
            Ok(r) if r.status().is_success() => match r.json::<Value>().await {
                Ok(v) => (Some(v), None),
                Err(e) => (None, Some(format!("its answer was not read ({e})"))),
            },
            Ok(r) => (None, Some(format!("it answered HTTP {} (a worker older than v0.1.9 does not share its hardware)", r.status().as_u16()))),
            Err(e) => (None, Some(error_chain(&e))),
        };
        serde_json::json!({"name": w.name, "url": w.base(), "ready": w.is_ready(), "system": system, "error": error})
    });
    let machines: Vec<Value> = futures::future::join_all(reads).await;
    axum::Json(serde_json::json!({"workers": machines})).into_response()
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
        .json(&json!({"code": code, "master_id": s.cluster.node.id, "master_name": s.cluster.node.name,
                      "master_cert_sha256": s.cluster.identity.as_ref().map(|i| i.pin.clone())})).send().await;
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
    // the worker's encrypted listener: the same host, its TLS port, its certificate pinned
    let tls = match (v["tls_port"].as_u64(), v["cert_sha256"].as_str().filter(|p| crate::cluster_tls::valid_pin(p)), s.cluster.identity.as_ref()) {
        (Some(port), Some(pin), Some(_)) => reqwest::Url::parse(&url).ok().and_then(|u| u.host_str().map(|h| {
            let h = if h.contains(':') && !h.starts_with('[') { format!("[{h}]") } else { h.to_string() };
            (format!("https://{h}:{port}"), pin.to_ascii_lowercase())
        })),
        _ => None,
    };
    let encrypted = tls.is_some();
    s.cluster.add_worker(Worker::with_tls(name.clone(), url.clone(), key.to_string(), Some(node_id.to_string()), true, tls,
                                          s.cluster.identity.as_ref()));
    tracing::info!("cluster: paired with worker {name} ({url}{})", if encrypted { ", encrypted" } else { ", plain HTTP" });
    axum::Json(json!({"paired": name, "url": url, "node_id": node_id, "encrypted": encrypted})).into_response()
}

/// POST /api/cluster/join {"code", "master_id", "master_name"} (a worker; the code is the only proof): for the right
/// code, a key for that master alone.
pub async fn join(axum::extract::State(s): axum::extract::State<crate::AppState>, axum::Json(req): axum::Json<Value>) -> Response {
    let (code, id, name) = (req["code"].as_str().unwrap_or(""), req["master_id"].as_str().unwrap_or(""), req["master_name"].as_str().unwrap_or("master"));
    if id.is_empty() {
        return crate::error(StatusCode::BAD_REQUEST, "invalid_request_error", "send \"master_id\"");
    }
    match s.cluster.join(code, id, name, req["master_cert_sha256"].as_str()) {
        Ok(key) => {
            // the encrypted listener's port and this worker's pin, for a master that sent its own pin
            let tls = req["master_cert_sha256"].as_str().is_some_and(crate::cluster_tls::valid_pin) && s.cluster.identity.is_some();
            axum::Json(json!({"key": key, "node_id": s.cluster.node.id, "name": s.cluster.node.name,
                              "tls_port": if tls { s.cluster.tls_port(s.cluster.api_port.load(Ordering::Relaxed)) } else { None },
                              "cert_sha256": if tls { s.cluster.identity.as_ref().map(|i| i.pin.clone()) } else { None }})).into_response()
        }
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
        w.client.post(format!("{}/api/cluster/forget", w.base())).bearer_auth(&w.key).timeout(Duration::from_secs(5)).send().await
            .is_ok_and(|r| r.status().is_success())
    } else {
        false
    };
    let note = if w.paired { None } else { Some("this worker is in the run config: remove it there too, or it comes back at the next start") };
    axum::Json(json!({"removed": name, "worker_told": told, "note": note})).into_response()
}

#[cfg(test)]
mod tests {

    #[test]
    fn a_media_request_travels_only_when_its_files_are_in_it() {
        // data: URLs and web addresses go with the request; this PC's job ids, library ids and paths do not
        assert!(portable(&json!({"audio": "data:audio/wav;base64,UklG", "shots": [{"image": "https://x.org/a.png"}], "size": "1080x1920"})));
        assert!(!portable(&json!({"audio": "mj-12", "shots": []})));
        assert!(!portable(&json!({"shots": [{"clip": "mf_ab12cd"}]})));
        assert!(!portable(&json!({"audio": "/home/me/song.wav"})));
        assert!(!portable(&json!({"image": "file:///tmp/a.png"})));
        assert!(!portable(&json!({"image": "C:\\pics\\a.png"})));
        // the words are words: a prompt or caption that starts with a slash is not a path
        assert!(portable(&json!({"prompt": "/imagine a lighthouse", "captions": [[0.0, 1.0, "/ chorus"]]})));
    }

    #[test]
    fn a_media_job_goes_to_an_idle_worker_that_can_make_it() {
        let c = Cluster::empty();
        c.add_test_worker("mac", &["m"], true);
        c.add_test_worker("pc2", &["m"], true);
        for w in c.workers() {
            w.seen.lock().unwrap().media = if w.name == "mac" { vec!["compose".into(), "stt".into()] } else { vec!["image".into()] };
        }
        assert_eq!(c.pick_media("compose", true).map(|w| w.name.clone()).as_deref(), Some("mac"));
        assert_eq!(c.pick_media("image", true).map(|w| w.name.clone()).as_deref(), Some("pc2"));
        assert!(c.pick_media("music", false).is_none(), "nobody can make music");
        // busy with a job this master sent: not idle, so not picked while this PC could make it itself
        let mac = c.workers().into_iter().find(|w| w.name == "mac").unwrap();
        let held = Inflight::new(mac.clone());
        assert!(c.pick_media("compose", true).is_none());
        assert_eq!(c.pick_media("compose", false).map(|w| w.name.clone()).as_deref(), Some("mac"), "this PC cannot: the busy one still");
        drop(held);
        assert!(mac.idle());
    }

    #[test]
    fn a_workers_capabilities_name_the_kinds_it_can_make() {
        let caps = json!({"available": true, "compose": true, "kinds": {"image": {"usable": true}, "video": {"usable": false}, "stt": {"usable": true}}});
        let mut k = media_kinds(&caps);
        k.sort();
        assert_eq!(k, ["compose", "image", "stt"]);
        // an older worker says nothing of compose: it is not sent any
        assert_eq!(media_kinds(&json!({"kinds": {}})), Vec::<String>::new());
    }

    #[test]
    fn the_routed_tools_carry_the_profile_without_memory_and_the_body_turns_memory_off() {
        let p = crate::clients::Profile::generated("agent-1".into(), ["web", "image", "memory", "mcp:files"].iter().map(|s| s.to_string()).collect(), "a".into());
        let r = Routing { profile: Some(p), needs_master: true, tools: true, overflow_any: false, media_in: false };
        let h = r.headers();
        assert_eq!(h[0], ("x-strata-routed-tools", "web,image".to_string()));
        assert_eq!(h[1].1, "agent-1");
        // an agent never overflows to whatever model a worker has loaded
        assert!(!r.plain_chat());
        let b = r.body_for_worker(&axum::body::Bytes::from(r#"{"model":"m","strata_mcp":true,"strata_memory":true}"#));
        let v: Value = serde_json::from_slice(&b).unwrap();
        assert!(v.get("strata_mcp").is_none());
        assert_eq!(v["strata_memory"], false);
        // the main key's plain chat (the web app's) may go to another loaded model, with no extra headers
        let plain = Routing { profile: None, needs_master: false, tools: false, overflow_any: false, media_in: false };
        assert!(plain.plain_chat() && plain.headers().is_empty());
        // a request with a picture or a video (am2Strata's judge sends frames to a vision model) never goes to another
        // loaded model, which may not read images
        let judge = json!({"model": "minicpm-v", "messages": [{"role": "user", "content": [{"type": "text", "text": "brief"},
                           {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}}]}]});
        assert!(has_media_parts(&judge));
        assert!(!has_media_parts(&json!({"messages": [{"role": "user", "content": "hi"}]})));
        let seen = Routing { media_in: true, ..plain };
        assert!(!seen.plain_chat());
    }
    use super::*;

    fn worker(name: &str, models: &[&str], current: Option<&str>, waiting: usize, seen: bool) -> Arc<Worker> {
        let w = Worker::new(name.into(), "http://x".into(), "k".into(), None, false);
        *w.seen.lock().unwrap() = Seen { models: models.iter().map(|m| m.to_string()).collect(), current: current.map(str::to_string),
                                         waiting, running: false, at: seen.then(Instant::now), error: None, routed: 0, ..Default::default() };
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
        // busy here: an idle worker that has the model installed but not loaded takes it too
        let c = with(vec![worker("cold", &["tiel"], Some("other"), 0, true)]);
        assert_eq!(c.pick("tiel", true, true).unwrap().name, "cold");
    }

    #[test]
    fn a_busy_pc_overflows_to_an_idle_workers_loaded_model() {
        let c = with(vec![worker("busy", &["mini"], Some("mini"), 2, true), worker("mac", &["mini"], Some("mini"), 0, true),
                          worker("empty", &["x"], None, 0, true)]);
        let (w, m) = c.overflow_pick().unwrap();
        assert_eq!((w.name.as_str(), m.as_str()), ("mac", "mini"), "idle, with a model loaded");
        let mut c = with(vec![worker("mac", &["mini"], Some("mini"), 0, true)]);
        c.overflow = false;
        assert!(c.overflow_pick().is_none(), "cluster.overflow: false keeps chats here");
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
        assert_eq!(c.join(wrong, "m1", "master", None).unwrap_err().0, StatusCode::FORBIDDEN);
        let key = c.join(&code, "m1", "master", None).unwrap();
        assert!(c.is_master_key(&key, None) && !c.is_master_key("am2w-other", None));
        assert_ne!(c.code(), code, "a used code is not used again");
        for _ in 0..LOCK_AFTER {
            let _ = c.join(wrong, "m2", "x", None);
        }
        assert_eq!(c.join(&c.code(), "m2", "x", None).unwrap_err().0, StatusCode::TOO_MANY_REQUESTS, "locked, even for the right code");
        assert!(c.forget(&key) && !c.is_master_key(&key, None));
    }

    #[test]
    fn a_tls_paired_masters_key_needs_its_own_certificate() {
        let mut c = Cluster::empty();
        c.role = Role::Worker;
        let (pin, other) = ("a".repeat(64), "b".repeat(64));
        let key = c.join(&c.code(), "m1", "master", Some(&pin)).unwrap();
        assert!(c.is_master_key(&key, Some(&pin)), "over TLS with its certificate");
        assert!(!c.is_master_key(&key, Some(&other)), "another master's certificate");
        assert!(!c.is_master_key(&key, None), "plain HTTP");
        c.allow_plain = true;
        assert!(c.is_master_key(&key, None), "plain HTTP when the worker allows it");
        assert_eq!(c.master_pins(), vec![pin.clone()]);
        assert_eq!(c.tls_port(8000), Some(8443));
    }

    #[test]
    fn a_found_service_becomes_a_worker_candidate() {
        let props = [("name", "pc2"), ("version", "0.1.4"), ("node_id", "abc123")];
        let info = mdns_sd::ServiceInfo::new(SERVICE, "pc2-abc123", "pc2.local.", "100.72.13.89,192.168.8.50", 8000, &props[..]).unwrap();
        let (id, f) = found_of(&info).unwrap();
        assert_eq!((id.as_str(), f.name.as_str(), f.url.as_str(), f.version.as_str()), ("abc123", "pc2", "http://192.168.8.50:8000", "0.1.4"), "the LAN address before the VPN's");
    }
}
