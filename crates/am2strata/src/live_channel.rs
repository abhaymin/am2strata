//! The live channel: one continuous WebSocket between a client (the web app, a script, a cluster node) and the server,
//! for pushed updates in place of polling and for requests without a new connection each.
//!
//! `GET /v1/live` — `ws://` on the HTTP port, `wss://` on the HTTPS port (the same app serves both). JSON text frames.
//!
//! - **Key**: the Authorization header, else the first message `{"type":"auth","key":"…"}` within 5 s (a browser cannot
//!   set headers on a WebSocket; never in the URL). The Origin is checked as for every route that acts for the user.
//!   The server's own key gets everything; a client profile's key gets its own requests and the public topics.
//! - **Topics** (`subscribe` / `unsubscribe`): `status`, `queue`, `jobs`, `workshop`, `metrics`. A snapshot on
//!   subscribe, then an `update` whenever that state changes. One sampler task serves every connection: it runs only
//!   while someone is subscribed, is woken by the queue's and the Workshop's changes (`poke`), and sends only what
//!   changed. No subscriber: no task, no timer.
//! - **Requests**: `{"type":"request","id","method","path","body"}` runs through the same router as HTTP, in-process,
//!   with the connection's key — every rule, limit, queue priority and monitor applies as it does over HTTP. A JSON
//!   answer comes back as one `response`; a streamed one (SSE) as `event`s and a `done`. `cancel` drops the request as
//!   a closed HTTP connection does (the model stops). Several may run at once.
//! - `ping` → `pong`; the server pings every 25 s and closes a peer that stays silent.
//!
//! The message reference is in docs/DEPLOYMENT.md ("The live channel: ws:// and wss://").

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::response::Response;
use futures::stream::SplitSink;
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::{broadcast, mpsc, Notify};

use crate::{authorized, clients, cluster_tls, error, AppState};

/// The topics, in the order the sampler reads them (the cheap ones first).
pub const TOPICS: [&str; 5] = ["status", "queue", "jobs", "workshop", "metrics"];
/// What a client profile's key (and a cluster node's) may subscribe to.
const PUBLIC_TOPICS: [&str; 2] = ["status", "queue"];
/// One message from a client (a request's body included). Larger bodies go over HTTP.
const MAX_MESSAGE: usize = 16 << 20;
/// The largest answer forwarded over the socket (a media file is fetched over HTTP).
const MAX_RESPONSE: usize = 32 << 20;
const MAX_IN_FLIGHT: usize = 32;
const AUTH_WAIT: Duration = Duration::from_secs(5);
const PING_EVERY: Duration = Duration::from_secs(25);
/// No frame at all (not even a pong) for this long: the peer is gone.
const DEAD_AFTER: Duration = Duration::from_secs(60);
/// A peer that takes no data for this long is dropped (its buffers are full: it is not reading).
const SEND_WAIT: Duration = Duration::from_secs(30);
/// The sampler's pace while someone is subscribed and nothing woke it.
const SAMPLE_EVERY: Duration = Duration::from_secs(1);
/// Changes in a burst (a job leaves, the next takes the turn) become one update.
const COALESCE: Duration = Duration::from_millis(60);
/// Close code for a refused key, so a client can tell it from a network drop and not retry at once.
const CLOSE_UNAUTHORIZED: u16 = 4401;
const JOBS_SHOWN: usize = 50;

// ----------------------------------------------------------------------------------------------------------- the hub

struct Push {
    topic: usize,
    value: Value,
}

#[derive(Default)]
struct HubState {
    /// subscribers per topic
    counts: [usize; TOPICS.len()],
    sampling: bool,
}

struct Hub {
    tx: broadcast::Sender<Arc<Push>>,
    state: Mutex<HubState>,
    /// someone is subscribed (read by `poke`, which runs on the queue's paths: no lock)
    active: AtomicBool,
    wake: Notify,
}

static HUB: OnceLock<Hub> = OnceLock::new();
static APP: OnceLock<axum::Router> = OnceLock::new();

fn hub() -> &'static Hub {
    HUB.get_or_init(|| Hub { tx: broadcast::channel(64).0, state: Mutex::default(), active: AtomicBool::new(false), wake: Notify::new() })
}

/// The finished router, for the requests sent over the socket (set once at start).
pub fn set_app(app: axum::Router) {
    let _ = APP.set(app);
}

/// Something the topics show has changed (the queue, a Workshop job): subscribers hear it now instead of at the
/// sampler's next look. Costs one atomic read while nobody is subscribed.
pub fn poke() {
    if let Some(h) = HUB.get() {
        if h.active.load(Ordering::Relaxed) {
            h.wake.notify_one();
        }
    }
}

impl Hub {
    fn join(&self, s: &AppState, topic: usize) {
        let mut st = self.state.lock().unwrap();
        st.counts[topic] += 1;
        if !st.sampling {
            st.sampling = true;
            self.active.store(true, Ordering::Relaxed);
            tokio::spawn(sampler(s.clone()));
        }
    }

    fn leave(&self, topic: usize) {
        let mut st = self.state.lock().unwrap();
        st.counts[topic] = st.counts[topic].saturating_sub(1);
    }
}

/// The one task behind every subscription: reads the subscribed topics when woken (or once a second), and broadcasts
/// a topic only when what it shows changed. It ends when the last subscriber leaves.
async fn sampler(s: AppState) {
    let hub = hub();
    let mut last: [Option<u64>; TOPICS.len()] = [None; TOPICS.len()];
    let mut metrics_at: Option<Instant> = None;
    loop {
        let counts = {
            let mut st = hub.state.lock().unwrap();
            if st.counts.iter().all(|c| *c == 0) {
                st.sampling = false;
                hub.active.store(false, Ordering::Relaxed);
                return;
            }
            st.counts
        };
        for (i, topic) in TOPICS.iter().enumerate() {
            if counts[i] == 0 {
                last[i] = None;
                continue;
            }
            // the Monitor's figures move all the time while a request runs: once a second, however often it is woken
            if *topic == "metrics" {
                if metrics_at.is_some_and(|t| t.elapsed() < SAMPLE_EVERY - Duration::from_millis(100)) {
                    continue;
                }
                metrics_at = Some(Instant::now());
            }
            // a topic that cannot be read right now (a model is being stopped) never holds the others back
            let Ok(value) = tokio::time::timeout(Duration::from_millis(300), snapshot(&s, topic, None)).await else { continue };
            let f = fingerprint(topic, &value);
            if last[i] != Some(f) {
                last[i] = Some(f);
                let _ = hub.tx.send(Arc::new(Push { topic: i, value }));
            }
        }
        tokio::select! {
            _ = hub.wake.notified() => tokio::time::sleep(COALESCE).await,
            _ = tokio::time::sleep(SAMPLE_EVERY) => {}
        }
    }
}

// -------------------------------------------------------------------------------------------------------- the topics

/// A topic as the server's own key sees it. `page`: the Monitor's request-history page (metrics only).
async fn snapshot(s: &AppState, topic: &str, page: Option<(usize, usize)>) -> Value {
    match topic {
        "status" => status(s),
        "queue" => s.mgr.queue.snapshot(),
        "jobs" => json!({"jobs": s.media_jobs.newest(JOBS_SHOWN)}),
        "workshop" => json!({"jobs": crate::workshop::get().map(|w| w.list().iter().map(|j| j.view(false)).collect::<Vec<_>>()).unwrap_or_default()}),
        "metrics" => crate::web::metrics_value(s, Some(page.unwrap_or((1, crate::monitor::PAGE_DEFAULT)))).await,
        _ => Value::Null,
    }
}

/// The model and what the server is doing, with nothing of anyone's requests in it (every key may see it). Read
/// without waiting for a lock a model switch holds, so "switching" is said while it happens.
fn status(s: &AppState) -> Value {
    let q = s.mgr.queue.snapshot();
    let loaded = q["resident"]["text_model"].as_str().map(str::to_string);
    let shown = loaded.clone().unwrap_or_else(|| s.mgr.default_id.clone());
    let name = s.mgr.entry(&shown).map(|e| e.display_name()).unwrap_or_else(|| shown.clone());
    let live = s.mgr.monitor.metrics(Value::Null, loaded.is_some(), Some((1, 1)))["live"].take();
    let running = &q["running"];
    let kind = running["kind"].as_str();
    let mut state = live["state"].as_str().unwrap_or("idle").to_string();
    let mut target = Value::Null;
    if kind == Some("switch") {
        state = "switching".into();
        target = running["label"].clone();
    } else if kind == Some("unload") {
        state = "unloading".into();
    } else if state == "unloaded" && running["need"]["text_model"].is_string() {
        // a request has the turn and no model is in memory yet: it is being loaded for it
        state = "loading".into();
        target = running["need"]["text_model"].clone();
    }
    let media = if live["state"] == "media" { json!({"kind": live["kind"], "model": live["model"], "step": live["step"], "steps": live["steps"]}) }
                else { Value::Null };
    json!({"model": {"id": shown, "name": name}, "loaded": loaded.is_some(), "default": s.mgr.default_id,
           "state": state, "phase": live["phase"], "target": target, "media": media,
           "queue": {"running": kind, "waiting": q["waiting"].as_array().map(|a| a.len()).unwrap_or(0)},
           "alerts": s.mgr.monitor.alerts().len()})
}

/// The queue for a client profile's key: how busy the server is, and its own waiting jobs only.
fn queue_of(full: &Value, client: &str) -> Value {
    let waiting = full["waiting"].as_array().cloned().unwrap_or_default();
    let mine: Vec<Value> = waiting.iter().filter(|w| w["client"] == client).cloned().collect();
    json!({"running": if full["running"].is_null() { Value::Null } else { json!({"kind": full["running"]["kind"]}) },
           "waiting": waiting.len(), "mine": mine})
}

/// What a topic shows, without the clocks that tick while nothing happens (a job's seconds so far): a change of this
/// is a change worth sending.
fn fingerprint(topic: &str, v: &Value) -> u64 {
    fn strip(v: &mut Value, keys: &[&str]) {
        match v {
            Value::Object(o) => {
                o.retain(|k, _| !keys.contains(&k.as_str()));
                o.values_mut().for_each(|x| strip(x, keys));
            }
            Value::Array(a) => a.iter_mut().for_each(|x| strip(x, keys)),
            _ => {}
        }
    }
    use std::hash::{Hash, Hasher};
    let mut hs = std::collections::hash_map::DefaultHasher::new();
    let mut c = v.clone();
    match topic {
        "queue" => strip(&mut c, &["running_s", "waiting_s", "eta_s"]),
        "metrics" => {
            if let Some(o) = c.as_object_mut() {
                o.remove("time");
            }
            strip(&mut c["queue"], &["running_s", "waiting_s", "eta_s"]);
        }
        _ => {}
    }
    c.to_string().hash(&mut hs);
    hs.finish()
}

// ----------------------------------------------------------------------------------------------------------- the key

#[derive(Clone, Debug, PartialEq)]
enum Role {
    /// the server's own key (or a server without one)
    Owner,
    /// a master of this worker, with the key the worker gave it (cluster.rs)
    Node,
    /// a client profile's key
    Client(String),
}

impl Role {
    fn name(&self) -> &'static str {
        match self { Role::Owner => "owner", Role::Node => "node", Role::Client(_) => "client" }
    }

    fn may(&self, topic: &str) -> bool {
        *self == Role::Owner || PUBLIC_TOPICS.contains(&topic)
    }
}

fn role_of(s: &AppState, headers: &HeaderMap) -> Option<Role> {
    let given = clients::given_key(headers);
    // a client profile first: on a server without a key of its own `authorized` passes everyone
    if let Some(p) = given.and_then(|k| s.clients.by_key(k)) {
        return Some(Role::Client(p.name.clone()));
    }
    if authorized(s, headers) {
        return Some(Role::Owner);
    }
    given.is_some_and(|k| s.cluster.is_master_key(k, cluster_tls::peer_of(headers))).then_some(Role::Node)
}

/// What the connection was opened with: carried into each request it sends, so the router sees it as it would an
/// HTTP request on that connection (its address, whether it came over TLS, a cluster peer's pin, the page's Origin).
struct Ctx {
    headers: HeaderMap,
    addr: Option<ConnectInfo<std::net::SocketAddr>>,
    tls: Option<crate::server_tls::OverTls>,
    peer: Option<cluster_tls::Peer>,
}

impl Ctx {
    /// The upgrade's headers with `key` as the Authorization (the key that came in the first message).
    fn with_key(&self, key: &str) -> Option<HeaderMap> {
        let mut h = self.headers.clone();
        h.remove("x-api-key");
        h.remove("x-goog-api-key");
        h.insert(axum::http::header::AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {key}")).ok()?);
        Some(h)
    }
}

// ------------------------------------------------------------------------------------------------------- the socket

/// GET /v1/live.
pub async fn live(State(s): State<AppState>, ws: Result<WebSocketUpgrade, axum::extract::ws::rejection::WebSocketUpgradeRejection>,
                  req: axum::extract::Request) -> Response {
    let headers = req.headers().clone();
    // a plain request (not a WebSocket): the key first, like every route, then "use a WebSocket"
    let Ok(ws) = ws else {
        if role_of(&s, &headers).is_none() {
            return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
        }
        return error(StatusCode::UPGRADE_REQUIRED, "invalid_request_error", "this is a WebSocket (the live channel): connect with ws:// or wss://");
    };
    // another site's page must not open this socket with the user's browser (cross-site WebSocket hijacking)
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("the live channel can be opened");
    }
    let ext = req.extensions();
    let ctx = Ctx { addr: ext.get::<ConnectInfo<std::net::SocketAddr>>().cloned(), tls: ext.get::<crate::server_tls::OverTls>().copied(),
                    peer: ext.get::<cluster_tls::Peer>().cloned(), headers };
    ws.max_message_size(MAX_MESSAGE).max_frame_size(MAX_MESSAGE).on_upgrade(move |sock| session(s, ctx, sock))
}

type Sink = SplitSink<WebSocket, Message>;

async fn send(tx: &mut Sink, v: &Value) -> bool {
    send_text(tx, v.to_string()).await
}

async fn send_text(tx: &mut Sink, text: String) -> bool {
    matches!(tokio::time::timeout(SEND_WAIT, tx.send(Message::Text(text.into()))).await, Ok(Ok(())))
}

fn error_msg(id: Option<&str>, message: impl Into<String>) -> Value {
    match id {
        Some(id) => json!({"type": "error", "id": id, "message": message.into()}),
        None => json!({"type": "error", "message": message.into()}),
    }
}

struct Sub {
    /// the Monitor's request-history page this connection asked for (metrics)
    page: Option<(usize, usize)>,
    /// what was last sent (`fingerprint`), so the same state is never sent twice
    sent: Option<u64>,
}

struct Conn {
    s: AppState,
    ctx: Arc<Ctx>,
    role: Role,
    key: Option<String>,
    tx: Sink,
    subs: HashMap<usize, Sub>,
    /// the requests in flight: each a task, ended by `cancel` or with the connection
    tasks: tokio::task::JoinSet<String>,
    in_flight: HashMap<String, (tokio::task::AbortHandle, u64)>,
    /// each request's number on this connection, and the last cancelled ones: what a cancelled request had already
    /// queued is not sent after its `cancelled`
    next: u64,
    cancelled: std::collections::VecDeque<u64>,
    out: mpsc::Sender<(u64, String)>,
}

impl Drop for Conn {
    fn drop(&mut self) {
        for topic in self.subs.keys() {
            hub().leave(*topic);
        }
        // (the JoinSet aborts the requests still running: their models stop as for a closed HTTP connection)
    }
}

async fn session(s: AppState, ctx: Ctx, sock: WebSocket) {
    let (mut tx, mut rx) = sock.split();
    // the key: the Authorization header (a CLI, a node), else the first message (the browser)
    let mut key = clients::given_key(&ctx.headers).map(str::to_string);
    let mut role = role_of(&s, &ctx.headers);
    if role.is_none() {
        if let Ok(Some(Ok(Message::Text(t)))) = tokio::time::timeout(AUTH_WAIT, rx.next()).await {
            let v: Value = serde_json::from_str(t.as_str()).unwrap_or_default();
            if let (Some("auth"), Some(k)) = (v["type"].as_str(), v["key"].as_str()) {
                role = ctx.with_key(k).and_then(|h| role_of(&s, &h));
                key = Some(k.to_string());
            }
        }
    }
    let Some(role) = role else {
        let _ = send(&mut tx, &json!({"type": "error", "code": 401,
            "message": "a valid API key is required: the Authorization header, or {\"type\":\"auth\",\"key\":\"...\"} as the first message"})).await;
        let _ = tx.send(Message::Close(Some(CloseFrame { code: CLOSE_UNAUTHORIZED, reason: "a valid API key is required".into() }))).await;
        return;
    };
    let (out, mut out_rx) = mpsc::channel::<(u64, String)>(256);
    let mut c = Conn { s, ctx: Arc::new(ctx), role, key, tx, subs: HashMap::new(), tasks: tokio::task::JoinSet::new(),
                       in_flight: HashMap::new(), next: 0, cancelled: Default::default(), out };
    if !c.ready().await {
        return;
    }
    let mut pushes = hub().tx.subscribe();
    let mut ping = tokio::time::interval_at(tokio::time::Instant::now() + PING_EVERY, PING_EVERY);
    ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut seen = Instant::now();
    loop {
        tokio::select! {
            msg = rx.next() => {
                seen = Instant::now();
                match msg {
                    Some(Ok(Message::Text(t))) => if !c.message(t.as_str()).await { break },
                    Some(Ok(Message::Binary(_))) => if !c.say(error_msg(None, "text frames only: each message is one JSON object")).await { break },
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(_)) => {} // a ping (answered by the socket itself) or a pong
                    Some(Err(e)) => {
                        // a message over the size cap, or a broken frame: said plainly, then the connection ends
                        let _ = c.say(error_msg(None, format!("the connection is closed: {e} (one message may be at most {} MB; larger bodies go over HTTP)", MAX_MESSAGE >> 20))).await;
                        break;
                    }
                }
            }
            Some((n, text)) = out_rx.recv() => if !c.cancelled.contains(&n) && !send_text(&mut c.tx, text).await { break },
            Some(done) = c.tasks.join_next(), if !c.tasks.is_empty() => match done {
                Ok(id) => { c.in_flight.remove(&id); }
                Err(_) => c.in_flight.retain(|_, (h, _)| !h.is_finished()),
            },
            p = pushes.recv() => match p {
                Ok(p) => if !c.push(&p).await { break },
                // this connection fell behind: every topic it follows is read again
                Err(broadcast::error::RecvError::Lagged(_)) => if !c.refresh().await { break },
                Err(broadcast::error::RecvError::Closed) => break,
            },
            _ = ping.tick() => {
                if seen.elapsed() > DEAD_AFTER {
                    break;
                }
                if !matches!(tokio::time::timeout(SEND_WAIT, c.tx.send(Message::Ping(Default::default()))).await, Ok(Ok(()))) {
                    break;
                }
            }
        }
    }
    let _ = c.tx.send(Message::Close(None)).await;
}

impl Conn {
    async fn say(&mut self, v: Value) -> bool {
        send(&mut self.tx, &v).await
    }

    async fn ready(&mut self) -> bool {
        let topics: Vec<&str> = TOPICS.iter().copied().filter(|t| self.role.may(t)).collect();
        let mut v = json!({"type": "ready", "role": self.role.name(), "topics": topics, "server": "am2Strata", "version": env!("CARGO_PKG_VERSION")});
        if let Role::Client(name) = &self.role {
            v["client"] = json!(name);
        }
        self.say(v).await
    }

    /// One message from the client. false: the connection is gone.
    async fn message(&mut self, text: &str) -> bool {
        let v: Value = match serde_json::from_str(text) {
            Ok(v @ Value::Object(_)) => v,
            Ok(_) => return self.say(error_msg(None, "each message is one JSON object with a \"type\"")).await,
            Err(e) => return self.say(error_msg(None, format!("not JSON: {e}"))).await,
        };
        let id = request_id(&v);
        match v["type"].as_str().unwrap_or("") {
            "ping" => {
                let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
                self.say(json!({"type": "pong", "t": t})).await
            }
            // the key again (the web app always sends it first, also to a server that needs none): the connection
            // keeps the access it was opened with
            "auth" => self.ready().await,
            "subscribe" => self.subscribe(&v).await,
            "unsubscribe" => {
                let Some(names) = topic_names(&v) else { return self.say(error_msg(None, "\"topics\" is a list of topic names")).await };
                for i in names.iter().filter_map(|n| TOPICS.iter().position(|t| t == n)) {
                    if self.subs.remove(&i).is_some() {
                        hub().leave(i);
                    }
                }
                self.say(json!({"type": "unsubscribed", "topics": names})).await
            }
            "request" => {
                let Some(id) = id else { return self.say(error_msg(None, "a request needs an \"id\": a string of 1 to 128 characters")).await };
                if self.in_flight.contains_key(&id) {
                    return self.say(error_msg(Some(&id), "a request with this id is still running")).await;
                }
                if self.in_flight.len() >= MAX_IN_FLIGHT {
                    return self.say(error_msg(Some(&id), format!("{MAX_IN_FLIGHT} requests are already running on this connection"))).await;
                }
                let req = match build_request(&self.ctx, self.key.as_deref(), &v) {
                    Ok(r) => r,
                    Err(e) => return self.say(error_msg(Some(&id), e)).await,
                };
                let Some(app) = APP.get().cloned() else { return self.say(error_msg(Some(&id), "the server is still starting")).await };
                self.next += 1;
                let (out, rid, n) = (self.out.clone(), id.clone(), self.next);
                let handle = self.tasks.spawn(async move {
                    run_request(app, req, &rid, Out { n, tx: out }).await;
                    rid
                });
                self.in_flight.insert(id, (handle, n));
                true
            }
            "cancel" => {
                let Some(id) = id else { return self.say(error_msg(None, "cancel needs the request's \"id\"")).await };
                match self.in_flight.remove(&id) {
                    Some((h, n)) => {
                        h.abort();
                        if self.cancelled.len() >= 64 {
                            self.cancelled.pop_front();
                        }
                        self.cancelled.push_back(n);
                        self.say(json!({"type": "cancelled", "id": id})).await
                    }
                    None => self.say(error_msg(Some(&id), "no request with this id is running (it may have finished)")).await,
                }
            }
            "" => self.say(error_msg(id.as_deref(), "a message needs a \"type\": auth, subscribe, unsubscribe, request, cancel or ping")).await,
            other => self.say(error_msg(id.as_deref(), format!("unknown type {other:?}: auth, subscribe, unsubscribe, request, cancel or ping"))).await,
        }
    }

    async fn subscribe(&mut self, v: &Value) -> bool {
        let Some(names) = topic_names(v) else { return self.say(error_msg(None, "\"topics\" is a list of topic names")).await };
        let mut added = Vec::new();
        for name in &names {
            let Some(i) = TOPICS.iter().position(|t| t == name) else {
                if !self.say(error_msg(None, format!("unknown topic {name:?}: {}", TOPICS.join(", ")))).await { return false }
                continue;
            };
            if !self.role.may(TOPICS[i]) {
                if !self.say(error_msg(None, format!("the topic {name:?} needs the server's own API key"))).await { return false }
                continue;
            }
            // the Monitor's request-history page: {"params": {"metrics": {"page": 2, "per": 50}}}
            let p = &v["params"][TOPICS[i]];
            let page = (TOPICS[i] == "metrics" && (p["page"].is_u64() || p["per"].is_u64()))
                .then(|| (p["page"].as_u64().unwrap_or(1) as usize, p["per"].as_u64().unwrap_or(crate::monitor::PAGE_DEFAULT as u64) as usize));
            if self.subs.insert(i, Sub { page, sent: None }).is_none() {
                hub().join(&self.s, i);
            }
            added.push(TOPICS[i]);
        }
        if !self.say(json!({"type": "subscribed", "topics": added})).await {
            return false;
        }
        // where each stands now; changes follow
        for name in added {
            let i = TOPICS.iter().position(|t| *t == name).unwrap();
            if !self.update(i, None, true).await {
                return false;
            }
        }
        true
    }

    /// The sampler saw a topic change.
    async fn push(&mut self, p: &Push) -> bool {
        if !self.subs.contains_key(&p.topic) {
            return true;
        }
        self.update(p.topic, Some(&p.value), false).await
    }

    async fn refresh(&mut self) -> bool {
        for i in self.subs.keys().copied().collect::<Vec<_>>() {
            if !self.update(i, None, false).await {
                return false;
            }
        }
        true
    }

    /// Send topic `i` as this connection may see it, unless that is what it was sent last. `base`: the topic as the
    /// sampler just read it (None: read now).
    async fn update(&mut self, i: usize, base: Option<&Value>, first: bool) -> bool {
        let topic = TOPICS[i];
        let page = self.subs.get(&i).and_then(|s| s.page);
        let read;
        let full: &Value = match base {
            // another page of the Monitor's history than the one the sampler reads
            Some(v) if page.is_none() => v,
            _ => match tokio::time::timeout(Duration::from_secs(3), snapshot(&self.s, topic, page)).await {
                Ok(v) => { read = v; &read }
                Err(_) => return true, // busy (a model is being stopped): the next change brings it
            },
        };
        let own;
        let data: &Value = match (&self.role, topic) {
            (Role::Client(name), "queue") => { own = queue_of(full, name); &own }
            _ => full,
        };
        let f = fingerprint(topic, data);
        let Some(sub) = self.subs.get_mut(&i) else { return true };
        if sub.sent == Some(f) && !first {
            return true;
        }
        sub.sent = Some(f);
        let mut msg = json!({"type": "update", "topic": topic, "data": data});
        if first {
            msg["snapshot"] = json!(true);
        }
        self.say(msg).await
    }
}

fn request_id(v: &Value) -> Option<String> {
    let id = match &v["id"] {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        _ => return None,
    };
    (!id.is_empty() && id.len() <= 128).then_some(id)
}

/// "topics": ["a", "b"] (or "topic": "a").
fn topic_names(v: &Value) -> Option<Vec<String>> {
    match (&v["topics"], &v["topic"]) {
        (Value::Array(a), _) => a.iter().map(|x| x.as_str().map(str::to_string)).collect(),
        (Value::String(s), _) | (_, Value::String(s)) => Some(vec![s.clone()]),
        _ => None,
    }
}

// ------------------------------------------------------------------------------------------------------ the requests

/// Headers a request over the socket may set itself; the rest come from the connection (its key, Host and Origin).
fn header_allowed(name: &str) -> bool {
    matches!(name, "accept" | "accept-language" | "anthropic-version" | "anthropic-beta" | "openai-beta" | "idempotency-key")
        || (name.starts_with("x-strata-") && name != cluster_tls::PEER_HEADER)
}

/// The HTTP request a `request` message stands for, as the router would get it from this connection.
fn build_request(ctx: &Ctx, key: Option<&str>, v: &Value) -> Result<Request<Body>, String> {
    let method = match v["method"].as_str().unwrap_or("GET").to_ascii_uppercase().as_str() {
        "GET" => Method::GET, "POST" => Method::POST, "PUT" => Method::PUT, "DELETE" => Method::DELETE, "PATCH" => Method::PATCH,
        other => return Err(format!("method {other:?} is not one of GET, POST, PUT, DELETE, PATCH")),
    };
    let path = v["path"].as_str().ok_or("a request needs a \"path\", e.g. /v1/chat/completions")?;
    let uri: axum::http::Uri = path.parse().map_err(|_| format!("{path:?} is not a path"))?;
    let p = uri.path();
    if uri.scheme().is_some() || uri.authority().is_some() || !(p.starts_with("/v1/") || p.starts_with("/api/")) {
        return Err("only paths under /v1/ and /api/ can be called over the live channel".into());
    }
    if p.trim_end_matches('/') == "/v1/live" {
        return Err("the live channel cannot be opened through itself".into());
    }
    let mut req = Request::builder().method(method).uri(uri);
    let h = req.headers_mut().ok_or("bad request")?;
    for name in ["host", "origin", "user-agent"] {
        if let Some(val) = ctx.headers.get(name) {
            h.insert(HeaderName::from_static(name), val.clone());
        }
    }
    if let Some(extra) = v["headers"].as_object() {
        for (k, val) in extra {
            let k = k.to_ascii_lowercase();
            if !header_allowed(&k) {
                return Err(format!("the header {k:?} cannot be set on a request over the live channel (x-strata-*, accept, anthropic-version, anthropic-beta, openai-beta, idempotency-key)"));
            }
            let (Ok(n), Some(Ok(val))) = (HeaderName::from_bytes(k.as_bytes()), val.as_str().map(HeaderValue::from_str)) else {
                return Err(format!("the header {k:?} needs a text value"));
            };
            h.insert(n, val);
        }
    }
    if let Some(k) = key.filter(|k| !k.is_empty()) {
        h.insert(axum::http::header::AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {k}")).map_err(|_| "bad key")?);
    }
    let body = match &v["body"] {
        Value::Null => Body::empty(),
        b => {
            h.insert(axum::http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
            Body::from(serde_json::to_vec(b).map_err(|e| e.to_string())?)
        }
    };
    let mut req = req.body(body).map_err(|e| e.to_string())?;
    let ext = req.extensions_mut();
    if let Some(a) = ctx.addr { ext.insert(a); }
    if let Some(t) = ctx.tls { ext.insert(t); }
    if let Some(p) = ctx.peer.clone() { ext.insert(p); }
    Ok(req)
}

/// Where the first complete SSE event in `buf` ends, and the length of the blank line after it.
fn sse_boundary(buf: &[u8]) -> Option<(usize, usize)> {
    let lf = buf.windows(2).position(|w| w == b"\n\n").map(|i| (i, 2));
    let crlf = buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| (i, 4));
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(if a.0 < b.0 { a } else { b }),
        (a, b) => a.or(b),
    }
}

/// One SSE event as an `event` message: its data (JSON where it is JSON, else the text, e.g. "[DONE]"), its name if it
/// has one, or a comment line (the server's ": queued: 2 ahead" while a request waits its turn).
fn sse_event(id: &str, block: &str) -> Option<Value> {
    let (mut name, mut data, mut comment) = (None, Vec::new(), Vec::new());
    for line in block.lines() {
        if let Some(c) = line.strip_prefix(':') {
            comment.push(c.trim());
        } else if let Some(n) = line.strip_prefix("event:") {
            name = Some(n.trim());
        } else if let Some(d) = line.strip_prefix("data:") {
            data.push(d.strip_prefix(' ').unwrap_or(d));
        }
    }
    let mut ev = json!({"type": "event", "id": id});
    if let Some(n) = name {
        ev["event"] = json!(n);
    }
    if !data.is_empty() {
        let d = data.join("\n");
        ev["data"] = serde_json::from_str(&d).unwrap_or(Value::String(d));
    } else if !comment.is_empty() {
        ev["comment"] = json!(comment.join("\n"));
    } else if name.is_none() {
        return None;
    }
    Some(ev)
}

/// A request's way to its connection's socket.
struct Out {
    n: u64,
    tx: mpsc::Sender<(u64, String)>,
}

impl Out {
    async fn send(&self, text: String) -> Result<(), ()> {
        self.tx.send((self.n, text)).await.map_err(|_| ())
    }
}

/// Run one request through the router and send its answer: `response`, or `event`s and `done` for a stream. Dropped
/// (cancel, or the connection closed) it drops the router's future or the body it was reading, as hyper does when an
/// HTTP client goes away.
async fn run_request(app: axum::Router, req: Request<Body>, id: &str, out: Out) {
    let resp = match tower::ServiceExt::oneshot(app, req).await {
        Ok(r) => r,
        Err(never) => match never {},
    };
    let (head, body) = resp.into_parts();
    let status = head.status.as_u16();
    let header = |n: &str| head.headers.get(n).and_then(|v| v.to_str().ok()).map(str::to_string);
    let ctype = header("content-type").unwrap_or_default().to_ascii_lowercase();
    let mut shown = serde_json::Map::new();
    for n in ["content-type", "retry-after", "x-strata-warning", "location"] {
        if let Some(v) = header(n) {
            shown.insert(n.to_string(), json!(v));
        }
    }
    let mut body = body.into_data_stream();
    let too_large = || error_msg(Some(id), format!("the answer is over {} MB: ask for it over HTTP", MAX_RESPONSE >> 20)).to_string();
    let mut buf: Vec<u8> = Vec::new();
    if ctype.starts_with("text/event-stream") {
        while let Some(chunk) = body.next().await {
            let Ok(chunk) = chunk else {
                let _ = out.send(error_msg(Some(id), "the stream broke before its end").to_string()).await;
                return;
            };
            buf.extend_from_slice(&chunk);
            while let Some((end, gap)) = sse_boundary(&buf) {
                let block: Vec<u8> = buf.drain(..end + gap).take(end).collect();
                if let Some(ev) = sse_event(id, &String::from_utf8_lossy(&block)) {
                    if out.send(ev.to_string()).await.is_err() {
                        return;
                    }
                }
            }
            if buf.len() > MAX_RESPONSE {
                let _ = out.send(too_large()).await;
                return;
            }
        }
        if let Some(ev) = sse_event(id, &String::from_utf8_lossy(&buf)) {
            let _ = out.send(ev.to_string()).await;
        }
        let _ = out.send(json!({"type": "done", "id": id, "status": status}).to_string()).await;
        return;
    }
    while let Some(chunk) = body.next().await {
        let Ok(chunk) = chunk else {
            let _ = out.send(error_msg(Some(id), "the answer broke before its end").to_string()).await;
            return;
        };
        buf.extend_from_slice(&chunk);
        if buf.len() > MAX_RESPONSE {
            let _ = out.send(too_large()).await;
            return;
        }
    }
    let mut msg = json!({"type": "response", "id": id, "status": status, "headers": shown});
    if buf.is_empty() {
        msg["body"] = Value::Null;
    } else if let Some(j) = ctype.contains("json").then(|| serde_json::from_slice::<Value>(&buf).ok()).flatten() {
        msg["body"] = j;
    } else if let Some(t) = (ctype.starts_with("text/") || ctype.contains("json") || ctype.is_empty()).then(|| String::from_utf8(buf.clone()).ok()).flatten() {
        msg["body"] = json!(t);
    } else {
        use base64::Engine as _;
        msg["body_base64"] = json!(base64::engine::general_purpose::STANDARD.encode(&buf));
    }
    let _ = out.send(msg.to_string()).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> Ctx {
        let mut headers = HeaderMap::new();
        headers.insert("host", HeaderValue::from_static("127.0.0.1:8000"));
        headers.insert("origin", HeaderValue::from_static("http://127.0.0.1:8000"));
        headers.insert("cookie", HeaderValue::from_static("not-carried"));
        Ctx { headers, addr: Some(ConnectInfo(([127, 0, 0, 1], 5000).into())), tls: None, peer: None }
    }

    #[test]
    fn only_api_paths_and_never_the_channel_itself() {
        let ok = |path: &str| build_request(&ctx(), Some("k"), &json!({"method": "GET", "path": path}));
        assert!(ok("/v1/models").is_ok());
        assert!(ok("/api/queue?x=1").is_ok());
        for bad in ["/settings", "/config", "/metrics", "/", "v1/models", "/v1", "http://other.example/v1/models", "//other.example/v1/models",
                    "/v1/live", "/v1/live/", "/v1/live?x=1", "/web/app.js", "/preview/1"] {
            assert!(ok(bad).is_err(), "{bad} must be refused");
        }
        assert!(build_request(&ctx(), None, &json!({"method": "TRACE", "path": "/v1/models"})).is_err());
        assert!(build_request(&ctx(), None, &json!({"method": "GET"})).is_err());
    }

    #[test]
    fn a_request_carries_the_connections_key_and_origin_not_its_own() {
        let r = build_request(&ctx(), Some("the-key"), &json!({"method": "post", "path": "/v1/chat/completions", "body": {"a": 1},
                                                              "headers": {"X-Strata-Priority": "low"}})).unwrap();
        assert_eq!(r.method(), Method::POST);
        assert_eq!(r.headers()["authorization"], "Bearer the-key");
        assert_eq!(r.headers()["origin"], "http://127.0.0.1:8000");
        assert_eq!(r.headers()["host"], "127.0.0.1:8000");
        assert_eq!(r.headers()["content-type"], "application/json");
        assert_eq!(r.headers()["x-strata-priority"], "low");
        assert!(r.headers().get("cookie").is_none());
        assert!(r.extensions().get::<ConnectInfo<std::net::SocketAddr>>().is_some(), "the peer's address goes with it");
        // a request may not name its own key, host, origin or a cluster peer
        for h in ["authorization", "x-api-key", "host", "origin", "cookie", cluster_tls::PEER_HEADER] {
            assert!(build_request(&ctx(), Some("k"), &json!({"path": "/v1/models", "headers": {h: "x"}})).is_err(), "{h}");
        }
        // no key (a server without one): no Authorization at all
        assert!(build_request(&ctx(), None, &json!({"path": "/v1/models"})).unwrap().headers().get("authorization").is_none());
    }

    #[test]
    fn sse_events_become_messages() {
        assert_eq!(sse_boundary(b"data: 1\n\ndata: 2\n\n"), Some((7, 2)));
        assert_eq!(sse_boundary(b"data: 1\r\n\r\nrest"), Some((7, 4)));
        assert_eq!(sse_boundary(b"data: 1\n"), None);
        assert_eq!(sse_event("a", "data: {\"x\":1}").unwrap(), json!({"type": "event", "id": "a", "data": {"x": 1}}));
        assert_eq!(sse_event("a", "data: [DONE]").unwrap()["data"], "[DONE]");
        assert_eq!(sse_event("a", "event: message_start\ndata: {}").unwrap(), json!({"type": "event", "id": "a", "event": "message_start", "data": {}}));
        assert_eq!(sse_event("a", ": queued: 2 ahead").unwrap()["comment"], "queued: 2 ahead");
        assert_eq!(sse_event("a", "data: one\ndata: two").unwrap()["data"], "one\ntwo");
        assert!(sse_event("a", "").is_none());
    }

    #[test]
    fn clocks_alone_are_not_a_change_and_a_client_sees_only_its_own_jobs() {
        let q = |s: u64| json!({"running": {"id": 1, "kind": "chat", "label": "secret prompt", "running_s": s},
                                "waiting": [{"position": 1, "id": 2, "client": "laptop", "label": "mine", "waiting_s": s, "eta_s": s},
                                            {"position": 2, "id": 3, "client": "phone", "label": "theirs", "waiting_s": s, "eta_s": s}]});
        assert_eq!(fingerprint("queue", &q(1)), fingerprint("queue", &q(9)));
        let mut other = q(1);
        other["waiting"].as_array_mut().unwrap().pop();
        assert_ne!(fingerprint("queue", &q(1)), fingerprint("queue", &other));
        let mine = queue_of(&q(1), "laptop");
        assert_eq!(mine["waiting"], 2);
        assert_eq!(mine["running"], json!({"kind": "chat"}));
        assert_eq!(mine["mine"].as_array().unwrap().len(), 1);
        assert!(!mine.to_string().contains("theirs") && !mine.to_string().contains("secret"), "{mine}");
    }

    #[test]
    fn topics_by_role() {
        assert!(TOPICS.iter().all(|t| Role::Owner.may(t)));
        for r in [Role::Node, Role::Client("laptop".into())] {
            assert!(r.may("status") && r.may("queue"));
            assert!(!r.may("jobs") && !r.may("workshop") && !r.may("metrics"));
        }
    }
}
