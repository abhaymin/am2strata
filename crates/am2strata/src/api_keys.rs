//! Generated API keys and usage (docs/DEPLOYMENT.md, "API keys and usage").
//!
//! The owner makes keys while the server runs (Settings › Clients, `am2strata-run keys`, POST /api/keys): one per
//! app, user or cluster client. A generated key is a client profile (clients.rs) made at run time: it opens the model
//! APIs (chat, messages, responses, Gemini, media) with its tool sets and memory space, and never the owner's routes
//! (settings, the Workshop, the Computer tab, the keys themselves).
//!
//! - `<data-dir>/api-keys.json` (mode 0600) holds each key's SHA-256 hash, never the key. The key itself
//!   (`am2_` + 40 random URL-safe characters) is given once, in the answer that creates it, and is not logged.
//! - One layer around the model APIs refuses a revoked or expired key (401), a key past its day's limits (429) or
//!   asking for a model outside its list (403), and counts every caller's use: per day (UTC), per key, per model:
//!   requests, prompt / completion / cached tokens, media jobs, errors, last use. Streams are read as they go out.
//! - `<data-dir>/usage/YYYY-MM.json` keeps the counts, written every 20 s when they changed and at shutdown.
//!
//! Usage is counted for the main key ("owner"), each run-config client profile ("client:<name>"), a cluster master
//! ("cluster") and each generated key (by its id).

use crate::{authorized, clients, error, AppState};
use axum::body::{Body, Bytes};
use axum::extract::{Path as UrlPath, Query, Request, State};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

/// How every generated key starts (only such a key is looked up in the store).
pub const KEY_PREFIX: &str = "am2_";
const KEY_RANDOM_CHARS: usize = 40;
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
/// The identity of the main key in the usage counts.
pub const OWNER: &str = "owner";
const CLUSTER: &str = "cluster";
/// Names a key may not take: they are the queue's and the usage counts' own.
const RESERVED: [&str; 5] = ["default", "web", "api", OWNER, CLUSTER];
const FLUSH_EVERY: std::time::Duration = std::time::Duration::from_secs(20);

// ------------------------------------------------------------------ days (UTC)

/// The UTC day of a Unix time, as YYYY-MM-DD.
pub fn day_of(ts: u64) -> String {
    // civil-from-days (Howard Hinnant's algorithm)
    let z = (ts / 86400) as i64 + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// "30d", "12h", "90m", "45s" or plain seconds, as seconds.
pub fn parse_duration(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let (num, mult) = match s.chars().last() {
        Some('d') => (&s[..s.len() - 1], 86400),
        Some('h') => (&s[..s.len() - 1], 3600),
        Some('m') => (&s[..s.len() - 1], 60),
        Some('s') => (&s[..s.len() - 1], 1),
        _ => (s, 1),
    };
    let n: u64 = num.trim().parse().map_err(|_| format!("{s:?} is not a time span (30d, 12h, 90m, 45s or seconds)"))?;
    if n == 0 {
        return Err("the time span must be more than zero".into());
    }
    n.checked_mul(mult).ok_or_else(|| format!("{s:?} is too long"))
}

// ------------------------------------------------------------------ keys

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Limits {
    /// model API requests this key may send in a UTC day
    #[serde(default)]
    pub requests_per_day: Option<u64>,
    /// prompt + completion tokens this key may use in a UTC day
    #[serde(default)]
    pub tokens_per_day: Option<u64>,
}

/// One generated key as kept in api-keys.json: its hash, never the key.
#[derive(Clone, Serialize, Deserialize)]
pub struct KeyRecord {
    pub id: String,
    pub name: String,
    /// the key's first 8 characters, to tell keys apart
    pub prefix: String,
    /// SHA-256 of the key, hex
    hash: String,
    pub created_at: u64,
    #[serde(default)]
    pub expires_at: Option<u64>,
    #[serde(default)]
    pub revoked: bool,
    #[serde(default)]
    pub revoked_at: Option<u64>,
    /// tool sets, as a client profile's: clients::SETS and mcp:<server> / mcp:*
    #[serde(default)]
    pub tools: BTreeSet<String>,
    /// its memory space (default: one of its own, from its name)
    #[serde(default)]
    pub memory: Option<String>,
    #[serde(default)]
    pub limits: Limits,
    /// the models it may ask for (None: every model)
    #[serde(default)]
    pub models: Option<Vec<String>>,
}

impl KeyRecord {
    pub fn expired(&self, now: u64) -> bool {
        self.expires_at.is_some_and(|e| now >= e)
    }

    pub fn active(&self, now: u64) -> bool {
        !self.revoked && !self.expired(now)
    }

    fn status(&self, now: u64) -> &'static str {
        if self.revoked { "revoked" } else if self.expired(now) { "expired" } else { "active" }
    }

    fn memory_space(&self) -> String {
        self.memory.clone().unwrap_or_else(|| {
            let s: String = self.name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' }).collect();
            format!("key-{s}")
        })
    }

    /// This key as a client profile: its tools and memory space.
    fn profile(&self) -> clients::Profile {
        clients::Profile::generated(self.name.clone(), self.tools.clone(), self.memory_space())
    }

    /// What the owner sees: everything but the hash.
    pub fn view(&self, now: u64) -> Value {
        json!({"id": self.id, "name": self.name, "prefix": self.prefix, "created_at": self.created_at,
               "expires_at": self.expires_at, "revoked": self.revoked, "revoked_at": self.revoked_at,
               "status": self.status(now), "tools": self.tools, "memory": self.memory_space(),
               "limits": self.limits, "models": self.models})
    }
}

/// What a given key is.
pub enum Lookup {
    Unknown,
    Active(KeyRecord),
    /// the key's name
    Revoked(String),
    Expired(String),
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hash_key(key: &str) -> String {
    hex(&Sha256::digest(key.as_bytes()))
}

/// Compared in constant time (no early exit on the first byte that differs).
fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

fn new_key() -> String {
    use rand::Rng;
    let mut rng = rand::rngs::OsRng;
    let mut k = String::with_capacity(KEY_PREFIX.len() + KEY_RANDOM_CHARS);
    k.push_str(KEY_PREFIX);
    for _ in 0..KEY_RANDOM_CHARS {
        k.push(ALPHABET[rng.gen_range(0..ALPHABET.len())] as char);
    }
    k
}

/// A file only this account reads: written beside it and moved into place.
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = path.with_extension("json.tmp");
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    let mut f = o.open(&tmp)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // a file left from before keeps its old mode otherwise
        f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    f.write_all(bytes)?;
    f.sync_all()?;
    std::fs::rename(&tmp, path)
}

fn check_name(name: &str) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() || name.chars().count() > 40 {
        return Err("a key's name has 1 to 40 characters".into());
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || " ._-".contains(c)) {
        return Err("a key's name has letters, digits, spaces, dots, dashes and underscores only".into());
    }
    if RESERVED.contains(&name.to_ascii_lowercase().as_str()) || name.starts_with("client:") {
        return Err(format!("{name:?} is a name the server uses itself; choose another"));
    }
    Ok(name.to_string())
}

/// A JSON number that is null (cleared), absent (unchanged) or a count.
fn opt_count(v: Option<&Value>, what: &str) -> Result<Option<Option<u64>>, String> {
    match v {
        None => Ok(None),
        Some(Value::Null) => Ok(Some(None)),
        Some(n) => match n.as_u64() {
            Some(0) | None => Err(format!("{what}: expected a whole number above zero, or null for no limit")),
            Some(n) => Ok(Some(Some(n))),
        },
    }
}

/// The fields of a create or update body put on `rec` (absent fields stay as they are).
fn apply(rec: &mut KeyRecord, body: &Value, now: u64) -> Result<(), String> {
    if !body.is_object() {
        return Err("expected a JSON object".into());
    }
    if let Some(n) = body.get("name") {
        rec.name = check_name(n.as_str().ok_or("name: expected text")?)?;
    }
    if let Some(t) = body.get("tools").filter(|t| !t.is_null()) {
        let names: Vec<String> = match t {
            Value::String(s) => s.split(',').map(str::to_owned).collect(),
            Value::Array(a) => a.iter().map(|x| x.as_str().map(str::to_owned).ok_or("tools: expected names")).collect::<Result<_, _>>()?,
            _ => return Err("tools: expected a list of names".into()),
        };
        rec.tools = names.iter().filter(|n| !n.trim().is_empty())
            .map(|n| clients::valid_set(n).ok_or_else(|| format!("tools: {:?} is not a tool set (one of {}, or mcp:<server>, mcp:*)", n.trim(), clients::SETS.join(", "))))
            .collect::<Result<_, _>>()?;
    }
    if let Some(m) = body.get("memory") {
        rec.memory = match m {
            Value::Null => None,
            Value::String(s) => Some(s.chars().filter(|c| c.is_ascii_alphanumeric() || "-_".contains(*c)).take(40).collect::<String>()).filter(|s| !s.is_empty()),
            _ => return Err("memory: expected the name of a memory space".into()),
        };
    }
    if let Some(e) = body.get("expires_in").filter(|e| !e.is_null()) {
        let secs = match e {
            Value::String(s) => parse_duration(s)?,
            other => other.as_u64().filter(|n| *n > 0).ok_or("expires_in: expected a time span (30d, 12h) or seconds")?,
        };
        rec.expires_at = Some(now.saturating_add(secs));
    } else if let Some(e) = body.get("expires_at") {
        rec.expires_at = match e {
            Value::Null => None,
            other => {
                let at = other.as_u64().ok_or("expires_at: expected a Unix time in seconds, or null for no expiry")?;
                if at <= now {
                    return Err("expires_at is in the past (revoke the key to stop it now)".into());
                }
                Some(at)
            }
        };
    }
    if let Some(l) = body.get("limits") {
        if l.is_null() {
            rec.limits = Limits::default();
        } else if l.is_object() {
            if let Some(v) = opt_count(l.get("requests_per_day"), "limits.requests_per_day")? {
                rec.limits.requests_per_day = v;
            }
            if let Some(v) = opt_count(l.get("tokens_per_day"), "limits.tokens_per_day")? {
                rec.limits.tokens_per_day = v;
            }
        } else {
            return Err("limits: expected {\"requests_per_day\": N, \"tokens_per_day\": N}".into());
        }
    }
    if let Some(m) = body.get("models") {
        let list: Vec<String> = match m {
            Value::Null => Vec::new(),
            Value::String(s) => s.split(',').map(|x| x.trim().to_string()).collect(),
            Value::Array(a) => a.iter().map(|x| x.as_str().map(|s| s.trim().to_string()).ok_or("models: expected model ids")).collect::<Result<_, _>>()?,
            _ => return Err("models: expected a list of model ids".into()),
        };
        let list: Vec<String> = list.into_iter().filter(|x| !x.is_empty()).collect();
        if list.len() > 64 || list.iter().any(|x| x.len() > 200) {
            return Err("models: at most 64 ids".into());
        }
        rec.models = (!list.is_empty()).then_some(list);
    }
    Ok(())
}

/// The generated keys and the usage counts.
pub struct ApiKeys {
    file: Option<PathBuf>,
    keys: RwLock<Vec<KeyRecord>>,
    pub usage: Usage,
}

impl std::fmt::Debug for ApiKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ApiKeys({} keys)", self.keys.read().map(|k| k.len()).unwrap_or(0))
    }
}

impl ApiKeys {
    /// The keys and counts kept in `data_dir` (None: in memory only).
    pub fn open(data_dir: Option<&Path>) -> Arc<ApiKeys> {
        let file = data_dir.map(|d| d.join("api-keys.json"));
        let mut keys = Vec::new();
        if let Some(text) = file.as_ref().and_then(|f| std::fs::read_to_string(f).ok()) {
            match serde_json::from_str::<Value>(&text).ok().and_then(|v| serde_json::from_value::<Vec<KeyRecord>>(v["keys"].clone()).ok()) {
                Some(k) => keys = k,
                None => tracing::error!("api-keys.json is not readable: no generated key works until it is repaired or removed"),
            }
        }
        Arc::new(ApiKeys { file, keys: RwLock::new(keys), usage: Usage::open(data_dir.map(|d| d.join("usage"))) })
    }

    fn save(&self, keys: &[KeyRecord]) -> Result<(), String> {
        let Some(f) = &self.file else { return Ok(()) };
        let body = serde_json::to_vec_pretty(&json!({"version": 1, "keys": keys})).map_err(|e| e.to_string())?;
        write_private(f, &body).map_err(|e| format!("the keys could not be saved ({}): {e}", f.display()))
    }

    /// A new key from a create body. Returns the key itself (told once, kept nowhere) and its record.
    /// `taken`: names the run config's client profiles have.
    pub fn create(&self, body: &Value, now: u64, taken: &dyn Fn(&str) -> bool) -> Result<(String, KeyRecord), String> {
        let key = new_key();
        let mut rec = KeyRecord { id: uuid::Uuid::new_v4().simple().to_string()[..12].to_string(), name: String::new(),
                                  prefix: key[..8].to_string(), hash: hash_key(&key), created_at: now, expires_at: None,
                                  revoked: false, revoked_at: None, tools: BTreeSet::new(), memory: None,
                                  limits: Limits::default(), models: None };
        if body.get("name").is_none() {
            return Err("a key needs a \"name\"".into());
        }
        apply(&mut rec, body, now)?;
        let mut g = self.keys.write().unwrap();
        if taken(&rec.name) || g.iter().any(|k| !k.revoked && k.name.eq_ignore_ascii_case(&rec.name)) {
            return Err(format!("the name {:?} is taken by another key or client profile", rec.name));
        }
        g.push(rec.clone());
        if let Err(e) = self.save(&g) {
            g.pop();
            return Err(e);
        }
        Ok((key, rec))
    }

    /// Rename, change tools, limits, expiry, models. Ok(None): no such key.
    pub fn update(&self, id: &str, body: &Value, now: u64, taken: &dyn Fn(&str) -> bool) -> Result<Option<KeyRecord>, String> {
        let mut g = self.keys.write().unwrap();
        let Some(i) = g.iter().position(|k| k.id == id) else { return Ok(None) };
        if g[i].revoked {
            return Err("this key is revoked; make a new one".into());
        }
        let before = g[i].clone();
        let mut rec = before.clone();
        apply(&mut rec, body, now)?;
        if rec.name != before.name && (taken(&rec.name) || g.iter().any(|k| k.id != id && !k.revoked && k.name.eq_ignore_ascii_case(&rec.name))) {
            return Err(format!("the name {:?} is taken by another key or client profile", rec.name));
        }
        g[i] = rec.clone();
        if let Err(e) = self.save(&g) {
            g[i] = before;
            return Err(e);
        }
        Ok(Some(rec))
    }

    /// Revoke: the key stops working at once (its record and usage stay). Ok(None): no such key.
    pub fn revoke(&self, id: &str, now: u64) -> Result<Option<KeyRecord>, String> {
        let mut g = self.keys.write().unwrap();
        let Some(i) = g.iter().position(|k| k.id == id) else { return Ok(None) };
        if !g[i].revoked {
            g[i].revoked = true;
            g[i].revoked_at = Some(now);
            if let Err(e) = self.save(&g) {
                g[i].revoked = false;
                g[i].revoked_at = None;
                return Err(e);
            }
        }
        Ok(Some(g[i].clone()))
    }

    pub fn list(&self) -> Vec<KeyRecord> {
        self.keys.read().unwrap().clone()
    }

    pub fn get(&self, id: &str) -> Option<KeyRecord> {
        self.keys.read().unwrap().iter().find(|k| k.id == id).cloned()
    }

    /// What `given` is. Its hash is compared with every stored one in constant time.
    pub fn lookup(&self, given: &str, now: u64) -> Lookup {
        if !given.starts_with(KEY_PREFIX) {
            return Lookup::Unknown;
        }
        let h = hash_key(given);
        let g = self.keys.read().unwrap();
        let mut found = None;
        for k in g.iter() {
            if same(&h, &k.hash) {
                found = Some(k);
            }
        }
        match found {
            None => Lookup::Unknown,
            Some(k) if k.revoked => Lookup::Revoked(k.name.clone()),
            Some(k) if k.expired(now) => Lookup::Expired(k.name.clone()),
            Some(k) => Lookup::Active(k.clone()),
        }
    }

    /// The profile of the working key `given` (clients.rs asks).
    pub fn profile_by_key(&self, given: &str) -> Option<clients::Profile> {
        match self.lookup(given, crate::now()) {
            Lookup::Active(k) => Some(k.profile()),
            _ => None,
        }
    }

    /// The profile of the working key called `name` (a task's client, for its limits).
    pub fn profile_by_name(&self, name: &str) -> Option<clients::Profile> {
        let now = crate::now();
        self.keys.read().unwrap().iter().find(|k| k.active(now) && k.name == name).map(KeyRecord::profile)
    }

    /// Write the usage counts every 20 s while they change (and once more at shutdown, from main).
    pub fn start_flush(self: &Arc<Self>) {
        let me = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(FLUSH_EVERY).await;
                me.usage.flush();
            }
        });
    }
}

// ------------------------------------------------------------------ usage

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Counter {
    #[serde(default)]
    pub requests: u64,
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    /// the part of prompt_tokens the engine reused from its cache
    #[serde(default)]
    pub cached_tokens: u64,
    #[serde(default)]
    pub media_jobs: u64,
    #[serde(default)]
    pub errors: u64,
    /// Unix time of the last request
    #[serde(default)]
    pub last_used: u64,
}

impl Counter {
    fn add(&mut self, o: &Counter) {
        self.requests += o.requests;
        self.prompt_tokens += o.prompt_tokens;
        self.completion_tokens += o.completion_tokens;
        self.cached_tokens += o.cached_tokens;
        self.media_jobs += o.media_jobs;
        self.errors += o.errors;
        self.last_used = self.last_used.max(o.last_used);
    }

    fn tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }

    fn view(&self) -> Value {
        let mut v = serde_json::to_value(self).unwrap_or_default();
        v["total_tokens"] = json!(self.tokens());
        v
    }
}

/// day -> who -> model -> counts
type Days = BTreeMap<String, BTreeMap<String, BTreeMap<String, Counter>>>;

#[derive(Default)]
struct UsageInner {
    days: Days,
    /// months (YYYY-MM) changed since the last write
    dirty: BTreeSet<String>,
    /// requests let in and not finished yet, per caller (they count toward the day's request limit)
    in_flight: HashMap<String, u64>,
}

#[derive(Default)]
pub struct Usage {
    dir: Option<PathBuf>,
    inner: Mutex<UsageInner>,
}

impl Usage {
    pub fn open(dir: Option<PathBuf>) -> Usage {
        let mut days = Days::new();
        for e in dir.as_ref().and_then(|d| std::fs::read_dir(d).ok()).into_iter().flatten().flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            // YYYY-MM.json
            if name.len() != 12 || !name.ends_with(".json") {
                continue;
            }
            let Some(v) = std::fs::read(e.path()).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) else {
                tracing::warn!("usage: {} is not readable; its counts are left out", e.path().display());
                continue;
            };
            if let Ok(d) = serde_json::from_value::<Days>(v["days"].clone()) {
                days.extend(d.into_iter().filter(|(day, _)| day.starts_with(&name[..7])));
            }
        }
        Usage { dir, inner: Mutex::new(UsageInner { days, ..Default::default() }) }
    }

    /// One request of `who` on `model`, finished: its counts added to today's.
    pub fn record(&self, who: &str, model: &str, c: Counter, now: u64) {
        let day = day_of(now);
        let mut g = self.inner.lock().unwrap();
        g.dirty.insert(day[..7].to_string());
        g.days.entry(day).or_default().entry(who.to_string()).or_default().entry(model.to_string()).or_default().add(&c);
    }

    /// What `who` used on the day of `now`, every model together.
    pub fn today(&self, who: &str, now: u64) -> Counter {
        let g = self.inner.lock().unwrap();
        let mut t = Counter::default();
        for c in g.days.get(&day_of(now)).and_then(|d| d.get(who)).into_iter().flat_map(|m| m.values()) {
            t.add(c);
        }
        t
    }

    /// Let one request of `who` in, unless today's limits are used up (Err: the 429's message). A request let in
    /// must be ended with `leave`.
    fn admit(&self, who: &str, name: &str, limits: &Limits, now: u64) -> Result<(), String> {
        // counted and let in under one lock: a request that ends meanwhile is never missed
        let mut g = self.inner.lock().unwrap();
        let mut used = Counter::default();
        for c in g.days.get(&day_of(now)).and_then(|d| d.get(who)).into_iter().flat_map(|m| m.values()) {
            used.add(c);
        }
        let running = g.in_flight.get(who).copied().unwrap_or(0);
        if let Some(max) = limits.requests_per_day {
            if used.requests + running >= max {
                return Err(format!("this key ({name}) has used its {max} requests for today (UTC); it works again tomorrow, or its owner raises the limit"));
            }
        }
        if let Some(max) = limits.tokens_per_day {
            if used.tokens() >= max {
                return Err(format!("this key ({name}) has used its {max} tokens for today (UTC); it works again tomorrow, or its owner raises the limit"));
            }
        }
        *g.in_flight.entry(who.to_string()).or_default() += 1;
        Ok(())
    }

    fn leave(&self, who: &str) {
        let mut g = self.inner.lock().unwrap();
        if let Some(n) = g.in_flight.get_mut(who) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                g.in_flight.remove(who);
            }
        }
    }

    /// The months that changed, written to <dir>/YYYY-MM.json.
    pub fn flush(&self) {
        let Some(dir) = &self.dir else { return };
        let out: Vec<(String, Vec<u8>)> = {
            let mut g = self.inner.lock().unwrap();
            let months = std::mem::take(&mut g.dirty);
            months.into_iter().map(|m| {
                let days: BTreeMap<&String, _> = g.days.iter().filter(|(d, _)| d.starts_with(&m)).collect();
                let body = serde_json::to_vec(&json!({"version": 1, "month": m, "days": days})).unwrap_or_default();
                (m, body)
            }).collect()
        };
        for (month, body) in out {
            if let Err(e) = write_private(&dir.join(format!("{month}.json")), &body) {
                tracing::warn!("usage: {month} could not be written: {e}");
                self.inner.lock().unwrap().dirty.insert(month);
            }
        }
    }

    /// The counts of the last `days` days (today included), of everyone or of `who` only:
    /// totals, per day (every day, also the empty ones), per model, per caller (with its models).
    pub fn report(&self, days: u64, who: Option<&str>, now: u64) -> Report {
        let days = days.clamp(1, 366);
        let names: Vec<String> = (0..days).rev().map(|i| day_of(now.saturating_sub(i * 86400))).collect();
        let g = self.inner.lock().unwrap();
        let mut r = Report { from: names[0].clone(), to: names[names.len() - 1].clone(), ..Default::default() };
        for day in &names {
            let mut of_day = Counter::default();
            for (w, models) in g.days.get(day).into_iter().flatten().filter(|(w, _)| who.is_none_or(|x| x == w.as_str())) {
                for (model, c) in models {
                    of_day.add(c);
                    r.by_model.entry(model.clone()).or_default().add(c);
                    let k = r.by_who.entry(w.clone()).or_default();
                    k.0.add(c);
                    k.1.entry(model.clone()).or_default().add(c);
                }
            }
            r.total.add(&of_day);
            r.by_day.push((day.clone(), of_day));
        }
        r
    }
}

#[derive(Default)]
pub struct Report {
    pub from: String,
    pub to: String,
    pub total: Counter,
    pub by_day: Vec<(String, Counter)>,
    pub by_model: BTreeMap<String, Counter>,
    /// caller -> its total and its models
    pub by_who: BTreeMap<String, (Counter, BTreeMap<String, Counter>)>,
}

fn models_view(m: &BTreeMap<String, Counter>) -> Vec<Value> {
    let mut rows: Vec<(&String, &Counter)> = m.iter().collect();
    rows.sort_by(|a, b| b.1.tokens().cmp(&a.1.tokens()).then(b.1.requests.cmp(&a.1.requests)));
    rows.into_iter().map(|(model, c)| {
        let mut v = c.view();
        v["model"] = json!(model);
        v
    }).collect()
}

/// A caller of the usage counts as the owner reads it: its kind and name.
fn who_view(keys: &ApiKeys, who: &str) -> (String, String) {
    if who == OWNER {
        ("owner".into(), "the main key".into())
    } else if who == CLUSTER {
        ("cluster".into(), "cluster master".into())
    } else if let Some(n) = who.strip_prefix("client:") {
        ("client".into(), n.to_string())
    } else {
        ("key".into(), keys.get(who).map(|k| k.name).unwrap_or_else(|| format!("removed key {who}")))
    }
}

fn report_view(keys: &ApiKeys, days: u64, who: Option<&str>, now: u64) -> Value {
    let r = keys.usage.report(days, who, now);
    let by_day: Vec<Value> = r.by_day.iter().map(|(d, c)| {
        let mut v = c.view();
        v["day"] = json!(d);
        v
    }).collect();
    let mut by_key: Vec<Value> = r.by_who.iter().map(|(w, (c, models))| {
        let (kind, name) = who_view(keys, w);
        let mut v = c.view();
        v["key"] = json!(w);
        v["kind"] = json!(kind);
        v["name"] = json!(name);
        v["models"] = json!(models_view(models));
        v["today"] = keys.usage.today(w, now).view();
        v
    }).collect();
    by_key.sort_by_key(|v| std::cmp::Reverse((v["total_tokens"].as_u64().unwrap_or(0), v["requests"].as_u64().unwrap_or(0))));
    json!({"days": r.by_day.len(), "from": r.from, "to": r.to, "timezone": "UTC", "totals": r.total.view(),
           "by_day": by_day, "by_model": models_view(&r.by_model), "by_key": by_key})
}

// ------------------------------------------------------------------ the layer

/// Who sends a request, for the counts.
enum Caller {
    /// no key this server knows: the handlers answer (401, or a route that needs none)
    Nobody,
    /// a generated key that no longer works: the 401's message
    Refused(String),
    /// the counts' name, and the generated key's record if it is one
    Is(String, Option<Box<KeyRecord>>),
}

fn caller(s: &AppState, headers: &HeaderMap, now: u64) -> Caller {
    let given = clients::given_key(headers);
    if let Some(k) = given {
        match s.keys.lookup(k, now) {
            Lookup::Active(rec) => return Caller::Is(rec.id.clone(), Some(Box::new(rec))),
            Lookup::Revoked(name) => return Caller::Refused(format!("this API key ({name}) was revoked; ask the server's owner for a new one")),
            Lookup::Expired(name) => return Caller::Refused(format!("this API key ({name}) has expired; ask the server's owner for a new one")),
            Lookup::Unknown => {}
        }
    }
    if authorized(s, headers) {
        return Caller::Is(OWNER.into(), None);
    }
    let Some(k) = given else { return Caller::Nobody };
    if let Some(p) = s.clients.config_by_key(k) {
        return Caller::Is(format!("client:{}", p.name), None);
    }
    if s.cluster.is_master_key(k, crate::cluster_tls::peer_of(headers)) {
        return Caller::Is(CLUSTER.into(), None);
    }
    Caller::Nobody
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// chat, messages, responses, Gemini: the answer says its model and token counts
    Text,
    /// images, videos, audio, productions: one media job
    Media,
}

/// The model API calls that are counted (and limited).
fn metered(method: &Method, path: &str) -> Option<Kind> {
    if method != Method::POST {
        return None;
    }
    if matches!(path, "/v1/chat/completions" | "/v1/messages" | "/v1/responses") {
        return Some(Kind::Text);
    }
    if let Some(target) = path.strip_prefix("/v1beta/models/") {
        return (target.ends_with(":generateContent") || target.ends_with(":streamGenerateContent")).then_some(Kind::Text);
    }
    ["/v1/images/", "/v1/videos/", "/v1/audio/", "/v1/media/"].iter().any(|p| path.starts_with(p)).then_some(Kind::Media)
}

/// The model a call uses when its body names none.
async fn default_model(s: &AppState, path: &str) -> String {
    if path.starts_with("/v1/audio/speech") {
        s.media.tts_model()
    } else if path.starts_with("/v1/audio") {
        s.media.stt_model()
    } else if path.starts_with("/v1/images") {
        s.media.model()
    } else if path.starts_with("/v1/videos") {
        s.media.video_model()
    } else if path.starts_with("/v1/media") {
        "media".into()
    } else {
        s.mgr.current().await.map(|l| l.entry.id.clone()).unwrap_or_else(|| s.mgr.default_id.clone())
    }
}

/// What the answer of one call says so far: its model, its usage (fields of any of the APIs' shapes, later ones
/// over earlier ones: an Anthropic stream says the prompt at the start and the rest at the end), an error.
#[derive(Default)]
struct Seen {
    model: Option<String>,
    usage: Map<String, Value>,
    error: bool,
}

impl Seen {
    fn take(&mut self, v: &Value) {
        let model = [v.get("model"), v.pointer("/message/model"), v.pointer("/response/model"), v.get("modelVersion")]
            .into_iter().flatten().find_map(|m| m.as_str().filter(|m| !m.is_empty()));
        if let Some(m) = model {
            self.model = Some(m.to_string());
        }
        for u in [v.get("usage"), v.pointer("/message/usage"), v.pointer("/response/usage"), v.get("usageMetadata")].into_iter().flatten() {
            if let Some(o) = u.as_object() {
                self.usage.extend(o.clone());
            }
        }
        let failed = |e: Option<&Value>| e.is_some_and(|e| !e.is_null());
        if failed(v.get("error")) || failed(v.pointer("/response/error")) || v.get("type").and_then(Value::as_str) == Some("error") {
            self.error = true;
        }
    }

    /// (prompt, completion, cached) tokens, whichever API's names the usage has.
    fn tokens(&self) -> (u64, u64, u64) {
        let n = |k: &str| self.usage.get(k).and_then(Value::as_u64);
        let inner = |k: &str| self.usage.get(k).and_then(|d| d.get("cached_tokens")).and_then(Value::as_u64).unwrap_or(0);
        if let Some(p) = n("promptTokenCount") {
            // Gemini
            (p, n("candidatesTokenCount").unwrap_or(0), n("cachedContentTokenCount").unwrap_or(0))
        } else if let Some(p) = n("prompt_tokens") {
            // OpenAI chat
            (p, n("completion_tokens").unwrap_or(0), inner("prompt_tokens_details"))
        } else if n("cache_read_input_tokens").is_some() || n("cache_creation_input_tokens").is_some() {
            // Anthropic: input_tokens is the part that was not cached
            let read = n("cache_read_input_tokens").unwrap_or(0);
            (n("input_tokens").unwrap_or(0) + read + n("cache_creation_input_tokens").unwrap_or(0), n("output_tokens").unwrap_or(0), read)
        } else {
            // Responses (and an Anthropic answer without cache fields)
            (n("input_tokens").unwrap_or(0), n("output_tokens").unwrap_or(0), inner("input_tokens_details"))
        }
    }
}

/// One counted call: written to the counts when it ends (the answer was sent, or the client left).
struct Meter {
    keys: Arc<ApiKeys>,
    who: String,
    kind: Kind,
    /// the model the request named, or the default one
    model: String,
    failed: bool,
    seen: Seen,
    /// the unfinished line of a stream
    buf: String,
}

impl Meter {
    /// The lines of a stream's chunk: each `data:` event is read for the model, usage and errors.
    fn feed(&mut self, chunk: &[u8]) {
        self.buf.push_str(&String::from_utf8_lossy(chunk));
        while let Some(nl) = self.buf.find('\n') {
            let line: String = self.buf.drain(..=nl).collect();
            if let Some(data) = line.trim_end().strip_prefix("data:") {
                // only the events that can carry what is counted are parsed
                if data.contains("usage") || data.contains("\"error\"") || self.seen.model.is_none() {
                    if let Ok(v) = serde_json::from_str::<Value>(data.trim()) {
                        self.seen.take(&v);
                    }
                }
            }
        }
    }
}

impl Drop for Meter {
    fn drop(&mut self) {
        let now = crate::now();
        let (prompt_tokens, completion_tokens, cached_tokens) = self.seen.tokens();
        let failed = self.failed || self.seen.error;
        let c = Counter { requests: 1, prompt_tokens, completion_tokens, cached_tokens,
                          media_jobs: u64::from(self.kind == Kind::Media && !failed), errors: u64::from(failed), last_used: now };
        let model = self.seen.model.take().unwrap_or_else(|| std::mem::take(&mut self.model));
        self.keys.usage.record(&self.who, &model, c, now);
        self.keys.usage.leave(&self.who);
    }
}

/// The model a request names: Gemini's in the path, the others in the JSON body or the upload's form.
fn asked_model(path: &str, headers: &HeaderMap, body: &[u8]) -> Option<String> {
    if let Some(target) = path.strip_prefix("/v1beta/models/") {
        return target.split(':').next().filter(|m| !m.is_empty()).map(str::to_string);
    }
    let multipart = headers.get(axum::http::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).is_some_and(|c| c.starts_with("multipart/"));
    let m = if multipart {
        form_field(body, "model")
    } else {
        serde_json::from_slice::<Value>(body).ok().and_then(|v| v.get("model").and_then(Value::as_str).map(str::to_string))
    };
    m.filter(|m| !m.is_empty())
}

/// A text field of a multipart/form-data body, by its name.
fn form_field(body: &[u8], name: &str) -> Option<String> {
    let key = format!("name=\"{name}\"");
    let at = body.windows(key.len()).position(|w| w == key.as_bytes())?;
    let rest = &body[at + key.len()..];
    let start = rest.windows(4).position(|w| w == b"\r\n\r\n")? + 4;
    let end = rest[start..].windows(2).position(|w| w == b"\r\n")?;
    String::from_utf8(rest[start..start + end].to_vec()).ok().map(|v| v.trim().to_string())
}

/// The layer around every route: a revoked or expired key is refused anywhere; the model APIs are limited and counted.
pub async fn layer(State(s): State<AppState>, req: Request, next: Next) -> Response {
    let now = crate::now();
    let (who, rec) = match caller(&s, req.headers(), now) {
        Caller::Refused(msg) => return error(StatusCode::UNAUTHORIZED, "authentication_error", msg),
        Caller::Nobody => return next.run(req).await,
        Caller::Is(who, rec) => (who, rec),
    };
    let path = req.uri().path().to_string();
    let Some(kind) = metered(req.method(), &path) else { return next.run(req).await };
    let keys = s.keys.clone();
    let mut model = None;
    let mut req = req;
    if let Some(rec) = &rec {
        // the models this key may ask for (the body is read here only for a key that has such a list)
        if let Some(allowed) = &rec.models {
            let (parts, body) = req.into_parts();
            let Ok(bytes) = axum::body::to_bytes(body, 256 << 20).await else {
                return error(StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error", "the request body is too large");
            };
            let m = match asked_model(&path, &parts.headers, &bytes) {
                Some(m) => m,
                None => default_model(&s, &path).await,
            };
            if !allowed.iter().any(|a| a.eq_ignore_ascii_case(&m)) {
                return error(StatusCode::FORBIDDEN, "permission_error",
                             format!("this key ({}) may not use the model {m:?}; it may use: {}", rec.name, allowed.join(", ")));
            }
            model = Some(m);
            req = Request::from_parts(parts, Body::from(bytes));
        }
        if let Err(msg) = keys.usage.admit(&who, &rec.name, &rec.limits, now) {
            let mut r = error(StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", msg);
            // until the next UTC day
            if let Ok(v) = axum::http::HeaderValue::from_str(&(86400 - now % 86400).to_string()) {
                r.headers_mut().insert("retry-after", v);
            }
            return r;
        }
    } else if let Err(msg) = keys.usage.admit(&who, &who, &Limits::default(), now) {
        // never: no limits
        return error(StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", msg);
    }
    let model = match model {
        Some(m) => m,
        None => default_model(&s, &path).await,
    };
    let mut meter = Meter { keys, who, kind, model, failed: false, seen: Seen::default(), buf: String::new() };
    let resp = next.run(req).await;
    meter.failed = resp.status().is_client_error() || resp.status().is_server_error();
    if kind == Kind::Media {
        // a picture or a sound: nothing to read in it
        return resp;
    }
    let sse = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|c| c.starts_with("text/event-stream"));
    let (parts, body) = resp.into_parts();
    if !sse {
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap_or_default();
        if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
            meter.seen.take(&v);
        }
        return Response::from_parts(parts, Body::from(bytes));
    }
    // a stream: read as it goes out; the meter is dropped with the body (the stream ended or the client left)
    let stream = body.into_data_stream().map(move |chunk| {
        if let Ok(b) = &chunk {
            meter.feed(b);
        }
        chunk.map(|b: Bytes| b)
    });
    Response::from_parts(parts, Body::from_stream(stream))
}

// ------------------------------------------------------------------ the owner's API

/// Only the server's own key manages keys and reads everyone's usage.
fn owner(s: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    let other_key = clients::given_key(headers).is_some_and(|k| s.clients.by_key(k).is_some());
    if !authorized(s, headers) || other_key {
        return Err(error(StatusCode::UNAUTHORIZED, "authentication_error", "API keys are managed with the server's own API key"));
    }
    Ok(())
}

fn key_with_usage(keys: &ApiKeys, k: &KeyRecord, now: u64) -> Value {
    let mut v = k.view(now);
    let r = keys.usage.report(30, Some(&k.id), now);
    let last_used = r.total.last_used;
    v["usage"] = json!({"today": keys.usage.today(&k.id, now).view(), "days_30": r.total.view(),
                        "models": r.by_model.keys().collect::<Vec<_>>()});
    v["last_used"] = if last_used > 0 { json!(last_used) } else { Value::Null };
    v
}

/// GET /api/keys: the keys (no secrets), each with today's and 30 days' use.
pub async fn list(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(r) = owner(&s, &headers) {
        return r;
    }
    let now = crate::now();
    let mut keys = s.keys.list();
    keys.sort_by_key(|k| (k.revoked, std::cmp::Reverse(k.created_at)));
    Json(json!({"keys": keys.iter().map(|k| key_with_usage(&s.keys, k, now)).collect::<Vec<_>>(),
                "tool_sets": clients::SETS, "main_key_set": s.api_key.as_deref().is_some_and(|k| !k.is_empty())})).into_response()
}

/// POST /api/keys: a new key. The answer holds the key itself, this once.
pub async fn create(State(s): State<AppState>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    if let Err(r) = owner(&s, &headers) {
        return r;
    }
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("API keys can be made");
    }
    let now = crate::now();
    match s.keys.create(&body, now, &|n| s.clients.has_name(n)) {
        Ok((key, rec)) => {
            tracing::info!("API key made: {} ({})", rec.name, rec.id);
            let mut v = rec.view(now);
            v["key"] = json!(key);
            (StatusCode::CREATED, Json(v)).into_response()
        }
        Err(e) => error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    }
}

/// POST /api/keys/{id}: rename, change tools, memory, limits, expiry, models.
pub async fn update(State(s): State<AppState>, headers: HeaderMap, UrlPath(id): UrlPath<String>, Json(body): Json<Value>) -> Response {
    if let Err(r) = owner(&s, &headers) {
        return r;
    }
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("API keys can be changed");
    }
    let now = crate::now();
    match s.keys.update(&id, &body, now, &|n| s.clients.has_name(n)) {
        Ok(Some(rec)) => Json(key_with_usage(&s.keys, &rec, now)).into_response(),
        Ok(None) => error(StatusCode::NOT_FOUND, "not_found", "no such key"),
        Err(e) => error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    }
}

/// DELETE /api/keys/{id}: revoke. The key stops working at once; its usage stays.
pub async fn revoke(State(s): State<AppState>, headers: HeaderMap, UrlPath(id): UrlPath<String>) -> Response {
    if let Err(r) = owner(&s, &headers) {
        return r;
    }
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("API keys can be revoked");
    }
    let now = crate::now();
    match s.keys.revoke(&id, now) {
        Ok(Some(rec)) => {
            tracing::info!("API key revoked: {} ({})", rec.name, rec.id);
            Json(rec.view(now)).into_response()
        }
        Ok(None) => error(StatusCode::NOT_FOUND, "not_found", "no such key"),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e),
    }
}

fn days_of(q: &HashMap<String, String>) -> u64 {
    q.get("days").and_then(|d| d.parse::<u64>().ok()).unwrap_or(30).clamp(1, 366)
}

/// GET /api/usage?days=30&key=<id>: totals, per day, per model, per key (`key`: a key's id, "owner", or
/// "client:<name>").
pub async fn usage(State(s): State<AppState>, headers: HeaderMap, Query(q): Query<HashMap<String, String>>) -> Response {
    if let Err(r) = owner(&s, &headers) {
        return r;
    }
    let who = q.get("key").map(String::as_str).filter(|k| !k.is_empty());
    Json(report_view(&s.keys, days_of(&q), who, crate::now())).into_response()
}

/// GET /v1/usage?days=30: the usage of the key that asks, and its limits.
pub async fn own_usage(State(s): State<AppState>, headers: HeaderMap, Query(q): Query<HashMap<String, String>>) -> Response {
    let now = crate::now();
    let (who, rec) = match caller(&s, &headers, now) {
        Caller::Is(who, rec) => (who, rec),
        // a server without a key: everyone is the owner
        Caller::Nobody if authorized(&s, &headers) => (OWNER.to_string(), None),
        Caller::Nobody => return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required"),
        Caller::Refused(msg) => return error(StatusCode::UNAUTHORIZED, "authentication_error", msg),
    };
    let mut v = report_view(&s.keys, days_of(&q), Some(&who), now);
    let (kind, name) = who_view(&s.keys, &who);
    v["key"] = json!({"id": who, "kind": kind, "name": name});
    v["today"] = s.keys.usage.today(&who, now).view();
    if let Some(rec) = rec {
        v["key"]["prefix"] = json!(rec.prefix);
        v["key"]["expires_at"] = json!(rec.expires_at);
        v["key"]["tools"] = json!(rec.tools);
        v["key"]["models"] = json!(rec.models);
        v["limits"] = json!(rec.limits);
    }
    Json(v).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("am2-api-keys-{name}-{}-{}", std::process::id(), uuid::Uuid::new_v4().simple()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    const FREE: &dyn Fn(&str) -> bool = &|_| false;

    #[test]
    fn days_and_time_spans() {
        assert_eq!(day_of(0), "1970-01-01");
        assert_eq!(day_of(951_782_400), "2000-02-29");
        assert_eq!(day_of(1_794_614_399), "2026-11-13");
        assert_eq!(day_of(1_794_614_400), "2026-11-14");
        assert_eq!(day_of(1_735_689_599), "2024-12-31");
        assert_eq!(parse_duration("30d"), Ok(30 * 86400));
        assert_eq!(parse_duration("12h"), Ok(43200));
        assert_eq!(parse_duration("90m"), Ok(5400));
        assert_eq!(parse_duration("45"), Ok(45));
        assert!(parse_duration("soon").is_err() && parse_duration("0d").is_err() && parse_duration("").is_err());
    }

    #[test]
    fn a_key_is_told_once_and_only_its_hash_is_kept() {
        let dir = temp("store");
        let keys = ApiKeys::open(Some(&dir));
        let (key, rec) = keys.create(&json!({"name": "laptop", "tools": ["web", "Memory", "mcp:files"], "expires_in": "30d",
                                             "limits": {"requests_per_day": 100}, "models": ["a", "b"]}), 1000, FREE).unwrap();
        assert!(key.starts_with(KEY_PREFIX) && key.len() == 44, "am2_ and 40 characters");
        assert!(key[4..].bytes().all(|b| ALPHABET.contains(&b)), "URL-safe characters only");
        assert_eq!(rec.prefix, &key[..8]);
        assert_eq!(rec.expires_at, Some(1000 + 30 * 86400));
        assert_eq!(rec.tools, ["memory", "mcp:files", "web"].iter().map(|s| s.to_string()).collect());
        let text = std::fs::read_to_string(dir.join("api-keys.json")).unwrap();
        assert!(!text.contains(&key) && !text.contains(&key[8..]), "the file holds no key");
        assert!(text.contains(&hash_key(&key)));
        assert!(!rec.view(1000).to_string().contains("hash"), "the view has no hash");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(dir.join("api-keys.json")).unwrap().permissions().mode() & 0o777, 0o600);
        }
        // read again at the next start; another key of another name is another key
        let again = ApiKeys::open(Some(&dir));
        assert!(matches!(again.lookup(&key, 2000), Lookup::Active(k) if k.id == rec.id));
        assert!(matches!(again.lookup("am2_0000000000000000000000000000000000000000", 2000), Lookup::Unknown));
        assert!(matches!(again.lookup("some-other-key", 2000), Lookup::Unknown));
        let (key2, _) = again.create(&json!({"name": "phone"}), 2000, FREE).unwrap();
        assert_ne!(key, key2);
        // expiry, then revocation
        assert!(matches!(again.lookup(&key, 1000 + 30 * 86400), Lookup::Expired(n) if n == "laptop"));
        again.revoke(&rec.id, 3000).unwrap().unwrap();
        assert!(matches!(again.lookup(&key, 3000), Lookup::Revoked(_)));
        assert!(again.profile_by_key(&key).is_none());
        assert!(matches!(ApiKeys::open(Some(&dir)).lookup(&key, 3000), Lookup::Revoked(_)), "revocation is kept");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn names_tools_and_limits_are_checked() {
        let keys = ApiKeys::open(None);
        assert!(keys.create(&json!({}), 1, FREE).is_err(), "no name");
        for bad in ["", "owner", "Default", "client:x", "a/b", &"x".repeat(41)] {
            assert!(keys.create(&json!({"name": bad}), 1, FREE).is_err(), "{bad:?}");
        }
        assert!(keys.create(&json!({"name": "a", "tools": ["shell"]}), 1, FREE).is_err());
        assert!(keys.create(&json!({"name": "a", "limits": {"requests_per_day": 0}}), 1, FREE).is_err());
        assert!(keys.create(&json!({"name": "a", "expires_at": 1}), 5, FREE).is_err(), "born expired");
        assert!(keys.create(&json!({"name": "laptop"}), 1, &|n| n == "laptop").is_err(), "a run-config profile's name");
        let (key, a) = keys.create(&json!({"name": "a", "tools": "web, image"}), 1, FREE).unwrap();
        assert!(keys.create(&json!({"name": "A"}), 1, FREE).is_err(), "taken");
        // as a client profile: its tools, a memory space of its own
        let p = keys.profile_by_key(&key).unwrap();
        assert!(p.has("web") && p.has("image") && !p.has("memory"));
        assert_eq!((p.name.as_str(), p.memory_space().as_str()), ("a", "key-a"));
        // changed while it runs; absent fields stay
        let b = keys.update(&a.id, &json!({"name": "b", "limits": {"tokens_per_day": 5}, "models": ["m"]}), 2, FREE).unwrap().unwrap();
        assert_eq!((b.name.as_str(), b.limits.tokens_per_day, b.tools.len()), ("b", Some(5), 2));
        let c = keys.update(&a.id, &json!({"limits": {"tokens_per_day": null}, "models": null, "tools": []}), 3, FREE).unwrap().unwrap();
        assert_eq!((c.limits.tokens_per_day, c.models, c.tools.len()), (None, None, 0));
        assert!(keys.profile_by_name("b").is_some() && keys.profile_by_name("a").is_none());
        assert!(keys.update("nope", &json!({}), 3, FREE).unwrap().is_none());
        // a revoked key's name is free again; the revoked key cannot be changed back
        keys.revoke(&a.id, 4).unwrap();
        assert!(keys.update(&a.id, &json!({"name": "z"}), 5, FREE).is_err());
        assert!(keys.create(&json!({"name": "b"}), 5, FREE).is_ok());
    }

    #[test]
    fn usage_is_counted_limited_and_kept() {
        let dir = temp("usage");
        let u = Usage::open(Some(dir.clone()));
        let day1 = 1_794_614_400; // 2026-11-14
        let c = |p, o| Counter { requests: 1, prompt_tokens: p, completion_tokens: o, cached_tokens: 1, last_used: day1, ..Default::default() };
        u.record("k1", "model-a", c(10, 5), day1);
        u.record("k1", "model-b", c(1, 1), day1 + 60);
        u.record("owner", "model-a", c(100, 50), day1 - 86400);
        assert_eq!(u.today("k1", day1 + 100).requests, 2);
        assert_eq!(u.today("k1", day1 + 100).tokens(), 17);
        assert_eq!(u.today("k1", day1 + 86400).requests, 0, "a new day starts at zero");
        // limits: requests (those still running count), then tokens
        let lim = Limits { requests_per_day: Some(3), tokens_per_day: None };
        assert!(u.admit("k1", "one", &lim, day1).is_ok());
        assert!(u.admit("k1", "one", &lim, day1).is_err(), "2 done + 1 running = 3");
        u.leave("k1");
        assert!(u.admit("k1", "one", &Limits { requests_per_day: None, tokens_per_day: Some(17) }, day1).is_err());
        assert!(u.admit("k1", "one", &Limits { requests_per_day: None, tokens_per_day: Some(18) }, day1).is_ok());
        u.leave("k1");
        // the report: every day of the range, per model, per caller
        let r = u.report(7, None, day1);
        assert_eq!((r.from.as_str(), r.to.as_str(), r.by_day.len()), ("2026-11-08", "2026-11-14", 7));
        assert_eq!(r.total.requests, 3);
        assert_eq!(r.by_model["model-a"].prompt_tokens, 110);
        assert_eq!(r.by_who["k1"].1.len(), 2);
        assert_eq!(u.report(7, Some("k1"), day1).total.tokens(), 17);
        assert_eq!(u.report(1, None, day1).total.requests, 2, "today only");
        // written, and read at the next start
        u.flush();
        assert!(dir.join("2026-11.json").exists());
        let again = Usage::open(Some(dir.clone()));
        assert_eq!(again.report(7, None, day1).total, r.total);
        assert_eq!(again.today("k1", day1).cached_tokens, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_apis_usage_is_read() {
        let read = |events: &[Value]| {
            let mut s = Seen::default();
            for e in events {
                s.take(e);
            }
            (s.model.clone(), s.tokens(), s.error)
        };
        // OpenAI chat: an answer, and a stream's last chunk
        assert_eq!(read(&[json!({"model": "m", "usage": {"prompt_tokens": 10, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 6}}})]),
                   (Some("m".into()), (10, 4, 6), false));
        // Anthropic stream: the prompt at the start, the rest (and what was cached) at the end
        assert_eq!(read(&[json!({"type": "message_start", "message": {"model": "m", "usage": {"input_tokens": 10, "output_tokens": 0}}}),
                          json!({"type": "message_delta", "usage": {"input_tokens": 4, "cache_read_input_tokens": 6, "output_tokens": 3}})]),
                   (Some("m".into()), (10, 3, 6), false));
        // Responses
        assert_eq!(read(&[json!({"type": "response.completed", "response": {"model": "m", "usage": {"input_tokens": 9, "output_tokens": 2,
                          "input_tokens_details": {"cached_tokens": 1}}}})]), (Some("m".into()), (9, 2, 1), false));
        // Gemini
        assert_eq!(read(&[json!({"modelVersion": "m", "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2, "cachedContentTokenCount": 3}})]),
                   (Some("m".into()), (7, 2, 3), false));
        // an error in a stream that began well
        assert!(read(&[json!({"model": "m"}), json!({"error": {"message": "the engine stopped"}})]).2);
    }

    #[test]
    fn only_model_api_posts_are_metered() {
        assert!(metered(&Method::POST, "/v1/chat/completions") == Some(Kind::Text));
        assert!(metered(&Method::POST, "/v1beta/models/m:streamGenerateContent") == Some(Kind::Text));
        assert!(metered(&Method::POST, "/v1beta/models/m:countTokens").is_none());
        assert!(metered(&Method::POST, "/v1/images/generations") == Some(Kind::Media));
        assert!(metered(&Method::POST, "/v1/messages/count_tokens").is_none());
        assert!(metered(&Method::GET, "/v1/models").is_none() && metered(&Method::POST, "/settings").is_none());
        assert_eq!(asked_model("/v1beta/models/gem:generateContent", &HeaderMap::new(), b""), Some("gem".into()));
        assert_eq!(asked_model("/v1/chat/completions", &HeaderMap::new(), br#"{"model":"x"}"#), Some("x".into()));
        assert_eq!(asked_model("/v1/chat/completions", &HeaderMap::new(), br#"{"messages":[]}"#), None);
    }
}
