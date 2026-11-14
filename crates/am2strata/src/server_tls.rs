//! HTTPS for the API and the web app: TLS 1.2/1.3 (rustls), HSTS, and `wss://` for anything that streams.
//!
//! Off unless the run config has a `"tls"` block:
//!
//! ```json
//! "tls": {"mode": "self-signed" | "tailscale" | "files", "https_port": 8001,
//!         "cert": "/path/cert.pem", "key": "/path/key.pem",          // "files" only
//!         "http": "local-only" | "keep" | "redirect",                 // what the plain-HTTP port does
//!         "hsts_max_age": 31536000, "hsts_include_subdomains": false}
//! ```
//!
//! - **self-signed**: a certificate made once with rcgen for every name this PC answers to (localhost, 127.0.0.1, ::1,
//!   its hostname and `.local`, its LAN and Tailscale addresses, its Tailscale name), kept in
//!   `<data>/server-tls/{cert.pem,key.pem}` (the key 0600) and made again only when that list of names changes. Its
//!   SHA-256 fingerprint is logged, so a browser's warning can be checked against it.
//! - **tailscale**: `tailscale cert` for this PC's tailnet name (a certificate browsers trust, for `https://<name>.ts.net`),
//!   written to the same folder and renewed every day. If tailscale is missing, logged out, or HTTPS certificates are not
//!   enabled for the tailnet, the reason is logged and HTTPS does not start; it never falls back silently.
//! - **files**: any PEM certificate and key (another CA, or `tailscale cert` run by hand); reloaded when they change.
//!
//! The plain-HTTP port (the API port) keeps working for this PC's own tools: with `"http": "local-only"` (the default)
//! a request from 127.0.0.1/::1 is answered as before and any other is redirected (308) to HTTPS; `"redirect"` sends
//! every request there except `/health`; `"keep"` answers everything over plain HTTP too.
//!
//! **HSTS** (`Strict-Transport-Security`) is sent on HTTPS responses only, never on plain HTTP. Browsers ignore it for
//! bare IP addresses and for certificates they do not trust, so it takes effect on a trusted name (the Tailscale name).

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use rustls::server::{ClientHello, ResolvesServerCert};
use rustls::sign::CertifiedKey;
use serde_json::Value;

/// What the plain-HTTP port does once HTTPS is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMode {
    /// this PC's own requests answered as before; every other request redirected to HTTPS
    LocalOnly,
    /// every request redirected to HTTPS (except /health)
    Redirect,
    /// plain HTTP answers everything, as without TLS
    Keep,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    SelfSigned,
    Tailscale,
    Files { cert: PathBuf, key: PathBuf },
}

#[derive(Debug, Clone)]
pub struct Config {
    pub mode: Mode,
    pub https_port: u16,
    pub http: HttpMode,
    pub hsts_max_age: u64,
    pub hsts_include_subdomains: bool,
}

/// HTTPS and wss next to the API's default 8000 (and clear of the cluster's TLS port, the API port + 443).
pub const DEFAULT_HTTPS_PORT: u16 = 8001;

impl Config {
    /// The `"tls"` block of the run config, or None when HTTPS is off. A block that is wrong is an error, so a typo
    /// never leaves the server on plain HTTP by surprise.
    pub fn from_config(rest: &std::collections::BTreeMap<String, Value>) -> Result<Option<Config>, String> {
        let Some(t) = rest.get("tls") else { return Ok(None) };
        if t == &Value::Bool(false) || t.is_null() {
            return Ok(None);
        }
        let mode = match t["mode"].as_str().unwrap_or("self-signed") {
            "self-signed" | "selfsigned" => Mode::SelfSigned,
            "tailscale" => Mode::Tailscale,
            "files" => Mode::Files {
                cert: t["cert"].as_str().map(expand).ok_or("\"tls\": mode \"files\" needs \"cert\" (a PEM file)")?,
                key: t["key"].as_str().map(expand).ok_or("\"tls\": mode \"files\" needs \"key\" (a PEM file)")?,
            },
            other => return Err(format!("\"tls\": unknown mode {other:?} (self-signed, tailscale or files)")),
        };
        let http = match t["http"].as_str().unwrap_or("local-only") {
            "local-only" | "local" => HttpMode::LocalOnly,
            "redirect" => HttpMode::Redirect,
            "keep" => HttpMode::Keep,
            other => return Err(format!("\"tls\": unknown \"http\" {other:?} (local-only, redirect or keep)")),
        };
        let https_port = match t.get("https_port") {
            None => DEFAULT_HTTPS_PORT,
            Some(v) => v.as_u64().filter(|p| (1..=65535).contains(p)).ok_or("\"tls\": \"https_port\" must be a port number")? as u16,
        };
        Ok(Some(Config {
            mode, https_port, http,
            hsts_max_age: t["hsts_max_age"].as_u64().unwrap_or(31_536_000),
            hsts_include_subdomains: t["hsts_include_subdomains"].as_bool().unwrap_or(false),
        }))
    }

    pub fn hsts_value(&self) -> String {
        if self.hsts_include_subdomains {
            format!("max-age={}; includeSubDomains", self.hsts_max_age)
        } else {
            format!("max-age={}", self.hsts_max_age)
        }
    }
}

fn expand(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(rest) => std::env::var("HOME").map(|h| PathBuf::from(h).join(rest)).unwrap_or_else(|_| PathBuf::from(p)),
        None => PathBuf::from(p),
    }
}

// ------------------------------------------------------------------------------------------------- certificates

/// The folder the server's certificate lives in.
pub fn tls_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("server-tls")
}

/// Every name and address this PC answers to, for a self-signed certificate's subject alternative names.
pub fn local_names() -> Vec<String> {
    let mut v: Vec<String> = vec!["localhost".into(), "127.0.0.1".into(), "::1".into()];
    if let Ok(h) = std::fs::read_to_string("/etc/hostname") {
        let h = h.trim().to_lowercase();
        if !h.is_empty() {
            v.push(format!("{h}.local"));
            v.push(h);
        }
    }
    // the LAN addresses: the source address of a route out (no packet is sent by connect on UDP)
    for target in ["192.0.2.1:9", "[2001:db8::1]:9"] {
        if let Ok(s) = std::net::UdpSocket::bind(if target.starts_with('[') { "[::]:0" } else { "0.0.0.0:0" }) {
            if s.connect(target).is_ok() {
                if let Ok(a) = s.local_addr() {
                    v.push(a.ip().to_string());
                }
            }
        }
    }
    if let Some(ts) = tailscale_self() {
        v.extend(ts.ips);
        if let Some(n) = ts.dns_name {
            v.push(n);
        }
    }
    v.sort();
    v.dedup();
    v
}

/// This PC as Tailscale sees it, when tailscale runs and is logged in.
#[derive(Debug, Clone, Default)]
pub struct TailscaleSelf {
    pub dns_name: Option<String>,
    pub ips: Vec<String>,
}

pub fn tailscale_self() -> Option<TailscaleSelf> {
    let o = std::process::Command::new("tailscale").args(["status", "--json"]).output().ok()?;
    if !o.status.success() {
        return None;
    }
    let j: Value = serde_json::from_slice(&o.stdout).ok()?;
    if j["BackendState"].as_str().is_some_and(|s| s != "Running") {
        return None;
    }
    Some(TailscaleSelf {
        dns_name: j["Self"]["DNSName"].as_str().map(|d| d.trim_end_matches('.').to_lowercase()).filter(|d| !d.is_empty()),
        ips: j["Self"]["TailscaleIPs"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default(),
    })
}

fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = path.with_extension("tmp");
    {
        use std::io::Write;
        let mut o = std::fs::OpenOptions::new();
        o.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            o.mode(0o600);
        }
        let mut f = o.open(&tmp)?;
        f.write_all(bytes)?;
    }
    std::fs::rename(&tmp, path)
}

/// The self-signed certificate for `names`, made once and kept; made again only when the names change.
/// Returns (cert.pem, key.pem) paths.
pub fn ensure_self_signed(dir: &Path, names: &[String]) -> Result<(PathBuf, PathBuf), String> {
    let (cert, key, sans) = (dir.join("cert.pem"), dir.join("key.pem"), dir.join("names.json"));
    let wanted = serde_json::to_string(names).unwrap();
    if cert.exists() && key.exists() && std::fs::read_to_string(&sans).ok().as_deref() == Some(wanted.as_str()) {
        return Ok((cert, key));
    }
    let mut params = rcgen::CertificateParams::new(names.to_vec()).map_err(|e| format!("certificate names: {e}"))?;
    params.distinguished_name.push(rcgen::DnType::CommonName, "am2Strata (self-signed)");
    params.not_before = rcgen::date_time_ymd(2024, 1, 1);
    params.not_after = rcgen::date_time_ymd(2036, 1, 1);
    let kp = rcgen::KeyPair::generate().map_err(|e| format!("key: {e}"))?;
    let c = params.self_signed(&kp).map_err(|e| format!("self-signed certificate: {e}"))?;
    write_private(&key, kp.serialize_pem().as_bytes()).map_err(|e| format!("{}: {e}", key.display()))?;
    std::fs::write(&cert, c.pem()).map_err(|e| format!("{}: {e}", cert.display()))?;
    std::fs::write(&sans, wanted).map_err(|e| format!("{}: {e}", sans.display()))?;
    Ok((cert, key))
}

/// `tailscale cert` for this PC's tailnet name into `dir`; (cert, key, name). Errors say what to do.
pub fn tailscale_cert(dir: &Path) -> Result<(PathBuf, PathBuf, String), String> {
    let me = tailscale_self().ok_or("tailscale is not installed, not running or not logged in (`tailscale status`)")?;
    let name = me.dns_name.ok_or("tailscale has no MagicDNS name for this PC: turn on MagicDNS in the tailnet admin")?;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let (cert, key) = (dir.join("cert.pem"), dir.join("key.pem"));
    let o = std::process::Command::new("tailscale")
        .arg("cert").arg("--cert-file").arg(&cert).arg("--key-file").arg(&key).arg(&name).output()
        .map_err(|e| format!("tailscale cert: {e}"))?;
    if !o.status.success() {
        let why = String::from_utf8_lossy(&o.stderr).trim().to_string();
        let hint = if why.contains("Access denied") || why.contains("permission") {
            " (run once: sudo tailscale set --operator=$USER, so this user may ask for certificates)"
        } else if why.contains("HTTPS") || why.contains("not enabled") {
            " (turn on HTTPS certificates in the Tailscale admin console, DNS page)"
        } else { "" };
        return Err(format!("tailscale cert {name} failed: {why}{hint}"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600));
    }
    Ok((cert, key, name))
}

/// A certificate and key the user uploaded (Settings › Secure connection › Load, or `am2strata-run tls load`): checked
/// to load and to belong together, then kept in `<data>/server-tls/uploaded-{cert,key}.pem` (the key 0600).
pub fn save_uploaded(data_dir: &Path, cert_pem: &str, key_pem: &str) -> Result<(PathBuf, PathBuf), String> {
    let dir = tls_dir(data_dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let (tc, tk) = (dir.join("upload-check-cert.pem"), dir.join("upload-check-key.pem"));
    std::fs::write(&tc, cert_pem).map_err(|e| e.to_string())?;
    write_private(&tk, key_pem.as_bytes()).map_err(|e| e.to_string())?;
    let checked = load_pem(&tc, &tk).and_then(|ck| {
        ck.keys_match().map_err(|_| "the key does not belong to the certificate".to_string())
    });
    if let Err(e) = checked {
        let _ = std::fs::remove_file(&tc);
        let _ = std::fs::remove_file(&tk);
        return Err(e.replace(&tc.display().to_string(), "the certificate").replace(&tk.display().to_string(), "the key"));
    }
    let (c, k) = (dir.join("uploaded-cert.pem"), dir.join("uploaded-key.pem"));
    std::fs::rename(&tc, &c).map_err(|e| e.to_string())?;
    std::fs::rename(&tk, &k).map_err(|e| e.to_string())?;
    Ok((c, k))
}

/// A certificate and key from PEM files, as rustls wants them.
pub fn load_pem(cert: &Path, key: &Path) -> Result<CertifiedKey, String> {
    use rustls::pki_types::pem::PemObject;
    let chain: Vec<rustls::pki_types::CertificateDer<'static>> = rustls::pki_types::CertificateDer::pem_file_iter(cert)
        .map_err(|e| format!("{}: {e}", cert.display()))?
        .collect::<Result<_, _>>().map_err(|e| format!("{}: {e}", cert.display()))?;
    if chain.is_empty() {
        return Err(format!("{}: no certificate in it", cert.display()));
    }
    let der = rustls::pki_types::PrivateKeyDer::from_pem_file(key).map_err(|e| format!("{}: {e}", key.display()))?;
    let signer = rustls::crypto::ring::sign::any_supported_type(&der).map_err(|e| format!("{}: {e}", key.display()))?;
    Ok(CertifiedKey::new(chain, signer))
}

/// SHA-256 of a certificate's DER, as browsers show it (AA:BB:...).
pub fn fingerprint(ck: &CertifiedKey) -> String {
    use sha2::Digest;
    let d = sha2::Sha256::digest(ck.cert[0].as_ref());
    d.iter().map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(":")
}

/// The certificate the listener presents, swappable while it runs (renewals, changed files).
#[derive(Debug)]
pub struct Resolver(RwLock<Arc<CertifiedKey>>);

impl Resolver {
    pub fn new(ck: CertifiedKey) -> Arc<Resolver> {
        Arc::new(Resolver(RwLock::new(Arc::new(ck))))
    }
    pub fn set(&self, ck: CertifiedKey) {
        *self.0.write().unwrap() = Arc::new(ck);
    }
    pub fn current(&self) -> Arc<CertifiedKey> {
        self.0.read().unwrap().clone()
    }
}

impl ResolvesServerCert for Resolver {
    fn resolve(&self, _hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        Some(self.current())
    }
}

/// Where the certificate comes from, so a watcher can reload or renew it.
#[derive(Debug, Clone)]
pub struct Source {
    pub cert: PathBuf,
    pub key: PathBuf,
    pub renew_with_tailscale: bool,
    pub tailnet_name: Option<String>,
}

/// Get the certificate for `cfg` (making it if needed): (the certificate, where it came from).
pub fn certificate(cfg: &Config, data_dir: &Path) -> Result<(CertifiedKey, Source), String> {
    let dir = tls_dir(data_dir);
    let src = match &cfg.mode {
        Mode::SelfSigned => {
            let (cert, key) = ensure_self_signed(&dir, &local_names())?;
            Source { cert, key, renew_with_tailscale: false, tailnet_name: None }
        }
        Mode::Tailscale => {
            let (cert, key, name) = tailscale_cert(&dir)?;
            Source { cert, key, renew_with_tailscale: true, tailnet_name: Some(name) }
        }
        Mode::Files { cert, key } => Source { cert: cert.clone(), key: key.clone(), renew_with_tailscale: false, tailnet_name: None },
    };
    Ok((load_pem(&src.cert, &src.key)?, src))
}

fn mtime(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

/// Keep the presented certificate current: reload when the files change, and with Tailscale ask for a fresh one daily.
pub fn watch(resolver: Arc<Resolver>, source: Arc<RwLock<Source>>, data_dir: PathBuf) {
    tokio::spawn(async move {
        let first = source.read().unwrap().clone();
        let mut seen = (mtime(&first.cert), mtime(&first.key));
        let mut last_renew = std::time::Instant::now();
        loop {
            tokio::time::sleep(Duration::from_secs(60)).await;
            let src = source.read().unwrap().clone();
            if src.renew_with_tailscale && last_renew.elapsed() > Duration::from_secs(24 * 3600) {
                last_renew = std::time::Instant::now();
                let dir = tls_dir(&data_dir);
                if let Err(e) = tokio::task::spawn_blocking(move || tailscale_cert(&dir)).await.unwrap_or_else(|e| Err(e.to_string())) {
                    tracing::warn!("https: renewing the Tailscale certificate failed ({e}); the current one stays");
                }
            }
            let now = (mtime(&src.cert), mtime(&src.key));
            if now != seen {
                seen = now;
                match load_pem(&src.cert, &src.key) {
                    Ok(ck) => {
                        tracing::info!("https: certificate reloaded (SHA-256 {})", fingerprint(&ck));
                        resolver.set(ck);
                    }
                    Err(e) => tracing::warn!("https: the changed certificate did not load ({e}); the previous one stays"),
                }
            }
        }
    });
}

// -------------------------------------------------------------------------------------------------------- serving

/// Marks a request that came in over TLS (set by the listener, read by `security_headers` and `http_policy`).
#[derive(Clone, Copy, Debug)]
pub struct OverTls;

/// The HSTS and plain-HTTP rules in force.
#[derive(Clone)]
pub struct Policy {
    pub https_port: u16,
    pub http: HttpMode,
    pub hsts: HeaderValue,
}

impl Policy {
    pub fn new(cfg: &Config) -> Policy {
        Policy { https_port: cfg.https_port, http: cfg.http, hsts: HeaderValue::from_str(&cfg.hsts_value()).unwrap() }
    }
}

/// HTTPS while the server runs: turned on, given another certificate, or off — from the run config at start, the
/// web app's Settings or the CLI — without a restart. One per process.
pub struct Runtime {
    data_dir: PathBuf,
    host: String,
    api_port: u16,
    /// the cluster's TLS port when this server is a worker (HTTPS must not take it)
    cluster_port: Option<u16>,
    app: std::sync::OnceLock<axum::Router>,
    policy: RwLock<Option<Policy>>,
    inner: tokio::sync::Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    cfg: Option<Config>,
    resolver: Option<Arc<Resolver>>,
    /// the port the listener is bound to (it stays for the process; another port needs a restart)
    bound: Option<u16>,
    enabled: Option<Arc<std::sync::atomic::AtomicBool>>,
    source: Option<Arc<RwLock<Source>>>,
    watching: bool,
    last_error: Option<String>,
}

static RUNTIME: std::sync::OnceLock<Arc<Runtime>> = std::sync::OnceLock::new();

/// The process's HTTPS runtime (set up by main).
pub fn runtime() -> Option<Arc<Runtime>> {
    RUNTIME.get().cloned()
}

impl Runtime {
    pub fn init(data_dir: PathBuf, host: String, api_port: u16, cluster_port: Option<u16>) -> Arc<Runtime> {
        RUNTIME.get_or_init(|| Arc::new(Runtime {
            data_dir, host, api_port, cluster_port, app: std::sync::OnceLock::new(), policy: RwLock::new(None),
            inner: tokio::sync::Mutex::new(Inner::default()),
        })).clone()
    }

    /// The app the HTTPS listener serves (the same router as plain HTTP).
    pub fn set_app(&self, app: axum::Router) {
        let _ = self.app.set(app);
    }

    pub fn policy(&self) -> Option<Policy> {
        self.policy.read().unwrap().clone()
    }

    /// Turn HTTPS on with `cfg` (getting or making its certificate), or off with None. On an error nothing changes:
    /// the server keeps serving exactly as before.
    pub async fn apply(self: &Arc<Self>, cfg: Option<Config>) -> Result<(), String> {
        let mut inner = self.inner.lock().await;
        let Some(cfg) = cfg else {
            if let Some(e) = &inner.enabled {
                e.store(false, std::sync::atomic::Ordering::SeqCst);
            }
            *self.policy.write().unwrap() = None;
            inner.cfg = None;
            inner.last_error = None;
            tracing::info!("https: off (plain HTTP on {} answers everything)", self.api_port);
            return Ok(());
        };
        let fail = |inner: &mut Inner, e: String| { inner.last_error = Some(e.clone()); Err(e) };
        if cfg.https_port == self.api_port {
            return fail(&mut inner, format!("the HTTPS port {} is the API's own port; pick another (8001 by default)", cfg.https_port));
        }
        if Some(cfg.https_port) == self.cluster_port {
            return fail(&mut inner, format!("port {} is this worker's cluster TLS port; pick another", cfg.https_port));
        }
        if let Some(b) = inner.bound.filter(|b| *b != cfg.https_port) {
            return fail(&mut inner, format!("HTTPS is already listening on {b}; another port takes a restart of the server"));
        }
        let (c2, data) = (cfg.clone(), self.data_dir.clone());
        let (ck, src) = match tokio::task::spawn_blocking(move || certificate(&c2, &data)).await {
            Ok(Ok(x)) => x,
            Ok(Err(e)) => return fail(&mut inner, e),
            Err(e) => return fail(&mut inner, e.to_string()),
        };
        let fp = fingerprint(&ck);
        match inner.resolver.clone() {
            Some(r) => r.set(ck),
            None => {
                let r = Resolver::new(ck);
                let Some(app) = self.app.get().cloned() else { return fail(&mut inner, "the server is still starting".into()) };
                let listener = match tokio::net::TcpListener::bind((self.host.as_str(), cfg.https_port)).await {
                    Ok(l) => l,
                    Err(e) => return fail(&mut inner, format!("bind {}:{}: {e}", self.host, cfg.https_port)),
                };
                let enabled = Arc::new(std::sync::atomic::AtomicBool::new(true));
                tokio::spawn(accept_loop(listener, app, r.clone(), enabled.clone()));
                inner.enabled = Some(enabled);
                inner.bound = Some(cfg.https_port);
                inner.resolver = Some(r);
            }
        }
        if let Some(e) = &inner.enabled {
            e.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        match &inner.source {
            Some(s) => *s.write().unwrap() = src.clone(),
            None => inner.source = Some(Arc::new(RwLock::new(src.clone()))),
        }
        if !inner.watching {
            inner.watching = true;
            watch(inner.resolver.clone().unwrap(), inner.source.clone().unwrap(), self.data_dir.clone());
        }
        *self.policy.write().unwrap() = Some(Policy::new(&cfg));
        let name = src.tailnet_name.clone().unwrap_or_else(|| "localhost".into());
        tracing::info!("https: on at https://{name}:{}/ (certificate SHA-256 {fp}; {:?}); plain HTTP on {}: {:?}",
                       cfg.https_port, cfg.mode, self.api_port, cfg.http);
        inner.cfg = Some(cfg);
        inner.last_error = None;
        Ok(())
    }

    /// What the web app and `am2strata-run tls status` show.
    pub async fn status(&self) -> Value {
        let inner = self.inner.lock().await;
        let on = inner.cfg.is_some() && inner.enabled.as_ref().is_some_and(|e| e.load(std::sync::atomic::Ordering::SeqCst));
        let mut v = serde_json::json!({
            "on": on, "api_port": self.api_port, "last_error": inner.last_error,
            "tailscale": tailscale_self().map(|t| serde_json::json!({"name": t.dns_name, "ips": t.ips})),
        });
        if let Some(c) = &inner.cfg {
            v["mode"] = serde_json::json!(match &c.mode { Mode::SelfSigned => "self-signed", Mode::Tailscale => "tailscale", Mode::Files { .. } => "files" });
            v["https_port"] = serde_json::json!(c.https_port);
            v["http"] = serde_json::json!(match c.http { HttpMode::LocalOnly => "local-only", HttpMode::Redirect => "redirect", HttpMode::Keep => "keep" });
            v["hsts"] = serde_json::json!(c.hsts_value());
        }
        if let (Some(r), Some(src)) = (&inner.resolver, &inner.source) {
            let ck = r.current();
            v["certificate"] = cert_info(&ck);
            let src = src.read().unwrap();
            v["certificate"]["file"] = serde_json::json!(src.cert.display().to_string());
            if let (true, Some(c)) = (on, &inner.cfg) {
                let mut urls: Vec<String> = vec![];
                if let Some(n) = &src.tailnet_name { urls.push(format!("https://{n}:{}/", c.https_port)); }
                for n in v["certificate"]["names"].as_array().into_iter().flatten().filter_map(|x| x.as_str()) {
                    let n = if n.contains(':') { format!("[{n}]") } else { n.to_string() };
                    let u = format!("https://{n}:{}/", c.https_port);
                    if !urls.contains(&u) { urls.push(u); }
                }
                v["urls"] = serde_json::json!(urls);
            }
        }
        v
    }
}

/// The certificate's names, issuer, validity and fingerprint.
pub fn cert_info(ck: &CertifiedKey) -> Value {
    let mut v = serde_json::json!({"sha256": fingerprint(ck)});
    if let Ok((_, x)) = x509_parser::parse_x509_certificate(ck.cert[0].as_ref()) {
        v["subject"] = serde_json::json!(x.subject().to_string());
        v["issuer"] = serde_json::json!(x.issuer().to_string());
        v["self_signed"] = serde_json::json!(x.subject() == x.issuer());
        v["not_before"] = serde_json::json!(x.validity().not_before.to_rfc2822().unwrap_or_default());
        v["not_after"] = serde_json::json!(x.validity().not_after.to_rfc2822().unwrap_or_default());
        v["expires_in_days"] = serde_json::json!(x.validity().time_to_expiration().map(|d| d.whole_days()).unwrap_or(0));
        let names: Vec<String> = x.subject_alternative_name().ok().flatten().map(|s| s.value.general_names.iter().filter_map(|g| match g {
            x509_parser::extensions::GeneralName::DNSName(d) => Some(d.to_string()),
            x509_parser::extensions::GeneralName::IPAddress(ip) => match ip.len() {
                4 => Some(std::net::Ipv4Addr::new(ip[0], ip[1], ip[2], ip[3]).to_string()),
                16 => <[u8; 16]>::try_from(*ip).ok().map(|a| std::net::Ipv6Addr::from(a).to_string()),
                _ => None,
            },
            _ => None,
        }).collect()).unwrap_or_default();
        v["names"] = serde_json::json!(names);
    }
    v
}

async fn accept_loop(listener: tokio::net::TcpListener, app: axum::Router, resolver: Arc<Resolver>,
                     enabled: Arc<std::sync::atomic::AtomicBool>) {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut cfg = match rustls::ServerConfig::builder_with_provider(provider).with_safe_default_protocol_versions() {
        Ok(b) => b.with_no_client_auth().with_cert_resolver(resolver),
        Err(e) => {
            tracing::error!("https: TLS setup failed: {e}");
            return;
        }
    };
    cfg.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(cfg));
    loop {
        let Ok((tcp, addr)) = listener.accept().await else { continue };
        if !enabled.load(std::sync::atomic::Ordering::SeqCst) {
            continue; // HTTPS turned off: new connections are closed
        }
        let (acceptor, app) = (acceptor.clone(), app.clone());
        tokio::spawn(async move {
            let tls = match tokio::time::timeout(Duration::from_secs(10), acceptor.accept(tcp)).await {
                Ok(Ok(t)) => t,
                Ok(Err(e)) => {
                    tracing::debug!("https: TLS handshake with {addr} failed: {e}");
                    return;
                }
                Err(_) => return,
            };
            let svc = app.layer(axum::Extension(OverTls)).layer(axum::Extension(axum::extract::ConnectInfo(addr)));
            let svc = hyper_util::service::TowerToHyperService::new(svc);
            let _ = hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                .serve_connection_with_upgrades(hyper_util::rt::TokioIo::new(tls), svc).await;
        });
    }
}

/// HSTS on responses that went out over TLS, never on plain HTTP (the header is meaningless there, and it would tell a
/// browser to insist on HTTPS for a name that may not have it).
pub async fn security_headers(req: axum::extract::Request, next: axum::middleware::Next) -> Response {
    let tls = req.extensions().get::<OverTls>().is_some();
    let mut res = next.run(req).await;
    if let (true, Some(p)) = (tls, runtime().and_then(|r| r.policy())) {
        res.headers_mut().insert(axum::http::header::STRICT_TRANSPORT_SECURITY, p.hsts.clone());
    }
    res
}

/// What a plain-HTTP request gets once HTTPS is on: answered, or redirected (308, so the method and body are kept).
pub fn plain_http_answer(p: &Policy, peer: Option<std::net::IpAddr>, host: &str, path_and_query: &str) -> Option<Response> {
    let local = peer.is_some_and(|ip| ip.is_loopback());
    let redirect = match p.http {
        HttpMode::Keep => false,
        HttpMode::LocalOnly => !local,
        HttpMode::Redirect => path_and_query != "/health" && !path_and_query.starts_with("/health?"),
    };
    if !redirect {
        return None;
    }
    let name = strip_port(host);
    let name = if name.is_empty() { "localhost" } else { name };
    let port = if p.https_port == 443 { String::new() } else { format!(":{}", p.https_port) };
    let to = format!("https://{name}{port}{path_and_query}");
    let mut r = (StatusCode::PERMANENT_REDIRECT, format!("this server answers over HTTPS: {to}\n")).into_response();
    if let Ok(v) = HeaderValue::from_str(&to) {
        r.headers_mut().insert(axum::http::header::LOCATION, v);
    }
    Some(r)
}

fn strip_port(host: &str) -> &str {
    if let Some(rest) = host.strip_prefix('[') {
        return rest.split(']').next().map(|h| &host[..h.len() + 2]).unwrap_or(host);
    }
    match host.rsplit_once(':') {
        Some((h, p)) if p.chars().all(|c| c.is_ascii_digit()) => h,
        _ => host,
    }
}

/// The plain-HTTP side of the policy, as a middleware: requests over TLS pass; plain ones may be redirected.
pub async fn http_policy(req: axum::extract::Request, next: axum::middleware::Next) -> Response {
    if let Some(p) = runtime().and_then(|r| r.policy()) {
        if req.extensions().get::<OverTls>().is_none() {
            let peer = req.extensions().get::<axum::extract::ConnectInfo<std::net::SocketAddr>>().map(|c| c.0.ip());
            let host = req.headers().get("host").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
            let pq = req.uri().path_and_query().map(|x| x.as_str().to_string()).unwrap_or_else(|| "/".into());
            if let Some(r) = plain_http_answer(&p, peer, &host, &pq) {
                return r;
            }
        }
    }
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(v: Value) -> Result<Option<Config>, String> {
        Config::from_config(&v.as_object().unwrap().clone().into_iter().collect())
    }

    #[test]
    fn off_unless_configured_and_typos_are_errors() {
        assert!(cfg(serde_json::json!({})).unwrap().is_none());
        assert!(cfg(serde_json::json!({"tls": false})).unwrap().is_none());
        let c = cfg(serde_json::json!({"tls": {}})).unwrap().unwrap();
        assert_eq!((c.mode, c.https_port, c.http), (Mode::SelfSigned, DEFAULT_HTTPS_PORT, HttpMode::LocalOnly));
        assert!(cfg(serde_json::json!({"tls": {"mode": "selfsigned-typo"}})).unwrap_err().contains("unknown mode"));
        assert!(cfg(serde_json::json!({"tls": {"mode": "files"}})).unwrap_err().contains("needs \"cert\""));
        assert!(cfg(serde_json::json!({"tls": {"http": "nope"}})).unwrap_err().contains("unknown \"http\""));
        assert!(cfg(serde_json::json!({"tls": {"https_port": 70000}})).is_err());
    }

    #[test]
    fn hsts_value() {
        let mut c = cfg(serde_json::json!({"tls": {}})).unwrap().unwrap();
        assert_eq!(c.hsts_value(), "max-age=31536000");
        c.hsts_include_subdomains = true;
        assert_eq!(c.hsts_value(), "max-age=31536000; includeSubDomains");
    }

    #[test]
    fn plain_http_policy() {
        let p = |http| Policy { https_port: 8001, http, hsts: HeaderValue::from_static("max-age=1") };
        let lo: Option<std::net::IpAddr> = Some("127.0.0.1".parse().unwrap());
        let lan: Option<std::net::IpAddr> = Some("192.168.8.105".parse().unwrap());
        // local-only: this PC's tools keep plain HTTP; the LAN is sent to HTTPS on the same name
        assert!(plain_http_answer(&p(HttpMode::LocalOnly), lo, "127.0.0.1:8000", "/v1/models").is_none());
        let r = plain_http_answer(&p(HttpMode::LocalOnly), lan, "192.168.8.105:8000", "/v1/models?x=1").unwrap();
        assert_eq!(r.status(), StatusCode::PERMANENT_REDIRECT);
        assert_eq!(r.headers()["location"], "https://192.168.8.105:8001/v1/models?x=1");
        // redirect: everything except /health
        assert!(plain_http_answer(&p(HttpMode::Redirect), lo, "localhost:8000", "/health").is_none());
        assert!(plain_http_answer(&p(HttpMode::Redirect), lo, "localhost:8000", "/").is_some());
        // keep: nothing changes
        assert!(plain_http_answer(&p(HttpMode::Keep), lan, "x:8000", "/").is_none());
        // an IPv6 host keeps its brackets
        let r = plain_http_answer(&p(HttpMode::Redirect), lan, "[fd7a::1]:8000", "/").unwrap();
        assert_eq!(r.headers()["location"], "https://[fd7a::1]:8001/");
    }

    #[test]
    fn self_signed_is_made_once_kept_private_and_remade_when_names_change() {
        let d = std::env::temp_dir().join(format!("am2-server-tls-{}-{}", std::process::id(),
            SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos()));
        let names = vec!["localhost".to_string(), "127.0.0.1".to_string()];
        let (c, k) = ensure_self_signed(&d, &names).unwrap();
        let first = load_pem(&c, &k).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&k).unwrap().permissions().mode() & 0o777, 0o600, "the key is private");
        }
        let (c2, k2) = ensure_self_signed(&d, &names).unwrap();
        assert_eq!(fingerprint(&first), fingerprint(&load_pem(&c2, &k2).unwrap()), "kept, not remade");
        let more = vec!["localhost".to_string(), "127.0.0.1".to_string(), "pc.local".to_string()];
        let (c3, k3) = ensure_self_signed(&d, &more).unwrap();
        assert_ne!(fingerprint(&first), fingerprint(&load_pem(&c3, &k3).unwrap()), "new names, new certificate");
        let _ = std::fs::remove_dir_all(&d);
    }
}
