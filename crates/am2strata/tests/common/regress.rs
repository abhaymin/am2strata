//! What the regression suite's test files share (docs/REGRESSION.md): requests with any key, header and body, the
//! route list read from main.rs, JSON shapes for the golden files, and temp folders that clean up after themselves.

use super::*;
use std::collections::BTreeSet;

/// A client profile's key (`"clients": {"laptop": ...}` in the run config `client_model` writes).
pub const CLIENT_KEY: &str = "laptop-key-0123456789";
/// A key of the main key's length that is not it (a guess one character off).
pub const WRONG_KEY: &str = "test-kez";
/// Another site's page, as a browser names it.
pub const FOREIGN_ORIGIN: &str = "https://evil.example";

/// A model whose run config has one client profile (key `CLIENT_KEY`), plus `extra`.
pub fn client_model(extra: Value) -> Model<'static> {
    let mut e = json!({"clients": {"laptop": {"key": CLIENT_KEY, "tools": ["memory"]}}});
    if let (Some(a), Some(b)) = (e.as_object_mut(), extra.as_object()) {
        a.extend(b.clone());
    }
    Model { extra: e, ..Model::default() }
}

/// One request, said in full: nothing is added that the caller did not ask for.
pub struct Req<'a> {
    pub method: &'a str,
    pub url: String,
    pub key: Option<&'a str>,
    pub headers: Vec<(&'a str, String)>,
    /// (content type, bytes)
    pub body: Option<(&'a str, Vec<u8>)>,
}

pub struct Answer {
    pub status: u16,
    pub headers: reqwest::header::HeaderMap,
    pub text: String,
}

impl Answer {
    pub fn json(&self) -> Value {
        serde_json::from_str(&self.text).unwrap_or(Value::Null)
    }

    /// The SSE events of a streamed answer, as (event name, data).
    pub fn events(&self) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        let mut name = String::new();
        for line in self.text.lines() {
            if let Some(n) = line.strip_prefix("event:") {
                name = n.trim().to_string();
            } else if let Some(d) = line.strip_prefix("data:") {
                let d = d.trim();
                out.push((name.clone(), if d == "[DONE]" { json!("[DONE]") } else { serde_json::from_str(d).unwrap_or(json!(d)) }));
            }
        }
        out
    }
}

impl<'a> Req<'a> {
    pub fn new(method: &'a str, base: &str, path: &str) -> Req<'a> {
        Req { method, url: format!("{base}{path}"), key: None, headers: Vec::new(), body: None }
    }

    pub fn key(mut self, k: Option<&'a str>) -> Self {
        self.key = k;
        self
    }

    pub fn header(mut self, k: &'a str, v: &str) -> Self {
        self.headers.push((k, v.to_string()));
        self
    }

    pub fn json(mut self, v: &Value) -> Self {
        self.body = Some(("application/json", v.to_string().into_bytes()));
        self
    }

    pub fn raw(mut self, content_type: &'a str, bytes: &[u8]) -> Self {
        self.body = Some((content_type, bytes.to_vec()));
        self
    }

    pub fn send(self) -> Answer {
        let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(60))
            .redirect(reqwest::redirect::Policy::none()).build().unwrap();
        let mut rb = client.request(self.method.parse().expect("method"), &self.url);
        if let Some(k) = self.key {
            rb = rb.bearer_auth(k);
        }
        for (k, v) in &self.headers {
            rb = rb.header(*k, v);
        }
        if let Some((ct, b)) = self.body {
            rb = rb.header("content-type", ct).body(b);
        }
        let r = rb.send().unwrap_or_else(|e| panic!("{} {}: {e}", self.method, self.url));
        Answer { status: r.status().as_u16(), headers: r.headers().clone(), text: r.text().unwrap_or_default() }
    }
}

/// Every route main.rs registers, as (METHOD, path pattern): `.route("…", get(..).post(..))` one per method,
/// `.route_service` and `.nest_service` as GET. Panics if main.rs builds its router in a way this cannot read
/// (a merged or nested sub-router), so such a change is taught to the suite rather than slipping past it.
pub fn routes_in_main() -> BTreeSet<(String, String)> {
    let main = include_str!("../../src/main.rs");
    let app = main.split("let app = Router::new()").nth(1).expect("main.rs builds `let app = Router::new()`")
        .split(".with_state(").next().expect("the router ends in .with_state(");
    let code: String = app.lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n");
    for sub in [".merge(", ".nest(", ".fallback(", ".fallback_service(", ".route_layer("] {
        assert!(!code.contains(sub), "main.rs now uses `{sub}`: teach tests/common/regress.rs::routes_in_main to read its routes \
                                       (and add them to the table in tests/regression_api.rs)");
    }
    let mut out = BTreeSet::new();
    let method_re = regex::Regex::new(r"\b(get|post|put|delete|patch|head|options|any)\(").unwrap();
    for part in code.split(".route(\"").skip(1) {
        let (path, rest) = part.split_once('"').expect("a path");
        // up to the next builder call that is not a method of this route
        let rest = rest.split("\n").next().unwrap_or("");
        let methods: Vec<String> = method_re.captures_iter(rest).map(|c| c[1].to_uppercase()).collect();
        assert!(!methods.is_empty(), "no method read for route {path:?}: {rest}");
        for m in methods {
            assert!(m != "ANY", "route {path:?} answers any method: list its methods in the regression table by hand");
            out.insert((m, path.to_string()));
        }
    }
    for kind in [".route_service(\"", ".nest_service(\""] {
        for part in code.split(kind).skip(1) {
            let (path, _) = part.split_once('"').expect("a path");
            out.insert(("GET".to_string(), path.to_string()));
        }
    }
    out
}

/// Routers built anywhere else in the server's sources (outside tests): none are expected.
pub fn routers_outside_main() -> Vec<String> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().and_then(|x| x.to_str()) == Some("rs") {
                out.push(p);
            }
        }
    }
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&src, &mut files);
    let mut found = Vec::new();
    for f in files {
        if f.file_name().and_then(|n| n.to_str()) == Some("main.rs") {
            continue;
        }
        let text = std::fs::read_to_string(&f).unwrap_or_default();
        // unit tests (after `#[cfg(test)]`) may build a router of their own
        let live = text.split("#[cfg(test)]").next().unwrap_or("");
        if live.contains("Router::new()") {
            found.push(f.strip_prefix(&src).unwrap_or(&f).display().to_string());
        }
    }
    found
}

/// The SHAPE of a JSON value: keys and value types, no values. Objects keep their keys (sorted); an array becomes
/// the list of the distinct shapes of its items (so ten chunks of one shape are one entry); scalars become their
/// type's name. Compared with a golden file, this catches a renamed, dropped, added or retyped field and ignores ids,
/// timestamps, token counts and timings.
pub fn shape(v: &Value) -> Value {
    match v {
        Value::Null => json!("null"),
        Value::Bool(_) => json!("bool"),
        Value::Number(n) => json!(if n.is_f64() { "float" } else { "int" }),
        Value::String(_) => json!("string"),
        Value::Array(a) => {
            let mut seen: Vec<Value> = Vec::new();
            for x in a {
                let s = shape(x);
                if !seen.contains(&s) {
                    seen.push(s);
                }
            }
            seen.sort_by_key(|s| s.to_string());
            Value::Array(seen)
        }
        Value::Object(o) => {
            let mut keys: Vec<&String> = o.keys().collect();
            keys.sort();
            Value::Object(keys.into_iter().map(|k| (k.clone(), shape(&o[k]))).collect())
        }
    }
}

/// A folder under the system's temp folder, removed when dropped (tests never write into the repository).
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(what: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("am2strata-regress-{what}-{}-{}", std::process::id(), uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Poll `f` until it gives Some, or panic with `what` after `max` (no fixed sleeps in tests).
pub fn wait_for<T>(what: &str, max: Duration, mut f: impl FnMut() -> Option<T>) -> T {
    let t0 = Instant::now();
    loop {
        if let Some(v) = f() {
            return v;
        }
        assert!(t0.elapsed() < max, "timed out after {max:?} waiting for {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// A client that trusts only the server's own certificate (`<data dir>/server-tls/cert.pem`), once HTTPS answers on
/// `port`. Never "accept any certificate".
pub fn https_client(dir: &Path, port: u16) -> reqwest::blocking::Client {
    wait_for("HTTPS to come up", Duration::from_secs(30), || {
        let pem = std::fs::read(dir.join("server-tls/cert.pem")).ok()?;
        let c = reqwest::blocking::Client::builder().add_root_certificate(reqwest::Certificate::from_pem(&pem).ok()?)
            .redirect(reqwest::redirect::Policy::none()).timeout(Duration::from_secs(10)).build().ok()?;
        c.get(format!("https://localhost:{port}/health")).send().is_ok_and(|r| r.status().is_success()).then_some(c)
    })
}

/// One HTTP/1.1 request written by hand (a `Host` no HTTP client would send): the status code and the headers, or
/// None when the server closed the connection without an answer.
pub fn raw_http(port: u16, request: &str) -> Option<(u16, String)> {
    use std::io::{Read, Write};
    let mut t = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    t.set_read_timeout(Some(Duration::from_secs(10))).ok()?;
    t.write_all(request.as_bytes()).ok()?;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        match t.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
    let head = String::from_utf8_lossy(&buf).into_owned();
    let status = head.split_whitespace().nth(1)?.parse().ok()?;
    Some((status, head.split("\r\n\r\n").next().unwrap_or("").to_lowercase()))
}
