//! Tools from MCP (Model Context Protocol) servers for the web app's chats. Port of serve/mcp.py.
//!
//! The run config lists servers (`"mcp_servers"`, or Claude Desktop's `"mcpServers"` block pasted as is). Each is a
//! program this server starts (`command`, `args`, `env`, `cwd`: newline-delimited JSON-RPC 2.0 on its stdin/stdout)
//! or an address (`url`, `headers`: Streamable HTTP, JSON-RPC POSTed, answered as JSON or an event stream). The hub
//! starts them in the background, offers their tools to the model as `<server>__<tool>`, and routes a call back to
//! the server that has it. A server that does not start is left out with one log line; one that stops later is
//! started again on its next call. Results reach the model as text capped at `max_result_chars`; a failed call
//! becomes a result that starts with "error:".

use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{oneshot, Mutex};

const PROTOCOL: &str = "2025-06-18";

#[derive(Debug, Clone)]
pub struct Settings {
    pub timeout_s: f64,
    pub max_result_chars: usize,
    pub max_rounds: usize,
    pub start_timeout_s: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { timeout_s: 60.0, max_result_chars: 20000, max_rounds: 8, start_timeout_s: 120.0 }
    }
}

type Pending = Arc<StdMutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>;

/// The answer of a JSON-RPC response: its result, or its error as text.
fn answer(m: &Value) -> Result<Value, String> {
    match m.get("error") {
        Some(e) if !e.is_null() => Err(format!(
            "{} (code {})",
            e.get("message").and_then(Value::as_str).unwrap_or("error"),
            e.get("code").map(|c| c.to_string()).unwrap_or_default()
        )),
        _ => Ok(m.get("result").cloned().unwrap_or(json!({}))),
    }
}

// ------------------------------------------------------------------------------------------------ stdio

struct StdioTransport {
    stdin: Mutex<Option<tokio::process::ChildStdin>>,
    child: Mutex<Option<tokio::process::Child>>,
    pid: Option<u32>,
    pending: Pending,
    next_id: AtomicU64,
    ended: Arc<AtomicBool>,
    stderr: Arc<StdMutex<Vec<String>>>,
}

impl StdioTransport {
    async fn start(cfg: &Value) -> Result<Arc<Self>, String> {
        let command = cfg.get("command").and_then(Value::as_str).unwrap_or("").to_string();
        let mut cmd = tokio::process::Command::new(&command);
        for a in cfg.get("args").and_then(Value::as_array).into_iter().flatten() {
            cmd.arg(a.as_str().map(str::to_owned).unwrap_or_else(|| a.to_string()));
        }
        for (k, v) in cfg.get("env").and_then(Value::as_object).into_iter().flatten() {
            cmd.env(k, v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string()));
        }
        if let Some(cwd) = cfg.get("cwd").and_then(Value::as_str).filter(|c| !c.is_empty()) {
            cmd.current_dir(cwd);
        }
        #[cfg(unix)]
        cmd.process_group(0); // its own process group, so close() ends its children too
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
        let mut child = cmd.spawn().map_err(|e| format!("could not start {command:?}: {e}"))?;
        let (stdin, stdout, stderr) = (child.stdin.take(), child.stdout.take(), child.stderr.take());
        let t = Arc::new(StdioTransport {
            stdin: Mutex::new(stdin),
            pid: child.id(),
            child: Mutex::new(Some(child)),
            pending: Arc::default(),
            next_id: AtomicU64::new(0),
            ended: Arc::default(),
            stderr: Arc::default(),
        });
        if let Some(err) = stderr {
            let log = t.stderr.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(err).lines();
                while let Ok(Some(l)) = lines.next_line().await {
                    let l = l.trim_end().to_string();
                    if !l.is_empty() {
                        let mut g = log.lock().unwrap();
                        g.push(l);
                        let n = g.len();
                        if n > 40 {
                            g.drain(..n - 40);
                        }
                    }
                }
            });
        }
        if let Some(out) = stdout {
            let me = Arc::downgrade(&t);
            let (pending, ended, log) = (t.pending.clone(), t.ended.clone(), t.stderr.clone());
            tokio::spawn(async move {
                let mut lines = BufReader::new(out).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    let Ok(msg) = serde_json::from_str::<Value>(line) else {
                        log.lock().unwrap().push(line.to_string()); // a server that prints to stdout
                        continue;
                    };
                    let msgs = match msg {
                        Value::Array(a) => a,
                        m => vec![m],
                    };
                    for m in msgs {
                        if let Some(method) = m.get("method").and_then(Value::as_str) {
                            // a request from the server: no client features are offered (ping is answered)
                            if let (Some(id), Some(t)) = (m.get("id"), me.upgrade()) {
                                let reply = if method == "ping" {
                                    json!({"jsonrpc": "2.0", "id": id, "result": {}})
                                } else {
                                    json!({"jsonrpc": "2.0", "id": id,
                                           "error": {"code": -32601, "message": format!("method not supported: {method}")}})
                                };
                                let _ = t.send(&reply).await;
                            }
                            continue;
                        }
                        let id = m.get("id").and_then(Value::as_u64);
                        if let Some(tx) = id.and_then(|i| pending.lock().unwrap().remove(&i)) {
                            let _ = tx.send(answer(&m));
                        }
                    }
                }
                ended.store(true, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(200)).await; // the last log line says why it stopped
                let tail = log.lock().unwrap().last().map(|l| format!(": {}", &l[..l.len().min(300)])).unwrap_or_default();
                let slots: Vec<_> = pending.lock().unwrap().drain().collect();
                for (_, tx) in slots {
                    let _ = tx.send(Err(format!("the server stopped{tail}")));
                }
            });
        }
        Ok(t)
    }

    fn tail(&self) -> String {
        self.stderr.lock().unwrap().last().map(|l| format!(": {}", &l[..l.len().min(300)])).unwrap_or_default()
    }

    fn alive(&self) -> bool {
        !self.ended.load(Ordering::SeqCst)
    }

    async fn send(&self, msg: &Value) -> Result<(), String> {
        let mut data = serde_json::to_vec(msg).unwrap_or_default();
        data.push(b'\n');
        let mut g = self.stdin.lock().await;
        let Some(stdin) = g.as_mut() else { return Err(format!("the server stopped{}", self.tail())) };
        stdin.write_all(&data).await.map_err(|_| format!("the server stopped{}", self.tail()))?;
        stdin.flush().await.map_err(|_| format!("the server stopped{}", self.tail()))
    }

    async fn request(&self, method: &str, params: Value, timeout: Duration) -> Result<Value, String> {
        if !self.alive() {
            return Err(format!("the server is not running{}", self.tail()));
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst) + 1;
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, tx);
        if let Err(e) = self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})).await {
            self.pending.lock().unwrap().remove(&id);
            return Err(e);
        }
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(r)) => r,
            Ok(Err(_)) => Err(format!("the server stopped{}", self.tail())),
            Err(_) => {
                self.pending.lock().unwrap().remove(&id);
                let reason = format!("{method} took longer than {:.0} s", timeout.as_secs_f64());
                let _ = self.notify("notifications/cancelled", Some(json!({"requestId": id, "reason": reason}))).await;
                Err(reason)
            }
        }
    }

    async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), String> {
        let mut m = json!({"jsonrpc": "2.0", "method": method});
        if let Some(p) = params {
            m["params"] = p;
        }
        self.send(&m).await
    }

    /// As the spec says: close its stdin, give it a moment to exit, then end it (and what it started).
    async fn close(&self) {
        self.stdin.lock().await.take();
        let mut g = self.child.lock().await;
        if let Some(child) = g.as_mut() {
            if tokio::time::timeout(Duration::from_secs(2), child.wait()).await.is_err() {
                #[cfg(unix)]
                if let Some(pid) = self.pid {
                    let _ = std::process::Command::new("kill").args(["-TERM", &format!("-{pid}")]).status();
                }
                if tokio::time::timeout(Duration::from_secs(2), child.wait()).await.is_err() {
                    let _ = child.kill().await;
                }
            }
        }
        g.take();
    }
}

// ------------------------------------------------------------------------------------------------ Streamable HTTP

struct HttpTransport {
    url: String,
    headers: Vec<(String, String)>,
    client: reqwest::Client,
    session: StdMutex<Option<String>>,
    protocol: StdMutex<Option<String>>,
    next_id: AtomicU64,
    broken: StdMutex<Option<String>>,
}

impl HttpTransport {
    fn new(cfg: &Value) -> Arc<Self> {
        let headers = cfg
            .get("headers")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
            .map(|(k, v)| (k.clone(), v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string())))
            .collect();
        Arc::new(HttpTransport {
            url: cfg.get("url").and_then(Value::as_str).unwrap_or("").to_string(),
            headers,
            client: reqwest::Client::new(),
            session: StdMutex::new(None),
            protocol: StdMutex::new(None),
            next_id: AtomicU64::new(0),
            broken: StdMutex::new(None),
        })
    }

    fn alive(&self) -> bool {
        self.broken.lock().unwrap().is_none()
    }

    fn builder(&self, method: reqwest::Method, timeout: Duration) -> reqwest::RequestBuilder {
        let mut b = self
            .client
            .request(method, &self.url)
            .timeout(timeout)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .header("User-Agent", "strata");
        for (k, v) in &self.headers {
            b = b.header(k, v);
        }
        if let Some(s) = self.session.lock().unwrap().clone() {
            b = b.header("Mcp-Session-Id", s);
        }
        if let Some(p) = self.protocol.lock().unwrap().clone() {
            b = b.header("MCP-Protocol-Version", p);
        }
        b
    }

    fn pick(v: &Value, id: u64) -> Option<Value> {
        let items = match v {
            Value::Array(a) => a.clone(),
            m => vec![m.clone()],
        };
        items.into_iter().find(|m| m.get("id").and_then(Value::as_u64) == Some(id) && m.get("method").is_none())
    }

    async fn post(&self, msg: &Value, id: Option<u64>, timeout: Duration) -> Result<Option<Value>, String> {
        let mut r = self
            .builder(reqwest::Method::POST, timeout)
            .body(serde_json::to_vec(msg).unwrap_or_default())
            .send()
            .await
            .map_err(|e| format!("could not reach {}: {e}", self.url))?;
        if let Some(s) = r.headers().get("Mcp-Session-Id").and_then(|v| v.to_str().ok()) {
            *self.session.lock().unwrap() = Some(s.to_string());
        }
        let status = r.status();
        if !status.is_success() {
            let detail = r.text().await.unwrap_or_default();
            if status.as_u16() == 404 && self.session.lock().unwrap().is_some() {
                *self.broken.lock().unwrap() = Some("the server ended the session".into());
            }
            let detail = &detail[..detail.len().min(300)];
            return Err(format!("HTTP {}{}", status, if detail.is_empty() { String::new() } else { format!(": {detail}") }));
        }
        let Some(id) = id else { return Ok(None) };
        let ctype = r.headers().get("Content-Type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
        if ctype.contains("text/event-stream") {
            // the event stream until the event that answers `id` (notifications may come first)
            let (mut buf, mut data) = (String::new(), Vec::<String>::new());
            while let Some(chunk) = r.chunk().await.map_err(|e| e.to_string())? {
                buf.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(nl) = buf.find('\n') {
                    let line = buf[..nl].trim_end_matches('\r').to_string();
                    buf.drain(..=nl);
                    if let Some(d) = line.strip_prefix("data:") {
                        data.push(d.trim_start_matches(' ').to_string());
                    } else if line.is_empty() && !data.is_empty() {
                        let joined = std::mem::take(&mut data).join("\n");
                        if let Some(m) = serde_json::from_str::<Value>(&joined).ok().and_then(|v| Self::pick(&v, id)) {
                            return Ok(Some(m));
                        }
                    }
                }
            }
            return Err("the server's answer did not contain a response".into());
        }
        let body = r.text().await.map_err(|e| e.to_string())?;
        let v: Value = serde_json::from_str(&body).map_err(|e| format!("the server's answer is not JSON: {e}"))?;
        Self::pick(&v, id).map(Some).ok_or_else(|| "the server's answer did not contain a response".into())
    }

    async fn request(&self, method: &str, params: Value, timeout: Duration) -> Result<Value, String> {
        if let Some(b) = self.broken.lock().unwrap().clone() {
            return Err(b);
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst) + 1;
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        match tokio::time::timeout(timeout, self.post(&msg, Some(id), timeout + Duration::from_secs(5))).await {
            Ok(r) => answer(&r?.unwrap_or(Value::Null)),
            Err(_) => Err(format!("{method} took longer than {:.0} s", timeout.as_secs_f64())),
        }
    }

    async fn notify(&self, method: &str) -> Result<(), String> {
        self.post(&json!({"jsonrpc": "2.0", "method": method}), None, Duration::from_secs(10)).await.map(|_| ())
    }

    async fn close(&self) {
        if self.session.lock().unwrap().is_some() {
            let _ = self.builder(reqwest::Method::DELETE, Duration::from_secs(3)).send().await;
        }
    }
}

#[derive(Clone)]
enum Transport {
    Stdio(Arc<StdioTransport>),
    Http(Arc<HttpTransport>),
}

impl Transport {
    fn alive(&self) -> bool {
        match self {
            Transport::Stdio(t) => t.alive(),
            Transport::Http(t) => t.alive(),
        }
    }
    async fn request(&self, method: &str, params: Value, timeout: Duration) -> Result<Value, String> {
        match self {
            Transport::Stdio(t) => t.request(method, params, timeout).await,
            Transport::Http(t) => t.request(method, params, timeout).await,
        }
    }
    async fn notify(&self, method: &str) -> Result<(), String> {
        match self {
            Transport::Stdio(t) => t.notify(method, None).await,
            Transport::Http(t) => t.notify(method).await,
        }
    }
    async fn close(&self) {
        match self {
            Transport::Stdio(t) => t.close().await,
            Transport::Http(t) => t.close().await,
        }
    }
    fn set_protocol(&self, p: &str) {
        if let Transport::Http(t) = self {
            *t.protocol.lock().unwrap() = Some(p.to_string());
        }
    }
}

// ------------------------------------------------------------------------------------------------ one server

struct ServerState {
    /// idle -> starting -> ready | failed; ready -> stopped when it ends
    status: &'static str,
    error: Option<String>,
    tools: Vec<Value>,
    info: Map<String, Value>,
    transport: Option<Transport>,
    last_start: Option<Instant>,
}

pub struct McpServer {
    pub name: String,
    cfg: Value,
    kind: &'static str,
    state: StdMutex<ServerState>,
    /// one start at a time
    starting: Mutex<()>,
}

impl McpServer {
    fn snapshot(&self) -> (&'static str, Option<String>, Vec<Value>, Map<String, Value>) {
        let s = self.state.lock().unwrap();
        (s.status, s.error.clone(), s.tools.clone(), s.info.clone())
    }

    async fn start(&self, settings: &Settings) -> bool {
        let _one = self.starting.lock().await;
        let old = {
            let mut s = self.state.lock().unwrap();
            s.status = "starting";
            s.error = None;
            s.last_start = Some(Instant::now());
            s.transport.take()
        };
        if let Some(t) = old {
            t.close().await;
        }
        let timeout = Duration::from_secs_f64(settings.start_timeout_s);
        let t = if self.kind == "http" {
            Transport::Http(HttpTransport::new(&self.cfg))
        } else {
            match StdioTransport::start(&self.cfg).await {
                Ok(t) => Transport::Stdio(t),
                Err(e) => return self.failed(e, None).await,
            }
        };
        let init = json!({"protocolVersion": PROTOCOL, "capabilities": {},
                          "clientInfo": {"name": "am2strata", "title": "am2Strata", "version": "1"}});
        let res = match t.request("initialize", init, timeout).await {
            Ok(r) => r,
            Err(e) => return self.failed(e, Some(t)).await,
        };
        let protocol = res.get("protocolVersion").and_then(Value::as_str).unwrap_or(PROTOCOL).to_string();
        t.set_protocol(&protocol);
        let mut info: Map<String, Value> = res
            .get("serverInfo")
            .and_then(Value::as_object)
            .map(|o| o.iter().filter(|(k, _)| matches!(k.as_str(), "name" | "title" | "version")).map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        info.insert("protocol".into(), json!(protocol));
        if let Err(e) = t.notify("notifications/initialized").await {
            return self.failed(e, Some(t)).await;
        }
        let mut tools: Vec<Value> = Vec::new();
        let mut cursor: Option<Value> = None;
        for _ in 0..100 {
            let params = cursor.as_ref().map(|c| json!({"cursor": c})).unwrap_or(json!({}));
            let page = match t.request("tools/list", params, timeout).await {
                Ok(p) => p,
                Err(e) if e.contains("code -32601") && tools.is_empty() => break, // prompts/resources only
                Err(e) => return self.failed(e, Some(t)).await,
            };
            tools.extend(
                page.get("tools").and_then(Value::as_array).into_iter().flatten()
                    .filter(|x| x.get("name").and_then(Value::as_str).is_some_and(|n| !n.is_empty()))
                    .cloned(),
            );
            cursor = page.get("nextCursor").filter(|c| !c.is_null() && c.as_str() != Some("")).cloned();
            if cursor.is_none() {
                break;
            }
        }
        let names: Vec<&str> = tools.iter().take(8).filter_map(|x| x["name"].as_str()).collect();
        tracing::info!("MCP server {:?}: {} tool(s) ({}{})", self.name, tools.len(), names.join(", "), if tools.len() > 8 { ", ..." } else { "" });
        let mut s = self.state.lock().unwrap();
        s.tools = tools;
        s.info = info;
        s.status = "ready";
        s.transport = Some(t);
        true
    }

    async fn failed(&self, e: String, t: Option<Transport>) -> bool {
        if let Some(t) = t {
            t.close().await;
        }
        tracing::warn!("MCP server {:?} did not start: {e}. Its tools are left out; the chat works without them.", self.name);
        let mut s = self.state.lock().unwrap();
        s.status = "failed";
        s.error = Some(e);
        s.tools.clear();
        false
    }

    async fn call(&self, tool: &str, arguments: Value, settings: &Settings) -> Result<Value, String> {
        let t = {
            let s = self.state.lock().unwrap();
            s.transport.clone().filter(|t| t.alive()).ok_or((s.status, s.error.clone(), s.last_start))
        };
        let t = match t {
            Ok(t) => t,
            Err((status, error, last)) => {
                // it stopped since its last call: start it again; one that failed is tried at most every 10 s
                if status == "failed" && last.is_some_and(|l| l.elapsed() < Duration::from_secs(10)) {
                    return Err(format!("the server is not running ({})", error.unwrap_or_else(|| "it stopped".into())));
                }
                tracing::info!("MCP server {:?} had stopped; starting it again", self.name);
                if !self.start(settings).await {
                    let e = self.state.lock().unwrap().error.clone().unwrap_or_default();
                    return Err(format!("the server could not be started again: {e}"));
                }
                self.state.lock().unwrap().transport.clone().ok_or("the server is not running")?
            }
        };
        let r = t.request("tools/call", json!({"name": tool, "arguments": arguments}), Duration::from_secs_f64(settings.timeout_s)).await;
        if let Err(e) = &r {
            if !t.alive() {
                let mut s = self.state.lock().unwrap();
                s.status = "stopped";
                s.error = Some(e.clone());
            }
        }
        r
    }

    async fn close(&self) {
        let t = {
            let mut s = self.state.lock().unwrap();
            if matches!(s.status, "ready" | "starting") {
                s.status = "stopped";
            }
            s.transport.take()
        };
        if let Some(t) = t {
            t.close().await;
        }
    }
}

// ------------------------------------------------------------------------------------------------ the hub

/// An MCP tool result as the text the model reads: the text blocks as they are, other blocks named.
pub fn result_text(result: &Value) -> String {
    let mut parts = Vec::new();
    for block in result.get("content").and_then(Value::as_array).into_iter().flatten() {
        let Some(b) = block.as_object() else { continue };
        let s = |k: &str| b.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        match b.get("type").and_then(Value::as_str) {
            Some("text") => parts.push(s("text")),
            Some("resource") => {
                let res = b.get("resource").cloned().unwrap_or(json!({}));
                parts.push(match res.get("text").and_then(Value::as_str) {
                    Some(t) => t.to_string(),
                    None => format!("[resource {} ({}), not shown]", res["uri"].as_str().unwrap_or(""),
                                    res["mimeType"].as_str().unwrap_or("binary")),
                });
            }
            Some("resource_link") => parts.push(format!("[resource link: {} {}]", s("name"), s("uri")).replace("  ", " ")),
            Some(k @ ("image" | "audio")) => {
                let size = s("data").len() * 3 / 4;
                parts.push(format!("[{k} ({}, {} bytes): not shown to the model]", b.get("mimeType").and_then(Value::as_str).unwrap_or("?"), group(size)));
            }
            _ => parts.push(strata_frontend::output::py_dumps(block, false)),
        }
    }
    if parts.is_empty() {
        if let Some(sc) = result.get("structuredContent").filter(|v| !v.is_null()) {
            parts.push(strata_frontend::output::py_dumps(sc, false));
        }
    }
    parts.join("\n")
}

/// 12345 -> "12,345" (Python's `{:,}`).
fn group(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn clean(name: &str) -> String {
    let c: String = name.chars().map(|ch| if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' { ch } else { '_' }).collect();
    if c.is_empty() { "x".into() } else { c }
}

pub struct McpHub {
    pub settings: Settings,
    pub servers: Vec<Arc<McpServer>>,
}

impl McpHub {
    /// The run config's `"mcp_servers"` / `"mcpServers"` (entries with `"disabled": true` left out) and its `"mcp"`
    /// settings block. None when there are no servers; Err names a bad entry.
    pub fn from_config(rest: &std::collections::BTreeMap<String, Value>) -> Result<Option<Arc<McpHub>>, String> {
        let mut servers: Vec<(String, Value)> = Vec::new();
        for key in ["mcp_servers", "mcpServers"] {
            let Some(block) = rest.get(key).filter(|b| !b.is_null()) else { continue };
            let obj = block.as_object().ok_or(format!("config {key}: the MCP servers must be an object {{\"name\": {{...}}}}"))?;
            for (name, c) in obj {
                if c.get("disabled") == Some(&Value::Bool(true)) {
                    continue;
                }
                check_server(name, c, key)?;
                servers.retain(|(n, _)| n != name);
                servers.push((name.clone(), c.clone()));
            }
        }
        if servers.is_empty() {
            return Ok(None);
        }
        let mut settings = Settings::default();
        for (k, v) in rest.get("mcp").and_then(Value::as_object).into_iter().flatten() {
            let n = v.as_f64().filter(|_| !v.is_boolean());
            match k.as_str() {
                "timeout_s" | "start_timeout_s" => {
                    let x = n.filter(|x| *x > 0.0).ok_or(format!("config mcp.{k}={v}: expected a number of seconds > 0"))?;
                    if k == "timeout_s" { settings.timeout_s = x } else { settings.start_timeout_s = x }
                }
                "max_result_chars" | "max_rounds" => {
                    let x = n.filter(|x| *x >= 1.0 && x.fract() == 0.0).ok_or(format!("config mcp.{k}={v}: expected a positive integer"))?;
                    if k == "max_rounds" { settings.max_rounds = x as usize } else { settings.max_result_chars = x as usize }
                }
                _ => tracing::warn!("config mcp.{k}={v}: unknown key, ignored"),
            }
        }
        let servers = servers
            .into_iter()
            .map(|(name, cfg)| {
                Arc::new(McpServer {
                    kind: if cfg.get("url").is_some_and(|u| !u.is_null()) { "http" } else { "stdio" },
                    name,
                    cfg,
                    state: StdMutex::new(ServerState { status: "idle", error: None, tools: Vec::new(), info: Map::new(), transport: None, last_start: None }),
                    starting: Mutex::new(()),
                })
            })
            .collect();
        Ok(Some(Arc::new(McpHub { settings, servers })))
    }

    /// Start every server in the background (npx may download a package first: that must not delay the chat).
    pub fn start(self: &Arc<Self>) {
        for s in &self.servers {
            let (s, settings) = (s.clone(), self.settings.clone());
            tokio::spawn(async move {
                s.start(&settings).await;
            });
        }
    }

    /// Wait (up to `timeout` in all) for servers that are still starting.
    pub async fn wait(&self, timeout: Duration) {
        let end = Instant::now() + timeout;
        while Instant::now() < end && self.servers.iter().any(|s| matches!(s.snapshot().0, "starting" | "idle")) {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// tool name the model sees -> (server, the tool's own name). A server that stopped keeps its tools: a call
    /// starts it again.
    pub fn routes(&self) -> Vec<(String, Arc<McpServer>, Value)> {
        let mut out: Vec<(String, Arc<McpServer>, Value)> = Vec::new();
        for s in &self.servers {
            let (status, _, tools, _) = s.snapshot();
            if !matches!(status, "ready" | "stopped") {
                continue;
            }
            for t in tools {
                let tool = t["name"].as_str().unwrap_or("");
                let mut name: String = format!("{}__{}", clean(&s.name), clean(tool)).chars().take(64).collect();
                let mut n = 2;
                while out.iter().any(|(x, _, _)| *x == name) {
                    name = format!("{}_{n}", name.chars().take(60).collect::<String>());
                    n += 1;
                }
                out.push((name, s.clone(), t));
            }
        }
        out
    }

    /// The tools in the chat template's form; `exclude`: names the request brought itself (those win).
    pub fn template_tools(&self, exclude: &[String]) -> Vec<Value> {
        self.routes()
            .into_iter()
            .filter(|(name, _, _)| !exclude.contains(name))
            .map(|(name, _, t)| {
                let schema = t.get("inputSchema").filter(|s| s.as_object().is_some_and(|o| !o.is_empty())).cloned();
                let desc = t.get("description").and_then(Value::as_str).filter(|d| !d.is_empty())
                    .or_else(|| t.get("title").and_then(Value::as_str)).unwrap_or("");
                json!({"name": name, "description": desc, "parameters": schema.unwrap_or(json!({"type": "object", "properties": {}}))})
            })
            .collect()
    }

    /// Run one tool -> {"ok", "text" (capped), "chars", "truncated", "ms", "server", "tool"}. Errors become text that
    /// starts with "error:".
    pub async fn call(&self, name: &str, arguments: &Value) -> Value {
        let t0 = Instant::now();
        let route = self.routes().into_iter().find(|(n, _, _)| n == name);
        let (server, tool) = match &route {
            Some((_, s, t)) => (Some(s.clone()), t["name"].as_str().unwrap_or(name).to_string()),
            None => (None, name.to_string()),
        };
        let (text, ok) = match &server {
            None => (format!("error: there is no tool named '{name}'"), false),
            Some(s) => {
                let args = if arguments.is_object() { arguments.clone() } else { json!({}) };
                match s.call(&tool, args, &self.settings).await {
                    Ok(res) => {
                        let text = result_text(&res);
                        if res.get("isError") == Some(&Value::Bool(true)) {
                            (format!("error: {}", if text.is_empty() { "the tool reported an error".to_string() } else { text }), false)
                        } else {
                            (text, true)
                        }
                    }
                    Err(e) => (format!("error: {e}"), false),
                }
            }
        };
        let cap = self.settings.max_result_chars;
        let full = text.chars().count();
        let truncated = cap > 0 && full > cap;
        let text = if truncated {
            format!("{}\n\n[... truncated: the tool returned {} characters; only the first {} are shown]",
                    text.chars().take(cap).collect::<String>(), group(full), group(cap))
        } else {
            text
        };
        json!({"server": server.map(|s| s.name.clone()), "tool": tool, "ok": ok, "text": text, "chars": full,
               "truncated": truncated, "ms": t0.elapsed().as_millis() as u64})
    }

    /// GET /mcp: every server, its state and its tools.
    pub fn status(&self) -> Value {
        let routes = self.routes();
        let servers: Vec<Value> = self
            .servers
            .iter()
            .map(|s| {
                let (status, error, tools, info) = s.snapshot();
                let tools: Vec<Value> = tools
                    .iter()
                    .map(|t| {
                        let tool = t["name"].as_str().unwrap_or("");
                        let name = routes.iter().find(|(_, srv, x)| Arc::ptr_eq(srv, s) && x["name"] == tool).map(|(n, _, _)| n.as_str()).unwrap_or(tool);
                        let desc: String = t.get("description").and_then(Value::as_str).unwrap_or("").chars().take(300).collect();
                        json!({"name": name, "tool": tool, "description": desc})
                    })
                    .collect();
                json!({"name": s.name, "transport": s.kind, "status": status, "error": error, "info": info, "tools": tools})
            })
            .collect();
        json!({"servers": servers, "tools": routes.len(),
               "settings": {"timeout_s": self.settings.timeout_s, "max_result_chars": self.settings.max_result_chars,
                            "max_rounds": self.settings.max_rounds}})
    }

    pub async fn close(&self) {
        for s in &self.servers {
            s.close().await;
        }
    }
}

fn check_server(name: &str, c: &Value, key: &str) -> Result<(), String> {
    let Some(o) = c.as_object() else {
        return Err(format!("config {key}: MCP server {name:?} must be an object with \"command\" or \"url\""));
    };
    if o.get("url").is_some_and(|u| !u.is_null()) {
        if o.get("type").and_then(Value::as_str) == Some("sse") {
            return Err(format!("config {key}: MCP server {name:?} uses the old SSE transport (\"type\": \"sse\"); \
                                Streamable HTTP is spoken here - most servers offer it at /mcp"));
        }
        if o.get("headers").is_some_and(|h| !h.is_null() && !h.is_object()) {
            return Err(format!("config {key}: MCP server {name:?}: \"headers\" must be an object"));
        }
        return Ok(());
    }
    if !o.get("command").and_then(Value::as_str).is_some_and(|c| !c.trim().is_empty()) {
        return Err(format!("config {key}: MCP server {name:?} needs \"command\" (a program to start) or \"url\""));
    }
    if o.get("args").is_some_and(|a| !a.is_null() && !a.is_array()) {
        return Err(format!("config {key}: MCP server {name:?}: \"args\" must be a list"));
    }
    if o.get("env").is_some_and(|e| !e.is_null() && !e.is_object()) {
        return Err(format!("config {key}: MCP server {name:?}: \"env\" must be an object"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_become_text() {
        let r = json!({"content": [{"type": "text", "text": "a"}, {"type": "image", "mimeType": "image/png", "data": "AAAA"},
                                   {"type": "resource", "resource": {"uri": "f://x", "text": "body"}}]});
        assert_eq!(result_text(&r), "a\n[image (image/png, 3 bytes): not shown to the model]\nbody");
        assert_eq!(result_text(&json!({"content": [], "structuredContent": {"k": 1}})), "{\"k\": 1}");
        assert_eq!(group(1234567), "1,234,567");
        assert_eq!(clean("my server.v2"), "my_server_v2");
    }

    #[test]
    fn config_entries_are_checked() {
        let mut rest = std::collections::BTreeMap::new();
        rest.insert("mcp_servers".into(), json!({"a": {"command": "x"}, "b": {"url": "http://h/mcp"}, "c": {"command": "y", "disabled": true}}));
        let hub = McpHub::from_config(&rest).unwrap().unwrap();
        assert_eq!(hub.servers.len(), 2);
        rest.insert("mcp_servers".into(), json!({"a": {"url": "http://h", "type": "sse"}}));
        assert!(McpHub::from_config(&rest).is_err());
    }
}
