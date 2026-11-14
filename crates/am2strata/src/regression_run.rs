//! `am2strata-run regression`: a read-mostly checklist against a RUNNING server (docs/REGRESSION.md), for after an
//! update: is it up, does every API still answer in its own shape, are the Monitor, the queue, HTTPS, the Workshop,
//! the Computer tab and the sandbox VM's status there. A pass/fail table with timings; a non-zero exit on any failure.
//!
//! - `--quick` (the default): changes nothing on the server and makes at most 3 short model calls (a chat, a streamed
//!   chat, an Anthropic message); the other APIs are checked through calls that generate nothing (token counts, lists).
//! - `--full`: also one real call to every compatibility API and a tool round trip (about 8 short model calls). It
//!   still changes no setting, starts no job, and never starts the VM (status only).
//!
//! The API key is never printed: not in the table, not in `--json`, not in an error (anything the server echoes is
//! masked).

use serde_json::{json, Value};
use std::time::{Duration, Instant};

pub struct Options {
    pub url: String,
    pub key: Option<String>,
    pub full: bool,
    pub json: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Outcome {
    Pass,
    Fail,
    Skip,
}

struct Row {
    name: &'static str,
    outcome: Outcome,
    ms: u128,
    detail: String,
}

/// What a check found: Ok(what was seen), or why it failed, or why it does not apply to this server.
enum Check {
    Ok(String),
    Fail(String),
    Skip(String),
}

struct Answer {
    status: u16,
    headers: reqwest::header::HeaderMap,
    text: String,
}

impl Answer {
    fn json(&self) -> Value {
        serde_json::from_str(&self.text).unwrap_or(Value::Null)
    }

    /// The server's own words for a refusal, short.
    fn why(&self) -> String {
        let j = self.json();
        let m = j["error"]["message"].as_str().map(str::to_string).unwrap_or_else(|| self.text.chars().take(120).collect());
        format!("HTTP {}: {}", self.status, m.chars().take(160).collect::<String>())
    }

    fn events(&self) -> Vec<(String, Value)> {
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

/// How the key is sent: each API family's own header.
#[derive(Clone, Copy)]
enum Auth {
    None,
    Bearer,
    XApiKey,
    Goog,
    Wrong,
}

struct Runner {
    url: String,
    key: Option<String>,
    http: reqwest::blocking::Client,
    model_calls: usize,
}

/// A model call may wait for the model to load and for its turn; everything else answers at once.
const MODEL_TIMEOUT: Duration = Duration::from_secs(600);
const QUICK_TIMEOUT: Duration = Duration::from_secs(30);

impl Runner {
    fn send(&self, method: &str, path: &str, auth: Auth, body: Option<&Value>, timeout: Duration) -> Result<Answer, String> {
        let m: reqwest::Method = method.parse().map_err(|_| format!("method {method}"))?;
        let mut rb = self.http.request(m, format!("{}{path}", self.url)).timeout(timeout);
        match (auth, &self.key) {
            (Auth::Bearer, Some(k)) => rb = rb.bearer_auth(k),
            (Auth::XApiKey, Some(k)) => rb = rb.header("x-api-key", k).header("anthropic-version", "2023-06-01"),
            (Auth::Goog, Some(k)) => rb = rb.header("x-goog-api-key", k),
            (Auth::Wrong, _) => rb = rb.bearer_auth("am2strata-regression-not-a-key"),
            _ => {}
        }
        if let Some(b) = body {
            rb = rb.json(b);
        }
        let r = rb.send().map_err(|e| format!("no answer ({})", self.mask(&e.to_string())))?;
        Ok(Answer { status: r.status().as_u16(), headers: r.headers().clone(), text: r.text().unwrap_or_default() })
    }

    fn get(&self, path: &str) -> Result<Answer, String> {
        self.send("GET", path, Auth::Bearer, None, QUICK_TIMEOUT)
    }

    /// A call that makes the model write (counted: `--quick` allows three).
    fn model(&mut self, path: &str, auth: Auth, body: &Value) -> Result<Answer, String> {
        self.model_calls += 1;
        self.send("POST", path, auth, Some(body), MODEL_TIMEOUT)
    }

    /// The key never leaves this program in its output: anything that might hold it is masked.
    fn mask(&self, s: &str) -> String {
        match self.key.as_deref().filter(|k| k.len() >= 4) {
            Some(k) => s.replace(k, "***"),
            None => s.to_string(),
        }
    }
}

fn ok200(a: Result<Answer, String>) -> Result<Answer, Check> {
    match a {
        Err(e) => Err(Check::Fail(e)),
        Ok(a) if a.status == 200 => Ok(a),
        Ok(a) => Err(Check::Fail(a.why())),
    }
}

/// 200 with JSON, or Skip when the server has no such route (an older version, another platform).
fn ok200_or_absent(a: Result<Answer, String>, what: &str) -> Result<Answer, Check> {
    match a {
        Ok(a) if a.status == 404 || a.status == 405 => Err(Check::Skip(format!("this server has no {what}"))),
        other => ok200(other),
    }
}

macro_rules! tri {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(c) => return c,
        }
    };
}

const WEATHER: &str = r#"{"type": "function", "function": {"name": "get_weather", "description": "The weather in a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}"#;

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

const PROMPT: &str = "Reply with the single word: ok";

// ------------------------------------------------------------------------------------------------------- the checks

fn health(r: &mut Runner) -> Check {
    let a = tri!(ok200(r.send("GET", "/health", Auth::None, None, QUICK_TIMEOUT)));
    let j = a.json();
    if j["status"] != "ok" {
        return Check::Fail(format!("status is {}", j["status"]));
    }
    Check::Ok(format!("{}, model {} ({})", j["server"].as_str().unwrap_or("?"), j["model"].as_str().unwrap_or("?"),
                      if j["loaded"] == true { "loaded" } else { "not loaded" }))
}

fn key_is_needed(r: &mut Runner) -> Check {
    if r.key.is_none() {
        return Check::Skip("no API key was given, so this is not checked".into());
    }
    let none = tri!(r.send("GET", "/v1/models", Auth::None, None, QUICK_TIMEOUT).map_err(Check::Fail));
    let wrong = tri!(r.send("GET", "/v1/models", Auth::Wrong, None, QUICK_TIMEOUT).map_err(Check::Fail));
    let settings = tri!(r.send("GET", "/settings", Auth::Wrong, None, QUICK_TIMEOUT).map_err(Check::Fail));
    if none.status == 200 && wrong.status == 200 {
        return Check::Fail("the API answers without a key: this server has no API key set".into());
    }
    if (none.status, wrong.status, settings.status) != (401, 401, 401) {
        return Check::Fail(format!("without a key: {}, with a wrong key: {}, settings with a wrong key: {} (401 each expected)", none.status, wrong.status, settings.status));
    }
    Check::Ok("401 without a key and with a wrong one".into())
}

fn models(r: &mut Runner) -> Check {
    let a = tri!(ok200(r.get("/v1/models")));
    let j = a.json();
    let n = j["data"].as_array().map(Vec::len).unwrap_or(0);
    if j["object"] != "list" || n == 0 {
        return Check::Fail(format!("no model in the list: {}", a.text.chars().take(120).collect::<String>()));
    }
    let all = tri!(ok200(r.get("/api/models"))).json()["models"].as_array().map(Vec::len).unwrap_or(0);
    Check::Ok(format!("{n} reachable now ({}), {all} installed", j["data"][0]["id"].as_str().unwrap_or("?")))
}

fn chat(r: &mut Runner) -> Check {
    let a = tri!(ok200(r.model("/v1/chat/completions", Auth::Bearer, &json!({"messages": [user(PROMPT)], "max_tokens": 64}))));
    let j = a.json();
    let m = &j["choices"][0]["message"];
    if j["object"] != "chat.completion" || m["role"] != "assistant" || !(m["content"].is_string() || m["reasoning_content"].is_string()) {
        return Check::Fail(format!("not a chat completion: {}", a.text.chars().take(160).collect::<String>()));
    }
    let u = &j["usage"];
    let (p, c, t) = (u["prompt_tokens"].as_u64(), u["completion_tokens"].as_u64(), u["total_tokens"].as_u64());
    match (p, c, t) {
        (Some(p), Some(c), Some(t)) if p > 0 && p + c == t => Check::Ok(format!("{c} tokens, finish {}", j["choices"][0]["finish_reason"].as_str().unwrap_or("?"))),
        _ => Check::Fail(format!("the usage block is wrong: {u}")),
    }
}

fn chat_stream(r: &mut Runner) -> Check {
    let a = tri!(ok200(r.model("/v1/chat/completions", Auth::Bearer, &json!({"messages": [user(PROMPT)], "max_tokens": 64, "stream": true}))));
    if !a.headers.get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|c| c.starts_with("text/event-stream")) {
        return Check::Fail("not an event stream".into());
    }
    let ev = a.events();
    let chunks = ev.iter().filter(|(_, v)| v["object"] == "chat.completion.chunk").count();
    let text: usize = ev.iter().filter_map(|(_, v)| v["choices"][0]["delta"]["content"].as_str().or(v["choices"][0]["delta"]["reasoning_content"].as_str())).map(str::len).sum();
    let finished = ev.iter().any(|(_, v)| v["choices"][0]["finish_reason"].is_string());
    if chunks == 0 || text == 0 || !finished || ev.last().map(|(_, v)| v) != Some(&json!("[DONE]")) {
        return Check::Fail(format!("{chunks} chunks, {text} characters, finish reason {finished}, ends with [DONE]: {}", ev.last().map(|(_, v)| v == "[DONE]").unwrap_or(false)));
    }
    Check::Ok(format!("{chunks} chunks, then [DONE]"))
}

fn messages(r: &mut Runner) -> Check {
    let a = tri!(ok200(r.model("/v1/messages", Auth::XApiKey, &json!({"model": "claude", "max_tokens": 64, "messages": [user(PROMPT)]}))));
    let j = a.json();
    let blocks = j["content"].as_array().map(Vec::len).unwrap_or(0);
    if j["type"] != "message" || j["role"] != "assistant" || blocks == 0 || !j["usage"]["output_tokens"].is_u64() || !j["stop_reason"].is_string() {
        return Check::Fail(format!("not a message: {}", a.text.chars().take(160).collect::<String>()));
    }
    Check::Ok(format!("{blocks} content block(s), stop {}", j["stop_reason"].as_str().unwrap_or("?")))
}

fn count_tokens(r: &mut Runner) -> Check {
    let a = tri!(ok200(r.send("POST", "/v1/messages/count_tokens", Auth::XApiKey, Some(&json!({"model": "claude", "messages": [user(PROMPT)]})), MODEL_TIMEOUT)));
    match a.json()["input_tokens"].as_u64() {
        Some(n) if n > 0 => Check::Ok(format!("{n} input tokens")),
        _ => Check::Fail(format!("no input_tokens: {}", a.text.chars().take(120).collect::<String>())),
    }
}

fn responses_stateless(r: &mut Runner) -> Check {
    let a = tri!(r.get("/v1/responses/resp_regression").map_err(Check::Fail));
    if a.status != 404 || !a.json()["error"]["message"].is_string() {
        return Check::Fail(format!("GET /v1/responses/<id> should be a 404 with an error body (nothing is stored): {}", a.why()));
    }
    Check::Ok("answers, and stores nothing (404 for a stored id)".into())
}

fn responses(r: &mut Runner) -> Check {
    let a = tri!(ok200(r.model("/v1/responses", Auth::Bearer, &json!({"input": PROMPT, "max_output_tokens": 64}))));
    let j = a.json();
    let items = j["output"].as_array().map(Vec::len).unwrap_or(0);
    if j["object"] != "response" || j["status"] != "completed" || items == 0 || !j["usage"]["output_tokens"].is_u64() {
        return Check::Fail(format!("not a response: {}", a.text.chars().take(160).collect::<String>()));
    }
    Check::Ok(format!("{items} output item(s)"))
}

fn gemini_lists(r: &mut Runner) -> Check {
    let a = tri!(ok200(r.send("GET", "/v1beta/models", Auth::Goog, None, QUICK_TIMEOUT)));
    let n = a.json()["models"].as_array().map(Vec::len).unwrap_or(0);
    if n == 0 {
        return Check::Fail("no model in Gemini's list".into());
    }
    let body = json!({"contents": [{"role": "user", "parts": [{"text": PROMPT}]}]});
    let c = tri!(ok200(r.send("POST", "/v1beta/models/gemini-2.5-pro:countTokens", Auth::Goog, Some(&body), MODEL_TIMEOUT)));
    match c.json()["totalTokens"].as_u64() {
        Some(t) if t > 0 => Check::Ok(format!("{n} model(s), countTokens {t}")),
        _ => Check::Fail(format!("countTokens: {}", c.text.chars().take(120).collect::<String>())),
    }
}

fn gemini(r: &mut Runner) -> Check {
    let body = json!({"contents": [{"role": "user", "parts": [{"text": PROMPT}]}], "generationConfig": {"maxOutputTokens": 64}});
    let a = tri!(ok200(r.model("/v1beta/models/gemini-2.5-pro:generateContent", Auth::Goog, &body)));
    let j = a.json();
    let parts = j["candidates"][0]["content"]["parts"].as_array().map(Vec::len).unwrap_or(0);
    if parts == 0 || !j["candidates"][0]["finishReason"].is_string() || !j["usageMetadata"]["totalTokenCount"].is_u64() {
        return Check::Fail(format!("not a Gemini answer: {}", a.text.chars().take(160).collect::<String>()));
    }
    Check::Ok(format!("{parts} part(s), finish {}", j["candidates"][0]["finishReason"].as_str().unwrap_or("?")))
}

fn tool_round_trip(r: &mut Runner) -> Check {
    let tool: Value = serde_json::from_str(WEATHER).expect("the tool's JSON");
    let ask = user("What is the weather in Pune? Use the tool.");
    let first = json!({"messages": [ask.clone()], "tools": [tool.clone()], "max_tokens": 200,
                       "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                       "chat_template_kwargs": {"enable_thinking": false}});
    let a = tri!(ok200(r.model("/v1/chat/completions", Auth::Bearer, &first)));
    let j = a.json();
    let call = j["choices"][0]["message"]["tool_calls"][0].clone();
    let args = call["function"]["arguments"].as_str().and_then(|s| serde_json::from_str::<Value>(s).ok());
    if j["choices"][0]["finish_reason"] != "tool_calls" || call["function"]["name"] != "get_weather" || !args.as_ref().is_some_and(Value::is_object) {
        return Check::Fail(format!("the forced call did not come back as a tool call: {}", a.text.chars().take(200).collect::<String>()));
    }
    let second = json!({"messages": [ask, {"role": "assistant", "content": null, "tool_calls": [call.clone()]},
                                     {"role": "tool", "tool_call_id": call["id"], "content": "31 C, clear sky"}],
                        "tools": [tool], "max_tokens": 64});
    let b = tri!(ok200(r.model("/v1/chat/completions", Auth::Bearer, &second)));
    let m = &b.json()["choices"][0]["message"];
    if !(m["content"].is_string() || m["reasoning_content"].is_string() || m["tool_calls"].is_array()) {
        return Check::Fail(format!("no answer after the tool's result: {}", b.text.chars().take(160).collect::<String>()));
    }
    Check::Ok(format!("get_weather({}) called, its result answered", args.unwrap_or_default()))
}

fn metrics(r: &mut Runner) -> Check {
    let a = tri!(ok200(r.get("/metrics?per=1")));
    let j = a.json();
    if !j["version"].is_string() || !j["queue"].is_object() {
        return Check::Fail("no version or queue in /metrics".into());
    }
    let p = tri!(ok200(r.get("/metrics/prometheus")));
    if !p.text.contains("# ") {
        return Check::Fail("/metrics/prometheus is not Prometheus text".into());
    }
    Check::Ok(format!("version {}, Prometheus text {} lines", j["version"].as_str().unwrap_or("?"), p.text.lines().count()))
}

fn queue(r: &mut Runner) -> Check {
    let a = tri!(ok200(r.get("/api/queue")));
    let j = a.json();
    if !j.is_object() {
        return Check::Fail("not a JSON object".into());
    }
    let waiting = j["waiting"].as_array().map(Vec::len).or(j["waiting"].as_u64().map(|n| n as usize));
    Check::Ok(match waiting { Some(n) => format!("{n} waiting"), None => "answers".into() })
}

fn status_and_system(r: &mut Runner) -> Check {
    tri!(ok200(r.get("/api/system")));
    tri!(ok200(r.get("/v1/status")));
    let cur = tri!(ok200(r.get("/api/models/current"))).json();
    Check::Ok(format!("current model: {}", cur["model"]["id"].as_str().unwrap_or("none loaded")))
}

fn web_app(r: &mut Runner) -> Check {
    let a = tri!(ok200(r.send("GET", "/", Auth::None, None, QUICK_TIMEOUT)));
    let h = |k: &str| a.headers.get(k).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    if !h("content-type").starts_with("text/html") || a.text.trim().is_empty() {
        return Check::Fail(format!("/ is not the web app's page (content type {:?}, {} bytes)", h("content-type"), a.text.len()));
    }
    if h("x-frame-options") != "SAMEORIGIN" || h("x-content-type-options") != "nosniff" {
        return Check::Fail("the page lacks its security headers (x-frame-options, x-content-type-options)".into());
    }
    Check::Ok(format!("{} KB, security headers set", a.text.len() / 1024))
}

fn https(r: &mut Runner) -> Check {
    let a = tri!(ok200_or_absent(r.get("/api/tls"), "HTTPS settings (/api/tls)"));
    let j = a.json();
    let over_tls = r.url.starts_with("https://");
    // plain HTTP never carries HSTS
    if !over_tls {
        let plain = tri!(ok200(r.send("GET", "/health", Auth::None, None, QUICK_TIMEOUT)));
        if plain.headers.contains_key("strict-transport-security") {
            return Check::Fail("plain HTTP carries an HSTS header".into());
        }
    }
    if j["on"] != true {
        return if over_tls { Check::Fail("reached over https://, yet /api/tls says HTTPS is off".into()) }
               else { Check::Ok("HTTPS is off; plain HTTP carries no HSTS".into()) };
    }
    let days = j["certificate"]["expires_in_days"].as_i64();
    if days.is_some_and(|d| d < 1) {
        return Check::Fail(format!("the certificate has expired ({} days)", days.unwrap_or(0)));
    }
    // the HTTPS port itself: /health, no key sent; the certificate is not checked here (a self-signed one is usual), its
    // facts come from /api/tls
    let port = j["https_port"].as_u64().unwrap_or(8001);
    let host = r.url.split("://").nth(1).unwrap_or("").split('/').next().unwrap_or("");
    let name = if host.starts_with('[') { host.split(']').next().map(|h| format!("{h}]")).unwrap_or_default() }
               else { host.rsplit_once(':').map(|(n, _)| n).unwrap_or(host).to_string() };
    let probe = match reqwest::blocking::Client::builder().danger_accept_invalid_certs(true).timeout(QUICK_TIMEOUT).build() {
        Ok(c) => c,
        Err(e) => return Check::Fail(e.to_string()),
    };
    let resp = match probe.get(format!("https://{name}:{port}/health")).send() {
        Ok(x) => x,
        Err(e) => return Check::Fail(format!("HTTPS is on, but https://{name}:{port}/health does not answer ({e})")),
    };
    let hsts = resp.headers().get("strict-transport-security").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    if resp.status().as_u16() != 200 || hsts.is_empty() {
        return Check::Fail(format!("https://{name}:{port}/health: HTTP {}, HSTS {hsts:?} (an HSTS header is expected over TLS)", resp.status().as_u16()));
    }
    Check::Ok(format!("on, {} certificate, port {port}, {} days left, HSTS \"{hsts}\", plain HTTP: {}",
                      j["mode"].as_str().unwrap_or("?"), days.map(|d| d.to_string()).unwrap_or_else(|| "?".into()), j["http"].as_str().unwrap_or("?")))
}

fn workshop(r: &mut Runner) -> Check {
    let a = tri!(ok200_or_absent(r.get("/v1/workshop/files"), "Workshop"));
    let j = a.json();
    let n = |k: &str| j[k].as_array().map(Vec::len);
    match (n("gguf"), n("hf_models"), n("adapters")) {
        (Some(g), Some(h), Some(ad)) => {
            let jobs = tri!(ok200(r.get("/v1/workshop/jobs"))).json()["jobs"].as_array().map(Vec::len).unwrap_or(0);
            Check::Ok(format!("{g} GGUF, {h} HF models, {ad} adapters, {jobs} jobs; command {}", if j["installed"] == true { "installed" } else { "NOT installed" }))
        }
        _ => Check::Fail("the files listing lacks gguf / hf_models / adapters".into()),
    }
}

fn computer(r: &mut Runner) -> Check {
    let a = tri!(ok200_or_absent(r.get("/api/computer/status"), "Computer tab"));
    let j = a.json();
    if !j["on"].is_boolean() || !j["mode"].is_string() {
        return Check::Fail("the status lacks on / mode".into());
    }
    Check::Ok(format!("computer use {}, mode {}", if j["on"] == true { "on" } else { "off" }, j["mode"].as_str().unwrap_or("?")))
}

/// Status only: this never starts, stops or changes the VM.
fn vm(r: &mut Runner) -> Check {
    let a = tri!(ok200_or_absent(r.get("/api/vm"), "sandbox VM (Linux only)"));
    let j = a.json();
    match j["phase"].as_str() {
        Some(p) => Check::Ok(format!("phase {p}, image {}", if j["image"] == true { "downloaded" } else { "not downloaded" })),
        None => Check::Fail("the status has no phase".into()),
    }
}

fn usage(r: &mut Runner) -> Check {
    let a = tri!(ok200_or_absent(r.get("/v1/usage?days=1"), "usage counts (/v1/usage)"));
    if !a.json().is_object() {
        return Check::Fail("not a JSON object".into());
    }
    Check::Ok("answers".into())
}

/// The live channel is a WebSocket: a plain request with the key is told so (426). No socket is opened here.
fn live_channel(r: &mut Runner) -> Check {
    let a = tri!(r.get("/v1/live").map_err(Check::Fail));
    match a.status {
        426 => Check::Ok("there (a plain request is told to use a WebSocket)".into()),
        404 | 405 => Check::Skip("this server has no live channel (/v1/live)".into()),
        _ => Check::Fail(a.why()),
    }
}

fn media_and_tools(r: &mut Runner) -> Check {
    let caps = tri!(ok200(r.get("/api/media/capabilities")));
    let mcp = tri!(ok200(r.get("/mcp")));
    let mem = tri!(ok200(r.get("/api/memory")));
    if !caps.json().is_object() || !mcp.json().is_object() || !mem.json().is_object() {
        return Check::Fail("media capabilities, MCP or memory did not answer with JSON".into());
    }
    Check::Ok("media capabilities, MCP servers and memory answer".into())
}

fn unknown_is_404(r: &mut Runner) -> Check {
    let a = tri!(r.get("/v1/regression-no-such-route").map_err(Check::Fail));
    let b = tri!(r.send("POST", "/v1/chat/completions", Auth::Bearer, Some(&json!({"messages": "not a list"})), QUICK_TIMEOUT).map_err(Check::Fail));
    if a.status != 404 {
        return Check::Fail(format!("an unknown path gives {}", a.status));
    }
    if b.status != 400 || !b.json()["error"]["message"].is_string() {
        return Check::Fail(format!("a wrong request should be a 400 with error.message: {}", b.why()));
    }
    Check::Ok("unknown path 404; a wrong request 400 with error.message".into())
}

type CheckFn = fn(&mut Runner) -> Check;

/// The checklist, in order: (name, only with --full, the check).
const CHECKS: &[(&str, bool, CheckFn)] = &[
    ("health", false, health),
    ("the API key is needed", false, key_is_needed),
    ("models list", false, models),
    ("chat (OpenAI)", false, chat),
    ("chat, streamed", false, chat_stream),
    ("messages (Anthropic)", false, messages),
    ("count_tokens (Anthropic)", false, count_tokens),
    ("responses: stateless", false, responses_stateless),
    ("responses (OpenAI)", true, responses),
    ("Gemini: models, countTokens", false, gemini_lists),
    ("Gemini: generateContent", true, gemini),
    ("tool round trip", true, tool_round_trip),
    ("errors: 404 and 400", false, unknown_is_404),
    ("metrics", false, metrics),
    ("queue", false, queue),
    ("status and system", false, status_and_system),
    ("web app", false, web_app),
    ("HTTPS and HSTS", false, https),
    ("Workshop files", false, workshop),
    ("Computer status", false, computer),
    ("VM status", false, vm),
    ("usage", false, usage),
    ("live channel", false, live_channel),
    ("media, MCP, memory", true, media_and_tools),
];

/// The most model calls `--quick` may make.
pub const QUICK_MODEL_CALLS: usize = 3;

/// Run the checklist and print it. True when nothing failed.
pub fn run(o: Options) -> bool {
    let http = match reqwest::blocking::Client::builder().redirect(reqwest::redirect::Policy::none())
        // a self-signed certificate is the usual one on a home server: the same trust the browser was given by hand
        .danger_accept_invalid_certs(o.url.starts_with("https://")).build() {
        Ok(c) => c,
        Err(e) => { eprintln!("regression: {e}"); return false; }
    };
    let mut r = Runner { url: o.url.trim_end_matches('/').to_string(), key: o.key.filter(|k| !k.is_empty()), http, model_calls: 0 };
    let mut rows: Vec<Row> = Vec::new();
    let started = Instant::now();
    for (name, full_only, check) in CHECKS {
        if *full_only && !o.full {
            continue;
        }
        let t0 = Instant::now();
        let (outcome, detail) = match check(&mut r) {
            Check::Ok(d) => (Outcome::Pass, d),
            Check::Fail(d) => (Outcome::Fail, d),
            Check::Skip(d) => (Outcome::Skip, d),
        };
        rows.push(Row { name, outcome, ms: t0.elapsed().as_millis(), detail: r.mask(&detail) });
        // a server that does not answer at all: the rest would only repeat it
        if *name == "health" && outcome == Outcome::Fail {
            break;
        }
    }
    if !o.full && r.model_calls > QUICK_MODEL_CALLS {
        rows.push(Row { name: "quick mode's budget", outcome: Outcome::Fail, ms: 0,
                        detail: format!("{} model calls were made; --quick allows {QUICK_MODEL_CALLS}", r.model_calls) });
    }
    let count = |o: Outcome| rows.iter().filter(|r| r.outcome == o).count();
    let (passed, failed, skipped) = (count(Outcome::Pass), count(Outcome::Fail), count(Outcome::Skip));
    let word = |o: Outcome| match o { Outcome::Pass => "pass", Outcome::Fail => "FAIL", Outcome::Skip => "skip" };
    if o.json {
        let v = json!({"url": r.mask(&r.url), "mode": if o.full { "full" } else { "quick" }, "passed": passed, "failed": failed, "skipped": skipped,
                       "model_calls": r.model_calls, "seconds": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0,
                       "rows": rows.iter().map(|x| json!({"check": x.name, "result": word(x.outcome).to_lowercase(), "ms": x.ms as u64, "detail": x.detail})).collect::<Vec<_>>()});
        println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
    } else {
        println!("am2Strata regression ({}) against {}", if o.full { "full" } else { "quick" }, r.mask(&r.url));
        let w = rows.iter().map(|x| x.name.chars().count()).max().unwrap_or(10);
        for x in &rows {
            println!("  {}  {:<w$}  {:>7} ms  {}", word(x.outcome), x.name, x.ms, x.detail);
        }
        println!("{passed} passed, {failed} failed, {skipped} skipped; {} model calls; {:.1} s", r.model_calls, started.elapsed().as_secs_f64());
    }
    failed == 0
}
