//! Remote models (docs/internal/REMOTE-MODELS.md) end to end, with no network: a mock OpenAI-compatible and a mock
//! Anthropic-compatible provider (small axum servers on 127.0.0.1, run here) that record what they receive. Off,
//! nothing changes; on, every client API reaches both provider APIs, streamed and not, with the server's own tool
//! loop, the usage counts, the key's permissions and the mix; the provider's key is never in an answer, the log or
//! the API monitor.

mod common;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use common::*;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const PROVIDER_KEY: &str = "sk-test-PROVIDERSECRET-0123456789abcdef";

// ------------------------------------------------------------------ the mock providers

#[derive(Clone, Default)]
struct Mock {
    seen: Arc<Mutex<Vec<Value>>>,
    /// a slow answer's connection was closed (the client left and the server ended the call)
    closed: Arc<AtomicBool>,
}

impl Mock {
    fn bodies(&self) -> Vec<Value> {
        self.seen.lock().unwrap().iter().filter(|v| v["path"] != "/v1/models").map(|v| v["body"].clone()).collect()
    }
}

struct Closed(Arc<AtomicBool>);
impl Drop for Closed {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn record(m: &Mock, path: &str, headers: &HeaderMap, body: &Value) -> Result<(), Response> {
    let auth = headers.get("authorization").or_else(|| headers.get("x-api-key")).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    m.seen.lock().unwrap().push(json!({"path": path, "auth": auth, "body": body}));
    if !auth.ends_with(PROVIDER_KEY) {
        return Err((StatusCode::UNAUTHORIZED, Json(json!({"error": {"message": format!("Incorrect API key provided: {}***", &auth.chars().take(12).collect::<String>())}}))).into_response());
    }
    Ok(())
}

fn sse(events: Vec<String>) -> Response {
    Response::builder().header("content-type", "text/event-stream").body(Body::from(events.concat())).unwrap()
}

/// A stream that writes its first event, then only keep-alives until the client goes.
fn slow(first: String, closed: Arc<AtomicBool>) -> Response {
    let guard = Closed(closed);
    let s = futures::stream::unfold((0u32, guard), move |(n, g)| {
        let first = first.clone();
        async move {
            if n > 0 {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            if n > 600 {
                return None;
            }
            let chunk = if n == 0 { first } else { ": ping\n\n".to_string() };
            Some((Ok::<_, std::convert::Infallible>(chunk), (n + 1, g)))
        }
    });
    Response::builder().header("content-type", "text/event-stream").body(Body::from_stream(s)).unwrap()
}

fn failure(model: &str) -> Option<Response> {
    match model {
        "fail500" => Some((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": {"message": "the mock fell over"}}))).into_response()),
        "fail429" => Some(Response::builder().status(429).header("retry-after", "7").header("content-type", "application/json")
            .body(Body::from(json!({"error": {"message": "slow down"}}).to_string())).unwrap()),
        "bad400" => Some((StatusCode::BAD_REQUEST, Json(json!({"error": {"message": "max_tokens is too large"}}))).into_response()),
        _ => None,
    }
}

async fn openai_chat(State(m): State<Mock>, headers: HeaderMap, Json(b): Json<Value>) -> Response {
    if let Err(r) = record(&m, "/v1/chat/completions", &headers, &b) {
        return r;
    }
    let model = b["model"].as_str().unwrap_or("").to_string();
    if let Some(r) = failure(&model) {
        return r;
    }
    let chunk = |delta: Value, finish: Value| format!("data: {}\n\n", json!({"id": "x", "object": "chat.completion.chunk", "model": model,
                                                                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}));
    let usage = format!("data: {}\n\n", json!({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 2}}}));
    let last = b["messages"].as_array().and_then(|a| a.last()).cloned().unwrap_or(Value::Null);
    match model.as_str() {
        "slow" => slow(chunk(json!({"content": "Starting"}), Value::Null), m.closed.clone()),
        "json" => Json(json!({"id": "x", "object": "chat.completion", "model": model, "choices": [{"index": 0,
            "message": {"role": "assistant", "content": "Whole answer."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}})).into_response(),
        "tool" if last["role"] == "tool" => sse(vec![chunk(json!({"content": format!("The sum is {}.", last["content"].as_str().unwrap_or("?"))}), json!("stop")),
                                                    usage, "data: [DONE]\n\n".into()]),
        "tool" => sse(vec![chunk(json!({"tool_calls": [{"index": 0, "id": "call_abc", "type": "function", "function": {"name": "calc__add", "arguments": "{\"a\": 2,"}}]}), Value::Null),
                           chunk(json!({"tool_calls": [{"index": 0, "function": {"arguments": " \"b\": 3}"}}]}), json!("tool_calls")),
                           usage, "data: [DONE]\n\n".into()]),
        "client-tool" => sse(vec![chunk(json!({"tool_calls": [{"index": 0, "id": "call_w", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\"}"}}]}), json!("tool_calls")),
                                  usage, "data: [DONE]\n\n".into()]),
        _ => sse(vec![chunk(json!({"role": "assistant", "content": ""}), Value::Null), chunk(json!({"reasoning_content": "Hmm."}), Value::Null),
                      chunk(json!({"content": "Hello from "}), Value::Null), chunk(json!({"content": model}), json!("stop")), usage, "data: [DONE]\n\n".into()]),
    }
}

async fn anthropic_messages(State(m): State<Mock>, headers: HeaderMap, Json(b): Json<Value>) -> Response {
    if let Err(r) = record(&m, "/v1/messages", &headers, &b) {
        return r;
    }
    let model = b["model"].as_str().unwrap_or("").to_string();
    if let Some(r) = failure(&model) {
        return r;
    }
    let ev = |name: &str, v: Value| format!("event: {name}\ndata: {v}\n\n");
    let start = ev("message_start", json!({"type": "message_start", "message": {"id": "msg_1", "model": model, "usage": {"input_tokens": 20, "cache_read_input_tokens": 5}}}));
    let text = |i: u32, t: &str| vec![
        ev("content_block_start", json!({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})),
        ev("content_block_delta", json!({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": t}})),
        ev("content_block_stop", json!({"type": "content_block_stop", "index": i}))];
    let end = |reason: &str| vec![ev("message_delta", json!({"type": "message_delta", "delta": {"stop_reason": reason}, "usage": {"output_tokens": 4}})),
                                  ev("message_stop", json!({"type": "message_stop"}))];
    let last = b["messages"].as_array().and_then(|a| a.last()).cloned().unwrap_or(Value::Null);
    let result = last["content"].as_array().and_then(|c| c.iter().find(|x| x["type"] == "tool_result")).cloned();
    let mut out = vec![start];
    match (model.as_str(), result) {
        ("tool", Some(r)) => {
            let said = r["content"].as_str().map(str::to_string)
                .or_else(|| r["content"][0]["text"].as_str().map(str::to_string)).unwrap_or_default();
            out.extend(text(0, &format!("The sum is {said}.")));
            out.extend(end("end_turn"));
        }
        ("tool", None) => {
            out.push(ev("content_block_start", json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_x1", "name": "calc__add", "input": {}}})));
            out.push(ev("content_block_delta", json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"a\": 2, \"b\": 3}"}})));
            out.push(ev("content_block_stop", json!({"type": "content_block_stop", "index": 0})));
            out.extend(end("tool_use"));
        }
        _ => {
            out.extend(text(0, &format!("Hello from {model}")));
            out.extend(end("end_turn"));
        }
    }
    sse(out)
}

async fn models(State(m): State<Mock>, headers: HeaderMap) -> Response {
    if let Err(r) = record(&m, "/v1/models", &headers, &Value::Null) {
        return r;
    }
    Json(json!({"object": "list", "data": [{"id": "listed-1"}, {"id": "listed-2"}]})).into_response()
}

async fn images(State(m): State<Mock>, headers: HeaderMap, Json(b): Json<Value>) -> Response {
    if let Err(r) = record(&m, "/v1/images/generations", &headers, &b) {
        return r;
    }
    Json(json!({"created": 1, "data": [{"b64_json": "iVBORw0KGgo="}]})).into_response()
}

/// Both mock providers on one port: OpenAI's API under /v1 (chat, models, images), Anthropic's /v1/messages.
fn mock_providers() -> (u16, Mock) {
    let m = Mock::default();
    let port = free_port();
    let app = axum::Router::new()
        .route("/v1/chat/completions", axum::routing::post(openai_chat))
        .route("/v1/messages", axum::routing::post(anthropic_messages))
        .route("/v1/models", axum::routing::get(models))
        .route("/v1/images/generations", axum::routing::post(images))
        .with_state(m.clone());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        rt.block_on(async move {
            let l = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.unwrap();
            tx.send(()).unwrap();
            axum::serve(l, app).await.unwrap();
        });
    });
    rx.recv().unwrap();
    (port, m)
}

fn remote_block(port: u16) -> Value {
    json!({
        "providers": {
            "openai": {"kind": "openai", "base_url": format!("http://127.0.0.1:{port}/v1"), "key": "env:AM2_TEST_PROVIDER_KEY",
                       "models": ["m1", "tool", "client-tool", "json", "slow", "fail500", "fail429", "bad400", "busy-model"], "kinds": ["chat", "image"]},
            "anthropic": {"kind": "anthropic", "base_url": format!("http://127.0.0.1:{port}"), "key": "env:AM2_TEST_PROVIDER_KEY",
                          "models": ["claude-x", "tool", "fail500"]},
            "listed": {"kind": "openai", "base_url": format!("http://127.0.0.1:{port}/v1"), "key": "env:AM2_TEST_PROVIDER_KEY", "models": []}
        },
        "mix": {"default": "local", "routes": [
            {"when": {"tokens_over": 2000}, "use": "anthropic/claude-x"},
            {"when": {"has_images": true}, "use": "openai/m1"},
            {"when": {"local_busy": true}, "use": "openai/busy-model"}]},
        "mixes": {
            "f500": {"default": "anthropic/claude-x", "routes": [{"when": {"tokens_over": 0}, "use": "openai/fail500"}]},
            "f429": {"default": "local", "routes": [{"when": {"tokens_over": 0}, "use": "openai/fail429"}]}
        }
    })
}

fn start(port: u16, extra_cfg: Value, replies: &[Vec<u32>], extra: Value) -> Server {
    let mut cfg = json!({"remote": remote_block(port), "api_monitor": true,
                         "mcp_servers": {"calc": {"command": env!("CARGO_BIN_EXE_strata-mock-mcp")}}});
    if let (Some(c), Some(e)) = (cfg.as_object_mut(), extra_cfg.as_object()) {
        c.extend(e.clone());
    }
    let mut x = json!({"server_log": true, "env": {"AM2_TEST_PROVIDER_KEY": PROVIDER_KEY}});
    if let (Some(c), Some(e)) = (x.as_object_mut(), extra.as_object()) {
        c.extend(e.clone());
    }
    Server::start(&[Model { vision: false, extra: cfg, ..Model::default() }], replies, x)
}

fn wait_mcp(s: &Server) {
    let t0 = Instant::now();
    while s.get("/mcp", true).1["tools"] != 1 && t0.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// POST with a key; the status, the headers and the body text.
fn call(s: &Server, key: &str, path: &str, body: &Value) -> (u16, HeaderMap, String) {
    let r = reqwest::blocking::Client::new().post(format!("{}{path}", s.base)).bearer_auth(key).json(body)
        .timeout(Duration::from_secs(60)).send().expect("request");
    let (code, h) = (r.status().as_u16(), r.headers().clone());
    let h2: HeaderMap = h.iter().filter_map(|(k, v)| Some((axum::http::HeaderName::from_bytes(k.as_str().as_bytes()).ok()?,
                                                            axum::http::HeaderValue::from_bytes(v.as_bytes()).ok()?))).collect();
    (code, h2, r.text().unwrap_or_default())
}

fn header(h: &HeaderMap, k: &str) -> String {
    h.get(k).and_then(|v| v.to_str().ok()).unwrap_or("").to_string()
}

fn sse_data(text: &str) -> Vec<Value> {
    text.lines().filter_map(|l| l.strip_prefix("data:")).map(str::trim).filter(|d| *d != "[DONE]")
        .filter_map(|d| serde_json::from_str(d).ok()).collect()
}

fn user(t: &str) -> Value {
    json!({"role": "user", "content": t})
}

// ------------------------------------------------------------------ off

#[test]
fn off_nothing_changes() {
    let s = Server::start(&[Model { vision: false, ..Model::default() }], &[ids("ok.\n</think>\n\nlocal answer")], json!({}));
    let (_, m) = s.get("/v1/models", true);
    let ids: Vec<&str> = m["data"].as_array().unwrap().iter().map(|x| x["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["mock-a"], "only the local model");
    assert!(!m.to_string().contains("remote") && !m.to_string().contains("\"mix\""));
    let (_, a) = s.get("/api/models", true);
    assert_eq!(a["models"].as_array().unwrap().len(), 1);
    // a remote name is just a name: the loaded model answers, as before
    let (code, h, body) = call(&s, KEY, "/v1/chat/completions", &json!({"model": "openai/m1", "messages": [user("hi")]}));
    assert_eq!(code, 200, "{body}");
    assert!(body.contains("local answer") && h.get("x-strata-routed").is_none());
    // the owner's page is there (to turn it on) and says the feature is off; the rest of its API is 404
    assert_eq!(s.get("/rtenai", false).0, 200);
    let (code, st) = s.get("/api/rtenai", true);
    assert_eq!((code, st["enabled"].as_bool()), (200, Some(false)), "{st}");
    assert_eq!(s.get("/api/rtenai", false).0, 401, "the owner's key only");
    assert_eq!(s.get("/api/rtenai/models/openai", true).0, 404);
    assert_eq!(s.post("/api/rtenai/test/openai", &json!({})).0, 404);
    assert_eq!(s.post("/api/rtenai/keys/openai", &json!({"key": "x"})).0, 404);
    // turned on from the page: the block is written (no key in it) and used at once
    let (code, v) = s.post("/api/rtenai", &json!({"remote": {"providers": {"p": {"kind": "openai", "base_url": "http://127.0.0.1:9/v1", "models": ["a"]}}}}));
    assert_eq!((code, v["enabled"].as_bool()), (200, Some(true)), "{v}");
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(s.dir.join("strata-mock-a.json")).unwrap()).unwrap();
    assert_eq!(cfg["remote"]["providers"]["p"]["models"], json!(["a"]));
    let (_, m) = s.get("/v1/models", true);
    assert!(m["data"].as_array().unwrap().iter().any(|x| x["id"] == "p/a" && x["remote"] == true), "{m}");
    // a key itself is never taken into the run config
    let (code, _) = s.post("/api/rtenai", &json!({"remote": {"providers": {"p": {"kind": "openai", "base_url": "http://127.0.0.1:9/v1", "key": "sk-plain"}}}}));
    assert_eq!(code, 400);
    // and off again
    let (code, v) = s.post("/api/rtenai", &json!({"remote": null}));
    assert_eq!((code, v["enabled"].as_bool()), (200, Some(false)));
    let (_, m) = s.get("/v1/models", true);
    assert_eq!(m["data"].as_array().unwrap().len(), 1);
}

// ------------------------------------------------------------------ on

#[test]
fn every_client_api_reaches_both_providers_streamed_and_not() {
    let (port, mock) = mock_providers();
    let s = start(port, json!({}), &[ids("x")], json!({}));
    // the model lists: the owner sees the remote models (a provider's own list too) and the mixes
    let (_, m) = s.get("/v1/models", true);
    let ids: Vec<String> = m["data"].as_array().unwrap().iter().map(|x| x["id"].as_str().unwrap().to_string()).collect();
    for want in ["mock-a", "openai/m1", "anthropic/claude-x", "listed/listed-1", "mix", "mix/f500"] {
        assert!(ids.iter().any(|i| i == want), "{want} in {ids:?}");
    }
    let (_, app) = s.get("/api/models", true);
    assert!(app["models"].as_array().unwrap().iter().any(|x| x["id"] == "openai/m1" && x["provider"] == "openai" && x["remote"] == true));

    // OpenAI chat, not streamed, to each provider
    for (model, said) in [("openai/m1", "Hello from m1"), ("anthropic/claude-x", "Hello from claude-x"), ("openai/json", "Whole answer.")] {
        let (code, h, body) = call(&s, KEY, "/v1/chat/completions", &json!({"model": model, "messages": [user("hi")], "max_tokens": 50, "temperature": 0.3}));
        assert_eq!(code, 200, "{model}: {body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], said, "{v}");
        assert_eq!(v["model"], model);
        assert_eq!(header(&h, "x-strata-routed"), model);
    }
    let v: Value = serde_json::from_str(&call(&s, KEY, "/v1/chat/completions", &json!({"model": "openai/m1", "messages": [user("hi")]})).2).unwrap();
    assert_eq!((v["usage"]["prompt_tokens"].as_u64(), v["usage"]["completion_tokens"].as_u64(), v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64()),
               (Some(12), Some(3), Some(2)), "the provider's usage: {v}");
    assert_eq!(v["choices"][0]["message"]["reasoning_content"], "Hmm.");
    // what the providers received: their own API, the bare model name, the key
    let b = mock.bodies();
    let oa = b.iter().find(|x| x["model"] == "m1").unwrap();
    assert_eq!((oa["max_tokens"].as_u64(), oa["temperature"].as_f64(), oa["stream"].as_bool()), (Some(50), Some(0.3), Some(true)), "{oa}");
    let an = b.iter().find(|x| x["model"] == "claude-x").unwrap();
    assert_eq!((an["max_tokens"].as_u64(), an["messages"][0]["content"][0]["text"].as_str()), (Some(50), Some("hi")), "{an}");

    // OpenAI chat streamed
    let (code, _, body) = call(&s, KEY, "/v1/chat/completions", &json!({"model": "anthropic/claude-x", "messages": [user("hi")], "stream": true}));
    assert_eq!(code, 200);
    let content: String = sse_data(&body).iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str().map(str::to_string)).collect();
    assert_eq!(content, "Hello from claude-x");
    let last = sse_data(&body).into_iter().rev().find(|c| c.get("usage").is_some()).unwrap();
    assert_eq!((last["usage"]["prompt_tokens"].as_u64(), last["usage"]["completion_tokens"].as_u64()), (Some(25), Some(4)), "{last}");

    // Anthropic Messages, not streamed and streamed
    let (code, h, body) = call(&s, KEY, "/v1/messages", &json!({"model": "openai/m1", "max_tokens": 64, "system": "Be kind.", "messages": [user("hi")]}));
    assert_eq!(code, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert!(v["content"].as_array().unwrap().iter().any(|c| c["type"] == "text" && c["text"] == "Hello from m1"), "{v}");
    assert_eq!((v["usage"]["output_tokens"].as_u64(), header(&h, "x-strata-routed").as_str()), (Some(3), "openai/m1"));
    let sys = mock.bodies().into_iter().rev().find(|x| x["model"] == "m1").unwrap();
    assert_eq!(sys["messages"][0], json!({"role": "system", "content": "Be kind."}));
    let (_, _, body) = call(&s, KEY, "/v1/messages", &json!({"model": "anthropic/claude-x", "max_tokens": 64, "stream": true, "messages": [user("hi")]}));
    let text: String = sse_data(&body).iter().filter_map(|e| e["delta"]["text"].as_str().map(str::to_string)).collect();
    assert_eq!(text, "Hello from claude-x");
    assert!(sse_data(&body).iter().any(|e| e["type"] == "message_stop"));

    // Responses and Gemini
    let (code, _, body) = call(&s, KEY, "/v1/responses", &json!({"model": "anthropic/claude-x", "input": "hi"}));
    assert_eq!(code, 200, "{body}");
    assert!(body.contains("Hello from claude-x"));
    let (code, _, body) = call(&s, KEY, "/v1beta/models/openai%2Fm1:generateContent", &json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}));
    assert_eq!(code, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert!(v["candidates"][0]["content"]["parts"].as_array().unwrap().iter().any(|p| p["text"] == "Hello from m1"), "{v}");
    let (code, _, body) = call(&s, KEY, "/v1beta/models/anthropic%2Fclaude-x:streamGenerateContent", &json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}));
    assert_eq!(code, 200);
    assert!(body.contains("Hello from claude-x"), "{body}");

    // a client's own tool: the provider's call goes back to the client
    let (_, _, body) = call(&s, KEY, "/v1/chat/completions", &json!({"model": "openai/client-tool", "messages": [user("weather?")],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]}));
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!((v["choices"][0]["finish_reason"].as_str(), v["choices"][0]["message"]["tool_calls"][0]["function"]["name"].as_str()),
               (Some("tool_calls"), Some("get_weather")), "{v}");

    // a remote image model
    let (code, h, body) = call(&s, KEY, "/v1/images/generations", &json!({"model": "openai/m1", "prompt": "a cat"}));
    assert_eq!((code, header(&h, "x-strata-routed").as_str()), (200, "openai/m1"), "{body}");
    assert!(body.contains("b64_json"));
    // a provider's refusal in the client's shape, with its status
    let (code, _, body) = call(&s, KEY, "/v1/messages", &json!({"model": "openai/bad400", "max_tokens": 9, "messages": [user("x")]}));
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!((code, v["type"].as_str()), (400, Some("error")), "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("max_tokens is too large"));
    // a model the provider is not given for
    assert_eq!(call(&s, KEY, "/v1/chat/completions", &json!({"model": "openai/unknown", "messages": [user("x")]})).0, 404);
}

#[test]
fn server_tools_run_here_for_remote_models_in_both_apis() {
    let (port, mock) = mock_providers();
    let s = start(port, json!({}), &[ids("x")], json!({}));
    wait_mcp(&s);
    for model in ["openai/tool", "anthropic/tool"] {
        // OpenAI client, streamed: the MCP call runs on this server, the model goes on with its result
        let (code, h, body) = call(&s, KEY, "/v1/chat/completions", &json!({"model": model, "messages": [user("2+3?")], "stream": true, "strata_mcp": true}));
        assert_eq!(code, 200, "{body}");
        let ev = sse_data(&body);
        let mcp: Vec<&Value> = ev.iter().filter_map(|v| v.get("strata_mcp")).collect();
        let kinds: Vec<&str> = mcp.iter().map(|e| e["event"].as_str().unwrap()).collect();
        assert!(kinds.contains(&"call") && kinds.contains(&"result"), "{model}: {kinds:?}");
        assert!(mcp.iter().any(|e| e["event"] == "result" && e["text"] == "5"), "{mcp:?}");
        let content: String = ev.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str().map(str::to_string)).collect();
        assert_eq!(content, "The sum is 5.", "{model}");
        assert_eq!(header(&h, "x-strata-routed"), model);
        assert!(!ev.iter().any(|c| c["choices"][0]["delta"].get("tool_calls").is_some()), "the MCP call is not the client's");
        // usage: both rounds' prompts (the provider bills each)
        let last = ev.iter().rev().find(|c| c.get("usage").is_some()).unwrap();
        assert!(last["usage"]["prompt_tokens"].as_u64().unwrap() >= 24, "{last}");
        // Anthropic client, not streamed
        let (code, _, body) = call(&s, KEY, "/v1/messages", &json!({"model": model, "max_tokens": 200, "messages": [user("2+3?")], "strata_mcp": true}));
        assert_eq!(code, 200, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert!(v["content"].as_array().unwrap().iter().any(|c| c["text"] == "The sum is 5."), "{v}");
        assert_eq!(v["stop_reason"], "end_turn");
    }
    // the second round carried the call and its result back, in each provider's own shape
    let b = mock.bodies();
    let oa = b.iter().find(|x| x["model"] == "tool" && x["messages"].as_array().is_some_and(|m| m.iter().any(|m| m["role"] == "tool"))).unwrap();
    let msgs = oa["messages"].as_array().unwrap();
    let call_id = msgs.iter().find_map(|m| m["tool_calls"][0]["id"].as_str()).unwrap();
    assert_eq!(msgs.iter().find(|m| m["role"] == "tool").unwrap()["tool_call_id"], call_id);
    assert!(oa["tools"].as_array().unwrap().iter().any(|t| t["function"]["name"] == "calc__add"), "the server's tool was offered");
    let an = b.iter().find(|x| x["model"] == "tool" && x.get("system").is_none() && x["messages"].as_array()
        .is_some_and(|m| m.iter().any(|m| m["content"].as_array().is_some_and(|c| c.iter().any(|c| c["type"] == "tool_result"))))).unwrap();
    let use_id = an["messages"].as_array().unwrap().iter().flat_map(|m| m["content"].as_array().cloned().unwrap_or_default())
        .find(|c| c["type"] == "tool_use").unwrap()["id"].clone();
    let res_id = an["messages"].as_array().unwrap().iter().flat_map(|m| m["content"].as_array().cloned().unwrap_or_default())
        .find(|c| c["type"] == "tool_result").unwrap()["tool_use_id"].clone();
    assert_eq!(use_id, res_id);
}

#[test]
fn keys_need_permission_usage_is_counted_and_the_provider_key_never_shows() {
    let (port, _mock) = mock_providers();
    let s = start(port, json!({"clients": {"laptop": {"key": "laptop-key-0123456789", "tools": [], "models": ["anthropic/*"]}}}), &[ids("x"), ids("y")], json!({}));
    let mut seen_text = String::new();
    // a generated key with no model list, and one listing a local model: no remote models, 403 by name
    for models in [Value::Null, json!(["mock-a"])] {
        let (code, made) = s.post("/api/keys", &json!({"name": format!("k{}", models.is_null()), "models": models}));
        assert_eq!(code, 201, "{made}");
        let key = made["key"].as_str().unwrap().to_string();
        let r = reqwest::blocking::Client::new().get(format!("{}/v1/models", s.base)).bearer_auth(&key).send().unwrap().text().unwrap();
        assert!(!r.contains("openai/") && !r.contains("\"mix\""), "{r}");
        let (code, _, body) = call(&s, &key, "/v1/chat/completions", &json!({"model": "openai/m1", "messages": [user("hi")]}));
        assert_eq!(code, 403, "{body}");
        assert_eq!(call(&s, &key, "/v1/chat/completions", &json!({"model": "mix", "messages": [user("hi")]})).0, 403);
        assert_eq!(reqwest::blocking::Client::new().get(format!("{}/api/rtenai", s.base)).bearer_auth(&key).send().unwrap().status().as_u16(), 401);
    }
    // with remote/*: it sees them and may use them; its use is counted per key and model
    let (_, made) = s.post("/api/keys", &json!({"name": "remote-user", "models": ["remote/*"]}));
    let (key, id) = (made["key"].as_str().unwrap().to_string(), made["id"].as_str().unwrap().to_string());
    let r = reqwest::blocking::Client::new().get(format!("{}/v1/models", s.base)).bearer_auth(&key).send().unwrap().text().unwrap();
    assert!(r.contains("openai/m1") && !r.contains("\"mix\""), "{r}");
    let (code, _, body) = call(&s, &key, "/v1/chat/completions", &json!({"model": "openai/m1", "messages": [user("hi")]}));
    assert_eq!(code, 200, "{body}");
    seen_text.push_str(&body);
    let (_, _, body) = call(&s, &key, "/v1/messages", &json!({"model": "openai/m1", "max_tokens": 9, "stream": true, "messages": [user("hi")]}));
    seen_text.push_str(&body);
    let t0 = Instant::now();
    let row = loop {
        let (_, u) = s.get(&format!("/api/usage?key={id}&days=1"), true);
        let row = u["by_model"].as_array().and_then(|a| a.iter().find(|m| m["model"] == "openai/m1").cloned());
        if row.as_ref().is_some_and(|r| r["requests"] == 2) || t0.elapsed() > Duration::from_secs(5) {
            break row.unwrap_or(json!({}));
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    assert_eq!((row["requests"].as_u64(), row["prompt_tokens"].as_u64(), row["completion_tokens"].as_u64()), (Some(2), Some(24), Some(6)), "{row}");
    // a run-config profile with "models": ["anthropic/*"]: those only
    assert_eq!(call(&s, "laptop-key-0123456789", "/v1/chat/completions", &json!({"model": "anthropic/claude-x", "messages": [user("hi")]})).0, 200);
    assert_eq!(call(&s, "laptop-key-0123456789", "/v1/chat/completions", &json!({"model": "openai/m1", "messages": [user("hi")]})).0, 403);
    // the owner's view of usage per provider
    let (_, u) = s.get("/api/rtenai/usage?days=1", true);
    assert!(u["providers"]["openai"]["completion_tokens"].as_u64().unwrap() >= 6, "{u}");

    // a key set from the page: kept 0600, shown as set with its last 4 characters only
    let (code, v) = s.post("/api/rtenai/keys/listed", &json!({"key": PROVIDER_KEY}));
    assert_eq!((code, v["key"]["set"].as_bool(), v["key"]["last4"].as_str()), (200, Some(true), Some("cdef")), "{v}");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(s.dir.join("remote-keys.json")).unwrap().permissions().mode() & 0o777, 0o600);
    }
    let (code, t) = s.post("/api/rtenai/test/listed", &json!({}));
    assert_eq!((code, t["ok"].as_bool()), (200, Some(true)), "{t}");
    assert_eq!(t["models"], json!(["listed-1", "listed-2"]));
    // the provider's key is in no answer, no log line and no monitor record
    for path in ["/api/rtenai", "/api/rtenai/usage", "/api/rtenai/models/openai", "/v1/models", "/api/models", "/api/requests?limit=100"] {
        seen_text.push_str(&s.get(path, true).1.to_string());
    }
    seen_text.push_str(&s.post("/api/rtenai/test/openai", &json!({})).1.to_string());
    std::thread::sleep(Duration::from_millis(300));
    let log = std::fs::read_to_string(s.dir.join("server.log")).unwrap_or_default();
    assert!(log.contains("remote"), "the log was written: {}", log.chars().take(300).collect::<String>());
    let calls = std::fs::read_dir(s.dir.join("api-calls")).into_iter().flatten().flatten()
        .map(|e| std::fs::read_to_string(e.path()).unwrap_or_default()).collect::<String>();
    let (_, mon) = s.get("/api/requests?limit=100", true);
    assert!(mon.to_string().contains("openai/m1"), "the monitor recorded the remote calls: {mon}");
    for (what, text) in [("answers", &seen_text), ("log", &log), ("monitor files", &calls)] {
        assert!(!text.contains(PROVIDER_KEY) && !text.contains("PROVIDERSECRET"), "the provider key is in the {what}");
    }
}

#[test]
fn a_wrong_provider_key_is_the_servers_problem_not_the_clients() {
    let (port, _mock) = mock_providers();
    let s = start(port, json!({}), &[ids("x")], json!({"env": {"AM2_TEST_PROVIDER_KEY": "sk-wrong-key-0123456789"}}));
    let (code, _, body) = call(&s, KEY, "/v1/chat/completions", &json!({"model": "openai/m1", "messages": [user("hi")]}));
    assert_eq!(code, 502, "{body}");
    assert!(body.contains("refused the server's key") && !body.contains("sk-wrong"), "{body}");
    let (code, _, body) = call(&s, KEY, "/v1/messages", &json!({"model": "anthropic/claude-x", "max_tokens": 9, "messages": [user("hi")]}));
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!((code, v["type"].as_str(), v["error"]["type"].as_str()), (502, Some("error"), Some("api_error")), "{v}");
}

#[test]
fn a_key_file_others_can_read_is_refused() {
    let (port, _mock) = mock_providers();
    let dir = std::env::temp_dir().join(format!("am2-remote-test-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("p.key");
    std::fs::write(&f, PROVIDER_KEY).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o644)).unwrap();
    }
    let mut block = remote_block(port);
    block["providers"]["openai"]["key"] = json!(format!("file:{}", f.display()));
    let s = start(port, json!({"remote": block}), &[ids("x")], json!({}));
    let (_, st) = s.get("/api/rtenai", true);
    let k = st["providers"].as_array().unwrap().iter().find(|p| p["name"] == "openai").unwrap()["key"].clone();
    #[cfg(unix)]
    {
        assert_eq!(k["set"], false, "{k}");
        assert!(k["error"].as_str().unwrap().contains("chmod 600"), "{k}");
        let (code, _, body) = call(&s, KEY, "/v1/chat/completions", &json!({"model": "openai/m1", "messages": [user("hi")]}));
        assert_eq!(code, 502, "{body}");
        assert!(body.contains("644"), "{body}");
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    assert_eq!(call(&s, KEY, "/v1/chat/completions", &json!({"model": "openai/m1", "messages": [user("hi")]})).0, 200, "0600 is used");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_mix_routes_by_size_pictures_and_falls_back_once() {
    let (port, _mock) = mock_providers();
    let s = start(port, json!({}), &[ids("Thinking.\n</think>\n\nlocal answer")], json!({}));
    // small, no pictures, local idle: the local model
    let (code, h, body) = call(&s, KEY, "/v1/chat/completions", &json!({"model": "mix", "messages": [user("hi")]}));
    assert_eq!(code, 200, "{body}");
    assert_eq!(header(&h, "x-strata-routed"), "mock-a");
    assert!(body.contains("local answer"));
    // a long conversation: tokens_over 2000
    let long = "word ".repeat(3000);
    let (code, h, body) = call(&s, KEY, "/v1/chat/completions", &json!({"model": "mix", "messages": [user(&long)]}));
    assert_eq!((code, header(&h, "x-strata-routed").as_str()), (200, "anthropic/claude-x"), "{body}");
    // a picture: has_images
    let (code, h, body) = call(&s, KEY, "/v1/chat/completions", &json!({"model": "mix", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is it?"}, {"type": "image_url", "image_url": {"url": png()}}]}]}));
    assert_eq!((code, header(&h, "x-strata-routed").as_str()), (200, "openai/m1"), "{body}");
    // a 500 before the answer: the next model, said in the headers
    let (code, h, body) = call(&s, KEY, "/v1/chat/completions", &json!({"model": "mix/f500", "messages": [user("hi")]}));
    assert_eq!((code, header(&h, "x-strata-routed").as_str()), (200, "anthropic/claude-x"), "{body}");
    assert!(header(&h, "x-strata-note").contains("openai/fail500 did not answer (HTTP 500)"), "{h:?}");
    // a 429: the fallback (local here), streamed too
    let (code, h, body) = call(&s, KEY, "/v1/chat/completions", &json!({"model": "mix/f429", "messages": [user("hi")], "stream": true}));
    assert_eq!((code, header(&h, "x-strata-routed").as_str()), (200, "mock-a"), "{body}");
    assert!(header(&h, "x-strata-note").contains("HTTP 429"));
    // asked by name, a 429 is the client's 429 with the provider's Retry-After
    let (code, h, body) = call(&s, KEY, "/v1/chat/completions", &json!({"model": "openai/fail429", "messages": [user("hi")]}));
    assert_eq!((code, header(&h, "retry-after").as_str()), (429, "7"), "{body}");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["error"]["type"], "rate_limit_error");
    // a mix that is not in the config
    assert_eq!(call(&s, KEY, "/v1/chat/completions", &json!({"model": "mix/nope", "messages": [user("hi")]})).0, 404);
}

#[test]
fn the_mix_sends_work_away_while_the_local_model_is_busy() {
    let (port, _mock) = mock_providers();
    let long = ids(&format!("Thinking.\n</think>\n\n{}", "slow ".repeat(30)));
    let s = start(port, json!({}), &[long.clone(), long.clone(), long], json!({"delay_ms": 150}));
    let base = s.base.clone();
    let busy: Vec<_> = (0..2).map(|_| {
        let base = base.clone();
        std::thread::spawn(move || {
            let _ = reqwest::blocking::Client::new().post(format!("{base}/v1/chat/completions")).bearer_auth(KEY)
                .json(&json!({"model": "mock-a", "messages": [{"role": "user", "content": "go"}]})).timeout(Duration::from_secs(60)).send();
        })
    }).collect();
    let t0 = Instant::now();
    while s.get("/api/queue", true).1["waiting"].as_array().is_none_or(|w| w.is_empty()) && t0.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(30));
    }
    let (code, h, body) = call(&s, KEY, "/v1/chat/completions", &json!({"model": "mix", "messages": [user("hi")]}));
    assert_eq!((code, header(&h, "x-strata-routed").as_str()), (200, "openai/busy-model"), "{body}");
    assert!(t0.elapsed() < Duration::from_secs(8), "it did not wait for the local queue");
    for b in busy {
        let _ = b.join();
    }
}

#[test]
fn a_client_that_leaves_ends_the_providers_call() {
    let (port, mock) = mock_providers();
    let s = start(port, json!({}), &[ids("x")], json!({}));
    let mut r = reqwest::blocking::Client::new().post(format!("{}/v1/chat/completions", s.base)).bearer_auth(KEY)
        .json(&json!({"model": "openai/slow", "messages": [user("hi")], "stream": true})).send().unwrap();
    let mut first = [0u8; 64];
    let n = std::io::Read::read(&mut r, &mut first).unwrap();
    assert!(n > 0);
    assert!(!mock.closed.load(Ordering::SeqCst), "still answering");
    drop(r);
    let t0 = Instant::now();
    while !mock.closed.load(Ordering::SeqCst) && t0.elapsed() < Duration::from_secs(20) {
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(mock.closed.load(Ordering::SeqCst), "the provider's connection was closed after the client left");
}

#[test]
fn the_command_line_shows_tests_and_sets_keys() {
    let run = |s: &Server, args: &[&str], stdin: Option<&str>| {
        use std::io::Write;
        let mut c = std::process::Command::new(env!("CARGO_BIN_EXE_am2strata-run"));
        c.arg("remote").args(args).args(["--url", &s.base, "--api-key", KEY, "--models-dir", "/nonexistent"])
            .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());
        let mut child = c.spawn().unwrap();
        if let Some(t) = stdin {
            child.stdin.take().unwrap().write_all(t.as_bytes()).unwrap();
        }
        let o = child.wait_with_output().unwrap();
        (o.status.success(), String::from_utf8_lossy(&o.stdout).to_string(), String::from_utf8_lossy(&o.stderr).to_string())
    };
    let off = Server::start(&[Model { vision: false, ..Model::default() }], &[ids("x")], json!({}));
    let (ok, _, err) = run(&off, &["status"], None);
    assert!(!ok && err.contains("remote models are off"), "{err}");
    let (port, _mock) = mock_providers();
    let s = start(port, json!({}), &[ids("x")], json!({}));
    let (ok, out, err) = run(&s, &["status"], None);
    assert!(ok, "{err}");
    assert!(out.contains("openai") && out.contains("key set (env") && out.contains("mix: 3 rule(s), default local"), "{out}");
    assert!(!out.contains(PROVIDER_KEY));
    let (ok, out, _) = run(&s, &["models", "listed"], None);
    assert!(ok && out.contains("listed/listed-1"), "{out}");
    let (ok, out, err) = run(&s, &["test", "anthropic"], None);
    assert!(ok && out.contains("ok: 2 models"), "{out} {err}");
    let (ok, out, err) = run(&s, &["set-key", "listed"], Some(&format!("{PROVIDER_KEY}\n")));
    assert!(ok && out.contains("key set (…cdef)"), "{out} {err}");
    assert!(!out.contains(PROVIDER_KEY) && !err.contains(PROVIDER_KEY));
}
