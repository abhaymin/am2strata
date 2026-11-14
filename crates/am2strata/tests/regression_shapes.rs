//! Golden response shapes of the compatibility APIs (docs/REGRESSION.md): what an OpenAI, Anthropic, Responses or
//! Gemini client parses. Each answer's SHAPE (its keys and value types, with the few naming fields such as `object`,
//! `type`, `role` and `finish_reason` kept by value) is compared with a file under tests/golden/. Ids, timestamps,
//! token counts and timings are not in a shape, so nothing here is volatile.
//!
//! A field renamed, dropped, added or retyped fails the test and shows the difference. When the change is meant:
//!
//!   UPDATE_GOLDEN=1 cargo test -p am2strata --test regression_shapes -- --nocapture
//!
//! writes the new shapes into a TEMP folder (printed; or the folder `GOLDEN_OUT` names) and never into the
//! repository: review them there, then copy the ones you mean into crates/am2strata/tests/golden/.
//! `deploy/regression.sh --update-golden` does the same and prints the copy command.
//!
//! This server has no /v1/completions and no /v1/embeddings; `the_apis_this_server_does_not_have_stay_404` pins that,
//! so adding either one asks for its golden file here.

mod common;

use common::regress::*;
use common::*;
use serde_json::{json, Value};
use std::path::PathBuf;

/// Fields whose VALUE names the kind of thing (a client switches on them): kept in the shape as "=value".
const NAMING: &[&str] = &["object", "type", "role", "finish_reason", "stop_reason", "finishReason", "status"];

/// `regress::shape`, with the naming fields kept by value.
fn named_shape(v: &Value) -> Value {
    match v {
        Value::Object(o) => {
            let mut keys: Vec<&String> = o.keys().collect();
            keys.sort();
            Value::Object(keys.into_iter().map(|k| {
                let x = &o[k];
                (k.clone(), match x.as_str() {
                    Some(s) if NAMING.contains(&k.as_str()) => json!(format!("={s}")),
                    _ => named_shape(x),
                })
            }).collect())
        }
        Value::Array(a) => {
            let mut seen: Vec<Value> = Vec::new();
            for x in a {
                let s = named_shape(x);
                if !seen.contains(&s) {
                    seen.push(s);
                }
            }
            seen.sort_by_key(|s| s.to_string());
            Value::Array(seen)
        }
        other => shape(other),
    }
}

/// A stream's shape: for each event name (or "data" when events are unnamed), the distinct shapes of its data.
fn stream_shape(events: &[(String, Value)]) -> Value {
    let mut by_name: std::collections::BTreeMap<String, Vec<Value>> = Default::default();
    for (name, data) in events {
        let key = if name.is_empty() { "data".to_string() } else { name.clone() };
        let s = if data == "[DONE]" { json!("=[DONE]") } else { named_shape(data) };
        let list = by_name.entry(key).or_default();
        if !list.contains(&s) {
            list.push(s);
        }
    }
    Value::Object(by_name.into_iter().map(|(k, mut v)| {
        v.sort_by_key(|s| s.to_string());
        (k, Value::Array(v))
    }).collect())
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

/// Where UPDATE_GOLDEN=1 writes: `GOLDEN_OUT`, else one folder per test run under the system's temp folder.
fn update_dir() -> Option<PathBuf> {
    if std::env::var("UPDATE_GOLDEN").ok().as_deref() != Some("1") {
        return None;
    }
    let dir = std::env::var_os("GOLDEN_OUT").map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join(format!("am2strata-golden-{}", std::process::id())));
    let inside_repo = dir.canonicalize().unwrap_or(dir.clone()).starts_with(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap());
    assert!(!inside_repo, "GOLDEN_OUT is inside the repository: tests never write there. Name a folder outside it and copy the files by hand");
    std::fs::create_dir_all(&dir).unwrap();
    Some(dir)
}

/// Compare `got` (already a shape) with tests/golden/<name>.json; with UPDATE_GOLDEN=1, write it to the temp folder
/// instead and say where.
fn check_golden(name: &str, got: &Value) {
    let got = &sorted(got);
    let pretty = serde_json::to_string_pretty(got).unwrap() + "\n";
    let file = golden_dir().join(format!("{name}.json"));
    if let Some(dir) = update_dir() {
        let out = dir.join(format!("{name}.json"));
        std::fs::write(&out, &pretty).unwrap();
        let same = std::fs::read_to_string(&file).is_ok_and(|old| old == pretty);
        eprintln!("golden {name}: {} -> {}\n  to accept: cp {} {}", if same { "unchanged" } else { "CHANGED" }, out.display(), out.display(), file.display());
        return;
    }
    let want = std::fs::read_to_string(&file).unwrap_or_else(|_| panic!(
        "no golden file {}: make it with UPDATE_GOLDEN=1 (docs/REGRESSION.md) and copy it in", file.display()));
    if want != pretty {
        let want_v: Value = serde_json::from_str(&want).unwrap_or(Value::Null);
        let mut diff = Vec::new();
        differences("", &want_v, got, &mut diff);
        panic!("the shape of {name} changed ({} difference(s)):\n  {}\n\nIf a client would notice, this is a regression: fix the server. \
                If the change is meant, refresh the file (docs/REGRESSION.md):\n  UPDATE_GOLDEN=1 cargo test -p am2strata --test regression_shapes -- --nocapture\n\n\
                the whole new shape:\n{pretty}", diff.len(), diff.join("\n  "));
    }
}

/// Object keys in alphabetical order at every depth, so a golden file reads the same however serde_json was built
/// (with or without `preserve_order`).
fn sorted(v: &Value) -> Value {
    match v {
        Value::Object(o) => {
            let mut keys: Vec<&String> = o.keys().collect();
            keys.sort();
            Value::Object(keys.into_iter().map(|k| (k.clone(), sorted(&o[k]))).collect())
        }
        Value::Array(a) => Value::Array(a.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

/// What differs between two shapes, as paths ("choices[].message.content: was string, now null").
fn differences(path: &str, want: &Value, got: &Value, out: &mut Vec<String>) {
    match (want, got) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in a {
                match b.get(k) {
                    Some(g) => differences(&format!("{path}.{k}"), v, g, out),
                    None => out.push(format!("{path}.{k}: gone (was {v})")),
                }
            }
            for (k, g) in b {
                if !a.contains_key(k) {
                    out.push(format!("{path}.{k}: new ({g})"));
                }
            }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == 1 && b.len() == 1 => differences(&format!("{path}[]"), &a[0], &b[0], out),
        (a, b) if a != b => out.push(format!("{path}: was {a}, now {b}")),
        _ => {}
    }
}

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

const ANSWER: &str = "I add them.\n</think>\n\nThe answer is 4.";
const CALL: &str = "Need the tool.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nPune\n</parameter>\n</function>\n</tool_call>";

fn post(s: &Server, path: &str, body: &Value) -> Answer {
    Req::new("POST", &s.base, path).key(Some(KEY)).json(body).send()
}

fn ok_json(a: &Answer) -> Value {
    assert_eq!(a.status, 200, "{}", a.text);
    assert!(a.headers.get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|c| c.starts_with("application/json")), "JSON content type");
    a.json()
}

fn ok_stream(a: &Answer) -> Vec<(String, Value)> {
    assert_eq!(a.status, 200, "{}", a.text);
    assert!(a.headers.get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|c| c.starts_with("text/event-stream")), "an event stream");
    let ev = a.events();
    assert!(!ev.is_empty(), "no events: {}", a.text);
    ev
}

// ------------------------------------------------------------------------------------------------------------ OpenAI

#[test]
fn openai_models_and_health() {
    let s = Server::start(&[Model::default()], &[ids(ANSWER)], json!({}));
    check_golden("openai_models", &named_shape(&ok_json(&Req::new("GET", &s.base, "/v1/models").key(Some(KEY)).send())));
    check_golden("health", &named_shape(&ok_json(&Req::new("GET", &s.base, "/health").send())));
}

#[test]
fn openai_chat_completion_whole_and_streamed() {
    let s = Server::start(&[Model::default()], &[ids(ANSWER)], json!({}));
    let req = json!({"messages": [user("What is 2+2?")], "max_tokens": 100});
    let r = ok_json(&post(&s, "/v1/chat/completions", &req));
    assert_eq!(r["choices"][0]["message"]["content"], "The answer is 4.");
    assert!(r["id"].as_str().unwrap().starts_with("chatcmpl-"), "{}", r["id"]);
    check_golden("openai_chat", &named_shape(&r));
    // the usage block on its own: the three counts every client reads, and that they add up
    let u = &r["usage"];
    assert_eq!(u["prompt_tokens"].as_u64().unwrap() + u["completion_tokens"].as_u64().unwrap(), u["total_tokens"].as_u64().unwrap(), "{u}");

    let mut sreq = req.clone();
    sreq["stream"] = json!(true);
    let ev = ok_stream(&post(&s, "/v1/chat/completions", &sreq));
    assert_eq!(ev.last().unwrap().1, "[DONE]", "the stream ends with [DONE]");
    check_golden("openai_chat_stream", &stream_shape(&ev));
}

fn weather_openai() -> Value {
    json!({"type": "function", "function": {"name": "get_weather", "description": "Weather.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})
}

#[test]
fn openai_tool_calls_whole_and_streamed() {
    let s = Server::start(&[Model::default()], &[ids(CALL)], json!({}));
    let req = json!({"messages": [user("Weather in Pune?")], "tools": [weather_openai()]});
    let r = ok_json(&post(&s, "/v1/chat/completions", &req));
    assert_eq!(r["choices"][0]["finish_reason"], "tool_calls");
    assert!(r["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"].is_string(), "arguments are a JSON string, as OpenAI sends them");
    check_golden("openai_chat_tool_call", &named_shape(&r));
    let mut sreq = req.clone();
    sreq["stream"] = json!(true);
    check_golden("openai_chat_tool_call_stream", &stream_shape(&ok_stream(&post(&s, "/v1/chat/completions", &sreq))));
    // the tool's result sent back: the conversation a client continues with is accepted
    let call = r["choices"][0]["message"]["tool_calls"][0].clone();
    let follow = json!({"messages": [user("Weather in Pune?"), {"role": "assistant", "content": null, "tool_calls": [call.clone()]},
                                     {"role": "tool", "tool_call_id": call["id"], "content": "31 C, clear"}], "tools": [weather_openai()]});
    assert_eq!(post(&s, "/v1/chat/completions", &follow).status, 200, "the tool round trip");
}

#[test]
fn openai_error_bodies() {
    let s = Server::start(&[Model { context: 64, ..Model::default() }], &[ids("x")], json!({}));
    let mut errors = serde_json::Map::new();
    let mut add = |name: &str, a: Answer, status: u16| {
        assert_eq!(a.status, status, "{name}: {}", a.text);
        assert!(a.json()["error"]["message"].as_str().is_some_and(|m| !m.is_empty()), "{name}: error.message ({})", a.text);
        errors.insert(name.to_string(), json!({"status": a.status, "body": named_shape(&a.json())}));
    };
    add("no_key", Req::new("POST", &s.base, "/v1/chat/completions").json(&json!({"messages": [user("hi")]})).send(), 401);
    add("wrong_key", Req::new("POST", &s.base, "/v1/chat/completions").key(Some(WRONG_KEY)).json(&json!({"messages": [user("hi")]})).send(), 401);
    add("messages_not_a_list", post(&s, "/v1/chat/completions", &json!({"messages": "not json"})), 400);
    add("body_not_an_object", post(&s, "/v1/chat/completions", &json!([1, 2])), 400);
    add("broken_json", Req::new("POST", &s.base, "/v1/chat/completions").key(Some(KEY)).raw("application/json", b"{\"messages\": ").send(), 400);
    add("not_json_content_type", Req::new("POST", &s.base, "/v1/chat/completions").key(Some(KEY)).raw("text/plain", b"hi").send(), 415);
    add("over_the_context", post(&s, "/v1/chat/completions", &json!({"messages": [user("hi")], "max_tokens": 5000})), 400);
    add("unknown_reasoning_effort", post(&s, "/v1/chat/completions", &json!({"messages": [user("x")], "reasoning_effort": "turbo"})), 400);
    add("models_no_key", Req::new("GET", &s.base, "/v1/models").send(), 401);
    check_golden("openai_errors", &Value::Object(errors));
}

#[test]
fn the_apis_this_server_does_not_have_stay_404() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    for path in ["/v1/completions", "/v1/embeddings"] {
        let a = post(&s, path, &json!({"prompt": "hi", "input": "hi"}));
        assert_eq!(a.status, 404, "{path} now answers: add its golden shape to tests/regression_shapes.rs, its row to \
                                   tests/regression_api.rs and its call to `am2strata-run regression`");
    }
}

// --------------------------------------------------------------------------------------------------------- Anthropic

fn weather_anthropic() -> Value {
    json!([{"name": "get_weather", "description": "Weather.", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}])
}

#[test]
fn anthropic_messages_whole_streamed_tool_use_and_count() {
    let s = Server::start(&[Model::default()], &[ids(ANSWER), ids(ANSWER), ids("Bonjour!"), ids(CALL), ids(CALL)], json!({}));
    let think = json!({"model": "x", "max_tokens": 100, "messages": [user("What is 2+2?")]});
    let r = ok_json(&post(&s, "/v1/messages", &think));
    assert_eq!((r["type"].as_str(), r["role"].as_str()), (Some("message"), Some("assistant")));
    assert!(r["id"].as_str().unwrap().starts_with("msg_"), "{}", r["id"]);
    check_golden("anthropic_message", &named_shape(&r));
    let mut st = think.clone();
    st["stream"] = json!(true);
    let ev = ok_stream(&post(&s, "/v1/messages", &st));
    let names: Vec<&str> = ev.iter().map(|(n, _)| n.as_str()).filter(|n| *n != "ping").collect();
    assert_eq!((names.first(), names.last()), (Some(&"message_start"), Some(&"message_stop")));
    for (n, v) in &ev {
        assert_eq!(v["type"], n.as_str(), "an event's name is its data's type");
    }
    check_golden("anthropic_message_stream", &stream_shape(&ev.into_iter().filter(|(n, _)| n != "ping").collect::<Vec<_>>()));

    let plain = json!({"model": "x", "max_tokens": 50, "thinking": {"type": "disabled"}, "messages": [user("Hello in French?")]});
    check_golden("anthropic_message_no_thinking", &named_shape(&ok_json(&post(&s, "/v1/messages", &plain))));

    let tool = json!({"model": "x", "max_tokens": 200, "messages": [user("Weather?")], "tools": weather_anthropic()});
    let r = ok_json(&post(&s, "/v1/messages", &tool));
    assert_eq!(r["stop_reason"], "tool_use", "{r}");
    check_golden("anthropic_message_tool_use", &named_shape(&r));
    let mut st = tool.clone();
    st["stream"] = json!(true);
    let ev = ok_stream(&post(&s, "/v1/messages", &st));
    check_golden("anthropic_message_tool_use_stream", &stream_shape(&ev.into_iter().filter(|(n, _)| n != "ping").collect::<Vec<_>>()));

    check_golden("anthropic_count_tokens", &named_shape(&ok_json(&post(&s, "/v1/messages/count_tokens", &json!({"model": "x", "messages": [user("How many?")]})))));
}

#[test]
fn anthropic_error_bodies() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    let mut errors = serde_json::Map::new();
    let mut add = |name: &str, a: Answer, status: u16| {
        assert_eq!(a.status, status, "{name}: {}", a.text);
        assert!(a.json()["error"]["message"].as_str().is_some_and(|m| !m.is_empty()), "{name}: error.message ({})", a.text);
        errors.insert(name.to_string(), json!({"status": a.status, "body": named_shape(&a.json())}));
    };
    let ok = json!({"model": "x", "max_tokens": 8, "messages": [user("hi")]});
    add("no_key", Req::new("POST", &s.base, "/v1/messages").json(&ok).send(), 401);
    add("wrong_x_api_key", Req::new("POST", &s.base, "/v1/messages").header("x-api-key", WRONG_KEY).json(&ok).send(), 401);
    add("messages_missing", post(&s, "/v1/messages", &json!({"model": "x", "max_tokens": 8})), 400);
    add("over_the_context", post(&s, "/v1/messages", &json!({"model": "x", "max_tokens": 50000, "messages": [user("hi")]})), 400);
    add("broken_json", Req::new("POST", &s.base, "/v1/messages").key(Some(KEY)).raw("application/json", b"{").send(), 400);
    check_golden("anthropic_errors", &Value::Object(errors));
    // the key in Anthropic's own header is accepted
    assert_eq!(Req::new("POST", &s.base, "/v1/messages").header("x-api-key", KEY).json(&ok).send().status, 200);
}

// --------------------------------------------------------------------------------------------------------- Responses

#[test]
fn responses_whole_streamed_and_function_calls() {
    let s = Server::start(&[Model::default()], &[ids(ANSWER), ids(ANSWER), ids(CALL), ids(CALL)], json!({}));
    let req = json!({"instructions": "Be brief.", "input": "What is 2+2?", "max_output_tokens": 100});
    let r = ok_json(&post(&s, "/v1/responses", &req));
    assert_eq!((r["object"].as_str(), r["status"].as_str()), (Some("response"), Some("completed")));
    assert!(r["id"].as_str().unwrap().starts_with("resp_"), "{}", r["id"]);
    check_golden("responses", &named_shape(&r));
    let mut st = req.clone();
    st["stream"] = json!(true);
    let ev = ok_stream(&post(&s, "/v1/responses", &st));
    for (n, v) in &ev {
        assert_eq!(v["type"], n.as_str(), "an event's name is its data's type");
    }
    let seq: Vec<u64> = ev.iter().filter_map(|(_, v)| v["sequence_number"].as_u64()).collect();
    assert!(seq.len() == ev.len() && seq.windows(2).all(|w| w[1] == w[0] + 1), "sequence numbers count up: {seq:?}");
    check_golden("responses_stream", &stream_shape(&ev));

    let tools = json!([{"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]);
    let call = json!({"input": [{"role": "user", "content": "Weather in Pune?"}], "tools": tools});
    let r = ok_json(&post(&s, "/v1/responses", &call));
    assert!(r["output"].as_array().unwrap().iter().any(|i| i["type"] == "function_call"), "{r}");
    check_golden("responses_function_call", &named_shape(&r));
    let mut st = call.clone();
    st["stream"] = json!(true);
    check_golden("responses_function_call_stream", &stream_shape(&ok_stream(&post(&s, "/v1/responses", &st))));
}

#[test]
fn responses_error_bodies() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    let mut errors = serde_json::Map::new();
    let mut add = |name: &str, a: Answer, status: u16| {
        assert_eq!(a.status, status, "{name}: {}", a.text);
        assert!(a.json()["error"]["message"].as_str().is_some_and(|m| !m.is_empty()), "{name}: error.message ({})", a.text);
        errors.insert(name.to_string(), json!({"status": a.status, "body": named_shape(&a.json())}));
    };
    add("no_key", Req::new("POST", &s.base, "/v1/responses").json(&json!({"input": "hi"})).send(), 401);
    add("stored_response_asked_for", post(&s, "/v1/responses", &json!({"input": "hi", "previous_response_id": "resp_1"})), 400);
    add("get_a_response", Req::new("GET", &s.base, "/v1/responses/resp_1").key(Some(KEY)).send(), 404);
    add("input_missing", post(&s, "/v1/responses", &json!({})), 400);
    check_golden("responses_errors", &Value::Object(errors));
}

// ------------------------------------------------------------------------------------------------------------ Gemini

#[test]
fn gemini_generate_stream_count_models_and_function_calls() {
    let s = Server::start(&[Model::default()], &[ids(ANSWER), ids(ANSWER), ids(CALL)], json!({}));
    let g = |path: &str, body: &Value| Req::new("POST", &s.base, path).header("x-goog-api-key", KEY).json(body).send();
    let body = json!({"contents": [{"role": "user", "parts": [{"text": "2+2?"}]}], "generationConfig": {"maxOutputTokens": 20}});
    let r = ok_json(&g("/v1beta/models/gemini-2.5-pro:generateContent", &body));
    assert_eq!(r["candidates"][0]["finishReason"], "STOP");
    check_golden("gemini_generate_content", &named_shape(&r));
    let ev = ok_stream(&g("/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse", &body));
    check_golden("gemini_stream_generate_content", &stream_shape(&ev));
    check_golden("gemini_count_tokens", &named_shape(&ok_json(&g("/v1beta/models/x:countTokens", &body))));
    let get = |path: &str| Req::new("GET", &s.base, path).header("x-goog-api-key", KEY).send();
    check_golden("gemini_models", &named_shape(&ok_json(&get("/v1beta/models"))));
    check_golden("gemini_model", &named_shape(&ok_json(&get("/v1beta/models/mock-a"))));

    let tools = json!({"contents": [{"role": "user", "parts": [{"text": "Weather in Pune?"}]}],
        "tools": [{"functionDeclarations": [{"name": "get_weather", "description": "Weather.",
                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]}]});
    let r = ok_json(&g("/v1beta/models/gemini-2.5-pro:generateContent", &tools));
    let parts = r["candidates"][0]["content"]["parts"].as_array().cloned().unwrap_or_default();
    assert!(parts.iter().any(|p| p["functionCall"]["name"] == "get_weather"), "{r}");
    check_golden("gemini_function_call", &named_shape(&r));
}

#[test]
fn gemini_error_bodies() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    let body = json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]});
    let mut errors = serde_json::Map::new();
    let mut add = |name: &str, a: Answer, status: u16| {
        assert_eq!(a.status, status, "{name}: {}", a.text);
        let j = a.json();
        assert!(j["error"]["message"].as_str().is_some_and(|m| !m.is_empty()) && j["error"]["code"].as_u64() == Some(status as u64)
                && j["error"]["status"].is_string(), "{name}: Gemini's error is {{error: {{code, message, status}}}} ({})", a.text);
        errors.insert(name.to_string(), json!({"status": a.status, "body": named_shape(&j)}));
    };
    add("no_key", Req::new("POST", &s.base, "/v1beta/models/x:generateContent").json(&body).send(), 401);
    add("wrong_key_in_query", Req::new("POST", &s.base, &format!("/v1beta/models/x:generateContent?key={WRONG_KEY}")).json(&body).send(), 401);
    add("unknown_method", Req::new("POST", &s.base, "/v1beta/models/x:embedContent").header("x-goog-api-key", KEY).json(&body).send(), 404);
    add("broken_json", Req::new("POST", &s.base, "/v1beta/models/x:generateContent").header("x-goog-api-key", KEY).raw("application/json", b"{").send(), 400);
    add("models_no_key", Req::new("GET", &s.base, "/v1beta/models").send(), 401);
    check_golden("gemini_errors", &Value::Object(errors));
}

// ------------------------------------------------------------------------------------------------- the goldens' own

/// Every golden file is read by a test (a file nobody compares is not a contract), and every one is a JSON shape.
#[test]
fn no_golden_file_is_orphaned() {
    let me = include_str!("regression_shapes.rs");
    let mut n = 0;
    for e in std::fs::read_dir(golden_dir()).expect("tests/golden").flatten() {
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) != Some("json") {
            continue;
        }
        let name = p.file_stem().unwrap().to_string_lossy().into_owned();
        assert!(me.contains(&format!("check_golden(\"{name}\"")), "tests/golden/{name}.json is compared by no test: remove it or check it");
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(serde_json::from_str::<Value>(&text).is_ok(), "{name}.json is not JSON");
        n += 1;
    }
    assert!(n >= 20, "only {n} golden files");
}

/// The shape function itself: values are gone, types and naming fields stay, a list is the set of its items' shapes.
#[test]
fn a_shape_keeps_types_and_names_and_drops_values() {
    let a = json!({"id": "chatcmpl-1", "object": "chat.completion", "created": 1731500000, "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"},
        {"index": 1, "message": {"role": "assistant", "content": "ho"}, "finish_reason": "stop"}], "usage": {"speed": 1.5}});
    let b = json!({"id": "chatcmpl-2", "object": "chat.completion", "created": 9, "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "other"}, "finish_reason": "stop"}], "usage": {"speed": 99.25}});
    assert_eq!(named_shape(&a), named_shape(&b), "ids, times, texts and the number of items are not in a shape");
    assert_eq!(named_shape(&a)["object"], "=chat.completion");
    assert_eq!(named_shape(&a)["choices"][0]["message"], json!({"content": "string", "role": "=assistant"}));
    let mut c = b.clone();
    c["choices"][0]["message"]["content"] = Value::Null;
    assert_ne!(named_shape(&b), named_shape(&c), "a retyped field is a change");
    let mut d = b.clone();
    d["usage"]["speed"] = json!(3);
    assert_ne!(named_shape(&b), named_shape(&d), "an integer where a float was is a change");
    let mut diff = Vec::new();
    differences("", &named_shape(&b), &named_shape(&c), &mut diff);
    assert_eq!(diff, [".choices[].message.content: was \"string\", now \"null\""]);
}
