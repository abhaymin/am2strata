//! OpenAI Chat Completions: /v1/chat/completions (streaming and not) and /v1/models.

use crate::run::{self, RunEvent, Summary};
use crate::{authorized, error, media, now, AppState};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::time::Duration;
use strata_frontend::openai_to_messages;
use strata_frontend::output::{py_dumps, Event};

pub async fn models(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let cur = s.mgr.current().await.map(|l| l.entry.id.clone());
    let data: Vec<Value> = s
        .mgr
        .models
        .iter()
        .map(|m| json!({"id": m.id, "object": "model", "created": now(), "owned_by": "polystrata",
                        "meta": {"n_ctx": m.cfg.max_context()},
                        "status": {"value": if cur.as_deref() == Some(m.id.as_str()) { "loaded" } else { "unloaded" }}}))
        .collect();
    Json(json!({"object": "list", "data": data})).into_response()
}

fn finish_reason(sum: &Summary, calls: usize, streamed_open: bool) -> &'static str {
    match sum.finish.as_str() {
        "stop" if calls > 0 && !streamed_open => "tool_calls",
        "length" => "length",
        _ => "stop",
    }
}

pub async fn chat(State(s): State<AppState>, headers: HeaderMap, Json(mut req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if let Err(e) = media::expand_videos(&mut req).await {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", e.to_string());
    }
    let loaded = match s.mgr.ensure().await {
        Ok(l) => l,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, "server_error", format!("{e:#}")),
    };
    let mut prep = match openai_to_messages(&req, &loaded.tool_orders) {
        Ok(p) => p,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e.0),
    };
    if req.get("tool_choice").and_then(Value::as_str) == Some("none") {
        prep.tools = None;
    }
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let started = match run::start(s.mgr.clone(), loaded, prep, req.clone(), stream).await {
        Ok(st) => st,
        Err(run::Refusal(code, msg)) => {
            return error(StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_REQUEST), "invalid_request_error", msg)
        }
    };
    let cid = format!("chatcmpl-{}", &uuid::Uuid::new_v4().simple().to_string()[..24]);
    let created = now();
    let model = started.model.clone();
    let prompt_tokens = started.prompt_tokens;
    let chunk = move |delta: Value, finish: Option<&str>| {
        json!({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    };
    let (tx, rx) = tokio::sync::mpsc::channel::<Option<Value>>(256);
    let mut events = started.events;
    tokio::spawn(async move {
        let _ = tx.send(Some(chunk(json!({"role": "assistant", "content": ""}), None))).await;
        let mut calls = 0usize;
        let mut streamed: HashMap<String, usize> = HashMap::new();
        let mut finished: Vec<String> = Vec::new();
        while let Some(ev) = events.recv().await {
            let out = match ev {
                RunEvent::Ping => None,
                RunEvent::Event(Event::Reasoning(t)) if !t.is_empty() => Some(chunk(json!({"reasoning_content": t}), None)),
                RunEvent::Event(Event::Content(t)) if !t.is_empty() => Some(chunk(json!({"content": t}), None)),
                RunEvent::Event(Event::ToolStart { id, name }) => {
                    streamed.insert(id.clone(), calls);
                    calls += 1;
                    Some(chunk(json!({"tool_calls": [{"index": streamed[&id], "id": id, "type": "function",
                                                      "function": {"name": name, "arguments": ""}}]}), None))
                }
                RunEvent::Event(Event::ToolArgs { id, text }) => {
                    let idx = streamed.get(&id).copied().unwrap_or(0);
                    Some(chunk(json!({"tool_calls": [{"index": idx, "function": {"arguments": text}}]}), None))
                }
                RunEvent::Event(Event::ToolCall(c)) if streamed.contains_key(&c.id) => {
                    finished.push(c.id);
                    None
                }
                RunEvent::Event(Event::ToolCall(c)) => {
                    let args = py_dumps(&Value::Object(c.arguments), false);
                    let v = chunk(json!({"tool_calls": [{"index": calls, "id": c.id, "type": "function",
                                                         "function": {"name": c.name, "arguments": args}}]}), None);
                    calls += 1;
                    Some(v)
                }
                RunEvent::Event(_) => None,
                RunEvent::Done(sum) => {
                    let open = streamed.keys().any(|k| !finished.contains(k));
                    let mut last = chunk(json!({}), Some(finish_reason(&sum, calls, open)));
                    let pt = prompt_tokens as u64;
                    last["usage"] = json!({"prompt_tokens": pt, "completion_tokens": sum.completion_tokens,
                                           "total_tokens": pt + sum.completion_tokens,
                                           "prompt_tokens_details": {"cached_tokens": sum.reused}});
                    if !sum.timings.is_null() {
                        last["timings"] = sum.timings.clone();
                    }
                    let _ = tx.send(Some(last)).await;
                    return;
                }
                RunEvent::Error(msg) => {
                    let _ = tx.send(Some(json!({"error": {"type": "server_error", "message": msg}}))).await;
                    return;
                }
            };
            if let Some(v) = out {
                if tx.send(Some(v)).await.is_err() {
                    return;
                }
            } else if tx.send(None).await.is_err() {
                return; // heartbeat failed: the client is gone
            }
        }
    });
    if stream {
        let stream = futures::stream::unfold(rx, |mut rx| async move {
            match rx.recv().await {
                Some(Some(v)) => Some((Ok::<_, Infallible>(SseEvent::default().data(v.to_string())), rx)),
                Some(None) => Some((Ok(SseEvent::default().comment("keep-alive")), rx)),
                None => None,
            }
        });
        let done = futures::stream::once(async { Ok::<_, Infallible>(SseEvent::default().data("[DONE]")) });
        use futures::StreamExt;
        return Sse::new(stream.chain(done)).keep_alive(KeepAlive::new().interval(Duration::from_secs(10))).into_response();
    }
    collect(rx).await
}

/// The streamed chunks merged into one chat.completion (serve/server.py openai_collect).
async fn collect(mut rx: tokio::sync::mpsc::Receiver<Option<Value>>) -> Response {
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut by_index: std::collections::BTreeMap<u64, Map<String, Value>> = Default::default();
    let mut last: Option<Value> = None;
    while let Some(c) = rx.recv().await {
        let Some(c) = c else { continue };
        if let Some(e) = c.get("error") {
            return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e["message"].as_str().unwrap_or("error").to_string());
        }
        let d = &c["choices"][0]["delta"];
        content.push_str(d.get("content").and_then(Value::as_str).unwrap_or(""));
        reasoning.push_str(d.get("reasoning_content").and_then(Value::as_str).unwrap_or(""));
        for tc in d.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
            let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(by_index.len() as u64);
            let cur = by_index.entry(idx).or_insert_with(|| {
                let mut m = Map::new();
                m.insert("id".into(), Value::Null);
                m.insert("type".into(), json!("function"));
                m.insert("function".into(), json!({"name": "", "arguments": ""}));
                m
            });
            if let Some(id) = tc.get("id").filter(|v| !v.is_null()) {
                cur.insert("id".into(), id.clone());
            }
            let f = tc.get("function").cloned().unwrap_or(json!({}));
            let name = format!("{}{}", cur["function"]["name"].as_str().unwrap_or(""), f.get("name").and_then(Value::as_str).unwrap_or(""));
            let args = format!("{}{}", cur["function"]["arguments"].as_str().unwrap_or(""), f.get("arguments").and_then(Value::as_str).unwrap_or(""));
            cur.insert("function".into(), json!({"name": name, "arguments": args}));
        }
        last = Some(c);
    }
    let Some(last) = last else {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "the engine ended the request without an answer");
    };
    let finish = last["choices"][0]["finish_reason"].clone();
    let mut calls: Vec<Value> = by_index.into_values().map(Value::Object).collect();
    if finish != "tool_calls" {
        calls.retain(|c| serde_json::from_str::<Value>(c["function"]["arguments"].as_str().unwrap_or("")).is_ok());
    }
    let mut msg = json!({"role": "assistant", "content": if content.is_empty() { Value::Null } else { json!(content) }});
    if !reasoning.is_empty() {
        msg["reasoning_content"] = json!(reasoning);
    }
    if !calls.is_empty() {
        msg["tool_calls"] = Value::Array(calls);
    }
    let mut out = json!({"id": last["id"], "object": "chat.completion", "created": last["created"], "model": last["model"],
                         "choices": [{"index": 0, "message": msg, "finish_reason": finish}]});
    if let Some(u) = last.get("usage") {
        out["usage"] = u.clone();
    }
    if let Some(t) = last.get("timings") {
        out["timings"] = t.clone();
    }
    Json(out).into_response()
}
