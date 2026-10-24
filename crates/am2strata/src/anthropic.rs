//! Anthropic Messages: /v1/messages (streaming and not) and /v1/messages/count_tokens.

use crate::run::{self, RunEvent};
use crate::{authorized, media, AppState};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::convert::Infallible;
use std::time::Duration;
use strata_frontend::output::{py_dumps, Event};
use strata_frontend::request::anthropic_to_messages;

fn anthropic_error(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (status, Json(json!({"type": "error", "error": {"type": kind, "message": msg.into()}}))).into_response()
}

pub async fn count_tokens(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return anthropic_error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let loaded = match s.mgr.ensure().await {
        Ok(l) => l,
        Err(e) => return anthropic_error(StatusCode::SERVICE_UNAVAILABLE, "api_error", format!("{e:#}")),
    };
    match anthropic_to_messages(&req, &loaded.tool_orders, true).and_then(|(p, _)| loaded.prompts.build(&p)) {
        Ok(p) => Json(json!({"input_tokens": p.ids.len()})).into_response(),
        Err(e) => anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", e.0),
    }
}

pub async fn messages(State(s): State<AppState>, headers: HeaderMap, Json(mut req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return anthropic_error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if let Err(e) = media::expand_videos(&mut req).await {
        return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", e.to_string());
    }
    let loaded = match s.mgr.ensure().await {
        Ok(l) => l,
        Err(e) => return anthropic_error(StatusCode::SERVICE_UNAVAILABLE, "api_error", format!("{e:#}")),
    };
    if let Err(e) = media::expand_documents(&mut req, loaded.vision.is_some(), true).await {
        return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", e.to_string());
    }
    {
        let shared = loaded.shared.read().unwrap().clone();
        crate::settings::with_shared(&mut req, &shared, loaded.config_effort().as_deref(), "anthropic");
    }
    let (mut prep, budget) = match anthropic_to_messages(&req, &loaded.tool_orders, true) {
        Ok(p) => p,
        Err(e) => return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", e.0),
    };
    if req.get("tool_choice").and_then(|t| t.get("type")).and_then(Value::as_str) == Some("none") {
        prep.tools = None;
    }
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let opts = run::RunOpts { stream_tools: stream, budget, tool_choice: req.get("tool_choice").cloned(),
                           memory: crate::memory::scope_of(&req, !crate::foreign_origin(&s, &headers) && headers.contains_key("origin")) };
    let started = match run::start(s.mgr.clone(), loaded, prep, req.clone(), opts).await {
        Ok(st) => st,
        Err(run::Refusal(code, msg)) => {
            return anthropic_error(StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_REQUEST), "invalid_request_error", msg)
        }
    };
    let mid = format!("msg_{}", &uuid::Uuid::new_v4().simple().to_string()[..24]);
    let model = started.model.clone();
    let n_prompt = started.prompt_tokens as u64;
    let (tx, rx) = tokio::sync::mpsc::channel::<Option<(String, Value)>>(256);
    let mut events = started.events;
    tokio::spawn(async move {
        let send = |name: &str, v: Value| Some((name.to_string(), v));
        let _ = tx
            .send(send("message_start", json!({"type": "message_start", "message": {
                "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
                "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": n_prompt, "output_tokens": 0}}})))
            .await;
        let mut index: i64 = -1;
        let mut open: Option<&'static str> = None;
        let mut used_tool = false;
        let (mut streamed, mut finished) = (HashSet::new(), HashSet::new());
        while let Some(ev) = events.recv().await {
            let mut out: Vec<Option<(String, Value)>> = Vec::new();
            match ev {
                RunEvent::Ping | RunEvent::Mcp(_) => out.push(None),
                RunEvent::Event(Event::ToolArgs { text, .. }) => out.push(send("content_block_delta", json!({
                    "type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": text}}))),
                RunEvent::Event(Event::ToolCall(c)) if streamed.contains(&c.id) => {
                    finished.insert(c.id);
                }
                RunEvent::Event(ev) => {
                    let (want, text, call): (&'static str, String, Option<(String, String, Option<String>)>) = match ev {
                        Event::Reasoning(t) => ("thinking", t, None),
                        Event::Content(t) => ("text", t, None),
                        Event::ToolStart { id, name } => {
                            streamed.insert(id.clone());
                            used_tool = true;
                            ("tool_use", String::new(), Some((id, name, None)))
                        }
                        Event::ToolCall(c) => {
                            used_tool = true;
                            let args = py_dumps(&Value::Object(c.arguments), false);
                            ("tool_use", String::new(), Some((c.id, c.name, Some(args))))
                        }
                        Event::ToolArgs { .. } => unreachable!(),
                    };
                    if call.is_none() && text.is_empty() {
                        continue;
                    }
                    if open != Some(want) || want == "tool_use" {
                        if open.is_some() {
                            out.push(send("content_block_stop", json!({"type": "content_block_stop", "index": index})));
                        }
                        index += 1;
                        open = Some(want);
                        let block = match want {
                            "thinking" => json!({"type": "thinking", "thinking": "", "signature": ""}),
                            "text" => json!({"type": "text", "text": ""}),
                            _ => {
                                let (id, name, _) = call.clone().unwrap();
                                json!({"type": "tool_use", "id": id, "name": name, "input": {}})
                            }
                        };
                        out.push(send("content_block_start", json!({"type": "content_block_start", "index": index, "content_block": block})));
                    }
                    match want {
                        "thinking" => out.push(send("content_block_delta", json!({"type": "content_block_delta", "index": index,
                                                     "delta": {"type": "thinking_delta", "thinking": text}}))),
                        "text" => out.push(send("content_block_delta", json!({"type": "content_block_delta", "index": index,
                                                 "delta": {"type": "text_delta", "text": text}}))),
                        _ => {
                            if let Some((_, _, Some(args))) = call {
                                out.push(send("content_block_delta", json!({"type": "content_block_delta", "index": index,
                                                "delta": {"type": "input_json_delta", "partial_json": args}})));
                            }
                        }
                    }
                }
                RunEvent::Done(sum) => {
                    if open.is_some() {
                        out.push(send("content_block_stop", json!({"type": "content_block_stop", "index": index})));
                    }
                    let stop = if sum.stop_sequence.is_some() {
                        "stop_sequence"
                    } else if used_tool && streamed.is_subset(&finished) && sum.finish == "stop" {
                        "tool_use"
                    } else {
                        match sum.finish.as_str() { "length" => "max_tokens", _ => "end_turn" }
                    };
                    let reused = sum.reused.min(n_prompt.saturating_sub(1));
                    out.push(send("message_delta", json!({"type": "message_delta",
                        "delta": {"stop_reason": stop, "stop_sequence": sum.stop_sequence},
                        "usage": {"input_tokens": n_prompt - reused, "cache_read_input_tokens": reused,
                                  "output_tokens": sum.completion_tokens}})));
                    out.push(send("message_stop", json!({"type": "message_stop"})));
                    for o in out {
                        let _ = tx.send(o).await;
                    }
                    return;
                }
                RunEvent::Error(msg) => {
                    let _ = tx.send(send("error", json!({"type": "error", "error": {"type": "api_error", "message": msg}}))).await;
                    return;
                }
            }
            for o in out {
                if tx.send(o).await.is_err() {
                    return;
                }
            }
        }
    });
    if stream {
        let stream = futures::stream::unfold(rx, |mut rx| async move {
            match rx.recv().await {
                Some(Some((name, v))) => Some((Ok::<_, Infallible>(SseEvent::default().event(name).data(v.to_string())), rx)),
                Some(None) => Some((Ok(SseEvent::default().event("ping").data(r#"{"type": "ping"}"#)), rx)),
                None => None,
            }
        });
        return Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10))).into_response();
    }
    collect(rx).await
}

/// The events merged into one message (serve/server.py anthropic_collect).
async fn collect(mut rx: tokio::sync::mpsc::Receiver<Option<(String, Value)>>) -> Response {
    let mut msg: Option<Value> = None;
    let mut blocks: Vec<Value> = Vec::new();
    let mut json_buf: Option<String> = None;
    while let Some(item) = rx.recv().await {
        let Some((name, e)) = item else { continue };
        match name.as_str() {
            "message_start" => msg = Some(e["message"].clone()),
            "content_block_start" => blocks.push(e["content_block"].clone()),
            "content_block_delta" => {
                let d = &e["delta"];
                let Some(b) = blocks.last_mut() else { continue };
                match d["type"].as_str() {
                    Some("text_delta") => b["text"] = json!(format!("{}{}", b["text"].as_str().unwrap_or(""), d["text"].as_str().unwrap_or(""))),
                    Some("thinking_delta") => b["thinking"] = json!(format!("{}{}", b["thinking"].as_str().unwrap_or(""), d["thinking"].as_str().unwrap_or(""))),
                    _ => json_buf = Some(json_buf.unwrap_or_default() + d["partial_json"].as_str().unwrap_or("")),
                }
            }
            "content_block_stop" => {
                if let Some(j) = json_buf.take() {
                    match serde_json::from_str::<Value>(if j.is_empty() { "{}" } else { &j }) {
                        Ok(v) => {
                            if let Some(b) = blocks.last_mut() {
                                b["input"] = v;
                            }
                        }
                        Err(_) => {
                            blocks.pop(); // a call the output ended inside: it has no input to give
                        }
                    }
                }
            }
            "message_delta" => {
                if let Some(m) = msg.as_mut() {
                    m["stop_reason"] = e["delta"]["stop_reason"].clone();
                    m["stop_sequence"] = e["delta"]["stop_sequence"].clone();
                    if let (Some(u), Some(add)) = (m["usage"].as_object_mut(), e["usage"].as_object()) {
                        for (k, v) in add {
                            u.insert(k.clone(), v.clone());
                        }
                    }
                }
            }
            "error" => return anthropic_error(StatusCode::INTERNAL_SERVER_ERROR, "api_error",
                                              e["error"]["message"].as_str().unwrap_or("error").to_string()),
            _ => {}
        }
    }
    let Some(mut m) = msg else {
        return anthropic_error(StatusCode::INTERNAL_SERVER_ERROR, "api_error", "no answer");
    };
    m["content"] = Value::Array(blocks);
    Json(m).into_response()
}
