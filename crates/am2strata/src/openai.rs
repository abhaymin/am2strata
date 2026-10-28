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
        .models()
        .iter()
        .map(|m| json!({"id": m.id, "object": "model", "created": now(), "owned_by": "am2strata",
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

pub async fn chat(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    // a stream that must wait for the model to load again (stopped for a media job) behind other jobs starts at once
    // and says its place (SSE comments); the answer follows on the same stream once the model is loaded
    if stream && s.mgr.current().await.is_none() && s.mgr.queue.busy() {
        return waiting_stream(s, headers, req);
    }
    chat_with(s, headers, req, None).await
}

fn waiting_stream(s: AppState, headers: HeaderMap, req: Value) -> Response {
    use futures::StreamExt;
    let (tx, rx) = tokio::sync::mpsc::channel::<axum::body::Bytes>(256);
    let keep = tx.clone();
    tokio::spawn(async move {
        let txp = tx.clone();
        let got = s.mgr.ensure_with_turn_told(move |p| {
            let _ = txp.try_send(format!(": queued: {} ahead\n\n", p.saturating_sub(1)).into());
        }).await;
        let r = match got {
            Ok(l) => chat_with(s, headers, req, Some(l)).await,
            Err(e) => error(StatusCode::SERVICE_UNAVAILABLE, "server_error", format!("{e:#}")),
        };
        let ok = r.status().is_success();
        let mut body = r.into_body().into_data_stream();
        if ok {
            while let Some(Ok(b)) = body.next().await {
                if tx.send(b).await.is_err() {
                    return;
                }
            }
        } else {
            let mut all = Vec::new();
            while let Some(Ok(b)) = body.next().await {
                all.extend_from_slice(&b);
            }
            let v: Value = serde_json::from_slice(&all).unwrap_or_else(|_| json!({"error": {"message": String::from_utf8_lossy(&all)}}));
            let _ = tx.send(format!("data: {v}\n\ndata: [DONE]\n\n").into()).await;
        }
    });
    // keep-alives while it waits (the answer's stream has its own); they stop when the answer's stream ends
    let weak = keep.downgrade();
    drop(keep);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(10)).await;
            match weak.upgrade() {
                Some(tx) => { let _ = tx.try_send(axum::body::Bytes::from_static(b": keep-alive\n\n")); }
                None => return,
            }
        }
    });
    let body = futures::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|b| (Ok::<_, Infallible>(b), rx)) });
    Response::builder().header("content-type", "text/event-stream").header("cache-control", "no-cache")
        .body(axum::body::Body::from_stream(body)).unwrap()
}

async fn chat_with(s: AppState, headers: HeaderMap, mut req: Value,
                   pre: Option<(std::sync::Arc<crate::manager::Loaded>, Option<crate::queue::Pass>)>) -> Response {
    if let Err(e) = media::expand_videos(&mut req).await {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", e.to_string());
    }
    let got = match pre {
        Some(l) => Ok(l),
        None => s.mgr.ensure_with_turn().await,
    };
    let (loaded, early_pass) = match got {
        Ok(l) => l,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, "server_error", format!("{e:#}")),
    };
    if let Err(e) = crate::media_jobs::expand_audio(&s, &mut req).await {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", e);
    }
    if let Err(e) = media::expand_documents(&mut req, loaded.vision.is_some(), false).await {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", e.to_string());
    }
    {
        let shared = loaded.shared.read().unwrap().clone();
        crate::settings::with_shared(&mut req, &shared, loaded.config_effort().as_deref(), "openai");
    }
    let mut prep = match openai_to_messages(&req, &loaded.tool_orders) {
        Ok(p) => p,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e.0),
    };
    if req.get("tool_choice").and_then(Value::as_str) == Some("none") {
        prep.tools = None;
    }
    let validator = match crate::structured::prepare_format(req.get("response_format"), &mut prep.messages, false) {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    if validator.is_some() && (prep.tools.is_some() || req.get("strata_mcp") == Some(&Value::Bool(true))) {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "structured response_format with tools/MCP is not supported");
    }
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let own_page = !crate::foreign_origin(&s, &headers) && headers.contains_key("origin");
    let mut opts = run::RunOpts { stream_tools: stream, budget: None, tool_choice: req.get("tool_choice").cloned(),
                              memory: crate::memory::scope_of(&req, own_page), pass: std::sync::Arc::new(std::sync::Mutex::new(early_pass)),
                                  media: None };
    // the web app's opt-in: MCP tools run here, with the user's rights on this PC, so only for its own page
    let mut mcp = None;
    if let (Some(true), Some(hub)) = (req.get("strata_mcp").and_then(Value::as_bool), s.mcp.clone()) {
        if crate::foreign_origin(&s, &headers) {
            return crate::not_own_page("MCP tools can be used");
        }
        hub.wait(Duration::from_secs(10)).await; // servers still starting (only right after the start)
        let own: Vec<String> = prep.tools.iter().flatten().filter_map(|t| t.get("name").and_then(Value::as_str).map(str::to_owned)).collect();
        let extra = hub.template_tools(&own); // the request's own tools win a name clash
        if !extra.is_empty() {
            if strata_frontend::request::forced_call(opts.tool_choice.as_ref(), prep.tools.as_deref()).is_some() {
                return error(StatusCode::BAD_REQUEST, "invalid_request_error", "a forced tool_choice with MCP tools is not supported");
            }
            let names = extra.iter().filter_map(|t| t["name"].as_str().map(str::to_owned)).collect();
            prep.tools.get_or_insert_with(Vec::new).extend(extra);
            mcp = Some((hub, names));
        }
    }
    // the memory tools (remember / recall / forget), run here like MCP tools, for a request that opted in
    let mut mcp = mcp.map(|(h, n)| (Some(h), n));
    if opts.memory.is_some() && s.mgr.memory.settings.read().unwrap().tools && validator.is_none() {
        let own: Vec<String> = prep.tools.iter().flatten().filter_map(|t| t.get("name").and_then(Value::as_str).map(str::to_owned)).collect();
        let extra: Vec<Value> = crate::memory::tools().into_iter().filter(|t| !own.iter().any(|o| t["name"] == o.as_str())).collect();
        if !extra.is_empty() && strata_frontend::request::forced_call(opts.tool_choice.as_ref(), prep.tools.as_deref()).is_none() {
            let (hub, mut names) = mcp.take().unwrap_or((None, std::collections::HashSet::new()));
            names.extend(extra.iter().filter_map(|t| t["name"].as_str().map(str::to_owned)));
            prep.tools.get_or_insert_with(Vec::new).extend(extra);
            mcp = Some((hub, names));
        }
    }
    // the media tools (generate_image / generate_video / speak) for a chat that opted in, the kinds this PC can make
    if req.get("strata_media") == Some(&Value::Bool(true)) && validator.is_none() && s.media.available()
        && strata_frontend::request::forced_call(opts.tool_choice.as_ref(), prep.tools.as_deref()).is_none() {
        let caps = s.media_jobs.capabilities().await;
        let own: Vec<String> = prep.tools.iter().flatten().filter_map(|t| t.get("name").and_then(Value::as_str).map(str::to_owned)).collect();
        let extra: Vec<Value> = crate::media_jobs::Jobs::tools(&caps).into_iter().filter(|t| !own.iter().any(|o| t["name"] == o.as_str())).collect();
        if !extra.is_empty() {
            let (hub, mut names) = mcp.take().unwrap_or((None, std::collections::HashSet::new()));
            names.extend(extra.iter().filter_map(|t| t["name"].as_str().map(str::to_owned)));
            prep.tools.get_or_insert_with(Vec::new).extend(extra);
            mcp = Some((hub, names));
            opts.media = Some(caps);
        }
    }
    let pass_slot = opts.pass.clone();
    let has_pass = pass_slot.lock().unwrap().is_some();
    let (mgr, req2, model_id) = (s.mgr.clone(), req.clone(), loaded.entry.id.clone());
    let run_it = async move {
        match mcp {
            Some((hub, names)) => run::start_mcp(mgr, loaded, prep, req2, opts, hub, names).await,
            None => run::start(mgr, loaded, prep, req2, opts).await,
        }
    };
    // a stream that has to wait for its turn starts at once and says its place in the queue (SSE comments); its
    // refusals then come as an error event, not a status code
    if stream && validator.is_none() && !has_pass && s.mgr.queue.busy() {
        let (tx, rx) = tokio::sync::mpsc::channel::<Option<Value>>(256);
        let mgr = s.mgr.clone();
        tokio::spawn(async move {
            let txp = tx.clone();
            let label = format!("{model_id} chat (stream)");
            let pass = mgr.take_turn(&model_id, &label, move |p| {
                let _ = txp.try_send(Some(json!({"__comment": format!("queued: {} ahead", p.saturating_sub(1))})));
            }).await;
            match pass {
                Ok(p) => *pass_slot.lock().unwrap() = Some(p),
                Err(t) => {
                    let _ = tx.send(Some(json!({"error": {"type": "server_error", "message":
                        format!("waited {} s at place {} of the queue (queue_max_wait_s); try again later", t.waited_s, t.position)}}))).await;
                    return;
                }
            }
            match run_it.await {
                Ok(st) => pump(st, tx).await,
                Err(run::Refusal(_, msg)) => {
                    let _ = tx.send(Some(json!({"error": {"type": "invalid_request_error", "message": msg}}))).await;
                }
            }
        });
        return sse(rx);
    }
    let started = match run_it.await {
        Ok(st) => st,
        Err(run::Refusal(code, msg)) => {
            return error(StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_REQUEST), "invalid_request_error", msg)
        }
    };
    let (tx, rx) = tokio::sync::mpsc::channel::<Option<Value>>(256);
    tokio::spawn(pump(started, tx));
    let rx = match validator {
        Some(v) => structured(rx, v),
        None => rx,
    };
    if stream {
        return sse(rx);
    }
    collect(rx).await
}

/// The engine's events of one request as chat.completion chunks (None: a keep-alive).
async fn pump(started: run::Started, tx: tokio::sync::mpsc::Sender<Option<Value>>) {
    let cid = format!("chatcmpl-{}", &uuid::Uuid::new_v4().simple().to_string()[..24]);
    let created = now();
    let model = started.model.clone();
    let prompt_tokens = started.prompt_tokens;
    let chunk = move |delta: Value, finish: Option<&str>| {
        json!({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    };
    let mut events = started.events;
    {
        let _ = tx.send(Some(chunk(json!({"role": "assistant", "content": ""}), None))).await;
        let mut calls = 0usize;
        let mut streamed: HashMap<String, usize> = HashMap::new();
        let mut finished: Vec<String> = Vec::new();
        while let Some(ev) = events.recv().await {
            let out = match ev {
                RunEvent::Ping => None,
                RunEvent::Mcp(v) => {
                    let mut c = chunk(json!({}), None);
                    c["strata_mcp"] = v;
                    Some(c)
                }
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
    }
}

/// Chunks as server-sent events, ending with [DONE]; a {"__comment"} chunk is an SSE comment (the queue position).
fn sse(rx: tokio::sync::mpsc::Receiver<Option<Value>>) -> Response {
    let stream = futures::stream::unfold(rx, |mut rx| async move {
        match rx.recv().await {
            Some(Some(v)) if v.get("__comment").is_some() => Some((Ok::<_, Infallible>(SseEvent::default().comment(v["__comment"].as_str().unwrap_or(""))), rx)),
            Some(Some(v)) => Some((Ok::<_, Infallible>(SseEvent::default().data(v.to_string())), rx)),
            Some(None) => Some((Ok(SseEvent::default().comment("keep-alive")), rx)),
            None => None,
        }
    });
    let done = futures::stream::once(async { Ok::<_, Infallible>(SseEvent::default().data("[DONE]")) });
    use futures::StreamExt;
    Sse::new(stream.chain(done)).keep_alive(KeepAlive::new().interval(Duration::from_secs(10))).into_response()
}

/// Structured output: the chunks are held (keep-alives go on) until the answer is checked; then the first chunk, one
/// with the whole checked content, and the last (serve/server.py structured_chunks). A failed check is an error
/// chunk (or a 502 when not streaming).
fn structured(mut rx: tokio::sync::mpsc::Receiver<Option<Value>>, v: crate::structured::Validator)
              -> tokio::sync::mpsc::Receiver<Option<Value>> {
    let (tx, out) = tokio::sync::mpsc::channel::<Option<Value>>(256);
    tokio::spawn(async move {
        let mut held: Vec<Value> = Vec::new();
        while let Some(c) = rx.recv().await {
            match c {
                Some(c) if c.get("error").is_some() => {
                    let _ = tx.send(Some(c)).await;
                    return;
                }
                Some(c) => held.push(c),
                None => {}
            }
            if tx.send(None).await.is_err() {
                return; // the client is gone
            }
        }
        let (Some(first), Some(last)) = (held.first().cloned(), held.last().cloned()) else { return };
        let (mut content, mut reasoning) = (String::new(), String::new());
        for c in &held {
            let d = &c["choices"][0]["delta"];
            content.push_str(d.get("content").and_then(Value::as_str).unwrap_or(""));
            reasoning.push_str(d.get("reasoning_content").and_then(Value::as_str).unwrap_or(""));
        }
        let finish = last["choices"][0]["finish_reason"].as_str().unwrap_or("stop").to_string();
        match crate::structured::validated_json(&content, &v, &finish) {
            Ok(text) => {
                let mut delta = json!({"content": text});
                if !reasoning.is_empty() {
                    delta["reasoning_content"] = json!(reasoning);
                }
                let mut mid = first.clone();
                mid["choices"] = json!([{"index": 0, "delta": delta, "finish_reason": null}]);
                for c in [first, mid, last] {
                    let _ = tx.send(Some(c)).await;
                }
            }
            Err(crate::structured::StructuredError(m)) => {
                let _ = tx.send(Some(json!({"error": {"type": "structured_output_failed", "code": "structured_output_failed", "message": m}}))).await;
            }
        }
    });
    out
}

/// The streamed chunks merged into one chat.completion (serve/server.py openai_collect).
async fn collect(mut rx: tokio::sync::mpsc::Receiver<Option<Value>>) -> Response {
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut by_index: std::collections::BTreeMap<u64, Map<String, Value>> = Default::default();
    let mut last: Option<Value> = None;
    while let Some(c) = rx.recv().await {
        let Some(c) = c else { continue };
        if let Some(e) = c.get("error") {
            let msg = e["message"].as_str().unwrap_or("error").to_string();
            if e["type"] == "structured_output_failed" {
                return (StatusCode::BAD_GATEWAY, Json(json!({"error": {"type": "structured_output_failed",
                        "code": "structured_output_failed", "message": msg}}))).into_response();
            }
            return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", msg);
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
