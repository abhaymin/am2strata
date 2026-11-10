//! The Gemini API (/v1beta/models/<model>:generateContent, :streamGenerateContent, :countTokens, GET /v1beta/models),
//! for the Gemini CLI and Google's SDKs (GOOGLE_GEMINI_BASE_URL=http://<this server>). A request becomes this server's
//! own OpenAI chat request (the same engine, tools, parser and records), and its answer goes back in Gemini's shape:
//! text, thoughts (`thought: true`), function calls, usage. The key comes as `x-goog-api-key` or `?key=`.
//!
//! Function calls are sent whole: Gemini has no partial-argument deltas, so a streamed call comes in the chunk where
//! its arguments end. countTokens counts the text and function parts with the loaded model's tokenizer.

use crate::AppState;
use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

fn gerror(status: StatusCode, msg: impl Into<String>) -> Response {
    let s = match status.as_u16() {
        400 => "INVALID_ARGUMENT",
        401 | 403 => "PERMISSION_DENIED",
        404 => "NOT_FOUND",
        429 => "RESOURCE_EXHAUSTED",
        503 => "UNAVAILABLE",
        _ => "INTERNAL",
    };
    (status, Json(json!({"error": {"code": status.as_u16(), "message": msg.into(), "status": s}}))).into_response()
}

/// Gemini's key (x-goog-api-key, or ?key=) as the Bearer the rest of the server checks.
fn with_key(headers: &HeaderMap, q: &HashMap<String, String>) -> HeaderMap {
    let mut h = headers.clone();
    let key = headers.get("x-goog-api-key").and_then(|v| v.to_str().ok()).map(str::to_string).or_else(|| q.get("key").cloned());
    if let Some(k) = key.filter(|_| !headers.contains_key("authorization")) {
        if let Ok(v) = HeaderValue::from_str(&format!("Bearer {k}")) {
            h.insert("authorization", v);
        }
    }
    h
}

/// A JSON schema in Gemini's OpenAPI dialect ("OBJECT", "STRING", nullable) in plain JSON schema.
fn schema(v: &Value) -> Value {
    match v {
        Value::Object(m) => {
            let mut out = Map::new();
            for (k, x) in m {
                match k.as_str() {
                    "type" => {
                        out.insert(k.clone(), x.as_str().map(|t| json!(t.to_lowercase())).unwrap_or_else(|| schema(x)));
                    }
                    "nullable" | "format" if !x.is_object() => {
                        if k == "format" {
                            out.insert(k.clone(), x.clone());
                        }
                    }
                    _ => {
                        out.insert(k.clone(), schema(x));
                    }
                }
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(schema).collect()),
        x => x.clone(),
    }
}

fn text_of(parts: &[Value]) -> String {
    parts.iter().filter(|p| p["thought"] != true).filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("")
}

/// A Gemini GenerateContentRequest as an OpenAI chat request.
pub fn to_openai(model: &str, g: &Value) -> Result<Value, String> {
    let mut messages = Vec::new();
    if let Some(parts) = g["systemInstruction"]["parts"].as_array().or_else(|| g["system_instruction"]["parts"].as_array()) {
        let t = text_of(parts);
        if !t.is_empty() {
            messages.push(json!({"role": "system", "content": t}));
        }
    }
    // the calls the model made, by name, waiting for their answers (Gemini's functionResponse names the function;
    // newer ones also carry the call's id)
    let mut pending: Vec<(String, String)> = Vec::new();
    let mut n = 0usize;
    for c in g["contents"].as_array().ok_or("contents: a list of turns")? {
        let parts = c["parts"].as_array().cloned().unwrap_or_default();
        let role = c["role"].as_str().unwrap_or("user");
        if role == "model" {
            let mut calls = Vec::new();
            for p in &parts {
                if let Some(f) = p.get("functionCall") {
                    n += 1;
                    let id = f["id"].as_str().map(str::to_string).unwrap_or_else(|| format!("call_{n}"));
                    let name = f["name"].as_str().unwrap_or("").to_string();
                    pending.push((name.clone(), id.clone()));
                    calls.push(json!({"id": id, "type": "function",
                                      "function": {"name": name, "arguments": f.get("args").cloned().unwrap_or(json!({})).to_string()}}));
                }
            }
            let text = text_of(&parts);
            let mut m = json!({"role": "assistant", "content": if text.is_empty() && !calls.is_empty() { Value::Null } else { json!(text) }});
            if !calls.is_empty() {
                m["tool_calls"] = Value::Array(calls);
            }
            messages.push(m);
            continue;
        }
        let mut content = Vec::new();
        for p in &parts {
            if let Some(r) = p.get("functionResponse") {
                let name = r["name"].as_str().unwrap_or("");
                let id = match r["id"].as_str() {
                    Some(id) => { pending.retain(|(_, i)| i != id); id.to_string() }
                    None => match pending.iter().position(|(nm, _)| nm == name) {
                        Some(i) => pending.remove(i).1,
                        None => { n += 1; format!("call_{n}") }
                    },
                };
                let out = &r["response"];
                let text = out.get("output").and_then(Value::as_str).map(str::to_string)
                    .or_else(|| out.get("content").and_then(Value::as_str).map(str::to_string))
                    .unwrap_or_else(|| out.to_string());
                messages.push(json!({"role": "tool", "tool_call_id": id, "content": text}));
            } else if let Some(t) = p["text"].as_str() {
                content.push(json!({"type": "text", "text": t}));
            } else if let Some(d) = p.get("inlineData").or_else(|| p.get("inline_data")) {
                let mime = d["mimeType"].as_str().or_else(|| d["mime_type"].as_str()).unwrap_or("image/png");
                if mime.starts_with("image/") {
                    content.push(json!({"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{}", d["data"].as_str().unwrap_or(""))}}));
                } else {
                    return Err(format!("inlineData of type {mime}: only pictures are read"));
                }
            } else if p.get("fileData").is_some() {
                return Err("fileData (an uploaded file's URI) is not supported: send the bytes as inlineData".into());
            }
        }
        if !content.is_empty() {
            let only_text = content.iter().all(|c| c["type"] == "text");
            messages.push(json!({"role": "user", "content": if only_text {
                json!(content.iter().filter_map(|c| c["text"].as_str()).collect::<Vec<_>>().join(""))
            } else { Value::Array(content) }}));
        }
    }
    let mut req = json!({"messages": messages});
    if !model.is_empty() {
        req["model"] = json!(model);
    }
    let mut tools = Vec::new();
    // Gemini's own Google Search tool: the server's web search (allowed for a key whose profile has "web"), as
    // Chat Completions' web_search_options
    if g["tools"].as_array().into_iter().flatten().any(|t| ["google_search", "googleSearch", "google_search_retrieval", "googleSearchRetrieval"]
        .iter().any(|k| t.get(*k).is_some())) {
        req["web_search_options"] = json!({});
    }
    for t in g["tools"].as_array().cloned().unwrap_or_default() {
        for f in t["functionDeclarations"].as_array().or_else(|| t["function_declarations"].as_array()).cloned().unwrap_or_default() {
            let params = f.get("parametersJsonSchema").cloned().or_else(|| f.get("parameters").map(schema))
                .unwrap_or(json!({"type": "object", "properties": {}}));
            tools.push(json!({"type": "function", "function": {"name": f["name"], "description": f["description"].as_str().unwrap_or(""),
                                                               "parameters": params}}));
        }
    }
    if !tools.is_empty() {
        req["tools"] = Value::Array(tools);
        let fc = &g["toolConfig"]["functionCallingConfig"];
        match fc["mode"].as_str().unwrap_or("AUTO") {
            "NONE" => req["tool_choice"] = json!("none"),
            "ANY" => {
                let only = fc["allowedFunctionNames"].as_array().filter(|a| a.len() == 1).and_then(|a| a[0].as_str());
                req["tool_choice"] = match only {
                    Some(name) => json!({"type": "function", "function": {"name": name}}),
                    None => json!("required"),
                };
            }
            _ => {}
        }
    }
    let gc = &g["generationConfig"];
    for (from, to) in [("temperature", "temperature"), ("topP", "top_p"), ("topK", "top_k"), ("maxOutputTokens", "max_tokens"),
                       ("stopSequences", "stop"), ("seed", "seed"), ("presencePenalty", "presence_penalty"),
                       ("frequencyPenalty", "frequency_penalty")] {
        if let Some(v) = gc.get(from).filter(|v| !v.is_null()) {
            req[to] = v.clone();
        }
    }
    let think = &gc["thinkingConfig"];
    if let Some(b) = think["thinkingBudget"].as_i64() {
        // 0 turns thinking off; -1 is the model's own choice
        if b == 0 {
            req["chat_template_kwargs"] = json!({"enable_thinking": false});
        } else if b > 0 {
            req["reasoning_budget_tokens"] = json!(b);
        }
    }
    if gc["responseMimeType"] == "application/json" {
        let s = gc.get("responseJsonSchema").cloned().or_else(|| gc.get("responseSchema").map(schema));
        req["response_format"] = match s {
            Some(s) => json!({"type": "json_schema", "json_schema": {"name": "response", "schema": s}}),
            None => json!({"type": "json_object"}),
        };
    }
    Ok(req)
}

fn finish(reason: &str) -> &'static str {
    match reason {
        "length" => "MAX_TOKENS",
        "content_filter" => "SAFETY",
        _ => "STOP",
    }
}

fn usage(u: &Value) -> Value {
    let p = u["prompt_tokens"].as_u64().unwrap_or(0);
    let c = u["completion_tokens"].as_u64().unwrap_or(0);
    let mut m = json!({"promptTokenCount": p, "candidatesTokenCount": c, "totalTokenCount": p + c});
    if let Some(r) = u["completion_tokens_details"]["reasoning_tokens"].as_u64() {
        m["thoughtsTokenCount"] = json!(r);
    }
    if let Some(r) = u["prompt_tokens_details"]["cached_tokens"].as_u64() {
        m["cachedContentTokenCount"] = json!(r);
    }
    m
}

fn call_part(c: &Value) -> Value {
    let args: Value = serde_json::from_str(c["function"]["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({}));
    json!({"functionCall": {"id": c["id"], "name": c["function"]["name"], "args": args}})
}

/// An OpenAI chat completion as a Gemini GenerateContentResponse.
pub fn from_openai(r: &Value, model: &str) -> Value {
    let ch = &r["choices"][0];
    let msg = &ch["message"];
    let mut parts = Vec::new();
    if let Some(t) = msg["reasoning_content"].as_str().filter(|t| !t.is_empty()) {
        parts.push(json!({"text": t, "thought": true}));
    }
    if let Some(t) = msg["content"].as_str().filter(|t| !t.is_empty()) {
        parts.push(json!({"text": t}));
    }
    for c in msg["tool_calls"].as_array().cloned().unwrap_or_default() {
        parts.push(call_part(&c));
    }
    json!({"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish(ch["finish_reason"].as_str().unwrap_or("stop")),
                           "index": 0}],
           "usageMetadata": usage(&r["usage"]), "modelVersion": r["model"].as_str().unwrap_or(model)})
}

/// POST /v1beta/models/<model>:<method>
pub async fn call(State(s): State<AppState>, headers: HeaderMap, Path(target): Path<String>, Query(q): Query<HashMap<String, String>>,
                  body: Bytes) -> Response {
    let headers = with_key(&headers, &q);
    if !crate::api_authorized(&s, &headers) {
        return gerror(StatusCode::UNAUTHORIZED, "a valid API key is required (x-goog-api-key or ?key=)");
    }
    let Some((model, method)) = target.rsplit_once(':') else {
        return gerror(StatusCode::NOT_FOUND, "expected models/<model>:generateContent, :streamGenerateContent or :countTokens");
    };
    let g: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return gerror(StatusCode::BAD_REQUEST, format!("the body is not JSON: {e}")),
    };
    // a Gemini model name (gemini-2.5-pro, the CLI's default) runs on the loaded model; an installed one is used
    let model = model.trim_start_matches("models/");
    let installed = s.mgr.entry(model).is_some();
    let model_for_engine = if installed { model } else { "" };
    let req = match to_openai(model_for_engine, &g) {
        Ok(r) => r,
        Err(e) => return gerror(StatusCode::BAD_REQUEST, e),
    };
    match method {
        "countTokens" => count(s, headers, &req).await,
        "generateContent" => {
            let r = crate::openai::chat(State(s), headers, Json(req)).await;
            let status = r.status();
            let bytes = axum::body::to_bytes(r.into_body(), 64 << 20).await.unwrap_or_default();
            let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
            if !status.is_success() {
                return gerror(status, v["error"]["message"].as_str().unwrap_or("the request failed"));
            }
            Json(from_openai(&v, model)).into_response()
        }
        "streamGenerateContent" => stream(s, headers, req, model.to_string()).await,
        m => gerror(StatusCode::NOT_FOUND, format!("{m}: generateContent, streamGenerateContent or countTokens")),
    }
}

async fn stream(s: AppState, headers: HeaderMap, mut req: Value, model: String) -> Response {
    req["stream"] = json!(true);
    req["stream_options"] = json!({"include_usage": true});
    let r = crate::openai::chat(State(s), headers, Json(req)).await;
    if !r.status().is_success() {
        let status = r.status();
        let bytes = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap_or_default();
        let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        return gerror(status, v["error"]["message"].as_str().unwrap_or("the request failed"));
    }
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::convert::Infallible>>(256);
    tokio::spawn(async move {
        let mut body = r.into_body().into_data_stream();
        let mut buf = String::new();
        // calls by index: id, name, arguments so far
        let mut calls: Vec<(String, String, String)> = Vec::new();
        let mut last_finish: Option<String> = None;
        let send = |v: Value| Bytes::from(format!("data: {v}\n\n"));
        while let Some(Ok(b)) = body.next().await {
            buf.push_str(&String::from_utf8_lossy(&b));
            while let Some(i) = buf.find('\n') {
                let line = buf[..i].trim().to_string();
                buf.drain(..=i);
                let Some(data) = line.strip_prefix("data: ") else { continue };
                if data == "[DONE]" {
                    continue;
                }
                let Ok(c) = serde_json::from_str::<Value>(data) else { continue };
                let ch = &c["choices"][0];
                let d = &ch["delta"];
                let mut parts = Vec::new();
                if let Some(t) = d["reasoning_content"].as_str().filter(|t| !t.is_empty()) {
                    parts.push(json!({"text": t, "thought": true}));
                }
                if let Some(t) = d["content"].as_str().filter(|t| !t.is_empty()) {
                    parts.push(json!({"text": t}));
                }
                for tc in d["tool_calls"].as_array().cloned().unwrap_or_default() {
                    let i = tc["index"].as_u64().unwrap_or(0) as usize;
                    while calls.len() <= i {
                        calls.push(Default::default());
                    }
                    let e = &mut calls[i];
                    if let Some(id) = tc["id"].as_str() { e.0 = id.to_string(); }
                    if let Some(n) = tc["function"]["name"].as_str() { e.1.push_str(n); }
                    if let Some(a) = tc["function"]["arguments"].as_str() { e.2.push_str(a); }
                }
                let fin = ch["finish_reason"].as_str().map(str::to_string);
                if fin.is_some() {
                    // the calls are whole now
                    for (id, name, args) in calls.drain(..) {
                        parts.push(call_part(&json!({"id": id, "function": {"name": name, "arguments": args}})));
                    }
                    last_finish = fin.clone();
                }
                let mut out = json!({"modelVersion": c["model"].as_str().unwrap_or(&model)});
                if !parts.is_empty() || fin.is_some() {
                    let mut cand = json!({"content": {"role": "model", "parts": parts}, "index": 0});
                    if let Some(f) = &fin {
                        cand["finishReason"] = json!(finish(f));
                    }
                    out["candidates"] = json!([cand]);
                }
                if c["usage"].is_object() {
                    out["usageMetadata"] = usage(&c["usage"]);
                    if out.get("candidates").is_none() {
                        out["candidates"] = json!([{"content": {"role": "model", "parts": []}, "index": 0,
                                                    "finishReason": finish(last_finish.as_deref().unwrap_or("stop"))}]);
                    }
                }
                if out.get("candidates").is_some() && tx.send(Ok(send(out))).await.is_err() {
                    return;
                }
            }
        }
    });
    Response::builder().header("content-type", "text/event-stream").header("cache-control", "no-cache")
        .body(axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx))).unwrap()
}

/// countTokens: the request's text, tool calls and answers in the loaded model's tokens, through the Anthropic count
/// (the same tokenizer and chat template).
async fn count(s: AppState, headers: HeaderMap, req: &Value) -> Response {
    let mut msgs = Vec::new();
    let mut system = String::new();
    for m in req["messages"].as_array().cloned().unwrap_or_default() {
        let text = match &m["content"] {
            Value::String(t) => t.clone(),
            Value::Array(a) => a.iter().filter_map(|c| c["text"].as_str()).collect::<Vec<_>>().join(""),
            _ => String::new(),
        } + &m["tool_calls"].as_array().map(|c| Value::Array(c.clone()).to_string()).unwrap_or_default();
        match m["role"].as_str() {
            Some("system") => system.push_str(&text),
            Some("assistant") => msgs.push(json!({"role": "assistant", "content": text})),
            _ => msgs.push(json!({"role": "user", "content": text})),
        }
    }
    if msgs.is_empty() {
        msgs.push(json!({"role": "user", "content": ""}));
    }
    let mut a = json!({"messages": msgs});
    if !system.is_empty() {
        a["system"] = json!(system);
    }
    let r = crate::anthropic::count_tokens(State(s), headers, Json(a)).await;
    let status = r.status();
    let bytes = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap_or_default();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    if !status.is_success() {
        return gerror(status, v["error"]["message"].as_str().unwrap_or("the count failed"));
    }
    Json(json!({"totalTokens": v["input_tokens"]})).into_response()
}

fn model_entry(id: &str, name: &str) -> Value {
    json!({"name": format!("models/{id}"), "baseModelId": id, "version": "am2strata", "displayName": name,
           "supportedGenerationMethods": ["generateContent", "streamGenerateContent", "countTokens"]})
}

/// GET /v1beta/models: the installed models, in Gemini's list.
pub async fn models(State(s): State<AppState>, headers: HeaderMap, Query(q): Query<HashMap<String, String>>) -> Response {
    let headers = with_key(&headers, &q);
    if !crate::api_authorized(&s, &headers) {
        return gerror(StatusCode::UNAUTHORIZED, "a valid API key is required (x-goog-api-key or ?key=)");
    }
    let list = s.mgr.list().await;
    let models: Vec<Value> = list["models"].as_array().cloned().unwrap_or_default().iter()
        .map(|m| model_entry(m["id"].as_str().unwrap_or(""), m["display_name"].as_str().unwrap_or(""))).collect();
    Json(json!({"models": models})).into_response()
}

/// GET /v1beta/models/<model>
pub async fn model(State(s): State<AppState>, headers: HeaderMap, Path(id): Path<String>, Query(q): Query<HashMap<String, String>>) -> Response {
    let headers = with_key(&headers, &q);
    if !crate::api_authorized(&s, &headers) {
        return gerror(StatusCode::UNAUTHORIZED, "a valid API key is required (x-goog-api-key or ?key=)");
    }
    match s.mgr.entry(&id) {
        Some(e) => Json(model_entry(&e.id, &e.display_name())).into_response(),
        None => gerror(StatusCode::NOT_FOUND, format!("no installed model {id:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn google_search_becomes_the_server_s_web_search() {
        let r = to_openai("m", &json!({"contents": [{"role": "user", "parts": [{"text": "news?"}]}], "tools": [{"google_search": {}}]})).unwrap();
        assert_eq!(r["web_search_options"], json!({}));
        assert!(r.get("tools").is_none(), "no function tools");
    }

    #[test]
    fn a_gemini_turn_with_a_function_call_and_its_answer_becomes_openai_messages() {
        let g = json!({
            "systemInstruction": {"parts": [{"text": "Be brief."}]},
            "contents": [
                {"role": "user", "parts": [{"text": "List the files."}]},
                {"role": "model", "parts": [{"functionCall": {"name": "ls", "args": {"path": "."}}}]},
                {"role": "user", "parts": [{"functionResponse": {"name": "ls", "response": {"output": "a.rs b.rs"}}}]}],
            "tools": [{"functionDeclarations": [{"name": "ls", "description": "list", "parameters": {"type": "OBJECT",
                "properties": {"path": {"type": "STRING", "nullable": true}}, "required": ["path"]}}]}],
            "generationConfig": {"maxOutputTokens": 100, "temperature": 0.2, "thinkingConfig": {"thinkingBudget": 0}}});
        let r = to_openai("", &g).unwrap();
        let m = r["messages"].as_array().unwrap();
        assert_eq!((m[0]["role"].as_str(), m[1]["content"].as_str()), (Some("system"), Some("List the files.")));
        assert_eq!(m[2]["tool_calls"][0]["function"]["arguments"], "{\"path\":\".\"}");
        assert_eq!((m[3]["role"].as_str(), m[3]["tool_call_id"].clone(), m[3]["content"].as_str()),
                   (Some("tool"), m[2]["tool_calls"][0]["id"].clone(), Some("a.rs b.rs")));
        assert_eq!(r["tools"][0]["function"]["parameters"]["properties"]["path"], json!({"type": "string"}));
        assert_eq!((r["max_tokens"].as_u64(), r["chat_template_kwargs"]["enable_thinking"].as_bool()), (Some(100), Some(false)));
        assert!(r.get("model").is_none(), "a Gemini model name runs on the loaded model");
    }

    #[test]
    fn an_openai_answer_becomes_a_gemini_candidate() {
        let r = json!({"model": "tiel", "choices": [{"finish_reason": "tool_calls", "message": {"content": "", "reasoning_content": "hm",
            "tool_calls": [{"id": "c1", "function": {"name": "ls", "arguments": "{\"path\":\".\"}"}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}});
        let g = from_openai(&r, "tiel");
        let parts = g["candidates"][0]["content"]["parts"].as_array().unwrap();
        assert_eq!(parts[0], json!({"text": "hm", "thought": true}));
        assert_eq!(parts[1]["functionCall"]["args"]["path"], ".");
        assert_eq!((g["candidates"][0]["finishReason"].as_str(), g["usageMetadata"]["totalTokenCount"].as_u64()), (Some("STOP"), Some(15)));
    }
}
