//! OpenAI Responses: POST /v1/responses (streaming and not), stateless. Port of serve/responses.py.
//!
//! A Responses request becomes a Chat Completions request (instructions + input items -> messages, function /
//! namespace / custom tools -> function tools), runs through the same front end and run loop, and its events come
//! back as Responses output items and typed SSE events. Nothing is stored: the client sends the whole conversation
//! every time (`store: false`, as Codex CLI does), with the reasoning, message and function_call items of earlier
//! answers.

use crate::run::{self, RunEvent, Summary};
use crate::{authorized, media, now, AppState};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine as _;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::time::Duration;
use strata_frontend::openai_to_messages;
use strata_frontend::output::{py_dumps, Event, ToolCall};

/// Our own reasoning replay strings; other servers' encrypted_content is ignored.
const ENCRYPTED_PREFIX: &str = "strata.r1:";

#[derive(Debug)]
pub struct RespError {
    pub message: String,
    pub param: Option<String>,
    pub code: Option<&'static str>,
    pub status: u16,
    pub kind: &'static str,
}

impl RespError {
    fn new(message: impl Into<String>, param: impl Into<Option<String>>) -> Self {
        RespError { message: message.into(), param: param.into(), code: None, status: 400, kind: "invalid_request_error" }
    }
    fn code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }
    fn body(&self) -> Value {
        error_body(&self.message, self.kind, self.param.as_deref(), self.code)
    }
    fn into_response(self) -> Response {
        (StatusCode::from_u16(self.status).unwrap_or(StatusCode::BAD_REQUEST), Json(self.body())).into_response()
    }
}

fn error_body(message: &str, kind: &str, param: Option<&str>, code: Option<&str>) -> Value {
    json!({"error": {"message": message, "type": kind, "param": param, "code": code}})
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

/// Template name -> (namespace, name, kind) for the tools of one request.
pub type ToolNames = HashMap<String, (Option<String>, String, String)>;

// ------------------------------------------------------------------------------------------------ request -> chat

/// A message's (or tool output's) content -> a chat content: a string, or text and image_url parts.
fn content(c: Option<&Value>, param: &str) -> Result<Value, RespError> {
    let parts = match c {
        None | Some(Value::Null) => return Ok(json!("")),
        Some(Value::String(s)) => return Ok(json!(s)),
        Some(Value::Array(a)) => a,
        Some(_) => return Err(RespError::new("expected a string or an array of content parts", param.to_string())),
    };
    let mut out = Vec::new();
    for (j, part) in parts.iter().enumerate() {
        if let Some(s) = part.as_str() {
            out.push(json!({"type": "text", "text": s}));
            continue;
        }
        let Some(p) = part.as_object() else {
            return Err(RespError::new("expected a content part object", format!("{param}[{j}]")));
        };
        let text = |k: &str| p.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        match p.get("type").and_then(Value::as_str) {
            Some("input_text" | "output_text" | "text" | "summary_text" | "reasoning_text") => {
                out.push(json!({"type": "text", "text": text("text")}))
            }
            Some("refusal") => out.push(json!({"type": "text", "text": text("refusal")})),
            Some("input_image" | "image_url") => {
                let url = match p.get("image_url") {
                    Some(Value::String(u)) if !u.is_empty() => u.clone(),
                    Some(Value::Object(o)) => o.get("url").and_then(Value::as_str).unwrap_or("").to_string(),
                    _ => String::new(),
                };
                if url.is_empty() {
                    return Err(RespError::new(
                        "only images given as image_url (a data: or http(s) URL) are supported; this server keeps no files",
                        format!("{param}[{j}]")).code("unsupported_parameter"));
                }
                out.push(json!({"type": "image_url", "image_url": {"url": url}}));
            }
            other => {
                return Err(RespError::new(
                    format!("content parts of type {} are not supported (text and images are)", py_repr(other)),
                    format!("{param}[{j}].type")).code("unsupported_parameter"))
            }
        }
    }
    Ok(Value::Array(out))
}

fn py_repr(s: Option<&str>) -> String {
    s.map(|s| format!("'{s}'")).unwrap_or_else(|| "None".into())
}

fn text_only(c: &Value) -> String {
    match c {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter(|p| p["type"] == "text").filter_map(|p| p["text"].as_str()).collect(),
        _ => String::new(),
    }
}

/// The thinking a reasoning item carries: its reasoning_text content, else our own encrypted_content.
fn reasoning_text(item: &Value, param: &str) -> Result<String, RespError> {
    let texts: Vec<&str> = item
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|p| matches!(p.get("type").and_then(Value::as_str), Some("reasoning_text" | "text")))
        .map(|p| p.get("text").and_then(Value::as_str).unwrap_or(""))
        .collect();
    if !texts.is_empty() {
        return Ok(texts.concat());
    }
    if let Some(enc) = item.get("encrypted_content").and_then(Value::as_str).and_then(|e| e.strip_prefix(ENCRYPTED_PREFIX)) {
        let damaged = || RespError::new("this reasoning item's encrypted_content is damaged", format!("{param}.encrypted_content"))
            .code("invalid_encrypted_content");
        let raw = base64::engine::general_purpose::STANDARD.decode(enc).map_err(|_| damaged())?;
        let v: Value = serde_json::from_slice(&raw).map_err(|_| damaged())?;
        return v.get("text").and_then(Value::as_str).map(str::to_owned).ok_or_else(damaged);
    }
    Ok(String::new())
}

/// A function_call's arguments -> the object the template renders (text that is not a JSON object is kept).
fn arguments(raw: Option<&Value>, param: &str) -> Result<Value, RespError> {
    Ok(match raw {
        None | Some(Value::Null) => json!({}),
        Some(Value::Object(_)) => raw.cloned().unwrap(),
        Some(Value::String(s)) if s.trim().is_empty() => json!({}),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(v @ Value::Object(_)) => v,
            Ok(v) => json!({"arguments": v}),
            Err(_) => json!({"arguments": s}),
        },
        Some(_) => return Err(RespError::new("function_call arguments must be a JSON string", param.to_string())),
    })
}

fn tool_output(item: &Value, param: &str) -> Result<Value, RespError> {
    match item.get("output") {
        Some(o @ Value::Object(_)) => Ok(json!(py_dumps(o, false))),
        Some(Value::String(s)) => Ok(json!(s)),
        other => content(other, &format!("{param}.output")),
    }
}

/// `instructions` + `input` -> chat messages (assistant tool calls carry their call ids, tool results their
/// tool_call_id, so the front end puts the results in the order of the calls).
pub fn input_messages(req: &Value) -> Result<Vec<Value>, RespError> {
    let mut messages: Vec<Value> = Vec::new();
    match req.get("instructions") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) if s.is_empty() => {}
        Some(Value::String(s)) => messages.push(json!({"role": "system", "content": s})),
        Some(other) => messages.push(json!({"role": "system", "content": text_only(&content(Some(other), "instructions")?)})),
    }
    let items = match req.get("input") {
        None | Some(Value::Null) => {
            return Err(RespError::new("input is required: a string or an array of input items", "input".to_string())
                .code("missing_required_parameter"))
        }
        Some(Value::String(s)) => vec![json!({"type": "message", "role": "user", "content": s})],
        Some(Value::Array(a)) => a.clone(),
        Some(_) => return Err(RespError::new("input must be a string or an array of input items", "input".to_string())),
    };
    let mut thinking: Vec<String> = Vec::new();
    let mut open_turn: Option<usize> = None; // the assistant message the next function_call joins

    fn turn(messages: &mut Vec<Value>, thinking: &mut Vec<String>, content: Value) -> usize {
        let mut m = json!({"role": "assistant", "content": content});
        if !thinking.is_empty() {
            m["reasoning_content"] = json!(thinking.concat());
            thinking.clear();
        }
        messages.push(m);
        messages.len() - 1
    }

    for (i, item) in items.iter().enumerate() {
        let param = format!("input[{i}]");
        if !item.is_object() {
            return Err(RespError::new("expected an input item object", param));
        }
        let kind = item.get("type").and_then(Value::as_str).or(if item.get("role").is_some() { Some("message") } else { None });
        match kind {
            Some("message") => {
                let role = item.get("role").and_then(Value::as_str);
                let Some(role @ ("user" | "assistant" | "system" | "developer")) = role else {
                    return Err(RespError::new(format!("unknown message role {}", py_repr(role)), format!("{param}.role")));
                };
                let c = content(item.get("content"), &format!("{param}.content"))?;
                if role == "assistant" {
                    open_turn = Some(turn(&mut messages, &mut thinking, json!(text_only(&c))));
                } else {
                    open_turn = None;
                    thinking.clear();
                    let role = if role == "developer" { "system" } else { role };
                    // system text is a string for the template; user content keeps its pictures
                    let c = if role == "system" { json!(text_only(&c)) } else { c };
                    messages.push(json!({"role": role, "content": c}));
                }
            }
            Some("reasoning") => {
                let text = reasoning_text(item, &param)?;
                if !text.is_empty() {
                    if let Some(t) = open_turn {
                        let m = &messages[t];
                        if m.get("tool_calls").is_some() || !text_only(&m["content"]).is_empty() {
                            open_turn = None; // thinking starts a new model turn
                        }
                    }
                    thinking.push(text);
                }
            }
            Some(k @ ("function_call" | "custom_tool_call")) => {
                let name = item.get("name").and_then(Value::as_str).filter(|n| !n.is_empty());
                let Some(name) = name else {
                    return Err(RespError::new("a function_call needs a name", format!("{param}.name")));
                };
                let name = match item.get("namespace").and_then(Value::as_str).filter(|n| !n.is_empty()) {
                    Some(ns) => format!("{ns}.{name}"),
                    None => name.to_string(),
                };
                let args = if k == "function_call" {
                    arguments(item.get("arguments"), &format!("{param}.arguments"))?
                } else {
                    json!({"input": item.get("input").and_then(Value::as_str).unwrap_or("")})
                };
                let target = match open_turn {
                    Some(t) if thinking.is_empty() => t,
                    _ => turn(&mut messages, &mut thinking, json!("")),
                };
                open_turn = Some(target);
                let mut call = json!({"type": "function", "function": {"name": name, "arguments": args}});
                if let Some(id) = item.get("call_id").filter(|v| v.is_string()) {
                    call["id"] = id.clone();
                }
                let m = messages[target].as_object_mut().unwrap();
                m.entry("tool_calls").or_insert_with(|| json!([])).as_array_mut().unwrap().push(call);
            }
            Some("function_call_output" | "custom_tool_call_output") => {
                let mut m = json!({"role": "tool", "content": tool_output(item, &param)?});
                if let Some(id) = item.get("call_id").filter(|v| v.is_string()) {
                    m["tool_call_id"] = id.clone();
                }
                messages.push(m);
                open_turn = None;
                thinking.clear();
            }
            Some("additional_tools") => {} // tools the client adds as an input item: see request_tools
            Some("item_reference") => {
                return Err(RespError::new(
                    "item references need stored responses, and this server keeps none: send the items themselves", param)
                    .code("unsupported_parameter"))
            }
            other => {
                return Err(RespError::new(format!("input items of type {} are not supported", py_repr(other)),
                                          format!("{param}.type")).code("unsupported_parameter"))
            }
        }
    }
    // leading system/developer messages (Codex: instructions, then its developer message) become one system message
    let lead = messages.iter().take_while(|m| m["role"] == "system" && m["content"].is_string()).count();
    if lead > 1 {
        let joined: Vec<&str> = messages[..lead].iter().map(|m| m["content"].as_str().unwrap()).collect();
        let merged = json!({"role": "system", "content": joined.join("\n\n")});
        messages.splice(..lead, [merged]);
    }
    Ok(messages)
}

/// -> (chat function tools, template name -> (namespace, name, kind), names of hosted tools left out).
pub fn request_tools(req: &Value) -> Result<(Vec<Value>, ToolNames, Vec<String>), RespError> {
    let (mut tools, mut names, mut skipped) = (Vec::new(), ToolNames::new(), Vec::new());
    let mut given: Vec<Value> = match req.get("tools") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a.clone(),
        Some(_) => return Err(RespError::new("tools must be an array", "tools".to_string())),
    };
    // Codex: `additional_tools` input items carry more tool definitions
    for item in req.get("input").and_then(Value::as_array).into_iter().flatten() {
        if item.get("type").and_then(Value::as_str) == Some("additional_tools") {
            given.extend(item.get("tools").and_then(Value::as_array).cloned().unwrap_or_default());
        }
    }
    let mut add = |tool: &Value, param: String, namespace: Option<&str>, ns_description: &str| -> Result<(), RespError> {
        let kind = tool.get("type").and_then(Value::as_str).unwrap_or("function");
        let Some(name) = tool.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()) else {
            return Err(RespError::new("a tool needs a name", format!("{param}.name")));
        };
        let flat = namespace.map(|ns| format!("{ns}.{name}")).unwrap_or_else(|| name.to_string());
        let mut description = tool.get("description").and_then(Value::as_str).unwrap_or("").to_string();
        if !ns_description.is_empty() {
            description = if description.is_empty() { ns_description.to_string() } else { format!("{ns_description}\n\n{description}") };
        }
        let params = if kind == "function" {
            tool.get("parameters").filter(|p| !p.is_null()).cloned().unwrap_or(json!({"type": "object", "properties": {}}))
        } else {
            // custom: free-form text, maybe in a grammar the model reads
            if let Some(fmt) = tool.get("format").and_then(Value::as_object) {
                if fmt.get("type").and_then(Value::as_str) == Some("grammar") {
                    if let Some(def) = fmt.get("definition").and_then(Value::as_str).filter(|d| !d.is_empty()) {
                        let syntax = fmt.get("syntax").and_then(Value::as_str).unwrap_or("");
                        description.push_str(&format!("\n\nThe input must follow this {syntax} grammar:\n{def}"));
                    }
                }
            }
            json!({"type": "object", "properties": {"input": {"type": "string", "description": "the raw input"}}, "required": ["input"]})
        };
        tools.push(json!({"type": "function", "function": {"name": flat, "description": description, "parameters": params}}));
        let entry = (namespace.map(str::to_owned), name.to_string(), kind.to_string());
        names.insert(flat, entry.clone());
        if let Some(ns) = namespace {
            // Qwen writes Codex's MCP tools as `mcp__server__tool`: that spelling maps back as well
            names.entry(format!("{ns}__{name}")).or_insert(entry);
        }
        Ok(())
    };
    for (i, tool) in given.iter().enumerate() {
        let param = format!("tools[{i}]");
        if !tool.is_object() {
            return Err(RespError::new("expected a tool object", param));
        }
        match tool.get("type").and_then(Value::as_str) {
            Some("function" | "custom") => add(tool, param, None, "")?,
            Some("namespace") => {
                let ns = tool.get("name").and_then(Value::as_str);
                let nsd = tool.get("description").and_then(Value::as_str).unwrap_or("");
                for (j, member) in tool.get("tools").and_then(Value::as_array).into_iter().flatten().enumerate() {
                    let ok = member.is_object()
                        && matches!(member.get("type").and_then(Value::as_str).unwrap_or("function"), "function" | "custom");
                    if !ok {
                        return Err(RespError::new("a namespace holds function tools", format!("{param}.tools[{j}]")));
                    }
                    let mut m = member.clone();
                    m["type"] = json!("function");
                    add(&m, format!("{param}.tools[{j}]"), ns, nsd)?;
                }
            }
            Some(other) => skipped.push(other.to_string()), // hosted tools (web search, ...): the model cannot run them
            None => return Err(RespError::new("a tool needs a type", format!("{param}.type"))),
        }
    }
    Ok((tools, names, skipped))
}

/// What this stateless server cannot do, refused before anything runs.
pub fn check_request(req: &Value) -> Result<(), RespError> {
    let set = |k: &str| req.get(k).is_some_and(strata_frontend_truthy);
    let refuse = |msg: &str, p: &str| Err(RespError::new(msg, p.to_string()).code("unsupported_parameter"));
    if set("previous_response_id") {
        return refuse("this server keeps no responses (stateless): send the whole conversation in input instead of \
                       previous_response_id", "previous_response_id");
    }
    if set("conversation") {
        return refuse("this server keeps no conversations (stateless): send the whole conversation in input", "conversation");
    }
    if set("background") {
        return refuse("background responses are not supported: use stream or wait for the answer", "background");
    }
    if set("prompt") {
        return refuse("stored prompt templates are not supported: send instructions and input", "prompt");
    }
    if let Some(n) = req.get("max_output_tokens").filter(|v| !v.is_null()) {
        if n.as_i64().is_none_or(|n| n < 1) {
            return Err(RespError::new("max_output_tokens must be a positive whole number", "max_output_tokens".to_string()));
        }
    }
    if req.get("reasoning").is_some_and(|r| !r.is_null() && !r.is_object()) {
        return Err(RespError::new("reasoning must be an object", "reasoning".to_string()));
    }
    Ok(())
}

fn strata_frontend_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// The whole Responses request -> (an equivalent Chat Completions request, the tool names).
pub fn to_chat(req: &Value) -> Result<(Value, ToolNames), RespError> {
    check_request(req)?;
    let messages = input_messages(req)?;
    let (tools, names, skipped) = request_tools(req)?;
    if !skipped.is_empty() {
        tracing::info!("/v1/responses: left out the hosted tools {} (the model cannot run them)", skipped.join(", "));
    }
    let mut chat = json!({"messages": messages});
    if !tools.is_empty() && req.get("tool_choice").and_then(Value::as_str) != Some("none") {
        chat["tools"] = json!(tools);
    }
    if let Some(n) = req.get("max_output_tokens").filter(|v| !v.is_null()) {
        chat["max_tokens"] = n.clone();
    }
    if let Some(e) = req.get("reasoning").and_then(|r| r.get("effort")).filter(|v| !v.is_null()) {
        chat["reasoning_effort"] = e.clone();
    }
    for k in ["temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty",
              "seed", "stop", "tool_choice", "chat_template_kwargs", "reasoning_budget_tokens"] {
        if let Some(v) = req.get(k).filter(|v| !v.is_null()) {
            chat[k] = v.clone();
        }
    }
    Ok((chat, names))
}

// ------------------------------------------------------------------------------------------------ output

/// One response: its output items and typed events, built from the run's events. The non-streaming answer is the
/// `response` of the last event, so both modes give the same items.
pub struct Assembler {
    names: ToolNames,
    encrypted: bool,
    prompt_tokens: u64,
    seq: u64,
    /// the index (in response.output) of the open item
    open: Option<usize>,
    pub response: Value,
}

impl Assembler {
    pub fn new(req: &Value, model: &str, prompt_tokens: u64, names: ToolNames, encrypted: bool) -> Self {
        let reasoning = req.get("reasoning").and_then(Value::as_object).cloned().unwrap_or_default();
        let g = |k: &str| req.get(k).cloned().unwrap_or(Value::Null);
        let response = json!({
            "id": new_id("resp"), "object": "response", "created_at": now(), "status": "in_progress",
            "background": false, "error": null, "incomplete_details": null,
            "instructions": g("instructions"), "max_output_tokens": g("max_output_tokens"),
            "model": model, "output": [], "parallel_tool_calls": req.get("parallel_tool_calls").cloned().unwrap_or(json!(true)),
            "previous_response_id": null,
            "reasoning": {"effort": reasoning.get("effort").cloned().unwrap_or(Value::Null),
                          "summary": reasoning.get("summary").cloned().unwrap_or(Value::Null)},
            "store": false, "temperature": g("temperature"),
            "text": req.get("text").filter(|t| t.is_object()).cloned().unwrap_or(json!({"format": {"type": "text"}})),
            "tool_choice": req.get("tool_choice").cloned().unwrap_or(json!("auto")),
            "tools": req.get("tools").filter(|t| !t.is_null()).cloned().unwrap_or(json!([])),
            "top_p": g("top_p"), "truncation": "disabled", "usage": null, "user": null,
            "metadata": req.get("metadata").filter(|t| !t.is_null()).cloned().unwrap_or(json!({})),
        });
        Assembler { names, encrypted, prompt_tokens, seq: 0, open: None, response }
    }

    fn event(&mut self, kind: &str, data: Value) -> Value {
        let mut e = json!({"type": kind, "sequence_number": self.seq});
        if let (Some(e), Value::Object(d)) = (e.as_object_mut(), data) {
            e.extend(d);
        }
        self.seq += 1;
        e
    }

    fn output(&mut self) -> &mut Vec<Value> {
        self.response["output"].as_array_mut().unwrap()
    }

    fn item(&mut self) -> Option<&mut Value> {
        let i = self.open?;
        self.output().get_mut(i)
    }

    fn item_type(&self) -> Option<&str> {
        self.open.and_then(|i| self.response["output"][i]["type"].as_str())
    }

    pub fn start(&mut self) -> Vec<Value> {
        let snap = self.response.clone();
        vec![self.event("response.created", json!({"response": snap.clone()})), self.event("response.in_progress", json!({"response": snap}))]
    }

    /// A keep-alive a client counts as activity (Codex's idle timer ignores SSE comments).
    pub fn in_progress(&mut self) -> Value {
        let snap = self.response.clone();
        self.event("response.in_progress", json!({"response": snap}))
    }

    fn open_item(&mut self, mut item: Value) -> Vec<Value> {
        let is_msg = item["type"] == "message";
        if is_msg {
            item["content"] = json!([]);
        }
        self.output().push(item.clone());
        let at = self.response["output"].as_array().unwrap().len() - 1;
        self.open = Some(at);
        let mut out = vec![self.event("response.output_item.added", json!({"output_index": at, "item": item}))];
        if is_msg {
            let part = json!({"type": "output_text", "text": "", "annotations": [], "logprobs": []});
            self.item().unwrap()["content"].as_array_mut().unwrap().push(part.clone());
            let id = self.response["output"][at]["id"].clone();
            out.push(self.event("response.content_part.added", json!({"item_id": id, "output_index": at, "content_index": 0, "part": part})));
        }
        out
    }

    pub fn close(&mut self, status: &str) -> Vec<Value> {
        let Some(at) = self.open.take() else { return Vec::new() };
        let encrypted = self.encrypted;
        let item = &mut self.response["output"][at];
        item["status"] = json!(status);
        let id = item["id"].clone();
        let mut pending: Vec<(&str, Value)> = Vec::new();
        match item["type"].as_str() {
            Some("message") => {
                let part = item["content"][0].clone();
                pending.push(("response.output_text.done", json!({"item_id": id, "output_index": at, "content_index": 0,
                                                                   "text": part["text"], "logprobs": []})));
                pending.push(("response.content_part.done", json!({"item_id": id, "output_index": at, "content_index": 0, "part": part})));
            }
            Some("reasoning") => {
                let text = item["content"][0]["text"].as_str().unwrap_or("").to_string();
                pending.push(("response.reasoning_text.done", json!({"item_id": id, "output_index": at, "content_index": 0, "text": text})));
                if encrypted {
                    let raw = py_dumps(&json!({"text": text}), false);
                    item["encrypted_content"] = json!(format!("{ENCRYPTED_PREFIX}{}", base64::engine::general_purpose::STANDARD.encode(raw)));
                }
            }
            Some("function_call") => pending.push(("response.function_call_arguments.done",
                json!({"item_id": id, "output_index": at, "name": item["name"], "arguments": item["arguments"]}))),
            Some("custom_tool_call") => pending.push(("response.custom_tool_call_input.done",
                json!({"item_id": id, "output_index": at, "input": item["input"]}))),
            _ => {}
        }
        let snap = item.clone();
        pending.push(("response.output_item.done", json!({"output_index": at, "item": snap})));
        pending.into_iter().map(|(k, d)| self.event(k, d)).collect()
    }

    fn call_item(&self, id: &str, call_name: &str) -> Value {
        let (namespace, name, kind) =
            self.names.get(call_name).cloned().unwrap_or((None, call_name.to_string(), "function".into()));
        let mut item = if kind == "custom" {
            json!({"id": new_id("ctc"), "type": "custom_tool_call", "status": "in_progress", "call_id": id, "name": name, "input": ""})
        } else {
            json!({"id": new_id("fc"), "type": "function_call", "status": "in_progress", "call_id": id, "name": name, "arguments": ""})
        };
        if let Some(ns) = namespace {
            item["namespace"] = json!(ns);
        }
        item
    }

    fn open_call_id(&self) -> Option<String> {
        self.open.and_then(|i| self.response["output"][i]["call_id"].as_str().map(str::to_owned))
    }

    pub fn feed(&mut self, ev: Event) -> Vec<Value> {
        let mut out = Vec::new();
        match ev {
            Event::Reasoning(t) if !t.is_empty() => {
                if self.item_type() != Some("reasoning") {
                    out.extend(self.close("completed"));
                    out.extend(self.open_item(json!({"id": new_id("rs"), "type": "reasoning", "status": "in_progress",
                                                     "summary": [], "content": [{"type": "reasoning_text", "text": ""}]})));
                }
                let at = self.open.unwrap();
                let item = self.item().unwrap();
                let id = item["id"].clone();
                append(&mut item["content"][0]["text"], &t);
                out.push(self.event("response.reasoning_text.delta", json!({"item_id": id, "output_index": at, "content_index": 0, "delta": t})));
            }
            Event::Content(t) if !t.is_empty() => {
                if self.item_type() != Some("message") {
                    out.extend(self.close("completed"));
                    out.extend(self.open_item(json!({"id": new_id("msg"), "type": "message", "status": "in_progress", "role": "assistant"})));
                }
                let at = self.open.unwrap();
                let item = self.item().unwrap();
                let id = item["id"].clone();
                append(&mut item["content"][0]["text"], &t);
                out.push(self.event("response.output_text.delta",
                                    json!({"item_id": id, "output_index": at, "content_index": 0, "delta": t, "logprobs": []})));
            }
            Event::ToolStart { id, name } => {
                out.extend(self.close("completed"));
                let item = self.call_item(&id, &name);
                out.extend(self.open_item(item));
            }
            Event::ToolArgs { id, text } => {
                if self.open_call_id().as_deref() != Some(id.as_str()) {
                    return out;
                }
                let at = self.open.unwrap();
                let item = self.item().unwrap();
                if item["type"] == "function_call" {
                    // a custom tool's input is sent whole, at its end
                    append(&mut item["arguments"], &text);
                    let iid = item["id"].clone();
                    out.push(self.event("response.function_call_arguments.delta", json!({"item_id": iid, "output_index": at, "delta": text})));
                }
            }
            Event::ToolCall(c) => out.extend(self.whole_call(c)),
            _ => {}
        }
        out
    }

    fn whole_call(&mut self, c: ToolCall) -> Vec<Value> {
        let mut out = Vec::new();
        if self.open_call_id().as_deref() != Some(c.id.as_str()) {
            // a call that was not announced while it was written
            out.extend(self.close("completed"));
            let item = self.call_item(&c.id, &c.name);
            out.extend(self.open_item(item));
            if self.item_type() == Some("function_call") {
                let at = self.open.unwrap();
                let args = py_dumps(&Value::Object(c.arguments.clone()), false);
                let item = self.item().unwrap();
                item["arguments"] = json!(args);
                let iid = item["id"].clone();
                out.push(self.event("response.function_call_arguments.delta", json!({"item_id": iid, "output_index": at, "delta": args})));
            }
        }
        if self.item_type() == Some("custom_tool_call") {
            let at = self.open.unwrap();
            let input = match c.arguments.get("input") {
                Some(Value::String(s)) => s.clone(),
                _ => py_dumps(&Value::Object(c.arguments.clone()), false),
            };
            let item = self.item().unwrap();
            item["input"] = json!(input);
            let iid = item["id"].clone();
            out.push(self.event("response.custom_tool_call_input.delta", json!({"item_id": iid, "output_index": at, "delta": input})));
        }
        out.extend(self.close("completed"));
        out
    }

    /// The end: close what is open, then response.completed (or .incomplete when max_output_tokens ran out).
    pub fn finish(&mut self, sum: &Summary) -> Vec<Value> {
        let length = sum.finish == "length";
        // a call still open was cut off: incomplete, whatever the finish
        let unfinished = matches!(self.item_type(), Some("function_call" | "custom_tool_call"));
        let mut out = self.close(if length || unfinished { "incomplete" } else { "completed" });
        let n = sum.completion_tokens;
        let prompt = self.prompt_tokens;
        self.response["usage"] = json!({
            "input_tokens": prompt, "input_tokens_details": {"cached_tokens": sum.reused.min(prompt)},
            "output_tokens": n, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": prompt + n});
        if length {
            self.response["status"] = json!("incomplete");
            self.response["incomplete_details"] = json!({"reason": "max_output_tokens"});
            let snap = self.response.clone();
            out.push(self.event("response.incomplete", json!({"response": snap})));
            return out;
        }
        self.response["status"] = json!("completed");
        self.response["completed_at"] = json!(now());
        let snap = self.response.clone();
        out.push(self.event("response.completed", json!({"response": snap})));
        out
    }

    /// response.failed: an error after the stream started (the HTTP status is already sent).
    pub fn failed(&mut self, message: &str) -> Value {
        if let Some(item) = self.item() {
            item["status"] = json!("incomplete");
        }
        self.open = None;
        self.response["status"] = json!("failed");
        self.response["error"] = json!({"code": "server_error", "message": message});
        let snap = self.response.clone();
        self.event("response.failed", json!({"response": snap}))
    }
}

fn append(v: &mut Value, t: &str) {
    let mut s = v.as_str().unwrap_or("").to_string();
    s.push_str(t);
    *v = Value::String(s);
}

// ------------------------------------------------------------------------------------------------ handlers

/// GET/DELETE /v1/responses/{id} and friends: nothing is stored.
pub async fn not_stored(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return unauthorized();
    }
    (StatusCode::NOT_FOUND, Json(error_body(
        "this server keeps no responses (stateless): send the whole conversation to POST /v1/responses",
        "invalid_request_error", None, Some("not_found")))).into_response()
}

fn unauthorized() -> Response {
    (StatusCode::UNAUTHORIZED, Json(error_body("a valid API key is required", "authentication_error", None, None))).into_response()
}

pub async fn responses(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return unauthorized();
    }
    let (mut chat, names) = match to_chat(&req) {
        Ok(c) => c,
        Err(e) => return e.into_response(),
    };
    if let Err(e) = media::expand_videos(&mut chat).await {
        return RespError::new(e.to_string(), None).into_response();
    }
    let loaded = match s.mgr.ensure().await {
        Ok(l) => l,
        Err(e) => {
            let mut r = RespError::new(format!("{e:#}"), None);
            (r.status, r.kind) = (503, "server_error");
            return r.into_response();
        }
    };
    {
        let shared = loaded.shared.read().unwrap().clone();
        crate::settings::with_shared(&mut chat, &shared, loaded.config_effort().as_deref(), "openai");
    }
    let prep = match openai_to_messages(&chat, &loaded.tool_orders) {
        Ok(p) => p,
        Err(e) => return RespError::new(e.0, None).into_response(),
    };
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let opts = run::RunOpts { stream_tools: stream, budget: None, tool_choice: chat.get("tool_choice").cloned() };
    let started = match run::start(s.mgr.clone(), loaded, prep, chat.clone(), opts).await {
        Ok(st) => st,
        Err(run::Refusal(code, msg)) => {
            let mut r = RespError::new(msg, None);
            r.status = code;
            return r.into_response();
        }
    };
    let encrypted = req
        .get("include")
        .and_then(Value::as_array)
        .is_some_and(|a| a.iter().any(|x| x.as_str() == Some("reasoning.encrypted_content")));
    let mut asm = Assembler::new(&req, &started.model, started.prompt_tokens as u64, names, encrypted);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Value>(256);
    let mut events = started.events;
    tokio::spawn(async move {
        for e in asm.start() {
            let _ = tx.send(e).await;
        }
        while let Some(ev) = events.recv().await {
            let out = match ev {
                RunEvent::Ping => vec![asm.in_progress()],
                RunEvent::Event(e) => asm.feed(e),
                RunEvent::Done(sum) => {
                    for e in asm.finish(&sum) {
                        let _ = tx.send(e).await;
                    }
                    return;
                }
                RunEvent::Error(msg) => {
                    let _ = tx.send(asm.failed(&msg)).await;
                    return;
                }
            };
            for e in out {
                if tx.send(e).await.is_err() {
                    return; // the client is gone: dropping `events` stops the engine
                }
            }
        }
    });
    if stream {
        let body = futures::stream::unfold(rx, |mut rx| async move {
            let e = rx.recv().await?;
            let kind = e["type"].as_str().unwrap_or("message").to_string();
            Some((Ok::<_, Infallible>(SseEvent::default().event(kind).data(e.to_string())), rx))
        });
        return Sse::new(body).keep_alive(KeepAlive::new().interval(Duration::from_secs(10))).into_response();
    }
    let mut last: Option<Value> = None;
    while let Some(e) = rx.recv().await {
        last = Some(e);
    }
    match last {
        Some(e) if e["type"] == "response.failed" => {
            let msg = e["response"]["error"]["message"].as_str().unwrap_or("error").to_string();
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error_body(&msg, "server_error", None, None))).into_response()
        }
        Some(mut e) => Json(e["response"].take()).into_response(),
        None => (StatusCode::INTERNAL_SERVER_ERROR,
                 Json(error_body("the engine ended the request without an answer", "server_error", None, None))).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codex_conversation_becomes_chat_messages() {
        let req = json!({
            "instructions": "Be brief.",
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Sandbox: on."}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List files"}]},
                {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "Use ls."}]},
                {"type": "function_call", "name": "shell", "call_id": "c1", "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call", "name": "read", "namespace": "mcp__fs", "call_id": "c2", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c2", "output": "ok"},
                {"type": "function_call_output", "call_id": "c1", "output": "a.txt"},
            ],
            "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}},
                      {"type": "namespace", "name": "mcp__fs", "description": "Files.", "tools": [{"name": "read"}]},
                      {"type": "web_search"}],
        });
        let (chat, names) = to_chat(&req).unwrap();
        let m = chat["messages"].as_array().unwrap();
        assert_eq!(m[0], json!({"role": "system", "content": "Be brief.\n\nSandbox: on."}));
        assert_eq!(m[2]["reasoning_content"], "Use ls.");
        assert_eq!(m[2]["tool_calls"][0]["function"]["arguments"], json!({"cmd": "ls"}));
        assert_eq!(m[2]["tool_calls"][1]["function"]["name"], "mcp__fs.read");
        let tools: Vec<&str> = chat["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
        assert_eq!(tools, ["shell", "mcp__fs.read"], "the hosted tool is left out");
        assert_eq!(names["mcp__fs__read"], (Some("mcp__fs".into()), "read".into(), "function".into()));
    }

    #[test]
    fn stateless_refusals() {
        assert_eq!(to_chat(&json!({"input": "x", "previous_response_id": "r"})).unwrap_err().code, Some("unsupported_parameter"));
        assert_eq!(to_chat(&json!({})).unwrap_err().code, Some("missing_required_parameter"));
        assert!(to_chat(&json!({"input": "x", "max_output_tokens": 0})).is_err());
    }

    #[test]
    fn encrypted_reasoning_round_trips() {
        let mut a = Assembler::new(&json!({}), "m", 3, ToolNames::new(), true);
        a.feed(Event::Reasoning("deep thought".into()));
        a.close("completed");
        let item = a.response["output"][0].clone();
        let back = reasoning_text(&json!({"encrypted_content": item["encrypted_content"]}), "x").unwrap();
        assert_eq!(back, "deep thought");
    }
}
