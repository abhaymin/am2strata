//! A request as the server's front end normalized it (strata_frontend::Prepared: the same for the OpenAI, Anthropic,
//! Responses and Gemini APIs) -> the provider's own API, and the provider's answer (a stream, or one JSON) -> the
//! engine's events (run::RunEvent), which each API's own code then writes in the client's shape.

use super::config::{Kind, Provider};
use crate::run::{RunEvent, RunOpts, Summary};
use base64::Engine as _;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap, VecDeque};
use strata_frontend::output::{Event, ToolCall};
use strata_frontend::request::tool_choice_of;
use strata_frontend::Prepared;

/// What was not sent to the provider, and why (the `x-strata-note` header).
pub type Notes = Vec<String>;

/// A thinking block's text and its signature (Anthropic): kept for the next round of the same request, whose
/// assistant turn must carry it back.
pub type Signatures = Vec<(String, String)>;

fn text_of(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts.iter().filter(|p| p["type"] == "text").filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

fn images_of(content: &Value) -> Vec<String> {
    content.as_array().into_iter().flatten().filter(|p| p["type"] == "image").filter_map(|p| p["source"].as_str().map(str::to_string)).collect()
}

/// A picture's media type from its first bytes.
fn sniff(b: &[u8]) -> &'static str {
    if b.starts_with(&[0x89, b'P', b'N', b'G']) {
        "image/png"
    } else if b.starts_with(&[0xff, 0xd8]) {
        "image/jpeg"
    } else if b.starts_with(b"GIF8") {
        "image/gif"
    } else if b.len() > 12 && &b[..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        "image/webp"
    } else {
        "image/png"
    }
}

/// Every picture of the conversation as a data: URL (a file or an address is read here, under the server's own
/// rules for them; the provider is sent the bytes, never a path on this PC).
pub async fn inline_images(prep: &mut Prepared) -> Result<(), String> {
    for m in prep.messages.iter_mut() {
        let Some(parts) = m.get_mut("content").and_then(Value::as_array_mut) else { continue };
        for p in parts.iter_mut().filter(|p| p["type"] == "image") {
            let src = p["source"].as_str().unwrap_or("").to_string();
            if src.starts_with("data:") && src.contains(";base64,") {
                continue;
            }
            let bytes = crate::media::load_source(&src).await.map_err(|e| e.to_string())?;
            p["source"] = json!(format!("data:{};base64,{}", sniff(&bytes), base64::engine::general_purpose::STANDARD.encode(&bytes)));
        }
    }
    Ok(())
}

/// The data: URL's media type and base64 data.
fn data_url(src: &str) -> (String, String) {
    let rest = src.strip_prefix("data:").unwrap_or(src);
    let (head, data) = rest.split_once(',').unwrap_or(("", rest));
    let mut mime = head.split(';').next().unwrap_or("").to_string();
    if !mime.starts_with("image/") {
        let first = base64::engine::general_purpose::STANDARD.decode(data.get(..32.min(data.len()) / 4 * 4).unwrap_or("")).unwrap_or_default();
        mime = sniff(&first).to_string();
    }
    (mime, data.to_string())
}

/// The request's max tokens, in whichever API's name it came.
fn max_tokens(req: &Value) -> Option<u64> {
    req.get("max_tokens").or_else(|| req.get("max_completion_tokens")).and_then(Value::as_u64).filter(|n| *n > 0)
}

/// The reasoning effort the client asked for: "low", "medium", "high", or "none" (thinking off).
fn effort(prep: &Prepared, req: &Value) -> Option<String> {
    if prep.kwargs.get("enable_thinking") == Some(&Value::Bool(false)) {
        return Some("none".into());
    }
    let asked = req.get("reasoning_effort").and_then(Value::as_str)
        .or_else(|| req.pointer("/reasoning/effort").and_then(Value::as_str))
        .or_else(|| prep.kwargs.get("reasoning_effort").and_then(Value::as_str))?;
    Some(match asked {
        "xhigh" | "max" | "maximum" | "high" => "high",
        "low" | "minimal" => "low",
        "none" | "off" | "disabled" => "none",
        _ => "medium",
    }.to_string())
}

/// A call's id the provider takes (Anthropic: letters, digits, - and _).
fn safe_id(id: &str) -> String {
    let s: String = id.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').take(64).collect();
    if s.is_empty() { "call".into() } else { s }
}

/// The calls of the assistant message `m` (index `i`) with an id each: their own, or one made from the position (the
/// server's tool loop keeps none).
fn calls_of(m: &Value, i: usize) -> Vec<(String, String, Value)> {
    m["tool_calls"].as_array().into_iter().flatten().enumerate().map(|(j, c)| {
        let id = c["id"].as_str().map(safe_id).unwrap_or_else(|| format!("call_am2_{i}_{j}"));
        let f = c.get("function").unwrap_or(c);
        (id, f["name"].as_str().unwrap_or("").to_string(), if f["arguments"].is_object() { f["arguments"].clone() } else { json!({}) })
    }).collect()
}

// ------------------------------------------------------------------ OpenAI Chat Completions

pub fn openai_messages(msgs: &[Value], notes: &mut Notes) -> Vec<Value> {
    let mut out = Vec::new();
    let mut pending: VecDeque<String> = VecDeque::new();
    let mut tool_images: Vec<String> = Vec::new();
    for (i, m) in msgs.iter().enumerate() {
        let role = m["role"].as_str().unwrap_or("user");
        if role != "tool" && !tool_images.is_empty() {
            let mut parts = vec![json!({"type": "text", "text": "[the pictures of the tool results above]"})];
            parts.extend(tool_images.drain(..).map(|u| json!({"type": "image_url", "image_url": {"url": u}})));
            out.push(json!({"role": "user", "content": parts}));
        }
        match role {
            "system" => out.push(json!({"role": "system", "content": text_of(&m["content"])})),
            "assistant" => {
                let text = text_of(&m["content"]);
                let mut a = json!({"role": "assistant", "content": if text.is_empty() { Value::Null } else { json!(text) }});
                let calls = calls_of(m, i);
                if !calls.is_empty() {
                    pending = calls.iter().map(|(id, _, _)| id.clone()).collect();
                    a["tool_calls"] = Value::Array(calls.into_iter().map(|(id, name, args)| json!({"id": id, "type": "function",
                        "function": {"name": name, "arguments": args.to_string()}})).collect());
                }
                out.push(a);
            }
            "tool" => {
                let id = match m["tool_call_id"].as_str() {
                    Some(given) => {
                        let id = safe_id(given);
                        pending.retain(|p| *p != id);
                        id
                    }
                    None => pending.pop_front().unwrap_or_else(|| format!("call_am2_{i}")),
                };
                tool_images.extend(images_of(&m["content"]));
                out.push(json!({"role": "tool", "tool_call_id": id, "content": text_of(&m["content"])}));
            }
            _ => {
                let images = images_of(&m["content"]);
                if images.is_empty() {
                    out.push(json!({"role": "user", "content": text_of(&m["content"])}));
                } else {
                    let parts: Vec<Value> = m["content"].as_array().into_iter().flatten().filter_map(|p| match p["type"].as_str() {
                        Some("text") => Some(json!({"type": "text", "text": p["text"]})),
                        Some("image") => Some(json!({"type": "image_url", "image_url": {"url": p["source"]}})),
                        _ => None,
                    }).collect();
                    out.push(json!({"role": "user", "content": parts}));
                }
            }
        }
    }
    if !tool_images.is_empty() {
        notes.push("pictures in the last tool results were sent as a user message (OpenAI's tool messages carry text)".into());
        let mut parts = vec![json!({"type": "text", "text": "[the pictures of the tool results above]"})];
        parts.extend(tool_images.drain(..).map(|u| json!({"type": "image_url", "image_url": {"url": u}})));
        out.push(json!({"role": "user", "content": parts}));
    }
    out
}

/// The body of POST {base}/chat/completions.
pub fn openai_body(model: &str, p: &Provider, prep: &Prepared, req: &Value, opts: &RunOpts, notes: &mut Notes) -> Value {
    let mut b = json!({"model": model, "messages": openai_messages(&prep.messages, notes), "stream": true,
                       "stream_options": {"include_usage": true}});
    for k in ["temperature", "top_p", "seed", "presence_penalty", "frequency_penalty"] {
        if let Some(v) = req.get(k).filter(|v| !v.is_null()) {
            b[k] = v.clone();
        }
    }
    if req.get("top_k").is_some_and(|v| !v.is_null()) {
        notes.push(format!("top_k is not sent to {} (the OpenAI API has none)", p.name));
    }
    if let Some(n) = max_tokens(req) {
        b[p.max_tokens_field.as_str()] = json!(n);
    }
    if let Ok(stops) = crate::run::stop_strings(req) {
        if !stops.is_empty() {
            b["stop"] = json!(stops.into_iter().take(4).collect::<Vec<_>>());
        }
    }
    if let Some(tools) = prep.tools.as_ref().filter(|t| !t.is_empty()) {
        b["tools"] = Value::Array(tools.iter().map(|t| json!({"type": "function", "function": {
            "name": t["name"], "description": t.get("description").cloned().unwrap_or(json!("")),
            "parameters": t.get("parameters").filter(|p| p.is_object()).cloned().unwrap_or(json!({"type": "object", "properties": {}}))}})).collect());
        match tool_choice_of(opts.tool_choice.as_ref()) {
            ("none", _) => b["tool_choice"] = json!("none"),
            ("required", _) => b["tool_choice"] = json!("required"),
            ("named", Some(n)) => b["tool_choice"] = json!({"type": "function", "function": {"name": n}}),
            _ => {}
        }
    }
    match effort(prep, req) {
        Some(e) if p.reasoning_effort => {
            b["reasoning_effort"] = json!(if e == "none" { "minimal".to_string() } else { e });
        }
        Some(e) if e != "none" => notes.push(format!("the reasoning effort is not sent to {} (\"reasoning_effort\": true on the provider sends it)", p.name)),
        _ => {}
    }
    b
}

// ------------------------------------------------------------------ Anthropic Messages

fn anthropic_content(content: &Value) -> Vec<Value> {
    match content {
        Value::Array(parts) => parts.iter().filter_map(|p| match p["type"].as_str() {
            Some("text") => p["text"].as_str().filter(|t| !t.is_empty()).map(|t| json!({"type": "text", "text": t})),
            Some("image") => {
                let (mime, data) = data_url(p["source"].as_str().unwrap_or(""));
                Some(json!({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}))
            }
            _ => None,
        }).collect(),
        other => {
            let t = text_of(other);
            if t.is_empty() { Vec::new() } else { vec![json!({"type": "text", "text": t})] }
        }
    }
}

/// (system, messages) for the Messages API. `signed`: thinking blocks this request's earlier rounds got.
pub fn anthropic_messages(msgs: &[Value], signed: &Signatures) -> (Option<String>, Vec<Value>) {
    let mut system: Vec<String> = Vec::new();
    let mut out: Vec<(String, Vec<Value>)> = Vec::new();
    let mut pending: VecDeque<String> = VecDeque::new();
    for (i, m) in msgs.iter().enumerate() {
        let role = m["role"].as_str().unwrap_or("user");
        let (r, blocks) = match role {
            "system" => {
                system.push(text_of(&m["content"]));
                continue;
            }
            "assistant" => {
                let mut blocks = Vec::new();
                if let Some(r) = m["reasoning_content"].as_str() {
                    if let Some((_, sig)) = signed.iter().find(|(t, _)| t.trim() == r.trim()) {
                        blocks.push(json!({"type": "thinking", "thinking": r, "signature": sig}));
                    }
                }
                blocks.extend(anthropic_content(&m["content"]));
                let calls = calls_of(m, i);
                pending = calls.iter().map(|(id, _, _)| id.clone()).collect();
                blocks.extend(calls.into_iter().map(|(id, name, args)| json!({"type": "tool_use", "id": id, "name": name, "input": args})));
                ("assistant", blocks)
            }
            "tool" => {
                let id = match m["tool_call_id"].as_str() {
                    Some(given) => {
                        let id = safe_id(given);
                        pending.retain(|p| *p != id);
                        id
                    }
                    None => pending.pop_front().unwrap_or_else(|| format!("call_am2_{i}")),
                };
                let content = anthropic_content(&m["content"]);
                let content = if content.is_empty() { json!("") } else { Value::Array(content) };
                ("user", vec![json!({"type": "tool_result", "tool_use_id": id, "content": content})])
            }
            _ => ("user", anthropic_content(&m["content"])),
        };
        if blocks.is_empty() {
            continue;
        }
        // one turn per role in a row (the API takes them alternating)
        match out.last_mut() {
            Some((last, b)) if last == r => b.extend(blocks),
            _ => out.push((r.to_string(), blocks)),
        }
    }
    if out.first().is_some_and(|(r, _)| r == "assistant") {
        out.insert(0, ("user".into(), vec![json!({"type": "text", "text": "(continue)"})]));
    }
    let system = system.join("\n\n");
    (if system.is_empty() { None } else { Some(system) },
     out.into_iter().map(|(r, b)| json!({"role": r, "content": b})).collect())
}

/// The body of POST {base}/v1/messages.
pub fn anthropic_body(model: &str, p: &Provider, prep: &Prepared, req: &Value, opts: &RunOpts, signed: &Signatures, notes: &mut Notes) -> Value {
    let (system, messages) = anthropic_messages(&prep.messages, signed);
    let max = max_tokens(req).unwrap_or(p.max_tokens);
    let mut b = json!({"model": model, "max_tokens": max, "messages": messages, "stream": true});
    if let Some(s) = system {
        b["system"] = json!(s);
    }
    for k in ["temperature", "top_p", "top_k"] {
        if let Some(v) = req.get(k).filter(|v| !v.is_null()) {
            b[k] = v.clone();
        }
    }
    if let Ok(stops) = crate::run::stop_strings(req) {
        if !stops.is_empty() {
            b["stop_sequences"] = json!(stops);
        }
    }
    let mut forced = false;
    if let Some(tools) = prep.tools.as_ref().filter(|t| !t.is_empty()) {
        b["tools"] = Value::Array(tools.iter().map(|t| json!({"name": t["name"], "description": t.get("description").cloned().unwrap_or(json!("")),
            "input_schema": t.get("parameters").filter(|p| p.is_object()).cloned().unwrap_or(json!({"type": "object", "properties": {}}))})).collect());
        match tool_choice_of(opts.tool_choice.as_ref()) {
            ("none", _) => b["tool_choice"] = json!({"type": "none"}),
            ("required", _) => { b["tool_choice"] = json!({"type": "any"}); forced = true; }
            ("named", Some(n)) => { b["tool_choice"] = json!({"type": "tool", "name": n}); forced = true; }
            _ => {}
        }
    }
    // thinking: the client's budget, else its effort
    let budget = match (opts.budget, effort(prep, req).as_deref()) {
        (_, Some("none")) => None,
        (Some(n), _) => Some(n),
        (None, Some("low")) => Some(1024),
        (None, Some("medium")) => Some(4096),
        (None, Some("high")) => Some(16000),
        _ => None,
    };
    if let Some(mut n) = budget {
        // the turn the tool loop goes on from must carry its signed thinking back; a client's history has none
        let last_assistant = prep.messages.iter().rposition(|m| m["role"] == "assistant");
        let open_call = last_assistant.is_some_and(|i| prep.messages[i]["tool_calls"].as_array().is_some_and(|c| !c.is_empty())
            && prep.messages[i + 1..].iter().all(|m| m["role"] == "tool"));
        let signed_turn = last_assistant.and_then(|i| prep.messages[i]["reasoning_content"].as_str())
            .is_some_and(|r| signed.iter().any(|(t, _)| t.trim() == r.trim()));
        if n >= max {
            n = max.saturating_sub(1);
        }
        if forced {
            notes.push("thinking is off for this request: Anthropic does not think with a forced tool_choice".into());
        } else if open_call && !signed_turn {
            notes.push("thinking is off for this request: the tool call it continues has no signed thinking block".into());
        } else if n < 1024 {
            notes.push("thinking is off for this request: max_tokens leaves no room for Anthropic's smallest thinking budget (1024)".into());
        } else {
            b["thinking"] = json!({"type": "enabled", "budget_tokens": n});
            for k in ["temperature", "top_k", "top_p"] {
                if b.as_object_mut().and_then(|o| o.remove(k)).is_some() {
                    notes.push(format!("{k} is not sent with thinking on (Anthropic refuses it)"));
                }
            }
        }
    }
    b
}

// ------------------------------------------------------------------ answers

/// (prompt, completion, cached) tokens of a provider's usage block, either API's names.
fn usage_of(u: &Value) -> Option<(u64, u64, u64)> {
    let n = |k: &str| u.get(k).and_then(Value::as_u64);
    if let Some(p) = n("prompt_tokens") {
        return Some((p, n("completion_tokens").unwrap_or(0), u.pointer("/prompt_tokens_details/cached_tokens").and_then(Value::as_u64).unwrap_or(0)));
    }
    let input = n("input_tokens")?;
    let read = n("cache_read_input_tokens").unwrap_or(0);
    Some((input + read + n("cache_creation_input_tokens").unwrap_or(0), n("output_tokens").unwrap_or(0), read))
}

fn arguments(text: &str) -> Map<String, Value> {
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(m)) => m,
        _ if text.trim().is_empty() => Map::new(),
        _ => {
            let mut m = Map::new();
            m.insert("arguments".into(), json!(text));
            m
        }
    }
}

fn summary(finish: &str, usage: Option<(u64, u64, u64)>, stop_sequence: Option<String>) -> Summary {
    let (p, c, cached) = usage.unwrap_or((0, 0, 0));
    Summary { finish: finish.to_string(), completion_tokens: c, reused: cached, stop_sequence, timings: Value::Null,
              prompt_tokens: usage.map(|_| p) }
}

fn openai_finish(f: &str) -> &'static str {
    match f {
        "length" => "length",
        _ => "stop",
    }
}

/// An OpenAI Chat Completions stream, chunk by chunk.
pub struct OpenAiStream {
    stream_tools: bool,
    /// by index: id, name, arguments so far, whether its start was told
    calls: BTreeMap<u64, (String, String, String, bool)>,
    finish: Option<String>,
    usage: Option<(u64, u64, u64)>,
}

impl OpenAiStream {
    pub fn new(stream_tools: bool) -> Self {
        OpenAiStream { stream_tools, calls: BTreeMap::new(), finish: None, usage: None }
    }

    pub fn feed(&mut self, v: &Value) -> Vec<RunEvent> {
        let mut out = Vec::new();
        if let Some(e) = v.get("error").filter(|e| !e.is_null()) {
            out.push(RunEvent::Error(format!("the provider: {}", e["message"].as_str().unwrap_or("an error"))));
            return out;
        }
        if let Some(u) = v.get("usage").and_then(usage_of) {
            self.usage = Some(u);
        }
        let Some(ch) = v["choices"].get(0) else { return out };
        let d = ch.get("delta").or_else(|| ch.get("message")).cloned().unwrap_or(Value::Null);
        for k in ["reasoning_content", "reasoning"] {
            if let Some(t) = d[k].as_str().filter(|t| !t.is_empty()) {
                out.push(RunEvent::Event(Event::Reasoning(t.to_string())));
                break;
            }
        }
        if let Some(t) = d["content"].as_str().filter(|t| !t.is_empty()) {
            out.push(RunEvent::Event(Event::Content(t.to_string())));
        }
        for (k, tc) in d["tool_calls"].as_array().into_iter().flatten().enumerate() {
            let idx = tc["index"].as_u64().unwrap_or(k as u64);
            let e = self.calls.entry(idx).or_default();
            if let Some(id) = tc["id"].as_str().filter(|i| !i.is_empty()) {
                e.0 = id.to_string();
            }
            if let Some(n) = tc["function"]["name"].as_str() {
                e.1.push_str(n);
            }
            let frag = match &tc["function"]["arguments"] {
                Value::String(s) => s.clone(),
                Value::Object(_) => tc["function"]["arguments"].to_string(),
                _ => String::new(),
            };
            e.2.push_str(&frag);
            if self.stream_tools && !e.1.is_empty() {
                if e.0.is_empty() {
                    e.0 = format!("call_{}", &uuid::Uuid::new_v4().simple().to_string()[..24]);
                }
                if !e.3 {
                    e.3 = true;
                    out.push(RunEvent::Event(Event::ToolStart { id: e.0.clone(), name: e.1.clone() }));
                    if !e.2.is_empty() {
                        out.push(RunEvent::Event(Event::ToolArgs { id: e.0.clone(), text: e.2.clone() }));
                    }
                } else if !frag.is_empty() {
                    out.push(RunEvent::Event(Event::ToolArgs { id: e.0.clone(), text: frag }));
                }
            }
        }
        if let Some(f) = ch["finish_reason"].as_str() {
            self.finish = Some(f.to_string());
        }
        out
    }

    /// The stream ended: the whole calls, then the end. `complete`: it ended as the API ends a stream.
    pub fn finish(&mut self, complete: bool) -> Vec<RunEvent> {
        if !complete && self.finish.is_none() {
            return vec![RunEvent::Error("the provider's answer stopped before its end".into())];
        }
        let mut out: Vec<RunEvent> = std::mem::take(&mut self.calls).into_values().map(|(id, name, args, _)| {
            let id = if id.is_empty() { format!("call_{}", &uuid::Uuid::new_v4().simple().to_string()[..24]) } else { id };
            RunEvent::Event(Event::ToolCall(ToolCall { id, name, arguments: arguments(&args) }))
        }).collect();
        out.push(RunEvent::Done(summary(openai_finish(self.finish.as_deref().unwrap_or("stop")), self.usage, None)));
        out
    }
}

/// A whole OpenAI chat.completion (a provider that answered without streaming).
pub fn openai_json(v: &Value) -> Vec<RunEvent> {
    let mut s = OpenAiStream::new(false);
    let mut msg = v["choices"][0]["message"].clone();
    // the message's calls carry no index
    if let Some(calls) = msg.get_mut("tool_calls").and_then(Value::as_array_mut) {
        for (i, c) in calls.iter_mut().enumerate() {
            c["index"] = json!(i);
        }
    }
    let mut out = s.feed(&json!({"choices": [{"delta": msg, "finish_reason": v["choices"][0]["finish_reason"]}], "usage": v["usage"],
                                 "error": v.get("error").cloned().unwrap_or(Value::Null)}));
    if matches!(out.last(), Some(RunEvent::Error(_))) {
        return out;
    }
    out.extend(s.finish(true));
    out
}

#[derive(Default)]
struct Block {
    kind: String,
    id: String,
    name: String,
    text: String,
    signature: String,
}

/// An Anthropic Messages stream, event by event.
pub struct AnthropicStream {
    stream_tools: bool,
    blocks: HashMap<u64, Block>,
    usage: Map<String, Value>,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    /// the thinking blocks with their signatures, for the next round
    pub signed: Signatures,
    pub ended: bool,
}

impl AnthropicStream {
    pub fn new(stream_tools: bool) -> Self {
        AnthropicStream { stream_tools, blocks: HashMap::new(), usage: Map::new(), stop_reason: None, stop_sequence: None,
                          signed: Vec::new(), ended: false }
    }

    pub fn feed(&mut self, v: &Value) -> Vec<RunEvent> {
        let mut out = Vec::new();
        let idx = v["index"].as_u64().unwrap_or(0);
        match v["type"].as_str().unwrap_or("") {
            "ping" => out.push(RunEvent::Ping),
            "error" => out.push(RunEvent::Error(format!("the provider: {}", v["error"]["message"].as_str().unwrap_or("an error")))),
            "message_start" => {
                if let Some(u) = v["message"]["usage"].as_object() {
                    self.usage.extend(u.clone());
                }
            }
            "content_block_start" => {
                let cb = &v["content_block"];
                let b = Block { kind: cb["type"].as_str().unwrap_or("").to_string(), id: cb["id"].as_str().unwrap_or("").to_string(),
                                name: cb["name"].as_str().unwrap_or("").to_string(), ..Default::default() };
                match b.kind.as_str() {
                    "text" => {
                        if let Some(t) = cb["text"].as_str().filter(|t| !t.is_empty()) {
                            out.push(RunEvent::Event(Event::Content(t.to_string())));
                        }
                    }
                    "tool_use" if self.stream_tools => out.push(RunEvent::Event(Event::ToolStart { id: b.id.clone(), name: b.name.clone() })),
                    _ => {}
                }
                self.blocks.insert(idx, b);
            }
            "content_block_delta" => {
                let d = &v["delta"];
                let b = self.blocks.entry(idx).or_default();
                match d["type"].as_str().unwrap_or("") {
                    "text_delta" => {
                        if let Some(t) = d["text"].as_str().filter(|t| !t.is_empty()) {
                            out.push(RunEvent::Event(Event::Content(t.to_string())));
                        }
                    }
                    "thinking_delta" => {
                        if let Some(t) = d["thinking"].as_str().filter(|t| !t.is_empty()) {
                            b.text.push_str(t);
                            out.push(RunEvent::Event(Event::Reasoning(t.to_string())));
                        }
                    }
                    "signature_delta" => b.signature.push_str(d["signature"].as_str().unwrap_or("")),
                    "input_json_delta" => {
                        let t = d["partial_json"].as_str().unwrap_or("");
                        b.text.push_str(t);
                        if self.stream_tools && !t.is_empty() {
                            out.push(RunEvent::Event(Event::ToolArgs { id: b.id.clone(), text: t.to_string() }));
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                if let Some(b) = self.blocks.remove(&idx) {
                    match b.kind.as_str() {
                        "tool_use" => out.push(RunEvent::Event(Event::ToolCall(ToolCall { id: b.id, name: b.name, arguments: arguments(&b.text) }))),
                        "thinking" if !b.signature.is_empty() => self.signed.push((b.text, b.signature)),
                        _ => {}
                    }
                }
            }
            "message_delta" => {
                if let Some(r) = v["delta"]["stop_reason"].as_str() {
                    self.stop_reason = Some(r.to_string());
                }
                if let Some(s) = v["delta"]["stop_sequence"].as_str() {
                    self.stop_sequence = Some(s.to_string());
                }
                if let Some(u) = v["usage"].as_object() {
                    self.usage.extend(u.clone());
                }
            }
            "message_stop" => self.ended = true,
            _ => {}
        }
        out
    }

    pub fn finish(&mut self) -> Vec<RunEvent> {
        if !self.ended && self.stop_reason.is_none() {
            return vec![RunEvent::Error("the provider's answer stopped before its end".into())];
        }
        let finish = match self.stop_reason.as_deref() {
            Some("max_tokens") => "length",
            _ => "stop",
        };
        vec![RunEvent::Done(summary(finish, usage_of(&Value::Object(self.usage.clone())), self.stop_sequence.take()))]
    }
}

/// A whole Anthropic message (a provider that answered without streaming).
pub fn anthropic_json(v: &Value, signed: &mut Signatures) -> Vec<RunEvent> {
    if v["type"] == "error" {
        return vec![RunEvent::Error(format!("the provider: {}", v["error"]["message"].as_str().unwrap_or("an error")))];
    }
    let mut s = AnthropicStream::new(false);
    let mut out = s.feed(&json!({"type": "message_start", "message": {"usage": v["usage"]}}));
    for (i, b) in v["content"].as_array().into_iter().flatten().enumerate() {
        let mut start = b.clone();
        for k in ["text", "thinking", "input", "signature"] {
            if let Some(o) = start.as_object_mut() {
                o.remove(k);
            }
        }
        out.extend(s.feed(&json!({"type": "content_block_start", "index": i, "content_block": start})));
        let delta = match b["type"].as_str() {
            Some("text") => vec![json!({"type": "text_delta", "text": b["text"]})],
            Some("thinking") => vec![json!({"type": "thinking_delta", "thinking": b["thinking"]}), json!({"type": "signature_delta", "signature": b["signature"]})],
            Some("tool_use") => vec![json!({"type": "input_json_delta", "partial_json": b["input"].to_string()})],
            _ => vec![],
        };
        for d in delta {
            out.extend(s.feed(&json!({"type": "content_block_delta", "index": i, "delta": d})));
        }
        out.extend(s.feed(&json!({"type": "content_block_stop", "index": i})));
    }
    out.extend(s.feed(&json!({"type": "message_delta", "delta": {"stop_reason": v["stop_reason"], "stop_sequence": v["stop_sequence"]}})));
    out.extend(s.feed(&json!({"type": "message_stop"})));
    out.extend(s.finish());
    signed.append(&mut s.signed);
    out
}

/// The provider kind's body.
pub fn body(kind: Kind, model: &str, p: &Provider, prep: &Prepared, req: &Value, opts: &RunOpts, signed: &Signatures, notes: &mut Notes) -> Value {
    match kind {
        Kind::OpenAi => openai_body(model, p, prep, req, opts, notes),
        Kind::Anthropic => anthropic_body(model, p, prep, req, opts, signed, notes),
    }
}

/// About how many tokens a request's text is (4 characters a token; pictures not counted).
pub fn estimate_tokens(v: &Value) -> u64 {
    fn chars(v: &Value) -> usize {
        match v {
            Value::String(s) if s.starts_with("data:") => 0,
            Value::String(s) => s.chars().count(),
            Value::Array(a) => a.iter().map(chars).sum(),
            Value::Object(o) => o.iter().filter(|(k, _)| !matches!(k.as_str(), "data" | "image_url" | "inlineData" | "inline_data")).map(|(_, v)| chars(v)).sum(),
            _ => 0,
        }
    }
    (chars(v) as u64).div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::config::Config;
    use strata_frontend::request::anthropic_to_messages;
    use strata_frontend::{openai_to_messages, ToolOrders};

    fn providers() -> (Provider, Provider) {
        let c = Config::parse(&json!({"providers": {
            "openai": {"kind": "openai", "base_url": "https://api.openai.com/v1"},
            "anthropic": {"kind": "anthropic", "base_url": "https://api.anthropic.com"}}})).unwrap();
        (c.provider("openai").unwrap().clone(), c.provider("anthropic").unwrap().clone())
    }

    const PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAAEElEQVR4nGP4z8DAwMDAAAAQAAFJ6qrNAAAAAElFTkSuQmCC";

    fn openai_req() -> Value {
        json!({"model": "openai/gpt-5.5", "max_tokens": 300, "temperature": 0.2, "stop": ["END"], "reasoning_effort": "high",
               "messages": [{"role": "system", "content": "Be brief."},
                            {"role": "user", "content": [{"type": "text", "text": "What is this?"}, {"type": "image_url", "image_url": {"url": PNG}}]},
                            {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "add", "arguments": "{\"a\":2,\"b\":3}"}}]},
                            {"role": "tool", "tool_call_id": "call_1", "content": "5"}],
               "tools": [{"type": "function", "function": {"name": "add", "description": "Add.", "parameters": {"type": "object", "properties": {"a": {"type": "number"}}}}}],
               "tool_choice": "auto"})
    }

    fn opts_of(req: &Value, budget: Option<u64>) -> RunOpts {
        RunOpts { tool_choice: req.get("tool_choice").cloned(), budget, ..Default::default() }
    }

    #[test]
    fn an_openai_client_request_in_both_provider_apis() {
        let (o, a) = providers();
        let req = openai_req();
        let prep = openai_to_messages(&req, &ToolOrders::default()).unwrap();
        let mut notes = Notes::new();
        let b = openai_body("gpt-5.5", &o, &prep, &req, &opts_of(&req, None), &mut notes);
        assert_eq!(b["model"], "gpt-5.5");
        assert_eq!((b["max_completion_tokens"].as_u64(), b["temperature"].as_f64(), b["stop"].clone()), (Some(300), Some(0.2), json!(["END"])));
        assert_eq!(b["reasoning_effort"], "high");
        assert_eq!(b["messages"][0], json!({"role": "system", "content": "Be brief."}));
        assert_eq!(b["messages"][1]["content"][1], json!({"type": "image_url", "image_url": {"url": PNG}}));
        assert_eq!(b["messages"][2]["tool_calls"][0]["id"], "call_1");
        assert_eq!(serde_json::from_str::<Value>(b["messages"][2]["tool_calls"][0]["function"]["arguments"].as_str().unwrap()).unwrap(), json!({"a": 2, "b": 3}));
        assert_eq!(b["messages"][3], json!({"role": "tool", "tool_call_id": "call_1", "content": "5"}));
        assert_eq!(b["tools"][0]["function"]["name"], "add");
        assert!(b.get("tool_choice").is_none(), "auto is the default");
        assert!(notes.is_empty(), "{notes:?}");

        let mut notes = Notes::new();
        let b = anthropic_body("claude-sonnet-5-5", &a, &prep, &req, &opts_of(&req, None), &Vec::new(), &mut notes);
        assert_eq!((b["max_tokens"].as_u64(), b["system"].as_str()), (Some(300), Some("Be brief.")));
        assert_eq!(b["stop_sequences"], json!(["END"]));
        let m = b["messages"].as_array().unwrap();
        assert_eq!(m[0]["content"][1]["source"]["type"], "base64");
        assert_eq!(m[0]["content"][1]["source"]["media_type"], "image/png");
        assert_eq!(m[1]["content"][0], json!({"type": "tool_use", "id": "call_1", "name": "add", "input": {"a": 2, "b": 3}}));
        assert_eq!(m[2]["content"][0]["type"], "tool_result");
        assert_eq!(m[2]["content"][0]["tool_use_id"], "call_1");
        assert_eq!(b["tools"][0]["input_schema"]["properties"]["a"]["type"], "number");
        // high effort asks thinking, which max_tokens 300 has no room for: said, not sent
        assert!(b.get("thinking").is_none());
        assert!(notes.iter().any(|n| n.contains("thinking is off")), "{notes:?}");
    }

    #[test]
    fn an_anthropic_client_request_in_both_provider_apis() {
        let (o, a) = providers();
        let req = json!({"model": "anthropic/claude-sonnet-5-5", "max_tokens": 20000, "system": "S.", "stop_sequences": ["X"],
            "thinking": {"type": "enabled", "budget_tokens": 4000}, "temperature": 0.5,
            "tools": [{"name": "add", "input_schema": {"type": "object"}}], "tool_choice": {"type": "tool", "name": "add"},
            "messages": [{"role": "user", "content": "2+3?"},
                         {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 2}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "5"}]}]});
        let (prep, budget) = anthropic_to_messages(&req, &ToolOrders::default(), true).unwrap();
        let mut notes = Notes::new();
        let b = anthropic_body("claude-sonnet-5-5", &a, &prep, &req, &opts_of(&req, budget), &Vec::new(), &mut notes);
        assert_eq!(b["tool_choice"], json!({"type": "tool", "name": "add"}));
        assert!(b.get("thinking").is_none(), "a forced tool_choice: no thinking");
        assert_eq!(b["temperature"], 0.5);
        assert_eq!(b["messages"][1]["content"][0]["id"], "toolu_1");
        assert_eq!(b["messages"][2]["content"][0]["tool_use_id"], "toolu_1");
        // without the forced call it thinks, and temperature goes (Anthropic refuses both together)
        let mut req2 = req.clone();
        req2.as_object_mut().unwrap().remove("tool_choice");
        req2["messages"] = json!([{"role": "user", "content": "2+3?"}]);
        let (prep2, budget2) = anthropic_to_messages(&req2, &ToolOrders::default(), true).unwrap();
        let mut notes = Notes::new();
        let b = anthropic_body("m", &a, &prep2, &req2, &opts_of(&req2, budget2), &Vec::new(), &mut notes);
        assert_eq!(b["thinking"], json!({"type": "enabled", "budget_tokens": 4000}));
        assert!(b.get("temperature").is_none() && notes.iter().any(|n| n.contains("temperature")), "{notes:?}");

        let mut notes = Notes::new();
        let b = openai_body("gpt", &o, &prep, &req, &opts_of(&req, budget), &mut notes);
        assert_eq!(b["messages"][0], json!({"role": "system", "content": "S."}));
        assert_eq!(b["messages"][2]["tool_calls"][0]["id"], "toolu_1");
        assert_eq!(b["messages"][3]["tool_call_id"], "toolu_1");
        assert_eq!(b["tool_choice"], json!({"type": "function", "function": {"name": "add"}}));
        assert_eq!(b["stop"], json!(["X"]));
        assert_eq!(b["max_completion_tokens"], 20000);
    }

    #[test]
    fn responses_and_gemini_requests_reach_the_providers_too() {
        let (o, a) = providers();
        let r = json!({"model": "openai/gpt-5.5", "instructions": "Terse.", "max_output_tokens": 64,
                       "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]});
        let (chat, _) = crate::responses::to_chat(&r).ok().unwrap();
        let prep = openai_to_messages(&chat, &ToolOrders::default()).unwrap();
        let mut notes = Notes::new();
        let b = openai_body("gpt-5.5", &o, &prep, &chat, &RunOpts::default(), &mut notes);
        assert_eq!(b["messages"], json!([{"role": "system", "content": "Terse."}, {"role": "user", "content": "hi"}]));
        assert_eq!(b["max_completion_tokens"], 64);
        let g = json!({"systemInstruction": {"parts": [{"text": "Sys."}]}, "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                       "generationConfig": {"maxOutputTokens": 50, "stopSequences": ["Z"]}});
        let chat = crate::gemini::to_openai("anthropic/claude-sonnet-5-5", &g).unwrap();
        let prep = openai_to_messages(&chat, &ToolOrders::default()).unwrap();
        let b = anthropic_body("claude-sonnet-5-5", &a, &prep, &chat, &RunOpts::default(), &Vec::new(), &mut notes);
        assert_eq!((b["system"].as_str(), b["max_tokens"].as_u64()), (Some("Sys."), Some(50)));
        assert_eq!(b["messages"], json!([{"role": "user", "content": [{"type": "text", "text": "hello"}]}]));
    }

    #[test]
    fn the_tool_loops_turns_get_ids_and_signed_thinking() {
        let (_, a) = providers();
        // the server's tool loop keeps no ids: they are made from the positions, the same on both sides
        let msgs = vec![json!({"role": "user", "content": "x"}),
                        json!({"role": "assistant", "content": "", "reasoning_content": "think", "tool_calls": [{"function": {"name": "calc__add", "arguments": {"a": 1}}}]}),
                        json!({"role": "tool", "content": "1"})];
        let mut notes = Notes::new();
        let o = openai_messages(&msgs, &mut notes);
        assert_eq!(o[1]["tool_calls"][0]["id"], o[2]["tool_call_id"]);
        let (_, m) = anthropic_messages(&msgs, &vec![("think".into(), "sig".into())]);
        assert_eq!(m[1]["content"][0], json!({"type": "thinking", "thinking": "think", "signature": "sig"}));
        assert_eq!(m[1]["content"][1]["id"], m[2]["content"][0]["tool_use_id"]);
        let prep = Prepared { messages: msgs, tools: None, kwargs: Map::new() };
        let b = anthropic_body("m", &a, &prep, &json!({"max_tokens": 9000}), &RunOpts { budget: Some(2000), ..Default::default() },
                               &vec![("think".into(), "sig".into())], &mut notes);
        assert_eq!(b["thinking"]["budget_tokens"], 2000, "a signed turn goes on thinking");
        let b = anthropic_body("m", &a, &prep, &json!({"max_tokens": 9000}), &RunOpts { budget: Some(2000), ..Default::default() }, &Vec::new(), &mut notes);
        assert!(b.get("thinking").is_none(), "an unsigned one cannot");
    }

    fn texts(evs: &[RunEvent]) -> (String, String, Vec<ToolCall>, Option<Summary>) {
        let (mut c, mut r, mut calls, mut done) = (String::new(), String::new(), Vec::new(), None);
        for e in evs {
            match e {
                RunEvent::Event(Event::Content(t)) => c.push_str(t),
                RunEvent::Event(Event::Reasoning(t)) => r.push_str(t),
                RunEvent::Event(Event::ToolCall(t)) => calls.push(t.clone()),
                RunEvent::Done(s) => done = Some(s.clone()),
                _ => {}
            }
        }
        (c, r, calls, done)
    }

    #[test]
    fn an_openai_stream_becomes_events() {
        let mut s = OpenAiStream::new(true);
        let mut evs = Vec::new();
        for c in [json!({"choices": [{"delta": {"role": "assistant", "reasoning_content": "hm"}}]}),
                  json!({"choices": [{"delta": {"content": "Hel"}}]}), json!({"choices": [{"delta": {"content": "lo"}}]}),
                  json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_9", "function": {"name": "add", "arguments": "{\"a\""}}]}}]}),
                  json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ":2}"}}]}, "finish_reason": "tool_calls"}]}),
                  json!({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 3}}})] {
            evs.extend(s.feed(&c));
        }
        evs.extend(s.finish(true));
        assert!(evs.iter().any(|e| matches!(e, RunEvent::Event(Event::ToolStart { id, name }) if id == "call_9" && name == "add")));
        let args: String = evs.iter().filter_map(|e| match e { RunEvent::Event(Event::ToolArgs { text, .. }) => Some(text.clone()), _ => None }).collect();
        assert_eq!(args, "{\"a\":2}");
        let (c, r, calls, done) = texts(&evs);
        assert_eq!((c.as_str(), r.as_str()), ("Hello", "hm"));
        assert_eq!((calls[0].id.as_str(), calls[0].arguments["a"].as_i64()), ("call_9", Some(2)));
        let d = done.unwrap();
        assert_eq!((d.finish.as_str(), d.prompt_tokens, d.completion_tokens, d.reused), ("stop", Some(11), 7, 3));
        // cut off before its end
        let mut s = OpenAiStream::new(false);
        s.feed(&json!({"choices": [{"delta": {"content": "x"}}]}));
        assert!(matches!(s.finish(false)[0], RunEvent::Error(_)));
        // the same answer as one JSON
        let evs = openai_json(&json!({"choices": [{"message": {"content": "Hi", "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]},
                                                    "finish_reason": "length"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}));
        let (c, _, calls, done) = texts(&evs);
        assert_eq!((c.as_str(), calls.len(), done.unwrap().finish.as_str()), ("Hi", 1, "length"));
    }

    #[test]
    fn an_anthropic_stream_becomes_events() {
        let mut s = AnthropicStream::new(true);
        let mut evs = Vec::new();
        for e in [json!({"type": "message_start", "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 4}}}),
                  json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
                  json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "plan"}}),
                  json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "SIG"}}),
                  json!({"type": "content_block_stop", "index": 0}),
                  json!({"type": "ping"}),
                  json!({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
                  json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Sure."}}),
                  json!({"type": "content_block_stop", "index": 1}),
                  json!({"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_7", "name": "add", "input": {}}}),
                  json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"b\": 4}"}}),
                  json!({"type": "content_block_stop", "index": 2}),
                  json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
                  json!({"type": "message_stop"})] {
            evs.extend(s.feed(&e));
        }
        evs.extend(s.finish());
        let (c, r, calls, done) = texts(&evs);
        assert_eq!((c.as_str(), r.as_str()), ("Sure.", "plan"));
        assert_eq!((calls[0].id.as_str(), calls[0].arguments["b"].as_i64()), ("toolu_7", Some(4)));
        let d = done.unwrap();
        assert_eq!((d.finish.as_str(), d.prompt_tokens, d.completion_tokens, d.reused), ("stop", Some(14), 9, 4));
        assert_eq!(s.signed, vec![("plan".to_string(), "SIG".to_string())]);
        // stop_sequence and max_tokens
        let mut signed = Vec::new();
        let evs = anthropic_json(&json!({"content": [{"type": "text", "text": "abc"}], "stop_reason": "stop_sequence", "stop_sequence": "END",
                                          "usage": {"input_tokens": 3, "output_tokens": 1}}), &mut signed);
        let d = texts(&evs).3.unwrap();
        assert_eq!(d.stop_sequence.as_deref(), Some("END"));
        let evs = anthropic_json(&json!({"content": [], "stop_reason": "max_tokens", "usage": {"input_tokens": 3, "output_tokens": 1}}), &mut signed);
        assert_eq!(texts(&evs).3.unwrap().finish, "length");
        let evs = anthropic_json(&json!({"type": "error", "error": {"message": "overloaded"}}), &mut signed);
        assert!(matches!(&evs[0], RunEvent::Error(m) if m.contains("overloaded")));
    }

    #[test]
    fn tokens_are_estimated_without_the_pictures() {
        assert_eq!(estimate_tokens(&json!({"messages": [{"role": "user", "content": "abcdefgh"}]})), 3);
        assert_eq!(estimate_tokens(&json!({"messages": [{"content": [{"type": "image_url", "image_url": {"url": PNG}}]}]})), 3);
    }
}
