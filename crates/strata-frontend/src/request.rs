//! OpenAI Chat Completions request -> (template messages, template tools, template kwargs).
//! Port of serve/frontend.py `openai_to_messages` and its helpers.

use crate::{RequestError, Result};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

const TEXT_PARTS: [&str; 2] = ["text", "input_text"];
const IMAGE_PARTS: [&str; 3] = ["image_url", "input_image", "image"];

#[derive(Debug, Clone, PartialEq)]
pub struct Prepared {
    pub messages: Vec<Value>,
    pub tools: Option<Vec<Value>>,
    pub kwargs: Map<String, Value>,
}

fn bad(msg: impl Into<String>) -> RequestError {
    RequestError(msg.into())
}

/// Python truthiness of a JSON value.
pub(crate) fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// A part counts as text when its type is text / input_text, or it has no type at all.
fn is_text_part(p: &Value) -> bool {
    match p.get("type") {
        None | Some(Value::Null) => true,
        Some(Value::String(t)) => TEXT_PARTS.contains(&t.as_str()),
        _ => false,
    }
}

fn is_image_part(p: &Value) -> bool {
    p.get("type").and_then(Value::as_str).is_some_and(|t| IMAGE_PARTS.contains(&t))
}

pub(crate) fn text_of(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter(|p| p.is_object() && is_text_part(p))
            .map(|p| p.get("text").and_then(Value::as_str).unwrap_or(""))
            .collect(),
        _ => String::new(),
    }
}

pub(crate) fn has_image(content: &Value) -> bool {
    content.as_array().is_some_and(|parts| parts.iter().any(|p| p.is_object() && is_image_part(p)))
}

fn image_source(part: &Value) -> String {
    if part.get("type").and_then(Value::as_str) == Some("image") {
        let src = part.get("source").cloned().unwrap_or(json!({}));
        if src.get("type").and_then(Value::as_str) == Some("base64") {
            return format!(
                "data:{};base64,{}",
                src.get("media_type").and_then(Value::as_str).unwrap_or("image/png"),
                src.get("data").and_then(Value::as_str).unwrap_or("")
            );
        }
        return src.get("url").or_else(|| src.get("path")).and_then(Value::as_str).unwrap_or("").to_string();
    }
    match part.get("image_url") {
        Some(Value::Object(o)) => o.get("url").and_then(Value::as_str).unwrap_or("").to_string(),
        Some(Value::String(s)) => s.clone(),
        _ => String::new(),
    }
}

/// Content for the template: a string without images, else text and image items in order.
fn parts_of(content: &Value) -> Value {
    if !has_image(content) {
        return Value::String(text_of(content));
    }
    let mut items = Vec::new();
    for part in content.as_array().into_iter().flatten() {
        if !part.is_object() {
            continue;
        }
        if is_image_part(part) {
            items.push(json!({"type": "image", "source": image_source(part)}));
        } else if is_text_part(part) && part.get("text").is_some() {
            items.push(json!({"type": "text", "text": part.get("text").cloned().unwrap_or(json!(""))}));
        }
    }
    Value::Array(items)
}

/// A refusal part reads as its text; other parts pass through (unknown types are left out by text_of/parts_of).
fn check_parts(content: &Value) -> Value {
    let Some(parts) = content.as_array() else { return content.clone() };
    Value::Array(
        parts
            .iter()
            .map(|p| {
                if p.get("type").and_then(Value::as_str) == Some("refusal") {
                    json!({"type": "text", "text": p.get("refusal").and_then(Value::as_str).unwrap_or("")})
                } else {
                    p.clone()
                }
            })
            .collect(),
    )
}

/// `messages` (or `tool_calls`) as a list of objects; a JSON string is decoded first (#460).
fn object_list(value: Option<&Value>, name: &str) -> Result<Vec<Value>> {
    let decoded;
    let v = match value {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::String(s)) => {
            decoded = serde_json::from_str::<Value>(s)
                .map_err(|_| bad(format!("{name} must be a list of objects (a string was sent that is not JSON)")))?;
            &decoded
        }
        Some(v) => v,
    };
    match v.as_array() {
        Some(a) if a.iter().all(Value::is_object) => Ok(a.clone()),
        _ => Err(bad(format!("{name} must be a list of objects"))),
    }
}

/// A call's arguments from a client's history -> the mapping the template renders (#510).
fn tool_arguments(raw: Option<&Value>) -> Value {
    match raw {
        None | Some(Value::Null) => json!({}),
        Some(Value::String(s)) if s.trim().is_empty() => json!({}),
        Some(Value::Object(_)) => raw.cloned().unwrap(),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(v @ Value::Object(_)) => v,
            _ => json!({"arguments": s}),
        },
        Some(other) => json!({"arguments": py_dumps(other)}),
    }
}

/// json.dumps(x, ensure_ascii=False) for the rare non-object arguments value.
fn py_dumps(v: &Value) -> String {
    let s = serde_json::to_string(v).unwrap_or_default();
    // serde writes "a":1,[1,2]; Python writes ", " and ": "
    let mut out = String::with_capacity(s.len() + 8);
    let mut in_str = false;
    let mut esc = false;
    for c in s.chars() {
        out.push(c);
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
        } else if c == '"' {
            in_str = true;
        } else if c == ',' || c == ':' {
            out.push(' ');
        }
    }
    out
}

const EFFORT: [(&str, Option<&str>); 11] = [
    ("none", None), ("off", None), ("minimal", None), ("disabled", None), ("false", None),
    ("low", Some("low")), ("medium", Some("medium")), ("high", Some("xhigh")), ("xhigh", Some("xhigh")),
    ("max", Some("xhigh")), ("maximum", Some("xhigh")),
];

/// A reasoning effort as given by a client -> the template's kwargs.
fn effort_kwargs(value: Option<&Value>) -> Result<Map<String, Value>> {
    let mut out = Map::new();
    let key = match value {
        None | Some(Value::Null) => return Ok(out),
        Some(Value::String(s)) if s.is_empty() => return Ok(out),
        Some(Value::Bool(false)) => {
            out.insert("enable_thinking".into(), json!(false));
            return Ok(out);
        }
        Some(Value::String(s)) => s.trim().to_lowercase(),
        Some(Value::Bool(true)) => "true".into(),
        Some(other) => other.to_string().trim().to_lowercase(),
    };
    match EFFORT.iter().find(|(k, _)| *k == key) {
        Some((_, None)) => {
            out.insert("enable_thinking".into(), json!(false));
        }
        Some((_, Some(level))) => {
            out.insert("reasoning_effort".into(), json!(level));
        }
        None => {
            let shown = match value {
                Some(Value::String(s)) => format!("'{s}'"),
                Some(v) => v.to_string(),
                None => String::new(),
            };
            return Err(bad(format!("unknown reasoning effort {shown}: use none, low, medium or high")));
        }
    }
    Ok(out)
}

/// The same tool set comes back in the order it was first seen (prefix-cache stability across MCP registration
/// order). Process-wide, at most 64 sets remembered, as in Python.
#[derive(Default)]
pub struct ToolOrders {
    inner: Mutex<VecDeque<(Vec<String>, Vec<String>)>>, // (sorted names as key, first order)
}

impl ToolOrders {
    const MAX: usize = 64;

    fn order(&self, tools: Vec<Value>, name_of: impl Fn(&Value) -> String) -> Vec<Value> {
        if tools.len() < 2 {
            return tools;
        }
        let names: Vec<String> = tools.iter().map(&name_of).collect();
        if names.iter().collect::<HashSet<_>>().len() != names.len() {
            return tools;
        }
        let mut key = names.clone();
        key.sort();
        let mut q = self.inner.lock().expect("tool order lock");
        let found = q.iter().position(|(k, _)| *k == key);
        let order = match found {
            None => {
                q.push_back((key, names));
                while q.len() > Self::MAX {
                    q.pop_front();
                }
                return tools;
            }
            Some(i) => {
                let entry = q.remove(i).expect("present");
                let order = entry.1.clone();
                q.push_back(entry);
                order
            }
        };
        drop(q);
        let pos: HashMap<&String, usize> = order.iter().enumerate().map(|(i, n)| (n, i)).collect();
        let mut tools = tools;
        tools.sort_by_key(|t| pos[&name_of(t)]);
        tools
    }

    pub fn clear(&self) {
        self.inner.lock().expect("tool order lock").clear();
    }
}

fn function_of(t: &Value) -> &Value {
    if t.get("type").and_then(Value::as_str) == Some("function") {
        t.get("function").unwrap_or(t)
    } else {
        t
    }
}

fn tool_list(value: Option<&Value>, orders: &ToolOrders) -> Result<Vec<Value>> {
    if value.is_none_or(|v| !truthy(v)) {
        return Ok(Vec::new());
    }
    let shape = r#"{"type": "function", "function": {"name": ..., "parameters": {...}}}"#;
    let tools = object_list(value, "tools").map_err(|_| bad(format!("tools must be a list of tool objects, each {shape}")))?;
    for (i, t) in tools.iter().enumerate() {
        let f = function_of(t);
        let name = f.get("name").and_then(Value::as_str).filter(|n| !n.is_empty());
        let Some(name) = name.filter(|_| f.is_object()) else {
            return Err(bad(format!("tools[{i}] has no name: each tool must be {shape}")));
        };
        if let Some(schema) = f.get("parameters") {
            if !schema.is_null() && !schema.is_object() {
                return Err(bad(format!(
                    "tools[{i}] ({name}): \"parameters\" must be an object (the JSON schema of its parameters)")));
            }
        }
    }
    Ok(orders.order(tools, |t| function_of(t).get("name").and_then(Value::as_str).unwrap_or("").to_string()))
}

/// Mid-conversation system messages become user messages, in place.
fn late_system_to_user(messages: Vec<Value>) -> Vec<Value> {
    messages
        .into_iter()
        .enumerate()
        .map(|(i, mut m)| {
            if i > 0 && m.get("role").and_then(Value::as_str) == Some("system") {
                m["role"] = json!("user");
            }
            m
        })
        .collect()
}

/// Tool answers put in the order of the calls they answer, when each names a different call of the message before.
fn answers_in_call_order(messages: Vec<Value>) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len());
    let mut i = 0;
    while i < messages.len() {
        let ids: Vec<Option<String>> = messages[i]
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(|calls| calls.iter().map(|c| c.get("id").and_then(Value::as_str).map(str::to_owned)).collect())
            .unwrap_or_default();
        out.push(messages[i].clone());
        i += 1;
        let mut end = i;
        while end < messages.len() && messages[end].get("role").and_then(Value::as_str) == Some("tool") {
            end += 1;
        }
        let mut run: Vec<Value> = messages[i..end].to_vec();
        let named: Vec<Option<String>> =
            run.iter().map(|t| t.get("tool_call_id").and_then(Value::as_str).map(str::to_owned)).collect();
        let all_ids = ids.iter().all(Option::is_some);
        let unique = named.iter().collect::<HashSet<_>>().len() == named.len();
        let subset = named.iter().all(|n| ids.contains(n));
        if run.len() > 1 && all_ids && unique && subset {
            run.sort_by_key(|t| {
                let n = t.get("tool_call_id").and_then(Value::as_str).map(str::to_owned);
                ids.iter().position(|x| *x == n).unwrap_or(usize::MAX)
            });
        }
        out.extend(run);
        i = end;
    }
    out
}

/// A client's `tool_choice` in either API's shape -> ("auto" | "none" | "required" | "named" | "unknown", name).
pub fn tool_choice_of(tool_choice: Option<&Value>) -> (&'static str, Option<String>) {
    match tool_choice {
        None | Some(Value::Null) => ("auto", None),
        Some(Value::String(s)) => match s.as_str() {
            "auto" => ("auto", None),
            "none" => ("none", None),
            "required" => ("required", None),
            _ => ("unknown", None),
        },
        Some(Value::Object(o)) => match o.get("type").and_then(Value::as_str) {
            Some("auto") => ("auto", None),
            Some("none") => ("none", None),
            Some("any" | "required") => ("required", None),
            Some("function" | "tool") => {
                let name = o.get("function").and_then(|f| f.get("name")).or_else(|| o.get("name")).and_then(Value::as_str);
                match name.filter(|n| !n.is_empty()) {
                    Some(n) => ("named", Some(n.to_string())),
                    None => ("unknown", None),
                }
            }
            _ => ("unknown", None),
        },
        Some(_) => ("unknown", None),
    }
}

/// `tool_choice` -> the text that opens the call the reply must make, or None (the model decides). There is no
/// grammar: the server writes this opening itself, so the model can only go on with a call. A choice that cannot be
/// honoured (an unknown shape, a name that is not one of the tools) acts as "auto".
pub fn forced_call(tool_choice: Option<&Value>, tools: Option<&[Value]>) -> Option<String> {
    let (mut kind, mut name) = tool_choice_of(tool_choice);
    let names: Vec<&str> = tools.into_iter().flatten().filter_map(|t| t.get("name").and_then(Value::as_str)).collect();
    if matches!(kind, "auto" | "none") {
        return None;
    }
    if kind == "required" && names.len() == 1 {
        kind = "named";
        name = Some(names[0].to_string());
    }
    if kind == "required" && !names.is_empty() {
        return Some("<tool_call>\n<function=".into());
    }
    match name {
        Some(n) if kind == "named" && names.contains(&n.as_str()) => Some(format!("<tool_call>\n<function={n}>\n")),
        _ => None,
    }
}

const BILLING_HEADER: &str = "x-anthropic-billing-header:";

/// Claude Code's billing stamp (`cch=` changes every request, the version's 4th part every session) pinned to f's,
/// so the prompt start stays the same and the conversation cache keeps it (llama.cpp #21793). Only a header at
/// the very start of the system text, inside its first 160 characters.
fn pin_billing_stamp(system: &str) -> String {
    if !system.starts_with(BILLING_HEADER) {
        return system.to_string();
    }
    let mut s: Vec<char> = system.chars().collect();
    let chars: Vec<char> = s.clone();
    let find = |pat: &str, from: usize| -> Option<usize> {
        let p: Vec<char> = pat.chars().collect();
        (from..chars.len().saturating_sub(p.len() - 1)).find(|&i| chars[i..i + p.len()] == p[..])
    };
    if let Some(cch) = find("cch=", BILLING_HEADER.len()).filter(|c| *c <= 160) {
        let v = cch + 4;
        if let Some(end) = find(";", v) {
            if end > v && end - v <= 16 {
                for c in &mut s[v..end] {
                    *c = 'f';
                }
            }
        }
    }
    if let Some(cv) = find("cc_version=", 0).filter(|c| *c <= 160) {
        let v = cv + "cc_version=".len();
        if let Some(end) = find(";", cv) {
            if v < end && end - v <= 64 {
                let ver: String = chars[v..end].iter().collect();
                let parts: Vec<&str> = ver.split('.').collect();
                if parts.len() > 3 {
                    let tail = v + parts[..3].join(".").chars().count() + 1;
                    for c in &mut s[tail..end] {
                        *c = 'f';
                    }
                }
            }
        }
    }
    s.into_iter().collect()
}

fn budget_effort(tokens: &Value) -> Map<String, Value> {
    let mut out = Map::new();
    let n = tokens.as_i64().or_else(|| tokens.as_f64().map(|f| f as i64)).or_else(|| tokens.as_str().and_then(|s| s.trim().parse().ok()));
    if let Some(n) = n {
        out.insert("reasoning_effort".into(), json!(if n < 2048 { "low" } else if n < 8192 { "medium" } else { "xhigh" }));
    }
    out
}

/// Anthropic Messages -> (template messages, template tools, template kwargs). `think_unasked`: a request without
/// "thinking", an effort or a budget gets the template's default (it thinks). Also returns the thinking budget the
/// request asks for (Anthropic's budget_tokens, or the request's own reasoning_budget_tokens).
pub fn anthropic_to_messages(req: &Value, orders: &ToolOrders, think_unasked: bool) -> Result<(Prepared, Option<u64>)> {
    let mut messages = Vec::new();
    if let Some(system) = req.get("system").filter(|s| truthy(s)) {
        messages.push(json!({"role": "system", "content": pin_billing_stamp(&text_of(&check_parts(system)))}));
    }
    for m in object_list(req.get("messages"), "messages")? {
        let role = m.get("role").cloned().ok_or_else(|| bad("each message needs a role"))?;
        let content = check_parts(m.get("content").unwrap_or(&Value::Null));
        if content.is_string() {
            messages.push(json!({"role": role, "content": content}));
            continue;
        }
        let blocks: Vec<Value> = content.as_array().cloned().unwrap_or_default();
        let has_tool_result = blocks.iter().any(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"));
        if role.as_str() == Some("user") && has_image(&content) && !has_tool_result {
            messages.push(json!({"role": "user", "content": parts_of(&content)}));
            continue;
        }
        let (mut text, mut reasoning, mut calls, mut parts) = (String::new(), String::new(), Vec::new(), Vec::new());
        let (mut any_text, mut any_reasoning) = (false, false);
        for block in &blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    text.push_str(block.get("text").and_then(Value::as_str).unwrap_or(""));
                    any_text = true;
                    parts.push(block.clone());
                }
                Some(k) if IMAGE_PARTS.contains(&k) => parts.push(block.clone()),
                Some("thinking") => {
                    reasoning.push_str(block.get("thinking").and_then(Value::as_str).unwrap_or(""));
                    any_reasoning = true;
                }
                Some("tool_use") => {
                    let mut call = Map::new();
                    call.insert("function".into(), json!({
                        "name": block.get("name").cloned().unwrap_or(Value::Null),
                        "arguments": tool_arguments(block.get("input")),
                    }));
                    if let Some(id) = block.get("id").filter(|v| truthy(v)) {
                        call.insert("id".into(), id.clone());
                    }
                    calls.push(Value::Object(call));
                }
                Some("tool_result") => {
                    let mut t = Map::new();
                    t.insert("role".into(), json!("tool"));
                    t.insert("content".into(), parts_of(&check_parts(block.get("content").unwrap_or(&Value::Null))));
                    if let Some(id) = block.get("tool_use_id").filter(|v| truthy(v)) {
                        t.insert("tool_call_id".into(), id.clone());
                    }
                    messages.push(Value::Object(t));
                }
                _ => {}
            }
        }
        if any_text || !calls.is_empty() || any_reasoning || !parts.is_empty() {
            let parts_v = Value::Array(parts);
            let mut out = Map::new();
            out.insert("role".into(), role.clone());
            out.insert("content".into(), if has_image(&parts_v) { parts_of(&parts_v) } else { json!(text) });
            if !reasoning.is_empty() {
                out.insert("reasoning_content".into(), json!(reasoning));
            }
            if !calls.is_empty() {
                out.insert("tool_calls".into(), Value::Array(calls));
            }
            messages.push(Value::Object(out));
        }
    }
    let tools: Vec<Value> = anthropic_tool_list(req.get("tools"), orders)?
        .iter()
        .map(|t| json!({
            "name": t.get("name").cloned().unwrap_or(Value::Null),
            "description": t.get("description").cloned().unwrap_or(json!("")),
            "parameters": t.get("input_schema").cloned().unwrap_or(json!({})),
        }))
        .collect();
    let tools = if tools.is_empty() { None } else { Some(tools) };

    let thinking = req.get("thinking");
    let effort = req.get("output_config").and_then(Value::as_object).and_then(|o| o.get("effort")).filter(|e| truthy(e));
    let disabled = thinking.and_then(|t| t.get("type")).and_then(Value::as_str) == Some("disabled");
    let bt = if thinking.is_some_and(Value::is_object) && !disabled {
        thinking.and_then(|t| t.get("budget_tokens")).and_then(Value::as_u64).filter(|n| *n > 0)
    } else {
        None
    };
    let own_budget = req.get("reasoning_budget_tokens").filter(|v| !v.is_null()).and_then(Value::as_u64);
    let budget = own_budget.or(bt);
    let mut kwargs = Map::new();
    if disabled {
        kwargs.insert("enable_thinking".into(), json!(false));
    } else if thinking.is_none_or(Value::is_null) && budget.is_none_or(|b| b == 0) && !think_unasked {
        kwargs.insert("enable_thinking".into(), json!(false));
    } else if let Some(e) = effort {
        kwargs.extend(effort_kwargs(Some(e))?);
    } else if let Some(b) = thinking.and_then(|t| t.get("budget_tokens")).filter(|b| truthy(b)) {
        kwargs.extend(budget_effort(b));
    }
    let messages = answers_in_call_order(late_system_to_user(messages));
    Ok((Prepared { messages, tools, kwargs }, budget))
}

/// Anthropic tools: each `{"name": ..., "input_schema": {...}}`.
fn anthropic_tool_list(value: Option<&Value>, orders: &ToolOrders) -> Result<Vec<Value>> {
    if value.is_none_or(|v| !truthy(v)) {
        return Ok(Vec::new());
    }
    let shape = r#"{"name": ..., "input_schema": {...}}"#;
    let tools = object_list(value, "tools").map_err(|_| bad(format!("tools must be a list of tool objects, each {shape}")))?;
    for (i, t) in tools.iter().enumerate() {
        let Some(name) = t.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()) else {
            return Err(bad(format!("tools[{i}] has no name: each tool must be {shape}")));
        };
        if let Some(schema) = t.get("input_schema") {
            if !schema.is_null() && !schema.is_object() {
                return Err(bad(format!(
                    "tools[{i}] ({name}): \"input_schema\" must be an object (the JSON schema of its parameters)")));
            }
        }
    }
    Ok(orders.order(tools, |t| t.get("name").and_then(Value::as_str).unwrap_or("").to_string()))
}

pub fn openai_to_messages(req: &Value, orders: &ToolOrders) -> Result<Prepared> {
    let mut messages = Vec::new();
    for m in object_list(req.get("messages"), "messages")? {
        let mut role = m.get("role").cloned().unwrap_or(Value::Null);
        if role.as_str() == Some("developer") {
            role = json!("system");
        }
        let images = matches!(role.as_str(), Some("user" | "tool" | "assistant"));
        let content = check_parts(m.get("content").unwrap_or(&Value::Null));
        let mut out = Map::new();
        out.insert("role".into(), role.clone());
        out.insert("content".into(), if images { parts_of(&content) } else { Value::String(text_of(&content)) });
        if let Some(r) = m.get("reasoning_content").filter(|r| truthy(r)) {
            out.insert("reasoning_content".into(), r.clone());
        }
        if role.as_str() == Some("tool") {
            if let Some(id) = m.get("tool_call_id").filter(|v| truthy(v)) {
                out.insert("tool_call_id".into(), id.clone());
            }
        }
        if m.get("tool_calls").is_some_and(truthy) {
            let mut calls = Vec::new();
            for c in object_list(m.get("tool_calls"), "tool_calls")? {
                let f = c.get("function").unwrap_or(&c);
                if !f.is_object() {
                    return Err(bad("tool_calls must be a list of objects (each with a \"function\" object)"));
                }
                let mut call = Map::new();
                call.insert("function".into(), json!({
                    "name": f.get("name").cloned().unwrap_or(Value::Null),
                    "arguments": tool_arguments(f.get("arguments")),
                }));
                if let Some(id) = c.get("id").filter(|v| truthy(v)) {
                    call.insert("id".into(), id.clone());
                }
                calls.push(Value::Object(call));
            }
            out.insert("tool_calls".into(), Value::Array(calls));
        }
        messages.push(Value::Object(out));
    }
    let tools: Vec<Value> = tool_list(req.get("tools"), orders)?.iter().map(|t| function_of(t).clone()).collect();
    let tools = if tools.is_empty() { None } else { Some(tools) };

    let reasoning = req.get("reasoning").and_then(Value::as_object);
    let given = req.get("reasoning_effort").filter(|g| !g.is_null() && g.as_str() != Some(""));
    let mut kwargs = effort_kwargs(given.or_else(|| reasoning.and_then(|r| r.get("effort"))))?;
    let ctk = req.get("chat_template_kwargs").and_then(Value::as_object);
    if let Some(ctk) = ctk {
        for (k, v) in ctk {
            if k == "enable_thinking" && !truthy(v) {
                kwargs = Map::new();
                kwargs.insert("enable_thinking".into(), json!(false));
            } else if k == "reasoning_effort" && !kwargs.contains_key("enable_thinking") {
                kwargs.extend(effort_kwargs(Some(v))?);
            }
        }
        if let Some(p @ Value::Bool(_)) = ctk.get("preserve_thinking") {
            kwargs.insert("preserve_thinking".into(), p.clone());
        }
    }
    Ok(Prepared { messages: answers_in_call_order(late_system_to_user(messages)), tools, kwargs })
}
