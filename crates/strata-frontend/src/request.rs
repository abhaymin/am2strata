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
