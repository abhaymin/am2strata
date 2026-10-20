//! Literal special-token strings inside a message are text, not markers (PolyStrata #537, #554, #2026).
//!
//! Before rendering, each literal (`<think>`, `</think>`, the vision markers, the tool-call tags, and every CONTROL
//! token's text such as `<|im_end|>`) inside a message is swapped for a private-use character; after rendering the
//! marks are turned back into the text and their byte spans are encoded as ordinary text. The template's own markers
//! are untouched. Port of serve/frontend.py `literal_tags`, `mark_think_literals`, `unmark_think_literals`.

use crate::request::{has_image, text_of};
use serde_json::{Map, Value};

const THINK_TAGS: [(&str, char); 2] = [("<think>", '\u{F0E01}'), ("</think>", '\u{F0E02}')];
const VISION_TAGS: [(&str, char); 4] = [
    ("<|vision_start|>", '\u{F0E03}'), ("<|image_pad|>", '\u{F0E04}'), ("<|vision_end|>", '\u{F0E05}'),
    ("<|video_pad|>", '\u{F0E06}'),
];
const TOOL_LITERAL_TAGS: [(&str, char); 4] = [
    ("</tool_call>", '\u{F0E07}'), ("<tool_call>", '\u{F0E08}'), ("<tool_response>", '\u{F0E09}'),
    ("</tool_response>", '\u{F0E0A}'),
];
const CONTROL_MARK0: u32 = 0xF0E10;

/// The tags and their marks, longest tag first (stable: equal lengths keep the base order, then the controls').
#[derive(Debug, Clone)]
pub struct Literals {
    pub tags: Vec<(String, char)>,
}

impl Literals {
    pub fn new(controls: &[&str]) -> Self {
        let mut tags: Vec<(String, char)> =
            THINK_TAGS.iter().chain(&VISION_TAGS).chain(&TOOL_LITERAL_TAGS).map(|(t, c)| (t.to_string(), *c)).collect();
        let base: Vec<String> = tags.iter().map(|(t, _)| t.clone()).collect();
        for (k, c) in controls.iter().filter(|c| !base.iter().any(|b| b == *c)).enumerate() {
            tags.push((c.to_string(), char::from_u32(CONTROL_MARK0 + k as u32).expect("private-use mark")));
        }
        tags.sort_by_key(|(t, _)| std::cmp::Reverse(t.chars().count()));
        Self { tags }
    }

    /// What an assistant message keeps as written: everything but the tool-call tags.
    fn own(&self) -> Vec<(String, char)> {
        self.tags.iter().filter(|(t, _)| !TOOL_LITERAL_TAGS.iter().any(|(x, _)| x == t)).cloned().collect()
    }
}

fn mark(text: &str, tags: &[(String, char)]) -> String {
    let mut s = text.to_string();
    for (tag, m) in tags {
        if s.contains(tag.as_str()) {
            s = s.replace(tag.as_str(), &m.to_string());
        }
    }
    s
}

fn mark_deep(v: &Value, tags: &[(String, char)]) -> Value {
    match v {
        Value::String(s) => Value::String(mark(s, tags)),
        Value::Object(o) => Value::Object(o.iter().map(|(k, x)| (k.clone(), mark_deep(x, tags))).collect()),
        Value::Array(a) => Value::Array(a.iter().map(|x| mark_deep(x, tags)).collect()),
        other => other.clone(),
    }
}

fn has_tag(v: &Value, tags: &[(String, char)]) -> bool {
    match v {
        Value::String(s) => tags.iter().any(|(t, _)| s.contains(t.as_str())),
        Value::Object(o) => o.values().any(|x| has_tag(x, tags)),
        Value::Array(a) => a.iter().any(|x| has_tag(x, tags)),
        _ => false,
    }
}

const OPEN: &str = "<tool_response>";
const CLOSE: &str = "</tool_response>";

/// A user message that is wholly `<tool_response>...</tool_response>` keeps that outer pair as markers.
fn keep_tool_result_wrap(content: &Value, tags: &[(String, char)]) -> Option<Value> {
    let wrap = |text: &str, first: bool, last: bool| -> String {
        let i = if first { text.find(OPEN).unwrap() + OPEN.len() } else { 0 };
        let j = if last { text.rfind(CLOSE).unwrap() } else { text.len() };
        let head = if first { format!("{}{OPEN}", mark(&text[..i - OPEN.len()], tags)) } else { String::new() };
        let tail = if last { format!("{CLOSE}{}", mark(&text[j + CLOSE.len()..], tags)) } else { String::new() };
        format!("{head}{}{tail}", mark(&text[i..j], tags))
    };
    let encloses = |body: &str| body.len() >= OPEN.len() + CLOSE.len() && body.starts_with(OPEN) && body.ends_with(CLOSE);
    match content {
        Value::String(s) => encloses(s.trim()).then(|| Value::String(wrap(s, true, true))),
        Value::Array(parts) if !parts.is_empty() && parts.iter().all(|x| x.get("text").is_some()) => {
            let texts: Option<Vec<&str>> = parts.iter().map(|x| x["text"].as_str()).collect();
            let texts = texts?;
            let body: String = texts.concat();
            if !encloses(body.trim()) {
                return None;
            }
            if !(texts[0].trim_start().starts_with(OPEN) && texts[texts.len() - 1].trim_end().ends_with(CLOSE)) {
                return None;
            }
            let with_text = |p: &Value, t: String| {
                let mut p = p.clone();
                p["text"] = Value::String(t);
                p
            };
            if parts.len() == 1 {
                return Some(Value::Array(vec![with_text(&parts[0], wrap(texts[0], true, true))]));
            }
            let mut out = vec![with_text(&parts[0], wrap(texts[0], true, false))];
            out.extend(parts[1..parts.len() - 1].iter().map(|x| mark_deep(x, tags)));
            out.push(with_text(&parts[parts.len() - 1], wrap(texts[texts.len() - 1], false, true)));
            Some(Value::Array(out))
        }
        _ => None,
    }
}

/// (messages, tools) with literal tags in their text swapped for marks, and whether anything changed.
pub fn mark_think_literals(messages: &[Value], tools: Option<&[Value]>, lit: &Literals)
    -> (Vec<Value>, Option<Vec<Value>>, bool) {
    let tags = &lit.tags;
    let tools_tagged = tools.is_some_and(|t| t.iter().any(|x| has_tag(x, tags)));
    if !messages.iter().any(|m| has_tag(m, tags)) && !tools_tagged {
        return (messages.to_vec(), tools.map(<[Value]>::to_vec), false);
    }
    let own = lit.own();
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let role = m.get("role").and_then(Value::as_str);
        let content = m.get("content").cloned().unwrap_or(Value::Null);
        let mine: &[(String, char)] = if role == Some("assistant") { &own } else { tags };
        let mut mm = Map::new();
        for (k, v) in m.as_object().into_iter().flatten() {
            mm.insert(k.clone(), if k == "role" { v.clone() } else { mark_deep(v, mine) });
        }
        if role == Some("user") {
            if let Some(kept) = keep_tool_result_wrap(&content, tags) {
                mm.insert("content".into(), kept);
            }
        }
        if role == Some("assistant") {
            if let Value::String(c) = &content {
                if c.trim_start().starts_with("<think>") && c.contains("</think>") {
                    let i = c.find("<think>").unwrap() + "<think>".len();
                    let j = c.find("</think>").unwrap();
                    if j >= i {
                        let v = format!("{}{}</think>{}", &c[..i], mark(&c[i..j], &own), mark(&c[j + "</think>".len()..], &own));
                        mm.insert("content".into(), Value::String(v));
                    }
                }
            }
        }
        out.push(Value::Object(mm));
    }
    let tools = tools.map(|t| t.iter().map(|x| mark_deep(x, tags)).collect());
    (out, tools, true)
}

/// The rendered prompt with marks turned back into tags, and the tags' (start, end) byte spans in it.
pub fn unmark_think_literals(prompt: &str, lit: &Literals) -> (String, Vec<(usize, usize)>) {
    let mut out = String::with_capacity(prompt.len());
    let mut spans = Vec::new();
    for c in prompt.chars() {
        match lit.tags.iter().find(|(_, m)| *m == c) {
            Some((tag, _)) => {
                let start = out.len();
                out.push_str(tag);
                spans.push((start, out.len()));
            }
            None => out.push(c),
        }
    }
    (out, spans)
}

// re-exported for the prompt builder's empty-turn rule
#[allow(dead_code)]
pub(crate) fn is_text_message(content: &Value) -> bool {
    !text_of(content).trim().is_empty() || has_image(content)
}
