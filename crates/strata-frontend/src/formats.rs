//! The tool-call formats other than Qwen's `<function=...>`: GLM's `<arg_key>/<arg_value>`, Gemma 4's
//! `call:NAME{...}`, DeepSeek V4's DSML invokes and LFM2's Python-call list. Each is read whole: a call is announced
//! when its name is complete and its arguments follow when it ends. Port of serve/frontend.py.

use crate::output::{param_value, ToolCall};
use serde_json::{json, Map, Value};

/// How the model writes a tool call (serve/server.py picks it by the tokenizer's pre-tokenizer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CallFormat {
    #[default]
    Function,
    Glm,
    Gemma,
    Dsml,
    Lfm2,
}

impl CallFormat {
    /// serve/server.py: "glm" for glm4/glm5, "gemma" for gemma4, "dsml" for joyai-llm/deepseek-v3, "lfm2" for lfm2.
    pub fn from_pre(pre: &str) -> Self {
        match pre {
            // Flint 40B-A2.5B writes GLM's calls (PolyStrata 4f773df4)
            "glm4" | "glm5" | "flint" => CallFormat::Glm,
            "gemma4" => CallFormat::Gemma,
            // #4137: any DeepSeek pre-tokenizer, so a config that names no arch still gets DSML calls
            "joyai-llm" => CallFormat::Dsml,
            p if p.starts_with("deepseek") => CallFormat::Dsml,
            "lfm2" => CallFormat::Lfm2,
            _ => CallFormat::Function,
        }
    }

    /// PolyStrata's names: "function", "glm", "gemma", "dsml", "lfm2".
    pub fn from_name(name: &str) -> Self {
        match name {
            "glm" => CallFormat::Glm,
            "gemma" => CallFormat::Gemma,
            "dsml" => CallFormat::Dsml,
            "lfm2" => CallFormat::Lfm2,
            _ => CallFormat::Function,
        }
    }

    /// The tokens that open a chat turn in this format's template (the last one in a prompt begins the reply).
    pub fn turn_openers(self) -> &'static [&'static str] {
        match self {
            CallFormat::Function | CallFormat::Lfm2 => &["<|im_start|>"],
            CallFormat::Gemma => &["<|turn>"],
            CallFormat::Glm => &["<|system|>", "<|developer|>", "<|user|>", "<|assistant|>", "<|observation|>"],
            CallFormat::Dsml => &["<｜User｜>", "<｜Assistant｜>"],
        }
    }

    /// (the call opener, the call closer) the parser looks for.
    pub fn call_tags(self) -> (&'static str, &'static str) {
        match self {
            CallFormat::Gemma => (GEMMA_CALL, GEMMA_CALL_END),
            CallFormat::Dsml => (DSML_INVOKE, DSML_INVOKE_END),
            CallFormat::Lfm2 => (LFM2_CALL, LFM2_CALL_END),
            _ => ("<tool_call>", "</tool_call>"),
        }
    }

    /// (what opens the reasoning in the reply, what ends it): Gemma opens it itself as a channel; GLM's template opens
    /// <think> in the prompt, and Flint, which writes GLM's calls, writes it itself (PolyStrata frontend.py).
    pub fn think_tags(self) -> (&'static str, &'static str) {
        match self {
            CallFormat::Gemma => (GEMMA_THOUGHT, "<channel|>"),
            CallFormat::Glm => ("<think>", "</think>"),
            _ => ("", "</think>"),
        }
    }

    /// Tags around the calls that are neither a call nor content (DeepSeek V4's block of calls).
    pub fn noise(self) -> &'static [&'static str] {
        match self {
            CallFormat::Dsml => &[DSML_BLOCK_START, DSML_BLOCK_END],
            _ => &[],
        }
    }

    /// The call's name once it is complete, None while it is still arriving.
    pub fn call_name(self, text: &str) -> Option<String> {
        match self {
            CallFormat::Glm => glm_call_name(text),
            CallFormat::Gemma => gemma_call_name(text),
            CallFormat::Dsml => text.find('"').map(|i| text[..i].trim().to_string()),
            CallFormat::Lfm2 => lfm2_call_name(text),
            CallFormat::Function => None,
        }
    }

    /// Where the call's body ends (the closer's index), None while it is not complete.
    pub fn call_end(self, text: &str) -> Option<usize> {
        match self {
            CallFormat::Glm => tagged_end(text, CALL_END, ARG_VALUE_END, ARG_KEY),
            CallFormat::Gemma => gemma_call_end(text),
            CallFormat::Dsml => tagged_end(text, DSML_INVOKE_END, DSML_PARAM_END, DSML_PARAM),
            CallFormat::Lfm2 => text.find(LFM2_CALL_END),
            CallFormat::Function => None,
        }
    }

    /// The calls a whole body holds (LFM2's block can hold several).
    pub fn parse(self, body: &str, schemas: &std::collections::HashMap<String, Value>) -> Vec<ToolCall> {
        let (_, stop) = self.call_tags();
        let schema = self.call_name(&format!("{body}{stop}")).and_then(|n| schemas.get(&n));
        match self {
            CallFormat::Glm => vec![parse_glm(body, schema)],
            CallFormat::Gemma => vec![parse_gemma(body, schema)],
            CallFormat::Dsml => vec![parse_dsml(body, schema)],
            CallFormat::Lfm2 => parse_lfm2(body),
            CallFormat::Function => Vec::new(),
        }
    }
}

const CALL_END: &str = "</tool_call>";
const ARG_KEY: &str = "<arg_key>";
const ARG_KEY_END: &str = "</arg_key>";
const ARG_VALUE: &str = "<arg_value>";
const ARG_VALUE_END: &str = "</arg_value>";
pub const GEMMA_CALL: &str = "<|tool_call>";
pub const GEMMA_CALL_END: &str = "<tool_call|>";
pub const GEMMA_QUOTE: &str = "<|\"|>";
pub const GEMMA_THOUGHT: &str = "<|channel>thought\n";
const DSML_INVOKE: &str = "<｜DSML｜invoke name=\"";
const DSML_INVOKE_END: &str = "</｜DSML｜invoke>";
const DSML_PARAM: &str = "<｜DSML｜parameter";
const DSML_PARAM_END: &str = "</｜DSML｜parameter>";
const DSML_BLOCK_START: &str = "<｜DSML｜tool_calls>";
const DSML_BLOCK_END: &str = "</｜DSML｜tool_calls>";
pub const LFM2_CALL: &str = "<|tool_call_start|>";
pub const LFM2_CALL_END: &str = "<|tool_call_end|>";

fn props(schema: Option<&Value>) -> Map<String, Value> {
    schema
        .and_then(|s| s.get("parameters"))
        .and_then(|p| p.get("properties"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

fn declared<'a>(props: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    props.get(key).and_then(|p| p.get("type")).and_then(Value::as_str)
}

fn py_dumps(v: &Value) -> String {
    crate::output::py_dumps(v, false)
}

/// The first `end` right after the name or after a value's closer (`value_end`), so a value may contain the tag.
fn tagged_end(text: &str, end: &str, value_end: &str, key: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(rel) = text[from..].find(end) {
        let at = from + rel;
        let before = text[..at].trim_end();
        if before.ends_with(value_end) || !before.contains(key) {
            return Some(at);
        }
        from = at + 1;
        while !text.is_char_boundary(from) {
            from += 1;
        }
    }
    None
}

// ------------------------------------------------------------------------------------------------ GLM

fn glm_call_name(text: &str) -> Option<String> {
    let at = [text.find(ARG_KEY), text.find(CALL_END)].into_iter().flatten().min()?;
    Some(text[..at].trim().to_string())
}

/// `NAME<arg_key>K</arg_key><arg_value>V</arg_value>...` -> ToolCall.
fn parse_glm(body: &str, schema: Option<&Value>) -> ToolCall {
    let (name, rest) = body.split_once(ARG_KEY).unwrap_or((body, ""));
    let props = props(schema);
    let mut args = Map::new();
    if !rest.is_empty() {
        for part in rest.split(ARG_KEY) {
            let (key, tail) = part.split_once(ARG_KEY_END).unwrap_or((part, ""));
            let value = tail.split_once(ARG_VALUE).map(|(_, v)| v).unwrap_or("");
            let value = match value.rfind(ARG_VALUE_END) {
                Some(e) => &value[..e],
                None => value,
            };
            let key = key.trim();
            args.insert(key.to_string(), param_value(value, declared(&props, key)));
        }
    }
    ToolCall::new(name.trim().to_string(), args)
}

// ------------------------------------------------------------------------------------------------ Gemma 4

fn gemma_call_name(text: &str) -> Option<String> {
    let at = [text.find('{'), text.find(GEMMA_CALL_END)].into_iter().flatten().min()?;
    let n = text[..at].trim();
    Some(n.strip_prefix("call:").unwrap_or(n).to_string())
}

/// The first `<tool_call|>` that is not inside a `<|"|>` string.
fn gemma_call_end(text: &str) -> Option<usize> {
    let mut pos = 0;
    loop {
        let quote = text[pos..].find(GEMMA_QUOTE).map(|q| q + pos);
        let end = text[pos..].find(GEMMA_CALL_END).map(|e| e + pos);
        if let Some(e) = end {
            if quote.is_none_or(|q| e < q) {
                return Some(e);
            }
        }
        let q = quote?;
        let close = text[q + GEMMA_QUOTE.len()..].find(GEMMA_QUOTE)? + q + GEMMA_QUOTE.len();
        pos = close + GEMMA_QUOTE.len();
    }
}

/// The value of Gemma's call notation that starts at s[i] -> (value, the index after it).
fn gemma_value(s: &str, mut i: usize) -> (Value, usize) {
    let b = s.as_bytes();
    while i < s.len() && (b[i] as char).is_ascii_whitespace() {
        i += 1;
    }
    if s[i..].starts_with(GEMMA_QUOTE) {
        i += GEMMA_QUOTE.len();
        return match s[i..].find(GEMMA_QUOTE) {
            Some(e) => (json!(&s[i..i + e]), i + e + GEMMA_QUOTE.len()),
            None => (json!(&s[i..]), s.len()),
        };
    }
    if i < s.len() && (b[i] == b'{' || b[i] == b'[') {
        let mapping = b[i] == b'{';
        let close = if mapping { b'}' } else { b']' };
        let mut obj = Map::new();
        let mut arr = Vec::new();
        i += 1;
        loop {
            while i < s.len() && ((b[i] as char).is_ascii_whitespace() || b[i] == b',') {
                i += 1;
            }
            if i >= s.len() || b[i] == close {
                let v = if mapping { Value::Object(obj) } else { Value::Array(arr) };
                return (v, i + 1);
            }
            if mapping {
                let Some(colon) = s[i..].find(':').map(|c| c + i) else {
                    return (Value::Object(obj), s.len());
                };
                let k = s[i..colon].trim();
                let k = k.strip_prefix(GEMMA_QUOTE).unwrap_or(k);
                let k = k.strip_suffix(GEMMA_QUOTE).unwrap_or(k).to_string();
                let (v, next) = gemma_value(s, colon + 1);
                obj.insert(k, v);
                i = next;
            } else {
                let (v, next) = gemma_value(s, i);
                arr.push(v);
                i = next;
            }
        }
    }
    let mut end = i;
    while end < s.len() && !matches!(b[end], b',' | b'}' | b']') {
        end += 1;
    }
    let word = s[i..end].trim();
    (serde_json::from_str(word).unwrap_or_else(|_| json!(word)), end)
}

/// `call:NAME{key:value,...}` -> ToolCall. A parameter the schema declares a string stays one.
fn parse_gemma(body: &str, schema: Option<&Value>) -> ToolCall {
    let mut args = match body.find('{') {
        Some(b) => gemma_value(body, b).0.as_object().cloned().unwrap_or_default(),
        None => Map::new(),
    };
    let props = props(schema);
    for (k, v) in args.iter_mut() {
        if declared(&props, k) == Some("string") && !v.is_string() {
            *v = json!(py_dumps(v));
        }
    }
    ToolCall::new(gemma_call_name(&format!("{body}{GEMMA_CALL_END}")).unwrap_or_default(), args)
}

// ------------------------------------------------------------------------------------------------ DeepSeek V4 (DSML)

/// `NAME">` and its `<｜DSML｜parameter name="K" string="true|false">V</｜DSML｜parameter>` lines -> ToolCall.
fn parse_dsml(body: &str, schema: Option<&Value>) -> ToolCall {
    let name = body.find('"').map(|i| body[..i].trim()).unwrap_or(body.trim()).to_string();
    let props = props(schema);
    let mut args = Map::new();
    let mut rest = body;
    while let Some(p) = rest.find(DSML_PARAM) {
        let after = &rest[p + DSML_PARAM.len()..];
        // \s+name="K"(\s+string="true|false")?\s*>(.*?)END
        let trimmed = after.trim_start();
        if trimmed.len() == after.len() || !trimmed.starts_with("name=\"") {
            rest = after;
            continue;
        }
        let t = &trimmed[6..];
        let Some(q) = t.find('"') else { break };
        let key = &t[..q];
        let mut t2 = &t[q + 1..];
        let mut kind = None;
        let ws = t2.trim_start();
        if ws.len() < t2.len() {
            if let Some(x) = ws.strip_prefix("string=\"true\"") {
                kind = Some(true);
                t2 = x;
            } else if let Some(x) = ws.strip_prefix("string=\"false\"") {
                kind = Some(false);
                t2 = x;
            }
        }
        let t3 = t2.trim_start();
        let Some(t3) = t3.strip_prefix('>') else {
            rest = t2;
            continue;
        };
        let Some(e) = t3.find(DSML_PARAM_END) else { break };
        let value = &t3[..e];
        let d = declared(&props, key);
        let v = match kind {
            Some(true) => json!(value),
            Some(false) => serde_json::from_str(value).unwrap_or_else(|_| param_value(value, d)),
            None => param_value(value, d),
        };
        args.insert(key.to_string(), v);
        rest = &t3[e + DSML_PARAM_END.len()..];
    }
    ToolCall::new(name, args)
}

// ------------------------------------------------------------------------------------------------ LFM2

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `[\s,\[]*([\w.\-]+)\s*\(` at i -> (name, the index after the parenthesis).
fn lfm2_name_at(s: &str, i: usize) -> Option<(String, usize)> {
    let rest = &s[i..];
    let t = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ',' || c == '[');
    let n: String = t.chars().take_while(|c| is_word(*c) || *c == '.' || *c == '-').collect();
    if n.is_empty() {
        return None;
    }
    let after = t[n.len()..].trim_start();
    let after = after.strip_prefix('(')?;
    Some((n, s.len() - after.len()))
}

fn lfm2_call_name(text: &str) -> Option<String> {
    lfm2_name_at(text, 0).map(|(n, _)| n)
}

/// `[\s,]*(\w+)\s*=\s*` at i -> (key, the index after it).
fn lfm2_key_at(s: &str, i: usize) -> Option<(String, usize)> {
    let t = s[i..].trim_start_matches(|c: char| c.is_whitespace() || c == ',');
    let k: String = t.chars().take_while(|c| is_word(*c)).collect();
    if k.is_empty() {
        return None;
    }
    let after = t[k.len()..].trim_start().strip_prefix('=')?;
    let after = after.trim_start();
    Some((k, s.len() - after.len()))
}

/// The index after the string, list or mapping that starts at s[i] (the end of s when it does not close).
fn lfm2_close(s: &str, mut i: usize) -> usize {
    let b = s.as_bytes();
    let (mut depth, mut quote) = (0i32, 0u8);
    while i < b.len() {
        let c = b[i];
        if quote != 0 {
            if c == b'\\' {
                i += 1;
            } else if c == quote {
                quote = 0;
                if depth == 0 {
                    return i + 1;
                }
            }
        } else if c == b'\'' || c == b'"' {
            quote = c;
        } else if c == b'[' || c == b'{' {
            depth += 1;
        } else if c == b']' || c == b'}' {
            depth -= 1;
            if depth == 0 {
                return i + 1;
            }
        }
        i += 1;
    }
    s.len()
}

/// A small `ast.literal_eval`: strings (either quote, the usual escapes), numbers, True/False/None, lists, tuples
/// and dicts. None when the text is not such a literal.
fn py_literal(s: &str) -> Option<Value> {
    let mut p = PyLit { s: s.as_bytes(), i: 0, src: s };
    let v = p.value()?;
    p.ws();
    (p.i == p.s.len()).then_some(v)
}

struct PyLit<'a> {
    s: &'a [u8],
    i: usize,
    src: &'a str,
}

impl PyLit<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && (self.s[self.i] as char).is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn value(&mut self) -> Option<Value> {
        self.ws();
        let c = *self.s.get(self.i)?;
        match c {
            b'\'' | b'"' => self.string(),
            b'[' | b'(' => {
                let close = if c == b'[' { b']' } else { b')' };
                self.i += 1;
                let mut out = Vec::new();
                loop {
                    self.ws();
                    if self.s.get(self.i) == Some(&close) {
                        self.i += 1;
                        return Some(Value::Array(out));
                    }
                    out.push(self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(x) if *x == close => {}
                        _ => return None,
                    }
                }
            }
            b'{' => {
                self.i += 1;
                let mut out = Map::new();
                loop {
                    self.ws();
                    if self.s.get(self.i) == Some(&b'}') {
                        self.i += 1;
                        return Some(Value::Object(out));
                    }
                    let k = self.value()?;
                    self.ws();
                    if self.s.get(self.i) != Some(&b':') {
                        return None;
                    }
                    self.i += 1;
                    let v = self.value()?;
                    let key = match k {
                        Value::String(s) => s,
                        other => other.to_string(),
                    };
                    out.insert(key, v);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {}
                        _ => return None,
                    }
                }
            }
            _ => {
                let start = self.i;
                while self.i < self.s.len() && !matches!(self.s[self.i], b',' | b']' | b')' | b'}' | b':') && !(self.s[self.i] as char).is_ascii_whitespace() {
                    self.i += 1;
                }
                match &self.src[start..self.i] {
                    "True" => Some(json!(true)),
                    "False" => Some(json!(false)),
                    "None" => Some(Value::Null),
                    w => w.parse::<i64>().map(|n| json!(n)).ok().or_else(|| w.parse::<f64>().ok().map(|f| json!(f))),
                }
            }
        }
    }
    fn string(&mut self) -> Option<Value> {
        let q = self.s[self.i];
        self.i += 1;
        let mut out = String::new();
        let start_chars = self.src;
        loop {
            let c = *self.s.get(self.i)?;
            if c == q {
                self.i += 1;
                return Some(json!(out));
            }
            if c == b'\\' {
                let e = *self.s.get(self.i + 1)?;
                self.i += 2;
                match e {
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    b'0' => out.push('\0'),
                    b'\\' | b'\'' | b'"' => out.push(e as char),
                    b'u' => {
                        let hex = start_chars.get(self.i..self.i + 4)?;
                        out.push(char::from_u32(u32::from_str_radix(hex, 16).ok()?)?);
                        self.i += 4;
                    }
                    _ => {
                        out.push('\\');
                        out.push(e as char);
                    }
                }
                continue;
            }
            let ch = start_chars[self.i..].chars().next()?;
            out.push(ch);
            self.i += ch.len_utf8();
        }
    }
}

/// The argument value at s[i] -> (value, the index after it).
fn lfm2_value(s: &str, i: usize) -> (Value, usize) {
    let b = s.as_bytes();
    if i < b.len() && matches!(b[i], b'\'' | b'"' | b'[' | b'{') {
        let end = lfm2_close(s, i);
        let text = &s[i..end];
        if matches!(b[i], b'[' | b'{') {
            if let Ok(v) = serde_json::from_str(text) {
                return (v, end);
            }
        }
        return (py_literal(text).unwrap_or_else(|| json!(text)), end);
    }
    let mut end = i;
    while end < b.len() && b[end] != b',' && b[end] != b')' {
        end += 1;
    }
    let word = s[i..end].trim();
    let v = match word {
        "true" | "True" => json!(true),
        "false" | "False" => json!(false),
        "null" | "None" => Value::Null,
        w => serde_json::from_str(w).unwrap_or_else(|_| json!(w)),
    };
    (v, end)
}

/// `[f(a='x', b=1), g()]` -> one ToolCall a call (the values carry their types).
fn parse_lfm2(body: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut i = 0;
    while let Some((name, after)) = lfm2_name_at(body, i) {
        i = after;
        let mut args = Map::new();
        while let Some((k, at)) = lfm2_key_at(body, i) {
            let (v, next) = lfm2_value(body, at);
            args.insert(k, v);
            i = next;
        }
        let rest = &body[i..];
        let t = rest.trim_start();
        if let Some(x) = t.strip_prefix(')') {
            i = body.len() - x.len();
        }
        calls.push(ToolCall::new(name, args));
    }
    calls
}

/// Gemma's template opens no turn for the model after an answer that came with tool calls and their results: the
/// prompt is ended as the template ends every other prompt.
pub fn gemma_model_turn(prompt: &str, thinking: bool) -> String {
    let mut p = prompt.to_string();
    if p.ends_with(&format!("<turn|>\n{GEMMA_THOUGHT}")) {
        p.truncate(p.len() - GEMMA_THOUGHT.len());
    }
    if p.ends_with("<turn|>\n") {
        p.push_str("<|turn>model\n");
        if !thinking {
            p.push_str(GEMMA_THOUGHT);
            p.push_str("<channel|>");
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_literals() {
        assert_eq!(py_literal("'x\\'y'"), Some(json!("x'y")));
        assert_eq!(py_literal("[1, 'a', None, (True,)]"), Some(json!([1, "a", null, [true]])));
        assert_eq!(py_literal("{'k': 2.5}"), Some(json!({"k": 2.5})));
        assert_eq!(py_literal("'unterminated"), None);
    }

    #[test]
    fn gemma_turn_is_closed() {
        assert_eq!(gemma_model_turn("a<turn|>\n", true), "a<turn|>\n<|turn>model\n");
        assert_eq!(gemma_model_turn("a<turn|>\n<|channel>thought\n", false), "a<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
    }
}

#[cfg(test)]
mod flint_tests {
    use super::*;
    use crate::output::{Event, OutputParser};

    const TEXT: &str = "<think>the user wants the capital</think>The capital of France is Paris.";

    fn parts(thinking: bool, size: usize) -> (String, String) {
        // Flint's template ends the prompt on <|assistant|>: the reply opens <think> itself (not `opened`)
        let mut p = OutputParser::with_format(thinking, None, false, CallFormat::from_pre("flint"), false);
        let chars: Vec<char> = TEXT.chars().collect();
        let mut evs = Vec::new();
        for c in chars.chunks(size) {
            evs.extend(p.feed(&c.iter().collect::<String>()).unwrap());
        }
        evs.extend(p.finish(None));
        let (mut r, mut c) = (String::new(), String::new());
        for e in evs {
            match e {
                Event::Reasoning(t) => r.push_str(&t),
                Event::Content(t) => c.push_str(&t),
                _ => {}
            }
        }
        (r, c)
    }

    /// PolyStrata serve/test_flint.py Reasoning
    #[test]
    fn flint_opens_its_own_thinking() {
        assert_eq!(CallFormat::from_pre("flint"), CallFormat::Glm);
        for size in [1, 3, TEXT.len()] {
            assert_eq!(parts(true, size), ("the user wants the capital".into(), "The capital of France is Paris.".into()), "size {size}");
        }
        // its template has no switch to turn thinking off, so it thinks whatever the request says
        assert_eq!(parts(false, 4), ("the user wants the capital".into(), "The capital of France is Paris.".into()));
    }

    /// PolyStrata serve/test_flint.py ServiceForFlint: llama.cpp ends a GLM-format reply at <|user|>, <|observation|>
    /// and <|developer|>, which also open turns
    #[test]
    fn glm_turn_openers_include_the_developer_turn() {
        let t = CallFormat::Glm.turn_openers();
        for o in ["<|system|>", "<|developer|>", "<|user|>", "<|assistant|>", "<|observation|>"] {
            assert!(t.contains(&o), "{o}");
        }
    }
}
