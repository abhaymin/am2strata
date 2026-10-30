//! Incremental parser of the model's output: reasoning (`...</think>`), content, and tool calls in the template's
//! XML form (`<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n</function>\n</tool_call>`).
//! A tag split across deltas is held back until it is complete, so clients never see `<tool_` or `</thi`.
//!
//! Port of Strata / PolyStrata serve/frontend.py `OutputParser` for call_format "function", including streamed tool
//! arguments, quoted `</think>` (#537), calls written inside the reasoning (#804, #1058), openers inside code (#1058),
//! and from Strata 0.2.8: values that quote a call (#4078), a left-out `</function>` or `</tool_call>` (#4427), damaged
//! JSON arguments (#4427), a stray second `</think>` (#4077) and the recoveries (a bare `<function=`, `<parameter=NAME>`
//! as the opener, several calls in one wrapper). The other call formats are in formats.rs.

use crate::formats::CallFormat;
use serde_json::{Map, Value};
use std::collections::HashMap;

const THINK_END: &str = "</think>";
const CALL_START: &str = "<tool_call>";
const CALL_END: &str = "</tool_call>";
const PARAM_START: &str = "<parameter=";
const PARAM_END: &str = "</parameter>";
const FUNC_START: &str = "<function=";
const FUNC_END: &str = "</function>";
/// the most of a `<tool_call>` inside the reasoning that is held while waiting for its end (#804)
const RCALL_MAX: usize = 32768;

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Map<String, Value>,
}

impl ToolCall {
    pub(crate) fn new(name: String, arguments: Map<String, Value>) -> Self {
        let hex: String = uuid_hex();
        Self { id: format!("call_{}", &hex[..24]), name, arguments }
    }
}

fn uuid_hex() -> String {
    // 32 hex characters, as uuid4().hex; std-only so the crate needs no rng dependency
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let mut s = String::with_capacity(32);
    for _ in 0..2 {
        let mut h = RandomState::new().build_hasher();
        h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
        s.push_str(&format!("{:016x}", h.finish()));
    }
    s
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Reasoning(String),
    Content(String),
    /// a streamed call's name is known (stream_tools)
    ToolStart { id: String, name: String },
    /// a piece of a streamed call's JSON arguments (stream_tools)
    ToolArgs { id: String, text: String },
    /// a complete call
    ToolCall(ToolCall),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Reasoning,
    RCall,
    Content,
    Call,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Scan {
    Name,
    Between,
    Str,
    Raw,
    Done,
}

/// A request the parser cannot read (a malformed call the model wrote): the server ends the request with it.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct ParseError(pub String);

fn find_from(s: &str, pat: &str, from: usize) -> Option<usize> {
    s.get(from..).and_then(|t| t.find(pat)).map(|i| i + from)
}

/// How many calls `text` quotes: `<function=NAME>` followed (after whitespace) by a `<parameter=` or `</function>`.
fn quoted_opens(text: &str) -> usize {
    let mut n = 0;
    let mut at = 0;
    while let Some(p) = find_from(text, FUNC_START, at) {
        at = p + FUNC_START.len();
        let rest = &text[at..];
        let Some(gt) = rest.find(|c: char| c == '<' || c == '>' || c.is_whitespace()) else { break };
        if gt > 0 && rest[gt..].starts_with('>') {
            let after = rest[gt + 1..].trim_start();
            if after.starts_with(PARAM_START) || after.starts_with(FUNC_END) {
                n += 1;
            }
        }
    }
    n
}

/// Where a parameter value in `text` ends: the first `</parameter>` followed (after whitespace) by the next
/// `<parameter=` or `</function>`, or by `</tool_call>` when the call's `</function>` was left out (#4427). A close inside
/// a call the value itself quotes, or one followed by the same parameter `name` again, is part of the value (#4078).
/// -1: none yet; -2: a candidate whose follower has not arrived (`fin` accepts it).
fn param_end(text: &str, fin: bool, name: Option<&str>) -> isize {
    let mut at = text.find(PARAM_END);
    while let Some(a) = at {
        if quoted_opens(&text[..a]) <= text[..a].matches(FUNC_END).count() {
            let after = text[a + PARAM_END.len()..].trim_start();
            if after.starts_with(PARAM_START) {
                let Some(gt) = after.find('>') else { return if fin { a as isize } else { -2 } };
                if name.is_none_or(|n| after[PARAM_START.len()..gt].trim() != n) {
                    return a as isize;
                }
            } else if after.starts_with(FUNC_END) || after.starts_with(CALL_END) {
                return a as isize;
            } else if after.is_empty() || PARAM_START.starts_with(after) || FUNC_END.starts_with(after) || CALL_END.starts_with(after) {
                return if fin { a as isize } else { -2 };
            }
        }
        at = find_from(text, PARAM_END, a + 1);
    }
    -1
}

/// The parameter name between `<parameter=` at `pos` and the `>` at `gt`.
fn param_name(text: &str, pos: usize, gt: usize) -> &str {
    text[pos..gt].split_once('=').map_or("", |x| x.1).trim()
}

/// Where a `<function=...>` body ends: just after its own `</function>`. -1: not complete, or not such a body.
fn function_end(text: &str) -> isize {
    let s = text.trim_start();
    let mut pos = text.len() - s.len();
    if !s.starts_with(FUNC_START) {
        return -1;
    }
    let Some(gt) = find_from(text, ">", pos) else { return -1 };
    pos = gt + 1;
    loop {
        let rest = &text[pos..];
        let s = rest.trim_start();
        pos += rest.len() - s.len();
        if s.starts_with(PARAM_START) {
            let Some(gt) = find_from(text, ">", pos) else { return -1 };
            let end = param_end(&text[gt + 1..], false, Some(param_name(text, pos, gt)));
            if end < 0 {
                return -1;
            }
            pos = gt + 1 + end as usize + PARAM_END.len();
        } else if s.starts_with(FUNC_END) {
            return (pos + FUNC_END.len()) as isize;
        } else {
            return -1;
        }
    }
}

/// `<tool_call>` + `<function=NAME>` at the start of `text`: a following call, which confirms an omitted closer.
fn next_wrapped_call(text: &str) -> bool {
    let Some(r) = text.strip_prefix(CALL_START) else { return false };
    let r = r.trim_start();
    let Some(r) = r.strip_prefix(FUNC_START) else { return false };
    let r = r.trim_start();
    let n = r.find(|c: char| c == '<' || c == '>' || c.is_whitespace()).unwrap_or(r.len());
    n > 0 && r[n..].trim_start().starts_with('>')
}

/// Where a tool call's body ends: the `</tool_call>` after the call's own `</function>`. -1: not complete yet.
fn call_end(text: &str) -> isize {
    let s = text.trim_start();
    let mut pos = text.len() - s.len();
    if !s.starts_with(FUNC_START) {
        return if !FUNC_START.starts_with(s) { text.find(CALL_END).map_or(-1, |i| i as isize) } else { -1 };
    }
    let Some(gt) = find_from(text, ">", pos) else { return -1 };
    pos = gt + 1;
    loop {
        let rest = &text[pos..];
        let s = rest.trim_start();
        pos += rest.len() - s.len();
        if s.starts_with(PARAM_START) {
            let Some(gt) = find_from(text, ">", pos) else { return -1 };
            let end = param_end(&text[gt + 1..], false, Some(param_name(text, pos, gt)));
            if end < 0 {
                return -1;
            }
            pos = gt + 1 + end as usize + PARAM_END.len();
        } else if s.starts_with(FUNC_END) {
            let rest = &text[pos + FUNC_END.len()..];
            let s = rest.trim_start();
            if s.starts_with(CALL_END) {
                return (pos + FUNC_END.len() + rest.len() - s.len()) as isize;
            }
            return if CALL_END.starts_with(s) { -1 } else { find_from(text, CALL_END, pos).map_or(-1, |i| i as isize) };
        } else if s.is_empty() || PARAM_START.starts_with(s) || FUNC_END.starts_with(s) {
            return -1;
        } else {
            return find_from(text, CALL_END, pos).map_or(-1, |i| i as isize);
        }
    }
}

/// A parameter's text as the type its schema declares.
pub(crate) fn param_value(value: &str, declared: Option<&str>) -> Value {
    coerce_value(value, declared, &mut Vec::new())
}

/// The longest value `lenient_json` tries.
const LENIENT_MAX: usize = 1 << 20;
/// A declared type list that holds "string" (a value that is not JSON is that string, never repaired).
const TYPE_LIST_WITH_STRING: &str = "[string]";

/// The JSON array or object `text` was meant to be, read past a model's slips (#4427): a trailing or doubled comma, single
/// quotes, Python's True / False / None, a bare string between two members of an object, brackets left open at the end,
/// extra closers at the end. It never guesses at a value. Err for anything else.
pub fn lenient_json(text: &str) -> Result<Value, ()> {
    let mut t = text.trim();
    if let Some(rest) = strip_json_fence(t) {
        t = rest.trim_end();
        t = t.strip_suffix("```").map_or(t, str::trim_end);
    }
    if !(t.starts_with('[') || t.starts_with('{')) || t.len() > LENIENT_MAX {
        return Err(());
    }
    let mut r = Lenient { s: t.chars().collect(), i: 0 };
    let v = r.value(0)?;
    let tail: String = r.s[r.i.min(r.s.len())..].iter().collect();
    if tail.trim_matches(|c| matches!(c, ']' | '}' | ')' | ' ' | '\t' | '\r' | '\n')).is_empty() { Ok(v) } else { Err(()) }
}

/// What follows "```" + an optional `json` + a line break (and white space) at the start of `t`.
fn strip_json_fence(t: &str) -> Option<&str> {
    let r = t.strip_prefix("```")?.trim_start_matches([' ', '\t']);
    let r = if r.len() >= 4 && r.is_char_boundary(4) && r[..4].eq_ignore_ascii_case("json") { &r[4..] } else { r };
    let r = r.trim_start_matches([' ', '\t']);
    let r = r.strip_prefix('\r').unwrap_or(r);
    r.strip_prefix('\n').map(str::trim_start)
}

struct Lenient {
    s: Vec<char>,
    i: usize,
}

impl Lenient {
    fn peek(&mut self) -> Option<char> {
        while self.s.get(self.i).is_some_and(|c| matches!(c, ' ' | '\t' | '\r' | '\n')) {
            self.i += 1;
        }
        self.s.get(self.i).copied()
    }

    fn string(&mut self) -> Result<String, ()> {
        let q = self.s[self.i];
        self.i += 1;
        let mut out = String::new();
        while let Some(&c) = self.s.get(self.i) {
            if c == q {
                self.i += 1;
                return Ok(out);
            }
            if c == '\\' {
                self.i += 1;
                let e = *self.s.get(self.i).ok_or(())?;
                if e == 'u' {
                    let hex: String = self.s.get(self.i + 1..self.i + 5).ok_or(())?.iter().collect();
                    let cp = u32::from_str_radix(&hex, 16).map_err(|_| ())?;
                    out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                    self.i += 5;
                    continue;
                }
                out.push(match e {
                    '"' | '\'' | '\\' | '/' => e,
                    'b' => '\u{08}',
                    'f' => '\u{0C}',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    _ => return Err(()),
                });
            } else {
                out.push(c);
            }
            self.i += 1;
        }
        Err(())
    }

    fn value(&mut self, depth: usize) -> Result<Value, ()> {
        if depth > 64 {
            return Err(());
        }
        match self.peek() {
            Some('{') => return self.obj(depth + 1),
            Some('[') => return self.arr(depth + 1),
            Some('"' | '\'') => return self.string().map(Value::String),
            _ => {}
        }
        let start = self.i;
        while self.s.get(self.i).is_some_and(|c| !(c.is_whitespace() || matches!(c, ',' | ']' | '}' | ':'))) {
            self.i += 1;
        }
        if self.i == start {
            return Err(());
        }
        let word: String = self.s[start..self.i].iter().collect();
        match word.as_str() {
            "true" | "True" => Ok(Value::Bool(true)),
            "false" | "False" => Ok(Value::Bool(false)),
            "null" | "None" => Ok(Value::Null),
            w => serde_json::from_str(w).map_err(|_| ()), // a number, or an error
        }
    }

    fn obj(&mut self, depth: usize) -> Result<Value, ()> {
        self.i += 1;
        let mut out = Map::new();
        loop {
            match self.peek() {
                None | Some('}') => {
                    self.i += 1; // (no closer: the brackets were left open at the end)
                    return Ok(Value::Object(out));
                }
                Some(',') => self.i += 1,
                Some('"' | '\'') => {
                    let key = self.string()?;
                    match self.peek() {
                        Some(':') => {
                            self.i += 1;
                            let v = self.value(depth)?;
                            out.insert(key, v);
                        }
                        None | Some(',' | '}') => {} // a bare string where a member belongs is dropped
                        _ => return Err(()),
                    }
                }
                _ => return Err(()),
            }
        }
    }

    fn arr(&mut self, depth: usize) -> Result<Value, ()> {
        self.i += 1;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                None | Some(']') => {
                    self.i += 1;
                    return Ok(Value::Array(out));
                }
                Some(',') => self.i += 1,
                _ => {
                    out.push(self.value(depth)?);
                    if !matches!(self.peek(), None | Some(',' | ']')) {
                        return Err(());
                    }
                }
            }
        }
    }
}

/// A non-string parameter's text -> its value by the declared type: JSON when it parses; else an array or object read
/// with `lenient_json` (with a note in `notes`); else the text itself. An array or object that arrived as a JSON string
/// of itself is decoded once more.
pub(crate) fn coerce_value(value: &str, declared: Option<&str>, notes: &mut Vec<String>) -> Value {
    if declared == Some("string") {
        return Value::String(value.into());
    }
    match serde_json::from_str::<Value>(value) {
        Ok(Value::String(v)) if matches!(declared, Some("array" | "object")) && v.trim_start().starts_with(['[', '{']) => {
            match serde_json::from_str::<Value>(&v) {
                Ok(inner) => inner,
                Err(_) => coerce_value(&v, declared, notes),
            }
        }
        Ok(v) => v,
        Err(_) => {
            if declared == Some("boolean") && matches!(value.trim().to_lowercase().as_str(), "true" | "false") {
                return Value::Bool(value.trim().eq_ignore_ascii_case("true"));
            }
            if !value.trim_start().starts_with(['[', '{', '`']) || declared == Some(TYPE_LIST_WITH_STRING) {
                return Value::String(value.into());
            }
            match lenient_json(value) {
                Ok(v) => {
                    notes.push(format!("a damaged JSON {} argument was read as written", if v.is_array() { "array" } else { "object" }));
                    v
                }
                Err(()) => Value::String(value.into()),
            }
        }
    }
}

fn declared_types(schema: Option<&Value>) -> HashMap<String, Option<String>> {
    schema
        .and_then(|s| s.get("parameters"))
        .and_then(|p| p.get("properties"))
        .and_then(Value::as_object)
        .map(|props| {
            props.iter().map(|(k, v)| {
                let t = v.get("type");
                let with_string = t.and_then(Value::as_array).is_some_and(|l| l.iter().any(|x| x == "string"));
                (k.clone(), if with_string { Some(TYPE_LIST_WITH_STRING.to_string()) } else { t.and_then(Value::as_str).map(str::to_owned) })
            }).collect()
        })
        .unwrap_or_default()
}

fn strip_one_newline(mut v: &str) -> &str {
    if let Some(s) = v.strip_prefix('\n') {
        v = s;
    }
    v.strip_suffix('\n').unwrap_or(v)
}

/// recover: `{"name": N, "arguments": {...}}` (or "parameters", or the arguments as a JSON string), possibly in a
/// ```json fence, for a declared tool N -> ToolCall; anything else -> None (the text stays text).
fn json_call(inner: &str, schemas: &HashMap<String, Value>) -> Option<ToolCall> {
    let mut t = inner.trim();
    if let Some(rest) = t.strip_prefix("```") {
        t = rest.trim_start_matches(|c: char| c.is_ascii_alphabetic()).trim();
        t = t.strip_suffix("```").unwrap_or(t).trim();
    }
    let v: Value = serde_json::from_str(t).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    if !schemas.contains_key(&name) {
        return None;
    }
    let args = match v.get("arguments").or_else(|| v.get("parameters")) {
        Some(Value::Object(m)) => m.clone(),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s).ok()? { Value::Object(m) => m, _ => return None },
        None => Map::new(),
        _ => return None,
    };
    Some(ToolCall::new(name, args))
}

/// `<function=NAME>\n<parameter=P>\nVALUE\n</parameter>...</function>` -> ToolCall.
pub fn parse_tool_call(body: &str, schema: Option<&Value>) -> Result<ToolCall, ParseError> {
    parse_tool_call_noted(body, schema, &mut Vec::new())
}

/// `parse_tool_call`, with what was repaired in the arguments added to `notes`.
pub fn parse_tool_call_noted(body: &str, schema: Option<&Value>, notes: &mut Vec<String>) -> Result<ToolCall, ParseError> {
    let body = body.trim();
    let malformed = || ParseError(format!("malformed tool call: {}", body.chars().take(80).collect::<String>()));
    if !body.starts_with(FUNC_START) || !body.contains('>') {
        return Err(malformed());
    }
    let gt = body.find('>').unwrap();
    let name = body[FUNC_START.len()..gt].trim().to_string();
    let mut rest = &body[gt + 1..];
    let types = declared_types(schema);
    let mut args = Map::new();
    while let Some(p) = rest.find(PARAM_START) {
        rest = &rest[p + PARAM_START.len()..];
        let gt = rest.find('>').ok_or_else(malformed)?;
        let pname = rest[..gt].to_string();
        rest = &rest[gt + 1..];
        let end = param_end(rest, true, Some(pname.trim()));
        let value = if end >= 0 { &rest[..end as usize] } else { rest };
        let next = if end >= 0 { &rest[end as usize + PARAM_END.len()..] } else { "" };
        let v = coerce_value(strip_one_newline(value), types.get(&pname).and_then(|t| t.as_deref()), notes);
        args.insert(pname, v);
        rest = next;
    }
    Ok(ToolCall::new(name, args))
}

/// Python json.dumps of a string; `ascii` = ensure_ascii.
fn py_str(s: &str, ascii: bool) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ascii && (c as u32) > 0x7E => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", u));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Python json.dumps of a value (", " and ": " separators).
pub fn py_dumps(v: &Value, ascii: bool) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => py_str(s, ascii),
        Value::Array(a) => format!("[{}]", a.iter().map(|x| py_dumps(x, ascii)).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter().map(|(k, x)| format!("{}: {}", py_str(k, ascii), py_dumps(x, ascii))).collect::<Vec<_>>().join(", ")
        ),
    }
}

pub struct OutputParser {
    state: State,
    buf: String,
    lead: bool,
    schemas: HashMap<String, Value>,
    stream_tools: bool,
    /// the config's "tool_call_recovery": a declared tool's call written as JSON inside <tool_call> is the call
    recover: bool,
    pending: Vec<(String, Option<ToolCall>)>,
    /// what was repaired in the arguments of the calls (a damaged JSON array or object)
    pub notes: Vec<String>,
    /// #4077: the reasoning ended at a bare `</think>`, so a second one at the start of a line in the answer is a stray...
    after_think: bool,
    /// ... and the strays of that kind dropped from the answer
    pub stray_think: usize,
    /// recover: the call being read opened with `<function=` alone, no `<tool_call>` around it
    bare: bool,
    /// recover: the next call follows in the same `<tool_call>`
    batch: bool,
    pub rescued: usize,
    pub refused: usize,
    fence: String,
    line: String,
    ticks: usize,
    call_end_checked: isize,
    sp: usize,
    ss: Scan,
    scall: Option<ToolCall>,
    sfirst: bool,
    sval_started: bool,
    sdeclared: HashMap<String, Option<String>>,
    stype: Option<String>,
    spname: Option<String>,
    svstart: usize,
    format: CallFormat,
    call_start: &'static str,
    call_stop: &'static str,
    think_open: &'static str,
    think_end: &'static str,
    noise: &'static [&'static str],
    /// #537 tells a quoted </think> by the line break after it; the other formats' answers follow their end tag at once
    quotes_end: bool,
    /// the reply may still start with the format's own opening of its reasoning (Gemma)
    opening: bool,
}

impl OutputParser {
    /// `thinking`: the prompt opened the reasoning. `tools`: the request's tools (template form, each with a name).
    pub fn new(thinking: bool, tools: Option<&[Value]>, stream_tools: bool) -> Self {
        Self::with_format(thinking, tools, stream_tools, CallFormat::Function, false)
    }

    /// `format`: how the model writes calls. `opened`: the prompt ended on the format's own opening of the
    /// reasoning (Gemma after a tool's answer), so the reply starts inside it.
    /// Opt in to reading a declared tool's call written as JSON inside the call wrapper (`<tool_call>{"name": ...,
    /// "arguments": {...}}</tool_call>`, also in a ```json fence): one of serve/frontend.py's recoveries, the one
    /// Kolibri 1 needs (PolyStrata's setup writes "tool_call_recovery": true for it).
    pub fn set_recover(&mut self, on: bool) {
        self.recover = on;
    }

    pub fn with_format(thinking: bool, tools: Option<&[Value]>, stream_tools: bool, format: CallFormat, opened: bool) -> Self {
        let (call_start, call_stop) = format.call_tags();
        let (think_open, think_end) = format.think_tags();
        let schemas = tools
            .into_iter()
            .flatten()
            .filter_map(|t| t.get("name").and_then(Value::as_str).map(|n| (n.to_string(), t.clone())))
            .collect();
        let state = if thinking && (opened || think_open.is_empty()) { State::Reasoning } else { State::Content };
        let mut p = Self {
            state,
            buf: String::new(),
            lead: state == State::Content,
            schemas,
            stream_tools,
            recover: false,
            pending: Vec::new(),
            notes: Vec::new(),
            after_think: false,
            stray_think: 0,
            bare: false,
            batch: false,
            rescued: 0,
            refused: 0,
            fence: String::new(),
            line: String::new(),
            ticks: 0,
            call_end_checked: 0,
            sp: 0,
            ss: Scan::Name,
            scall: None,
            sfirst: true,
            sval_started: false,
            sdeclared: HashMap::new(),
            stype: None,
            spname: None,
            svstart: 0,
            format,
            call_start,
            call_stop,
            think_open,
            think_end,
            noise: format.noise(),
            quotes_end: format == CallFormat::Function,
            opening: !think_open.is_empty() && !opened,
        };
        p.reset_scan();
        p
    }

    fn reset_scan(&mut self) {
        self.call_end_checked = 0;
        self.sp = 0;
        self.ss = Scan::Name;
        self.scall = None;
        self.sfirst = true;
        self.sval_started = false;
        self.sdeclared.clear();
        self.stype = None;
        self.spname = None;
        self.svstart = 0;
    }

    /// Follow the text that has gone out: the open code fence, the current line, and the backticks of the paragraph.
    fn track(&mut self, text: &str) -> String {
        if !text.contains('\n') {
            self.line.push_str(text);
            return text.to_string();
        }
        let parts: Vec<&str> = text.split('\n').collect();
        for (k, part) in parts.iter().enumerate() {
            if k < parts.len() - 1 {
                let line = std::mem::take(&mut self.line) + part;
                let s = line.trim_start();
                if !self.fence.is_empty() {
                    if s.starts_with(&self.fence.repeat(3)) {
                        self.fence.clear();
                    }
                } else if (s.starts_with("```") || s.starts_with("~~~"))
                    && !s[3..].contains(&s[..1].repeat(3))
                {
                    self.fence = s[..1].to_string();
                } else if s.is_empty() {
                    self.ticks = 0;
                } else {
                    self.ticks += line.matches('`').count();
                }
            } else if !part.is_empty() {
                self.line.push_str(part);
            }
        }
        text.to_string()
    }

    /// Whether an opener at buf[p] stands at the start of a line, outside code (the text before it decides).
    fn ok_at(&mut self, p: usize) -> bool {
        let snap = (self.fence.clone(), self.line.clone(), self.ticks);
        let before = self.buf[..p].to_string();
        self.track(&before);
        let ok = self.opener_ok();
        (self.fence, self.line, self.ticks) = snap;
        ok
    }

    /// recover: the first `<function=` in buf that opens a call without the wrapper (start of a line, outside code, a
    /// declared tool) -> (position, decided). A name still arriving is undecided.
    fn bare_opener(&mut self) -> Option<(usize, bool)> {
        let mut p = self.buf.find(FUNC_START);
        while let Some(at) = p {
            if self.ok_at(at) {
                let from = at + FUNC_START.len();
                let gt = find_from(&self.buf, ">", from);
                let raw = &self.buf[from..gt.unwrap_or(self.buf.len())];
                if gt.is_none() && !raw.trim().contains(|c: char| c == '<' || c.is_whitespace()) {
                    return Some((at, false));
                }
                if gt.is_some() && self.schemas.contains_key(raw.trim()) {
                    return Some((at, true));
                }
            }
            p = find_from(&self.buf, FUNC_START, at + 1);
        }
        None
    }

    /// recover: `<parameter=NAME>` with a declared NAME where a call opens. None: not decided yet.
    fn param_opener(&self, after: &str) -> Option<bool> {
        let Some(gt) = after.find('>') else { return if after.contains('\n') { Some(false) } else { None } };
        Some(self.schemas.contains_key(&after[PARAM_START.len()..gt]))
    }

    /// #4077: where buf has a `</think>` at the start of a line, in the answer of a turn whose reasoning already ended at
    /// one. A fence or backtick is no reason to keep it: the reasoning may have left one open.
    fn stray_think_at(&self) -> Option<usize> {
        let mut k = self.buf.find(THINK_END);
        while let Some(at) = k {
            let before = &self.buf[..at];
            let line = match before.rfind('\n') {
                Some(n) => before[n + 1..].to_string(),
                None => format!("{}{before}", self.line),
            };
            if line.trim().is_empty() {
                return Some(at);
            }
            k = find_from(&self.buf, THINK_END, at + 1);
        }
        None
    }

    fn opener_ok(&self) -> bool {
        self.fence.is_empty() && self.line.trim().is_empty() && self.ticks % 2 == 0
    }

    fn in_code(&self) -> bool {
        !self.fence.is_empty() || (self.ticks + self.line.matches('`').count()) % 2 == 1
    }

    /// #537: the `</think>` at buf[i] quoted while thinking (after a quote or backtick, followed by anything but a
    /// line break). None while the character after it has not arrived.
    fn quoted_think_end(&self, i: usize) -> Option<bool> {
        let before = if i > 0 { self.buf[..i].chars().last() } else { self.line.chars().last() };
        if !matches!(before, Some('"' | '\'' | '`')) {
            return Some(false);
        }
        let after = &self.buf[i + THINK_END.len()..];
        let c = after.chars().next()?;
        Some(c != '\r' && c != '\n')
    }

    fn release(&mut self, deliver: bool) -> Vec<Event> {
        let mut out = Vec::new();
        for (raw, call) in std::mem::take(&mut self.pending) {
            match call {
                Some(c) if deliver => {
                    out.push(Event::ToolCall(c));
                    self.rescued += 1;
                }
                call => {
                    if call.is_some() {
                        self.refused += 1;
                    }
                    if !raw.is_empty() {
                        let t = self.track(&raw);
                        out.push(Event::Reasoning(t));
                    }
                }
            }
        }
        out
    }

    /// Length of the longest suffix of `text` that is a proper prefix of one of `tags`.
    fn hold(text: &str, tags: &[&str]) -> usize {
        let mut best = 0;
        let size = text.len();
        for tag in tags {
            if tag.len() <= 1 {
                continue;
            }
            let first = &tag[..1];
            let mut start = find_from(text, first, size.saturating_sub(tag.len() - 1));
            while let Some(s) = start {
                if size - s <= best {
                    break;
                }
                if tag.starts_with(&text[s..]) {
                    best = size - s;
                    break;
                }
                start = find_from(text, first, s + 1);
            }
        }
        best
    }

    fn wrapped_call_end(&mut self) -> isize {
        if self.call_end_checked >= 0 {
            let start = (self.call_end_checked as usize).saturating_sub(CALL_END.len() - 1);
            if find_from(&self.buf, CALL_END, start).is_none() {
                self.call_end_checked = self.buf.len() as isize;
                return -1;
            }
            self.call_end_checked = -1;
        }
        call_end(&self.buf)
    }

    fn args(&self, out: &mut Vec<Event>, s: String) {
        if !s.is_empty() {
            if let Some(c) = &self.scall {
                out.push(Event::ToolArgs { id: c.id.clone(), text: s });
            }
        }
    }

    /// Advance the streaming view of the call body in buf (stream_tools).
    fn scan(&mut self) -> Vec<Event> {
        let mut out = Vec::new();
        loop {
            let rest = self.buf[self.sp..].to_string();
            match self.ss {
                Scan::Name => {
                    let Some(a) = rest.find(FUNC_START) else { return out };
                    let Some(b) = find_from(&rest, ">", a + FUNC_START.len()) else { return out };
                    let name = rest[a + FUNC_START.len()..b].to_string();
                    let call = ToolCall::new(name.clone(), Map::new());
                    self.sdeclared = declared_types(self.schemas.get(&name));
                    out.push(Event::ToolStart { id: call.id.clone(), name });
                    self.scall = Some(call);
                    self.args(&mut out, "{".into());
                    self.sp += b + 1;
                    self.ss = Scan::Between;
                }
                Scan::Between => {
                    let stripped = rest.trim_start();
                    self.sp += rest.len() - stripped.len();
                    if stripped.starts_with(PARAM_START) {
                        let Some(b) = stripped.find('>') else { return out };
                        let pname = stripped[PARAM_START.len()..b].to_string();
                        let lead = if self.sfirst { "" } else { "," };
                        self.args(&mut out, format!("{lead}{}:", py_str(&pname, true)));
                        self.sfirst = false;
                        self.sp += b + 1;
                        self.spname = Some(pname.trim().to_string());
                        if self.sdeclared.get(&pname).and_then(|t| t.as_deref()) == Some("string") {
                            self.args(&mut out, "\"".into());
                            self.ss = Scan::Str;
                            self.sval_started = false;
                        } else {
                            self.ss = Scan::Raw;
                            self.stype = self.sdeclared.get(&pname).cloned().flatten();
                        }
                    } else if stripped.starts_with(FUNC_END) {
                        self.args(&mut out, "}".into());
                        self.sp += FUNC_END.len();
                        self.ss = Scan::Done;
                    } else {
                        return out;
                    }
                }
                Scan::Str => {
                    let mut rest = rest.as_str();
                    if !self.sval_started {
                        if rest.is_empty() {
                            return out;
                        }
                        if let Some(r) = rest.strip_prefix('\n') {
                            self.sp += 1;
                            rest = r;
                        }
                        self.sval_started = true;
                        self.svstart = self.sp;
                    }
                    // the walk starts at the value (a quoted call's opener counts); only a close in the new text can end it
                    let mut end = if rest.contains(PARAM_END) {
                        param_end(&self.buf[self.svstart..], false, self.spname.as_deref())
                    } else {
                        -1
                    };
                    if end >= 0 {
                        end -= (self.sp - self.svstart) as isize;
                        if end < 0 {
                            end = -1;
                        }
                    }
                    if end >= 0 {
                        let value = &rest[..end as usize];
                        let value = value.strip_suffix('\n').unwrap_or(value);
                        let q = py_str(value, true);
                        self.args(&mut out, format!("{}\"", &q[1..q.len() - 1]));
                        self.sp += end as usize + PARAM_END.len();
                        self.ss = Scan::Between;
                        continue;
                    }
                    let mut safe = if end == -2 {
                        rest.find(PARAM_END).unwrap_or(0)
                    } else {
                        rest.len() - Self::hold(rest, &[PARAM_END])
                    };
                    if safe > 0 && rest.as_bytes()[safe - 1] == b'\n' {
                        safe -= 1;
                    }
                    if safe > 0 {
                        let q = py_str(&rest[..safe], true);
                        self.args(&mut out, q[1..q.len() - 1].to_string());
                        self.sp += safe;
                    }
                    return out;
                }
                Scan::Raw => {
                    let end = param_end(&rest, false, self.spname.as_deref());
                    if end < 0 {
                        return out;
                    }
                    let value = strip_one_newline(&rest[..end as usize]);
                    let v = param_value(value, self.stype.as_deref()); // the note comes with the final call
                    self.args(&mut out, py_dumps(&v, false));
                    self.sp += end as usize + PARAM_END.len();
                    self.ss = Scan::Between;
                }
                Scan::Done => return out,
            }
        }
    }

    /// A streamed call that ended without a clean </function>: close its JSON so clients can still parse it.
    fn close_scan(&mut self) -> Vec<Event> {
        let mut out = Vec::new();
        if self.scall.is_none() || self.ss == Scan::Done {
            return out;
        }
        let mut tail = String::new();
        if self.ss == Scan::Str {
            tail.push('"');
        } else if self.ss == Scan::Raw {
            tail.push_str(&py_str(self.buf[self.sp..].trim_matches('\n'), true));
        }
        tail.push('}');
        self.args(&mut out, tail);
        self.ss = Scan::Done;
        out
    }

    /// A whole call body -> its tool_call event. With recover, one that still does not parse is the text it is instead of
    /// an error that ends the request.
    fn finish_call(&mut self, body: &str) -> Result<Vec<Event>, ParseError> {
        let t = body.trim();
        let name = t.get(FUNC_START.len()..).unwrap_or("").split('>').next().unwrap_or("").trim().to_string();
        let bare = std::mem::take(&mut self.bare);
        let schema = self.schemas.get(&name).cloned();
        let mut call = match parse_tool_call_noted(body, schema.as_ref(), &mut self.notes) {
            Ok(c) => c,
            Err(e) if !self.recover => return Err(e),
            Err(_) => {
                self.reset_scan();
                self.state = State::Content;
                let raw = if bare { body.to_string() } else { format!("{CALL_START}{body}{CALL_END}") };
                let t = self.track(&raw);
                return Ok(vec![Event::Content(t)]);
            }
        };
        if let Some(s) = &self.scall {
            call.id = s.id.clone();
        }
        self.reset_scan();
        self.state = State::Content;
        self.lead = true;
        Ok(vec![Event::ToolCall(call)])
    }

    pub fn feed(&mut self, delta: &str) -> Result<Vec<Event>, ParseError> {
        self.buf.push_str(delta);
        let mut out = Vec::new();
        let tools = !self.schemas.is_empty() && self.format == CallFormat::Function;
        let whole = self.format != CallFormat::Function;
        loop {
            match self.state {
                State::RCall => {
                    let body = self.buf[CALL_START.len()..].to_string();
                    let mut end = call_end(&body);
                    let mut drop = CALL_END.len();
                    let f = function_end(&body); // #4427: as in the answer, a following call opener confirms an omitted closer
                    if f >= 0 && (end < 0 || f < end) && next_wrapped_call(body[f as usize..].trim_start()) {
                        (end, drop) = (f, 0);
                    }
                    let think = body.find(THINK_END);
                    if end >= 0 && think.is_none_or(|t| t >= end as usize) {
                        let end = end as usize;
                        let mut call = None;
                        let head = body[..end].trim();
                        let name = head.get(FUNC_START.len()..).unwrap_or("").split('>').next().unwrap_or("").trim();
                        if let Some(schema) = self.schemas.get(name).cloned() {
                            call = parse_tool_call_noted(&body[..end], Some(&schema), &mut self.notes).ok();
                        }
                        let raw = self.buf[..CALL_START.len() + end + drop].to_string();
                        if call.is_some() || !self.pending.is_empty() {
                            self.pending.push((raw, call));
                        } else {
                            let t = self.track(&raw);
                            out.push(Event::Reasoning(t));
                        }
                        self.buf = body[end + drop..].to_string();
                        self.state = State::Reasoning;
                    } else if let Some(think) = think {
                        out.extend(self.release(false));
                        let head = self.buf[..CALL_START.len() + think].to_string();
                        let t = self.track(&head);
                        out.push(Event::Reasoning(t));
                        self.buf = body[think..].to_string();
                        self.state = State::Reasoning;
                    } else if body.chars().count() > RCALL_MAX {
                        out.extend(self.release(false));
                        let all = std::mem::take(&mut self.buf);
                        let t = self.track(&all);
                        out.push(Event::Reasoning(t));
                        self.state = State::Reasoning;
                    } else {
                        return Ok(out);
                    }
                }
                State::Reasoning => {
                    if !self.pending.is_empty() {
                        let stripped_len = self.buf.trim_start().len();
                        if stripped_len < self.buf.len() {
                            let ws: String = self.buf[..self.buf.len() - stripped_len].to_string();
                            self.pending.last_mut().unwrap().0.push_str(&ws);
                            self.buf = self.buf[self.buf.len() - stripped_len..].to_string();
                        }
                        if self.buf.is_empty() {
                            return Ok(out);
                        }
                        if self.buf.starts_with(CALL_START) {
                            self.state = State::RCall;
                        } else if self.buf.starts_with(THINK_END) {
                            out.extend(self.release(true));
                        } else if CALL_START.starts_with(self.buf.as_str()) || THINK_END.starts_with(self.buf.as_str()) {
                            return Ok(out);
                        } else {
                            out.extend(self.release(false));
                        }
                        continue;
                    }
                    let think_end = self.think_end;
                    let i = self.buf.find(think_end);
                    let tool = if tools { self.buf.find(CALL_START) } else { None };
                    if let Some(tool) = tool.filter(|t| i.is_none_or(|i| *t < i)) {
                        if tool > 0 {
                            let head = self.buf[..tool].to_string();
                            let t = self.track(&head);
                            out.push(Event::Reasoning(t));
                        }
                        if self.opener_ok() {
                            self.buf = self.buf[tool..].to_string();
                            self.state = State::RCall;
                        } else {
                            let t = self.track(CALL_START);
                            out.push(Event::Reasoning(t));
                            self.buf = self.buf[tool + CALL_START.len()..].to_string();
                        }
                        continue;
                    }
                    let Some(i) = i else {
                        let keep = if tools { Self::hold(&self.buf, &[think_end, CALL_START]) } else { Self::hold(&self.buf, &[think_end]) };
                        if self.buf.len() > keep {
                            let cut = self.buf.len() - keep;
                            let head = self.buf[..cut].to_string();
                            let t = self.track(&head);
                            out.push(Event::Reasoning(t));
                            self.buf = self.buf[cut..].to_string();
                        }
                        return Ok(out);
                    };
                    let quoted = if self.quotes_end { self.quoted_think_end(i) } else { Some(false) };
                    match quoted {
                        None => {
                            if i > 0 {
                                let head = self.buf[..i].to_string();
                                let t = self.track(&head);
                                out.push(Event::Reasoning(t));
                                self.buf = self.buf[i..].to_string();
                            }
                            return Ok(out);
                        }
                        Some(true) => {
                            let head = self.buf[..i + THINK_END.len()].to_string();
                            let t = self.track(&head);
                            out.push(Event::Reasoning(t));
                            self.buf = self.buf[i + THINK_END.len()..].to_string();
                            continue;
                        }
                        Some(false) => {}
                    }
                    if i > 0 {
                        let head = self.buf[..i].to_string();
                        let t = self.track(&head);
                        out.push(Event::Reasoning(t));
                    }
                    self.buf = self.buf[i + think_end.len()..].to_string();
                    self.state = State::Content;
                    self.lead = true;
                    self.after_think = self.format == CallFormat::Function;
                    self.fence.clear();
                    self.line.clear();
                    self.ticks = 0;
                }
                State::Content => {
                    if self.opening {
                        // the model's own opening of its reasoning
                        if let Some(rest) = self.buf.strip_prefix(self.think_open) {
                            self.buf = rest.to_string();
                            self.state = State::Reasoning;
                            self.opening = false;
                            continue;
                        }
                        if self.think_open.starts_with(self.buf.as_str()) {
                            return Ok(out);
                        }
                        self.opening = false;
                    }
                    for tag in self.noise {
                        if self.buf.contains(tag) {
                            self.buf = self.buf.replace(tag, "");
                        }
                    }
                    if self.lead {
                        let stripped = self.buf.trim_start_matches('\n');
                        if stripped.is_empty() {
                            self.buf.clear();
                            return Ok(out);
                        }
                        self.buf = stripped.to_string();
                        self.lead = false;
                    }
                    let opener = self.buf.find(self.call_start);
                    let stray = if self.after_think { self.stray_think_at() } else { None };
                    if let Some(st) = stray.filter(|st| opener.is_none_or(|i| *st < i)) {
                        // #4077: the reasoning ended at a `</think>` the model wrote while thinking, and this is the one
                        // that ends it. What came between went out as the answer; the tag is dropped and the text after
                        // it starts over, so a call that follows is not taken for a quote.
                        if st > 0 {
                            let head = self.buf[..st].to_string();
                            let t = self.track(&head);
                            out.push(Event::Content(t));
                        }
                        self.buf = self.buf[st + THINK_END.len()..].to_string();
                        self.fence.clear();
                        self.line.clear();
                        self.ticks = 0;
                        self.lead = true;
                        self.stray_think += 1;
                        continue;
                    }
                    let recovering = self.recover && tools;
                    if let Some((b, decided)) = if recovering { self.bare_opener() } else { None }.filter(|(b, _)| opener.is_none_or(|i| *b < i)) {
                        // recover: <function=NAME> alone at the start of a line, the call without its <tool_call>. Until
                        // the name has arrived it is held.
                        let mut j = b;
                        while j > 0 && self.buf.as_bytes()[j - 1] == b'\n' {
                            j -= 1;
                        }
                        if j > 0 && !self.buf[..j].trim().is_empty() {
                            let head = self.buf[..j].to_string();
                            let t = self.track(&head);
                            out.push(Event::Content(t));
                        }
                        if !decided {
                            self.buf = self.buf[j..].to_string();
                            return Ok(out);
                        }
                        self.buf = self.buf[b..].to_string();
                        self.state = State::Call;
                        self.bare = true;
                        continue;
                    }
                    let Some(i) = opener else {
                        let mut tags = vec![self.call_start];
                        tags.extend(self.noise);
                        if self.after_think {
                            tags.push(THINK_END);
                        }
                        let mut j = self.buf.len() - Self::hold(&self.buf, &tags);
                        if recovering {
                            // a line that may be becoming a bare <function=
                            let k = self.buf.rfind('\n').map_or(0, |n| n + 1);
                            let tail = self.buf[k..].to_string();
                            if !tail.is_empty() && FUNC_START.starts_with(tail.as_str()) && self.ok_at(k) {
                                j = j.min(k);
                            }
                        }
                        while j > 0 && self.buf.as_bytes()[j - 1] == b'\n' {
                            j -= 1;
                        }
                        if j > 0 {
                            let head = self.buf[..j].to_string();
                            let t = self.track(&head);
                            out.push(Event::Content(t));
                            self.buf = self.buf[j..].to_string();
                        }
                        return Ok(out);
                    };
                    if whole {
                        // GLM's, Gemma's, DSML's and LFM2's opener is a call where it stands
                        if i > 0 && !self.buf[..i].trim().is_empty() {
                            let head = self.buf[..i].trim_end_matches('\n').to_string();
                            let t = self.track(&head);
                            out.push(Event::Content(t));
                        }
                        self.buf = self.buf[i + self.call_start.len()..].to_string();
                        self.state = State::Call;
                        continue;
                    }
                    // #1058: an opener inside a code fence or inline code is text, decided by the text before it
                    let snap = (self.fence.clone(), self.line.clone(), self.ticks);
                    let before = self.buf[..i].to_string();
                    self.track(&before);
                    let in_code = self.in_code();
                    (self.fence, self.line, self.ticks) = snap;
                    if in_code {
                        let head = self.buf[..i + CALL_START.len()].to_string();
                        let t = self.track(&head);
                        out.push(Event::Content(t));
                        self.buf = self.buf[i + CALL_START.len()..].to_string();
                        continue;
                    }
                    let after = self.buf[i + CALL_START.len()..].trim_start().to_string();
                    if self.recover && (after.starts_with('{') || after.starts_with("```")) {
                        let inner_at = i + CALL_START.len();
                        let Some(end) = self.buf[inner_at..].find(CALL_END) else {
                            // the call is still arriving: what came before it goes out now
                            if i > 0 && !self.buf[..i].trim().is_empty() {
                                let head = self.buf[..i].trim_end_matches('\n').to_string();
                                let t = self.track(&head);
                                out.push(Event::Content(t));
                                self.buf = self.buf[i..].to_string();
                            }
                            return Ok(out);
                        };
                        if let Some(call) = json_call(&self.buf[inner_at..inner_at + end], &self.schemas) {
                            if i > 0 && !self.buf[..i].trim().is_empty() {
                                let head = self.buf[..i].trim_end_matches('\n').to_string();
                                let t = self.track(&head);
                                out.push(Event::Content(t));
                            }
                            if self.stream_tools {
                                out.push(Event::ToolStart { id: call.id.clone(), name: call.name.clone() });
                                out.push(Event::ToolArgs { id: call.id.clone(), text: py_dumps(&Value::Object(call.arguments.clone()), false) });
                            }
                            out.push(Event::ToolCall(call));
                            self.buf = self.buf[inner_at + end + CALL_END.len()..].to_string();
                            self.reset_scan();
                            self.lead = true;
                            continue;
                        }
                    }
                    if self.recover && tools && (after.starts_with(PARAM_START) || (!after.is_empty() && PARAM_START.starts_with(after.as_str()))) {
                        let decided = if after.starts_with(PARAM_START) { self.param_opener(&after) } else { None };
                        if decided.is_none() {
                            return Ok(out); // the name is still arriving
                        }
                        if decided == Some(true) {
                            // <parameter=NAME> as the opener: the call's <function=NAME>
                            if i > 0 && !self.buf[..i].trim().is_empty() {
                                let head = self.buf[..i].trim_end_matches('\n').to_string();
                                let t = self.track(&head);
                                out.push(Event::Content(t));
                            }
                            self.buf = format!("{FUNC_START}{}", &after[PARAM_START.len()..]);
                            self.state = State::Call;
                            continue;
                        }
                    }
                    if !after.is_empty() && !after.starts_with(FUNC_START) && !FUNC_START.starts_with(after.as_str()) {
                        let head = self.buf[..i + CALL_START.len()].to_string();
                        let t = self.track(&head);
                        out.push(Event::Content(t));
                        self.buf = self.buf[i + CALL_START.len()..].to_string();
                        continue;
                    }
                    if !after.starts_with(FUNC_START) {
                        let mut j = i;
                        while j > 0 && self.buf.as_bytes()[j - 1] == b'\n' {
                            j -= 1;
                        }
                        if j > 0 {
                            let head = self.buf[..j].to_string();
                            let t = self.track(&head);
                            out.push(Event::Content(t));
                            self.buf = self.buf[j..].to_string();
                        }
                        return Ok(out);
                    }
                    if i > 0 && !self.buf[..i].trim().is_empty() {
                        let head = self.buf[..i].trim_end_matches('\n').to_string();
                        let t = self.track(&head);
                        out.push(Event::Content(t));
                    }
                    self.buf = self.buf[i + CALL_START.len()..].to_string();
                    self.state = State::Call;
                }
                State::Call if whole => {
                    // the call is announced when its name is complete; its arguments follow whole when it ends
                    if self.stream_tools && self.scall.is_none() {
                        if let Some(name) = self.format.call_name(&self.buf) {
                            let c = ToolCall::new(name.clone(), Map::new());
                            out.push(Event::ToolStart { id: c.id.clone(), name });
                            self.scall = Some(c);
                        }
                    }
                    let Some(i) = self.format.call_end(&self.buf) else { return Ok(out) };
                    let body = self.buf[..i].to_string();
                    self.buf = self.buf[i + self.call_stop.len()..].to_string();
                    let mut calls = self.format.parse(&body, &self.schemas);
                    if calls.is_empty() {
                        let raw = format!("{}{body}{}", self.call_start, self.call_stop);
                        let t = self.track(&raw);
                        out.push(Event::Content(t));
                    } else if let Some(s) = &self.scall {
                        calls[0].id = s.id.clone();
                        out.push(Event::ToolArgs { id: s.id.clone(), text: py_dumps(&Value::Object(calls[0].arguments.clone()), false) });
                    }
                    out.extend(calls.into_iter().map(Event::ToolCall));
                    self.reset_scan();
                    self.state = State::Content;
                    self.lead = true;
                }
                State::Call => {
                    let (mut i, mut drop): (isize, Option<usize>);
                    if self.bare {
                        // no wrapper: the call ends at its </function>; a stray </tool_call> after it is the wrapper's closer
                        i = function_end(&self.buf);
                        drop = None;
                        if i >= 0 {
                            let tail = &self.buf[i as usize..];
                            let after = tail.trim_start();
                            if after.starts_with(CALL_END) {
                                drop = Some(tail.len() - after.len() + CALL_END.len());
                            } else if !after.is_empty() && !CALL_END.starts_with(after) {
                                drop = Some(0);
                            }
                        }
                    } else {
                        i = self.wrapped_call_end();
                        drop = (i >= 0).then_some(CALL_END.len());
                        // only a following call opener confirms an omitted wrapper closer (#4427)
                        let f = function_end(&self.buf);
                        if f >= 0 {
                            let at = self.buf.len() - self.buf[f as usize..].trim_start().len();
                            if next_wrapped_call(&self.buf[at..]) {
                                (i, drop) = (f, Some(0));
                            }
                            if self.recover {
                                // a batch in one wrapper: the next call follows </function>
                                let nxt = self.buf[at..].to_string();
                                if nxt.starts_with(PARAM_START) && self.param_opener(&nxt) == Some(true) {
                                    self.buf = format!("{}{FUNC_START}{}", &self.buf[..at], &nxt[PARAM_START.len()..]);
                                }
                                if self.buf[at..].starts_with(FUNC_START) {
                                    (i, drop) = (f, Some(0));
                                    self.batch = true;
                                }
                            }
                        }
                    }
                    if self.stream_tools {
                        if i >= 0 {
                            let whole = std::mem::take(&mut self.buf);
                            self.buf = whole[..i as usize].to_string();
                            out.extend(self.scan());
                            out.extend(self.close_scan());
                            self.buf = whole;
                        } else {
                            out.extend(self.scan());
                        }
                    }
                    let (Some(drop), true) = (drop, i >= 0) else { return Ok(out) };
                    let i = i as usize;
                    let body = self.buf[..i].to_string();
                    self.buf = self.buf[i + drop..].to_string();
                    let batch = std::mem::take(&mut self.batch);
                    out.extend(self.finish_call(&body)?);
                    if batch && matches!(out.last(), Some(Event::ToolCall(_))) {
                        self.state = State::Call;
                        self.lead = false;
                    }
                }
            }
        }
    }

    /// Still inside the reasoning (for counting a thinking budget).
    pub fn in_reasoning(&self) -> bool {
        self.state == State::Reasoning
    }

    /// In the answer, with nothing held back: a forced call can be opened here.
    pub fn at_clean_content(&self) -> bool {
        self.state == State::Content && self.buf.is_empty()
    }

    /// Nothing held back (no partial tag), so a pass can end here.
    pub fn is_clean(&self) -> bool {
        self.buf.is_empty()
    }

    /// The reasoning's held text is exactly a `</think>` that has not been confirmed as the end yet.
    /// What ends this format's reasoning (</think>, or Gemma's <channel|>).
    pub fn think_end(&self) -> &'static str {
        self.think_end
    }

    pub fn holds_think_end(&self) -> bool {
        self.state == State::Reasoning && self.buf == self.think_end
    }

    /// End of generation. `reason`: how the turn ended (`None` or "stop" = a natural stop); calls waiting from the
    /// reasoning become real calls only on a natural stop.
    pub fn finish(&mut self, reason: Option<&str>) -> Vec<Event> {
        let mut out = Vec::new();
        let natural = matches!(reason, None | Some("stop"));
        if self.state == State::RCall && natural {
            // #4427: a call in the reasoning whose `</function>` ends the output (the `</tool_call>` left out) is whole
            let body = self.buf[CALL_START.len()..].to_string();
            let f = function_end(&body);
            if f >= 0 && body[f as usize..].trim().is_empty() {
                let name = body.trim_start()[FUNC_START.len()..].split('>').next().unwrap_or("").trim().to_string();
                if let Some(schema) = self.schemas.get(&name).cloned() {
                    if let Ok(call) = parse_tool_call_noted(&body[..f as usize], Some(&schema), &mut self.notes) {
                        let raw = std::mem::take(&mut self.buf);
                        self.pending.push((raw, Some(call)));
                        self.state = State::Reasoning;
                    }
                }
            }
        }
        if !self.pending.is_empty() {
            let deliver = matches!(reason, None | Some("stop")) && self.state == State::Reasoning && self.buf.is_empty();
            out.extend(self.release(deliver));
        }
        if self.state == State::Call && self.bare && function_end(&self.buf) >= 0 {
            // recover: a bare call is whole at its </function>
            let i = function_end(&self.buf) as usize;
            let (body, rest) = (self.buf[..i].to_string(), self.buf[i..].trim().to_string());
            self.buf.clear();
            if let Ok(events) = self.finish_call(&body) {
                out.extend(events);
            }
            if !rest.is_empty() && !CALL_END.starts_with(rest.as_str()) {
                let t = self.track(&rest);
                out.push(Event::Content(t));
            }
            return out;
        }
        if self.state == State::Call && self.stream_tools && self.scall.is_some() {
            out.extend(self.scan());
            if self.ss == Scan::Done {
                if natural {
                    // the final event carries the call's arguments, as the one finish_call sends does
                    let mut call = self.scall.clone().unwrap();
                    let schema = self.schemas.get(&call.name).cloned();
                    if let Ok(c) = parse_tool_call_noted(&self.buf, schema.as_ref(), &mut self.notes) {
                        call.arguments = c.arguments;
                    }
                    out.push(Event::ToolCall(call));
                } else {
                    out.push(Event::ToolCall(self.scall.clone().unwrap()));
                }
            }
            self.buf.clear();
            self.reset_scan();
            return out;
        }
        if self.state == State::Reasoning && self.buf == self.think_end {
            self.buf.clear();
            self.state = State::Content;
        }
        if !self.buf.is_empty() {
            let buf = std::mem::take(&mut self.buf);
            match self.state {
                State::Reasoning | State::RCall => out.push(Event::Reasoning(buf)),
                State::Call => out.push(Event::Content(format!("{}{buf}", self.call_start))),
                State::Content => out.push(Event::Content(buf)),
            }
        }
        out
    }
}
