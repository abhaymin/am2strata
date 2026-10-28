//! Incremental parser of the model's output: reasoning (`...</think>`), content, and tool calls in the template's
//! XML form (`<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n</function>\n</tool_call>`).
//! A tag split across deltas is held back until it is complete, so clients never see `<tool_` or `</thi`.
//!
//! Port of PolyStrata serve/frontend.py `OutputParser` for call_format "function" with recovery off (the server's
//! defaults), including streamed tool arguments, quoted `</think>` (#537), calls written inside the reasoning (#804,
//! #1058) and openers inside code (#1058). GLM, Gemma, DeepSeek and LFM2 call formats are not ported yet.

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

/// Where a parameter value in `text` ends: the first `</parameter>` followed (after whitespace) by the next
/// `<parameter=` or `</function>`. -1: none yet; -2: a candidate whose follower has not arrived (`fin` accepts it).
fn param_end(text: &str, fin: bool) -> isize {
    let mut at = text.find(PARAM_END);
    while let Some(a) = at {
        let after = text[a + PARAM_END.len()..].trim_start();
        if after.starts_with(PARAM_START) || after.starts_with(FUNC_END) {
            return a as isize;
        }
        if after.is_empty() || PARAM_START.starts_with(after) || FUNC_END.starts_with(after) {
            return if fin { a as isize } else { -2 };
        }
        at = find_from(text, PARAM_END, a + 1);
    }
    -1
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
            let end = param_end(&text[gt + 1..], false);
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
    if declared == Some("string") {
        return Value::String(value.into());
    }
    if let Ok(v) = serde_json::from_str::<Value>(value) {
        return v;
    }
    if declared == Some("boolean") && matches!(value.trim().to_lowercase().as_str(), "true" | "false") {
        return Value::Bool(value.trim().eq_ignore_ascii_case("true"));
    }
    Value::String(value.into())
}

fn declared_types(schema: Option<&Value>) -> HashMap<String, Option<String>> {
    schema
        .and_then(|s| s.get("parameters"))
        .and_then(|p| p.get("properties"))
        .and_then(Value::as_object)
        .map(|props| {
            props.iter().map(|(k, v)| (k.clone(), v.get("type").and_then(Value::as_str).map(str::to_owned))).collect()
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
        let end = param_end(rest, true);
        let value = if end >= 0 { &rest[..end as usize] } else { rest };
        let next = if end >= 0 { &rest[end as usize + PARAM_END.len()..] } else { "" };
        let v = param_value(strip_one_newline(value), types.get(&pname).and_then(|t| t.as_deref()));
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
                    }
                    let end = param_end(rest, false);
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
                    let end = param_end(&rest, false);
                    if end < 0 {
                        return out;
                    }
                    let value = strip_one_newline(&rest[..end as usize]);
                    let v = param_value(value, self.stype.as_deref());
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

    fn finish_call(&mut self, body: &str) -> Result<Vec<Event>, ParseError> {
        let t = body.trim();
        let name = t.get(FUNC_START.len()..).unwrap_or("").split('>').next().unwrap_or("").trim().to_string();
        let mut call = parse_tool_call(body, self.schemas.get(&name))?;
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
                    let end = call_end(&body);
                    let think = body.find(THINK_END);
                    if end >= 0 && think.is_none_or(|t| t >= end as usize) {
                        let end = end as usize;
                        let mut call = None;
                        let head = body[..end].trim();
                        let name = head.get(FUNC_START.len()..).unwrap_or("").split('>').next().unwrap_or("").trim();
                        if let Some(schema) = self.schemas.get(name) {
                            call = parse_tool_call(&body[..end], Some(schema)).ok();
                        }
                        let raw = self.buf[..CALL_START.len() + end + CALL_END.len()].to_string();
                        if call.is_some() || !self.pending.is_empty() {
                            self.pending.push((raw, call));
                        } else {
                            let t = self.track(&raw);
                            out.push(Event::Reasoning(t));
                        }
                        self.buf = body[end + CALL_END.len()..].to_string();
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
                    let Some(i) = self.buf.find(self.call_start) else {
                        let mut tags = vec![self.call_start];
                        tags.extend(self.noise);
                        let mut j = self.buf.len() - Self::hold(&self.buf, &tags);
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
                    let i = self.wrapped_call_end();
                    let drop = (i >= 0).then_some(CALL_END.len());
                    if self.stream_tools {
                        if i >= 0 {
                            let whole = std::mem::replace(&mut self.buf, String::new());
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
                    out.extend(self.finish_call(&body)?);
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
        if !self.pending.is_empty() {
            let deliver = matches!(reason, None | Some("stop")) && self.state == State::Reasoning && self.buf.is_empty();
            out.extend(self.release(deliver));
        }
        if self.state == State::Call && self.stream_tools && self.scall.is_some() {
            out.extend(self.scan());
            if self.ss == Scan::Done {
                out.push(Event::ToolCall(self.scall.clone().unwrap()));
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
