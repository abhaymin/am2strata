//! A model's Jinja chat template, rendered as PolyStrata's `serve/frontend.py` `ChatTemplate` renders it, which in
//! turn follows transformers' `apply_chat_template`: `trim_blocks`, `lstrip_blocks`, loop controls, a `tojson` filter
//! with Python `json.dumps` output (`ensure_ascii=False`, `", "` and `": "` separators), `raise_exception`, and the
//! Python string and dict methods templates call (`startswith`, `split`, `items`, ...).

use minijinja::value::{Kwargs, Value, ValueKind};
use minijinja::{Environment, Error as JError, ErrorKind};
use serde::Serialize;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reading {path}: {source}")]
    Io { path: String, source: std::io::Error },
    /// The template refused the request (e.g. raise_exception("No user query found")): a client error.
    #[error("{0}")]
    Request(String),
    #[error("chat template: {0}")]
    Template(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub struct ChatTemplate {
    env: Environment<'static>,
    source: String,
}

fn raise_exception(message: String) -> std::result::Result<Value, JError> {
    Err(JError::new(ErrorKind::InvalidOperation, format!("raise_exception:{message}")))
}

/// Python `repr(float)` for the values templates meet: integral values keep ".0".
fn py_float(f: f64) -> String {
    if f.is_nan() {
        "NaN".into()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity".into() } else { "-Infinity".into() }
    } else if f.fract() == 0.0 && f.abs() < 1e16 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

fn py_json_str(s: &str, out: &mut String) {
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
            c => out.push(c),
        }
    }
    out.push('"');
}

struct JsonStyle {
    indent: Option<usize>,
    item_sep: String,
    key_sep: String,
    sort_keys: bool,
}

impl JsonStyle {
    /// Between two items: the separator, minus its trailing space when an indent puts a newline there.
    fn sep(&self, out: &mut String) {
        if self.indent.is_some() {
            out.push_str(self.item_sep.trim_end());
        } else {
            out.push_str(&self.item_sep);
        }
    }

    fn newline(&self, out: &mut String, depth: usize) {
        if let Some(n) = self.indent {
            out.push('\n');
            out.push_str(&" ".repeat(n * depth));
        }
    }
}

/// Python json.dumps over a minijinja value.
fn py_json(v: &Value, style: &JsonStyle, depth: usize, out: &mut String) -> std::result::Result<(), JError> {
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
        ValueKind::Number => {
            // integers print as integers; floats as Python repr
            let s = v.to_string();
            if v.as_i64().is_some() && !s.contains(['.', 'e', 'E']) {
                out.push_str(&s);
            } else {
                let f = f64::try_from(v.clone())
                    .map_err(|_| JError::new(ErrorKind::InvalidOperation, "tojson: not a number"))?;
                out.push_str(&py_float(f));
            }
        }
        ValueKind::String => py_json_str(v.as_str().unwrap_or_default(), out),
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<Value> = v.try_iter()?.collect();
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    style.sep(out);
                }
                style.newline(out, depth + 1);
                py_json(item, style, depth + 1, out)?;
            }
            if !items.is_empty() {
                style.newline(out, depth);
            }
            out.push(']');
        }
        ValueKind::Map => {
            let mut keys: Vec<Value> = v.try_iter()?.collect();
            if style.sort_keys {
                keys.sort_by_key(|k| k.to_string());
            }
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    style.sep(out);
                }
                style.newline(out, depth + 1);
                match k.as_str() {
                    Some(s) => py_json_str(s, out),
                    None => py_json_str(&k.to_string(), out),
                }
                out.push_str(&style.key_sep);
                py_json(&v.get_item(k)?, style, depth + 1, out)?;
            }
            if !keys.is_empty() {
                style.newline(out, depth);
            }
            out.push('}');
        }
        _ => py_json_str(&v.to_string(), out),
    }
    Ok(())
}

fn tojson(v: Value, kwargs: Kwargs) -> std::result::Result<Value, JError> {
    let indent: Option<usize> = kwargs.get("indent")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let _ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let separators: Option<Vec<String>> = kwargs.get("separators")?;
    kwargs.assert_all_used()?;
    // Python's default separators: (", ", ": "), or (",", ": ") when an indent is given
    let (item_sep, key_sep) = match separators {
        Some(s) if s.len() == 2 => (s[0].clone(), s[1].clone()),
        _ if indent.is_some() => (",".into(), ": ".into()),
        _ => (", ".into(), ": ".into()),
    };
    let style = JsonStyle { indent, item_sep, key_sep, sort_keys: sort_keys.unwrap_or(false) };
    let mut out = String::new();
    py_json(&v, &style, 0, &mut out)?;
    Ok(Value::from_safe_string(out))
}

fn from_json(s: String) -> std::result::Result<Value, JError> {
    let v: serde_json::Value = serde_json::from_str(&s)
        .map_err(|e| JError::new(ErrorKind::InvalidOperation, format!("from_json: {e}")))?;
    Ok(Value::from_serialize(&v))
}

impl ChatTemplate {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = std::fs::read_to_string(path)
            .map_err(|e| Error::Io { path: path.display().to_string(), source: e })?;
        Self::from_source(source)
    }

    pub fn from_source(source: String) -> Result<Self> {
        // transformers' {% generation %} marks assistant text for training; render it as a block that always shows
        let open = regex::Regex::new(r"(\{%-?\s*)generation(\s*-?%\})").expect("static regex");
        let close = regex::Regex::new(r"(\{%-?\s*)endgeneration(\s*-?%\})").expect("static regex");
        let src = close.replace_all(&source, "${1}endif${2}");
        let src = open.replace_all(&src, "${1}if true${2}").into_owned();

        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_filter("from_json", from_json);
        env.add_function("raise_exception", raise_exception);
        env.add_template_owned("chat", src).map_err(|e| Error::Template(format!("{e:#}")))?;
        Ok(Self { env, source })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// Render `messages` (OpenAI-style chat messages), `tools`, and any template variables in `extra`
    /// (`enable_thinking`, `reasoning_effort`, ...). Completed empty assistant turns are dropped first, keeping
    /// the last message (PolyStrata #843).
    pub fn render(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        add_generation_prompt: bool,
        extra: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<String> {
        let last = messages.len().saturating_sub(1);
        let kept: Vec<&serde_json::Value> = messages
            .iter()
            .enumerate()
            .filter(|(i, m)| *i == last || !is_empty_assistant(m))
            .map(|(_, m)| m)
            .collect();
        #[derive(Serialize)]
        struct Ctx<'a> {
            messages: Vec<&'a serde_json::Value>,
            tools: Option<&'a serde_json::Value>,
            add_generation_prompt: bool,
            #[serde(flatten)]
            extra: &'a serde_json::Map<String, serde_json::Value>,
        }
        let ctx = Value::from_serialize(Ctx { messages: kept, tools, add_generation_prompt, extra });
        let tmpl = self.env.get_template("chat").map_err(|e| Error::Template(e.to_string()))?;
        tmpl.render(ctx).map_err(|e| {
            let msg = e.detail().unwrap_or_default().to_string();
            match msg.strip_prefix("raise_exception:") {
                Some(m) => Error::Request(m.to_string()),
                None => Error::Template(format!("{e:#}")),
            }
        })
    }
}

fn text_of(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter(|p| matches!(p.get("type").and_then(|t| t.as_str()), Some("text" | "input_text") | None))
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect(),
        _ => String::new(),
    }
}

fn has_image(content: &serde_json::Value) -> bool {
    content.as_array().is_some_and(|parts| {
        parts
            .iter()
            .any(|p| matches!(p.get("type").and_then(|t| t.as_str()), Some("image_url" | "input_image" | "image")))
    })
}

fn is_empty_assistant(m: &serde_json::Value) -> bool {
    if std::env::var("STRATA_KEEP_EMPTY_TURNS").as_deref() == Ok("1") {
        return false;
    }
    let null = serde_json::Value::Null;
    let content = m.get("content").unwrap_or(&null);
    m.get("role").and_then(|r| r.as_str()) == Some("assistant")
        && text_of(content).trim().is_empty()
        && !has_image(content)
        && m.get("tool_calls").map_or(true, |t| t.is_null() || t.as_array().is_some_and(|a| a.is_empty()))
}
