//! Training conversations → token ids and the mask that says which tokens are scored.
//!
//! The data is chat JSONL as the Workshop's `datagen` writes it (`{"messages": [{"role", "content"}, ...]}`); Soup's
//! other row shapes are read too and turned into messages the way Soup does: `alpaca` (`instruction`, `input`,
//! `output`, `system`), `sharegpt` (`conversations` of `from`/`value`) and `prompt`/`completion`.
//!
//! Each conversation is rendered with **the model's own chat template** (`tokenizer_config.json`'s `chat_template`, or
//! `chat_template.jinja`) by the renderer the server uses, then tokenised with the model's `tokenizer.json`. Only the
//! assistant's tokens are scored, the way `train_on_responses_only` works in Soup, Unsloth and Axolotl: each assistant
//! turn is located in the rendered text as "what the template adds after the generation prompt" — its content and the
//! end-of-turn marker — and a token is supervised when it starts inside such a span.

use std::path::Path;

use anyhow::{anyhow, bail, Context as _, Result};
use serde_json::{json, Map, Value};
use strata_template::ChatTemplate;

use crate::dataset::Record;

/// One tokenised conversation. `supervised[p]` says token `p` counts in the loss (predicted from position `p − 1`).
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    pub tokens: Vec<u32>,
    pub supervised: Vec<bool>,
}

impl Example {
    pub fn supervised_count(&self) -> usize {
        self.supervised.iter().skip(1).filter(|s| **s).count()
    }
}

/// The row shapes of a JSONL training file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    /// decide per row from its keys
    Auto,
    /// `{"messages": [...]}` (Soup calls it chatml)
    Messages,
    Alpaca,
    ShareGpt,
}

impl DataFormat {
    pub fn from_s(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(DataFormat::Auto),
            "messages" | "chatml" | "chat" => Ok(DataFormat::Messages),
            "alpaca" => Ok(DataFormat::Alpaca),
            "sharegpt" => Ok(DataFormat::ShareGpt),
            _ => bail!("data format {s:?} is not read by this trainer: use auto, chatml (messages), alpaca or sharegpt"),
        }
    }
}

fn text<'a>(v: &'a Value, what: &str) -> Result<&'a str> {
    v.as_str().ok_or_else(|| anyhow!("{what} is missing or not text"))
}

/// One decoded JSONL row → a conversation.
pub fn record_of(row: &Value, format: DataFormat) -> Result<Record> {
    let obj = row.as_object().ok_or_else(|| anyhow!("the row is not a JSON object"))?;
    let format = match format {
        DataFormat::Auto if obj.contains_key("messages") => DataFormat::Messages,
        DataFormat::Auto if obj.contains_key("conversations") => DataFormat::ShareGpt,
        DataFormat::Auto if obj.contains_key("instruction") => DataFormat::Alpaca,
        DataFormat::Auto if obj.contains_key("prompt") && obj.contains_key("completion") => {
            return Ok(Record { messages: vec![("user".into(), text(&row["prompt"], "prompt")?.into()), ("assistant".into(), text(&row["completion"], "completion")?.into())] });
        }
        DataFormat::Auto => bail!("the row has none of: messages, conversations, instruction + output, prompt + completion"),
        f => f,
    };
    let messages = match format {
        DataFormat::Messages => row["messages"].as_array().ok_or_else(|| anyhow!("messages is missing or not a list"))?.iter().enumerate()
            .map(|(i, m)| Ok((text(&m["role"], &format!("messages[{i}].role"))?.to_string(), text(&m["content"], &format!("messages[{i}].content"))?.to_string())))
            .collect::<Result<Vec<_>>>()?,
        DataFormat::ShareGpt => row["conversations"].as_array().ok_or_else(|| anyhow!("conversations is missing or not a list"))?.iter().enumerate()
            .map(|(i, m)| {
                let role = match text(&m["from"], &format!("conversations[{i}].from"))? {
                    "human" => "user",
                    "gpt" => "assistant",
                    other => other,
                };
                Ok((role.to_string(), text(&m["value"], &format!("conversations[{i}].value"))?.to_string()))
            })
            .collect::<Result<Vec<_>>>()?,
        DataFormat::Alpaca => {
            let instruction = text(&row["instruction"], "instruction")?;
            let input = row["input"].as_str().unwrap_or("");
            let user = if input.is_empty() { instruction.to_string() } else { format!("{instruction}\n{input}").trim().to_string() };
            let mut m = vec![("user".to_string(), user), ("assistant".to_string(), text(&row["output"], "output")?.to_string())];
            if let Some(sys) = row["system"].as_str().filter(|s| !s.is_empty()) {
                m.insert(0, ("system".to_string(), sys.to_string()));
            }
            m
        }
        DataFormat::Auto => unreachable!("resolved above"),
    };
    if !messages.iter().any(|(r, _)| r == "assistant") {
        bail!("the conversation has no assistant message: nothing to learn from");
    }
    Ok(Record { messages })
}

/// Read a JSONL file into conversations; bad rows come back with their line number and the reason.
pub fn read_jsonl(path: &Path, format: DataFormat) -> Result<(Vec<Record>, Vec<(usize, String)>)> {
    let text = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
    let (mut records, mut bad) = (Vec::new(), Vec::new());
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(line).map_err(|e| anyhow!("invalid JSON: {e}")).and_then(|v| record_of(&v, format)) {
            Ok(r) => records.push(r),
            Err(e) => bad.push((i + 1, e.to_string())),
        }
    }
    Ok((records, bad))
}

/// The model's tokenizer and chat template.
pub struct ChatTokenizer {
    tok: tokenizers::Tokenizer,
    template: ChatTemplate,
    /// `bos_token`, `eos_token`, ... as the template may name them
    vars: Map<String, Value>,
}

/// A special token as `tokenizer_config.json` stores it: a string, or an object with `content`.
fn token_text(v: &Value) -> Option<String> {
    v.as_str().map(str::to_string).or_else(|| v["content"].as_str().map(str::to_string))
}

impl ChatTokenizer {
    /// From a model folder. `template_file` replaces the model's chat template (for a base model without one).
    pub fn load(dir: &Path, template_file: Option<&Path>) -> Result<Self> {
        let tj = dir.join("tokenizer.json");
        let tok = tokenizers::Tokenizer::from_file(&tj).map_err(|e| anyhow!("{}: {e}", tj.display()))?;
        let cfg: Value = std::fs::read_to_string(dir.join("tokenizer_config.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
        let source = match template_file {
            Some(p) => std::fs::read_to_string(p).with_context(|| format!("{}", p.display()))?,
            None => match std::fs::read_to_string(dir.join("chat_template.jinja")) {
                Ok(s) => s,
                Err(_) => match &cfg["chat_template"] {
                    Value::String(s) => s.clone(),
                    // a list of named templates: the default one
                    Value::Array(list) => list.iter().find(|t| t["name"] == "default").and_then(|t| t["template"].as_str()).map(str::to_string)
                        .ok_or_else(|| anyhow!("{}: tokenizer_config.json has several chat templates and none is named default", dir.display()))?,
                    _ => bail!("{}: the model has no chat template (tokenizer_config.json chat_template or chat_template.jinja); pass one with --chat-template FILE", dir.display()),
                },
            },
        };
        let template = ChatTemplate::from_source(source).map_err(|e| anyhow!("chat template: {e}"))?;
        let mut vars = Map::new();
        for key in ["bos_token", "eos_token", "pad_token", "unk_token"] {
            if let Some(t) = token_text(&cfg[key]) {
                vars.insert(key.to_string(), json!(t));
            }
        }
        Ok(Self { tok, template, vars })
    }

    pub fn vocab_size(&self) -> usize {
        self.tok.get_vocab_size(true)
    }

    fn render(&self, messages: &[Value], add_generation_prompt: bool) -> Result<String> {
        self.template.render(messages, None, add_generation_prompt, &self.vars).map_err(|e| anyhow!("chat template: {e}"))
    }

    /// The rendered conversation and the byte ranges of its assistant turns (content and end-of-turn marker).
    pub fn render_with_spans(&self, record: &Record) -> Result<(String, Vec<(usize, usize)>)> {
        let messages: Vec<Value> = record.messages.iter().map(|(r, c)| json!({"role": r, "content": c})).collect();
        let full = self.render(&messages, false)?;
        let mut spans = Vec::new();
        let mut from = 0usize;
        for (i, (role, content)) in record.messages.iter().enumerate() {
            if role != "assistant" {
                continue;
            }
            // the text up to this answer, and up to the end of this answer
            let before = self.render(&messages[..i], true)?;
            let through = self.render(&messages[..=i], false)?;
            let span = if through.starts_with(&before) && full.starts_with(&through) && through.len() > before.len() {
                // trailing white space after the end-of-turn marker (ChatML's newline) is the next turn's business
                let end = before.len() + through[before.len()..].trim_end().len();
                (before.len(), end)
            } else {
                // a template that rewrites earlier turns (or adds nothing for the prompt): find the answer's own text
                let at = full[from..].find(content.as_str()).map(|p| p + from)
                    .ok_or_else(|| anyhow!("the chat template does not write assistant message {i} as it is; cannot tell which tokens to score"))?;
                (at, at + content.len())
            };
            if span.1 <= span.0 {
                bail!("assistant message {i} is empty");
            }
            from = span.1;
            spans.push(span);
        }
        if spans.is_empty() {
            bail!("the conversation has no assistant message");
        }
        Ok((full, spans))
    }

    /// Tokenise one conversation. `max_len` cuts the tail; a conversation whose answers are all cut away is refused.
    /// With `score_everything`, every token after the first is scored (Soup's `train_on_responses_only: false`).
    pub fn encode(&self, record: &Record, max_len: usize, score_everything: bool) -> Result<Example> {
        let (text, spans) = self.render_with_spans(record)?;
        // the template writes the special tokens itself (BOS included, when the model has one)
        let enc = self.tok.encode(text.as_str(), false).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let mut tokens: Vec<u32> = enc.get_ids().to_vec();
        let mut supervised: Vec<bool> = enc.get_offsets().iter()
            .map(|(start, end)| score_everything || spans.iter().any(|(s, e)| *start >= *s && *start < *e && end > start))
            .collect();
        if tokens.len() > max_len {
            tokens.truncate(max_len);
            supervised.truncate(max_len);
        }
        let ex = Example { tokens, supervised };
        if ex.tokens.len() < 2 || ex.supervised_count() == 0 {
            bail!("no assistant token is left within {max_len} tokens (the prompt alone is longer)");
        }
        Ok(ex)
    }

    /// The token count of a conversation (no cut), for choosing a sequence length.
    pub fn count(&self, record: &Record) -> Result<usize> {
        let (text, _) = self.render_with_spans(record)?;
        Ok(self.tok.encode(text.as_str(), false).map_err(|e| anyhow!("tokenizer: {e}"))?.len())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tok.decode(ids, false).map_err(|e| anyhow!("tokenizer: {e}"))
    }
}

/// What tokenising a dataset gave.
#[derive(Debug, Clone, Default)]
pub struct Tokenized {
    pub examples: Vec<Example>,
    /// conversations left out, with the reason (first few, for the log)
    pub skipped: Vec<String>,
    pub skipped_count: usize,
    /// conversations cut at the sequence-length cap
    pub truncated: usize,
}

/// Tokenise every conversation, leaving out (and counting) those that cannot be used.
pub fn tokenize(tok: &ChatTokenizer, records: &[Record], max_len: usize, score_everything: bool) -> Tokenized {
    let mut out = Tokenized::default();
    for (i, r) in records.iter().enumerate() {
        match tok.encode(r, max_len, score_everything) {
            Ok(ex) => {
                if ex.tokens.len() == max_len {
                    out.truncated += 1;
                }
                out.examples.push(ex);
            }
            Err(e) => {
                out.skipped_count += 1;
                if out.skipped.len() < 5 {
                    out.skipped.push(format!("conversation {}: {e}", i + 1));
                }
            }
        }
    }
    out
}
