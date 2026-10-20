//! The request's prompt: template rendered and tokenized, as serve/server.py `Service.encode_prompt` and
//! `render_prompt` do with default settings (effort_end and preserve_thinking off).

use crate::literals::{mark_think_literals, unmark_think_literals, Literals};
use crate::request::Prepared;
use crate::{RequestError, Result};
use serde_json::{json, Map, Value};
use strata_template::ChatTemplate;
use strata_tokenizer::Tokenizer;

pub struct PromptBuilder<'a> {
    pub template: &'a ChatTemplate,
    pub tok: &'a Tokenizer,
    literals: Literals,
    /// bos_token / eos_token under the names templates know them by
    template_tokens: Map<String, Value>,
}

#[derive(Debug, Clone)]
pub struct Prompt {
    pub text: String,
    pub ids: Vec<u32>,
    /// the rendered prompt leaves a <think> open: the reply starts inside the reasoning
    pub opens_reasoning: bool,
}

impl<'a> PromptBuilder<'a> {
    pub fn new(template: &'a ChatTemplate, tok: &'a Tokenizer) -> Self {
        let literals = Literals::new(&tok.control_tokens());
        let mut template_tokens = Map::new();
        for (name, key) in [("bos_token", "tokenizer.ggml.bos_token_id"), ("eos_token", "tokenizer.ggml.eos_token_id")] {
            if let Some(id) = tok.special_id(key) {
                if let Ok(text) = tok.decode(&[id]) {
                    template_tokens.insert(name.into(), json!(text));
                }
            }
        }
        Self { template, tok, literals, template_tokens }
    }

    pub fn literals(&self) -> &Literals {
        &self.literals
    }

    pub fn build(&self, p: &Prepared) -> Result<Prompt> {
        if p.messages.is_empty() {
            return Err(RequestError("No messages provided.".into()));
        }
        let (marked, marked_tools, changed) = mark_think_literals(&p.messages, p.tools.as_deref(), &self.literals);
        let off = p.kwargs.get("enable_thinking") == Some(&Value::Bool(false));
        let mut kwargs = self.template_tokens.clone();
        kwargs.extend(p.kwargs.clone());
        kwargs.insert("enable_thinking".into(), json!(!off));
        let tools_value = marked_tools.map(Value::Array);
        let mut text = self
            .template
            .render(&marked, tools_value.as_ref(), true, &kwargs)
            .map_err(|e| match e {
                strata_template::Error::Request(m) => RequestError(m),
                other => RequestError(other.to_string()),
            })?;
        // GLM's template always ends on an open <think>: thinking off closes it at once
        if off && text.ends_with("<think>") {
            text.push_str("</think>");
        }
        let (text, plain) = if changed { unmark_think_literals(&text, &self.literals) } else { (text, Vec::new()) };
        let ids = self.tok.encode_with_plain(&text, true, &plain).map_err(|e| RequestError(e.to_string()))?;
        let opens_reasoning = match (text.rfind("<think>"), text.rfind("</think>")) {
            (Some(o), Some(c)) => o > c,
            (Some(_), None) => true,
            _ => false,
        };
        Ok(Prompt { text, ids, opens_reasoning })
    }
}
