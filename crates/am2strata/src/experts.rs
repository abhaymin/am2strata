//! Specialist models the main model consults in the same turn: one server tool, `ask_expert(expert, question, context?)`.
//! The request carries the user's original request (the last user message of the chat) as well as the sub-question, so an
//! expert sees the whole job and not only the planner's brief. Each expert is an OpenAI-compatible endpoint (`experts` in
//! the run config), another am2Strata, a cluster node or llama-server; its key, when it has one, comes from the environment
//! variable `key_env` names and is never read into the struct, logged or returned. Offered only when a request asks
//! (`"strata_experts": true`) or the key's client profile has the set "experts", and only when at least one expert is
//! configured; never with a structured response_format or a forced tool_choice.

use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::time::{Duration, Instant};

pub const TOOL_NAMES: [&str; 1] = ["ask_expert"];

/// The most characters of an expert's answer the model is given (and where it is cut).
const ANSWER_CHARS: usize = 20_000;
/// A name is 1-32 of [a-z0-9_-].
const MAX_NAME: usize = 32;
/// The defaults when an expert omits them.
const DEFAULT_MAX_TOKENS: u64 = 1024;
const DEFAULT_TIMEOUT_S: u64 = 120;

/// One configured specialist. `key` is not stored (it is read from the environment at call time).
#[derive(Debug)]
struct Expert {
    name: String,
    about: String,
    /// an OpenAI-compatible base URL (trailing slash trimmed); `{url}/chat/completions` is called.
    url: String,
    model: String,
    /// the environment variable that holds the key, if any; read at call time only.
    key_env: Option<String>,
    max_tokens: usize,
    timeout: Duration,
}

#[derive(Debug, Default)]
pub struct Experts {
    experts: Vec<Expert>,
}

impl Experts {
    /// Build the specialists from the run config's `"experts": [{"name","about","url","model","key_env?","max_tokens?","timeout_s?"}]`.
    /// Names must be unique and 1-32 of [a-z0-9_-]; an entry that fails, or a name already used, is skipped with a warning.
    pub fn from_config(rest: &BTreeMap<String, Value>) -> Experts {
        let c = rest.get("experts").cloned().unwrap_or(Value::Null);
        let mut experts = Vec::new();
        let mut seen = HashSet::new();
        if let Some(list) = c.as_array() {
            for e in list {
                let name = e.get("name").and_then(Value::as_str).unwrap_or("").trim().to_string();
                let url = e.get("url").and_then(Value::as_str).unwrap_or("").trim().to_string();
                let about = e.get("about").and_then(Value::as_str).unwrap_or("").trim().to_string();
                let model = e.get("model").and_then(Value::as_str).unwrap_or("").trim().to_string();
                let key_env = e.get("key_env").and_then(Value::as_str).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
                if name.is_empty() || name.len() > MAX_NAME || !name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-') {
                    tracing::warn!("ask_expert expert {name:?}: a name is 1-32 of [a-z0-9_-] (skipped)");
                    continue;
                }
                if url.is_empty() {
                    tracing::warn!("ask_expert expert {name:?}: missing url (skipped)");
                    continue;
                }
                if model.is_empty() {
                    tracing::warn!("ask_expert expert {name:?}: missing model (skipped)");
                    continue;
                }
                if !seen.insert(name.clone()) {
                    tracing::warn!("ask_expert expert {name:?}: a name is used more than once (the later one is skipped)");
                    continue;
                }
                let max_tokens = e.get("max_tokens").and_then(Value::as_u64).filter(|n| *n > 0).unwrap_or(DEFAULT_MAX_TOKENS) as usize;
                let timeout = Duration::from_secs(e.get("timeout_s").and_then(Value::as_u64).filter(|n| *n > 0).unwrap_or(DEFAULT_TIMEOUT_S));
                experts.push(Expert { name, about, url: url.trim_end_matches('/').to_string(), model, key_env, max_tokens, timeout });
            }
        }
        Experts { experts }
    }

    /// The one tool that asks an expert, its description naming each expert and what they do.
    pub fn tools(&self) -> Vec<Value> {
        let mut desc = String::from(
            "Ask a specialist model a sub-question; it answers in this same turn and also gets the user's original request so it \
             sees the whole job, not just the brief. Give it the name of an expert below and the question."
        );
        for e in &self.experts {
            desc.push_str(&format!("\nThe {} expert ({model}): {about}.", e.name, model = e.model, about = e.about));
        }
        vec![json!({"name": "ask_expert", "description": desc,
            "parameters": {"type": "object", "properties": {
                "expert": {"type": "string", "description": "which expert to ask (one of the names above)"},
                "question": {"type": "string", "description": "the sub-question for the expert"},
                "context": {"type": "string", "description": "optional extra context for the question"}},"required": ["expert","question"]}})]
    }

    /// How many experts are configured.
    pub fn len(&self) -> usize { self.experts.len() }

    /// Whether any expert is configured: the set is offered only when at least one is (server_tools.rs).
    pub fn is_empty(&self) -> bool { self.experts.is_empty() }

    /// The system and user messages sent to an expert, using its own "about". Holds the user's original request and the
    /// question (and any context) so the expert sees the whole job; tested for both parts together.
    pub fn messages(&self, name: &str, question: &str, context: Option<&str>, original_request: &str) -> Vec<Value> {
        let about = self.experts.iter().find(|e| e.name == name).map(|e| e.about.clone()).unwrap_or_default();
        let mut msgs: Vec<Value> = Vec::with_capacity(2);
        msgs.push(json!({"role": "system", "content": format!(
            "You are the {name} expert. {about}. Answer the question; say plainly when you are unsure.")}));
        let mut user = String::from("The user's original request:\n");
        user.push_str(original_request);
        user.push_str("\n\nThe question for you:\n");
        user.push_str(question);
        if let Some(ctx) = context {
            user.push_str("\n\nContext:\n");
            user.push_str(ctx);
        }
        msgs.push(json!({"role": "user", "content": user}));
        msgs
    }

    /// The key an expert's requests are authorized with: read from the environment variable `key_env` names (empty -> none).
    /// It is never stored in this struct, logged or returned.
    fn key_for(&self, env_name: &str) -> Option<String> {
        std::env::var(env_name).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
    }

    /// A tool call to an expert -> the result for the model and the client: {"ok","text","chars","truncated","ms"}. An error
    /// (unknown expert, missing question, timeout, HTTP status, empty answer) names the expert and goes back as a tool error;
    /// the turn goes on. The expert is named by the "expert" argument (the tool itself is always `ask_expert`).
    pub async fn call(&self, args: &Value, original_request: &str) -> Value {
        let t0 = Instant::now();
        let out = self.run(args, original_request).await;
        let ms = t0.elapsed().as_millis() as u64;
        match out {
            Ok(text) => {
                let truncated = text.chars().count() > ANSWER_CHARS;
                let shown = if truncated { text.chars().take(ANSWER_CHARS).collect::<String>() } else { text };
                json!({"ok": true, "chars": shown.chars().count(), "text": shown, "truncated": truncated, "ms": ms})
            }
            Err(e) => json!({"ok": false, "text": format!("error: {e}"), "chars": 0, "truncated": false, "ms": ms}),
        }
    }

    async fn run(&self, args: &Value, original_request: &str) -> Result<String, String> {
        let name = args["expert"].as_str().map(str::trim).filter(|n| !n.is_empty()).ok_or("name an expert")?;
        let e = self.experts.iter().find(|x| x.name == name).ok_or_else(|| format!("there is no expert named {name}"))?;
        let question = args["question"].as_str().map(str::trim).filter(|q| !q.is_empty()).ok_or("give a question")?;
        let context = args.get("context").and_then(Value::as_str).map(|s| s.trim()).filter(|s| !s.is_empty());
        let url = format!("{}chat/completions", e.url);
        let body = self.request(&e, &url, question, context, original_request).await?; // send / HTTP status / timeout
        if body.trim().is_empty() {
            return Err(format!("the {name} expert returned nothing"));
        }
        Ok(answer_text(name, &e.model, &body))
    }

    async fn request(&self, exp: &Expert, url: &str, question: &str, context: Option<&str>, original_request: &str) -> Result<String, String> {
        let client = http_client()?;
        let mut req = client.post(url)
            .timeout(exp.timeout)
            .json(&json!({"model": exp.model, "messages": self.messages(&exp.name, question, context, original_request), "max_tokens": exp.max_tokens}));
        if let Some(env) = &exp.key_env {
            if let Some(key) = self.key_for(env) {
                req = req.header(reqwest::header::AUTHORIZATION, format!("Bearer {key}"));
            }
        }
        let resp = req.send().await.map_err(|e| e.to_string())?;
        let status = resp.status();
        if !status.is_success() {
            return Err(format!("the {} expert answered {}", exp.name, status));
        }
        let body = resp.text().await.map_err(|e| format!("the {} expert: {}", exp.name, e))?;
        let content = body_for(&body);
        Ok(content)
    }
}

/// The expert's answer, "Answer from the <name> expert (<model>):\n<text>", cut at ANSWER_CHARS with a note saying so.
fn answer_text(name: &str, model: &str, body: &str) -> String {
    let cut = body.chars().count() > ANSWER_CHARS;
    let shown = if cut { body.chars().take(ANSWER_CHARS).collect::<String>() } else { body.to_string() };
    if cut {
        format!("Answer from the {name} expert ({model}):\n{shown}\n\n(The answer was cut at {ANSWER_CHARS} characters; the expert may have said more.)")
    } else {
        format!("Answer from the {name} expert ({model}):\n{shown}")
    }
}

/// The answer text from a chat completion response ("choices[0].message.content"), as a string (a list of text parts joined).
fn body_for(raw: &str) -> String {
    let v: Value = serde_json::from_str(raw).unwrap_or(Value::Null);
    let content = v.get("choices").and_then(Value::as_array).and_then(|a| a.first())
        .and_then(|c| c.get("message")).and_then(|m| m.get("content"));
    match content {
        Some(c) => crate::memory::content_text(c),
        None => String::new(),
    }
}

fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (X11; Linux x86_64) am2Strata")
        .build()
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rest(json: Value) -> BTreeMap<String, Value> {
        serde_json::from_value(json!({"experts": json})).unwrap()
    }

    #[test]
    fn a_good_expert_is_kept_with_its_config() {
        let e = Experts::from_config(&rest(json!([{"name":"math","about":"checks arithmetic and algebra step by step",
            "url":"http://127.0.0.1:8081/v1/","model":"qwen-math","key_env":"MATH_KEY","max_tokens":512,"timeout_s":30}])));
        assert!(!e.is_empty());
        assert_eq!(e.experts[0].name, "math");
        assert_eq!(e.experts[0].about, "checks arithmetic and algebra step by step");
        assert_eq!(e.experts[0].url, "http://127.0.0.1:8081/v1", "a trailing slash is trimmed");
        assert_eq!(e.experts[0].model, "qwen-math");
        assert_eq!(e.experts[0].key_env.as_deref(), Some("MATH_KEY"));
        assert_eq!(e.experts[0].max_tokens, 512);
        assert_eq!(e.experts[0].timeout, Duration::from_secs(30));
    }

    #[test]
    fn the_config_uses_the_default_max_tokens_and_timeout_when_omitted() {
        let e = Experts::from_config(&rest(json!([{"name":"ocr","about":"reads text in a picture","url":"http://x/v1","model":"m"}])));
        assert_eq!(e.experts[0].max_tokens, DEFAULT_MAX_TOKENS as usize);
        assert_eq!(e.experts[0].timeout, Duration::from_secs(DEFAULT_TIMEOUT_S));
    }

    #[test]
    fn a_name_that_is_not_lowercase_alnum_underscore_dash_is_skipped() {
        let e = Experts::from_config(&rest(json!([{"name":"Bad Name!","about":"x","url":"http://x/v1","model":"m"}])));
        assert!(e.is_empty(), "a bad name is skipped");

        let long = "a".repeat(MAX_NAME + 1);
        let e2 = Experts::from_config(&rest(json!([{"name":long,"about":"x","url":"http://x/v1","model":"m"}])));
        assert!(e2.is_empty(), "a name longer than 32 chars is skipped");

        // a valid name still survives beside a bad one
        let e3 = Experts::from_config(&rest(json!([{"name":"Bad","about":"x","url":"http://x/v1","model":"m"},
                                                   {"name":"good","about":"x","url":"http://x/v1","model":"m"}])));
        assert_eq!(e3.experts[0].name, "good");
    }

    #[test]
    fn a_name_used_twice_keeps_only_the_first() {
        let e = Experts::from_config(&rest(json!([{"name":"math","about":"first","url":"http://x/v1","model":"m"},
                                                  {"name":"math","about":"second","url":"http://y/v1","model":"n"}])));
        assert_eq!(e.experts.len(), 1, "the duplicate is skipped");
        assert_eq!(e.experts[0].about, "first", "the first one is kept");
    }

    #[test]
    fn a_missing_url_is_skipped() {
        let e = Experts::from_config(&rest(json!([{"name":"math","about":"x","model":"m"}])));
        assert!(e.is_empty(), "no url is skipped");
    }

    #[test]
    fn the_key_comes_from_the_named_environment_variable_and_is_not_stored() {
        let e = Experts::from_config(&rest(json!([{"name":"math","about":"x","url":"http://x/v1","model":"m","key_env":"AM2_EXPERT_KEY_TEST"}])));
        std::env::set_var("AM2_EXPERT_KEY_TEST", "  secret-one-two-three  ");
        assert_eq!(e.key_for("AM2_EXPERT_KEY_TEST").as_deref(), Some("secret-one-two-three"), "read from the env var, trimmed");
        assert_eq!(e.key_for("AM2_EXPERT_KEY_MISSING").as_deref(), None, "unset -> none");
        std::env::remove_var("AM2_EXPERT_KEY_TEST");
        // the secret lives only in the environment: it is never held by the struct itself
        assert!(!format!("{:?}", e).contains("secret-one-two-three"));
    }

    #[test]
    fn the_message_for_the_expert_holds_both_the_original_request_and_the_question() {
        let e = Experts::from_config(&rest(json!([{"name":"math","about":"checks arithmetic and algebra step by step",
            "url":"http://x/v1","model":"qwen-math"}])));
        let msgs = e.messages("math", "7 * 6 + 1?", Some("show the steps"), "Explain and solve it, then give the number.");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        let user = msgs[1]["content"].as_str().unwrap();
        assert!(user.contains("Explain and solve it, then give the number."), "the original request: {user}");
        assert!(user.contains("7 * 6 + 1?"), "the question: {user}");
        assert!(user.contains("show the steps"), "the context went through too: {user}");
    }

    #[test]
    fn the_message_has_no_context_line_when_none_is_given() {
        let e = Experts::from_config(&rest(json!([{"name":"math","about":"x","url":"http://x/v1","model":"m"}])));
        let msgs = e.messages("math", "q", None, "the original request");
        let user = msgs[1]["content"].as_str().unwrap();
        assert!(user.contains("The user's original request:\nthe original request"));
        assert!(user.contains("The question for you:\nq"));
        assert!(!user.contains("Context:"));
    }

    #[test]
    fn the_answer_is_cut_at_twenty_thousand_characters_with_a_note() {
        let body = "x".repeat(ANSWER_CHARS + 50);
        let text = answer_text("math", "qwen-math", &body);
        assert!(text.starts_with("Answer from the math expert (qwen-math):"));
        assert!(text.contains(&format!("The answer was cut at {ANSWER_CHARS} characters")), "it says it was cut: {text}");
        // the full body (beyond the cut) is not present — only about twenty thousand characters are kept
        assert!(!text.contains(&body), "the body past the cut is gone: {text}");
    }

    #[test]
    fn a_short_answer_is_not_cut_and_has_no_note() {
        let text = answer_text("math", "qwen-math", "42");
        assert_eq!(text, "Answer from the math expert (qwen-math):\n42");
        assert!(!text.contains("cut at"));
    }

    #[test]
    fn every_expert_is_named_in_the_tool_description() {
        let e = Experts::from_config(&rest(json!([{"name":"math","about":"checks arithmetic and algebra step by step","url":"http://x/v1","model":"m"},
                                                  {"name":"ocr","about":"reads text in a picture","url":"http://y/v1","model":"n"}])));
        let tools = e.tools();
        let desc = tools[0]["description"].as_str().unwrap();
        assert!(desc.contains("The math expert (m): checks arithmetic and algebra step by step."));
        assert!(desc.contains("The ocr expert (n): reads text in a picture."));
    }
}
