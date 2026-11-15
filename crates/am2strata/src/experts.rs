//! Specialist models the main model consults in the same turn: one server tool, `ask_expert(expert, question, context?)`.
//! The request carries the user's original request (the last user message of the chat) as well as the sub-question, so an
//! expert sees the whole job and not only the planner's brief. Each expert is one of two forms (`experts` in the run config):
//! an OpenAI-compatible endpoint (another am2Strata, a cluster node or llama-server) as `{"url","model"}` — its key, when it
//! has one, comes from the environment variable `key_env` names and is never read into the struct, logged or returned; or a
//! paired cluster worker's model as `{"worker","model"}`, reached over that worker's pinned mutual-TLS client with the key
//! pairing gave (no new client, no config key read). A config entry must have exactly one of `url` or `worker`. A worker-form
//! expert is offered only while that worker is paired, ready and lists the model; its call records name the worker. Offered
//! only when a request asks ("strata_experts": true) or the key's client profile has the set "experts", and only when at
//! least one expert is configured; never with a structured response_format or a forced tool_choice.

use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const TOOL_NAMES: [&str; 1] = ["ask_expert"];

/// The most characters of an expert's answer the model is given (and where it is cut).
const ANSWER_CHARS: usize = 20_000;
/// A name is 1-32 of [a-z0-9_-].
const MAX_NAME: usize = 32;
/// The defaults when an expert omits them.
const DEFAULT_MAX_TOKENS: u64 = 1024;
const DEFAULT_TIMEOUT_S: u64 = 120;

/// One configured specialist. A worker-form expert (a paired cluster worker's model) has no url and its key comes from the
/// pairing, never stored here; a url-form expert reads its key from the environment variable `key_env` names at call time.
#[derive(Debug)]
struct Expert {
    name: String,
    about: String,
    /// an OpenAI-compatible base URL (trailing slash trimmed); `{url}/chat/completions` is called. Empty for a worker-form expert.
    url: String,
    /// a paired cluster worker this expert names (the `{"worker","model"}` form). None for a url-form expert.
    worker: Option<String>,
    model: String,
    /// the environment variable that holds the key, if any; read at call time only (url form).
    key_env: Option<String>,
    max_tokens: usize,
    timeout: Duration,
}

/// The configured specialists plus the server's own listen host+port (set at start from main.rs via `with_self_addr`):
/// an expert whose url is a loopback address on our own port would be *us* — calling it waits for our chat's own GPU turn
/// and hangs until the timeout — so that case can be refused before anything is sent.

pub struct Experts {
    experts: Vec<Expert>,
    own_host: String,
    own_port: u16,
    /// this node's cluster (set at start via `with_cluster`): a worker-form expert looks its worker up here at call time and
    /// goes through that worker's pinned mutual-TLS client with the key pairing gave.
    cluster: Option<Arc<crate::cluster::Cluster>>,
}


impl Expert {
    /// Whether this expert can be offered now: a url-form expert always; a worker-form expert only while its paired worker is
    /// ready and lists the model (the cluster status the master polls), so the tool description names only experts it can reach.
    fn is_offered(&self, cluster: &Option<Arc<crate::cluster::Cluster>>) -> bool {
        match &self.worker {
            Some(w) => cluster.as_ref().is_some_and(|c| c.worker_by_name(w).is_some_and(|w| w.is_ready() && w.has_model(&self.model))),
            None => true,
        }
    }
}

impl std::fmt::Debug for Experts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Experts").field("experts", &self.experts).field("own_host", &self.own_host)
            .field("own_port", &self.own_port).field("cluster_set", &self.cluster.is_some()).finish()
    }
}

impl Experts {
    /// Build the specialists from the run config's `"experts": [...]`, two forms: a url `{"name","about","url","model",...}`
    /// (an OpenAI-compatible endpoint) and a paired worker `{"name","about","worker","model",...}` (that worker's model). A
    /// name is unique and 1-32 of [a-z0-9_-]; an entry with neither or both of url/worker, or no model, is skipped with a
    /// warning. The server's own listen address is set with `with_self_addr`; the cluster with `with_cluster`.
    pub fn from_config(rest: &BTreeMap<String, Value>) -> Experts {
        let c = rest.get("experts").cloned().unwrap_or(Value::Null);
        let mut experts = Vec::new();
        let mut seen = HashSet::new();
        if let Some(list) = c.as_array() {
            for e in list {
                let name = e.get("name").and_then(Value::as_str).unwrap_or("").trim().to_string();
                let about = e.get("about").and_then(Value::as_str).unwrap_or("").trim().to_string();
                let model = e.get("model").and_then(Value::as_str).unwrap_or("").trim().to_string();
                let url = e.get("url").and_then(Value::as_str).unwrap_or("").trim().to_string();
                let worker = e.get("worker").and_then(Value::as_str).unwrap_or("").trim().to_string();
                let key_env = e.get("key_env").and_then(Value::as_str).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
                if name.is_empty() || name.len() > MAX_NAME || !name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-') {
                    tracing::warn!("ask_expert expert {name:?}: a name is 1-32 of [a-z0-9_-] (skipped)");
                    continue;
                }
                let has_url = !url.is_empty();
                let has_worker = !worker.is_empty();
                if has_url && has_worker {
                    tracing::warn!("ask_expert expert {name:?}: a form needs exactly one of url or worker (skipped)");
                    continue;
                }
                if !has_url && !has_worker {
                    tracing::warn!("ask_expert expert {name:?}: missing url or worker (skipped)");
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
                experts.push(Expert { name, about, url: url.trim_end_matches('/').to_string(), worker: if has_worker { Some(worker) } else { None }, model, key_env, max_tokens, timeout });
            }
        }
        Experts { experts, own_host: String::new(), own_port: 0, cluster: None }
    }

    /// Set the host we listen on (usually "127.0.0.1") and the port (main.rs), so a loopback expert url on our own port is seen
    /// as pointing at us. Empty host / port 0 (never called) means "unknown" — nothing counts as self.
    pub fn with_self_addr(mut self, host: &str, port: u16) -> Self {
        self.own_host = host.to_string();
        self.own_port = port;
        self
    }

    /// The node's cluster (main.rs), so a worker-form expert can reach its paired worker and go through that worker's pinned
    /// mutual-TLS client at call time with the key pairing gave. Without it, a worker-form expert's worker is not found.
    pub fn with_cluster(mut self, cluster: Arc<crate::cluster::Cluster>) -> Self {
        self.cluster = Some(cluster);
        self
    }

    /// The one tool that asks an expert, its description naming each available expert and what they do. A worker-form expert is
    /// named "(on worker X)" and offered only while that worker is paired, ready and lists the model (cluster status); if none
    /// are available the set is not offered (an empty vector).
    pub fn tools(&self) -> Vec<Value> {
        let mut desc = String::from(
            "Ask a specialist model a sub-question; it answers in this same turn and also gets the user's original request so it \
             sees the whole job, not just the brief. Give it the name of an expert below and the question."
        );
        let mut offered = false;
        for e in &self.experts {
            if !e.is_offered(&self.cluster) { continue; }
            offered = true;
            let who = match &e.worker { Some(w) => format!(" (on worker {w})"), None => String::new() };
            desc.push_str(&format!("\nThe {} expert ({model}){who}: {about}.", e.name, model = e.model, about = e.about));
        }
        if !offered { return Vec::new(); }
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

    /// Whether an expert's url points at this server itself — a loopback host (127.0.0.1 / localhost / ::1) and the port we
    /// listen on. Calling it would reach us and wait for our chat's own GPU turn, so the chat hangs until the timeout; run()
    /// refuses that with a clear error before anything is sent.
    fn points_at_self(&self, e: &Expert) -> bool {
        self_url(&e.url, &self.own_host, self.own_port)
    }

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
        key_from(|n| std::env::var(n).ok(), env_name)
    }

    /// A tool call to an expert -> the result for the model and the client: {"ok","text","chars","truncated","ms"}. An error
    /// (unknown expert, missing question, timeout, HTTP status, empty answer) names the expert and goes back as a tool error;
    /// the turn goes on. The expert is named by the "expert" argument (the tool itself is always `ask_expert`).
    pub async fn call(&self, args: &Value, original_request: &str) -> Value {
        let t0 = Instant::now();
        let out = self.run(args, original_request).await;
        let ms = t0.elapsed().as_millis() as u64;
        match out {
            Ok((text, truncated)) => json!({"ok": true, "chars": text.chars().count(), "text": text, "truncated": truncated, "ms": ms}),
            Err(e) => json!({"ok": false, "text": format!("error: {e}"), "chars": 0, "truncated": false, "ms": ms}),
        }
    }

    async fn run(&self, args: &Value, original_request: &str) -> Result<(String, bool), String> {
        let name = args["expert"].as_str().map(str::trim).filter(|n| !n.is_empty()).ok_or("name an expert")?;
        let e = self.experts.iter().find(|x| x.name == name).ok_or_else(|| format!("there is no expert named {name}"))?;
        let question = args["question"].as_str().map(str::trim).filter(|q| !q.is_empty()).ok_or("give a question")?;
        let context = args.get("context").and_then(Value::as_str).map(|s| s.trim()).filter(|s| !s.is_empty());

        // The request goes out over one of two clients: a paired cluster worker's — its base and the key pairing gave — or a
        // fresh client for a url-form expert authorized by the `key_env` variable. Both build the same body; both send it.
        let mut client = http_client()?;
        let mut url = format!("{}/chat/completions", e.url);
        let mut bearer: Option<String> = None;
        let mut on_worker: Option<&str> = None;
        if let Some(wname) = e.worker.as_deref() {
            // a paired cluster worker: reach it over this node's cluster, with the key pairing gave (no new client, no config key)
            let w = self.cluster.as_ref().and_then(|c| c.worker_by_name(wname))
                .ok_or_else(|| format!("the {name} expert's worker {wname} is not answering"))?;
            if !w.is_ready() {
                return Err(format!("the {name} expert's worker {wname} is not answering"));
            }
            url = format!("{}/v1/chat/completions", w.base().trim_end_matches('/'));
            client = w.client.clone();
            bearer = Some(w.auth_key().to_string());
            on_worker = Some(wname);
        } else if self.points_at_self(e) {
            return Err(format!("the {name} expert is this server itself ({}); point it at another server or a remote model", e.url));
        } else {
            bearer = match &e.key_env {
                Some(env) => self.key_for(env).map(|k| k.to_string()),
                None => None,
            };
        }
        let raw = self.post(&client, &url, &name, &e.model, e.max_tokens, e.timeout, bearer.as_deref(), &question, context, original_request).await?;
        let content = body_for(&raw);
        if content.trim().is_empty() {
            return Err(format!("the {name} expert returned nothing"));
        }
        Ok(answer_text(&name, &e.model, &content, on_worker))
    }

    /// POST the built `url` (an OpenAI chat-completions path) with the same body any expert gets, authorized by `bearer` when
    /// given (as `Bearer <value>`). Returns the raw completion body: call() parses and wraps it. Send / HTTP status / timeout
    /// errors name the expert and go back as a tool error; the turn goes on.
    async fn post(&self, client: &reqwest::Client, url: &str, name: &str, model: &str, max_tokens: usize, timeout: Duration,
                   bearer: Option<&str>, question: &str, context: Option<&str>, original_request: &str) -> Result<String, String> {
        let mut rq = client.post(url).timeout(timeout).json(&json!({"model": model, "messages": self.messages(name, question, context, original_request), "max_tokens": max_tokens}));
        if let Some(k) = bearer {
            rq = rq.header(reqwest::header::AUTHORIZATION, format!("Bearer {k}"));
        }
        let resp = rq.send().await.map_err(|e| {
            if e.is_timeout() {
                format!("the {name} expert did not answer within {} s", timeout.as_secs())
            } else {
                format!("the {name} expert: {e}")
            }
        })?;
        let status = resp.status();
        if !status.is_success() {
            return Err(format!("the {name} expert answered {}", status));
        }
        resp.text().await.map_err(|e| format!("the {name} expert: {e}"))
    }
}

/// The trimmed, non-empty value of `env_name` as `lookup` returns, or None when the variable is empty/whitespace only.
/// Extracted as a plain function so both key_for and the unit test can pass a lookup closure and never touch the shared
/// environment (parallel tests would set/remove the same variable).
fn key_from(lookup: impl Fn(&str) -> Option<String>, env_name: &str) -> Option<String> {
    lookup(env_name).map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// The expert's answer, "Answer from the <name> expert (<model>):\n<text>" — for a worker-form expert, "(on worker X)" too, so
/// the record of the call (the tool result in the chat and the monitor/API calls) names the worker. Cut at ANSWER_CHARS with a
/// note saying so. Cuts once here and reports `cut` back to call(), which no longer cuts again. Returns (wrapped text, cut?).
fn answer_text(name: &str, model: &str, body: &str, on_worker: Option<&str>) -> (String, bool) {
    let cut = body.chars().count() > ANSWER_CHARS;
    let shown = if cut { body.chars().take(ANSWER_CHARS).collect::<String>() } else { body.to_string() };
    let who = match on_worker { Some(w) => format!(" (on worker {w})"), None => String::new() };
    if cut {
        (format!("Answer from the {name} expert ({model}){who}:\n{shown}\n\n(The answer was cut at {ANSWER_CHARS} characters; the expert may have said more.)"), true)
    } else {
        (format!("Answer from the {name} expert ({model}){who}:\n{shown}"), false)
    }
}

/// Whether an OpenAI base `url` (like "http://127.0.0.1:8080/v1") is this server itself — a loopback host (127.0.0.1 /
/// localhost / ::1, any IP that routes back to us) and the port we listen on (`own_port`). With an unknown own address
/// (empty host or port 0) nothing counts as self. Parsing is hand-rolled: these base urls are always plain http://host[:port]/.
fn self_url(url: &str, own_host: &str, own_port: u16) -> bool {
    if own_host.is_empty() || own_port == 0 { return false; }
    let authority = url.split("://").nth(1).unwrap_or(url); // drop scheme
    let authority = authority.split('@').last().unwrap_or(authority); // strip any userinfo@
    let authority = authority.split('/').next().unwrap_or(""); // drop path/query
    let (host, port) = if authority.starts_with('[') {
        match authority.split_once("]:") {
            Some((h, p)) => (h.to_string(), Some(p)),
            None => (authority[1..].to_string(), None),
        }
    } else {
        match authority.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), Some(p)),
            None => (authority.to_string(), None),
        }
    };
    let host = host.trim().trim_start_matches('[');
    let is_loopback_host = host.eq_ignore_ascii_case("localhost")
        || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
    let same_port = port.and_then(|p| p.parse::<u16>().ok()) == Some(own_port);
    is_loopback_host && same_port
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
    fn the_key_lookup_trims_and_treats_empty_as_none() {
        let e = Experts::from_config(&rest(json!([{"name":"math","about":"x","url":"http://x/v1","model":"m"}])));
        // key_from reads via a lookup closure, so this never sets or removes anything in the shared environment (parallel-safe)
        assert_eq!(key_from(|_| Some("  secret-one-two-three  ".to_string()), "AM2_EXPERT_KEY_TEST").as_deref(), Some("secret-one-two-three"), "the value is trimmed");
        assert_eq!(key_from(|_| Some("   ".to_string()), "AM2_EXPERT_KEY_TEST").as_deref(), None, "whitespace only -> none");
        // the env-var path (read-only): a missing variable reads as none
        assert_eq!(e.key_for("AM2_EXPERT_KEY_MISSING").as_deref(), None, "unset -> none");
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
        let (text, truncated) = answer_text("math", "qwen-math", &body, None);
        assert!(truncated, "it reports truncated");
        assert!(text.starts_with("Answer from the math expert (qwen-math):"));
        assert!(text.contains(&format!("The answer was cut at {ANSWER_CHARS} characters")), "it says it was cut: {text}");
        // the full body (beyond the cut) is not present — only about twenty thousand characters are kept
        assert!(!text.contains(&body), "the body past the cut is gone: {text}");
    }

    #[test]
    fn a_short_answer_is_not_cut_and_has_no_note() {
        let (text, truncated) = answer_text("math", "qwen-math", "42", None);
        assert!(!truncated, "a short answer is not truncated");
        assert_eq!(text, "Answer from the math expert (qwen-math):\n42");
        assert!(!text.contains("cut at"));
    }

    #[test]
    fn a_25000_char_answer_is_cut_once_with_the_note_kept() {
        let body = "x".repeat(25_000);
        let (text, truncated) = answer_text("math", "qwen-math", &body, None);
        assert!(truncated, "it reports truncated");
        // exactly one cut of twenty thousand characters: the body line is precisely ANSWER_CHARS long, no more — a second
        // cut elsewhere would make it shorter; and the answer ends with the note (a double cut would have dropped it).
        assert_eq!(text.lines().nth(1).unwrap().chars().count(), ANSWER_CHARS, "one cut kept twenty thousand characters");
        assert!(text.ends_with(&format!("cut at {ANSWER_CHARS} characters; the expert may have said more.)")), "ends with the note: {}", text.len());
    }

    #[test]
    fn an_expert_on_this_servers_own_loopback_port_is_refused() {
        let looped = Experts::from_config(&rest(json!([{"name":"math","about":"x","url":"http://127.0.0.1:8081/v1","model":"m"}]))).with_self_addr("127.0.0.1", 8081);
        assert!(looped.points_at_self(&looped.experts[0]), "a loopback url on our own port is us");

        // the same host, a different port: another server — not refused
        let other_port = Experts::from_config(&rest(json!([{"name":"math","about":"x","url":"http://127.0.0.1:9000/v1","model":"m"}]))).with_self_addr("127.0.0.1", 8081);
        assert!(!other_port.points_at_self(&other_port.experts[0]), "our port is different");

        // a remote host on our port, and no self addr at all: never counts as us
        let remote = Experts::from_config(&rest(json!([{"name":"math","about":"x","url":"https://example.com/v1","model":"m"}])));
        assert!(!remote.points_at_self(&remote.experts[0]), "no self addr set");

        // localhost and [::1] on our port are us too (they all route back to this server)
        let lok = Experts::from_config(&rest(json!([{"name":"math","about":"x","url":"http://localhost:8081/v1","model":"m"}]))).with_self_addr("127.0.0.1", 8081);
        assert!(lok.points_at_self(&lok.experts[0]), "localhost on our port is us");
        let lo6 = Experts::from_config(&rest(json!([{"name":"math","about":"x","url":"http://[::1]:8081/v1","model":"m"}]))).with_self_addr("127.0.0.1", 8081);
        assert!(lo6.points_at_self(&lo6.experts[0]), "[::1] on our port is us");
    }

    /// The refusal happens before anything is sent (no network): run() returns the error, call() wraps it as a tool error. This
    /// is what main.rs's own listen address set via `with_self_addr` produces for an expert that points back at this server.
    #[test]
    fn call_refuses_an_expert_that_is_this_server_itself() {
        let experts = Experts::from_config(&rest(json!([{"name":"math","about":"x","url":"http://127.0.0.1:8081/v1","model":"m"}]))).with_self_addr("127.0.0.1", 8081);
        let args = json!({"expert":"math","question":"7 times 6"});
        let r = futures::executor::block_on(experts.call(&args, "the user's original request"));
        assert_eq!(r["ok"], false, "{:?}", r);
        let text = r["text"].as_str().unwrap();
        assert!(text.contains("is this server itself"), "the tool error says it is us: {text}");
        assert!(text.contains("math"), "the error names the expert: {text}");
        assert_eq!(r["chars"].as_u64(), Some(0), "nothing was fetched from us");
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

    #[test]
    fn a_worker_form_expert_names_its_worker_not_a_url() {
        let e = Experts::from_config(&rest(json!([{"name":"mac-small","about":"the mac's small model",
            "worker":"macm4","model":"mock-b"}])));
        assert_eq!(e.experts.len(), 1, "a worker-form expert is kept");
        assert_eq!(e.experts[0].name, "mac-small");
        assert_eq!(e.experts[0].worker.as_deref(), Some("macm4"), "it names the worker");
        assert!(e.experts[0].url.is_empty(), "a worker-form expert has no url");
        assert_eq!(e.experts[0].model, "mock-b");
    }

    #[test]
    fn a_config_entry_needs_exactly_one_of_url_or_worker() {
        // neither url nor worker: skipped with a warning
        let none = Experts::from_config(&rest(json!([{"name":"math","about":"x","model":"m"}])));
        assert!(none.is_empty(), "a form needs url or worker");
        // both at once: skipped with a warning (a form has exactly one)
        let both = Experts::from_config(&rest(json!([{"name":"math","about":"x","url":"http://x/v1","worker":"w","model":"m"}])));
        assert!(both.is_empty(), "a form needs exactly one of url or worker");
        // each form on its own is fine: one of each is kept
        let one = Experts::from_config(&rest(json!([{"name":"urlform","about":"x","url":"http://x/v1","model":"m"},
                                                    {"name":"workerform","about":"y","worker":"w","model":"n"}])));
        assert_eq!(one.experts.len(), 2, "each form on its own is kept");
    }

    #[test]
    fn a_worker_form_expert_that_is_not_ready_is_a_tool_error() {
        // the paired worker exists but has not answered for the poll: a clear tool error, the turn goes on
        let mut c = crate::cluster::Cluster::empty();
        c.add_test_worker("macm4", &["mock-b"], false);
        let e = Experts::from_config(&rest(json!([{"name":"small","about":"x","worker":"macm4","model":"mock-b"}])))
            .with_cluster(Arc::new(c));
        let r = futures::executor::block_on(e.call(&json!({"expert":"small","question":"2+2?"}), "the request"));
        assert_eq!(r["ok"], false, "{r:?}");
        assert!(r["text"].as_str().unwrap().contains("the small expert's worker macm4 is not answering"), "{r:?}");

        // the worker's name is unknown to this node: the same clear error (nothing was sent)
        let mut c = crate::cluster::Cluster::empty();
        c.add_test_worker("other", &["mock-b"], true);
        let e2 = Experts::from_config(&rest(json!([{"name":"small","about":"x","worker":"macm4","model":"mock-b"}])))
            .with_cluster(Arc::new(c));
        let r2 = futures::executor::block_on(e2.call(&json!({"expert":"small","question":"2+2?"}), "the request"));
        assert_eq!(r2["ok"], false, "{r2:?}");
        assert!(r2["text"].as_str().unwrap().contains("is not answering"), "{r2:?}");
        assert_eq!(r2["chars"].as_u64(), Some(0), "nothing was fetched from an unreachable worker");
    }

    #[test]
    fn a_worker_expert_is_offered_with_its_worker_only_while_ready() {
        // the paired worker ready and listing the model: offered, labelled "(on worker macm4)"
        let mut c = crate::cluster::Cluster::empty();
        c.add_test_worker("macm4", &["mock-b"], true);
        let e = Experts::from_config(&rest(json!([{"name":"small","about":"the mac's small model","worker":"macm4","model":"mock-b"}])))
            .with_cluster(Arc::new(c));
        let tools = e.tools();
        let desc = tools[0]["description"].as_str().unwrap();
        assert!(desc.contains("The small expert (mock-b) (on worker macm4): the mac's small model."), "offered with its worker: {desc}");

        // the same worker, not ready: the expert is no longer offered at all
        let mut c = crate::cluster::Cluster::empty();
        c.add_test_worker("macm4", &["mock-b"], false);
        let e2 = Experts::from_config(&rest(json!([{"name":"small","about":"x","worker":"macm4","model":"mock-b"}])))
            .with_cluster(Arc::new(c));
        assert!(e2.tools().is_empty(), "not offered while the worker is not ready");
    }

    #[test]
    fn a_worker_form_answer_names_the_worker() {
        // the recorded answer of a worker-form expert says which worker answered, so Monitor/API calls name it
        let (text, truncated) = answer_text("small", "mock-b", "Hi from macm4.", Some("macm4"));
        assert!(!truncated);
        assert_eq!(text, "Answer from the small expert (mock-b) (on worker macm4):\nHi from macm4.");
        // a url-form expert keeps the plain form (no worker)
        let (plain, _) = answer_text("small", "mock-b", "Hi", None);
        assert_eq!(plain, "Answer from the small expert (mock-b):\nHi");
    }
}
