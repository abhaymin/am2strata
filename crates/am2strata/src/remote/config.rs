//! The run config's `"remote"` block, read and checked (docs/internal/REMOTE-MODELS.md), and the providers' keys.

use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// The API a provider speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// OpenAI Chat Completions (OpenAI, Groq, Together, OpenRouter, Mistral, DeepSeek, vLLM, Ollama, ...)
    OpenAi,
    /// Anthropic Messages
    Anthropic,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::OpenAi => "openai",
            Kind::Anthropic => "anthropic",
        }
    }
}

/// What a provider may be used for.
pub const KINDS: [&str; 2] = ["chat", "image"];
/// Names a provider may not take: a request's model is read as `<provider>/<model>`, and these are aliases.
const RESERVED: [&str; 4] = ["mix", "local", "remote", "default"];
pub const DEFAULT_MAX_CONCURRENT: usize = 4;

#[derive(Clone, Debug, PartialEq)]
pub struct Provider {
    pub name: String,
    pub kind: Kind,
    /// OpenAI kind: up to and with /v1 (https://api.openai.com/v1); Anthropic kind: the host (https://api.anthropic.com)
    pub base_url: String,
    /// as written in the run config: "env:NAME", "file:PATH", a plain key, or None (set from Settings)
    pub key: Option<String>,
    /// the models offered; empty: the provider's own list (GET /models), kept 10 minutes
    pub models: Vec<String>,
    pub max_concurrent: usize,
    pub kinds: Vec<String>,
    /// the context the tool loop keeps a conversation within (tokens)
    pub context: usize,
    /// Anthropic kind: max_tokens when the client sends none (the Messages API needs one)
    pub max_tokens: u64,
    /// seconds an answer may take in all
    pub timeout_s: u64,
    /// OpenAI kind: "max_completion_tokens" (OpenAI's own API) or "max_tokens" (most others)
    pub max_tokens_field: String,
    /// OpenAI kind: send the client's reasoning effort as `reasoning_effort`
    pub reasoning_effort: bool,
}

impl Provider {
    /// The provider's own URL for `path` ("/chat/completions", "/models", "/messages", "/images/generations").
    pub fn url(&self, path: &str) -> String {
        match self.kind {
            Kind::OpenAi => format!("{}{path}", self.base_url),
            Kind::Anthropic => format!("{}/v1{path}", self.base_url.trim_end_matches("/v1")),
        }
    }

    pub fn has_kind(&self, k: &str) -> bool {
        self.kinds.iter().any(|x| x == k)
    }
}

/// One mix rule's condition: every field given must hold.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct When {
    pub tokens_over: Option<u64>,
    pub has_images: Option<bool>,
    pub tools: Option<bool>,
    pub local_busy: Option<bool>,
    pub key: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    pub when: When,
    /// "local", a local model id, or "<provider>/<model>"
    pub target: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mix {
    pub default: String,
    pub routes: Vec<Rule>,
}

impl Default for Mix {
    fn default() -> Self {
        Mix { default: "local".into(), routes: Vec::new() }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Config {
    pub providers: Vec<Provider>,
    /// what the model "mix" routes by
    pub mix: Mix,
    /// "mix/<name>"
    pub mixes: BTreeMap<String, Mix>,
}

fn name_ok(n: &str) -> bool {
    !n.is_empty() && n.len() <= 40 && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn parse_provider(name: &str, v: &Value) -> Result<Provider, String> {
    let at = |k: &str| format!("config remote.providers.{name}.{k}");
    let o = v.as_object().ok_or(format!("config remote.providers.{name}: expected an object"))?;
    if !name_ok(name) || RESERVED.contains(&name.to_ascii_lowercase().as_str()) {
        return Err(format!("config remote.providers: {name:?} is not a usable provider name (letters, digits, - and _; not mix, local, remote)"));
    }
    let kind = match o.get("kind").and_then(Value::as_str) {
        Some("openai") => Kind::OpenAi,
        Some("anthropic") => Kind::Anthropic,
        other => return Err(format!("{}: {other:?} is not \"openai\" or \"anthropic\"", at("kind"))),
    };
    let base_url = o.get("base_url").and_then(Value::as_str).map(|u| u.trim().trim_end_matches('/').to_string())
        .filter(|u| (u.starts_with("https://") || u.starts_with("http://")) && u.len() > 8)
        .ok_or(format!("{}: expected an http(s) address", at("base_url")))?;
    let key = match o.get("key") {
        None | Some(Value::Null) => None,
        Some(Value::String(k)) if k.trim().is_empty() => None,
        Some(Value::String(k)) => Some(k.trim().to_string()),
        Some(_) => return Err(format!("{}: expected \"env:NAME\" or \"file:PATH\"", at("key"))),
    };
    let models = match o.get("models") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a.iter().map(|m| m.as_str().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
            .ok_or(format!("{}: expected model names", at("models")))).collect::<Result<_, _>>()?,
        Some(_) => return Err(format!("{}: expected a list of model names ([] asks the provider)", at("models"))),
    };
    let max_concurrent = match o.get("max_concurrent") {
        None | Some(Value::Null) => DEFAULT_MAX_CONCURRENT,
        Some(n) => n.as_u64().filter(|n| (1..=64).contains(n)).ok_or(format!("{}: 1 to 64", at("max_concurrent")))? as usize,
    };
    let kinds = match o.get("kinds") {
        None | Some(Value::Null) => vec!["chat".to_string()],
        Some(Value::Array(a)) => {
            let k: Vec<String> = a.iter().map(|x| x.as_str().filter(|x| KINDS.contains(x)).map(str::to_string)
                .ok_or(format!("{}: each one of {}", at("kinds"), KINDS.join(", ")))).collect::<Result<_, _>>()?;
            if k.is_empty() {
                return Err(format!("{}: at least one of {}", at("kinds"), KINDS.join(", ")));
            }
            k
        }
        Some(_) => return Err(format!("{}: expected a list ({})", at("kinds"), KINDS.join(", "))),
    };
    let num = |k: &str, default: u64, min: u64, max: u64| -> Result<u64, String> {
        match o.get(k) {
            None | Some(Value::Null) => Ok(default),
            Some(n) => n.as_u64().filter(|n| (min..=max).contains(n)).ok_or(format!("{}: {min} to {max}", at(k))),
        }
    };
    let openai_itself = base_url.contains("://api.openai.com");
    let max_tokens_field = match o.get("max_tokens_field").and_then(Value::as_str) {
        None => if openai_itself { "max_completion_tokens" } else { "max_tokens" }.to_string(),
        Some(f @ ("max_tokens" | "max_completion_tokens")) => f.to_string(),
        Some(f) => return Err(format!("{}: {f:?} is not \"max_tokens\" or \"max_completion_tokens\"", at("max_tokens_field"))),
    };
    let reasoning_effort = match o.get("reasoning_effort") {
        None | Some(Value::Null) => openai_itself,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(format!("{}: true or false", at("reasoning_effort"))),
    };
    Ok(Provider {
        name: name.to_string(), kind, base_url, key, models, max_concurrent, kinds,
        context: num("context", 128_000, 1024, 10_000_000)? as usize,
        max_tokens: num("max_tokens", 8192, 1, 1_000_000)?,
        timeout_s: num("timeout_s", 600, 5, 86_400)?,
        max_tokens_field, reasoning_effort,
    })
}

fn parse_target(t: &Value, at: &str, providers: &[Provider]) -> Result<String, String> {
    let t = t.as_str().map(str::trim).filter(|t| !t.is_empty()).ok_or(format!("{at}: expected a model (\"local\", a local model id or <provider>/<model>)"))?;
    if t == "mix" || t.starts_with("mix/") {
        return Err(format!("{at}: a mix cannot route to a mix"));
    }
    if let Some((p, m)) = t.split_once('/') {
        if !providers.iter().any(|x| x.name == p) || m.is_empty() {
            return Err(format!("{at}: {t:?} names no provider of remote.providers (expected <provider>/<model>)"));
        }
    }
    Ok(t.to_string())
}

fn parse_mix(v: &Value, at: &str, providers: &[Provider]) -> Result<Mix, String> {
    let o = v.as_object().ok_or(format!("{at}: expected {{\"default\": ..., \"routes\": [...]}}"))?;
    let default = match o.get("default") {
        None | Some(Value::Null) => "local".to_string(),
        Some(t) => parse_target(t, &format!("{at}.default"), providers)?,
    };
    let mut routes = Vec::new();
    for (i, r) in o.get("routes").and_then(Value::as_array).into_iter().flatten().enumerate() {
        let at = format!("{at}.routes[{i}]");
        let target = parse_target(r.get("use").unwrap_or(&Value::Null), &format!("{at}.use"), providers)?;
        let mut when = When::default();
        for (k, c) in r.get("when").and_then(Value::as_object).into_iter().flatten() {
            let bool_of = |c: &Value| c.as_bool().ok_or(format!("{at}.when.{k}: true or false"));
            match k.as_str() {
                "tokens_over" => when.tokens_over = Some(c.as_u64().ok_or(format!("{at}.when.tokens_over: a number of tokens"))?),
                "has_images" => when.has_images = Some(bool_of(c)?),
                "tools" => when.tools = Some(bool_of(c)?),
                "local_busy" => when.local_busy = Some(bool_of(c)?),
                "key" => when.key = Some(c.as_str().filter(|s| !s.is_empty()).ok_or(format!("{at}.when.key: a generated key's id or name, a client profile's name, or \"owner\""))?.to_string()),
                "kind" => return Err(format!("{at}.when.kind: this server has no request classifier to tell code from chat; route by tokens_over, has_images, tools, local_busy or key")),
                other => return Err(format!("{at}.when.{other}: not a condition (tokens_over, has_images, tools, local_busy, key)")),
            }
        }
        if r.get("when").is_some_and(|w| !w.is_object() && !w.is_null()) {
            return Err(format!("{at}.when: expected an object"));
        }
        routes.push(Rule { when, target });
    }
    if o.get("routes").is_some_and(|r| !r.is_array() && !r.is_null()) {
        return Err(format!("{at}.routes: expected a list"));
    }
    Ok(Mix { default, routes })
}

impl Config {
    /// The `"remote"` block; Err names the first wrong entry.
    pub fn parse(v: &Value) -> Result<Config, String> {
        let o = v.as_object().ok_or("config remote: expected {\"providers\": {...}, \"mix\": {...}}")?;
        let mut providers = Vec::new();
        if let Some(ps) = o.get("providers").filter(|p| !p.is_null()) {
            for (name, p) in ps.as_object().ok_or("config remote.providers: expected {\"name\": {\"kind\": ..., \"base_url\": ...}}")? {
                providers.push(parse_provider(name, p)?);
            }
        }
        let mix = match o.get("mix").filter(|m| !m.is_null()) {
            Some(m) => parse_mix(m, "config remote.mix", &providers)?,
            None => Mix::default(),
        };
        let mut mixes = BTreeMap::new();
        for (name, m) in o.get("mixes").and_then(Value::as_object).into_iter().flatten() {
            if !name_ok(name) {
                return Err(format!("config remote.mixes: {name:?} is not a usable name"));
            }
            mixes.insert(name.clone(), parse_mix(m, &format!("config remote.mixes.{name}"), &providers)?);
        }
        Ok(Config { providers, mix, mixes })
    }

    pub fn provider(&self, name: &str) -> Option<&Provider> {
        self.providers.iter().find(|p| p.name == name)
    }

    /// The mix a model name asks for: "mix" or "mix/<name>".
    pub fn mix_of(&self, model: &str) -> Option<&Mix> {
        if model == "mix" {
            return Some(&self.mix);
        }
        self.mixes.get(model.strip_prefix("mix/")?)
    }

    /// The block as the run config keeps it (keys only as their env:/file: references or as they were written).
    pub fn to_value(&self) -> Value {
        let mix = |m: &Mix| {
            let routes: Vec<Value> = m.routes.iter().map(|r| {
                let mut w = Map::new();
                if let Some(n) = r.when.tokens_over { w.insert("tokens_over".into(), json!(n)); }
                if let Some(b) = r.when.has_images { w.insert("has_images".into(), json!(b)); }
                if let Some(b) = r.when.tools { w.insert("tools".into(), json!(b)); }
                if let Some(b) = r.when.local_busy { w.insert("local_busy".into(), json!(b)); }
                if let Some(k) = &r.when.key { w.insert("key".into(), json!(k)); }
                json!({"when": w, "use": r.target})
            }).collect();
            json!({"default": m.default, "routes": routes})
        };
        let mut providers = Map::new();
        for p in &self.providers {
            let mut v = json!({"kind": p.kind.name(), "base_url": p.base_url, "models": p.models, "max_concurrent": p.max_concurrent,
                               "kinds": p.kinds});
            if let Some(k) = &p.key {
                v["key"] = json!(k);
            }
            providers.insert(p.name.clone(), v);
        }
        let mut out = json!({"providers": providers, "mix": mix(&self.mix)});
        if !self.mixes.is_empty() {
            out["mixes"] = Value::Object(self.mixes.iter().map(|(n, m)| (n.clone(), mix(m))).collect());
        }
        out
    }
}

// ------------------------------------------------------------------ keys

/// The keys set from Settings (`<data>/remote-keys.json`, 0600), by provider name.
pub struct KeyStore {
    path: Option<PathBuf>,
    keys: RwLock<BTreeMap<String, String>>,
}

/// Err: `path` can be read by its group or by everyone (a key file must be its owner's alone).
pub fn private_enough(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            return Err(format!("{} can be read by other users (mode {mode:o}): the key is not used; run chmod 600 {}",
                               path.display(), path.display()));
        }
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

impl KeyStore {
    pub fn open(path: Option<PathBuf>) -> KeyStore {
        let mut keys = BTreeMap::new();
        if let Some(p) = path.as_ref().filter(|p| p.exists()) {
            match private_enough(p) {
                Ok(()) => {
                    let v: Value = std::fs::read_to_string(p).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
                    for (k, s) in v["keys"].as_object().into_iter().flatten() {
                        if let Some(s) = s.as_str() {
                            keys.insert(k.clone(), s.to_string());
                        }
                    }
                }
                Err(e) => tracing::error!("remote models: {e}"),
            }
        }
        KeyStore { path, keys: RwLock::new(keys) }
    }

    fn save(&self, keys: &BTreeMap<String, String>) -> Result<(), String> {
        let Some(p) = &self.path else { return Ok(()) };
        crate::cluster::write_private(p, &json!({"keys": keys})).map_err(|e| format!("{}: {e}", p.display()))
    }

    pub fn set(&self, provider: &str, key: &str) -> Result<(), String> {
        let mut g = self.keys.write().unwrap();
        g.insert(provider.to_string(), key.to_string());
        self.save(&g)
    }

    pub fn remove(&self, provider: &str) -> Result<bool, String> {
        let mut g = self.keys.write().unwrap();
        let had = g.remove(provider).is_some();
        self.save(&g)?;
        Ok(had)
    }

    pub fn get(&self, provider: &str) -> Option<String> {
        self.keys.read().unwrap().get(provider).cloned()
    }
}

/// Where a provider's key comes from, and the key. Err says why there is none (never the key itself).
pub fn resolve_key(p: &Provider, store: &KeyStore) -> Result<(&'static str, String), String> {
    if let Some(k) = store.get(&p.name) {
        return Ok(("settings", k));
    }
    match p.key.as_deref() {
        None => Err(format!("{}: no key (set one in Settings › Remote models, or \"key\": \"env:NAME\" / \"file:PATH\")", p.name)),
        Some(k) if k.starts_with("env:") => match std::env::var(&k[4..]) {
            Ok(v) if !v.trim().is_empty() => Ok(("env", v.trim().to_string())),
            _ => Err(format!("{}: the environment variable {} is not set", p.name, &k[4..])),
        },
        Some(k) if k.starts_with("file:") => {
            let path = crate::expand_path(&k[5..]);
            private_enough(&path).map_err(|e| format!("{}: {e}", p.name))?;
            match std::fs::read_to_string(&path) {
                Ok(t) if !t.trim().is_empty() => Ok(("file", t.trim().to_string())),
                Ok(_) => Err(format!("{}: {} is empty", p.name, path.display())),
                Err(e) => Err(format!("{}: {}: {e}", p.name, path.display())),
            }
        }
        Some(k) => Ok(("config", k.to_string())),
    }
}

/// A key as the owner may see it: set or not, its source and its last 4 characters (a long key only).
pub fn key_view(p: &Provider, store: &KeyStore) -> Value {
    match resolve_key(p, store) {
        Ok((source, k)) => {
            let last4 = (k.chars().count() >= 12).then(|| k.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect::<String>());
            json!({"set": true, "source": source, "last4": last4, "ref": p.key.as_deref().filter(|r| r.starts_with("env:") || r.starts_with("file:"))})
        }
        Err(e) => json!({"set": false, "error": e, "ref": p.key.as_deref().filter(|r| r.starts_with("env:") || r.starts_with("file:"))}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Value {
        json!({
            "providers": {
                "openai": {"kind": "openai", "base_url": "https://api.openai.com/v1/", "key": "env:AM2_TEST_NO_SUCH_VAR", "models": ["gpt-5.5"]},
                "anthropic": {"kind": "anthropic", "base_url": "https://api.anthropic.com", "models": ["claude-sonnet-5-5"]},
                "groq": {"kind": "openai", "base_url": "https://api.groq.com/openai/v1", "models": [], "max_concurrent": 2, "kinds": ["chat", "image"]}
            },
            "mix": {"default": "local", "routes": [
                {"when": {"tokens_over": 24000}, "use": "anthropic/claude-sonnet-5-5"},
                {"when": {"has_images": true}, "use": "openai/gpt-5.5"},
                {"when": {"local_busy": true}, "use": "groq/llama-4-70b"}]},
            "mixes": {"code": {"default": "openai/gpt-5.5"}}
        })
    }

    #[test]
    fn the_block_is_read_and_written_back() {
        let c = Config::parse(&sample()).unwrap();
        assert_eq!(c.providers.len(), 3);
        let o = c.provider("openai").unwrap();
        assert_eq!((o.base_url.as_str(), o.max_tokens_field.as_str(), o.reasoning_effort), ("https://api.openai.com/v1", "max_completion_tokens", true));
        assert_eq!(o.url("/chat/completions"), "https://api.openai.com/v1/chat/completions");
        let a = c.provider("anthropic").unwrap();
        assert_eq!(a.url("/messages"), "https://api.anthropic.com/v1/messages");
        let g = c.provider("groq").unwrap();
        assert_eq!((g.max_concurrent, g.max_tokens_field.as_str(), g.reasoning_effort, g.has_kind("image")), (2, "max_tokens", false, true));
        assert_eq!(c.mix.routes.len(), 3);
        assert_eq!(c.mix.routes[0].when.tokens_over, Some(24000));
        assert_eq!(c.mix_of("mix/code").unwrap().default, "openai/gpt-5.5");
        assert!(c.mix_of("mix/none").is_none());
        // written back and read again: the same
        assert_eq!(Config::parse(&c.to_value()).unwrap().providers, c.providers);
        assert_eq!(Config::parse(&c.to_value()).unwrap().mix, c.mix);
    }

    #[test]
    fn wrong_entries_are_named() {
        let bad = |v: Value| Config::parse(&v).unwrap_err();
        assert!(bad(json!({"providers": {"x": {"kind": "gemini", "base_url": "https://a.b"}}})).contains("kind"));
        assert!(bad(json!({"providers": {"x": {"kind": "openai", "base_url": "ftp://a.b"}}})).contains("base_url"));
        assert!(bad(json!({"providers": {"mix": {"kind": "openai", "base_url": "https://a.b"}}})).contains("not a usable provider name"));
        assert!(bad(json!({"providers": {"x": {"kind": "openai", "base_url": "https://a.b", "kinds": ["video"]}}})).contains("kinds"));
        assert!(bad(json!({"mix": {"routes": [{"when": {"kind": "code"}, "use": "local"}]}})).contains("no request classifier"));
        assert!(bad(json!({"mix": {"routes": [{"when": {"colour": 1}, "use": "local"}]}})).contains("not a condition"));
        assert!(bad(json!({"mix": {"default": "nope/model"}})).contains("names no provider"));
        assert!(bad(json!({"mix": {"default": "mix/x"}})).contains("cannot route to a mix"));
    }

    #[test]
    fn a_key_file_others_can_read_is_refused() {
        let dir = std::env::temp_dir().join(format!("am2-remote-key-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("k");
        std::fs::write(&f, "sk-test-not-a-real-key-0000\n").unwrap();
        let mut p = parse_provider("p", &json!({"kind": "openai", "base_url": "https://a.b/v1"})).unwrap();
        p.key = Some(format!("file:{}", f.display()));
        let store = KeyStore::open(None);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o644)).unwrap();
            let e = resolve_key(&p, &store).unwrap_err();
            assert!(e.contains("644") && e.contains("chmod 600"), "{e}");
            std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        assert_eq!(resolve_key(&p, &store).unwrap(), ("file", "sk-test-not-a-real-key-0000".to_string()));
        let v = key_view(&p, &store);
        assert_eq!((v["set"].as_bool(), v["last4"].as_str()), (Some(true), Some("0000")));
        assert!(!v.to_string().contains("sk-test"), "the view never has the key");
        // a key set from Settings wins
        let store = KeyStore::open(Some(dir.join("remote-keys.json")));
        store.set("p", "sk-from-settings-1234").unwrap();
        assert_eq!(resolve_key(&p, &store).unwrap().0, "settings");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(dir.join("remote-keys.json")).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
