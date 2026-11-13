//! Coding agents on am2Strata, on this PC or another one (`am2strata agent ...`, GET /api/agents): what
//! `unsloth start` and OmniRoute's `run` / `setup-<tool>` do, for this server.
//!
//! - **start**: the agent runs once against am2Strata, set up for that run only (environment variables, flags, or a
//!   config in a temporary folder); nothing of the user's changes.
//! - **setup**: the agent's own config file gets an am2Strata provider and its default model (the model override).
//!   The file is backed up first (<file>.am2strata-backup) and `reset` puts it back. The API key stays out of the
//!   file: the config names the variable AM2STRATA_API_KEY, which the user's shell sets.
//! - **guide**: what to enter, for agents whose config is a settings page or a format not written here.
//!
//! Each agent's format is from its own documentation; `checked` says how far it was tried here (opencode 1.19.5 ran
//! with the config this writes). The Gemini CLI speaks the Gemini API, which gemini.rs serves. agy (Antigravity) and
//! cursor-agent take no custom endpoint, so they are listed as not supported rather than reached by intercepting
//! their traffic.

// shared by the server (the Connect page's list and guides) and am2strata-run (start, setup, reset): each uses part
#![allow(dead_code)]

use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

pub const KEY_VAR: &str = "AM2STRATA_API_KEY";
const PROVIDER: &str = "am2strata";
const BACKUP: &str = "am2strata-backup";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Support {
    /// start and setup
    Full,
    /// start only (its setup is the user's shell profile)
    Start,
    /// setup only (it reads no per-run config)
    Setup,
    /// the guide only
    Guide,
    /// not reachable; the text says why
    No(&'static str),
}

pub struct Agent {
    pub name: &'static str,
    pub label: &'static str,
    /// the command the agent runs as
    pub exe: &'static str,
    /// the API it speaks to the server
    pub api: &'static str,
    pub support: Support,
    /// where the format is from, and whether it was run here
    pub checked: &'static str,
}

pub const AGENTS: &[Agent] = &[
    Agent { name: "claude", label: "Claude Code", exe: "claude", api: "anthropic", support: Support::Full,
            checked: "Claude Code's settings and environment variable docs" },
    Agent { name: "codex", label: "Codex CLI", exe: "codex", api: "responses", support: Support::Full,
            checked: "Codex's config.toml model_providers and -c overrides" },
    Agent { name: "opencode", label: "opencode", exe: "opencode", api: "openai", support: Support::Full,
            checked: "run here: opencode 1.19.5 read this config (OPENCODE_CONFIG_CONTENT and opencode.json)" },
    Agent { name: "pi", label: "Pi coding agent", exe: "pi", api: "openai", support: Support::Full,
            checked: "Pi's docs/models.md and docs/configuration.md (PI_CODING_AGENT_DIR)" },
    Agent { name: "droid", label: "Factory Droid", exe: "droid", api: "anthropic", support: Support::Setup,
            checked: "Factory's BYOK docs (~/.factory/settings.json customModels)" },
    Agent { name: "qwen", label: "Qwen Code", exe: "qwen", api: "openai", support: Support::Start,
            checked: "Qwen Code's OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL variables" },
    Agent { name: "aider", label: "Aider", exe: "aider", api: "openai", support: Support::Start,
            checked: "Aider's OpenAI-compatible API docs (OPENAI_API_BASE, --model openai/<id>)" },
    Agent { name: "goose", label: "Goose", exe: "goose", api: "openai", support: Support::Start,
            checked: "Goose's OpenAI provider variables (OPENAI_HOST, OPENAI_BASE_PATH)" },
    Agent { name: "gemini", label: "Gemini CLI", exe: "gemini", api: "gemini", support: Support::Start,
            checked: "am2Strata's /v1beta Gemini API (tested with Google's request shapes); the CLI's GOOGLE_GEMINI_BASE_URL, as OmniRoute runs it" },
    Agent { name: "omp", label: "Oh My Pi", exe: "omp", api: "openai", support: Support::Guide,
            checked: "Oh My Pi's README (~/.omp/agent/models.yml)" },
    Agent { name: "crush", label: "Crush", exe: "crush", api: "openai", support: Support::Guide,
            checked: "Crush's config docs (providers, $VAR keys)" },
    Agent { name: "hermes", label: "Hermes Agent", exe: "hermes", api: "openai", support: Support::Guide,
            checked: "a custom OpenAI-compatible endpoint, as Unsloth and OmniRoute list it" },
    Agent { name: "openclaw", label: "OpenClaw", exe: "openclaw", api: "openai", support: Support::Guide,
            checked: "a custom OpenAI-compatible provider, as Unsloth and OmniRoute list it" },
    Agent { name: "continue", label: "Continue", exe: "cn", api: "openai", support: Support::Guide,
            checked: "Continue's config.yaml models (provider: openai, apiBase)" },
    Agent { name: "cline", label: "Cline", exe: "cline", api: "openai", support: Support::Guide,
            checked: "Cline's OpenAI Compatible provider" },
    Agent { name: "kilocode", label: "Kilo Code", exe: "kilocode", api: "openai", support: Support::Guide,
            checked: "Kilo Code's OpenAI Compatible provider" },
    Agent { name: "zed", label: "Zed", exe: "zed", api: "openai", support: Support::Guide,
            checked: "Zed's language_models.openai_compatible settings" },
    Agent { name: "vscode", label: "VS Code (Copilot Chat)", exe: "code", api: "openai", support: Support::Guide,
            checked: "Copilot Chat's custom OpenAI-compatible model entry" },
    Agent { name: "agy", label: "Antigravity CLI", exe: "agy", api: "-", support: Support::No(
        "Antigravity takes no custom model endpoint; OmniRoute reaches it only by intercepting its traffic (MITM), which am2Strata does not do"),
            checked: "-" },
    Agent { name: "cursor-agent", label: "Cursor CLI", exe: "cursor-agent", api: "-", support: Support::No(
        "the Cursor CLI documents no custom base URL; its models run on Cursor's servers"),
            checked: "-" },
];

pub fn find(name: &str) -> Option<&'static Agent> {
    let n = name.to_lowercase();
    let n = match n.as_str() {
        "claude-code" => "claude",
        "oh-my-pi" => "omp",
        "antigravity" => "agy",
        "cursor" | "cursor-cli" => "cursor-agent",
        "qwen-code" => "qwen",
        "kilo" => "kilocode",
        "cn" => "continue",
        "factory" => "droid",
        other => other,
    };
    AGENTS.iter().find(|a| a.name == n)
}

/// The server an agent is pointed at: this PC's or a remote one.
#[derive(Clone, Debug)]
pub struct Target {
    /// http://host:port, no /v1
    pub url: String,
    pub key: Option<String>,
    /// the main model (Claude Code's opus and sonnet, Codex's model, ...)
    pub model: String,
    /// the small, fast one (Claude Code's haiku and background work); the main model when not given
    pub small: Option<String>,
    /// the server tools this run asks for (sent as the x-strata-tools header: "web,image"), within the key's profile
    pub tools: Option<String>,
}

impl Target {
    pub fn v1(&self) -> String {
        format!("{}/v1", self.url.trim_end_matches('/'))
    }
    fn root(&self) -> String {
        self.url.trim_end_matches('/').to_string()
    }
    fn small(&self) -> &str {
        self.small.as_deref().unwrap_or(&self.model)
    }
    /// what a config writes for the key: the variable's name (the shell sets it), or a placeholder for a server
    /// without a key (most agents refuse an empty one)
    fn key_or(&self, reference: &str) -> String {
        if self.key.is_some() { reference.to_string() } else { "am2strata-no-key".to_string() }
    }
}

/// One run of an agent: the command, its arguments and environment, and the files it needs (in a temporary folder).
#[derive(Debug, Default)]
pub struct Plan {
    pub exe: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub files: Vec<(PathBuf, String)>,
    pub notes: Vec<String>,
}

impl Plan {
    fn new(a: &Agent, t: &Target) -> Plan {
        let mut p = Plan { exe: a.exe.to_string(), ..Default::default() };
        if let Some(k) = &t.key {
            p.env.push((KEY_VAR.into(), k.clone()));
        }
        p
    }
    fn env(&mut self, k: &str, v: impl Into<String>) {
        self.env.push((k.to_string(), v.into()));
    }
}

/// How to run `name` once against `t`; `tmp` is a folder of its own for this run.
pub fn start(name: &str, t: &Target, tmp: &Path) -> Result<Plan, String> {
    let a = find(name).ok_or_else(|| format!("no agent called {name:?} (am2strata agent list)"))?;
    let mut p = Plan::new(a, t);
    let key = t.key.clone().unwrap_or_else(|| "am2strata-no-key".into());
    match a.name {
        "claude" => {
            p.env("ANTHROPIC_BASE_URL", t.root());
            p.env("ANTHROPIC_AUTH_TOKEN", key);
            p.env("ANTHROPIC_MODEL", &t.model);
            p.env("ANTHROPIC_DEFAULT_OPUS_MODEL", &t.model);
            p.env("ANTHROPIC_DEFAULT_SONNET_MODEL", &t.model);
            p.env("ANTHROPIC_DEFAULT_HAIKU_MODEL", t.small());
            p.env("CLAUDE_CODE_SUBAGENT_MODEL", &t.model);
            if let Some(tools) = &t.tools {
                p.env("ANTHROPIC_CUSTOM_HEADERS", format!("x-strata-tools: {tools}"));
            }
        }
        "codex" => {
            p.args = codex_overrides(t);
        }
        "opencode" => {
            p.env("OPENCODE_CONFIG_CONTENT", serde_json::to_string(&opencode_config(t)).unwrap());
        }
        "pi" => {
            let dir = tmp.join("pi-agent");
            p.files.push((dir.join("models.json"), serde_json::to_string_pretty(&json!({"providers": {PROVIDER: pi_provider(t)}})).unwrap()));
            p.env("PI_CODING_AGENT_DIR", dir.display().to_string());
            p.args = vec!["--provider".into(), PROVIDER.into(), "--model".into(), t.model.clone()];
            p.notes.push("this run keeps its sessions in a temporary folder; `agent setup pi` keeps them".into());
        }
        "qwen" => {
            p.env("OPENAI_BASE_URL", t.v1());
            p.env("OPENAI_API_KEY", key);
            p.env("OPENAI_MODEL", &t.model);
        }
        "aider" => {
            p.env("OPENAI_API_BASE", t.v1());
            p.env("OPENAI_API_KEY", key);
            p.args = vec!["--model".into(), format!("openai/{}", t.model)];
        }
        "gemini" => {
            // the CLI signs in with a Gemini API key from a home of its own for this run (its usual home keeps a Google
            // sign-in, which would win); the workspace trust prompt stays the CLI's (--skip-trust skips it)
            let home = tmp.join("gemini-home");
            p.files.push((home.join(".gemini/settings.json"),
                          json!({"security": {"auth": {"selectedType": "gemini-api-key"}}}).to_string()));
            p.env("GEMINI_CLI_HOME", home.display().to_string());
            p.env("GOOGLE_GEMINI_BASE_URL", t.root());
            p.env("GEMINI_API_KEY", key);
            p.env("GEMINI_DEFAULT_AUTH_TYPE", "gemini-api-key");
            p.args = vec!["-m".into(), t.model.clone()];
            p.notes.push("this run keeps the CLI's settings and history in a temporary home".into());
        }
        "goose" => {
            p.env("GOOSE_PROVIDER", "openai");
            p.env("OPENAI_HOST", t.root());
            p.env("OPENAI_BASE_PATH", "v1/chat/completions");
            p.env("OPENAI_API_KEY", key);
            p.env("GOOSE_MODEL", &t.model);
        }
        _ => return Err(match a.support {
            Support::No(why) => format!("{} is not supported: {why}", a.label),
            Support::Setup => format!("{} reads only its own settings file: run `am2strata agent setup {}` once, then start {} as usual",
                                      a.label, a.name, a.exe),
            _ => format!("{} is set up in its own settings: `am2strata agent show {}` says what to enter", a.label, a.name),
        }),
    }
    if t.tools.is_some() && !matches!(a.name, "claude" | "codex" | "opencode") {
        p.notes.push(format!("{} sends no extra headers: --tools is left out, and the key's client profile alone picks the server's tools", a.label));
    }
    Ok(p)
}

fn codex_provider_toml(t: &Target) -> String {
    let mut s = format!("name = \"am2Strata\", base_url = \"{}\", wire_api = \"responses\"", t.v1());
    if t.key.is_some() {
        s.push_str(&format!(", env_key = \"{KEY_VAR}\""));
    }
    if let Some(tools) = &t.tools {
        s.push_str(&format!(", http_headers = {{ \"x-strata-tools\" = \"{}\" }}", tools.replace('"', "")));
    }
    s
}

/// Codex's -c overrides: an am2Strata provider for this run, and the model.
fn codex_overrides(t: &Target) -> Vec<String> {
    vec!["-c".into(), format!("model_providers.{PROVIDER}={{ {} }}", codex_provider_toml(t)),
         "-c".into(), format!("model_provider=\"{PROVIDER}\""),
         "-m".into(), t.model.clone()]
}

fn opencode_options(t: &Target) -> Value {
    let mut o = json!({"baseURL": t.v1(), "apiKey": t.key_or(&format!("{{env:{KEY_VAR}}}"))});
    if let Some(tools) = &t.tools {
        o["headers"] = json!({"x-strata-tools": tools});
    }
    o
}

fn opencode_config(t: &Target) -> Value {
    let mut models = Map::new();
    for m in [Some(t.model.as_str()), t.small.as_deref()].into_iter().flatten() {
        models.insert(m.to_string(), json!({"name": m}));
    }
    json!({"$schema": "https://opencode.ai/config.json",
           "provider": {PROVIDER: {"npm": "@ai-sdk/openai-compatible", "name": "am2Strata",
                                   "options": opencode_options(t),
                                   "models": models}},
           "model": format!("{PROVIDER}/{}", t.model),
           "small_model": format!("{PROVIDER}/{}", t.small())})
}

fn pi_provider(t: &Target) -> Value {
    let mut models = vec![json!({"id": t.model})];
    if let Some(s) = t.small.as_deref().filter(|s| *s != t.model) {
        models.push(json!({"id": s}));
    }
    json!({"baseUrl": t.v1(), "api": "openai-completions", "apiKey": t.key_or(&format!("${KEY_VAR}")), "models": models})
}

/// One file `setup` writes: where, what it held, what it will hold.
#[derive(Debug)]
pub struct Change {
    pub path: PathBuf,
    pub before: Option<String>,
    pub after: String,
}

fn read_json(path: &Path) -> Result<(Option<String>, Value), String> {
    match std::fs::read_to_string(path) {
        Ok(s) if s.trim().is_empty() => Ok((Some(s), json!({}))),
        Ok(s) => match serde_json::from_str::<Value>(&s) {
            Ok(v) if v.is_object() => Ok((Some(s), v)),
            _ => Err(format!("{} is not plain JSON (comments?), so it is left alone; `agent show` gives the block to add by hand", path.display())),
        },
        Err(_) => Ok((None, json!({}))),
    }
}

fn obj<'a>(v: &'a mut Value, k: &str) -> &'a mut Map<String, Value> {
    if !v[k].is_object() {
        v[k] = json!({});
    }
    v[k].as_object_mut().unwrap()
}

/// What `setup` would write into `home` (nothing is written yet: `apply` does that).
pub fn setup(name: &str, t: &Target, home: &Path) -> Result<Vec<Change>, String> {
    let a = find(name).ok_or_else(|| format!("no agent called {name:?} (am2strata agent list)"))?;
    let json_change = |path: PathBuf, f: &dyn Fn(&mut Value)| -> Result<Change, String> {
        let (before, mut v) = read_json(&path)?;
        f(&mut v);
        Ok(Change { path, before, after: serde_json::to_string_pretty(&v).unwrap() + "\n" })
    };
    match a.name {
        "claude" => Ok(vec![json_change(home.join(".claude/settings.json"), &|v| {
            let env = obj(v, "env");
            env.insert("ANTHROPIC_BASE_URL".into(), json!(t.root()));
            env.insert("ANTHROPIC_MODEL".into(), json!(t.model));
            env.insert("ANTHROPIC_DEFAULT_OPUS_MODEL".into(), json!(t.model));
            env.insert("ANTHROPIC_DEFAULT_SONNET_MODEL".into(), json!(t.model));
            env.insert("ANTHROPIC_DEFAULT_HAIKU_MODEL".into(), json!(t.small()));
            env.insert("CLAUDE_CODE_SUBAGENT_MODEL".into(), json!(t.model));
            if t.key.is_some() {
                // the key comes from the shell at each start, not from this file
                env.remove("ANTHROPIC_AUTH_TOKEN");
                v["apiKeyHelper"] = json!(format!("printf %s \"${KEY_VAR}\""));
            } else {
                env.insert("ANTHROPIC_AUTH_TOKEN".into(), json!("am2strata-no-key"));
            }
        })?]),
        "codex" => {
            let path = home.join(".codex/config.toml");
            let before = std::fs::read_to_string(&path).ok();
            let after = codex_toml(before.as_deref().unwrap_or(""), t);
            Ok(vec![Change { path, before, after }])
        }
        "opencode" => {
            let dir = home.join(".config/opencode");
            if dir.join("opencode.jsonc").exists() {
                return Err(format!("{} is JSONC, so it is left alone; `agent show opencode` gives the block to add", dir.join("opencode.jsonc").display()));
            }
            let c = opencode_config(t);
            Ok(vec![json_change(dir.join("opencode.json"), &|v| {
                obj(v, "provider").insert(PROVIDER.into(), c["provider"][PROVIDER].clone());
                v["model"] = c["model"].clone();
                v["small_model"] = c["small_model"].clone();
                if v.get("$schema").is_none() {
                    v["$schema"] = c["$schema"].clone();
                }
            })?])
        }
        "pi" => Ok(vec![json_change(home.join(".pi/agent/models.json"), &|v| {
            obj(v, "providers").insert(PROVIDER.into(), pi_provider(t));
        })?]),
        "droid" => Ok(vec![json_change(home.join(".factory/settings.json"), &|v| {
            let mut list: Vec<Value> = v["customModels"].as_array().cloned().unwrap_or_default();
            list.retain(|m| !m["displayName"].as_str().is_some_and(|d| d.starts_with("am2Strata: ")));
            let mut add = |m: &str| {
                let mut e = json!({"model": m, "displayName": format!("am2Strata: {m}"), "baseUrl": t.root(), "provider": "anthropic"});
                if t.key.is_some() {
                    e["apiKey"] = json!(format!("${{{KEY_VAR}}}"));
                }
                list.push(e);
            };
            add(&t.model);
            if let Some(s) = t.small.as_deref().filter(|s| *s != t.model) {
                add(s);
            }
            v["customModels"] = Value::Array(list);
        })?]),
        _ => Err(match a.support {
            Support::No(why) => format!("{} is not supported: {why}", a.label),
            Support::Start => format!("{} is set up through variables: `am2strata agentrun {}` runs it, and `agent show {}` gives them for a shell profile",
                                      a.label, a.name, a.name),
            _ => format!("{} is set up in its own settings: `am2strata agent show {}` says what to enter", a.label, a.name),
        }),
    }
}

const BEGIN: &str = "# >>> am2strata (am2strata agent setup codex; agent reset codex removes it)";
const END: &str = "# <<< am2strata";

/// Codex's config.toml with am2Strata as the default provider. Top-level keys must come before the first table, so
/// the block goes first; a `model` or `model_provider` the user had is commented out (reset brings it back).
fn codex_toml(old: &str, t: &Target) -> String {
    let mut rest = String::new();
    let mut inside = false;
    let mut in_table = false;
    for line in old.lines() {
        if line.starts_with(BEGIN) {
            inside = true;
            continue;
        }
        if inside {
            if line.starts_with(END) {
                inside = false;
            }
            continue;
        }
        let l = line.trim_start();
        if l.starts_with('[') {
            in_table = true;
        }
        let top_key = !in_table && (l.starts_with("model ") || l.starts_with("model=") || l.starts_with("model_provider"));
        if top_key {
            rest.push_str(&format!("# am2strata: {line}\n"));
        } else if in_table && l.starts_with(&format!("[model_providers.{PROVIDER}]")) {
            rest.push_str(&format!("# am2strata (replaced by the block above): {line}\n"));
        } else {
            rest.push_str(line);
            rest.push('\n');
        }
    }
    format!("{BEGIN}\nmodel_provider = \"{PROVIDER}\"\nmodel = \"{}\"\nmodel_providers.{PROVIDER} = {{ {} }}\n{END}\n{rest}",
            t.model, codex_provider_toml(t))
}

/// Writes the changes, each file backed up first (the first backup is kept: it is the user's own file).
pub fn apply(changes: &[Change]) -> Result<(), String> {
    for c in changes {
        if let Some(dir) = c.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let backup = backup_path(&c.path);
        if !backup.exists() {
            // no file before: the backup says so, and reset removes the file
            std::fs::write(&backup, c.before.as_deref().unwrap_or("")).map_err(|e| format!("{}: {e}", backup.display()))?;
            if c.before.is_none() {
                std::fs::write(backup.with_extension(format!("{BACKUP}-none")), "").map_err(|e| e.to_string())?;
            }
        }
        std::fs::write(&c.path, &c.after).map_err(|e| format!("{}: {e}", c.path.display()))?;
    }
    Ok(())
}

fn backup_path(p: &Path) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(format!(".{BACKUP}"));
    PathBuf::from(s)
}

/// The files `setup` wrote for `name` put back as they were; the paths restored.
pub fn reset(name: &str, home: &Path) -> Result<Vec<PathBuf>, String> {
    let a = find(name).ok_or_else(|| format!("no agent called {name:?}"))?;
    let files: Vec<PathBuf> = match a.name {
        "claude" => vec![home.join(".claude/settings.json")],
        "codex" => vec![home.join(".codex/config.toml")],
        "opencode" => vec![home.join(".config/opencode/opencode.json")],
        "pi" => vec![home.join(".pi/agent/models.json")],
        "droid" => vec![home.join(".factory/settings.json")],
        _ => return Err(format!("{} has nothing set up by am2strata", a.label)),
    };
    let mut done = Vec::new();
    for f in files {
        let b = backup_path(&f);
        let none = b.with_extension(format!("{BACKUP}-none"));
        if !b.exists() {
            continue;
        }
        if none.exists() {
            let _ = std::fs::remove_file(&f);
            let _ = std::fs::remove_file(&none);
        } else {
            std::fs::copy(&b, &f).map_err(|e| format!("{}: {e}", f.display()))?;
        }
        let _ = std::fs::remove_file(&b);
        done.push(f);
    }
    Ok(done)
}

/// What to enter, for any agent: the Connect page's text and `agent show`. The key is never in it.
pub fn guide(name: &str, t: &Target) -> Result<String, String> {
    let a = find(name).ok_or_else(|| format!("no agent called {name:?}"))?;
    let key = if t.key.is_some() { format!("${KEY_VAR}") } else { "(none: this server has no API key)".into() };
    let cmd = |s: &str| if s == "start" { format!("am2strata agentrun {}{}", a.name, remote_flags(t)) }
                        else { format!("am2strata agent {s} {}{}", a.name, remote_flags(t)) };
    Ok(match a.name {
        "claude" | "codex" | "opencode" | "pi" => format!(
            "For one run (nothing of yours changes):\n  {}\nAs the default from now on (your config is backed up; `agent reset {}` restores it):\n  {}\n{}",
            cmd("start"), a.name, cmd("setup"),
            if t.key.is_some() { format!("Both read the key from {KEY_VAR} in your shell.") } else { String::new() }),
        "droid" => format!("{}\nThen pick \"am2Strata: {}\" with /model in droid.", cmd("setup"), t.model),
        "qwen" | "aider" | "goose" | "gemini" => {
            let p = start(a.name, t, Path::new("/tmp")).unwrap();
            let vars: Vec<String> = p.env.iter().filter(|(k, _)| k != KEY_VAR)
                .map(|(k, v)| if k.contains("KEY") || k.contains("TOKEN") { format!("export {k}={key}") } else { format!("export {k}={v}") }).collect();
            format!("For one run:\n  {}\nOr in your shell profile:\n  {}\n  {}{}", cmd("start"), vars.join("\n  "), a.exe,
                    if p.args.is_empty() { String::new() } else { format!(" {}", p.args.join(" ")) })
        }
        "omp" => format!("In ~/.omp/agent/models.yml:\nproviders:\n  am2strata:\n    baseUrl: {}\n    api: openai-completions\n    apiKey: {}\n    models:\n      - id: {}\nThen choose am2strata/{} in omp.", t.v1(), key, t.model, t.model),
        "crush" => format!("In ~/.config/crush/crush.json, under \"providers\":\n\"am2strata\": {{ \"type\": \"openai\", \"base_url\": \"{}\", \"api_key\": \"{}\", \"models\": [{{ \"id\": \"{}\", \"name\": \"{}\" }}] }}", t.v1(), key, t.model, t.model),
        "continue" => format!("In ~/.continue/config.yaml, under models:\n  - name: am2Strata\n    provider: openai\n    model: {}\n    apiBase: {}\n    apiKey: {}", t.model, t.v1(), key),
        "zed" => format!("In Zed's settings.json:\n\"language_models\": {{ \"openai_compatible\": {{ \"am2Strata\": {{ \"api_url\": \"{}\", \"available_models\": [{{ \"name\": \"{}\", \"max_tokens\": 131072 }}] }} }} }}\nZed asks for the key ({key}) the first time.", t.v1(), t.model),
        "hermes" | "openclaw" | "cline" | "kilocode" | "vscode" => format!(
            "In {}'s model settings, add an OpenAI-compatible provider:\n  Base URL: {}\n  API key:  {}\n  Model:    {}", a.label, t.v1(), key, t.model),
        _ => match a.support {
            Support::No(why) => format!("Not supported: {why}."),
            _ => String::new(),
        },
    })
}

fn remote_flags(t: &Target) -> String {
    let local = t.url.contains("://127.0.0.1") || t.url.contains("://localhost");
    let mut s = if local { String::new() } else { format!(" --url {}", t.url) };
    s.push_str(&format!(" --model {}", t.model));
    s
}

/// GET /api/agents: every agent with its support and guide, for the Connect page.
pub fn list(t: &Target) -> Value {
    Value::Array(AGENTS.iter().map(|a| {
        let (support, why) = match a.support {
            Support::Full => ("start, setup", None),
            Support::Start => ("start", None),
            Support::Setup => ("setup", None),
            Support::Guide => ("guide", None),
            Support::No(w) => ("not supported", Some(w)),
        };
        json!({"name": a.name, "label": a.label, "exe": a.exe, "api": a.api, "support": support, "why": why,
               "checked": a.checked, "guide": guide(a.name, t).unwrap_or_default()})
    }).collect())
}

#[cfg(test)]
mod tools_header_tests {
    use super::*;

    #[test]
    fn tools_become_the_x_strata_tools_header() {
        let t = Target { url: "http://h:8000".into(), key: Some("k".into()), model: "m".into(), small: None, tools: Some("web,image".into()) };
        let tmp = std::env::temp_dir();
        let c = start("claude", &t, &tmp).unwrap();
        assert!(c.env.contains(&("ANTHROPIC_CUSTOM_HEADERS".into(), "x-strata-tools: web,image".into())));
        let x = start("codex", &t, &tmp).unwrap();
        assert!(x.args.iter().any(|a| a.contains(r#"http_headers = { "x-strata-tools" = "web,image" }"#)), "{:?}", x.args);
        let o = start("opencode", &t, &tmp).unwrap();
        assert!(o.env.iter().any(|(k, v)| k == "OPENCODE_CONFIG_CONTENT" && v.contains(r#""x-strata-tools":"web,image""#)));
        let q = start("qwen", &t, &tmp).unwrap();
        assert!(q.notes.iter().any(|n| n.contains("no extra headers")));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(key: bool) -> Target {
        Target { url: "http://10.0.0.5:8000".into(), key: key.then(|| "sk-test".into()), model: "tiel".into(), small: Some("lfm2".into()), tools: None }
    }

    #[test]
    fn claude_runs_on_the_server_with_its_model_tiers() {
        let p = start("claude", &t(true), Path::new("/tmp")).unwrap();
        let env: std::collections::HashMap<_, _> = p.env.iter().cloned().collect();
        assert_eq!(env["ANTHROPIC_BASE_URL"], "http://10.0.0.5:8000");
        assert_eq!((env["ANTHROPIC_DEFAULT_OPUS_MODEL"].as_str(), env["ANTHROPIC_DEFAULT_HAIKU_MODEL"].as_str()), ("tiel", "lfm2"));
        assert_eq!(env["ANTHROPIC_AUTH_TOKEN"], "sk-test");
    }

    #[test]
    fn codex_gets_a_provider_for_the_run_and_a_block_in_its_config() {
        let p = start("codex", &t(true), Path::new("/tmp")).unwrap();
        assert!(p.args.contains(&"model_providers.am2strata={ name = \"am2Strata\", base_url = \"http://10.0.0.5:8000/v1\", wire_api = \"responses\", env_key = \"AM2STRATA_API_KEY\" }".to_string()), "{:?}", p.args);
        let old = "model = \"gpt-5\"\napproval_policy = \"never\"\n\n[model_providers.am2strata]\nname = \"old\"\n";
        let new = codex_toml(old, &t(true));
        assert!(new.starts_with(BEGIN) && new.contains("model = \"tiel\"") && new.contains("# am2strata: model = \"gpt-5\""), "{new}");
        assert!(new.contains("approval_policy = \"never\""));
        // again: one block, not two
        let again = codex_toml(&new, &t(true));
        assert_eq!(again.matches(BEGIN).count(), 1, "{again}");
    }

    #[test]
    fn setup_backs_up_and_reset_restores() {
        let home = std::env::temp_dir().join(format!("am2-agents-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(home.join(".claude")).unwrap();
        std::fs::write(home.join(".claude/settings.json"), "{\"theme\": \"dark\", \"env\": {\"FOO\": \"1\"}}").unwrap();
        apply(&setup("claude", &t(true), &home).unwrap()).unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(home.join(".claude/settings.json")).unwrap()).unwrap();
        assert_eq!((v["theme"].as_str(), v["env"]["FOO"].as_str(), v["env"]["ANTHROPIC_MODEL"].as_str()), (Some("dark"), Some("1"), Some("tiel")));
        assert!(v["env"].get("ANTHROPIC_AUTH_TOKEN").is_none() && v["apiKeyHelper"].as_str().unwrap().contains(KEY_VAR), "the key stays out: {v}");
        // a file that was not there goes again
        apply(&setup("pi", &t(false), &home).unwrap()).unwrap();
        assert!(home.join(".pi/agent/models.json").exists());
        assert_eq!(reset("claude", &home).unwrap().len(), 1);
        assert_eq!(std::fs::read_to_string(home.join(".claude/settings.json")).unwrap(), "{\"theme\": \"dark\", \"env\": {\"FOO\": \"1\"}}");
        reset("pi", &home).unwrap();
        assert!(!home.join(".pi/agent/models.json").exists());
        // JSONC is left alone (not checked on Windows, where the runner refused this scratch file just after making its
        // folder; the JSONC rule itself is the same code on every system)
        #[cfg(not(windows))]
        {
            std::fs::create_dir_all(home.join(".config/opencode")).unwrap();
            std::fs::write(home.join(".config/opencode/opencode.json"), "{ // mine\n}").unwrap();
            assert!(setup("opencode", &t(true), &home).unwrap_err().contains("not plain JSON"));
        }
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn every_agent_has_a_guide_and_the_unreachable_ones_say_why() {
        for a in AGENTS {
            let g = guide(a.name, &t(true)).unwrap();
            assert!(!g.is_empty(), "{}", a.name);
            assert!(!g.contains("sk-test"), "{}: the key is never shown", a.name);
        }
        assert!(start("agy", &t(true), Path::new("/tmp")).unwrap_err().contains("not supported"));
        assert_eq!(find("antigravity").unwrap().name, "agy");
    }
}
