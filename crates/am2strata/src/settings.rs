//! Settings: the Chat settings shared with other apps (GET/POST /settings, kept as
//! strata-<model>.shared-settings.json next to the run config), the run config's `sampling` block and
//! `reasoning_effort`, and the web page's view of a few documented run-config keys (GET/POST /config).
//! Port of serve/server.py clean_shared_defaults / with_shared / sampling_defaults_from_config and serve/runconfig.py.

use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

pub type Defaults = Map<String, Value>;

fn number(v: &Value) -> Option<f64> {
    if v.is_boolean() {
        return None;
    }
    v.as_f64()
}

fn whole(v: &Value) -> Option<i64> {
    number(v).filter(|f| f.fract() == 0.0).map(|f| f as i64)
}

/// The Chat settings other apps get: only known keys, each checked; Err names a bad one.
pub fn clean_shared_defaults(d: &Value) -> Result<Defaults, String> {
    let obj = match d {
        Value::Null => return Ok(Defaults::new()),
        Value::Object(o) => o,
        _ => return Err("defaults must be an object".into()),
    };
    let mut out = Defaults::new();
    for (key, value) in obj {
        if value.is_null() || value.as_str() == Some("") {
            continue;
        }
        let v = match key.as_str() {
            "reasoning_effort" => match value.as_str() {
                Some(e @ ("none" | "low" | "medium" | "high")) => json!(e),
                _ => return Err("reasoning_effort: none, low, medium or high".into()),
            },
            "temperature" => match number(value).filter(|t| (0.0..=2.0).contains(t)) {
                Some(t) => json!(t),
                None => return Err("temperature: 0..2".into()),
            },
            "top_p" => match number(value).filter(|t| *t > 0.0 && *t <= 1.0) {
                Some(t) => json!(t),
                None => return Err("top_p: 0 < top_p <= 1".into()),
            },
            "top_k" => match whole(value).filter(|k| (1..=64).contains(k)) {
                Some(k) => json!(k),
                None => return Err("top_k: an integer 1..64".into()),
            },
            "seed" | "max_tokens" => match whole(value).filter(|k| *k > 0) {
                Some(k) => json!(k),
                None => return Err(format!("{key}: a positive integer")),
            },
            "reasoning_budget_tokens" => match whole(value).filter(|k| *k >= 0) {
                Some(k) => json!(k),
                None => return Err("reasoning_budget_tokens: a whole number of tokens (0: no budget)".into()),
            },
            "experimental_speed_projection" => match value {
                Value::Bool(_) => value.clone(),
                _ => return Err("experimental_speed_projection: true or false".into()),
            },
            _ => return Err(format!("unknown setting '{key}'")),
        };
        out.insert(key.clone(), v);
    }
    Ok(out)
}

/// strata-<model>.json -> strata-<model>.shared-settings.json
pub fn shared_path(config: &Path) -> PathBuf {
    config.with_extension("shared-settings.json")
}

/// The shared settings saved the last time this model ran (none when missing or unreadable).
pub fn read_shared(config: &Path) -> Defaults {
    let Ok(text) = std::fs::read_to_string(shared_path(config)) else { return Defaults::new() };
    let d = serde_json::from_str::<Value>(&text).ok().and_then(|v| clean_shared_defaults(&v).ok()).unwrap_or_default();
    if !d.is_empty() {
        let list: Vec<String> = d.iter().map(|(k, v)| format!("{k}={v}")).collect();
        tracing::info!("other apps use the Chat settings: {}", list.join(", "));
    }
    d
}

/// Keep (or, empty, remove) the shared settings next to the run config.
pub fn write_shared(config: &Path, d: &Defaults) {
    let p = shared_path(config);
    let r = if d.is_empty() {
        match std::fs::remove_file(&p) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            other => other,
        }
    } else {
        std::fs::write(&p, serde_json::to_string_pretty(&Value::Object(d.clone())).unwrap_or_default())
    };
    if let Err(e) = r {
        tracing::warn!("could not save the shared settings: {e}");
    }
}

const SAMPLING_KEYS: [&str; 9] =
    ["temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "frequency_penalty", "penalty_last_n", "seed"];

/// The run config's `sampling` block, checked (llama.cpp's names repeat_penalty / repeat_last_n accepted).
pub fn config_sampling(cfg_rest: &std::collections::BTreeMap<String, Value>) -> Result<Defaults, String> {
    let mut out = Defaults::new();
    let Some(block) = cfg_rest.get("sampling").and_then(Value::as_object) else { return Ok(out) };
    for (k, v) in block {
        if v.is_null() {
            continue;
        }
        let key = match k.as_str() {
            "repeat_penalty" => "repetition_penalty",
            "repeat_last_n" => "penalty_last_n",
            k => k,
        };
        if !SAMPLING_KEYS.contains(&key) {
            tracing::warn!("config sampling: unknown key {k:?} ignored");
            continue;
        }
        if out.contains_key(key) {
            return Err(format!("config sampling: both {k:?} and {key:?} are set; they are the same setting, keep one"));
        }
        let ok = match key {
            "temperature" => number(v).is_some_and(|x| x >= 0.0),
            "top_p" => number(v).is_some_and(|x| x > 0.0 && x <= 1.0),
            "min_p" => number(v).is_some_and(|x| (0.0..=1.0).contains(&x)),
            "top_k" | "penalty_last_n" | "seed" => whole(v).is_some(),
            _ => number(v).is_some(),
        };
        if !ok {
            return Err(format!("config sampling.{k}={v}: not a valid value"));
        }
        out.insert(key.to_string(), v.clone());
    }
    Ok(out)
}

/// The request with the shared max tokens and thinking level filled in where it has none of its own.
/// `api`: "openai" (also the Responses API, as its chat request) or "anthropic".
pub fn with_shared(req: &mut Value, shared: &Defaults, config_effort: Option<&str>, api: &str) {
    let effort = shared.get("reasoning_effort").and_then(Value::as_str).or(config_effort);
    fn given(req: &Value, k: &str) -> bool {
        req.get(k).is_some_and(|v| !v.is_null() && v.as_str() != Some("") && v != &json!(0) && v != &json!(false))
    }
    if let Some(n) = shared.get("max_tokens") {
        if !given(req, "max_tokens") && !given(req, "max_completion_tokens") {
            req["max_tokens"] = n.clone();
        }
    }
    let Some(effort) = effort else { return };
    if api == "openai" {
        let ctk = req.get("chat_template_kwargs").and_then(Value::as_object);
        let in_ctk = ctk.is_some_and(|c| c.contains_key("enable_thinking") || c.contains_key("reasoning_effort"));
        let reasoning = req.get("reasoning").is_some_and(|r| r.as_object().is_some_and(|o| !o.is_empty()));
        if !given(req, "reasoning_effort") && !reasoning && !in_ctk {
            req["reasoning_effort"] = json!(effort);
        }
    } else if !given(req, "thinking") && !given(req, "output_config") {
        if effort == "none" {
            req["thinking"] = json!({"type": "disabled"});
        } else {
            req["thinking"] = json!({"type": "adaptive"});
            req["output_config"] = json!({"effort": effort});
        }
    }
}

// ------------------------------------------------------------------------------------------------ GET/POST /config

#[derive(Clone, Copy)]
enum Kind {
    Bool,
    IntGe0,
    NumGe0,
    Enum(&'static [&'static str]),
    Names,
    /// a key of the "sampling" block, with its rule
    Sampling(&'static str),
    /// an engine option kept in "args"
    Arg(&'static str),
}

const EDITABLE: [(&str, Kind, &str); 18] = [
    ("sampling.temperature", Kind::Sampling("num>=0"),
     "Default temperature for requests that send none (0 = greedy; without it the model card's: 1.0 thinking, 0.7 not)"),
    ("sampling.top_p", Kind::Sampling("0<x<=1"), "Default top_p for requests that send none"),
    ("sampling.top_k", Kind::Sampling("1..64"), "Default top_k for requests that send none (1-64)"),
    ("sampling.min_p", Kind::Sampling("0<=x<=1"), "Default min_p for requests that send none"),
    ("reasoning_budget_tokens", Kind::IntGe0, "Cap the thinking of every request at this many tokens (0 or empty: no cap)"),
    ("fit_max_tokens", Kind::Bool, "Shorten a max_tokens that does not fit the context instead of answering 400"),
    ("drop_oldest", Kind::Bool, "Leave out a conversation's oldest turns when it no longer fits the context, instead of \
                                 answering 400 (the system prompt and the last turn stay)"),
    ("preserve_thinking", Kind::Bool, "Keep the thinking of earlier turns in the prompt (templates that support it: \
                                       Qwen3.6, Ornith, Tiel Coder)"),
    ("anthropic_thinking", Kind::Enum(&["model", "on_request"]),
     "Anthropic requests that do not ask for thinking: think as the model does (model) or not (on_request)"),
    ("effort_position", Kind::Enum(&["start", "end"]),
     "Where a non-default reasoning effort goes: start (the default) or end (keeps the cache when it changes)"),
    ("aliases", Kind::Names, "Other model names the server lists and answers to (comma-separated)"),
    ("idle_unload_s", Kind::NumGe0, "Unload the model after this many seconds without requests (0 or empty: never)"),
    ("lazy_load", Kind::Bool, "Start without loading the model; the first request loads it (text only)"),
    ("engine_silence_s", Kind::NumGe0, "End a request when the engine says nothing for this long (default 300 s, 0 = wait)"),
    ("api_monitor", Kind::Bool, "Keep the last 100 requests' prompts and answers in memory for /api-monitor"),
    ("switch_on_request", Kind::Bool, "List every installed model in /v1/models and load the one a request names, the \
                                       loaded one stopped first (#38; read from the config the server starts with)"),
    ("open_browser", Kind::Bool, "Open the chat page in the browser when the model is ready"),
    ("vram_reserve_mib", Kind::Arg("--vram-reserve-mib"), "VRAM in MiB the engine leaves free for other programs (engine default 700)"),
];

fn kind_of(key: &str) -> Option<Kind> {
    EDITABLE.iter().find(|(k, _, _)| *k == key).map(|(_, kind, _)| *kind)
}

fn arg_value(cfg: &Value, flag: &str) -> Value {
    let a = cfg.get("args").and_then(Value::as_array).cloned().unwrap_or_default();
    let n = a.len();
    match a.iter().take(n.saturating_sub(1)).position(|x| x.as_str() == Some(flag)) {
        Some(i) => {
            let v = a[i + 1].as_str().map(str::to_owned).unwrap_or_else(|| a[i + 1].to_string());
            v.parse::<i64>().map(|n| json!(n)).unwrap_or(json!(v))
        }
        None => Value::Null,
    }
}

fn value_of(cfg: &Value, key: &str) -> Value {
    match kind_of(key) {
        Some(Kind::Sampling(_)) => cfg.get("sampling").and_then(|s| s.get(&key[9..])).cloned().unwrap_or(Value::Null),
        Some(Kind::Arg(flag)) => arg_value(cfg, flag),
        _ => cfg.get(key).cloned().unwrap_or(Value::Null),
    }
}

/// GET /config: the editable keys with their values (null: not set, the default applies).
pub fn view(cfg: &Value, path: &Path) -> Value {
    let keys: Vec<Value> = EDITABLE
        .iter()
        .map(|(key, kind, help)| {
            let (k, choices) = match kind {
                Kind::Bool => ("bool", None),
                Kind::Enum(c) => ("enum", Some(c.to_vec())),
                Kind::Names => ("names", None),
                _ => ("number", None),
            };
            let mut o = json!({"key": key, "value": value_of(cfg, key), "help": help, "kind": k});
            if let Some(c) = choices {
                o["choices"] = json!(c);
            }
            o
        })
        .collect();
    json!({"file": path.file_name().and_then(|n| n.to_str()).unwrap_or(""), "keys": keys,
           "note": "Saved to the run config; used from the next start of, or switch to, the model."})
}

fn num_in(key: &str, v: &Value, whole_only: bool, lo: f64, hi: Option<f64>, lo_open: bool) -> Result<Value, String> {
    let ok = number(v).filter(|x| {
        (!whole_only || x.fract() == 0.0) && (if lo_open { *x > lo } else { *x >= lo }) && hi.is_none_or(|h| *x <= h)
    });
    match ok {
        Some(x) if whole_only => Ok(json!(x as i64)),
        Some(x) => Ok(json!(x)),
        None => {
            let rng = format!("{} {lo}", if lo_open { "more than" } else { "at least" })
                + &hi.map(|h| format!(" and at most {h}")).unwrap_or_default();
            Err(format!("{key}: expected a {}number, {rng}, not {v}", if whole_only { "whole " } else { "" }))
        }
    }
}

/// The value to store for `key` (null removes the key: its default), or why not.
fn check(key: &str, v: &Value, cfg: &Value) -> Result<Value, String> {
    let Some(kind) = kind_of(key) else {
        return Err(format!("'{key}' cannot be changed here (only the keys the Settings view lists)"));
    };
    if v.is_null() {
        return Ok(Value::Null);
    }
    match kind {
        Kind::Bool => {
            if !v.is_boolean() {
                return Err(format!("{key}: expected true or false, not {v}"));
            }
            if key == "lazy_load" && v == &json!(true) && cfg.get("vision").is_some_and(|x| !x.is_null()) {
                return Err("lazy_load: lazy loading is text-only, and this model reads images (\"vision\")".into());
            }
            Ok(v.clone())
        }
        Kind::Enum(choices) => match v.as_str() {
            Some(s) if choices.contains(&s) => Ok(v.clone()),
            _ => Err(format!("{key}: expected one of {}, not {v}", choices.join(", "))),
        },
        Kind::Names => {
            let list: Vec<Value> = match v {
                Value::String(s) => s.split(',').map(str::trim).filter(|x| !x.is_empty()).map(|x| json!(x)).collect(),
                Value::Array(a) => a.clone(),
                _ => return Err(format!("{key}: expected a list of model names, not {v}")),
            };
            if !list.iter().all(|x| x.as_str().is_some_and(|s| !s.trim().is_empty() && s.len() <= 128)) {
                return Err(format!("{key}: expected a list of model names, not {v}"));
            }
            Ok(if list.is_empty() { Value::Null } else { Value::Array(list) })
        }
        Kind::IntGe0 | Kind::Arg(_) => num_in(key, v, true, 0.0, None, false),
        Kind::NumGe0 => num_in(key, v, false, 0.0, None, false),
        Kind::Sampling(rule) => match rule {
            "num>=0" => num_in(key, v, false, 0.0, None, false),
            "0<x<=1" => num_in(key, v, false, 0.0, Some(1.0), true),
            "0<=x<=1" => num_in(key, v, false, 0.0, Some(1.0), false),
            _ => num_in(key, v, true, 1.0, Some(64.0), false),
        },
    }
}

/// The config with `changes` made, and the keys that changed. Every value is checked first, so one bad value
/// changes nothing.
pub fn apply(cfg: &Value, changes: Option<&Value>) -> Result<(Value, Vec<String>), String> {
    let Some(changes) = changes.and_then(Value::as_object).filter(|c| !c.is_empty()) else {
        return Err("send {\"set\": {\"<key>\": <value or null>, ...}}".into());
    };
    let mut checked = Vec::new();
    for (k, v) in changes {
        checked.push((k.clone(), check(k, v, cfg)?));
    }
    let mut new = cfg.clone();
    let mut changed = Vec::new();
    for (key, v) in checked {
        if value_of(&new, &key) == v {
            continue;
        }
        let obj = new.as_object_mut().ok_or("the run config is not a JSON object")?;
        match kind_of(&key).unwrap() {
            Kind::Sampling(_) => {
                let mut s = obj.get("sampling").and_then(Value::as_object).cloned().unwrap_or_default();
                if v.is_null() {
                    s.remove(&key[9..]);
                } else {
                    s.insert(key[9..].to_string(), v);
                }
                if s.is_empty() {
                    obj.remove("sampling");
                } else {
                    obj.insert("sampling".into(), Value::Object(s));
                }
            }
            Kind::Arg(flag) => {
                let mut a = obj.get("args").and_then(Value::as_array).cloned().unwrap_or_default();
                let n = a.len();
                match a.iter().take(n.saturating_sub(1)).position(|x| x.as_str() == Some(flag)) {
                    Some(i) if v.is_null() => {
                        a.drain(i..i + 2);
                    }
                    Some(i) => a[i + 1] = json!(v.to_string()),
                    None if !v.is_null() => {
                        a.push(json!(flag));
                        a.push(json!(v.to_string()));
                    }
                    None => {}
                }
                obj.insert("args".into(), Value::Array(a));
            }
            _ if v.is_null() => {
                obj.remove(&key);
            }
            _ => {
                obj.insert(key.clone(), v);
            }
        }
        changed.push(key);
    }
    Ok((new, changed))
}

/// The config written whole (a temporary file moved over the old one), the earlier one kept as <name>.bak.
pub fn save(path: &Path, cfg: &Value) -> std::io::Result<PathBuf> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("config.json");
    let bak = path.with_file_name(format!("{name}.bak"));
    std::fs::copy(path, &bak)?;
    let tmp = path.with_file_name(format!("{name}.tmp"));
    std::fs::write(&tmp, serde_json::to_string_pretty(cfg).unwrap_or_default())?;
    std::fs::rename(&tmp, path)?;
    Ok(bak)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_defaults_are_checked() {
        let d = clean_shared_defaults(&json!({"reasoning_effort": "low", "temperature": 0.6, "top_k": 20, "max_tokens": 4096, "seed": null})).unwrap();
        assert_eq!(Value::Object(d), json!({"reasoning_effort": "low", "temperature": 0.6, "top_k": 20, "max_tokens": 4096}));
        assert!(clean_shared_defaults(&json!({"top_k": 65})).is_err());
        assert!(clean_shared_defaults(&json!({"colour": "red"})).is_err());
    }

    #[test]
    fn shared_fills_only_what_the_request_leaves_out() {
        let s = clean_shared_defaults(&json!({"reasoning_effort": "none", "max_tokens": 100})).unwrap();
        let mut a = json!({"messages": []});
        with_shared(&mut a, &s, None, "openai");
        assert_eq!(a["max_tokens"], 100);
        assert_eq!(a["reasoning_effort"], "none");
        let mut b = json!({"max_tokens": 5, "reasoning_effort": "high"});
        with_shared(&mut b, &s, None, "openai");
        assert_eq!(b, json!({"max_tokens": 5, "reasoning_effort": "high"}));
        let mut c = json!({});
        with_shared(&mut c, &s, None, "anthropic");
        assert_eq!(c["thinking"], json!({"type": "disabled"}));
    }

    #[test]
    fn config_changes_touch_only_their_keys() {
        let cfg = json!({"args": ["--model", "m.gguf", "--vram-reserve-mib", "964"], "api_key": "k", "sampling": {"top_p": 0.9}});
        let (new, changed) = apply(&cfg, Some(&json!({"sampling.temperature": 0.6, "sampling.top_p": null,
                                                         "vram_reserve_mib": 700, "fit_max_tokens": true}))).unwrap();
        assert_eq!(changed.len(), 4);
        assert_eq!(new["sampling"], json!({"temperature": 0.6}));
        assert_eq!(new["args"], json!(["--model", "m.gguf", "--vram-reserve-mib", "700"]));
        assert_eq!(new["api_key"], "k");
        assert!(apply(&cfg, Some(&json!({"api_key": "x"}))).is_err(), "network keys are not editable");
        assert!(apply(&cfg, Some(&json!({"sampling.top_k": 0, "fit_max_tokens": true}))).is_err(), "one bad value changes nothing");
        let v = view(&new, Path::new("/x/strata-m.json"));
        assert_eq!(v["file"], "strata-m.json");
        assert_eq!(v["keys"][0]["value"], 0.6);
    }
}
