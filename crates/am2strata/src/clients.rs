//! Client profiles (docs/SERVER-TOOLS-PLAN.md, part 1): a client's configuration kept on the server, picked by its
//! API key, so an agent that can send only a URL and a key (`unsloth start`, Codex, Claude Code) gets the server's
//! tools with no request flag at all.
//!
//! The run config:
//! ```json
//! "clients": {
//!   "laptop":  {"key": "...", "tools": ["memory", "web", "image", "speech", "mcp:*"], "memory": "laptop"},
//!   "default": {"tools": ["web"]}
//! }
//! ```
//! - `tools`: memory, web, image, video, speech, music, browser (the browser on this PC), code (run_python in the
//!   sandbox), and MCP servers by name (`mcp:search`) or every address
//!   server (`mcp:*`). A program (stdio) MCP server runs on this PC, so a profile gets it only by name, never by `*`.
//! - `memory`: the profile's memory space (default: the profile's name); two profiles naming the same space share it.
//! - `default` (no key): what the main key gets; without it, the main key gets only what its requests ask for.
//!
//! A client key opens the model APIs (chat, messages, responses, Gemini, media), not the server's settings. One request
//! may narrow its set with `x-strata-tools: web,image`; naming a set the profile lacks is refused.

use serde_json::Value;
use std::collections::BTreeSet;

pub const SETS: [&str; 8] = ["memory", "web", "image", "video", "speech", "music", "browser", "code"];

#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    pub name: String,
    key: Option<String>,
    /// the tool sets: SETS and "mcp:<server>" / "mcp:*"
    pub tools: BTreeSet<String>,
    pub memory: Option<String>,
    /// "limits": {"media_jobs_per_hour": N}: media jobs this key's chats may queue in an hour
    pub media_jobs_per_hour: Option<u32>,
    /// "media_in_turn": "defer" (default: an image or video is made after the answer) or "swap" (the answer waits
    /// for it: the text model stops, the job runs, the model loads again and the answer goes on with the result)
    pub media_in_turn: Option<String>,
    /// "limits": {"max_image_size": "1024x1024"}: the most pixels an image this key asks for may have
    pub max_image_pixels: Option<u64>,
    /// "limits": {"max_video_seconds": N}: the longest video this key may ask for
    pub max_video_seconds: Option<f64>,
    /// "limits": {"web_queries_per_turn": N}: web searches one answer may run for this key
    pub web_queries_per_turn: Option<u32>,
}

impl Profile {
    /// The memory space of this profile's requests.
    pub fn memory_space(&self) -> String {
        self.memory.clone().unwrap_or_else(|| self.name.clone())
    }

    pub fn has(&self, set: &str) -> bool {
        self.tools.contains(set)
    }

    /// The MCP servers it may use: names, and whether every address server ("mcp:*").
    pub fn mcp(&self) -> (Vec<String>, bool) {
        let names = self.tools.iter().filter_map(|t| t.strip_prefix("mcp:")).filter(|n| *n != "*").map(str::to_owned).collect();
        (names, self.tools.contains("mcp:*"))
    }

    /// This profile narrowed to `asked` (the x-strata-tools header); Err names a set it does not have.
    pub fn narrowed(&self, asked: &str) -> Result<Profile, String> {
        let mut tools = BTreeSet::new();
        for t in asked.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            let t = t.to_ascii_lowercase();
            // "mcp" alone: every MCP server the profile has
            if t == "mcp" {
                tools.extend(self.tools.iter().filter(|x| x.starts_with("mcp:")).cloned());
                continue;
            }
            let ok = self.tools.contains(&t) || (t.starts_with("mcp:") && t != "mcp:*" && self.tools.contains("mcp:*"));
            if !ok {
                return Err(format!("x-strata-tools: {t:?} is not one of this key's tools ({})",
                                   self.tools.iter().cloned().collect::<Vec<_>>().join(", ")));
            }
            tools.insert(t);
        }
        Ok(Profile { tools, ..self.clone() })
    }
}

#[derive(Clone, Debug, Default)]
pub struct Clients {
    profiles: Vec<Profile>,
    /// what the main key gets (the "default" profile)
    pub default: Option<Profile>,
}

fn check_set(name: &str, t: &str) -> Result<String, String> {
    let t = t.trim().to_ascii_lowercase();
    if SETS.contains(&t.as_str()) || (t.starts_with("mcp:") && t.len() > 4) {
        Ok(t)
    } else {
        Err(format!("config clients.{name}.tools: {t:?} is not a tool set (one of {}, or mcp:<server>, mcp:*)", SETS.join(", ")))
    }
}

impl Clients {
    /// The run config's `"clients"`; Err names a bad entry. `main_key`: the server's own key (a profile may not reuse it).
    pub fn from_config(rest: &std::collections::BTreeMap<String, Value>, main_key: Option<&str>) -> Result<Clients, String> {
        let mut c = Clients::default();
        let Some(block) = rest.get("clients").filter(|b| !b.is_null()) else { return Ok(c) };
        let obj = block.as_object().ok_or("config clients: expected {\"name\": {\"key\": ..., \"tools\": [...]}}")?;
        for (name, p) in obj {
            let o = p.as_object().ok_or(format!("config clients.{name}: expected an object"))?;
            let tools = o.get("tools").map(|t| t.as_array().ok_or(format!("config clients.{name}.tools: expected a list"))).transpose()?
                .into_iter().flatten()
                .map(|t| t.as_str().ok_or(format!("config clients.{name}.tools: expected names")).and_then(|t| check_set(name, t)))
                .collect::<Result<BTreeSet<_>, _>>()?;
            let memory = o.get("memory").and_then(Value::as_str).map(|m| m.chars().filter(|c| c.is_ascii_alphanumeric() || "-_".contains(*c)).take(40).collect::<String>())
                .filter(|m| !m.is_empty());
            let key = o.get("key").and_then(Value::as_str).map(str::trim).filter(|k| !k.is_empty()).map(str::to_owned);
            let media_jobs_per_hour = o.get("limits").and_then(|l| l.get("media_jobs_per_hour")).and_then(Value::as_u64).map(|n| n as u32);
            let media_in_turn = match o.get("media_in_turn").and_then(Value::as_str) {
                None => None,
                Some(m @ ("defer" | "swap")) => Some(m.to_string()),
                Some(m) => return Err(format!("config clients.{name}.media_in_turn: {m:?} is not \"defer\" or \"swap\"")),
            };
            let lim = o.get("limits");
            let max_image_pixels = match lim.and_then(|l| l.get("max_image_size")) {
                None => None,
                Some(v) => Some(v.as_str().and_then(|z| z.split_once('x')).and_then(|(a, b)| Some(a.trim().parse::<u64>().ok()? * b.trim().parse::<u64>().ok()?))
                    .ok_or(format!("config clients.{name}.limits.max_image_size: expected \"WIDTHxHEIGHT\""))?),
            };
            let max_video_seconds = lim.and_then(|l| l.get("max_video_seconds")).and_then(Value::as_f64);
            let web_queries_per_turn = lim.and_then(|l| l.get("web_queries_per_turn")).and_then(Value::as_u64).map(|n| n as u32);
            let profile = Profile { name: name.clone(), key: key.clone(), tools, memory, media_jobs_per_hour, media_in_turn,
                                    max_image_pixels, max_video_seconds, web_queries_per_turn };
            if name == "default" {
                if key.is_some() {
                    return Err("config clients.default: the default profile is the main key's; it has no key of its own".into());
                }
                c.default = Some(profile);
                continue;
            }
            let key = key.ok_or(format!("config clients.{name}: needs a \"key\" (the API key that picks this profile)"))?;
            if key.len() < 16 {
                return Err(format!("config clients.{name}.key: use at least 16 characters"));
            }
            if Some(key.as_str()) == main_key || c.profiles.iter().any(|q| q.key.as_deref() == Some(key.as_str())) {
                return Err(format!("config clients.{name}.key: each profile needs a key of its own (not the main key, not another profile's)"));
            }
            c.profiles.push(profile);
        }
        Ok(c)
    }

    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// The profile whose key is `given` (compared in constant time).
    pub fn by_key(&self, given: &str) -> Option<&Profile> {
        self.profiles.iter().find(|p| p.key.as_deref().is_some_and(|k| same(given, k)))
    }

    pub fn by_name(&self, name: &str) -> Option<&Profile> {
        self.profiles.iter().find(|p| p.name == name)
    }

    /// Settings › Clients: each profile without its key (the key's last 4 characters only).
    pub fn view(&self) -> Value {
        let show = |p: &Profile| serde_json::json!({"name": p.name, "tools": p.tools, "memory": p.memory_space(),
            "media_jobs_per_hour": p.media_jobs_per_hour, "media_in_turn": p.media_in_turn,
            "max_image_pixels": p.max_image_pixels, "max_video_seconds": p.max_video_seconds, "web_queries_per_turn": p.web_queries_per_turn,
            "key": p.key.as_deref().map(|k| format!("••••{}", &k[k.len().saturating_sub(4)..]))});
        serde_json::json!({"profiles": self.profiles.iter().map(show).collect::<Vec<_>>(),
                           "default": self.default.as_ref().map(show)})
    }

    pub fn names(&self) -> Vec<String> {
        self.profiles.iter().map(|p| p.name.clone()).collect()
    }
}

fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

/// The client profile this task serves (its key picked it), for its limits; None for the main key.
pub fn current(c: &Clients) -> Option<Profile> {
    c.by_name(&crate::queue::current_client()).cloned()
}

/// Err: the size `w`x`h` is past this task's profile's limit.
pub fn image_allowed(c: &Clients, w: u64, h: u64) -> Result<(), String> {
    match current(c).and_then(|p| p.max_image_pixels.map(|m| (p.name, m))) {
        Some((name, m)) if w * h > m => Err(format!("this key ({name}) may make images of at most {m} pixels; {w}x{h} is {}", w * h)),
        _ => Ok(()),
    }
}

/// Err: a video of `secs` seconds is past this task's profile's limit.
pub fn video_allowed(c: &Clients, secs: f64) -> Result<(), String> {
    match current(c).and_then(|p| p.max_video_seconds.map(|m| (p.name, m))) {
        Some((name, m)) if secs > m + 0.05 => Err(format!("this key ({name}) may make videos of at most {m} s; this one is {secs:.1} s")),
        _ => Ok(()),
    }
}

/// The key a request gives (Bearer or x-api-key).
pub fn given_key(headers: &axum::http::HeaderMap) -> Option<&str> {
    headers.get("authorization").and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))
        .or_else(|| headers.get("x-goog-api-key").and_then(|v| v.to_str().ok()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rest(v: Value) -> std::collections::BTreeMap<String, Value> {
        let mut m = std::collections::BTreeMap::new();
        m.insert("clients".into(), v);
        m
    }

    #[test]
    fn profiles_are_read_and_checked() {
        let c = Clients::from_config(&rest(json!({
            "laptop": {"key": "laptop-key-0123456789", "tools": ["memory", "web", "mcp:*"], "memory": "shared"},
            "phone": {"key": "phone-key-0123456789", "tools": ["Speech"]},
            "default": {"tools": ["web"]}})), Some("main-key-0123456789")).unwrap();
        let l = c.by_key("laptop-key-0123456789").unwrap();
        assert_eq!(l.name, "laptop");
        assert_eq!(l.memory_space(), "shared");
        assert!(l.has("web") && !l.has("image"));
        assert_eq!(c.by_key("phone-key-0123456789").unwrap().memory_space(), "phone");
        assert!(c.by_key("main-key-0123456789").is_none());
        assert!(c.default.as_ref().unwrap().has("web"));
        // bad entries
        assert!(Clients::from_config(&rest(json!({"x": {"tools": ["web"]}})), None).is_err(), "no key");
        assert!(Clients::from_config(&rest(json!({"x": {"key": "short", "tools": []}})), None).is_err());
        assert!(Clients::from_config(&rest(json!({"x": {"key": "main-key-0123456789"}})), Some("main-key-0123456789")).is_err());
        assert!(Clients::from_config(&rest(json!({"x": {"key": "k-0123456789abcdef", "tools": ["shell"]}})), None).is_err());
    }

    #[test]
    fn the_header_narrows_and_never_widens() {
        let p = Profile { name: "a".into(), key: None, tools: ["web", "image", "mcp:*", "mcp:files"].iter().map(|s| s.to_string()).collect(), memory: None, media_jobs_per_hour: None, media_in_turn: None, max_image_pixels: None, max_video_seconds: None, web_queries_per_turn: None };
        let n = p.narrowed("web, MCP:search").unwrap();
        assert_eq!(n.tools, ["mcp:search", "web"].iter().map(|s| s.to_string()).collect());
        assert!(p.narrowed("video").is_err());
        assert_eq!(p.narrowed("mcp").unwrap().tools.len(), 2);
        let (names, all) = p.mcp();
        assert_eq!((names, all), (vec!["files".to_string()], true));
    }
}
