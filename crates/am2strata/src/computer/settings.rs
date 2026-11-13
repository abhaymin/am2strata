//! The computer-use / browser-use / rule-preset settings parsed from the run config — the `"computer"`, `"browser"` and
//! `"rules"` blocks, off by default. This mirrors `browser::Settings::from_config`: each field is checked, unknown keys
//! are ignored, and everything absent stays at its (off) default. A `Config` also round-trips to the run-config value
//! that would produce it (see `mod tests`), so the web app can GET it and POST it back unchanged.

use crate::computer::rules::{CustomRules, Preset, RuleSet};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

/// The desktop mode: a virtual display with only the listed apps (`Sandbox`, the default when turned on), a read-only view
/// of the real desktop (`WatchOnly`), the real desktop where every action asks (`Assist`), or input approved once per job
/// (`TrustedJob`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode { #[default] Sandbox, WatchOnly, Assist, TrustedJob }

impl Mode {
    pub fn parse(s: &str) -> Mode {
        match s.trim().to_lowercase().replace(' ', "-").as_str() {
            "watch-only" => Mode::WatchOnly,
            "assist" => Mode::Assist,
            "trusted-job" => Mode::TrustedJob,
            _ => Mode::Sandbox,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Sandbox => "sandbox",
            Mode::WatchOnly => "watch-only",
            Mode::Assist => "assist",
            Mode::TrustedJob => "trusted-job",
        }
    }
    /// When computer use is turned on and no mode is set, this one is used (the sandbox).
    pub const DEFAULT: Mode = Mode::Sandbox;
}

/// The desktop control: switched on (`use_`), the mode it runs in, the apps the sandbox may open, whether downloads ask.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ComputerUse {
    pub use_: bool,
    pub mode: Mode,
    /// the apps the sandbox desktop may open (Strict requires a non-empty list)
    pub apps: Vec<String>,
    /// downloads to a folder auto-accept (`false`) or ask the user each time (`true`, the default when on)
    pub download_ask: Option<bool>,
    /// browser/site allow + block lists; empty means none configured (off by default)
    pub site_allow: Vec<String>,
    pub site_block: Vec<String>,
    /// the sandbox virtual display as "WxHxD" (e.g. "1280x800x24"); `None` uses the default 1280x800x24
    pub display_size: Option<String>,
}

/// Whether the existing headless-Chrome tools are offered again (`"browser": {"use": true}`), off by default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct BrowserUse {
    pub use_: bool,
}

/// The rule preset and, for `Custom`, every toggle the user set (the engine's independent rules).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct RulesConfig {
    pub preset: Preset,
    pub custom: CustomRules,
}

impl RulesConfig {
    /// The engine rule set this config produces for a desktop whose allowed apps are `apps`. Strict / Creator start with an
    /// empty list (Strict then refuses every open); Custom keeps whatever the user's toggles say.
    pub fn rule_set(&self, apps: Vec<String>) -> RuleSet {
        match self.preset {
            Preset::Strict => { let mut r = RuleSet::strict(); r.allowed_apps = apps; r }
            Preset::Creator => { let mut r = RuleSet::creator(); r.allowed_apps = apps; r }
            Preset::Custom => RuleSet::custom(apps, self.custom.no_password_fields, self.custom.no_payment_fields,
                                              self.custom.no_sign_in, self.custom.ask_send_delete_install_buy_publish),
        }
    }

    /// The run-config the user would write for this preset + custom toggles.
    fn to_value(&self) -> Value {
        let mut o = json!({"preset": self.preset.as_str()});
        if self.preset == Preset::Custom && !self.custom.is_empty() {
            o["custom"] = json!({
                "no_password_fields": self.custom.no_password_fields,
                "no_payment_fields": self.custom.no_payment_fields,
                "no_sign_in": self.custom.no_sign_in,
                "ask_send_delete_install_buy_publish": self.custom.ask_send_delete_install_buy_publish,
            });
        }
        o
    }
}

/// The whole feature's settings. `Config::from_config` reads the blocks; `to_value` writes them back (round-trip).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Config {
    pub computer: ComputerUse,
    pub browser: BrowserUse,
    pub rules: RulesConfig,
}

impl Config {
    /// Parse the run config. Every field is optional and off/empty by default, so an absent block means "computer use
    /// off". `data_dir` is accepted for parity with `browser::Settings` (the upload folder would live there).
    pub fn from_config(rest: &BTreeMap<String, Value>, _data_dir: &Path) -> Config {
        let computer = match rest.get("computer").and_then(Value::as_object) {
            Some(c) => ComputerUse {
                use_: c.get("use").and_then(Value::as_bool).unwrap_or(false),
                mode: if let Some(m) = c.get("mode").and_then(Value::as_str) { Mode::parse(m) } else { Mode::DEFAULT },
                apps: names_array(c.get("apps")),
                download_ask: c.get("download_ask").and_then(Value::as_bool),
                site_allow: names_array(c.get("site_allow")),
                site_block: names_array(c.get("site_block")),
                display_size: c.get("display_size").and_then(Value::as_str).map(str::to_string),
            },
            None => ComputerUse { mode: Mode::DEFAULT, ..Default::default() },
        };
        let browser_use = rest.get("browser").and_then(Value::as_object).and_then(|b| b.get("use")).and_then(Value::as_bool).unwrap_or(false);
        let rules = match rest.get("rules").and_then(Value::as_object) {
            Some(r) => RulesConfig {
                preset: r.get("preset").and_then(Value::as_str).map(Preset::parse).unwrap_or_default(),
                custom: CustomRules {
                    no_password_fields: r.get("custom").and_then(|c| c.get("no_password_fields")).and_then(Value::as_bool).unwrap_or(true),
                    no_payment_fields: r.get("custom").and_then(|c| c.get("no_payment_fields")).and_then(Value::as_bool).unwrap_or(true),
                    no_sign_in: r.get("custom").and_then(|c| c.get("no_sign_in")).and_then(Value::as_bool).unwrap_or(true),
                    ask_send_delete_install_buy_publish: r.get("custom").and_then(|c| c.get("ask_send_delete_install_buy_publish")).and_then(Value::as_bool).unwrap_or(true),
                },
            },
            None => RulesConfig { preset: Preset::Strict, ..Default::default() },
        };
        Config { computer, browser: BrowserUse { use_: browser_use }, rules }
    }

    /// The run-config value for this state (only set / non-default fields), the inverse of `from_config`.
    pub fn to_value(&self) -> Value {
        let mut computer = json!({"use": self.computer.use_, "mode": self.computer.mode.as_str()});
        if !self.computer.apps.is_empty() {
            computer["apps"] = json!(self.computer.apps.clone());
        }
        if let Some(d) = self.computer.download_ask {
            computer["download_ask"] = json!(d);
        }
        if !self.computer.site_allow.is_empty() {
            computer["site_allow"] = json!(self.computer.site_allow.clone());
        }
        if !self.computer.site_block.is_empty() {
            computer["site_block"] = json!(self.computer.site_block.clone());
        }
        if let Some(d) = &self.computer.display_size {
            computer["display_size"] = json!(d);
        }
        let mut browser = json!({"use": self.browser.use_});
        json!({"computer": computer, "browser": browser, "rules": self.rules.to_value()})
    }

    /// Whether either kind of computer use is on (so the tools can be offered).
    pub fn any_enabled(&self) -> bool {
        self.computer.use_ || self.browser.use_
    }
}

/// A `"apps"`/`"site_allow"` value: an array of strings, or a comma-separated string; nothing when absent or not a list.
fn names_array(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        Some(Value::String(s)) if !s.trim().is_empty() => s.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        _ => Vec::new(),
    }
}

// ------------------------------------------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn rest(obj: Value) -> BTreeMap<String, Value> {
        let mut m = BTreeMap::new();
        if let Value::Object(o) = obj { for (k, v) in o { m.insert(k, v); } }
        m
    }

    #[test]
    fn absent_blocks_are_all_off() {
        let c = Config::from_config(&rest(json!({})), Path::new("/tmp"));
        assert!(!c.computer.use_, "computer use off by default");
        assert_eq!(c.computer.mode, Mode::Sandbox, "the default mode when on");
        assert!(!c.browser.use_, "browser use off by default");
        assert_eq!(c.rules.preset, Preset::Strict, "the strict preset by default");
        assert!(c.rules.custom.is_empty(), "custom toggles default to the Strict behaviour");
    }

    #[test]
    fn a_full_config_parses_each_block() {
        let c = Config::from_config(&rest(json!({
            "computer": {"use": true, "mode": "Watch Only", "apps": ["firefox", "LibreOffice"],
                         "download_ask": false, "site_allow": ["youtube.com"], "site_block": ["ads.example"]},
            "browser": {"use": true, "auto_approve": true, "connect": "http://127.0.0.1:9222"},
            "rules": {"preset": "custom", "custom": {"no_password_fields": false, "no_sign_in": false}}
        })), Path::new("/tmp"));
        assert!(c.computer.use_);
        assert_eq!(c.computer.mode, Mode::WatchOnly);
        assert_eq!(c.computer.apps, vec!["firefox", "LibreOffice"]);
        assert_eq!(c.computer.download_ask, Some(false));
        assert_eq!(c.computer.site_allow, vec!["youtube.com"]);
        assert_eq!(c.computer.site_block, vec!["ads.example"]);
        assert!(c.browser.use_);
        // the rest of the "browser" block (auto_approve, connect) is left for browser::Settings to parse
        assert_eq!(c.rules.preset, Preset::Custom);
        assert_eq!(c.rules.custom.no_password_fields, false);
        assert_eq!(c.rules.custom.no_sign_in, false);
        assert_eq!(c.rules.custom.no_payment_fields, true, "an unset custom toggle stays the Strict default");
        assert!(c.any_enabled());
    }

    #[test]
    fn config_round_trips_through_its_value() {
        let original = Config::from_config(&rest(json!({
            "computer": {"use": true, "mode": "trusted-job", "apps": ["thunderbird"], "site_block": ["x"]},
            "rules": {"preset": "creator"}
        })), Path::new("/tmp"));
        // writing it back and reading again gives the same settings (no field lost or changed)
        let round = Config::from_config(&rest(original.to_value()), Path::new("/tmp"));
        assert_eq!(round, original);

        // and a hand-written strict config round-trips too
        let c2 = Config { computer: ComputerUse { use_: true, mode: Mode::Sandbox, download_ask: Some(true), ..Default::default() },
                          browser: BrowserUse::default(), rules: RulesConfig { preset: Preset::Strict, ..Default::default() } };
        assert_eq!(Config::from_config(&rest(c2.to_value()), Path::new("/tmp")), c2);
    }

    #[test]
    fn presets_build_the_expected_rule_set() {
        // strict fills the allowed-app list from the desktop; custom keeps its own (empty) list and its toggles
        let strict = Config::from_config(&rest(json!({"rules": {"preset": "strict"}, "computer": {"use": true, "apps": ["firefox"]}})), Path::new("/tmp"));
        assert_eq!(strict.rules.rule_set(vec!["firefox".into()]).allowed_apps, vec!["firefox".to_string()]);

        let rules = json!({"preset": "custom", "custom": {"no_password_fields": false, "no_sign_in": true}});
        let custom = Config::from_config(&rest(json!({"rules": rules})), Path::new("/tmp"));
        assert!(custom.rules.rule_set(Vec::new()).allowed_apps.is_empty(), "custom keeps its own empty list");
    }

    #[test]
    fn modes_parse_their_forms() {
        assert_eq!(Mode::parse("sandbox"), Mode::Sandbox);
        assert_eq!(Mode::parse("Watch Only"), Mode::WatchOnly);
        assert_eq!(Mode::parse("assist"), Mode::Assist);
        assert_eq!(Mode::parse("trusted job"), Mode::TrustedJob);
        assert_eq!(Mode::parse("weird"), Mode::Sandbox, "unknown falls back to the sandbox default");
        assert_eq!(Mode::Sandbox.as_str(), "sandbox");
    }
}
