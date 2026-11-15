//! Computer-use rules, enforced in the server before every step (never left to the model): a step that breaks a rule is
//! refused and the error names the rule. Pure functions with unit tests — this module has no OS or network dependency, so
//! it builds offline with the memory-capped `.agent/bin/cargo` and every decision here is exercised by `mod tests`.
//!
//! Presets (see docs/COMPUTER-USE-PLAN.md §2): **Strict** (default) requires an app allow list and never types into a
//! password or payment field, never signs in, never opens terminal / password manager / banking / system-settings apps,
//! asks before anything that sends, deletes, installs, buys or publishes, and caps each job at 30 minutes. **Creator** is
//! Strict plus, for the browser, publishing on listed sites needs only the job's yes. **Custom** makes every rule an
//! independent toggle.

use serde_json::Value;
use std::time::{Duration, Instant};

/// Words that make a click risky on purpose: it publishes or changes the site's state (Publish, Post, Send, Delete…).
/// Matched whole-word and case-insensitively against a control's label, so "Published 3 days ago" (a heading) does not
/// match — only a button/whose own label is the word. Kept here (not imported from browser.rs) so the rules engine stays a
/// standalone module with no dependency on the browser tools it sits in front of; the two matchers agree by construction.
fn publishing_keyword(text: &str) -> bool {
    const KEYS: [&str; 15] = ["publish", "post", "send", "submit", "upload", "save", "delete", "remove",
                               "share", "public", "make public", "schedule", "confirm", "buy", "pay"];
    let h = text.to_lowercase();
    KEYS.iter().any(|k| word_at(&h, k))
}

/// Whole-word containment (bounded by the ends or non-alphanumerics) for `publishing_keyword`.
fn word_at(haystack: &str, needle: &str) -> bool {
    if haystack.len() < needle.len() { return false; }
    let mut from = 0usize;
    loop {
        match haystack[from..].find(needle) {
            None => return false,
            Some(rel) => {
                let i = from + rel;
                let before = i == 0 || !haystack[..i].chars().next_back().unwrap().is_ascii_alphanumeric();
                let end = i + needle.len();
                let after = end >= haystack.len() || !haystack[end..].chars().next().unwrap().is_ascii_alphanumeric();
                if before && after { return true; }
                from = i + 1;
            }
        }
    }
}

/// A preset: Strict (default), Creator (owner) or Custom (the user edits each rule).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Preset { #[default] Strict, Creator, Custom }

impl Preset {
    pub fn parse(s: &str) -> Preset {
        match s.trim().to_lowercase().as_str() {
            "creator" => Preset::Creator,
            "custom" => Preset::Custom,
            _ => Preset::Strict,
        }
    }
    /// The run-config spelling of this preset (`"strict"`, `"creator"` or `"custom"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Preset::Strict => "strict",
            Preset::Creator => "creator",
            Preset::Custom => "custom",
        }
    }
    /// A one-line description for the Rules panel in the Settings UI.
    pub fn description(self) -> &'static str {
        match self {
            Preset::Strict => "Default: an app allow list, never a password or payment field, no sign-in, no terminal, ask before any action that sends, deletes, installs, buys or publishes.",
            Preset::Creator => "Owner-style: Strict, plus the browser may upload from the upload folder and publish on listed sites after the job's yes.",
            Preset::Custom => "Every rule below is a toggle you set.",
        }
    }
}

/// The kind of field a `computer_type` / `browser_type` step targets (from its `type`/`selector`), for password and
/// payment detection. Unknown or text → Text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind { Text, Email, Password, Payment }

fn detect_field(arg: &str) -> FieldKind {
    let s = arg.to_lowercase();
    if s.contains("password") || s.contains("passwd") || s.contains(".pw") || s == "password" {
        FieldKind::Password
    } else if s.contains("cvc") || s.contains("cvv") || s.contains("card") || s.contains("payment")
        || s.contains("credit card") || s.contains("-pin") || s.contains(" pin ") || s == "pin" {
        FieldKind::Payment
    } else if s.contains("email") || s.contains("@") || s.contains("e-mail") {
        FieldKind::Email
    } else {
        FieldKind::Text
    }
}

/// What a classified step does, at the level the rules reason about — not the raw tool name and arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Intent {
    Click { label: Option<String>, risky: bool }, // risky: publishes / sends / deletes / installs / buys (per its label)
    TypeIntoField(FieldKind),
    Key(Vec<String>),
    OpenApp(String),
    Scroll,
    Screenshot,
    Wait,
}

/// A browser step's context the rules can use: the site it is on (Strict and Creator differ only on listed sites).
#[derive(Clone, Debug, Default)]
pub struct Ctx<'a> {
    pub site: Option<&'a str>,
}

/// What the server does with a classified step. `Refused` carries the rule's name, which is put in the error message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Allowed,
    /// run the reused approval flow; the String is the human prompt suffix ("open 'Terminal'", "publish on youtube.com")
    Approval(String),
    Refused(&'static str),
}

/// The user's independent rule toggles for the `custom` preset (every one on == Strict; a custom edit turns some off).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CustomRules {
    pub no_password_fields: bool,
    pub no_payment_fields: bool,
    pub no_sign_in: bool,
    pub ask_send_delete_install_buy_publish: bool,
}

impl Default for CustomRules {
    /// A fresh custom preset equals Strict: every rule on. (derive(Default) would give false, which is the opposite.)
    fn default() -> Self {
        CustomRules { no_password_fields: true, no_payment_fields: true, no_sign_in: true, ask_send_delete_install_buy_publish: true }
    }
}

impl CustomRules {
    /// The toggles all equal the Strict defaults (every rule on), so a `custom` preset with none edited behaves like Strict.
    pub fn is_empty(&self) -> bool {
        self.no_password_fields && self.no_payment_fields && self.no_sign_in && self.ask_send_delete_install_buy_publish
    }
}

/// A preset's rules. Built from a preset (`strict`/`creator`) or from the user's custom toggles, and checked by
/// `RuleSet::check`. Kept as plain fields so Custom is just another constructor — no match arms to keep in step.
#[derive(Clone, Debug)]
pub struct RuleSet {
    /// Strict / Creator require this to be non-empty; Custom allows an empty list (then nothing opens) and owns every toggle.
    pub allowed_apps_required: bool,
    pub allowed_apps: Vec<String>,
    /// the app categories that are never openable on their own (independent of the allow list)
    forbidden_classes: &'static [&'static str],
    pub no_password_fields: bool,
    pub no_payment_fields: bool,
    pub no_sign_in: bool,
    pub ask_send_delete_install_buy_publish: bool,
    /// Creator only: publish on a listed site needs the job's yes (Allowed), not another ask. Strict always asks.
    pub allow_publish_on_listed_sites: bool,
    pub max_job_seconds: u64,
}

impl RuleSet {
    fn new(preset: Preset) -> RuleSet {
        let (allowed_apps_required, no_sign_in, ask, allow_listed) = match preset {
            Preset::Strict | Preset::Creator => (true, true, true, preset == Preset::Creator),
            Preset::Custom => (false, false, false, false),
        };
        RuleSet {
            allowed_apps_required,
            allowed_apps: Vec::new(),
            forbidden_classes: &["terminal", "password manager", "banking", "settings", "wallet", "authenticator"],
            no_password_fields: true,
            no_payment_fields: true,
            no_sign_in,
            ask_send_delete_install_buy_publish: ask,
            allow_publish_on_listed_sites: allow_listed,
            max_job_seconds: 30 * 60,
        }
    }

    pub fn strict() -> RuleSet { RuleSet::new(Preset::Strict) }
    pub fn creator() -> RuleSet { RuleSet::new(Preset::Creator) }

    /// A Custom rule set from the user's toggles and allow list (`rules.custom`). Every rule is independent.
    pub fn custom(allowed_apps: Vec<String>, no_password: bool, no_payment: bool, no_sign_in: bool, ask: bool) -> RuleSet {
        RuleSet {
            allowed_apps_required: false,
            allowed_apps,
            forbidden_classes: &["terminal", "password manager", "banking", "settings", "wallet", "authenticator"],
            no_password_fields: no_password,
            no_payment_fields: no_payment,
            no_sign_in,
            ask_send_delete_install_buy_publish: ask,
            allow_publish_on_listed_sites: false,
            max_job_seconds: 30 * 60,
        }
    }

    /// The decision for a step. `now` / `started_at` enforce the job time limit (`time_limit`).
    pub fn check(&self, intent: &Intent, ctx: &Ctx, started_at: Instant, now: Instant) -> Decision {
        if !time_allowed(started_at, now, self.max_job_seconds) {
            return Decision::Refused("time_limit");
        }
        match intent {
            Intent::OpenApp(name) => self.check_app(name),
            Intent::TypeIntoField(FieldKind::Password) if self.no_password_fields => Decision::Refused("no_password_fields"),
            Intent::TypeIntoField(FieldKind::Payment) if self.no_payment_fields => Decision::Refused("no_payment_fields"),
            // a text, email, or an unguarded password/payment field is ordinary (sign-in detection is separate: check_sign_in)
            Intent::TypeIntoField(FieldKind::Text | FieldKind::Email | FieldKind::Password | FieldKind::Payment) => Decision::Allowed,
            Intent::Click { label, risky } => self.check_click(label, *risky, ctx.site),
            Intent::Key(_) | Intent::Scroll | Intent::Screenshot | Intent::Wait => Decision::Allowed,
        }
    }

    /// A sign-in step (the server detects a username+password pair) is always refused, even in Custom without the field rule.
    pub fn check_sign_in(&self) -> Decision {
        if self.no_sign_in {
            Decision::Refused("no_sign_in")
        } else {
            Decision::Allowed
        }
    }

    fn check_app(&self, name: &str) -> Decision {
        let low = name.to_lowercase();
        if self.forbidden_classes.iter().any(|c| low.contains(c)) {
            return Decision::Refused("forbidden_apps");
        }
        // Strict / Creator require a non-empty allow list; Custom with the app check off trusts the owner (open anything
        // not in `forbidden_classes`, even an empty list). Empty or missing lists are only enforced when required.
        if self.allowed_apps_required && self.allowed_apps.iter().all(|a| a.trim().is_empty()) {
            return Decision::Refused("allowed_apps");
        }
        if self.allowed_apps_required && !self.allowed_apps.iter().any(|a| a.eq_ignore_ascii_case(name.trim())) {
            return Decision::Refused("allowed_apps");
        }
        Decision::Allowed
    }

    fn check_click(&self, label: &Option<String>, risky: bool, site: Option<&str>) -> Decision {
        if !risky || !self.ask_send_delete_install_buy_publish {
            return Decision::Allowed;
        }
        // publishing on a listed site needs only the job's yes in Creator (Strict still asks each time).
        if self.allow_publish_on_listed_sites && site.is_some() {
            return Decision::Allowed;
        }
        let prompt = label.as_ref().map(|l| format!("{} '{}'", click_verb(l), l)).unwrap_or_else(|| "a risky action".into());
        Decision::Approval(prompt)
    }
}

/// A verb for the approval prompt from a control's label ("Click 'Publish'", "Submit 'Send'").
fn click_verb(label: &str) -> &'static str {
    let l = label.to_lowercase();
    if l.contains("publish") || l.contains("post") || l.contains("share") || l.contains("make public") { "publish" }
    else if l.contains("send") || l.contains("submit") { "send" }
    else if l.contains("delete") || l.contains("remove") { "delete" }
    else if l.contains("install") || l.contains("download") { "install" }
    else if l.contains("buy") || l.contains("pay") || l.contains("checkout") { "buy" }
    else { "click" }
}

/// Whether a job has stayed within its time limit. Pure: take instants from an injected clock in tests.
pub fn time_allowed(started_at: Instant, now: Instant, max_seconds: u64) -> bool {
    now.duration_since(started_at) <= Duration::from_secs(max_seconds)
}

/// If this shell command is destructive to something on the owner's protected list (`computer.protect`), return what it hit —
/// so the VM's shell tool can refuse before running it (`engine.rs`), telling the model to work on a copy. It hits when it
/// names a protected entry directly (a verb's object or a `>` redirection target), or, only when the list is non-empty, when
/// a destructive command selects in bulk (acting on everything: `docker stop $(docker ps -q)`, `rm -rf /srv/*`,
/// `systemctl stop '*'`) and names none. Names the entry that is named; for a bulk selector reports the first listed protected
/// entry with "(it acts on everything)". Empty/blank list protects nothing (`nothing_is_protected_by_default`), so even a plainly
/// destructive command refuses to refuse.
pub fn protected_hit(command: &str, protect: &[String]) -> Option<String> {
    let listed: Vec<&str> = protect.iter().filter(|e| !e.trim().is_empty()).map(|e| e.as_str()).collect();
    if listed.is_empty() { return None; }

    let destructive = has_destructive_verb(command);
    // a destructive command that selects in bulk acts on more than the named entries — refuse and say so.
    if destructive && selects_in_bulk(command) {
        return Some(format!("{} (it acts on everything)", listed[0]));
    }

    // otherwise refuse only when a specific protected entry is actually named, by path or as a redirection target.
    for e in &listed {
        if command.contains(e) && (destructive || any_redirection_writes(command, e)) {
            return Some(e.to_string());
        }
    }
    None
}

/// Whether `command` runs something that removes or changes state on purpose — a remove/overwrite verb (`rm`, `rmdir`, `mv`,
/// `dd`, `mkfs`, `truncate`, `shred`, `chmod`, `chown`, `kill`, `pkill`, `killall`), an in-place edit (`sed -i`), or a
/// `docker` / `podman` / `systemctl` subcommand that stops, removes, disables or prunes. Redirection and bulk selection are
/// judged against the protected list separately (in `protected_hit`), not here.
fn has_destructive_verb(command: &str) -> bool {
    let h = command.to_lowercase();
    let tokens: Vec<&str> = h.split_whitespace().collect();
    // removing / changing words, matched whole-word so "address" does not count as "dd".
    for w in ["rm", "rmdir", "mv", "dd", "mkfs", "truncate", "shred", "chmod", "chown", "kill", "pkill", "killall"] {
        if word_at(&h, w) { return true; }
    }
    // an in-place edit: a `sed` with a `-i` flag somewhere later on the command.
    let mut saw_sed = false;
    for tok in &tokens {
        if *tok == "sed" { saw_sed = true; }
        else if saw_sed && tok.starts_with('-') && tok[1..].contains('i') { return true; }
    }
    follow_kill(&tokens, &["docker", "podman"], &["rm", "stop", "kill", "restart", "down", "prune"])
        || follow_kill(&tokens, &["systemctl"], &["stop", "restart", "disable", "mask"])
}

/// Whether `command` selects everything at runtime rather than one named thing, so (with a destructive verb above) it hits more
/// than the protected entries: command substitution (`$(…)` or backticks), a glob (`*`/`?`), `docker compose down`, a prune/all
/// flag (`-a` / `--all` / `prune`), or a `kill -1` / `kill -9 -1` (all processes). Only meaningful with `has_destructive_verb`.
fn selects_in_bulk(command: &str) -> bool {
    let h = command.to_lowercase();
    if h.contains("$(") || h.contains('`') { return true; }   // expands to many things at runtime
    if h.contains('*') || h.contains('?') { return true; }     // a glob matches every match, not one thing
    let tokens: Vec<&str> = h.split_whitespace().collect();
    if tokens.windows(2).any(|w| w[0] == "compose" && w[1] == "down") { return true; }  // docker compose down tears the stack
    if tokens.iter().any(|t| *t == "-a" || *t == "--all" || *t == "prune") { return true; }
    let mut saw_kill = false;
    for t in &tokens {
        if *t == "kill" || *t == "pkill" || *t == "killall" { saw_kill = true; continue; }
        if saw_kill && *t == "-1" { return true; }  // kill -1 / kill -9 -1 → all processes
    }
    false
}

/// The filenames a `>`/`>>` redirection writes to (the word after the operator, or in a `>file` token), skipping a leading fd
/// (`2>`) and fd-dups (`2>&1`, `&>/dev/null`) so error/output pipes don't read as writing to a protected path.
fn redirect_targets(command: &str) -> Vec<String> {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    let mut targets = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        if tok == ">" || tok == ">>" {                 // a redirection onto the next word
            if let Some(&next) = tokens.get(i + 1) { if !next.is_empty() { targets.push(next.to_string()); } }
            continue;
        }
        let stripped = tok.trim_start_matches(|c: char| c.is_ascii_digit()).trim_start_matches('&');  // drop the "2"/"&" in "2>", "&>"
        if stripped.starts_with('>') {                  // a redirection stuck to its target: ">file", ">>/x", "2>&1"
            let target = stripped.trim_start_matches('>');
            if !target.is_empty() && !target.starts_with('&') { targets.push(target.to_string()); }  // not a fd-dup like ">&1"
        }
    }
    targets
}

/// Whether any `>` redirection in `command` writes to a file whose path contains the protected entry `entry`.
fn any_redirection_writes(command: &str, entry: &str) -> bool {
    redirect_targets(command).iter().any(|t| t.contains(entry))
}

/// True when one of `bins` appears and, later in the command, one of `verbs` appears as a token (docker/podman/systemctl that
/// stop or remove something). The binary itself is not also treated as a verb.
fn follow_kill(tokens: &[&str], bins: &[&str], verbs: &[&str]) -> bool {
    let mut seen = false;
    for tok in tokens {
        if !seen && bins.contains(tok) { seen = true; continue; }
        if seen && verbs.contains(tok) { return true; }
    }
    false
}

/// Classify a tool call into an Intent (best-effort from the arguments), and whether it is risky by its label. This is
/// where password/payment field detection happens; pure and unit-tested with a mock, no OS needed.
pub fn classify(tool: &str, args: &Value) -> Option<Intent> {
    match tool {
        "computer_screenshot" => Some(Intent::Screenshot),
        "computer_scroll" => Some(Intent::Scroll),
        "computer_wait" | "browser_wait" => Some(Intent::Wait),
        "computer_open_app" => args["name"].as_str().map(|n| Intent::OpenApp(n.to_string())),
        "computer_type" => {
            let arg = args["selector"].as_str().or(args["type"].as_str()).unwrap_or("");
            Some(Intent::TypeIntoField(detect_field(arg)))
        }
        "computer_key" => Some(Intent::Key(keys_vec(&args["keys"]))),
        "browser_type" => {
            // browser tools carry their own risky detection (the DOM); here we can only see the raw selector.
            let arg = args["selector"].as_str().unwrap_or("");
            Some(Intent::TypeIntoField(detect_field(arg)))
        }
        "browser_click" => {
            // a point click (x, y) has no label to judge; risky detection stays with the DOM in run.rs.
            let text = args["text"].as_str().unwrap_or("");
            Some(Intent::Click { label: if text.is_empty() { None } else { Some(text.to_string()) }, risky: false })
        }
        "computer_click" => {
            let text = args["text"].as_str().or(args["aria"].as_str()).unwrap_or("");
            let label = if text.is_empty() { None } else { Some(text.to_string()) };
            let risky = label.as_ref().is_some_and(|t| publishing_keyword(t));
            Some(Intent::Click { label, risky })
        }
        _ => None,
    }
}

/// `computer_key` accepts either an array of keys (["Ctrl", "a"]) or a single string ("Enter"): return both forms.
fn keys_vec(keys: &Value) -> Vec<String> {
    match keys {
        Value::Array(a) => a.iter().filter_map(|k| k.as_str().map(str::to_string)).collect(),
        Value::String(s) => {
            if s.trim().is_empty() { Vec::new() } else { s.split('+').map(|s| s.trim().to_string()).collect() }
        }
        _ => Vec::new(),
    }
}

// ------------------------------------------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn strict_apps() -> RuleSet { let mut s = RuleSet::strict(); s.allowed_apps = vec!["firefox".into(), "libreoffice".into()]; s }
    fn now() -> Instant { Instant::now() }

    // --- Strict: allowed and refused steps ---------------------------------------------------------------

    #[test]
    fn strict_allows_ordinary_steps() {
        let r = strict_apps();
        assert_eq!(r.check(&Intent::Screenshot, &Ctx::default(), now(), now()), Decision::Allowed);
        assert_eq!(r.check(&Intent::Scroll, &Ctx::default(), now(), now()), Decision::Allowed);
        assert_eq!(r.check(&Intent::Key(vec!["a".into()]), &Ctx::default(), now(), now()), Decision::Allowed);
        // an allowed app opens
        assert_eq!(r.check(&Intent::OpenApp("firefox".into()), &Ctx::default(), now(), now()), Decision::Allowed);
    }

    #[test]
    fn strict_refuses_password_payment_signin_and_forbidden_apps() {
        let r = strict_apps();
        assert_eq!(r.check(&Intent::TypeIntoField(FieldKind::Password), &Ctx::default(), now(), now()), Decision::Refused("no_password_fields"));
        assert_eq!(r.check(&Intent::TypeIntoField(FieldKind::Payment), &Ctx::default(), now(), now()), Decision::Refused("no_payment_fields"));
        assert_eq!(r.check_sign_in(), Decision::Refused("no_sign_in"));
        for app in ["Terminal", "System Settings", "Authenticator", "Online Banking"] {
            assert_eq!(r.check(&Intent::OpenApp(app.into()), &Ctx::default(), now(), now()), Decision::Refused("forbidden_apps"), "{app}");
        }
    }

    #[test]
    fn strict_refuses_unknown_app_and_missing_allow_list() {
        let r = RuleSet::strict(); // no allowed apps
        assert_eq!(r.check(&Intent::OpenApp("firefox".into()), &Ctx::default(), now(), now()), Decision::Refused("allowed_apps"));
        let mut with = RuleSet::strict();
        with.allowed_apps = vec!["other".into()];
        assert_eq!(with.check(&Intent::OpenApp("firefox".into()), &Ctx::default(), now(), now()), Decision::Refused("allowed_apps"));
    }

    #[test]
    fn strict_asks_before_risky_click_but_not_ordinary() {
        let r = strict_apps();
        // a risky click (publish) asks via the reused approval flow
        assert_eq!(r.check(&Intent::Click { label: Some("Publish".into()), risky: true }, &Ctx::default(), now(), now()), Decision::Approval("publish 'Publish'".into()));
        // an ordinary click is allowed
        assert_eq!(r.check(&Intent::Click { label: Some("Next".into()), risky: false }, &Ctx::default(), now(), now()), Decision::Allowed);
    }

    // --- Creator = Strict plus publish on listed sites needs only the job's yes --------------------------

    #[test]
    fn creator_publishes_on_listed_sites_without_asking() {
        let mut c = RuleSet::creator();
        c.allowed_apps = vec!["firefox".into()];
        // Strict still asks a publish even on a listed site; Creator lets it through after the job's yes.
        assert_eq!(c.check(&Intent::Click { label: Some("Publish".into()), risky: true }, &Ctx { site: Some("youtube.com") }, now(), now()), Decision::Allowed);
        let mut s = RuleSet::strict();
        s.allowed_apps = vec!["firefox".into()];
        assert_eq!(s.check(&Intent::Click { label: Some("Publish".into()), risky: true }, &Ctx { site: Some("youtube.com") }, now(), now()), Decision::Approval("publish 'Publish'".into()));
    }

    // --- Custom: every rule is an independent toggle -----------------------------------------------------

    #[test]
    fn custom_allows_password_field_when_toggled_off() {
        let c = RuleSet::custom(vec!["app".into()], false, true, false, true);
        assert_eq!(c.check_sign_in(), Decision::Allowed, "sign-in allowed when its toggle is off");
        assert_eq!(c.check(&Intent::TypeIntoField(FieldKind::Password), &Ctx::default(), now(), now()), Decision::Allowed, "password allowed when its toggle is off");
        // but a payment field still refused (its own toggle) and an unknown app still refused only if required
        assert_eq!(c.check(&Intent::TypeIntoField(FieldKind::Payment), &Ctx::default(), now(), now()), Decision::Refused("no_payment_fields"));
        // a non-forbidden app opens with an empty Custom list (the allow-list check is off when not required)
        let none = RuleSet::custom(Vec::new(), true, true, true, true);
        assert_eq!(none.check(&Intent::OpenApp("anything".into()), &Ctx::default(), now(), now()), Decision::Allowed, "a non-forbidden app opens with an empty Custom list");
    }

    // --- password / payment field detection ------------------------------------------------------------

    #[test]
    fn field_detection_classes_their_inputs() {
        assert_eq!(detect_field("input[type=password]"), FieldKind::Password);
        assert_eq!(detect_field("login[passwd]"), FieldKind::Password);
        assert_eq!(detect_field("card-number"), FieldKind::Payment);
        assert_eq!(detect_field("[name=cvc]"), FieldKind::Payment);
        assert_eq!(detect_field("[name=email]"), FieldKind::Email);
        assert_eq!(detect_field("input[type=text]"), FieldKind::Text);
    }

    #[test]
    fn classify_type_reads_the_field_kind() {
        assert_eq!(classify("computer_type", &json!({"selector": "input[password]", "text": "x"})), Some(Intent::TypeIntoField(FieldKind::Password)));
        assert_eq!(classify("computer_type", &json!({"type": "card-number", "text": "4111"})), Some(Intent::TypeIntoField(FieldKind::Payment)));
    }

    // --- app allow list ----------------------------------------------------------------------------------

    #[test]
    fn allow_list_opens_allowed_and_refuses_unknown() {
        let r = strict_apps();
        assert_eq!(r.check(&Intent::OpenApp("LibreOffice".into()), &Ctx::default(), now(), now()), Decision::Allowed, "case-insensitive match");
        assert_eq!(r.check(&Intent::OpenApp("code".into()), &Ctx::default(), now(), now()), Decision::Refused("allowed_apps"));
    }

    // --- time limit ------------------------------------------------------------------------------------

    #[test]
    fn time_limit_allows_under_and_refuses_after() {
        let t0 = Instant::now();
        assert!(time_allowed(t0, t0 + Duration::from_secs(10), 30 * 60), "under 30 min is allowed");
        assert!(!time_allowed(t0, t0 + Duration::from_secs(31 * 60), 30 * 60), "past 30 min is refused");
        let r = strict_apps();
        assert_eq!(r.check(&Intent::Screenshot, &Ctx::default(), t0, t0 + Duration::from_secs(31 * 60)), Decision::Refused("time_limit"));
    }

    // --- preset constructors and parsing ---------------------------------------------------------------

    #[test]
    fn presets_parse_and_describe() {
        assert_eq!(Preset::parse("strict"), Preset::Strict);
        assert_eq!(Preset::parse("CREATOR"), Preset::Creator);
        assert_eq!(Preset::parse("custom"), Preset::Custom);
        assert_eq!(Preset::parse("weird"), Preset::Strict, "unknown falls back to Strict");
        assert!(Preset::Strict.description().is_ascii());
    }

    #[test]
    fn classify_matches_each_tool() {
        assert_eq!(classify("computer_screenshot", &json!({})), Some(Intent::Screenshot));
        assert_eq!(classify("computer_open_app", &json!({"name": "firefox"})), Some(Intent::OpenApp("firefox".into())));
        assert_eq!(classify("computer_key", &json!({"keys": ["Ctrl", "a"]})), Some(Intent::Key(vec!["Ctrl".into(), "a".into()])));
        assert_eq!(classify("computer_key", &json!({"keys": "Enter"})), Some(Intent::Key(vec!["Enter".into()])));
        assert_eq!(classify("computer_click", &json!({"text": "Publish", "x": 10.0, "y": 20.0})), Some(Intent::Click { label: Some("Publish".into()), risky: true }));
    }

    // --- click prompt verb -------------------------------------------------------------------------------

    #[test]
    fn click_prompt_uses_the_right_verb() {
        assert_eq!(click_verb("Publish now"), "publish");
        assert_eq!(click_verb("Send email"), "send");
        assert_eq!(click_verb("Delete this"), "delete");
        assert_eq!(click_verb("Buy now"), "buy");
        assert_eq!(click_verb("Next"), "click");
    }

    // --- protected things (computer.protect) ---------------------------------------------------------

    #[test]
    fn refuses_destructive_command_that_names_a_protected_thing() {
        let prod_db = vec!["prod-db".into()];
        assert_eq!(
            protected_hit("docker rm -f prod-db", &prod_db),
            Some("prod-db".to_string()),
            "removing a named container is refused"
        );

        let nginx = vec!["nginx-prod".into()];
        assert_eq!(
            protected_hit("systemctl restart nginx-prod", &nginx),
            Some("nginx-prod".to_string()),
            "restarting a named service is refused"
        );

        let srv = vec!["/srv/prod".into()];
        assert_eq!(
            protected_hit("sed -i s/a/b/ /srv/prod/app.conf", &srv),
            Some("/srv/prod".to_string()),
            "an in-place edit of a protected file is refused"
        );
        assert_eq!(
            protected_hit("echo x > /srv/prod/f", &srv),
            Some("/srv/prod".to_string()),
            "a `>` redirection onto a protected path is refused"
        );
    }

    #[test]
    fn does_not_refuse_reads_or_other_paths() {
        // reading a protected file is not destructive
        let srv = vec!["/srv/prod".into()];
        assert_eq!(protected_hit("cat /srv/prod/app.conf", &srv), None);
        // destructive, but names a different thing than what is protected
        let srv = vec!["/srv/prod".into()];
        assert_eq!(protected_hit("rm -rf /tmp/copy", &srv), None);
    }

    #[test]
    fn nothing_is_protected_by_default() {
        // even a plainly destructive command refuses to name something when the list is empty
        assert_eq!(protected_hit("rm -rf everything", &[]), None);
        let srv = vec![String::new()];  // an empty entry counts as "nothing listed"
        assert_eq!(protected_hit("rm -rf /srv/prod/f", &srv), None);
    }

    #[test]
    fn other_destructive_forms_are_recognised() {
        for c in ["truncate -s 0 /data", "shred file", "chmod 777 /etc/x", "dd if=/dev/zero of=/data",
                  "mkfs.ext4 /dev/sdb1", "mv a b", "pkill node", "podman stop web"] {
            assert!(has_destructive_verb(c), "expected destructive: {c}");
        }
    }

    #[test]
    fn refuses_bulk_commands_that_act_on_everything() {
        // the field-test hole: a command that selects in bulk names no protected entry, so it must still be refused.
        let prod_db = vec!["prod-db".into()];
        assert_eq!(protected_hit("docker stop $(docker ps -q)", &prod_db), Some("prod-db (it acts on everything)".to_string()));
        assert_eq!(protected_hit("docker rm -f $(docker ps -aq)", &prod_db), Some("prod-db (it acts on everything)".to_string()));
        assert_eq!(protected_hit("docker system prune -a", &prod_db), Some("prod-db (it acts on everything)".to_string()));
        assert_eq!(protected_hit("docker compose down", &prod_db), Some("prod-db (it acts on everything)".to_string()));

        let nginx = vec!["nginx-prod".into()];
        assert_eq!(protected_hit("systemctl stop '*'", &nginx), Some("nginx-prod (it acts on everything)".to_string()));
        assert_eq!(protected_hit("kill -9 -1", &nginx), Some("nginx-prod (it acts on everything)".to_string()));

        let srv = vec!["/srv/prod".into()];
        assert_eq!(protected_hit("rm -rf /srv/*", &srv), Some("/srv/prod (it acts on everything)".to_string()));

        // but with nothing listed, a bulk command names no entry to refuse.
        assert_eq!(protected_hit("docker stop $(docker ps -q)", &[]), None);
    }

    #[test]
    fn redirection_is_refused_only_onto_a_protected_target() {
        // the false refusal: `>` is only destructive when its target names a protected entry, not for any `>`.
        let srv = vec!["/srv/prod".into()];
        assert_eq!(protected_hit("cat /srv/prod/app.conf 2>/dev/null", &srv), None, "stderr to /dev/null is fine");
        assert_eq!(protected_hit("ls /srv/prod > /tmp/list", &srv), None, "a redirect onto /tmp is fine");
        assert_eq!(protected_hit("echo x >> /srv/prod/f", &srv), Some("/srv/prod".to_string()), ">> into a protected path is refused");
    }
}
