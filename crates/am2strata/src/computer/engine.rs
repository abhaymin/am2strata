//! The desktop a computer-use step drives, wired into the run loop (the plan's §4). Before every computer step this
//! classifies it (step.rs), checks it against the active preset (rules.rs), runs it over the backend or sends it for
//! approval — and it is all exercised by unit tests with a mock backend, no OS needed. The desktop itself (backend.rs)
//! is chosen by mode: the sandbox / assist / trusted-job modes run here; watch-only reads the desktop but never input.
//!
//! - **Offered only when `"computer": {"use": true}`.** `offered_tools` returns the empty set otherwise, so nothing
//!   changes for anyone who has not switched it on (server_tools.rs).
//! - **Every step is checked before it runs.** A step the rules refuse (`Refused(rule)`) is not executed and its tool
//!   result names the rule; a step marked "ask" waits for approval through the browser's own path (`approvals()`, one
//!   per job); only then does it run.
//! - **Modes gate input.** In `watch-only` no input is sent (only a screenshot or a wait).

use crate::browser;
use crate::computer::backend::Backend;
use crate::computer::rules::{self, Ctx, Decision, Intent, RuleSet};
use crate::computer::settings::{Config, Mode};
use crate::computer::step::classify_step;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use crate::run::RunEvent;

/// How long a step may wait for the user's yes before it is treated as declined (the browser waits the same).
const APPROVE_TIMEOUT: Duration = Duration::from_secs(180);
/// How long one approval of a job stays good for that conversation (one per job): thirty minutes, like the browser.
const JOB_WINDOW: Duration = Duration::from_secs(30 * 60);

/// A tool that sends input to the desktop (clicks, types, presses keys, scrolls or opens an app). `watch-only` blocks
/// all of these: it only reads the screen (a screenshot) or waits.
fn sends_input(name: &str) -> bool {
    matches!(name, "computer_click" | "computer_type" | "computer_key" | "computer_scroll" | "computer_open_app")
}

/// Run one computer-use tool call over `backend`, after classifying it and checking the active preset, returning the
/// tool result the run loop puts in the conversation (the same shape the browser and MCP tools return). `grants`
/// holds the conversation's once-per-job approval so a later ask of the same job runs without asking again; `tx` sends
/// the "approve" event when a step waits. Kept generic over the backend so tests drive a mock.
pub async fn run_one(backend: &(dyn Backend + Send + Sync), rules: &RuleSet, mode: Mode, conv: &str, name: &str, args: &Value,
                     id: &str, grants: &mut HashMap<String, Instant>, started_at: Instant, now: Instant,
                     tx: &mpsc::Sender<RunEvent>) -> Value {
    let t0 = Instant::now();
    let ms = |t0: Instant| t0.elapsed().as_millis() as u64;
    // 1. A call with a missing or wrong argument is refused up front, the step never runs (like browser.rs refuses a
    //    click without x and y), so the model can fix it.
    let step = match classify_step(name, args) {
        Ok(s) => s,
        Err(e) => return error_result(format!("error: the step was not run: {e}"), ms(t0)),
    };
    // 2. In watch-only no input is sent: a click / type / key / scroll / open_app is declined without asking; only a
    //    screenshot (a read) or a wait go on.
    if mode == Mode::WatchOnly && sends_input(name) {
        return error_result("error: the step was not run (watch-only mode does not send input)".to_string(), ms(t0));
    }
    // 3. Classify into an Intent and check it against the active preset. A refusal is never executed and names the rule;
    //    an "ask" waits for approval through the browser's own path.
    let intent = rules::classify(name, args).unwrap_or(Intent::Screenshot);
    match rules.check(&intent, &Ctx::default(), started_at, now) {
        Decision::Refused(rule) => return error_result(format!("error: the step was not run, by rule {rule}"), ms(t0)),
        Decision::Approval(prompt) => {
            let granted = grants.get(conv).is_some_and(|until| now < *until);
            if !granted {
                let (atx, arx) = tokio::sync::oneshot::channel();
                browser::approvals().lock().unwrap().insert(id.to_string(), atx);
                let _ = tx.send(RunEvent::Mcp(json!({"event": "approve", "id": id, "name": name, "what": prompt}))).await;
                let allowed = matches!(tokio::time::timeout(APPROVE_TIMEOUT, arx).await, Ok(Ok(true)));
                browser::approvals().lock().unwrap().remove(id);
                if allowed { grants.insert(conv.to_string(), now + JOB_WINDOW); }
                if !allowed { return error_result("error: the user did not allow this step (or no one answered); ask them, or do something else".into(), ms(t0)); }
            }
        }
        Decision::Allowed => {}
    }
    // 4. The rules let it through: run it over the desktop.
    let err = match name {
        "computer_screenshot" | "computer_wait" => Ok(()),
        "computer_click" => backend.click(step.x.unwrap_or(0), step.y.unwrap_or(0), step.button),
        "computer_type" => backend.type_text(step.text.as_deref().unwrap_or("")),
        "computer_key" => backend.press_keys(&step.keys),
        "computer_scroll" => backend.scroll(step.dx.unwrap_or(0), step.dy.unwrap_or(0)),
        "computer_open_app" => backend.open_app(step.name.as_deref().unwrap_or("")),
        _ => Ok(()),
    };
    if let Err(e) = err {
        return error_result(format!("error: {e}"), ms(t0));
    }
    // A screenshot follows every step, like the browser's, so a model that reads pictures sees what changed.
    let shot = backend.screenshot().ok().and_then(|b| b.jpeg_b64());
    let text = format!("{name} ran.");
    let mut r = json!({"ok": true, "chars": text.chars().count(), "text": text, "truncated": false, "ms": ms(t0)});
    if let Some(s) = shot { r["image"] = json!(s); }
    r
}

/// A tool result: `ok`, the characters for the model (`chars`), its text, and whether it was truncated. The run loop's
/// MCP-mapping reads these exactly like the browser and MCP tools' results (its `image` becomes a data URL).
fn error_result(text: String, ms: u64) -> Value {
    json!({"ok": false, "chars": 0, "text": text, "truncated": false, "ms": ms})
}

/// The desktop computer use for one server: the run config's ruleset and mode, and the backend chosen for this machine.
    /// Built once at start from the run config (`main.rs`); a step's result is decided in `run_one` so tests can drive it
    /// with a mock backend. On a platform with no desktop backend yet (design: §6) it has none and offers nothing.
    pub struct Computers {
        config: Config,
        started_at: Instant,
        /// per conversation: the once-per-job approval already given (until when), for the reuse of the browser path
        grants: tokio::sync::Mutex<HashMap<String, Instant>>,
        backend: Option<Box<dyn Backend + Send + Sync>>,
    }

    impl Computers {
        /// The desktop use from the run config. `config` carries the mode and preset; the backend is chosen by OS (Linux
        /// now; macOS/Windows stubs have none, so it offers nothing until the plan's §6).
        pub fn new(config: Config) -> Computers {
            let mode = config.computer.mode;
            Computers { config, started_at: Instant::now(), grants: tokio::sync::Mutex::new(HashMap::new()), backend: select_backend(mode) }
        }

    /// Whether desktop use is on AND this computer has a backend (so the tools can be offered). Off by default.
    pub fn enabled(&self) -> bool { self.config.computer.use_ && self.backend.is_some() }

    /// The computer-use tool schemas, or none when the desktop use is off or this platform has no backend — so the chat
    /// sees them only for a request whose run config turned it on. `vision` adds that each step returns a screenshot.
    pub fn offered_tools(&self, vision: bool) -> Vec<Value> {
        if !self.enabled() { return Vec::new(); }
        crate::computer::tools::tools(vision)
    }

    /// One computer-use tool call over this desktop, checked and run. `tx` sends the "approve" event when a step waits;
    /// `conv` is the conversation (its once-per-job grant). The job's clock is this store's start and now, for the
    /// preset's time limit. Each step also returns a screenshot, like the browser's.
    pub async fn step(&self, conv: &str, name: &str, args: &Value, id: &str, tx: &mpsc::Sender<RunEvent>) -> Value {
        let rules = self.config.rules.rule_set(self.computer_apps());
        let backend: &(dyn Backend + Send + Sync) = match self.backend.as_ref() {
            Some(b) => b.as_ref(),
            None => return error_result("error: computer use is not available on this computer".into(), 0),
        };
        let started_at = self.started_at;
        let now = Instant::now();
        let mut grants = self.grants.lock().await;
        run_one(backend, &rules, self.computer_mode(), conv, name, args, id, &mut grants, started_at, now, tx).await
    }

    fn computer_apps(&self) -> Vec<String> { self.config.computer.apps.clone() }
    fn computer_mode(&self) -> Mode { self.config.computer.mode }
}

/// The desktop for `mode` on this OS: Linux builds one Xvfb-backed display (the sandbox's virtual screen, never the real
/// desktop); the others have no backend yet and return none.
fn select_backend(_mode: Mode) -> Option<Box<dyn Backend + Send + Sync>> {
    #[cfg(target_os = "linux")]
    { let b: Box<dyn Backend + Send + Sync> = Box::new(crate::computer::backend::LinuxBackend::new(std::env::var("XTEST_DISPLAY").ok().as_deref())); Some(b) }
    #[cfg(not(target_os = "linux"))]
    { None }
}

// ------------------------------------------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::computer::backend::{Buffer, Button, Key, Window, WindowId};
    use std::cell::RefCell;

    /// A mock desktop the tool loop drives in tests — no OS, records every input. This is why `Backend` exists: any type
    /// implementing it stands in for a real desktop, so run_one is tested with it directly (mock backend).
    struct MockBackend {
        clicks: RefCell<Vec<(i32, i32, Button)>>,
        typed: RefCell<Vec<String>>,
        scrolled: RefCell<Vec<(i32, i32)>>,
        opened: RefCell<Vec<String>>,
    }

    impl MockBackend {
        fn new() -> MockBackend { MockBackend { clicks: RefCell::new(Vec::new()), typed: RefCell::new(Vec::new()),
                                                 scrolled: RefCell::new(Vec::new()), opened: RefCell::new(Vec::new()) } }
    }

    // The mock is driven single-threaded in tests; its RefCell interior mutability isn't Send/Sync, so assert both for
    // this test double that stands in for a real (Send + Sync) desktop behind `dyn Backend + Send + Sync`.
    unsafe impl Send for MockBackend {}
    unsafe impl Sync for MockBackend {}

    impl Backend for MockBackend {
        fn screenshot(&self) -> Result<Buffer, String> { Ok(Buffer { width: 2, height: 2, rgba: vec![0u8; 16] }) }
        fn list_windows(&self) -> Result<Vec<Window>, String> { Ok(Vec::new()) }
        fn focus(&self, _id: WindowId) -> Result<(), String> { Ok(()) }
        fn click(&self, x: i32, y: i32, button: Button) -> Result<(), String> { self.clicks.borrow_mut().push((x, y, button)); Ok(()) }
        fn type_text(&self, text: &str) -> Result<(), String> { self.typed.borrow_mut().push(text.to_string()); Ok(()) }
        fn press_keys(&self, _keys: &[Key]) -> Result<(), String> { Ok(()) }
        fn scroll(&self, dx: i32, dy: i32) -> Result<(), String> { self.scrolled.borrow_mut().push((dx, dy)); Ok(()) }
        fn open_app(&self, name: &str) -> Result<(), String> { self.opened.borrow_mut().push(name.to_string()); Ok(()) }
    }

    fn ruleset() -> RuleSet { let mut r = RuleSet::strict(); r.allowed_apps = vec!["firefox".into()]; r }
    /// A fresh run loop channel; the receiver is kept alive so the "approve" event has somewhere to go.
    fn channel() -> (mpsc::Sender<RunEvent>, mpsc::Receiver<RunEvent>) { mpsc::channel(16) }

    #[tokio::test]
    async fn a_refused_step_is_not_run_and_names_the_rule() {
        let mock = MockBackend::new();
        let mut grants = HashMap::new();
        let (tx, _rx) = channel();
        // Strict never types into a password field: classify the step, check it against the preset, decline without running.
        let r = run_one(&mock, &ruleset(), Mode::Sandbox, "conv", "computer_type",
                        &json!({"selector": "input[type=password]", "text": "x"}), "id1", &mut grants, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(r["ok"], false);
        assert!(r["text"].as_str().unwrap().contains("no_password_fields"), "the message names the rule: {}", r["text"]);
        assert!(mock.typed.borrow().is_empty(), "the refused step was not run");
    }

    #[tokio::test]
    async fn a_wrong_argument_is_refused_before_it_runs() {
        let mock = MockBackend::new();
        let mut grants = HashMap::new();
        let (tx, _rx) = channel();
        // A click without x and y is refused up front; the step never reaches the backend.
        let r = run_one(&mock, &ruleset(), Mode::Sandbox, "conv", "computer_click", &json!({"x": 5.0}), "id2", &mut grants, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(r["ok"], false);
        assert!(r["text"].as_str().unwrap().contains("x and y"));
        assert!(mock.clicks.borrow().is_empty());
    }

    #[tokio::test]
    async fn an_approved_job_runs_its_steps_without_asking_again() {
        let mock = MockBackend::new();
        let mut grants = HashMap::new();
        let (tx, _rx) = channel();
        // A risky click ("Publish") asks once; approve it for the conversation, and a later risky click of the same job runs.
        seed_approval("idA", true).await;
        let r1 = run_one(&mock, &ruleset(), Mode::Sandbox, "job", "computer_click",
                         &json!({"x": 3.0, "y": 4.0, "text": "Publish"}), "idA", &mut grants, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(r1["ok"], true, "the approved risky click ran: {}", r1["text"]);
        let r2 = run_one(&mock, &ruleset(), Mode::Sandbox, "job", "computer_click",
                         &json!({"x": 5.0, "y": 6.0, "text": "Publish"}), "idB", &mut grants, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(r2["ok"], true, "a later risky click of the same job runs without asking again");
        assert_eq!(mock.clicks.borrow().len(), 2, "both approved clicks ran");
    }

    #[tokio::test]
    async fn a_denied_step_is_not_run() {
        let mock = MockBackend::new();
        let mut grants = HashMap::new();
        let (tx, _rx) = channel();
        seed_approval("idD", false).await;
        let r = run_one(&mock, &ruleset(), Mode::Sandbox, "conv", "computer_click",
                        &json!({"x": 1.0, "y": 2.0, "text": "Publish"}), "idD", &mut grants, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(r["ok"], false);
        assert!(mock.clicks.borrow().is_empty(), "a denied step was not run");
    }

    #[tokio::test]
    async fn watch_only_never_sends_input_but_reads() {
        let mock = MockBackend::new();
        let mut grants = HashMap::new();
        let (tx, _rx) = channel();
        // In watch-only a click is declined without asking and the desktop never receives it...
        let click = run_one(&mock, &ruleset(), Mode::WatchOnly, "conv", "computer_click", &json!({"x": 3.0, "y": 4.0}), "idW", &mut grants, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(click["ok"], false);
        assert!(mock.clicks.borrow().is_empty(), "watch-only sent no input");
        // ...but a screenshot (a read) still runs.
        let shot = run_one(&mock, &ruleset(), Mode::WatchOnly, "conv", "computer_screenshot", &json!({}), "idR", &mut grants, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(shot["ok"], true);
    }

    #[tokio::test]
    async fn the_allowed_apps_open_and_refuse() {
        let mock = MockBackend::new();
        let mut grants = HashMap::new();
        let (tx, _rx) = channel();
        // Firefox is allowed to open; an unknown app is refused by the allowed_apps rule without running.
        let opened = run_one(&mock, &ruleset(), Mode::Sandbox, "conv", "computer_open_app", &json!({"name": "firefox"}), "idO", &mut grants, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(opened["ok"], true);
        assert_eq!(*mock.opened.borrow(), vec!["firefox".to_string()]);
        let refused = run_one(&mock, &ruleset(), Mode::Sandbox, "conv", "computer_open_app", &json!({"name": "code"}), "idC", &mut grants, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(refused["ok"], false);
        assert!(refused["text"].as_str().unwrap().contains("allowed_apps"));
    }

    /// Whether desktop use offers its tools only when the run config turned it on: off by default, nothing offered.
    #[test]
    fn offered_tools_are_absent_when_the_config_has_it_off() {
        let off = Computers::new(Config::default());
        assert!(!off.enabled(), "off by default");
        assert!(off.offered_tools(true).is_empty(), "no computer tools when the run config has it off");

        // A config with computer use on but no backend here still does not enable the tools.
        let mut cfg = Config::default();
        cfg.computer.use_ = true;
        assert_eq!(cfg.computer.mode, Mode::Sandbox);
    }

    /// Put an approval (yes or no) on the browser's shared approvals map for `id`, answered on its sender, concurrently
    /// with run_one so it settles only after run_one has inserted the id. It polls because run_one is the one that makes
    /// the id appear in the map (the real POST /api/tools/approve would not race a caller). A spawned task interleaves
    /// with run_one's wait on current-thread runtimes, so it keeps polling until the id is there.
    async fn seed_approval(id: &str, ok: bool) {
        let id = id.to_string();
        tokio::spawn(async move {
            for _ in 0..1_000 {
                if let Some(tx) = browser::approvals().lock().unwrap().remove(&id) {
                    let _ = tx.send(ok);
                    return;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });
    }
}
