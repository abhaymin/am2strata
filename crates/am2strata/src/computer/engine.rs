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
// `Backend` and `Size` are portable (every OS); `tool_available` too. `LinuxBackend` exists only on Linux. `SandboxDesktop`
// exists everywhere — the real Xvfb-backed one here, a tiny phantom stub on macOS / Windows — so this engine's fields compile
// on all OSes without an OS check (`backend.rs`).
#[cfg(target_os = "linux")]
use crate::computer::backend::{LinuxBackend, SandboxDesktop};
#[cfg(not(target_os = "linux"))]
use crate::computer::backend::SandboxDesktop;
use crate::computer::backend::{Backend, Size};
use crate::computer::rules::{self, Ctx, Decision, Intent, RuleSet};
use crate::computer::settings::{Config, Mode};
use crate::computer::step::classify_step;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
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
                      id: &str, grants: &mut HashMap<String, Instant>, step_counts: &mut HashMap<String, usize>, max_steps: u32,
                      display_size: Size, started_at: Instant, now: Instant, tx: &mpsc::Sender<RunEvent>) -> Value {
    let t0 = Instant::now();
    let ms = |t0: Instant| t0.elapsed().as_millis() as u64;
    // 1. A call with a missing or wrong argument is refused up front, the step never runs (like browser.rs refuses a
    //    click without x and y), so the model can fix it.
    let step = match classify_step(name, args) {
        Ok(s) => s,
        Err(e) => return error_result(format!("error: the step was not run: {e}"), ms(t0)),
    };
    // 1b. Refuse a click whose coordinates fall outside the sandbox display (item 3), before anything is sent; a click by
    //    window title (no x, y) has no pixel to bounds-check against.
    if name == "computer_click" {
        if let (Some(x), Some(y)) = (step.x, step.y) {
            if x < 0 || y < 0 || x >= display_size.width as i32 || y >= display_size.height as i32 {
                return error_result(format!("error: the click at ({x}, {y}) is outside the {}x{} sandbox display; choose coordinates within it",
                                             display_size.width, display_size.height), ms(t0));
            }
        }
    }
    // 2. In watch-only no input is sent: a click / type / key / scroll / open_app is declined without asking; only a
    //    screenshot (a read) or a wait go on.
    if mode == Mode::WatchOnly && sends_input(name) {
        return error_result("error: the step was not run (watch-only mode does not send input)".to_string(), ms(t0));
    }
    // 3. Classify into an Intent and check it against the active preset. A refusal is never executed and names the rule;
    //    an "ask" waits for approval through the browser's own path.
    let intent = rules::classify(name, args).unwrap_or(Intent::Screenshot);
    let decision = rules.check(&intent, &Ctx::default(), started_at, now);
    match &decision {
        Decision::Refused(rule) => return error_result(format!("error: the step was not run, by rule {rule}"), ms(t0)),
        Decision::Approval(prompt) => {
            let granted = grants.get(conv).is_some_and(|until| now < *until);
            if !granted {
                let (atx, arx) = tokio::sync::oneshot::channel();
                browser::approvals().lock().unwrap().insert(id.to_string(), atx);
                let _ = tx.send(RunEvent::Mcp(json!({"event": "approve", "id": id, "name": name, "what": prompt}))).await;
                let allowed = matches!(tokio::time::timeout(APPROVE_TIMEOUT, arx).await, Ok(Ok(true)));
                browser::approvals().lock().unwrap().remove(id);
                if allowed { grants.insert(conv.to_string(), now + JOB_WINDOW); step_counts.insert(conv.to_string(), 0); /* next approval starts a new job count (item 2) */ }
                if !allowed { return error_result("error: the user did not allow this step (or no one answered); ask them, or do something else".into(), ms(t0)); }
            }
        }
        Decision::Allowed => {}
    }
    // 4. The rules let it through: run it over the desktop. One job (conversation) is capped at `max_steps` steps (item 2):
    //    an ask is how you pass the limit ("the next approval starts a new job count"), so only freely-allowed steps hit this
    //    hard cap; once there, refuse with a message naming the limit until an approval resets it.
    if matches!(decision, Decision::Allowed) && step_counts.get(conv).copied().unwrap_or(0) >= max_steps as usize {
        return error_result(format!("error: this job reached its limit of {max_steps} steps; approve it again to start a new job"), ms(t0));
    }
    *step_counts.entry(conv.to_string()).or_insert(0) += 1;
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
        /// per conversation: how many steps this job has run, capped at `computer.max_steps` (item 2); a fresh approval resets it
        steps: tokio::sync::Mutex<HashMap<String, usize>>,
        backend: Option<Box<dyn Backend + Send + Sync>>,
        /// the sandbox's private Xvfb display, shared with the backend so it can be stopped after 10 minutes idle; `None` otherwise
        desktop: Option<Arc<SandboxDesktop>>,
        /// the idle watcher for the sandbox desktop: it stops the display after 10 min quiet, then frees this slot on exit so a
        /// restarted display gets a fresh one; `None` means none is alive (at most one at a time), and the Arc lets it clear the slot itself.
        watchdog: std::sync::Arc<std::sync::Mutex<Option<std::thread::JoinHandle<()>>>>,
        /// the operator took control in the Computer tab: the agent's steps wait until it is handed back
        human: std::sync::atomic::AtomicBool,
    }

    impl Computers {
        /// The desktop use from the run config. `config` carries the mode and preset; the backend is chosen by OS (Linux
        /// now; macOS/Windows stubs have none, so it offers nothing until the plan's §6).
        pub fn new(config: Config) -> Computers {
            let mode = config.computer.mode;
            let size = Size::parse(config.computer.display_size.as_deref());
            let (backend, desktop) = select_backend(mode, size);
            Computers { config, started_at: Instant::now(), grants: tokio::sync::Mutex::new(HashMap::new()), steps: tokio::sync::Mutex::new(HashMap::new()), backend, desktop, watchdog: Arc::new(std::sync::Mutex::new(None)),
                        human: std::sync::atomic::AtomicBool::new(false) }
        }

    /// Whether desktop use is on AND this computer has a backend (so the tools can be offered). Off by default. On a sandbox
    /// short of Xvfb or xdotool the backend is `None`, so nothing is offered (`server_tools.rs`; item 2) instead of panicking.
    pub fn enabled(&self) -> bool { self.config.computer.use_ && self.backend.is_some() }

    /// The computer-use tool schemas, or none when the desktop use is off or this platform has no backend — so the chat
    /// sees them only for a request whose run config turned it on. `vision` adds that each step returns a screenshot.
    pub fn offered_tools(&self, vision: bool) -> Vec<Value> {
        if !self.enabled() { return Vec::new(); }
        let mut t = crate::computer::tools::tools(vision);
        if self.computer_mode() == Mode::Vm {
            t.extend(crate::computer::tools::vm_tools());
        }
        t
    }

    /// One computer-use tool call over this desktop, checked and run. `tx` sends the "approve" event when a step waits;
    /// `conv` is the conversation (its once-per-job grant). The job's clock is this store's start and now, for the
    /// preset's time limit. Each step also returns a screenshot, like the browser's.
    pub async fn step(&self, conv: &str, name: &str, args: &Value, id: &str, tx: &mpsc::Sender<RunEvent>) -> Value {
        // the operator has control (Computer tab): wait until it is handed back, up to 30 minutes, so the agent never
        // acts on the desktop while a person is using it
        let t0 = Instant::now();
        while self.human_in_control() {
            if t0.elapsed() > Duration::from_secs(30 * 60) || tx.is_closed() {
                return error_result("error: the operator still has control of the desktop; the step was not run".into(), 0);
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        // the sandbox VM's shell and installs: a separate machine where the agent is root, so they run without asking
        if name == "computer_shell" || name == "computer_install" || name == "computer_ads" {
            return self.vm_tool(name, args).await;
        }
        // arm the idle watcher so a sandbox display is stopped after 10 minutes with no step (item 1); a non-sandbox server has none
        self.arm_watchdog();
        let rules = self.config.rules.rule_set(self.computer_apps());
        let backend: &(dyn Backend + Send + Sync) = match self.backend.as_ref() {
            Some(b) => b.as_ref(),
            None => return error_result("error: computer use is not available on this computer".into(), 0),
        };
        let started_at = self.started_at;
        let now = Instant::now();
        let max_steps = self.config.computer.max_steps;
        let disp_size = Size::parse(self.config.computer.display_size.as_deref());
        let mut grants = self.grants.lock().await;
        let mut step_counts = self.steps.lock().await;
        run_one(backend, &rules, self.computer_mode(), conv, name, args, id, &mut grants, &mut step_counts, max_steps, disp_size, started_at, now, tx).await
    }

    /// The sandbox VM, when computer use runs in it (`"mode": "vm"`).
    #[cfg(target_os = "linux")]
    pub fn vm(&self) -> Option<Arc<crate::computer::vm::Vm>> {
        (self.computer_mode() == Mode::Vm).then(crate::computer::vm::get).flatten()
    }

    async fn vm_tool(&self, name: &str, args: &Value) -> Value {
        #[cfg(target_os = "linux")]
        {
            let Some(vm) = self.vm() else {
                return error_result("error: the shell and installs are only in the sandbox VM mode (Settings › Computer and browser use › Mode: Sandbox VM)".into(), 0);
            };
            let (name, args, t0) = (name.to_string(), args.clone(), Instant::now());
            let r = tokio::task::spawn_blocking(move || -> Result<String, String> {
                vm.ensure()?;
                if name == "computer_ads" {
                    let allow: Vec<String> = args["allow"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default();
                    let lists: Vec<String> = args["lists"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default();
                    return vm.set_ads(args["block"].as_bool().unwrap_or(true), &lists, &allow);
                }
                if name == "computer_shell" {
                    let cmd = args["command"].as_str().filter(|c| !c.trim().is_empty()).ok_or("give \"command\"")?;
                    let secs = args["timeout_s"].as_u64().unwrap_or(300).clamp(1, 3600);
                    let (code, out) = vm.sh(cmd, args["root"].as_bool().unwrap_or(false), None, Duration::from_secs(secs), &|_| {})?;
                    let out = if out.len() > 20_000 { format!("…{}", &out[out.len() - 20_000..]) } else { out };
                    Ok(format!("exit {code}\n{out}"))
                } else {
                    let remove = args["remove"].as_bool().unwrap_or(false);
                    let (what, target) = match (args["app"].as_str(), args["kind"].as_str()) {
                        (Some(a), _) => (a.to_string(), String::new()),
                        (None, Some(k)) => (k.to_string(), args["target"].as_str().unwrap_or("").to_string()),
                        _ => return Err("give \"app\" (a catalog id) or \"kind\" and \"target\"".into()),
                    };
                    vm.install(&what, &target, !remove, &|_| {})
                }
            }).await.unwrap_or_else(|e| Err(e.to_string()));
            match r {
                Ok(s) => json!({"ok": true, "chars": s.chars().count(), "text": s, "truncated": false, "ms": t0.elapsed().as_millis() as u64}),
                Err(e) => error_result(format!("error: {e}"), t0.elapsed().as_millis() as u64),
            }
        }
        #[cfg(not(target_os = "linux"))]
        { let _ = (name, args); error_result("error: the sandbox VM needs Linux".into(), 0) }
    }

    /// The Computer tab's view: the sandbox desktop (started if it is not running), when this server has one.
    pub fn sandbox_display(&self) -> Result<String, String> {
        if !self.config.computer.use_ {
            return Err("computer use is off (Settings › Computer and browser use)".into());
        }
        let d = self.desktop.as_ref().ok_or("this computer has no sandbox desktop (computer use mode is not \"sandbox\", or Xvfb/xdotool are missing)")?;
        let disp = d.ensure()?;
        self.arm_watchdog();
        Ok(disp)
    }

    /// Keep the sandbox desktop from being stopped as idle while someone watches it.
    pub fn touch(&self) {
        if let Some(d) = &self.desktop {
            let _ = d.ensure();
        }
        #[cfg(target_os = "linux")]
        if let Some(vm) = self.vm() {
            vm.touch();
        }
    }

    /// The server is stopping: stop the sandbox desktop (its Xvfb and everything on it).
    pub fn shutdown(&self) {
        if let Some(d) = &self.desktop {
            d.stop();
        }
    }

    pub fn human_in_control(&self) -> bool { self.human.load(std::sync::atomic::Ordering::SeqCst) }
    pub fn set_human(&self, on: bool) { self.human.store(on, std::sync::atomic::Ordering::SeqCst) }
    pub fn mode_name(&self) -> &'static str { self.config.computer.mode.as_str() }
    pub fn apps(&self) -> Vec<String> { self.config.computer.apps.clone() }
    pub fn is_on(&self) -> bool { self.config.computer.use_ }

    fn computer_apps(&self) -> Vec<String> { self.config.computer.apps.clone() }
    fn computer_mode(&self) -> Mode { self.config.computer.mode }

    /// Arm the idle watcher (item 1): after 10 minutes with no computer step the sandbox display is stopped. Called on every step, so a
    /// display that restarted after an idle stop (which cleared `stopped`, backend.rs) gets a fresh watcher — at most one alive at a time.
    fn arm_watchdog(&self) {
        if let Some(desktop) = &self.desktop {
            start_watchdog(desktop.clone(), IDLE_TIMEOUT, self.watchdog.clone());
        }
    }
}

/// The idle watcher's quiet period: after 10 minutes with no computer step, the sandbox display is stopped (item 1).
const IDLE_TIMEOUT: Duration = Duration::from_secs(10 * 60);

/// Start a fresh idle watcher into `slot` only when none is already running, keeping at most one alive at a time (item 1): the previous
/// watcher frees this slot on exit, so a restarted display (whose `stopped` was cleared) can arm a new one. Returns false when one is live.
fn start_watchdog(desktop: Arc<SandboxDesktop>, period: Duration, slot: std::sync::Arc<std::sync::Mutex<Option<std::thread::JoinHandle<()>>>>) -> bool {
    // lock a clone so the closure can move `slot` (same underlying mutex), not the one the guard borrows.
    let shared = slot.clone();
    let mut w = shared.lock().unwrap();
    if w.is_some() { return false; }   // a watcher is already alive; don't start a second one (at most one at a time)
    *w = Some(std::thread::spawn(move || idle_watcher(desktop, period, slot)));
    true
}

/// Watches a sandbox desktop until it has been quiet for `period`, then stops it — and as soon as it is already stopping. The watcher frees
/// `slot` on exit so a restarted display can arm a fresh one (item 1); `period` is a parameter so tests drive this in milliseconds, and the
/// original bug was that idle only broke the loop, leaving Xvfb running forever with no watcher after a restart.
fn idle_watcher(desktop: Arc<SandboxDesktop>, period: Duration, slot: std::sync::Arc<std::sync::Mutex<Option<std::thread::JoinHandle<()>>>>) {
    loop {
        std::thread::sleep(period);
        if desktop.stopped.load(std::sync::atomic::Ordering::Relaxed) { break; }       // already stopping, nothing to do
        if desktop.is_idle(period) { desktop.stop(); break; }                         // idle the whole period → stop the display
    }
    // this watcher has exited: free the slot so the next (re)start arms a fresh one instead of leaving the display with no watcher (item 1).
    *slot.lock().unwrap() = None;
}

/// The desktop for `mode` on this OS: the sandbox builds a private Xvfb-backed display (never the real desktop), the others use
/// a fixed one (`XTEST_DISPLAY`); when the sandbox is short of Xvfb or xdotool it has no backend, so the tools are not offered
/// (`server_tools.rs`, item 2). Returns the backend and its Arc so the engine can stop the display after idle.
fn select_backend(_mode: Mode, _size: Size) -> (Option<Box<dyn Backend + Send + Sync>>, Option<Arc<SandboxDesktop>>) {
    #[cfg(target_os = "linux")]
    {
        if _mode == Mode::Vm {
            // the sandbox VM (vm/): available once the server made it (main.rs, before this)
            return match crate::computer::vm::get() {
                Some(vm) => (Some(Box::new(crate::computer::vm::backend::VmBackend { vm })), None),
                None => (None, None),
            };
        }
        if _mode == Mode::Sandbox && !(crate::computer::backend::tool_available("Xvfb") && crate::computer::backend::tool_available("xdotool")) {
            return (None, None);  // the sandbox needs both Xvfb and xdotool; without them the desktop tools are not offered
        }
        let desktop = if _mode == Mode::Sandbox { Some(Arc::new(SandboxDesktop::new(_size))) } else { None };
        let backend: Box<dyn Backend + Send + Sync> = if _mode == Mode::Sandbox {
            Box::new(LinuxBackend::new_sandbox(_size))
        } else {
            Box::new(LinuxBackend::new(std::env::var("XTEST_DISPLAY").ok().as_deref()))
        };
        (Some(backend), desktop)
    }
    #[cfg(not(target_os = "linux"))]
    { (None, None) }
}

// ------------------------------------------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    /// While the operator has the desktop (the Computer tab's "Take control"), the agent's steps wait; handing it back
    /// lets them go on. (What the step then does depends on this machine: what matters is when it runs.)
    #[tokio::test]
    async fn the_agent_waits_while_the_operator_has_control() {
        let c = std::sync::Arc::new(Computers::new(Config::default()));
        c.set_human(true);
        let (tx, _rx) = tokio::sync::mpsc::channel(4);
        let c2 = c.clone();
        let t0 = Instant::now();
        let step = tokio::spawn(async move { c2.step("conv", "computer_screenshot", &json!({}), "id", &tx).await });
        tokio::time::sleep(Duration::from_millis(1200)).await;
        assert!(!step.is_finished(), "the step waits while the operator has control");
        c.set_human(false);
        let r = tokio::time::timeout(Duration::from_secs(5), step).await.expect("it goes on once handed back").unwrap();
        assert!(t0.elapsed() >= Duration::from_millis(1200));
        // it ran only after the hand-back (with or without a desktop on this machine)
        assert!(r.is_object(), "{r}");
    }

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
    /// The per-job step cap used by the existing tests: large enough that only the dedicated limit tests hit it (item 2).
    const LIMIT: u32 = crate::computer::settings::DEFAULT_MAX_STEPS;
    /// A comfortably-large sandbox display so the existing small-coordinate clicks are always in bounds (item 3).
    const DEFAULT_DISPLAY: Size = Size::DEFAULT;

    #[tokio::test]
    async fn a_refused_step_is_not_run_and_names_the_rule() {
        let mock = MockBackend::new();
        let mut grants = HashMap::new();
        let mut steps = HashMap::new();
        let (tx, _rx) = channel();
        // Strict never types into a password field: classify the step, check it against the preset, decline without running.
        let r = run_one(&mock, &ruleset(), Mode::Sandbox, "conv", "computer_type",
                        &json!({"selector": "input[type=password]", "text": "x"}), "id1", &mut grants, &mut steps, LIMIT, DEFAULT_DISPLAY, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(r["ok"], false);
        assert!(r["text"].as_str().unwrap().contains("no_password_fields"), "the message names the rule: {}", r["text"]);
        assert!(mock.typed.borrow().is_empty(), "the refused step was not run");
    }

    #[tokio::test]
    async fn a_wrong_argument_is_refused_before_it_runs() {
        let mock = MockBackend::new();
        let mut grants = HashMap::new();
        let mut steps = HashMap::new();
        let (tx, _rx) = channel();
        // A click without x and y is refused up front; the step never reaches the backend.
        let r = run_one(&mock, &ruleset(), Mode::Sandbox, "conv", "computer_click", &json!({"x": 5.0}), "id2", &mut grants, &mut steps, LIMIT, DEFAULT_DISPLAY, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(r["ok"], false);
        assert!(r["text"].as_str().unwrap().contains("x and y"));
        assert!(mock.clicks.borrow().is_empty());
    }

    #[tokio::test]
    async fn an_approved_job_runs_its_steps_without_asking_again() {
        let mock = MockBackend::new();
        let mut grants = HashMap::new();
        let mut steps = HashMap::new();
        let (tx, _rx) = channel();
        // A risky click ("Publish") asks once; approve it for the conversation, and a later risky click of the same job runs.
        seed_approval("idA", true).await;
        let r1 = run_one(&mock, &ruleset(), Mode::Sandbox, "job", "computer_click",
                         &json!({"x": 3.0, "y": 4.0, "text": "Publish"}), "idA", &mut grants, &mut steps, LIMIT, DEFAULT_DISPLAY, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(r1["ok"], true, "the approved risky click ran: {}", r1["text"]);
        let r2 = run_one(&mock, &ruleset(), Mode::Sandbox, "job", "computer_click",
                         &json!({"x": 5.0, "y": 6.0, "text": "Publish"}), "idB", &mut grants, &mut steps, LIMIT, DEFAULT_DISPLAY, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(r2["ok"], true, "a later risky click of the same job runs without asking again");
        assert_eq!(mock.clicks.borrow().len(), 2, "both approved clicks ran");
    }

    #[tokio::test]
    async fn a_denied_step_is_not_run() {
        let mock = MockBackend::new();
        let mut grants = HashMap::new();
        let mut steps = HashMap::new();
        let (tx, _rx) = channel();
        seed_approval("idD", false).await;
        let r = run_one(&mock, &ruleset(), Mode::Sandbox, "conv", "computer_click",
                        &json!({"x": 1.0, "y": 2.0, "text": "Publish"}), "idD", &mut grants, &mut steps, LIMIT, DEFAULT_DISPLAY, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(r["ok"], false);
        assert!(mock.clicks.borrow().is_empty(), "a denied step was not run");
    }

    #[tokio::test]
    async fn watch_only_never_sends_input_but_reads() {
        let mock = MockBackend::new();
        let mut grants = HashMap::new();
        let mut steps = HashMap::new();
        let (tx, _rx) = channel();
        // In watch-only a click is declined without asking and the desktop never receives it...
        let click = run_one(&mock, &ruleset(), Mode::WatchOnly, "conv", "computer_click", &json!({"x": 3.0, "y": 4.0}), "idW", &mut grants, &mut steps, LIMIT, DEFAULT_DISPLAY, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(click["ok"], false);
        assert!(mock.clicks.borrow().is_empty(), "watch-only sent no input");
        // ...but a screenshot (a read) still runs.
        let shot = run_one(&mock, &ruleset(), Mode::WatchOnly, "conv", "computer_screenshot", &json!({}), "idR", &mut grants, &mut steps, LIMIT, DEFAULT_DISPLAY, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(shot["ok"], true);
    }

    #[tokio::test]
    async fn the_allowed_apps_open_and_refuse() {
        let mock = MockBackend::new();
        let mut grants = HashMap::new();
        let mut steps = HashMap::new();
        let (tx, _rx) = channel();
        // Firefox is allowed to open; an unknown app is refused by the allowed_apps rule without running.
        let opened = run_one(&mock, &ruleset(), Mode::Sandbox, "conv", "computer_open_app", &json!({"name": "firefox"}), "idO", &mut grants, &mut steps, LIMIT, DEFAULT_DISPLAY, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(opened["ok"], true);
        assert_eq!(*mock.opened.borrow(), vec!["firefox".to_string()]);
        let refused = run_one(&mock, &ruleset(), Mode::Sandbox, "conv", "computer_open_app", &json!({"name": "code"}), "idC", &mut grants, &mut steps, LIMIT, DEFAULT_DISPLAY, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(refused["ok"], false);
        assert!(refused["text"].as_str().unwrap().contains("allowed_apps"));
    }

    /// Whether desktop use offers its tools only when the run config turned it on: off by default, nothing offered.
    #[test]
    fn offered_tools_are_absent_when_the_config_has_it_off() {
        let off = Computers::new(Config::default());
        assert!(!off.enabled(), "off by default");
        assert!(off.offered_tools(true).is_empty(), "no computer tools when the run config has it off");

        // A config with computer use on but no backend here still does not enable the tools: in the sandbox, whether the tools
        // are offered depends on Xvfb + xdotool being installed (item 2) — assert it either way so this stays honest on any box.
        let mut cfg = Config::default();
        cfg.computer.use_ = true;
        assert_eq!(cfg.computer.mode, Mode::Sandbox);
        let c = Computers::new(cfg);
        if crate::computer::backend::tool_available("Xvfb") && crate::computer::backend::tool_available("xdotool") {
            assert!(c.enabled(), "the sandbox offers tools when Xvfb and xdotool are installed");
        } else {
            assert!(!c.enabled(), "the sandbox offers nothing when Xvfb/xdotool are missing (item 2)");
        }
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

    /// End-to-end through the engine on a real (sandbox) desktop: start the private Xvfb via `Computers.step`, click an area and take
    /// a screenshot over the backend — the whole point of starting a display. Ignored by default (this box has no Xvfb); bails early
    /// when Xvfb/xdotool are absent, like the ignored backend test. Run with `cargo test -- --ignored`.
    #[cfg(target_os = "linux")]
    #[ignore = "needs Xvfb + xdotool on a running session"]
    #[tokio::test]
    async fn sandbox_engine_starts_a_display_clicks_and_screenshots() -> Result<(), String> {
        // build the server's desktop from a sandbox config; if its tools are absent there is no backend to drive.
        let mut cfg = Config::default();
        cfg.computer.use_ = true;
        cfg.computer.mode = Mode::Sandbox;
        cfg.computer.apps = vec!["xterm".into()];
        let computers = Computers::new(cfg);
        if !computers.enabled() { return Ok(()); }  // no Xvfb/xdotool on this box: there is nothing to start
        let (tx, _rx) = channel();

        // a plain click is Allowed by the preset (not risky), so it runs without an approval wait and starts the private display
        let click = computers.step("job", "computer_click", &json!({"x": 3.0, "y": 4.0}), "idE", &tx).await;
        assert_eq!(click["ok"], true, "a sandbox click runs through the engine and starts Xvfb: {}", click["text"]);

        // a screenshot returns the frame from the virtual display; its presence depends on a screenshot tool being installed
        let shot = computers.step("job", "computer_screenshot", &json!({}), "idS", &tx).await;
        assert_eq!(shot["ok"], true, "a sandbox screenshot reads the virtual display: {}", shot["text"]);
        if crate::computer::backend::tool_available("convert") || crate::computer::backend::tool_available("import") || crate::computer::backend::tool_available("scrot") {
            assert!(shot.get("image").is_some(), "the screenshot carries a base64 JPEG for the model");
        }
        Ok(())
    }

    /// One job is capped at its configured step count (item 2): freely-allowed steps run until the limit, then are refused
    /// with a message naming it; they stay refused until an approval resets the count.
    #[tokio::test]
    async fn one_job_is_capped_at_its_step_limit() {
        let mock = MockBackend::new();
        let mut grants = HashMap::new();
        let mut steps = HashMap::new();
        let (tx, _rx) = channel();
        // ordinary clicks are Allowed by the strict preset; run them up to and past a limit of 2.
        for i in 0..3 {
            let r = run_one(&mock, &ruleset(), Mode::Sandbox, "job", "computer_click",
                            &json!({"x": 1.0 + i as f64, "y": 2.0}), "idC", &mut grants, &mut steps, 2, DEFAULT_DISPLAY, Instant::now(), Instant::now(), &tx).await;
            if i < 2 { assert_eq!(r["ok"], true, "the first {} allowed steps run", i); }
            else { assert_eq!(r["ok"], false, "a step at the limit is refused"); assert!(r["text"].as_str().unwrap().contains("limit of 2"), "{}", r["text"]); }
        }
        assert_eq!(mock.clicks.borrow().len(), 2, "only the two steps under the cap actually ran");
    }

    /// A user approval resets the job's count ("the next approval starts a new job count", item 2): after freely-allowed steps
    /// reach the plain cap of 2, approving one risky click resets it and lets more allowed steps run again.
    #[tokio::test]
    async fn an_approved_reset_allows_more_steps_than_the_plain_cap() {
        let mock = MockBackend::new();
        let mut grants = HashMap::new();
        let mut steps = HashMap::new();
        let (tx, _rx) = channel();
        seed_approval("idReset", true).await;  // approve the risky click so its approval resets the count
        let r1 = run_one(&mock, &ruleset(), Mode::Sandbox, "job", "computer_click", &json!({"x": 1.0, "y": 2.0}), "id1", &mut grants, &mut steps, 2, DEFAULT_DISPLAY, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(r1["ok"], true);
        let r2 = run_one(&mock, &ruleset(), Mode::Sandbox, "job", "computer_click", &json!({"x": 1.0, "y": 2.0}), "id2", &mut grants, &mut steps, 2, DEFAULT_DISPLAY, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(r2["ok"], true);
        // a risky click ("Publish") asks; approving it resets the count to start a new job.
        let r3 = run_one(&mock, &ruleset(), Mode::Sandbox, "job", "computer_click", &json!({"x": 1.0, "y": 2.0, "text": "Publish"}), "idReset", &mut grants, &mut steps, 2, DEFAULT_DISPLAY, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(r3["ok"], true, "the approved risky click ran and reset the count: {}", r3["text"]);
        // two more allowed steps run past the plain cap before the cap bites again.
        let r4 = run_one(&mock, &ruleset(), Mode::Sandbox, "job", "computer_click", &json!({"x": 1.0, "y": 2.0}), "id4", &mut grants, &mut steps, 2, DEFAULT_DISPLAY, Instant::now(), Instant::now(), &tx).await;
        let r5 = run_one(&mock, &ruleset(), Mode::Sandbox, "job", "computer_click", &json!({"x": 1.0, "y": 2.0}), "id5", &mut grants, &mut steps, 2, DEFAULT_DISPLAY, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(r4["ok"], true, "after a reset an allowed step runs again");
        assert_eq!(r5["ok"], false, "the cap bites once the count reaches it again");
        assert!(r5["text"].as_str().unwrap().contains("limit of 2"), "{}", r5["text"]);
        assert_eq!(mock.clicks.borrow().len(), 4, "reset let the job run past its plain cap of 2");
    }

    /// A click whose coordinates fall outside the sandbox display's size is refused with a message naming the size, before
    /// anything is sent to the backend (item 3). Within bounds an ordinary click runs.
    #[tokio::test]
    async fn a_click_outside_the_sandbox_display_is_refused_before_it_runs() {
        let mock = MockBackend::new();
        let mut grants = HashMap::new();
        let mut steps = HashMap::new();
        let (tx, _rx) = channel();
        let small = Size { width: 100, height: 100, depth: 24 };
        // inside the bounds, an ordinary click runs.
        let ok = run_one(&mock, &ruleset(), Mode::Sandbox, "job", "computer_click",
                         &json!({"x": 50.0, "y": 60.0}), "ok1", &mut grants, &mut steps, LIMIT, small, Instant::now(), Instant::now(), &tx).await;
        assert_eq!(ok["ok"], true, "{}", ok["text"]);
        // outside on either axis, or negative, is refused before anything is sent, naming the display size.
        for (tag, x, y) in [("x-axis", 200.0, 50.0), ("y-axis", 50.0, 300.0), ("negative", -1.0, 5.0)] {
            let r = run_one(&mock, &ruleset(), Mode::Sandbox, "job", "computer_click",
                            &json!({"x": x, "y": y}), "oob1", &mut grants, &mut steps, LIMIT, small, Instant::now(), Instant::now(), &tx).await;
            assert_eq!(r["ok"], false, "{tag}: an out-of-bounds click is refused");
            assert!(r["text"].as_str().unwrap().contains("outside the 100x100 sandbox display"), "{tag}: {}", r["text"]);
        }
        // only the one in-bounds click reached the backend: out-of-bounds clicks never send anything.
        assert_eq!(mock.clicks.borrow().len(), 1, "only the in-bounds click ran");
    }

    /// A quiet sandbox display is stopped by the idle watcher within its period — not just left running once it breaks out of the
    /// loop (item 1). A short period is passed so the test runs in milliseconds and needs no Xvfb (`new` marks it last-used now, so
    /// it reads idle once the quiet period elapses; watch `stopped` flip). The watcher clears its slot on exit, so the same slot can arm
    /// a fresh one for a restarted display (see the next test).
    // The sandbox display is a Linux-only Xvfb concept (the engine's `Computers.desktop` slot stays `None` on other OSes), so these
    // watcher tests — which build one and drive the idle stop — run only where there is a real `SandboxDesktop`. On macOS / Windows the
    // phantom stub would let them compile, but they test Linux display semantics, so gate them.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_idle_watcher_stops_a_quiet_sandbox() {
        let desktop = Arc::new(SandboxDesktop::new(DEFAULT_DISPLAY));
        let check = desktop.clone();
        let slot: std::sync::Arc<std::sync::Mutex<Option<std::thread::JoinHandle<()>>>> = std::sync::Arc::new(std::sync::Mutex::new(None));
        assert!(start_watchdog(desktop.clone(), Duration::from_millis(10), slot.clone()));
        for _ in 0..200 { if check.stopped.load(std::sync::atomic::Ordering::Relaxed) { break; } std::thread::sleep(Duration::from_millis(10)); }
        assert!(check.stopped.load(std::sync::atomic::Ordering::Relaxed), "the idle watcher stopped the quiet sandbox display");
    }

    /// At most one idle watcher is alive at a time (item 1): while a watcher runs, arming again does not start a second one, even if it
    /// has just stopped the display; and once it exits the slot frees so a restarted display can arm a fresh watcher that stops it again.
    // The sandbox display is Linux-only, so — like the previous test — these are gated to run only where a real `SandboxDesktop` exists.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_watcher_stops_itself_at_most_once_then_rearms() {
        // a single desktop with one watcher: arming while it is alive starts only that one (the second call returns false).
        let d1 = Arc::new(SandboxDesktop::new(DEFAULT_DISPLAY));
        let s1: std::sync::Arc<std::sync::Mutex<Option<std::thread::JoinHandle<()>>>> = std::sync::Arc::new(std::sync::Mutex::new(None));
        assert!(start_watchdog(d1.clone(), Duration::from_millis(10), s1.clone()));
        assert!(!start_watchdog(d1.clone(), Duration::from_millis(10), s1.clone()), "a watcher already running blocks a second one");
        assert!(s1.lock().unwrap().is_some(), "exactly one watcher handle is tracked");

        // once it has stopped and exited, the slot frees; a restarted display (stopped cleared) arms a fresh watcher that stops it again.
        let d2 = Arc::new(SandboxDesktop::new(DEFAULT_DISPLAY));
        let s2: std::sync::Arc<std::sync::Mutex<Option<std::thread::JoinHandle<()>>>> = std::sync::Arc::new(std::sync::Mutex::new(None));
        assert!(start_watchdog(d2.clone(), Duration::from_millis(10), s2.clone()));
        for _ in 0..200 { if d2.stopped.load(std::sync::atomic::Ordering::Relaxed) { break; } std::thread::sleep(Duration::from_millis(10)); }
        assert!(d2.stopped.load(std::sync::atomic::Ordering::Relaxed), "the first watcher stopped the display");
        // wait for it to exit and free the slot, so re-arming arms a genuinely fresh watcher.
        for _ in 0..200 { if s2.lock().unwrap().is_none() { break; } std::thread::sleep(Duration::from_millis(10)); }
        d2.stopped.store(false, std::sync::atomic::Ordering::Relaxed);   // a step restarted the display (ensure clears `stopped`, backend.rs)
        assert!(start_watchdog(d2.clone(), Duration::from_millis(10), s2.clone()), "a fresh watcher arms after the old one freed its slot");
        for _ in 0..200 { if d2.stopped.load(std::sync::atomic::Ordering::Relaxed) { break; } std::thread::sleep(Duration::from_millis(10)); }
        assert!(d2.stopped.load(std::sync::atomic::Ordering::Relaxed), "the fresh watcher stops the restarted display again");
    }
}
