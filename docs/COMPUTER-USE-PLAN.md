# Computer Use (Task 7) — Plan

Full computer + browser use, switched on in Settings, with rule presets that are enforced **in the server**, not left
to the model. This builds on `crates/am2strata/src/browser.rs` (the existing headless-Chrome computer use). The same
working rules apply: `.agent/bin/cargo` for everything (it caps memory; stock cargo 1.93 is there), tests only, each
step committed with its own tests.

The one thing that makes this safe — and the part we build first — is the **rules engine**: a pure module with unit
tests, covering every preset, password/payment field detection, the app allow list and the 30-minute job time limit.
Everything below the rules engine (the desktop tools, the OS backends, the Settings UI) depends on it.

## Current state (this branch's parent line)

`browser.rs` already drives one headless Chrome over the DevTools protocol, one page per conversation:
- Tools `TOOL_NAMES` = `browser_open/click/type/scroll/back/read/upload`. Offered by `server_tools.rs::offer()` when a
  request asks (`strata_browser: true`) or its client profile has `"browser"`, never for another web page.
- Private addresses blocked; risky steps (password field, form submit, publishing click, upload) ask via
  `Browsers::risk()` and the approvals map (`POST /api/tools/approve`); `"browser": {"approve": "each"|"job"}` repeats
  only the publishing ones once per job (the `JobMemory`, `publishing_job_allows`, `remember_publishing` in `run.rs`).
- Settings parse from `"browser": {...}` (`Settings::from_config`). Tools offered only when available.

## What to build

### 1. Settings › Computer use / Browser use (web app + run config), all off by default

Run config keys (parsed by `Settings::from_config` in a new module, tested via round-trip):

```jsonc
{
  "computer": {
    "use": false,                    // the whole feature, off by default
    "mode": "sandbox",               // sandbox | watch-only | assist | trusted-job (ignored if use==false)
    "apps": ["firefox", "libreoffice"], // allowed apps for the sandbox desktop; required in Strict
    "upload_dir": "/home/<me>/am2strata-uploads", // reused by browser too
    "download_ask": true,            // downloads ask (sandbox default) vs auto-accept into download folder
    "site_allow": [], "site_block": [] // browser allow/block lists
  },
  "browser": {
    "use": false,                    // off by default; when true the existing chrome tools are offered again
    "connect": "",                   // "" = own headless (default), else the user's Chrome endpoint
    "auto_approve": false,           // existing
    "approve": "each",               // existing (each|job)
    "upload_dir": null,              // existing
    "step_timeout_s": 45             // existing
  },
  "rules": { "preset": "strict" | "creator" | "custom", "custom": { ... } }  // shared by both
}
```

Web app (Settings): a **Computer use** switch and a **Browser use** switch, each `false` at first render. Turning on
Computer use shows the mode selector (Sandbox / Watch only / Assist / Trusted job), the allowed-apps editor and the
download ask control; turning on Browser use shows headless-vs-`connect`, the download ask and the upload folder (as it
does now). The **Rules** panel lists the three presets with one-line descriptions, lets the user pick one, and for
`custom` exposes each rule as an on/off toggle. Saving posts to `/config` (the `apply` in `settings.rs`) — these become
editable keys alongside `sampling.*`. Every new key is registered there so both GET and POST validate it.

The "Use a browser" chat flag stays `"strata_browser": true`. New chat flags: `"strata_computer": true` (offer the
desktop tools). The offering in `server_tools.rs::offer()` checks `computer.use` / `browser.use`, not just availability,
so an off setting hides the tools.

### 2. Rule presets

The engine (`crates/am2strata/src/computer/rules.rs`) classifies a step into an intent and returns a decision:

```rust
enum Decision { Allowed, Approval(String), Refused(&'static str) }   // refused carries the rule's name
```

- **Strict** (default): apps allow-list required and enforced (`allowed_apps`); never type into a password or payment
  field; never sign in; never open terminal / password manager / banking / system-settings apps; *ask* (reused approval)
  before anything that sends, deletes, installs, buys or publishes; max 30 min per job (a step past it is refused with
  `time_limit`); every step logged with a screenshot. A rule broken is **refused** and the name is in the error, e.g.
  `error: rules (forbidden_apps): cannot open 'Terminal'`.
- **Creator** (owner-style): Strict, plus the browser may upload from the upload folder and publish on listed sites after
  the job's yes (so a Creator publishing click only *asks*, it is not hard-refused like Strict). Everything else identical.
- **Custom**: every rule above is an independent toggle the user set; empty apps list is allowed (then no app is openable,
  or all are, per the toggle) and each `Refused` path is reachable only when its toggle is on.

Enforcement lives in the server: `run.rs` calls `rules.check(&step)` before every computer step — `Allowed` runs it,
`Approval(name)` runs the same waiting flow as the browser, `Refused(rule)` returns immediately with that rule's name. The
model can never do something the rules forbid even if it keeps asking.

### 3. Desktop tools (offered only when Computer use is on)

`computer_screenshot`, `computer_click {x,y,button}`, `computer_type {text}`, `computer_key {keys}`, `computer_scroll
{dx,dy}`, `computer_open_app {name}` (allowed list only), `computer_wait {seconds}`. Each result carries a screenshot when
the model reads pictures, else the window list and focused window title (from `xwininfo -tree`, present on this PC).

A backend is chosen by mode and OS and hidden behind one trait so it never leaks into the tool layer or the tests:

```rust
trait Backend { // in computer/backend.rs
    fn screenshot(&self) -> Result<Buffer, String>;
    fn list_windows(&self) -> Result<Vec<Window>, String>;
    fn focus(&self, id: WindowId) -> Result<(), String>;
    fn click(&self, x: i32, y: i32, button: Button) -> Result<(), String>;
    fn type_text(&self, text: &str) -> Result<(), String>;
    fn press_keys(&self, keys: &[Key]) -> Result<(), String>;
    fn scroll(&self, dx: i32, dy: i32) -> Result<(), String>;
    fn open_app(&self, name: &str) -> Result<(), String>;
}
```

- **Linux** (this PC — GNOME on Wayland): `Watch only` reads the real desktop's frames through the xdg-desktop-portal
  RemoteDesktop + ScreenCast portals (user approves once in the system dialog); input is ignored. `Sandbox` launches a
  virtual display with Xvfb (`Xvfb :99 -screen 0 1280x800x24`) or, if present, a headless compositor (`weston
  --backend=headless` / `cage`), exposing only the listed apps; input goes through XTest on that display. `Assist` and
  `Trusted job` drive the real desktop (Wayland portal for input too, or X11 XTest when running under Xorg/Xvfb). Screenshots
  are taken with `xwd`/`scrot`/portal capture and decoded by the already-present `image` crate. Input is issued by shelling
  out to standard tools on the chosen display (`XTEST_DISPLAY`, `xdotool`), which keeps the dependency list minimal and matches
  how `browser.rs` launches Chrome — no new OS crates, so it builds offline with the capped cargo. (An `x11rb` XTest backend is
  noted as a future refinement in §6; shelling out is enough to satisfy §2–§5.)
- **macOS** (`CGEvent` + ScreenCaptureKit, needs Accessibility permission): designed in `computer/backend.rs` with a stub that
  refuses with the permission message; build if time allows.
- **Windows** (`SendInput` + DXGI capture): designed as a stub that refuses with "needs the Screen Recording / Accessibility
  permission"; build if time allows.

Both stubs share the same trait and return `Err("computer use is not supported on this platform yet (design: §6)")`, so the
tool layer, tests and Settings work unchanged everywhere.

### 4. Safety built in

- A visible **"am2Strata is using the computer"** indicator in the web app (a small pill near the chat) plus a **Stop** button
  that ends the job at once; a **stop hotkey** (Ctrl+Shift+Q by default, configurable) sends `POST /api/computer/stop`. Both end
  the current run's tool loop exactly as the existing Stop does for the browser.
- The rules check runs before every step (§2).
- The approval flow is reused from the browser (`ApproveMode`, the approvals map, `/api/tools/approve`, the web UI in
  `app.js` lines ~1572–1600): Allow / Deny / Allow for this job.
- An **audit log per job**: one file under `<data>/computer-jobs/<job-id>.log`, each line `{time, tool, decision, rule, screenshot_path}`.

### 5. Browser completeness (same presets, on top of the existing chrome tools)

`browser_tabs {action: list|switch|close}`, `browser_select {selector, value}`, `browser_wait {for: "text"}` (waits until the
text is on the page or times out), downloads to a folder with ask/deny (replaces the current hard Deny with
`download_ask`), and the site allow/block lists (`computer.site_allow` / `computer.site_block`). These reuse the same
presets and approval flow; offered only when `browser.use`.

### 6. Tests

Pure rules engine (`computer/rules.rs`, std-only):
- Each preset: an allowed step passes, each forbidden step is refused with the correct rule name (password field →
  `no_password_fields`; payment → `no_payment_fields`; open Terminal → `forbidden_apps`; empty apps list in Strict →
  `allowed_apps`; time-limit exceeded → `time_limit`).
- Password-field and payment detection from a `type`/`selector` argument (`"input[password]"`, `"card-number"`, `"cvc"`).
- App allow list: allowed app opens, unknown app refused, forbidden categories always refused.
- Time limit: `JobClock` (injectable `Instant`) allows under 30 min, refuses after.

Other tests: the computer-job approval memory mirrors `JobMemory`'s (per conversation+site grant, 30 min); the settings round
trip (`from_config` → stored value → `from_config` again) for `computer.*`, `browser.use`, and the preset; a sandbox-desktop
test **ignored by default** (`#[ignore = "needs Xvfb + an X client"]`) that starts `Xvfb :99`, opens an allowed app, takes a
screenshot and clicks — it is skipped here because this box has no Xvfb/xdotool.

## Build order (one commit per step, each with its tests green)

1. **Rules engine** (`computer/rules.rs` + `computer/mod.rs`) and its tests. No new deps; builds offline. *(this run)*
2. **Settings**: `ComputerUse`, `BrowserUse`, `Rules` parsing in a new module; register the editable keys in `settings.rs`;
   round-trip test. *(this run, depends on 1)*
3. **Desktop backend trait + Linux implementation** (spawn Xvfb/xdotool/xwd), macOS/Windows stubs; tool registry
   (`TOOL_NAMES`), tool JSON, and a classifier that turns `(tool, args)` into a `ComputerStep`. Unit-tested with a mock
   backend (no OS needed). *(next run)*
4. **Wiring**: offer computer tools when `computer.use`; in `run.rs` run the rules check before every computer step
   (`Allowed` / `Approval(name)` via reused approval / `Refused(rule)`); audit log; indicator + Stop route + hotkey.
5. **Browser completeness** (tabs/select/wait/downloads/lists).
6. **Web app UI**: Computer/Browser switches, mode + apps editor, Rules panel, the "using the computer" pill and Stop button.

## Next steps (after this run)

This run builds the rules engine and the settings wiring (§1–§2), the parts with zero new dependencies that also make the rest
safe to build. Remaining:

- §3 desktop backend trait + Linux Xvfb/xdotool implementation + Wayland portal capture (input needs a running session);
  macOS/Windows stubs; screenshot decode via `image`; tests with a mock backend and the ignored real-Xvfb test.
- §4 the `/api/computer/stop` route, the audit-log writer, the "using the computer" indicator and Stop button in `app.js`, and
  wiring the rules check into `run.rs` around the existing browser dispatch (lines ~930–984).
- §5 browser tabs/select/wait/downloads/lists.
- §6 full web app Settings UI for computer mode, apps, download ask and the Rules panel.

## What this branch does

Rules engine + computer/browser settings — the in-server safety rules and their presets that every later step relies on.
