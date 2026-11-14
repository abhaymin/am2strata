//! The desktop a computer-use step drives, hidden behind one trait so neither the tool layer nor the tests see which
//! OS or display it is on (`computer/backend.rs`). One method per action: screenshot, list/focus windows, click, type,
//! keys, scroll, open an app. Each returns `String` for a failure the model can read and act on — like browser.rs — not
//! an anyhow error to be swallowed.
//!
//! A backend is chosen by mode (sandbox / watch-only / assist / trusted-job) and OS in `run.rs`; here it is just the
//! trait, its data types, this PC's Linux implementation (GNOME on Wayland: input via XTest/Xvfb through xdotool, and a
//! screenshot from ImageMagick on that display), plus a stub for macOS and Windows that refuses with the platform's
//! permission message. Both share the trait, so the tool layer, tests and Settings work unchanged everywhere.

use base64::Engine as _;

/// What to do when no real backend exists yet for this OS (`Unsupported`). Matches the plan (§6): Linux builds now, the
/// others are stubs that refuse until implemented.
const UNSUPPORTED: &str = "computer use is not supported on this platform yet (design: §6)";

/// The window a click or key goes to. Its `WindowId` comes from `list_windows()` / `focus()`, never guessed by the model.
pub type WindowId = u64;

/// A visible window on the desktop, for the list the model reads when it has no screenshot (`xwininfo -tree` on Linux).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    pub id: WindowId,
    pub title: String,
    /// the process that owns it, if known (None until parsed)
    pub process: Option<String>,
}

/// Which mouse button a click uses; the argument default is Left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Button { Left, Right, Middle }

impl Button {
    /// The name in the tool schema (`"left"`, `"right"`, `"middle"`).
    pub fn as_str(self) -> &'static str {
        match self { Button::Left => "left", Button::Right => "right", Button::Middle => "middle" }
    }
}

/// A key or key-combination, as typed in the tool argument (`"Enter"`, `"Ctrl"` + `"a"`). Kept as a string because xdotool
/// takes arbitrary key names joined by `+`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Key(String);

impl Key {
    pub fn new(s: &str) -> Key { Key(s.to_string()) }
    /// the raw token (`"Ctrl"`), already split from a `"Ctrl+a"` sequence
    pub fn as_str(&self) -> &str { &self.0 }
}

/// An image frame the model can see (or the client display): raw RGBA pixels and their dimensions, re-encoded to JPEG.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Buffer {
    pub width: u32,
    pub height: u32,
    /// RGBA pixels, `width * height * 4` bytes, in row-major order
    pub rgba: Vec<u8>,
}

impl Buffer {
    /// The screenshot as a base64 JPEG (base64 for the tool result like the browser's), for a model that reads pictures.
    /// Falls back to None when the frame cannot be encoded, so a tool step can still report its text.
    pub fn jpeg_b64(&self) -> Option<String> {
        let rgba = image::RgbaImage::from_raw(self.width, self.height, self.rgba.clone())?;
        // JPEG has no alpha channel: encode the RGB channels (drop every 4th byte).
        let rgb_bytes: Vec<u8> = rgba.chunks_exact(4).flat_map(|px| [px[0], px[1], px[2]]).collect();
        let rgb = image::RgbImage::from_raw(self.width, self.height, rgb_bytes)?;
        let mut out = Vec::new();
        {
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 80);
            enc.encode_image(&rgb).ok()?;
        }
        Some(base64::engine::general_purpose::STANDARD.encode(out))
    }
}

/// The virtual display geometry (`WxHxD`), defaulting to 1280x800x24 when the config gives none. Portable: every OS parses it,
/// so `engine.rs` uses it on Linux and the macOS / Windows stubs alike — there is no OS in this type (it is why `Size` lives here).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size { pub width: u32, pub height: u32, pub depth: u32 }

impl Size {
    /// The default sandbox display geometry (1280x800x24).
    pub const DEFAULT: Size = Size { width: 1280, height: 800, depth: 24 };
    /// Parse "WxHxD" (e.g. "1280x800x24"); unknown or empty falls back to the default.
    pub fn parse(s: Option<&str>) -> Size {
        match s.and_then(|v| v.split_once('x')) {
            Some((w, rest)) => match (w.trim().parse::<u32>(), rest.split_once('x')) {
                (Ok(w), Some((h, d))) => match (h.trim().parse::<u32>(), d.trim().parse::<u32>()) {
                    (Ok(h), Ok(d)) if w > 0 && h > 0 && d > 0 => Size { width: w, height: h, depth: d },
                    _ => Size::DEFAULT,
                },
                _ => Size::DEFAULT,
            },
            None => Size::DEFAULT,
        }
    }
}

impl Default for Size { fn default() -> Self { Size::DEFAULT } }

/// Whether a tool (`name`) is on the system's PATH, so a backend fails cleanly on a box without it instead of panicking. Portable
/// — it only walks `PATH`, no OS call — so every platform gates its tools on it (`engine.rs`, item 2), Linux and macOS / Windows alike.
pub fn tool_available(name: &str) -> bool {
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            if dir.join(name).is_file() { return true; }
        }
    }
    false
}

/// The desktop a step acts on. Chosen by mode and OS at runtime; in tests it is any type that implements it (a mock).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub trait Backend {
    fn screenshot(&self) -> Result<Buffer, String>;
    fn list_windows(&self) -> Result<Vec<Window>, String>;
    fn focus(&self, id: WindowId) -> Result<(), String>;
    fn click(&self, x: i32, y: i32, button: Button) -> Result<(), String>;
    fn type_text(&self, text: &str) -> Result<(), String>;
    fn press_keys(&self, keys: &[Key]) -> Result<(), String>;
    fn scroll(&self, dx: i32, dy: i32) -> Result<(), String>;
    fn open_app(&self, name: &str) -> Result<(), String>;
}

// -------------------------------------------------------------------------------------------------------------- Linux

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::os::unix::io::{AsRawFd, FromRawFd};

    impl Backend for LinuxBackend {
        fn screenshot(&self) -> Result<Buffer, String> {
            let disp = self.disp()?;
            let bytes = self.capture_screen(&disp)?;
            let img = image::load_from_memory(&bytes).map_err(|e| format!("decoding the screenshot: {e}"))?;
            Ok(Buffer { width: img.width(), height: img.height(), rgba: img.to_rgba8().into_raw() })
        }

        fn list_windows(&self) -> Result<Vec<Window>, String> {
            if !tool_available("xwininfo") {
                return Err("`xwininfo` is not on this system; the window list needs it (install x11-utils)".into());
            }
            let disp = self.disp()?;
            let out = run_cmd(&["xwininfo", "-root", "-tree", "-display", &disp], &disp)?;
            Ok(parse_wininfo(&String::from_utf8_lossy(&out)))
        }

        fn focus(&self, id: WindowId) -> Result<(), String> {
            let disp = self.disp()?;
            let win = format!("0x{id:x}");
            let mut c = dotool("windowactivate --sync", &disp)?;
            c.arg(&win);
            c.status_checked()
        }

        fn click(&self, x: i32, y: i32, button: Button) -> Result<(), String> {
            let disp = self.disp()?;
            let mut m = dotool("mousemove --sync", &disp)?;
            m.arg(x.to_string()).arg(y.to_string());
            m.status_checked()?;
            // xdotool click codes: 1 left, 2 middle, 3 right
            let code = match button { Button::Left => "1", Button::Middle => "2", Button::Right => "3" };
            let mut c = dotool("click", &disp)?;
            c.arg(code);
            c.status_checked()
        }

        fn type_text(&self, text: &str) -> Result<(), String> {
            // --clearmodifiers drops any stuck modifier keys; xdotool types the rest literally
            let disp = self.disp()?;
            let mut c = dotool("type --clearmodifiers --delay 3", &disp)?;
            c.arg(text);
            c.status_checked()
        }

        fn press_keys(&self, keys: &[Key]) -> Result<(), String> {
            let seq = keys_to_xdotool(keys);
            if seq.is_empty() { return Ok(()); }
            let disp = self.disp()?;
            let mut c = dotool("key --clearmodifiers", &disp)?;
            c.arg(seq);
            c.status_checked()
        }

        fn scroll(&self, dx: i32, dy: i32) -> Result<(), String> {
            // xdotool scrolls one direction at a time, so emit a scroll per non-zero axis: both dx and dy run when both are given.
            let disp = self.disp()?;
            for (dir, n) in scroll_parts(dx, dy) {
                let mut c = dotool("scroll", &disp)?;
                c.arg("0").arg("0");
                c.args(&["--direction".into(), dir.into(), "--lines".into(), n.to_string()]);
                c.status_checked()?;
            }
            Ok(())
        }

        fn open_app(&self, name: &str) -> Result<(), String> {
            // run on this display; an app not installed (NotFound) is a clear "is it in the allow list?" message. Pin DISPLAY and
            // drop WAYLAND_DISPLAY so a Wayland-capable app cannot open on the real desktop instead of the sandbox.
            let disp = self.disp()?;
            let mut cmd = sandbox_app(name, &disp);
            match cmd.spawn() {
                Ok(_) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound =>
                    Err(format!("'{name}' is not installed on this system")),
                Err(e) => Err(e.to_string()),
            }
        }
    }

    /// This PC's desktop. For the sandbox it is a private Xvfb virtual display started lazily on the first step (`SandboxDesktop`),
    /// for the watch-only / assist / trusted-job modes a fixed one; input is issued there with xdotool and its XTest, and the
    /// screenshot comes from ImageMagick on that display. The sandbox's display is never the environment's DISPLAY (item 3).
    #[derive(Debug)]
    pub struct LinuxBackend {
        sandbox: bool,
        /// a set display for the non-sandbox / test case (`XTEST_DISPLAY`, else :99) — never the environment's DISPLAY
        fixed_display: Option<String>,
        /// the private Xvfb-backed virtual desktop the sandbox starts lazily; owns the child that is killed on drop
        desktop: Option<std::sync::Arc<SandboxDesktop>>,
    }

    impl LinuxBackend {
        /// A fixed display (`XTEST_DISPLAY`, else :99), used by the watch-only / assist / trusted-job modes and tests.
        pub fn new(display: Option<&str>) -> LinuxBackend {
            LinuxBackend { sandbox: false, fixed_display: display.map(str::to_string).or_else(|| std::env::var("XTEST_DISPLAY").ok()), desktop: None }
        }

        /// A private sandbox desktop: a virtual Xvfb display started lazily on the first step (item 1) at `size`, never driving
        /// the real DISPLAY, killed when dropped and after 10 minutes with no computer step. A `None` config size uses the default geometry.
        pub fn new_sandbox(size: Size) -> LinuxBackend {
            LinuxBackend { sandbox: true, fixed_display: None, desktop: Some(std::sync::Arc::new(SandboxDesktop::new(size))) }
        }

        /// The display a step drives. For the sandbox it starts a private Xvfb on a free number the first time (item 1), so it
        /// never falls back to the environment's DISPLAY / WAYLAND_DISPLAY; for the fixed case it is just that number.
        fn disp(&self) -> Result<String, String> {
            if self.sandbox { self.desktop.as_ref().unwrap().ensure() }
            else { Ok(self.fixed_display.clone().unwrap_or_else(|| ":99".into())) }
        }

        /// Try a screenshot tool and return its image bytes (JPEG/PNG). The first that works wins; needs a running display.
        fn capture_screen(&self, disp: &str) -> Result<Vec<u8>, String> {
            if tool_available("convert") || tool_available("import") {
                let tool = if tool_available("convert") { "convert" } else { "import" };
                let b = run_cmd(&[tool, "-window", "root", "-display", disp, "jpg:-"], disp)?;
                if !b.is_empty() { return Ok(b); }
            }
            // scrot writes a PNG to `-` (its own piping), else we read the file it leaves behind.
            if tool_available("scrot") {
                let b = run_cmd(&["scrot", "-o", "-", "-display", disp], disp)?;
                if !b.is_empty() { return Ok(b); }
            }
            Err(format!("no screenshot tool (convert/import/scrot) is on this system for display {disp}"))
        }
    }


    /// Start `Xvfb` privately for the sandbox (`item 1`). It picks its own free display via `-displayfd`, which reports that
    /// display number back on a pipe, so there is no window between choosing a number and starting Xvfb for another server to
    /// take it (and we never talk to a display we did not start). Returns the running child and the display name it wrote
    /// (`:NN`); `-nolisten tcp` keeps it off TCP, and if Xvfb exits early or writes nothing within 5 s this returns an error.
    /// Sandbox displays left by an am2Strata that died without cleaning up (a crash, an out-of-memory kill, SIGKILL):
    /// Xvfb with exactly our arguments whose parent is no longer an am2Strata server. Stopped before a new one starts,
    /// so they do not pile up. Another running server's display (its parent is that server) is never touched.
    fn reap_orphan_sandboxes() {
        let Ok(rd) = std::fs::read_dir("/proc") else { return };
        let me = unsafe { libc::getuid() };
        for e in rd.flatten() {
            let Ok(pid) = e.file_name().to_string_lossy().parse::<i32>() else { continue };
            let Ok(cmd) = std::fs::read(format!("/proc/{pid}/cmdline")) else { continue };
            let args: Vec<String> = cmd.split(|b| *b == 0).filter(|a| !a.is_empty()).map(|a| String::from_utf8_lossy(a).into_owned()).collect();
            // our own: Xvfb -displayfd N -nolisten tcp -screen 0 WxHxD
            if args.len() != 8 || !args[0].ends_with("Xvfb") || args[1] != "-displayfd" || args[3] != "-nolisten" || args[5] != "-screen" {
                continue;
            }
            let Ok(status) = std::fs::read_to_string(format!("/proc/{pid}/status")) else { continue };
            let field = |k: &str| status.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u32>().ok());
            if field("Uid:") != Some(me) {
                continue;
            }
            let ppid = field("PPid:").unwrap_or(0);
            let parent = std::fs::read_to_string(format!("/proc/{ppid}/comm")).unwrap_or_default();
            if parent.trim() != "am2strata" {
                unsafe { libc::kill(pid, libc::SIGTERM); }
            }
        }
    }

    fn start_sandbox_xvfb(size: Size) -> Result<(std::process::Child, String), String> {
        reap_orphan_sandboxes();
        unsafe {
            // a pipe so Xvfb can write its chosen display number back on `-displayfd`; the read end is parsed below.
            let mut fds = [0i32; 2];
            if libc::pipe(fds.as_mut_ptr()) != 0 { return Err("making the pipe for Xvfb's display number failed".into()); }
            // wrap both ends in a File so each closes exactly once, when this scope leaves.
            let read_end = std::fs::File::from_raw_fd(fds[0]);
            let write_end = std::fs::File::from_raw_fd(fds[1]);

            let mut cmd = std::process::Command::new("Xvfb");
            cmd.arg("-displayfd").arg(write_end.as_raw_fd().to_string())
               .arg("-nolisten").arg("tcp")
               .arg("-screen").arg("0").arg(format!("{}x{}x{}", size.width, size.height, size.depth))
               .env_remove("DISPLAY").env_remove("WAYLAND_DISPLAY")
               .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());

            let mut child = cmd.spawn().map_err(|e| format!("starting Xvfb: the sandbox needs `Xvfb` (install xvfb) to run its virtual display: {e}"))?;
            // close our copy of the write end after spawning; Xvfb keeps its own so it can still report the number.
            drop(write_end);

            // read Xvfb's reported display number in a thread until it ends its line (non-blocking reads on the pipe).
            let (dtx, drx) = std::sync::mpsc::channel();
            {
                let rd = read_end.try_clone().map_err(|e| e.to_string())?;
                std::thread::spawn(move || report_display_number(rd, &dtx));
            }

            const DISPLAY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
            let started = std::time::Instant::now();
            loop {
                if let Ok(read) = drx.try_recv() {
                    return match parse_display_number(&read) {
                        Some(d) => Ok((child, d)),
                        None => { let _ = child.kill(); Err(format!("Xvfb started but reported an unreadable display number: {read:?}").into()) }
                    };
                }
                match child.try_wait().map_err(|e| e.to_string())? {
                    Some(_) => {
                        // Xvfb may write the display then close its end; take one final look before giving up.
                        if let Ok(read) = drx.try_recv() {
                            return match parse_display_number(&read) {
                                Some(d) => Ok((child, d)),
                                None => { let _ = child.kill(); Err(format!("Xvfb started but reported an unreadable display number: {read:?}").into()) }
                            };
                        }
                        let _ = child.kill();
                        return Err("Xvfb exited before it could report a display number".into());
                    }
                    None if started.elapsed() >= DISPLAY_TIMEOUT => {
                        let _ = child.kill();
                        return Err("Xvfb did not report a display number within 5 seconds".into());
                    }
                    _ => std::thread::sleep(std::time::Duration::from_millis(20)),
                }
            }
        }
    }

    /// Xvfb writes its chosen display (`:NN`) on `-displayfd` followed by a newline. Read that number back from the pipe
    /// without blocking forever (non-blocking reads until the line ends, or a short cap), then hand it to the channel.
    fn report_display_number(read_end: std::fs::File, dtx: &std::sync::mpsc::Sender<String>) {
        let fd = read_end.as_raw_fd();
        let mut data: Vec<u8> = Vec::new();
        unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFL);
            if flags >= 0 { let _ = libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK); }
            let mut buf = [0u8; 64];
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                match libc::read(fd, buf.as_mut_ptr() as *mut _, buf.len()) {
                    n if n > 0 => {
                        data.extend_from_slice(&buf[..n as usize]);
                        if data.iter().any(|&b| b == b'\n') { break; }  // full line read: the display number Xvfb chose
                    }
                    0 => break,                                       // EOF: Xvfb closed its end (it exited)
                    _ => {
                        let e = std::io::Error::last_os_error();
                        if e.kind() != std::io::ErrorKind::WouldBlock && e.kind() != std::io::ErrorKind::Interrupted { break; }
                        if std::time::Instant::now() >= deadline { break; }
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                }
            }
        }
        let _ = dtx.send(String::from_utf8_lossy(&data).into_owned());
    }

    /// The display number Xvfb reported on `-displayfd` (`:NN`, or just `NN`; possibly with trailing whitespace/newline), or
    /// None when it is not a usable ":N". Pure, so the read-back can be unit-tested without starting Xvfb (item 1).
    pub(super) fn parse_display_number(read: &str) -> Option<String> {
        let s = read.trim();
        let body = s.strip_prefix(':').unwrap_or(s);
        if body.is_empty() || !body.chars().all(|c| c.is_ascii_digit()) { return None; }
        Some(format!(":{body}"))
    }

    /// The sandbox's private virtual desktop. It owns an `Xvfb` started lazily on the first computer step and killed when the last
    /// reference drops (the server stops), shared by the backend that drives it and the engine that stops it after 10 minutes idle.
    pub struct SandboxDesktop {
        size: Size,
        inner: std::sync::Mutex<State>,
        /// when the display was last driven, so the idle watcher can tell it has been quiet (item 1); set on `new`, refreshed each step.
        /// A mutex keeps this interior-mutable (driven through `&self`) and Send + Sync like the rest of the desktop.
        last_used: std::sync::Mutex<std::time::Instant>,
        /// whether the display has been told to stop (set on drop and by the idle watcher), so the watcher can exit promptly
        pub stopped: std::sync::atomic::AtomicBool,
    }

    /// Whether Xvfb is running yet (`Idle`), on screen (`Running(child, display)`) or stopped.
    enum State {
        Idle,
        Running(std::process::Child, String),
    }

    impl std::fmt::Debug for SandboxDesktop {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let running = matches!(*self.inner.lock().unwrap(), State::Running(..));
            f.debug_struct("SandboxDesktop").field("size", &self.size).field("running", &running).finish()
        }
    }

    impl SandboxDesktop {
        /// Construct a sandbox desktop at `size`, not yet started (Xvfb runs on the first `ensure`).
        pub fn new(size: Size) -> SandboxDesktop {
            SandboxDesktop { size, inner: std::sync::Mutex::new(State::Idle), last_used: std::sync::Mutex::new(std::time::Instant::now()), stopped: std::sync::atomic::AtomicBool::new(false) }
        }

        /// The display a step drives, starting Xvfb lazily the first time (item 1) and refreshing the last-used clock.
        pub fn ensure(&self) -> Result<String, String> {
            *self.last_used.lock().unwrap() = std::time::Instant::now();   // a step ran, so the idle watcher should count from now regardless of Xvfb state
            let mut state = self.inner.lock().unwrap();
            if !matches!(*state, State::Running(..)) {
                // the display is (re)starting after an idle stop: clear `stopped` so a fresh idle watcher can run again (item 1).
                self.stopped.store(false, std::sync::atomic::Ordering::Relaxed);
                match start_sandbox_xvfb(self.size) {
                    Ok((child, disp)) => *state = State::Running(child, disp),
                    Err(e) => return Err(e),
                }
            }
            match &*state {
                State::Running(_, d) => Ok(d.clone()),
                State::Idle => Err("the sandbox display is not running".into()),
            }
        }

        /// Stop the virtual display (kill Xvfb); called on drop and by the idle watcher.
        pub fn stop(&self) {
            self.stopped.store(true, std::sync::atomic::Ordering::Relaxed);
            let mut state = self.inner.lock().unwrap();
            if let State::Running(child, _) = &mut *state { let _ = child.kill(); let _ = child.wait(); }
            *state = State::Idle;
        }

        /// True when it has been quiet at least `within` (the idle watcher's period), so the watcher can stop it.
        pub fn is_idle(&self, within: std::time::Duration) -> bool {
            std::time::Instant::now().duration_since(*self.last_used.lock().unwrap()) > within
        }
    }

    impl std::ops::Drop for SandboxDesktop {
        fn drop(&mut self) { self.stop(); }
    }


    /// A child X tool must run against the sandbox display with nothing inherited from the server's environment (the am2strata
    /// service runs with `DISPLAY=:0` / `WAYLAND_DISPLAY=wayland-0`, so a leaked DISPLAY would drive the owner's real screen),
    /// yet keep PATH so the binary resolves. Applies this pinned environment to any Command; xdotool reads its display only from
    /// DISPLAY, so no `-d` argument is needed or allowed anymore.
    /// An app started on the sandbox display (the Computer tab's "Open an app" and the agent's `open_app`). It keeps the
    /// user's environment (HOME and the rest: apps need them) but loses everything that leads to the real desktop session:
    /// WAYLAND_DISPLAY (a window on the real screen), XAUTHORITY, and the session D-Bus — through which a browser finds an
    /// instance already open on the real desktop and opens its window there. Toolkits are told to use X11. Firefox also
    /// gets its own profile (in ~/snap/firefox/common for a snap Firefox, which may not write elsewhere) and no remote.
    pub fn sandbox_app(name: &str, disp: &str) -> std::process::Command {
        let mut cmd = std::process::Command::new(name);
        for k in ["WAYLAND_DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS", "DESKTOP_STARTUP_ID", "XDG_ACTIVATION_TOKEN"] {
            cmd.env_remove(k);
        }
        cmd.env("DISPLAY", disp).env("GDK_BACKEND", "x11").env("QT_QPA_PLATFORM", "xcb")
            .env("MOZ_ENABLE_WAYLAND", "0").env("MOZ_DBUS_REMOTE", "0");
        let base = std::path::Path::new(name).file_name().and_then(|n| n.to_str()).unwrap_or(name).to_string();
        if base.starts_with("firefox") {
            if let Some(home) = std::env::var_os("HOME") {
                // a snap Firefox may only write in its own ~/snap/firefox (AppArmor), any other in ~/.local/share
                let home = std::path::PathBuf::from(home);
                let profile = if home.join("snap/firefox").is_dir() { home.join("snap/firefox/common/am2strata-sandbox-profile") }
                              else { home.join(".local/share/am2strata/sandbox-firefox-profile") };
                let _ = std::fs::create_dir_all(&profile);
                cmd.arg("--no-remote").arg("--profile").arg(profile);
            }
        }
        cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
        cmd
    }

    fn use_sandbox_display(mut cmd: std::process::Command, disp: &str) -> std::process::Command {
        // capture PATH before clearing (env_clear drops all inherited vars); DISPLAY is pinned to the sandbox display below.
        let path = std::env::var_os("PATH");
        cmd.env_clear();
        if let Some(path) = path { cmd.env("PATH", path); }
        cmd.env("DISPLAY", disp);
        cmd
    }

    /// xdotool invocation pinned to the sandbox display (`disp`), or an error naming the missing tool. Callers pass the action and
    /// its arguments; the sandbox DISPLAY is set on the environment (never inherited), so `-d` is dropped at every call site.
    pub(super) fn dotool(action: &str, disp: &str) -> Result<std::process::Command, String> {
        if !tool_available("xdotool") { return Err("`xdotool` is not on this system; computer use input needs it (install xdotool)".into()); }
        let mut c = std::process::Command::new("xdotool");
        c.args(action.split_whitespace()).stdin(std::process::Stdio::null());
        Ok(use_sandbox_display(c, disp))
    }

    /// Run a command against the sandbox display (`disp`), capture its stdout, and fail with stderr if it did not exit successfully.
    pub(super) fn run_cmd(args: &[&str], disp: &str) -> Result<Vec<u8>, String> {
        let mut child_cmd = std::process::Command::new(&args[0]);
        child_cmd.args(&args[1..]).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null());
        let child = use_sandbox_display(child_cmd, disp).spawn().map_err(|e| format!("running {}: {e}", args[0]))?;
        let out = child.wait_with_output().map_err(|e| e.to_string())?;
        if out.status.success() { Ok(out.stdout) } else { Err("the screenshot command failed".into()) }
    }

    /// `xwininfo -root -tree` lines: each window starts with a 0x id and ends with a quoted title; the root has none.
    fn parse_wininfo(out: &str) -> Vec<Window> {
        let mut windows = Vec::new();
        for line in out.lines() {
            if !line.contains('"') { continue; } // the root window, or a line of another format
            let Some(id_tok) = line.trim_start().split_whitespace().next() else { continue };
            let Ok(id) = id_tok.trim_start_matches("0x").parse::<WindowId>() else { continue };
            if id == 0 { continue; } // skip the root window
            windows.push(Window { id, title: between_quotes(line).unwrap_or_default(), process: None });
        }
        windows
    }

    /// The text between the first and last `"` on a line (the window title from `xwininfo`).
    fn between_quotes(s: &str) -> Option<String> {
        let open = s.find('"')?;
        let rest = &s[open + 1..];
        let close = rest.rfind('"')?;
        Some(rest[..close].to_string())
    }

    /// Join key tokens into an xdotool key sequence...
    pub(super) fn keys_to_xdotool(keys: &[Key]) -> String {
        let mut seq = Vec::new();
        for k in keys {
            if k.as_str().is_empty() { continue; }
            // a single token may itself be a sequence ("ctrl+a") already; split on '+' and lowercase each.
            seq.extend(k.as_str().split('+').map(|s| s.to_ascii_lowercase()));
        }
        seq.join("+")
    }

    /// The scroll actions for both axes: one (direction, line count) per non-zero delta, so a diagonal scrolls in both directions.
    /// Per the tool schema positive dx scrolls right and positive dy scrolls down; xdotool takes up/down/left/right directions.
    pub(super) fn scroll_parts(dx: i32, dy: i32) -> Vec<(&'static str, u32)> {
        let mut out = Vec::new();
        if dy != 0 { out.push((if dy > 0 { "down" } else { "up" }, dy.unsigned_abs())); }
        if dx != 0 { out.push((if dx > 0 { "right" } else { "left" }, dx.unsigned_abs())); }
        out
    }

    /// Run a finished command, turning failure into a readable error (the tool's name + the failing status).
    trait StatusExt { fn status_checked(self) -> Result<(), String>; }
    impl StatusExt for std::process::Command {
        fn status_checked(mut self) -> Result<(), String> {
            match self.status() {
                Ok(s) if s.success() => Ok(()),
                Ok(_) => Err("the input command failed".to_string()),
                Err(e) => Err(e.to_string()),
            }
        }
    }
}

#[cfg(target_os = "linux")]
pub use linux::{sandbox_app, LinuxBackend, SandboxDesktop};

// ------------------------------------------------------------------------------------------------- Non-Linux helpers

/// macOS / Windows: the same desktop handle type as Linux, so `engine.rs`'s `Computers.desktop` slot, its idle watcher and the
/// sandbox tests are well-typed everywhere. There is never a real display here (the backend is `None`, so no tool is offered), but
/// this keeps those fields and functions compiling; it only records that it was asked to stop. Kept behind cfg since Linux has the
/// real Xvfb-backed one in `mod linux` — both are named `SandboxDesktop` at this path, which is why the engine needs no OS check.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
#[cfg(not(target_os = "linux"))]
#[derive(Debug)]
pub struct SandboxDesktop {
    /// whether it has been told to stop (the idle watcher / engine read this), mirroring the Linux sandbox's flag.
    pub stopped: std::sync::atomic::AtomicBool,
}

#[cfg(not(target_os = "linux"))]
impl SandboxDesktop {
    /// Construct a phantom desktop at `size`; there is no Xvfb to launch on these platforms (nothing is ever started).
    pub fn new(_size: Size) -> SandboxDesktop { SandboxDesktop { stopped: std::sync::atomic::AtomicBool::new(false) } }

    /// There is nothing running here (`LinuxBackend` does not exist on this platform), so a step never drives it.
    pub fn ensure(&self) -> Result<String, String> { Ok(":99".into()) }

    /// Mark it stopped; there is no child process to kill.
    pub fn stop(&self) { self.stopped.store(true, std::sync::atomic::Ordering::Relaxed) }

    /// Report idle so the idle watcher stops a phantom display just like the real one (nothing depends on the quiet time here).
    pub fn is_idle(&self, _within: std::time::Duration) -> bool { true }
}

// -------------------------------------------------------------------------------------------------------- Non-Linux stub

/// macOS / Windows: the same trait, refusing until implemented. Both return `UNSUPPORTED` from every method, so a desktop
/// build compiles and behaves identically (the tool layer and tests do not change) — §6 builds them.
#[cfg(not(target_os = "linux"))]
pub struct Unsupported;

#[cfg(not(target_os = "linux"))]
impl Backend for Unsupported {
    fn screenshot(&self) -> Result<Buffer, String> { Err(UNSUPPORTED.into()) }
    fn list_windows(&self) -> Result<Vec<Window>, String> { Err(UNSUPPORTED.into()) }
    fn focus(&self, _id: WindowId) -> Result<(), String> { Err(UNSUPPORTED.into()) }
    fn click(&self, _x: i32, _y: i32, _button: Button) -> Result<(), String> { Err(UNSUPPORTED.into()) }
    fn type_text(&self, _text: &str) -> Result<(), String> { Err(UNSUPPORTED.into()) }
    fn press_keys(&self, _keys: &[Key]) -> Result<(), String> { Err(UNSUPPORTED.into()) }
    fn scroll(&self, _dx: i32, _dy: i32) -> Result<(), String> { Err(UNSUPPORTED.into()) }
    fn open_app(&self, _name: &str) -> Result<(), String> { Err(UNSUPPORTED.into()) }
}

// ------------------------------------------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A mock backend the model's tools drive in tests — no OS, records every call. This is why `Backend` exists: any type
    /// implementing it can stand in for a real desktop (interior mutability keeps the `&self` signature from the plan), and
    /// the tool layer never mentions an OS.
    struct MockBackend {
        shot: Vec<u8>,
        windows: Vec<Window>,
        clicks: RefCell<Vec<(i32, i32, Button)>>,
        typed: RefCell<Vec<String>>,
        scrolled: RefCell<Vec<(i32, i32)>>,
        opened: RefCell<Vec<String>>,
    }

    impl MockBackend {
        fn new() -> MockBackend {
            MockBackend { shot: vec![0u8; 16], windows: Vec::new(), clicks: RefCell::new(Vec::new()), typed: RefCell::new(Vec::new()),
                           scrolled: RefCell::new(Vec::new()), opened: RefCell::new(Vec::new()) }
        }
    }

    impl Backend for MockBackend {
        fn screenshot(&self) -> Result<Buffer, String> { Ok(Buffer { width: 2, height: 2, rgba: self.shot.clone() }) }
        fn list_windows(&self) -> Result<Vec<Window>, String> { Ok(self.windows.clone()) }
        fn focus(&self, _id: WindowId) -> Result<(), String> { Ok(()) }
        fn click(&self, x: i32, y: i32, button: Button) -> Result<(), String> { self.clicks.borrow_mut().push((x, y, button)); Ok(()) }
        fn type_text(&self, text: &str) -> Result<(), String> { self.typed.borrow_mut().push(text.to_string()); Ok(()) }
        fn press_keys(&self, _keys: &[Key]) -> Result<(), String> { Ok(()) }
        fn scroll(&self, dx: i32, dy: i32) -> Result<(), String> { self.scrolled.borrow_mut().push((dx, dy)); Ok(()) }
        fn open_app(&self, name: &str) -> Result<(), String> { self.opened.borrow_mut().push(name.to_string()); Ok(()) }
    }

    #[test]
    fn a_step_drives_a_mock_backend_without_an_os() {
        let m = MockBackend::new();
        assert_eq!(m.screenshot().unwrap(), Buffer { width: 2, height: 2, rgba: vec![0u8; 16] });
        m.focus(42).unwrap();
        m.click(3, 4, Button::Right).unwrap();
        m.type_text("hi").unwrap();
        m.press_keys(&[Key::new("Ctrl"), Key::new("a")]).unwrap();
        m.scroll(-1, 2).unwrap();
        m.open_app("firefox").unwrap();
        assert_eq!(*m.clicks.borrow(), vec![(3, 4, Button::Right)]);
        assert_eq!(*m.typed.borrow(), vec!["hi".to_string()]);
        assert_eq!(*m.scrolled.borrow(), vec![(-1, 2)]);
        assert_eq!(*m.opened.borrow(), vec!["firefox".to_string()]);
    }

    #[test]
    fn a_buffer_reencodes_to_a_base64_jpeg() {
        let b = Buffer { width: 2, height: 2, rgba: vec![10u8; 16] };
        assert!(b.jpeg_b64().is_some(), "a frame should encode to a base64 JPEG");
    }

    #[test]
    fn button_maps_to_the_tool_name() {
        assert_eq!(Button::Left.as_str(), "left");
        assert_eq!(Button::Right.as_str(), "right");
        assert_eq!(Button::Middle.as_str(), "middle");
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn the_stub_refuses_on_this_platform() {
        let u = Unsupported;
        assert_eq!(u.screenshot().unwrap_err(), UNSUPPORTED);
        assert_eq!(u.open_app("x").unwrap_err(), UNSUPPORTED);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn key_and_scroll_helpers_render_for_xdotool() {
        assert_eq!(linux::keys_to_xdotool(&[Key::new("Ctrl"), Key::new("a")]), "ctrl+a");
        assert_eq!(linux::keys_to_xdotool(&[Key::new("Enter")]), "enter");
        // positive dy scrolls down, negative dx scrolls left, and both axes come back (dy then dx) when both deltas are given.
        assert!(linux::scroll_parts(0, 0).is_empty(), "a zero scroll does nothing");
        assert_eq!(linux::scroll_parts(0, 2), vec![("down", 2)]);
        assert_eq!(linux::scroll_parts(-3, 4), vec![("down", 4), ("left", 3)], "both axes run when both are given");
    }

    /// End-to-end on a real (sandbox) desktop: start Xvfb :99, open an allowed app through LinuxBackend, list its windows,
    /// take a screenshot, click it and tear the display down. Ignored by default — run with `cargo test -- --ignored` —
    /// because this box has no Xvfb/xdotool (a running session is needed); it also returns early if neither is installed.
    #[cfg(target_os = "linux")]
    #[ignore = "needs Xvfb + an X client on a running session"]
    #[test]
    fn sandbox_launches_a_virtual_display_and_clicks_an_app() -> Result<(), String> {
        if !tool_available("Xvfb") || !tool_available("xdotool") { return Ok(()); }

        // the sandbox: a virtual display at 1280x800, driven on that display by LinuxBackend
        let mut xvfb = std::process::Command::new("Xvfb").args([":99", "-screen", "0", "1280x800x24"])
            .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null())
            .spawn().map_err(|e| format!("starting Xvfb :99: {e}"))?;

        // wait (briefly) for the display to come up, or bail if it never does
        let mut waited = 0u32;
        while linux::run_cmd(&["xwininfo", "-root", "-display", ":99"], ":99").is_err() && waited < 50 {
            std::thread::sleep(std::time::Duration::from_millis(100));
            waited += 1;
        }

        let backend = LinuxBackend::new(Some(":99"));
        let _ = backend.open_app("xterm"); // any allowed app: a small one that stays on screen
        std::thread::sleep(std::time::Duration::from_millis(300));

        let shot = backend.screenshot()?; // a frame from the virtual display
        assert!(shot.width >= 1 && shot.height >= 1, "the display produced a frame");
        if let Some(jpg) = shot.jpeg_b64() {
            assert!(!jpg.is_empty(), "a screenshot re-encodes to a base64 JPEG for the model");
        }

        // driving the desktop: point/click, type, and a keypress land through the trait
        backend.click(10, 10, Button::Left)?;
        backend.type_text("hi")?;
        backend.press_keys(&[Key::new("Enter")])?;
        let _ = backend.list_windows();

        // tear down the display; a stale one left by a killed run is left for the OS to reap
        let _ = xvfb.kill();
        let _ = xvfb.wait();
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_linux_backend_constructs_and_degrades_without_a_display() {
        // an unlikely-running virtual display; without a screenshot tool there it must fail cleanly, not panic
        let b = LinuxBackend::new(Some(":98"));
        let _ = b.screenshot();
        let _ = b.open_app("am2strata-no-such-app-xyz");
    }

    /// The sandbox display is handed to child X tools on their environment, never inherited from the server's DISPLAY (`/computer/backend.rs`, item 3).
    /// Unlike the old check (which only compared the stored string), this sets `DISPLAY=:0` in the process — like the real service would — and asserts what each
    /// built Command actually passes the child: DISPLAY is the sandbox display, there is no WAYLAND_DISPLAY leaked in, PATH is kept, and `-d` is no longer an xdotool
    /// argument (for click, type, key, scroll and focus). Without the fix the inherited `DISPLAY=:0` would hit the owner's real desktop.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_sandbox_backend_never_uses_the_environment_display() {
        let original = std::env::var("DISPLAY").ok();
        std::env::set_var("DISPLAY", ":0");  // the server's real display; a leaked one would drive the owner's desktop

        // Inspecting the built Command needs no running display, but building an xdotool command goes through the availability
        // guard — so on a box without xdotool (this CI box) there is nothing to build; the assertions below run wherever the
        // sandbox actually runs, which requires `xdotool` (`computer/engine.rs`, item 2).
        if !tool_available("xdotool") {
            match original { Some(o) => std::env::set_var("DISPLAY", o), None => std::env::remove_var("DISPLAY") }
            return;
        }

        const SANDBOX: &str = ":99";

        // focus: xdotool windowactivate --sync <win>
        let mut c = linux::dotool("windowactivate --sync", SANDBOX).unwrap(); c.arg("0x1f");
        assert_sandbox_env(&c, "DISPLAY", "WAYLAND_DISPLAY", "PATH", SANDBOX);

        // click: xdotool mousemove --sync <x> <y>, then xdotool click <code>
        let mut m = linux::dotool("mousemove --sync", SANDBOX).unwrap(); m.arg("10").arg("20");
        assert_sandbox_env(&m, "DISPLAY", "WAYLAND_DISPLAY", "PATH", SANDBOX);
        let mut k = linux::dotool("click", SANDBOX).unwrap(); k.arg("1");
        assert_sandbox_env(&k, "DISPLAY", "WAYLAND_DISPLAY", "PATH", SANDBOX);

        // type: xdotool type --clearmodifiers --delay 3 <text>
        let mut t = linux::dotool("type --clearmodifiers --delay 3", SANDBOX).unwrap(); t.arg("hello");
        assert_sandbox_env(&t, "DISPLAY", "WAYLAND_DISPLAY", "PATH", SANDBOX);

        // key: xdotool key --clearmodifiers <seq>
        let mut kp = linux::dotool("key --clearmodifiers", SANDBOX).unwrap(); kp.arg("enter");
        assert_sandbox_env(&kp, "DISPLAY", "WAYLAND_DISPLAY", "PATH", SANDBOX);

        // scroll: xdotool scroll 0 0 [direction]
        let mut s = linux::dotool("scroll", SANDBOX).unwrap(); s.arg("0").arg("0");
        assert_sandbox_env(&s, "DISPLAY", "WAYLAND_DISPLAY", "PATH", SANDBOX);

        match original { Some(o) => std::env::set_var("DISPLAY", o), None => std::env::remove_var("DISPLAY") }
    }

    /// What every built child X command must receive: DISPLAY pinned to the sandbox display, no WAYLAND_DISPLAY leaked in, PATH kept so
    /// the binary resolves, and no `-d` argument (xdotool reads its display only from the environment). Mirrors exactly what each backend
    /// method pushes onto its Command. `CommandEnvs` is an iterator over the env overrides, so it is scanned here rather than indexed.
    #[cfg(target_os = "linux")]
    fn assert_sandbox_env(c: &std::process::Command, disp_key: &str, wl_key: &str, path_key: &str, sandbox: &str) {
        assert!(!c.get_args().any(|a| a.to_string_lossy() == "-d"), "xdotool must not get a -d argument; it reads DISPLAY from the environment");
        let mut disp_value: Option<String> = None;
        let mut has_wl = false;
        let mut has_path = false;
        for (k, v) in c.get_envs() {
            let ks = k.to_string_lossy();
            if ks == disp_key {
                disp_value = Some(v.as_ref().and_then(|o| o.to_str()).unwrap_or("").to_string());
            } else if ks == wl_key {
                has_wl = true;
            } else if ks == path_key {
                has_path = true;
            }
        }
        assert_eq!(disp_value, Some(sandbox.to_string()), "child must get the sandbox DISPLAY, not an inherited one");
        assert!(!has_wl, "WAYLAND_DISPLAY must not reach the child");
        assert!(has_path, "PATH must be kept so the binary resolves");
    }

    /// The default geometry is used when the config gives none or an unparseable value.
    #[cfg(target_os = "linux")]
    #[test]
    fn size_falls_back_to_the_default_geometry() {
        assert_eq!(Size::default(), Size::DEFAULT);
        assert_eq!(Size::parse(None), Size::DEFAULT);
        assert_eq!(Size::parse(Some("garbage")), Size::DEFAULT);
        assert_eq!(Size::parse(Some("1024x768x24")), Size { width: 1024, height: 768, depth: 24 });
        assert_eq!(Size::parse(Some("0x768x24")), Size::DEFAULT, "a zero width falls back to the default");
    }

    /// Xvfb reports its chosen display on `-displayfd`; parse that number back (item 1). Pure, no Xvfb needed. Accepts a
    /// leading colon or just the digits, since different Xvfb builds write one form or the other.
    #[cfg(target_os = "linux")]
    #[test]
    fn xvfb_display_number_is_parsed_from_the_pipe() {
        assert_eq!(linux::parse_display_number(":99\n"), Some(":99".into()));
        assert_eq!(linux::parse_display_number("2\n"), Some(":2".into()), "some builds write the number without a colon");
        assert_eq!(linux::parse_display_number(" :123 "), Some(":123".into()));
        assert_eq!(linux::parse_display_number(""), None);
        assert_eq!(linux::parse_display_number("garbage"), None);
        assert_eq!(linux::parse_display_number(":-1"), None);
    }

    /// The sandbox's private Xvfb picks a free display via `-displayfd` and reports it back over the pipe (item 1). Ignored by
    /// default — this box has no Xvfb on a running session; run with `cargo test -- --ignored`. Bails when Xvfb is absent.
    #[cfg(target_os = "linux")]
    #[ignore = "needs Xvfb on a running session"]
    #[test]
    fn the_sandbox_xvfb_reports_a_display_number_that_is_usable() -> Result<(), String> {
        if !tool_available("Xvfb") { return Ok(()); }
        // a desktop at 640x480; Xvfb runs privately, reports its display over the pipe, stays up so it can be driven.
        let d = SandboxDesktop::new(Size { width: 640, height: 480, depth: 24 });
        let disp = d.ensure()?;
        assert!(disp.starts_with(':'), "Xvfb writes back a usable display number: {disp}");
        // it is the number Xvfb itself chose (not ours): :99 or whatever -displayfd handed out, never the server's DISPLAY.
        let _ = d.stop();
        Ok(())
    }

    /// After an idle stop the display is `stopped`; when a step restarts it, `ensure` clears that flag so a fresh idle watcher can run
    /// again (item 1). Ignored by default like the sibling Xvfb test — bail when Xvfb is absent.
    #[cfg(target_os = "linux")]
    #[ignore = "needs Xvfb on a running session"]
    #[test]
    fn ensure_clears_stopped_when_the_display_restarts() -> Result<(), String> {
        if !tool_available("Xvfb") { return Ok(()); }
        let d = SandboxDesktop::new(Size { width: 320, height: 240, depth: 24 });
        let _ = d.stop();                                       // an idle watcher stopped the display
        assert!(d.stopped.load(std::sync::atomic::Ordering::Relaxed), "it is stopped");
        d.ensure()?;                                            // a step restarts the display
        assert!(!d.stopped.load(std::sync::atomic::Ordering::Relaxed), "the restart cleared `stopped` so a fresh watcher can run (item 1)");
        let _ = d.stop();
        Ok(())
    }
}
