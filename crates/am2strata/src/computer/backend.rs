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

    impl Backend for LinuxBackend {
        fn screenshot(&self) -> Result<Buffer, String> {
            let bytes = self.capture_screen()?;
            let img = image::load_from_memory(&bytes).map_err(|e| format!("decoding the screenshot: {e}"))?;
            Ok(Buffer { width: img.width(), height: img.height(), rgba: img.to_rgba8().into_raw() })
        }

        fn list_windows(&self) -> Result<Vec<Window>, String> {
            if !tool_available("xwininfo") {
                return Err("`xwininfo` is not on this system; the window list needs it (install x11-utils)".into());
            }
            let disp = self.display.as_str();
            let out = run_cmd(&["xwininfo", "-root", "-tree", "-display", disp])?;
            Ok(parse_wininfo(&String::from_utf8_lossy(&out)))
        }

        fn focus(&self, id: WindowId) -> Result<(), String> {
            let win = format!("0x{id:x}");
            let mut c = dotool("windowactivate --sync")?;
            c.arg(&win).arg("-d").arg(&self.display);
            c.status_checked()
        }

        fn click(&self, x: i32, y: i32, button: Button) -> Result<(), String> {
            let mut m = dotool("mousemove --sync")?;
            m.arg(x.to_string()).arg(y.to_string()).arg("-d").arg(&self.display);
            m.status_checked()?;
            // xdotool click codes: 1 left, 2 middle, 3 right
            let code = match button { Button::Left => "1", Button::Middle => "2", Button::Right => "3" };
            let mut c = dotool("click")?;
            c.arg(code).arg("-d").arg(&self.display);
            c.status_checked()
        }

        fn type_text(&self, text: &str) -> Result<(), String> {
            // --clearmodifiers drops any stuck modifier keys; xdotool types the rest literally
            let mut c = dotool("type --clearmodifiers --delay 3")?;
            c.arg(text).arg("-d").arg(&self.display);
            c.status_checked()
        }

        fn press_keys(&self, keys: &[Key]) -> Result<(), String> {
            let seq = keys_to_xdotool(keys);
            if seq.is_empty() { return Ok(()); }
            let mut c = dotool("key --clearmodifiers")?;
            c.arg(seq).arg("-d").arg(&self.display);
            c.status_checked()
        }

        fn scroll(&self, dx: i32, dy: i32) -> Result<(), String> {
            let (h, v) = (scroll_part(dx), scroll_part(dy));
            if h.is_none() && v.is_none() { return Ok(()); } // nothing to do
            // xdotool scrolls lines from the current pointer (0,0); sign carries the direction.
            let mut c = dotool("scroll")?;
            c.arg("-d").arg(&self.display).arg("0").arg("0");
            if let Some((dir, n)) = v { c.args(&["--direction".into(), dir.into(), "--lines".into(), n.to_string()]); }
            else if let Some((dir, n)) = h { c.args(&["--direction".into(), dir.into(), "--lines".into(), n.to_string()]); }
            c.status_checked()
        }

        fn open_app(&self, name: &str) -> Result<(), String> {
            // run on this display; an app not installed (NotFound) is a clear "is it in the allow list?" message
            let mut cmd = std::process::Command::new(name);
            cmd.env("DISPLAY", &self.display).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
            match cmd.spawn() {
                Ok(_) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound =>
                    Err(format!("'{name}' is not installed on this system")),
                Err(e) => Err(e.to_string()),
            }
        }
    }

    /// This PC's desktop. For the sandbox it is a virtual display (Xvfb :99, or `weston --backend=headless`/`cage` if a
    /// headless compositor is set up); input is issued there with xdotool and its XTest, and the screenshot comes from
    /// ImageMagick on that display. The display defaults to :99, overridable with `XTEST_DISPLAY` (how the sandbox sets it).
    #[derive(Debug)]
    pub struct LinuxBackend { display: String }

    impl LinuxBackend {
        pub fn new(display: Option<&str>) -> LinuxBackend {
            let display = display.map(str::to_string)
                .or_else(|| std::env::var("XTEST_DISPLAY").ok())
                .unwrap_or_else(|| ":99".to_string());
            LinuxBackend { display }
        }

        /// Try a screenshot tool and return its image bytes (JPEG/PNG). The first that works wins.
        fn capture_screen(&self) -> Result<Vec<u8>, String> {
            let disp = self.display.as_str();
            // ImageMagick's `convert -window root` reads the whole display as one frame; `import -window root` is another entry point.
            if tool_available("convert") || tool_available("import") {
                let tool = if tool_available("convert") { "convert" } else { "import" };
                let b = run_cmd(&[tool, "-window", "root", "-display", disp, "jpg:-"])?;
                if !b.is_empty() { return Ok(b); }
            }
            // scrot writes a PNG to `-` (its own piping), else we read the file it leaves behind.
            if tool_available("scrot") {
                let b = run_cmd(&["scrot", "-o", "-", "-display", disp])?;
                if !b.is_empty() { return Ok(b); }
            }
            Err(format!("no screenshot tool (convert/import/scrot) is on this system for display {disp}"))
        }
    }

    /// The tools a real session needs, so a backend on a box without them fails cleanly instead of panicking.
    pub(super) fn tool_available(name: &str) -> bool {
        if let Some(path) = std::env::var_os("PATH") {
            for dir in std::env::split_paths(&path) {
                if dir.join(name).is_file() { return true; }
            }
        }
        false
    }

    /// xdotool invocation with a shared flag set, or an error naming the missing tool.
    pub(super) fn dotool(action: &str) -> Result<std::process::Command, String> {
        if !tool_available("xdotool") { return Err("`xdotool` is not on this system; computer use input needs it (install xdotool)".into()); }
        let mut c = std::process::Command::new("xdotool");
        c.args(action.split_whitespace()).stdin(std::process::Stdio::null());
        Ok(c)
    }

    /// Run a command, capture its stdout, and fail with stderr if it did not exit successfully.
    pub(super) fn run_cmd(args: &[&str]) -> Result<Vec<u8>, String> {
        let child = std::process::Command::new(&args[0]).args(&args[1..])
            .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null())
            .spawn().map_err(|e| format!("running {}: {e}", args[0]))?;
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

    /// The scroll part for one axis: (direction, line count) or None when the delta is zero.
    pub(super) fn scroll_part(d: i32) -> Option<(&'static str, u32)> {
        if d == 0 { return None; }
        Some((if d < 0 { "down" } else { "up" }, d.unsigned_abs()))
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
pub use linux::LinuxBackend;

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
        assert_eq!(linux::scroll_part(-2), Some(("down", 2)));
        assert_eq!(linux::scroll_part(0), None);
    }

    /// End-to-end on a real (sandbox) desktop: start Xvfb :99, open an allowed app through LinuxBackend, list its windows,
    /// take a screenshot, click it and tear the display down. Ignored by default — run with `cargo test -- --ignored` —
    /// because this box has no Xvfb/xdotool (a running session is needed); it also returns early if neither is installed.
    #[cfg(target_os = "linux")]
    #[ignore = "needs Xvfb + an X client on a running session"]
    #[test]
    fn sandbox_launches_a_virtual_display_and_clicks_an_app() -> Result<(), String> {
        if !linux::tool_available("Xvfb") || !linux::tool_available("xdotool") { return Ok(()); }

        // the sandbox: a virtual display at 1280x800, driven on that display by LinuxBackend
        let mut xvfb = std::process::Command::new("Xvfb").args([":99", "-screen", "0", "1280x800x24"])
            .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null())
            .spawn().map_err(|e| format!("starting Xvfb :99: {e}"))?;

        // wait (briefly) for the display to come up, or bail if it never does
        let mut waited = 0u32;
        while linux::run_cmd(&["xwininfo", "-root", "-display", ":99"]).is_err() && waited < 50 {
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

    #[test]
    fn the_linux_backend_constructs_and_degrades_without_a_display() {
        // an unlikely-running virtual display; without a screenshot tool there it must fail cleanly, not panic
        let b = LinuxBackend::new(Some(":98"));
        let _ = b.screenshot();
        let _ = b.open_app("am2strata-no-such-app-xyz");
    }
}
