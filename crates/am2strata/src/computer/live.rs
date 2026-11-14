//! The Computer tab: the sandbox desktop computer use works on, live in the web app, and the operator's own mouse,
//! keyboard and clipboard on it — so a person can watch a job, take over (a sign-in, a setting, a fix) and hand back,
//! without a separate remote-desktop tool.
//!
//! - `GET /api/computer/stream` (WebSocket; `wss://` on the HTTPS port): JPEG frames when the screen changes (about
//!   8 a second at most), the state as JSON, and the operator's input. The key comes in the first message
//!   (`{"type":"auth","key":...}`, a browser cannot set headers on a WebSocket; never in the URL) or the
//!   Authorization header; the page must be am2Strata's own (the Origin is checked).
//! - Input is accepted only while the operator has control (**Take control**); the agent's computer steps wait until
//!   it is handed back (`Computers::step`). Closing the tab hands control back.
//! - REST, for the CLI and scripts: `GET /api/computer/status`, `GET /api/computer/screenshot` (PNG),
//!   `POST /api/computer/control {"action":"take"|"give"}`, `GET|POST /api/computer/clipboard`,
//!   `POST /api/computer/type {"text"}`, `POST /api/computer/open {"app"}` (only apps in the allowed list).
//!
//! Frames come from the sandbox's own private X display (x11rb, a Rust X client) and input goes into that display
//! (XTest) — never the real desktop. The clipboard uses `xclip` on that display. Only the server's own key may use any
//! of this (not client profiles).

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};

use crate::{authorized, error, AppState};

fn owner_key(s: &AppState, headers: &HeaderMap) -> bool {
    authorized(s, headers) && !crate::clients::given_key(headers).is_some_and(|k| s.clients.by_key(k).is_some())
}

fn computers(s: &AppState) -> Result<Arc<crate::computer::Computers>, String> {
    s.mgr.computer.get().cloned().ok_or_else(|| "the server is still starting".to_string())
}

/// What the tab and `am2strata-run computer status` show.
pub fn status(c: &crate::computer::Computers) -> Value {
    json!({"on": c.is_on(), "mode": c.mode_name(), "available": cfg!(target_os = "linux"),
           "operator_in_control": c.human_in_control(), "apps": c.apps()})
}

// ----------------------------------------------------------------------------------------------------- the display

#[cfg(target_os = "linux")]
pub mod x11 {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::{ConnectionExt as _, ImageFormat};
    use x11rb::protocol::xtest::ConnectionExt as _;

    /// A connection to the sandbox's X display: frames out, the operator's input in.
    pub struct Display {
        conn: x11rb::rust_connection::RustConnection,
        root: u32,
        pub width: u16,
        pub height: u16,
        min_keycode: u8,
        per_keycode: u8,
        keysyms: Vec<u32>,
    }

    impl Display {
        pub fn open(name: &str) -> Result<Display, String> {
            let (conn, n) = x11rb::connect(Some(name)).map_err(|e| format!("display {name}: {e}"))?;
            let setup = conn.setup();
            let screen = &setup.roots[n];
            let (root, width, height) = (screen.root, screen.width_in_pixels, screen.height_in_pixels);
            let (min, max) = (setup.min_keycode, setup.max_keycode);
            let map = conn.get_keyboard_mapping(min, max - min + 1).map_err(|e| e.to_string())?.reply().map_err(|e| e.to_string())?;
            Ok(Display { conn, root, width, height, min_keycode: min, per_keycode: map.keysyms_per_keycode, keysyms: map.keysyms })
        }

        /// The screen as RGB bytes.
        pub fn grab(&self) -> Result<Vec<u8>, String> {
            let r = self.conn.get_image(ImageFormat::Z_PIXMAP, self.root, 0, 0, self.width, self.height, !0)
                .map_err(|e| e.to_string())?.reply().map_err(|e| e.to_string())?;
            let px = (self.width as usize) * (self.height as usize);
            if r.data.len() < px * 4 {
                return Err(format!("unexpected image format (depth {}, {} bytes)", r.depth, r.data.len()));
            }
            // 24-bit depth in 32-bit pixels, least significant byte first: B, G, R, X
            let mut rgb = Vec::with_capacity(px * 3);
            for p in r.data.chunks_exact(4).take(px) {
                rgb.extend_from_slice(&[p[2], p[1], p[0]]);
            }
            Ok(rgb)
        }

        fn fake(&self, kind: u8, detail: u8, x: i16, y: i16) -> Result<(), String> {
            self.conn.xtest_fake_input(kind, detail, x11rb::CURRENT_TIME, self.root, x, y, 0).map_err(|e| e.to_string())?;
            self.conn.flush().map_err(|e| e.to_string())
        }

        pub fn move_to(&self, x: i16, y: i16) -> Result<(), String> {
            self.fake(x11rb::protocol::xproto::MOTION_NOTIFY_EVENT, 0, x.clamp(0, self.width as i16 - 1), y.clamp(0, self.height as i16 - 1))
        }

        /// X button numbers: 1 left, 2 middle, 3 right, 4/5 wheel up/down, 6/7 wheel left/right.
        pub fn button(&self, b: u8, down: bool) -> Result<(), String> {
            let kind = if down { x11rb::protocol::xproto::BUTTON_PRESS_EVENT } else { x11rb::protocol::xproto::BUTTON_RELEASE_EVENT };
            self.fake(kind, b, 0, 0)
        }

        /// The keycode that types `keysym`, and whether it needs Shift (it is in the shifted column), if the map has it.
        pub fn keycode(&self, keysym: u32) -> Option<(u8, bool)> {
            let per = self.per_keycode.max(1) as usize;
            // the unshifted column first, so "a" is never typed as Shift+A's keycode with Shift
            for want_col in 0..per.min(2) {
                if let Some(i) = self.keysyms.chunks(per).position(|col| col.get(want_col) == Some(&keysym)) {
                    return Some((self.min_keycode + i as u8, want_col == 1));
                }
            }
            self.keysyms.chunks(per).position(|col| col.contains(&keysym)).map(|i| (self.min_keycode + i as u8, false))
        }

        pub fn shift_keycode(&self) -> Option<u8> {
            self.keycode(0xffe1).map(|(k, _)| k)
        }

        pub fn key(&self, keycode: u8, down: bool) -> Result<(), String> {
            let kind = if down { x11rb::protocol::xproto::KEY_PRESS_EVENT } else { x11rb::protocol::xproto::KEY_RELEASE_EVENT };
            self.fake(kind, keycode, 0, 0)
        }
    }
}

/// The X keysym for a browser `KeyboardEvent.key`.
pub fn keysym_of(key: &str) -> Option<u32> {
    let named = match key {
        "Enter" => 0xff0d, "Backspace" => 0xff08, "Tab" => 0xff09, "Escape" => 0xff1b, "Delete" => 0xffff,
        "Home" => 0xff50, "ArrowLeft" => 0xff51, "ArrowUp" => 0xff52, "ArrowRight" => 0xff53, "ArrowDown" => 0xff54,
        "PageUp" => 0xff55, "PageDown" => 0xff56, "End" => 0xff57, "Insert" => 0xff63, "Shift" => 0xffe1,
        "Control" => 0xffe3, "Alt" => 0xffe9, "AltGraph" => 0xfe03, "Meta" | "OS" => 0xffeb, "CapsLock" => 0xffe5,
        "ContextMenu" => 0xff67, " " => 0x20,
        f if f.len() >= 2 && f.starts_with('F') && f[1..].parse::<u32>().is_ok_and(|n| (1..=12).contains(&n)) => 0xffbd + f[1..].parse::<u32>().unwrap(),
        _ => 0,
    };
    if named != 0 {
        return Some(named);
    }
    let mut chars = key.chars();
    match (chars.next(), chars.next()) {
        // one character: Latin-1 keysyms are the code point; others are 0x01000000 + the code point
        (Some(c), None) => Some(if (c as u32) < 0x100 { c as u32 } else { 0x0100_0000 + c as u32 }),
        _ => None,
    }
}

/// A command run against the sandbox display only: environment cleared but PATH, DISPLAY set to the sandbox's.
fn on_display(program: &str, display: &str) -> std::process::Command {
    let mut c = std::process::Command::new(program);
    c.env_clear().env("DISPLAY", display);
    if let Ok(p) = std::env::var("PATH") {
        c.env("PATH", p);
    }
    c
}

pub fn clipboard_get(display: &str) -> Result<String, String> {
    let o = on_display("xclip", display).args(["-selection", "clipboard", "-o"]).output()
        .map_err(|e| format!("xclip: {e} (install xclip for clipboard sync)"))?;
    if !o.status.success() {
        let why = String::from_utf8_lossy(&o.stderr).trim().to_string();
        // an empty clipboard is not an error (xclip says so on stderr); anything else is
        if why.is_empty() || why.contains("target STRING not available") || why.contains("Error: target") {
            return Ok(String::new());
        }
        return Err(format!("xclip: {why}"));
    }
    Ok(String::from_utf8_lossy(&o.stdout).into_owned())
}

pub fn clipboard_set(display: &str, text: &str) -> Result<(), String> {
    use std::io::Write;
    let mut child = on_display("xclip", display).args(["-selection", "clipboard", "-i"])
        .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null())
        .spawn().map_err(|e| format!("xclip: {e} (install xclip for clipboard sync)"))?;
    child.stdin.take().unwrap().write_all(text.as_bytes()).map_err(|e| e.to_string())?;
    // xclip stays to serve the selection and exits when another owner takes it: reaped then, never left a zombie
    std::thread::spawn(move || { let _ = child.wait(); });
    Ok(())
}

/// Type `text` into the focused window (any characters, including ones the keyboard map lacks).
pub fn type_text(display: &str, text: &str) -> Result<(), String> {
    let s = on_display("xdotool", display).args(["type", "--clearmodifiers", "--delay", "4", "--", text]).status()
        .map_err(|e| format!("xdotool: {e}"))?;
    if s.success() { Ok(()) } else { Err(format!("xdotool type ended with {s}")) }
}

/// The screen as an image (JPEG at `quality`, or PNG).
#[cfg(target_os = "linux")]
fn encode(rgb: &[u8], w: u16, h: u16, jpeg_quality: Option<u8>) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    match jpeg_quality {
        Some(q) => image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, q)
            .encode(rgb, w as u32, h as u32, image::ExtendedColorType::Rgb8).map_err(|e| e.to_string())?,
        None => {
            use image::ImageEncoder;
            image::codecs::png::PngEncoder::new(&mut out)
                .write_image(rgb, w as u32, h as u32, image::ExtendedColorType::Rgb8).map_err(|e| e.to_string())?
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------------------------------------------- REST

fn guard(s: &AppState, headers: &HeaderMap) -> Result<Arc<crate::computer::Computers>, Response> {
    if !owner_key(s, headers) {
        return Err(error(StatusCode::UNAUTHORIZED, "authentication_error", "the Computer tab needs the server's own API key"));
    }
    computers(s).map_err(|e| error(StatusCode::SERVICE_UNAVAILABLE, "server_error", e))
}

fn bad(e: String) -> Response {
    error(StatusCode::CONFLICT, "invalid_request_error", e)
}

pub async fn status_get(State(s): State<AppState>, headers: HeaderMap) -> Response {
    match guard(&s, &headers) {
        Ok(c) => Json(status(&c)).into_response(),
        Err(r) => r,
    }
}

pub async fn screenshot_get(State(s): State<AppState>, headers: HeaderMap) -> Response {
    let c = match guard(&s, &headers) { Ok(c) => c, Err(r) => return r };
    #[cfg(target_os = "linux")]
    {
        let r = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, String> {
            let d = c.sandbox_display()?;
            let x = x11::Display::open(&d)?;
            encode(&x.grab()?, x.width, x.height, None)
        }).await.unwrap_or_else(|e| Err(e.to_string()));
        match r {
            Ok(png) => ([(axum::http::header::CONTENT_TYPE, HeaderValue::from_static("image/png"))], png).into_response(),
            Err(e) => bad(e),
        }
    }
    #[cfg(not(target_os = "linux"))]
    { let _ = c; bad("the desktop view needs Linux for now".into()) }
}

pub async fn control_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let c = match guard(&s, &headers) { Ok(c) => c, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("control of the desktop can be taken");
    }
    match req["action"].as_str() {
        Some("take") => c.set_human(true),
        Some("give") => c.set_human(false),
        _ => return error(StatusCode::BAD_REQUEST, "invalid_request_error", "\"action\" is take or give"),
    }
    tracing::info!("computer: the operator {} control of the desktop", if c.human_in_control() { "took" } else { "handed back" });
    Json(status(&c)).into_response()
}

pub async fn clipboard_get_h(State(s): State<AppState>, headers: HeaderMap) -> Response {
    let c = match guard(&s, &headers) { Ok(c) => c, Err(r) => return r };
    let r = tokio::task::spawn_blocking(move || c.sandbox_display().and_then(|d| clipboard_get(&d))).await.unwrap_or_else(|e| Err(e.to_string()));
    match r {
        Ok(t) => Json(json!({"text": t})).into_response(),
        Err(e) => bad(e),
    }
}

pub async fn clipboard_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let c = match guard(&s, &headers) { Ok(c) => c, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("the desktop's clipboard can be set");
    }
    let Some(text) = req["text"].as_str().map(str::to_string) else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "\"text\" is required");
    };
    let r = tokio::task::spawn_blocking(move || c.sandbox_display().and_then(|d| clipboard_set(&d, &text))).await.unwrap_or_else(|e| Err(e.to_string()));
    match r {
        Ok(()) => Json(json!({"ok": true})).into_response(),
        Err(e) => bad(e),
    }
}

pub async fn type_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let c = match guard(&s, &headers) { Ok(c) => c, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("text can be typed on the desktop");
    }
    if !c.human_in_control() {
        return bad("take control first (the agent may be acting on the desktop)".into());
    }
    let Some(text) = req["text"].as_str().map(str::to_string) else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "\"text\" is required");
    };
    let r = tokio::task::spawn_blocking(move || c.sandbox_display().and_then(|d| type_text(&d, &text))).await.unwrap_or_else(|e| Err(e.to_string()));
    match r {
        Ok(()) => Json(json!({"ok": true})).into_response(),
        Err(e) => bad(e),
    }
}

pub async fn open_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let c = match guard(&s, &headers) { Ok(c) => c, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("apps can be opened on the desktop");
    }
    let app = req["app"].as_str().unwrap_or("").to_string();
    if !c.apps().iter().any(|a| *a == app) {
        return error(StatusCode::FORBIDDEN, "permission_error",
                     format!("{app:?} is not in the allowed apps (Settings › Computer and browser use › Allowed apps)"));
    }
    let r = tokio::task::spawn_blocking(move || -> Result<(), String> {
        let d = c.sandbox_display()?;
        #[cfg(target_os = "linux")]
        { crate::computer::backend::sandbox_app(&app, &d).spawn().map(|_| ()).map_err(|e| format!("{app}: {e}")) }
        #[cfg(not(target_os = "linux"))]
        { let _ = (d, app); Err("opening apps on the desktop needs Linux for now".to_string()) }
    }).await.unwrap_or_else(|e| Err(e.to_string()));
    match r {
        Ok(()) => Json(json!({"ok": true})).into_response(),
        Err(e) => bad(e),
    }
}

// ------------------------------------------------------------------------------------------------------ the stream

pub async fn stream(State(s): State<AppState>, headers: HeaderMap,
                    ws: Result<WebSocketUpgrade, axum::extract::ws::rejection::WebSocketUpgradeRejection>) -> Response {
    // a plain request (not a WebSocket): the key first, like every route, then "use a WebSocket"
    let Ok(ws) = ws else {
        if !owner_key(&s, &headers) {
            return error(StatusCode::UNAUTHORIZED, "authentication_error", "the Computer tab needs the server's own API key");
        }
        return error(StatusCode::UPGRADE_REQUIRED, "invalid_request_error", "this is a WebSocket (the Computer tab's stream)");
    };
    // another site's page must not open this socket with the user's browser (cross-site WebSocket hijacking)
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("the desktop can be viewed");
    }
    // the browser cannot send a key header on a WebSocket: it sends it in the first message (session)
    ws.max_message_size(1 << 20).on_upgrade(move |sock| session(s, headers, sock))
}

async fn send_json(tx: &mut futures::stream::SplitSink<WebSocket, Message>, v: Value) -> bool {
    tx.send(Message::Text(v.to_string().into())).await.is_ok()
}

async fn session(s: AppState, headers: HeaderMap, sock: WebSocket) {
    let (mut tx, mut rx) = sock.split();
    // the key: the Authorization header (CLI), else the first message (the browser)
    let mut ok = owner_key(&s, &headers);
    if !ok {
        if let Ok(Some(Ok(Message::Text(t)))) = tokio::time::timeout(Duration::from_secs(5), rx.next()).await {
            let v: Value = serde_json::from_str(&t).unwrap_or_default();
            if let (Some("auth"), Some(k)) = (v["type"].as_str(), v["key"].as_str()) {
                let mut h = HeaderMap::new();
                if let Ok(val) = HeaderValue::from_str(&format!("Bearer {k}")) {
                    h.insert(axum::http::header::AUTHORIZATION, val);
                    ok = owner_key(&s, &h);
                }
            }
        }
    }
    if !ok {
        let _ = send_json(&mut tx, json!({"type": "error", "message": "the Computer tab needs the server's own API key"})).await;
        return;
    }
    let c = match computers(&s) {
        Ok(c) => c,
        Err(e) => { let _ = send_json(&mut tx, json!({"type": "error", "message": e})).await; return; }
    };
    #[cfg(not(target_os = "linux"))]
    {
        let _ = send_json(&mut tx, json!({"type": "error", "message": "the desktop view needs Linux for now"})).await;
        let _ = c;
    }
    #[cfg(target_os = "linux")]
    linux_session(c, tx, rx).await;
}

#[cfg(target_os = "linux")]
async fn linux_session(c: Arc<crate::computer::Computers>, mut tx: futures::stream::SplitSink<WebSocket, Message>,
                       mut rx: futures::stream::SplitStream<WebSocket>) {
    let c2 = c.clone();
    let opened = tokio::task::spawn_blocking(move || -> Result<(String, x11::Display), String> {
        let d = c2.sandbox_display()?;
        let x = x11::Display::open(&d)?;
        Ok((d, x))
    }).await.unwrap_or_else(|e| Err(e.to_string()));
    let (display, x) = match opened {
        Ok(v) => v,
        Err(e) => { let _ = send_json(&mut tx, json!({"type": "error", "message": e})).await; return; }
    };
    let (w, h) = (x.width, x.height);
    let x = Arc::new(std::sync::Mutex::new(x));
    let state = |c: &crate::computer::Computers| json!({"type": "state", "display": display, "width": w, "height": h,
                                                        "operator_in_control": c.human_in_control(), "mode": c.mode_name(), "apps": c.apps()});
    if !send_json(&mut tx, state(&c)).await {
        return;
    }
    let mut took = false;
    let mut last_hash = 0u64;
    let mut last_state = c.human_in_control();
    let mut last_touch = Instant::now();
    let mut tick = tokio::time::interval(Duration::from_millis(125));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            _ = tick.tick() => {
                if last_touch.elapsed() > Duration::from_secs(30) {
                    c.touch(); // watching keeps the sandbox from being stopped as idle
                    last_touch = Instant::now();
                }
                if c.human_in_control() != last_state {
                    last_state = c.human_in_control();
                    if !send_json(&mut tx, state(&c)).await { break; }
                }
                let xg = x.clone();
                let frame = tokio::task::spawn_blocking(move || -> Result<(u64, Vec<u8>), String> {
                    let rgb = xg.lock().unwrap().grab()?;
                    use std::hash::{Hash, Hasher};
                    let mut hs = std::collections::hash_map::DefaultHasher::new();
                    rgb.hash(&mut hs);
                    Ok((hs.finish(), rgb))
                }).await.unwrap_or_else(|e| Err(e.to_string()));
                match frame {
                    Ok((hash, rgb)) if hash != last_hash => {
                        last_hash = hash;
                        let jpeg = tokio::task::spawn_blocking(move || encode(&rgb, w, h, Some(70))).await.unwrap_or_else(|e| Err(e.to_string()));
                        match jpeg {
                            Ok(j) => if tx.send(Message::Binary(j.into())).await.is_err() { break },
                            Err(e) => { let _ = send_json(&mut tx, json!({"type": "error", "message": e})).await; break; }
                        }
                    }
                    Ok(_) => {}
                    Err(e) => { let _ = send_json(&mut tx, json!({"type": "error", "message": format!("the desktop stopped: {e}")})).await; break; }
                }
            }
            msg = rx.next() => {
                let Some(Ok(msg)) = msg else { break };
                let Message::Text(t) = msg else { continue };
                let v: Value = serde_json::from_str(&t).unwrap_or_default();
                let kind = v["type"].as_str().unwrap_or("").to_string();
                match kind.as_str() {
                    "take" => { c.set_human(true); took = true; tracing::info!("computer: the operator took control of the desktop"); }
                    "give" => { c.set_human(false); took = false; tracing::info!("computer: the operator handed the desktop back"); }
                    "clipboard_get" => {
                        let d = display.clone();
                        let r = tokio::task::spawn_blocking(move || clipboard_get(&d)).await.unwrap_or_else(|e| Err(e.to_string()));
                        let msg = match r { Ok(text) => json!({"type": "clipboard", "text": text}), Err(e) => json!({"type": "error", "message": e}) };
                        if !send_json(&mut tx, msg).await { break; }
                    }
                    _ => {
                        // everything else is input: only while the operator has control
                        if !c.human_in_control() {
                            let _ = send_json(&mut tx, json!({"type": "notice", "message": "take control first: the agent may be acting on the desktop"})).await;
                            continue;
                        }
                        let (xi, d) = (x.clone(), display.clone());
                        let r = tokio::task::spawn_blocking(move || input(&xi, &d, &kind, &v)).await.unwrap_or_else(|e| Err(e.to_string()));
                        if let Err(e) = r {
                            if !send_json(&mut tx, json!({"type": "notice", "message": e})).await { break; }
                        }
                    }
                }
            }
        }
    }
    // a closed tab hands control back, so the agent is not left waiting
    if took {
        c.set_human(false);
        tracing::info!("computer: the viewer closed; control handed back to the agent");
    }
}

/// One input message from the viewer, into the sandbox display.
#[cfg(target_os = "linux")]
fn input(x: &std::sync::Mutex<x11::Display>, display: &str, kind: &str, v: &Value) -> Result<(), String> {
    let xy = || (v["x"].as_f64().unwrap_or(0.0) as i16, v["y"].as_f64().unwrap_or(0.0) as i16);
    match kind {
        "move" => { let (px, py) = xy(); x.lock().unwrap().move_to(px, py) }
        "button" => {
            let (px, py) = xy();
            let b = v["button"].as_u64().unwrap_or(1).clamp(1, 3) as u8;
            let d = x.lock().unwrap();
            d.move_to(px, py)?;
            d.button(b, v["down"].as_bool().unwrap_or(true))
        }
        "wheel" => {
            let d = x.lock().unwrap();
            let (px, py) = xy();
            d.move_to(px, py)?;
            let (dy, dx) = (v["dy"].as_f64().unwrap_or(0.0), v["dx"].as_f64().unwrap_or(0.0));
            let (b, n) = if dy.abs() >= dx.abs() { (if dy < 0.0 { 4 } else { 5 }, (dy.abs() / 50.0).ceil().clamp(1.0, 10.0)) }
                         else { (if dx < 0.0 { 6 } else { 7 }, (dx.abs() / 50.0).ceil().clamp(1.0, 10.0)) };
            for _ in 0..n as u32 {
                d.button(b, true)?;
                d.button(b, false)?;
            }
            Ok(())
        }
        "key" => {
            let key = v["key"].as_str().unwrap_or("");
            let down = v["down"].as_bool().unwrap_or(true);
            let sym = keysym_of(key).ok_or_else(|| format!("unknown key {key:?}"))?;
            let d = x.lock().unwrap();
            match d.keycode(sym) {
                // a character that needs Shift, sent without it (a phone keyboard, a synthetic event): Shift around it
                Some((kc, true)) if key.chars().count() == 1 && !v["shift"].as_bool().unwrap_or(false) => {
                    if down {
                        if let Some(sk) = d.shift_keycode() { d.key(sk, true)?; d.key(kc, true)?; d.key(kc, false)?; return d.key(sk, false); }
                    }
                    Ok(())
                }
                Some((kc, _)) => d.key(kc, down),
                // a character the keyboard map lacks: type it (on key down only)
                None if down && key.chars().count() == 1 => { drop(d); type_text(display, key) }
                None => Ok(()),
            }
        }
        "text" => type_text(display, v["text"].as_str().unwrap_or("")),
        "clipboard_set" => clipboard_set(display, v["text"].as_str().unwrap_or("")),
        other => Err(format!("unknown message {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_keys_map_to_x_keysyms() {
        assert_eq!(keysym_of("a"), Some(0x61));
        assert_eq!(keysym_of("A"), Some(0x41));
        assert_eq!(keysym_of("Enter"), Some(0xff0d));
        assert_eq!(keysym_of("ArrowLeft"), Some(0xff51));
        assert_eq!(keysym_of("F1"), Some(0xffbe));
        assert_eq!(keysym_of("F12"), Some(0xffc9));
        assert_eq!(keysym_of(" "), Some(0x20));
        assert_eq!(keysym_of("é"), Some(0xe9), "Latin-1 is the code point");
        assert_eq!(keysym_of("€"), Some(0x0100_20ac), "beyond Latin-1: 0x01000000 + the code point");
        assert_eq!(keysym_of("Unidentified"), None);
        assert_eq!(keysym_of("F13"), None);
    }

    #[test]
    fn display_commands_never_inherit_the_real_display() {
        // (no set_var here: changing the environment while other test threads read it is undefined behaviour)
        let c = on_display("xclip", ":42");
        let envs: Vec<(String, Option<String>)> = c.get_envs()
            .map(|(k, v)| (k.to_string_lossy().into_owned(), v.map(|v| v.to_string_lossy().into_owned()))).collect();
        assert!(envs.iter().any(|(k, v)| k == "DISPLAY" && v.as_deref() == Some(":42")));
        assert!(!envs.iter().any(|(k, v)| k == "WAYLAND_DISPLAY" && v.is_some()), "env cleared: {envs:?}");
        assert!(envs.iter().all(|(k, _)| k == "DISPLAY" || k == "PATH"), "only DISPLAY and PATH are passed: {envs:?}");
    }
}
