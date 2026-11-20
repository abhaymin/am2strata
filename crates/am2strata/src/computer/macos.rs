//! The Mac's own desktop for computer use (macOS): the real screen, driven the way a person would, through the system's
//! frameworks with no new crates — Quartz events (CoreGraphics) for the mouse and keyboard, `screencapture` for the
//! screenshot, the window server's list for windows, the accessibility API for the control under a click and for
//! focusing a window, `open -a` for apps and `pbcopy`/`pbpaste` for the clipboard.
//!
//! There is no sandbox on a Mac: this drives the desktop the owner is using, so it is offered only in the Assist,
//! Trusted job and Watch only modes (engine.rs), with every step checked by the rules first. macOS asks the owner to
//! allow it once, in System Settings › Privacy & Security: **Accessibility** (input, the control under a click, window
//! focus) and **Screen Recording** (screenshots, window titles) for the `am2strata` binary. Until then each step fails with
//! that message instead of doing nothing.
//!
//! Coordinates: the model sees the screen scaled to fit 1280x800 (its aspect kept) and clicks in that picture's pixels;
//! here they are turned into the screen's points. Only the main display is used.

use super::backend::{Backend, Buffer, Button, Key, Size, Window, WindowId};
use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_long};
use std::process::{Command, Stdio};

// ------------------------------------------------------------------------------------------------------ the frameworks

type CFTypeRef = *const c_void;
type CFStringRef = *const c_void;
type CFArrayRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CFNumberRef = *const c_void;
type CGEventRef = *mut c_void;
type AXUIElementRef = *const c_void;
type CFIndex = c_long;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CGPoint { x: f64, y: f64 }
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CGSize { width: f64, height: f64 }
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CGRect { origin: CGPoint, size: CGSize }

const UTF8: u32 = 0x0800_0100; // kCFStringEncodingUTF8
const SINT64: CFIndex = 4; // kCFNumberSInt64Type
const HID_TAP: u32 = 0; // kCGHIDEventTap
const SCROLL_LINE: u32 = 1; // kCGScrollEventUnitLine
const CLICK_STATE: u32 = 1; // kCGMouseEventClickState
// kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
const ON_SCREEN: u32 = 1 | 16;

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(cf: CFTypeRef);
    fn CFGetTypeID(cf: CFTypeRef) -> usize;
    fn CFStringGetTypeID() -> usize;
    fn CFStringCreateWithCString(alloc: CFTypeRef, s: *const c_char, encoding: u32) -> CFStringRef;
    fn CFStringGetLength(s: CFStringRef) -> CFIndex;
    fn CFStringGetMaximumSizeForEncoding(len: CFIndex, encoding: u32) -> CFIndex;
    fn CFStringGetCString(s: CFStringRef, buf: *mut c_char, size: CFIndex, encoding: u32) -> bool;
    fn CFArrayGetCount(a: CFArrayRef) -> CFIndex;
    fn CFArrayGetValueAtIndex(a: CFArrayRef, i: CFIndex) -> CFTypeRef;
    fn CFDictionaryGetValue(d: CFDictionaryRef, key: CFTypeRef) -> CFTypeRef;
    fn CFNumberGetValue(n: CFNumberRef, kind: CFIndex, out: *mut c_void) -> bool;
    static kCFBooleanTrue: CFTypeRef;
}

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGMainDisplayID() -> u32;
    fn CGDisplayBounds(display: u32) -> CGRect;
    fn CGPreflightScreenCaptureAccess() -> bool;
    fn CGEventCreateMouseEvent(source: CFTypeRef, kind: u32, at: CGPoint, button: u32) -> CGEventRef;
    fn CGEventCreateKeyboardEvent(source: CFTypeRef, key: u16, down: bool) -> CGEventRef;
    fn CGEventCreateScrollWheelEvent2(source: CFTypeRef, units: u32, count: u32, w1: i32, w2: i32, w3: i32) -> CGEventRef;
    fn CGEventKeyboardSetUnicodeString(e: CGEventRef, len: CFIndex, chars: *const u16);
    fn CGEventSetFlags(e: CGEventRef, flags: u64);
    fn CGEventSetIntegerValueField(e: CGEventRef, field: u32, value: i64);
    fn CGEventPost(tap: u32, e: CGEventRef);
    fn CGWindowListCopyWindowInfo(option: u32, relative_to: u32) -> CFArrayRef;
    static kCGWindowNumber: CFStringRef;
    static kCGWindowOwnerName: CFStringRef;
    static kCGWindowOwnerPID: CFStringRef;
    static kCGWindowName: CFStringRef;
    static kCGWindowLayer: CFStringRef;
}

#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    fn AXIsProcessTrusted() -> bool;
    fn AXUIElementCreateSystemWide() -> AXUIElementRef;
    fn AXUIElementCreateApplication(pid: i32) -> AXUIElementRef;
    fn AXUIElementCopyElementAtPosition(app: AXUIElementRef, x: f32, y: f32, out: *mut AXUIElementRef) -> i32;
    fn AXUIElementCopyAttributeValue(e: AXUIElementRef, attr: CFStringRef, out: *mut CFTypeRef) -> i32;
    fn AXUIElementSetAttributeValue(e: AXUIElementRef, attr: CFStringRef, value: CFTypeRef) -> i32;
    fn AXUIElementPerformAction(e: AXUIElementRef, action: CFStringRef) -> i32;
}

/// A CF object released when it goes out of scope.
struct Owned(CFTypeRef);
impl Drop for Owned {
    fn drop(&mut self) { if !self.0.is_null() { unsafe { CFRelease(self.0) } } }
}

/// A CFString made from `s` (released on drop).
fn cfstr(s: &str) -> Owned {
    let c = CString::new(s).unwrap_or_default();
    Owned(unsafe { CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), UTF8) })
}

/// A CFString's text, or None when `v` is null or not a string.
fn string_of(v: CFTypeRef) -> Option<String> {
    unsafe {
        if v.is_null() || CFGetTypeID(v) != CFStringGetTypeID() { return None; }
        let cap = CFStringGetMaximumSizeForEncoding(CFStringGetLength(v), UTF8) + 1;
        let mut buf = vec![0 as c_char; cap.max(1) as usize];
        if !CFStringGetCString(v, buf.as_mut_ptr(), cap, UTF8) { return None; }
        Some(CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned())
    }
}

fn number_of(v: CFTypeRef) -> Option<i64> {
    if v.is_null() { return None; }
    let mut n: i64 = 0;
    unsafe { CFNumberGetValue(v, SINT64, &mut n as *mut i64 as *mut c_void) }.then_some(n)
}

/// An accessibility attribute of `e` as text.
fn ax_text(e: AXUIElementRef, attr: &str) -> Option<String> {
    let name = cfstr(attr);
    let mut out: CFTypeRef = std::ptr::null();
    if unsafe { AXUIElementCopyAttributeValue(e, name.0, &mut out) } != 0 { return None; }
    let v = Owned(out);
    string_of(v.0).filter(|s| !s.trim().is_empty())
}

// --------------------------------------------------------------------------------------------------------- the backend

/// The owner's Mac desktop (main display). Holds nothing: the screen's size is read on each step, so a display change
/// (resolution, a monitor plugged in) is picked up.
#[derive(Debug, Default)]
pub struct MacBackend;

const NEED_ACCESSIBILITY: &str = "macOS has not allowed am2Strata to control the Mac yet: the owner must turn on \
    am2strata in System Settings › Privacy & Security › Accessibility (then restart am2Strata)";
const NEED_SCREEN: &str = "macOS has not allowed am2Strata to see the screen yet: the owner must turn on am2strata in \
    System Settings › Privacy & Security › Screen Recording (then restart am2Strata)";

impl MacBackend {
    pub fn new() -> MacBackend { MacBackend }

    /// Whether macOS lets this process send input and read controls (Accessibility) and see the screen (Screen Recording).
    pub fn permissions() -> (bool, bool) {
        unsafe { (AXIsProcessTrusted(), CGPreflightScreenCaptureAccess()) }
    }

    fn input_allowed() -> Result<(), String> {
        if unsafe { AXIsProcessTrusted() } { Ok(()) } else { Err(NEED_ACCESSIBILITY.into()) }
    }

    /// The main display in points.
    fn bounds() -> CGRect { unsafe { CGDisplayBounds(CGMainDisplayID()) } }

    /// A point of the model's picture (`size()`) on the screen, in points.
    fn to_screen(&self, x: i32, y: i32) -> CGPoint {
        let (b, s) = (Self::bounds(), self.size());
        map_point(x, y, s, b.origin.x, b.origin.y, b.size.width, b.size.height)
    }

    fn post_mouse(kind: u32, at: CGPoint, button: u32, clicks: i64) -> Result<(), String> {
        unsafe {
            let e = CGEventCreateMouseEvent(std::ptr::null(), kind, at, button);
            if e.is_null() { return Err("macOS refused to make the mouse event".into()); }
            if clicks > 0 { CGEventSetIntegerValueField(e, CLICK_STATE, clicks); }
            CGEventPost(HID_TAP, e);
            CFRelease(e as CFTypeRef);
        }
        pause(12);
        Ok(())
    }

    fn post_key(code: u16, down: bool, flags: u64) -> Result<(), String> {
        unsafe {
            let e = CGEventCreateKeyboardEvent(std::ptr::null(), code, down);
            if e.is_null() { return Err("macOS refused to make the key event".into()); }
            CGEventSetFlags(e, flags);
            CGEventPost(HID_TAP, e);
            CFRelease(e as CFTypeRef);
        }
        pause(8);
        Ok(())
    }

    /// Type one run of characters (no newline) as Unicode, whatever the keyboard layout.
    fn post_text(chunk: &[u16]) -> Result<(), String> {
        for down in [true, false] {
            unsafe {
                let e = CGEventCreateKeyboardEvent(std::ptr::null(), 0, down);
                if e.is_null() { return Err("macOS refused to make the key event".into()); }
                CGEventSetFlags(e, 0);
                CGEventKeyboardSetUnicodeString(e, chunk.len() as CFIndex, chunk.as_ptr());
                CGEventPost(HID_TAP, e);
                CFRelease(e as CFTypeRef);
            }
        }
        pause(10);
        Ok(())
    }

    fn windows_raw() -> Result<Vec<(Window, i32)>, String> {
        unsafe {
            let list = Owned(CGWindowListCopyWindowInfo(ON_SCREEN, 0));
            if list.0.is_null() { return Err("macOS gave no window list".into()); }
            let mut out = Vec::new();
            for i in 0..CFArrayGetCount(list.0) {
                let d = CFArrayGetValueAtIndex(list.0, i);
                // layer 0: ordinary app windows (not the menu bar, the Dock, overlays)
                if number_of(CFDictionaryGetValue(d, kCGWindowLayer)) != Some(0) { continue; }
                let Some(id) = number_of(CFDictionaryGetValue(d, kCGWindowNumber)) else { continue };
                let owner = string_of(CFDictionaryGetValue(d, kCGWindowOwnerName));
                let pid = number_of(CFDictionaryGetValue(d, kCGWindowOwnerPID)).unwrap_or(0) as i32;
                // the title needs Screen Recording; without it the app's name stands in
                let title = string_of(CFDictionaryGetValue(d, kCGWindowName)).filter(|t| !t.is_empty())
                    .or_else(|| owner.clone()).unwrap_or_default();
                out.push((Window { id: id as WindowId, title, process: owner }, pid));
            }
            Ok(out)
        }
    }
}

impl Backend for MacBackend {
    fn size(&self) -> Size {
        let b = Self::bounds();
        fit(b.size.width, b.size.height)
    }

    fn screenshot(&self) -> Result<Buffer, String> {
        if !unsafe { CGPreflightScreenCaptureAccess() } { return Err(NEED_SCREEN.into()); }
        let path = std::env::temp_dir().join(format!("am2strata-shot-{}-{:?}.png", std::process::id(), std::thread::current().id()));
        // -x: no sound; -m: the main display; -C would add the cursor, which hides what is under it
        let st = Command::new("/usr/sbin/screencapture").args(["-x", "-m", "-t", "png"]).arg(&path)
            .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status()
            .map_err(|e| format!("running screencapture: {e}"))?;
        let bytes = std::fs::read(&path);
        let _ = std::fs::remove_file(&path);
        if !st.success() { return Err("screencapture failed (Screen Recording may be off for am2strata)".into()); }
        let img = image::load_from_memory(&bytes.map_err(|e| format!("reading the screenshot: {e}"))?)
            .map_err(|e| format!("decoding the screenshot: {e}"))?;
        let s = self.size();
        let img = if img.width() == s.width && img.height() == s.height { img.to_rgba8() } else {
            image::imageops::resize(&img.to_rgba8(), s.width, s.height, image::imageops::FilterType::Triangle)
        };
        Ok(Buffer { width: img.width(), height: img.height(), rgba: img.into_raw() })
    }

    fn list_windows(&self) -> Result<Vec<Window>, String> {
        Ok(Self::windows_raw()?.into_iter().map(|(w, _)| w).collect())
    }

    fn focus(&self, id: WindowId) -> Result<(), String> {
        Self::input_allowed()?;
        let (win, pid) = Self::windows_raw()?.into_iter().find(|(w, _)| w.id == id)
            .ok_or_else(|| format!("there is no window {id} on screen (list them with computer_windows)"))?;
        unsafe {
            let app = Owned(AXUIElementCreateApplication(pid));
            let front = cfstr("AXFrontmost");
            if AXUIElementSetAttributeValue(app.0, front.0, kCFBooleanTrue) != 0 {
                return Err(format!("macOS would not bring {} to the front", win.process.as_deref().unwrap_or("the app")));
            }
            // then raise the window with that title, if the app has several
            let attr = cfstr("AXWindows");
            let mut wins: CFTypeRef = std::ptr::null();
            if AXUIElementCopyAttributeValue(app.0, attr.0, &mut wins) == 0 {
                let wins = Owned(wins);
                let raise = cfstr("AXRaise");
                for i in 0..CFArrayGetCount(wins.0) {
                    let w = CFArrayGetValueAtIndex(wins.0, i);
                    if ax_text(w, "AXTitle").as_deref() == Some(win.title.as_str()) {
                        AXUIElementPerformAction(w, raise.0);
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    fn click(&self, x: i32, y: i32, button: Button) -> Result<(), String> {
        Self::input_allowed()?;
        let at = self.to_screen(x, y);
        let (down, up, b) = match button { Button::Left => (1, 2, 0), Button::Right => (3, 4, 1), Button::Middle => (25, 26, 2) };
        Self::post_mouse(5, at, 0, 0)?; // move there first, so hover effects and the click target match
        Self::post_mouse(down, at, b, 1)?;
        Self::post_mouse(up, at, b, 1)
    }

    fn double_click(&self, x: i32, y: i32) -> Result<(), String> {
        Self::input_allowed()?;
        let at = self.to_screen(x, y);
        Self::post_mouse(5, at, 0, 0)?;
        for n in [1, 2] {
            Self::post_mouse(1, at, 0, n)?;
            Self::post_mouse(2, at, 0, n)?;
        }
        Ok(())
    }

    fn drag(&self, x: i32, y: i32, to_x: i32, to_y: i32) -> Result<(), String> {
        Self::input_allowed()?;
        let (a, b) = (self.to_screen(x, y), self.to_screen(to_x, to_y));
        Self::post_mouse(5, a, 0, 0)?;
        Self::post_mouse(1, a, 0, 1)?;
        // a few steps on the way, so apps that track the drag see it move
        let mut moved = Ok(());
        for i in 1..=12 {
            let t = i as f64 / 12.0;
            let p = CGPoint { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
            moved = moved.and(Self::post_mouse(6, p, 0, 0));
            pause(10);
        }
        // let go even when a move failed, so the button is never left held down
        let up = Self::post_mouse(2, b, 0, 1);
        moved.and(up)
    }

    fn move_mouse(&self, x: i32, y: i32) -> Result<(), String> {
        Self::input_allowed()?;
        Self::post_mouse(5, self.to_screen(x, y), 0, 0)
    }

    fn type_text(&self, text: &str) -> Result<(), String> {
        Self::input_allowed()?;
        for (i, line) in text.split('\n').enumerate() {
            if i > 0 {
                Self::post_key(36, true, 0)?; // Return
                Self::post_key(36, false, 0)?;
            }
            let units: Vec<u16> = line.encode_utf16().collect();
            // macOS takes at most 20 characters per event
            for chunk in units.chunks(20) {
                Self::post_text(chunk)?;
            }
        }
        Ok(())
    }

    fn press_keys(&self, keys: &[Key]) -> Result<(), String> {
        Self::input_allowed()?;
        let (flags, codes) = combo(keys)?;
        let mut held: Vec<u16> = Vec::new();
        let mut flag_acc = 0u64;
        // modifiers down in order, then the keys, then everything up in reverse
        for (code, flag) in &codes {
            flag_acc |= flag;
            Self::post_key(*code, true, if *flag != 0 { flag_acc } else { flags })?;
            held.push(*code);
        }
        for code in held.iter().rev() {
            Self::post_key(*code, false, 0)?;
        }
        Ok(())
    }

    fn scroll(&self, dx: i32, dy: i32) -> Result<(), String> {
        Self::input_allowed()?;
        // the tool's units are screens: about 10 lines each; Quartz counts positive as up and left
        let (v, h) = (-dy * 10, -dx * 10);
        if v == 0 && h == 0 { return Ok(()); }
        unsafe {
            let e = CGEventCreateScrollWheelEvent2(std::ptr::null(), SCROLL_LINE, 2, v, h, 0);
            if e.is_null() { return Err("macOS refused to make the scroll event".into()); }
            CGEventPost(HID_TAP, e);
            CFRelease(e as CFTypeRef);
        }
        Ok(())
    }

    fn open_app(&self, name: &str) -> Result<(), String> {
        let out = Command::new("/usr/bin/open").arg("-a").arg(name).stdin(Stdio::null()).output()
            .map_err(|e| format!("running open: {e}"))?;
        if out.status.success() { Ok(()) } else {
            Err(format!("'{name}' could not be opened: {}", String::from_utf8_lossy(&out.stderr).trim()))
        }
    }

    fn clipboard_get(&self) -> Result<String, String> {
        let out = Command::new("/usr/bin/pbpaste").stdin(Stdio::null()).output().map_err(|e| format!("running pbpaste: {e}"))?;
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    fn clipboard_set(&self, text: &str) -> Result<(), String> {
        use std::io::Write;
        let mut c = Command::new("/usr/bin/pbcopy").stdin(Stdio::piped()).spawn().map_err(|e| format!("running pbcopy: {e}"))?;
        c.stdin.take().ok_or("pbcopy has no input")?.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
        if c.wait().map_err(|e| e.to_string())?.success() { Ok(()) } else { Err("pbcopy failed".into()) }
    }

    fn label_at(&self, x: i32, y: i32) -> Option<String> {
        if !unsafe { AXIsProcessTrusted() } { return None; }
        let at = self.to_screen(x, y);
        unsafe {
            let sys = Owned(AXUIElementCreateSystemWide());
            let mut el: AXUIElementRef = std::ptr::null();
            if AXUIElementCopyElementAtPosition(sys.0, at.x as f32, at.y as f32, &mut el) != 0 { return None; }
            let el = Owned(el);
            ["AXTitle", "AXDescription", "AXHelp"].iter().find_map(|a| ax_text(el.0, a))
        }
    }

    fn focused_is_secure(&self) -> bool {
        if !unsafe { AXIsProcessTrusted() } { return false; }
        unsafe {
            let sys = Owned(AXUIElementCreateSystemWide());
            let attr = cfstr("AXFocusedUIElement");
            let mut el: CFTypeRef = std::ptr::null();
            if AXUIElementCopyAttributeValue(sys.0, attr.0, &mut el) != 0 { return false; }
            let el = Owned(el);
            ax_text(el.0, "AXSubrole").as_deref() == Some("AXSecureTextField")
        }
    }
}

fn pause(ms: u64) { std::thread::sleep(std::time::Duration::from_millis(ms)) }

// ------------------------------------------------------------------------------------------- pure parts (unit-tested)

/// The model's picture for a screen of `w` x `h` points: scaled down to fit 1280x800 with its aspect kept (never up).
pub(crate) fn fit(w: f64, h: f64) -> Size {
    if !(w > 0.0 && h > 0.0) { return Size::DEFAULT; }
    let s = (1280.0 / w).min(800.0 / h).min(1.0);
    Size { width: (w * s).round().max(1.0) as u32, height: (h * s).round().max(1.0) as u32, depth: 24 }
}

/// A pixel of the model's picture (`size`) to a point on a screen at (ox, oy) of `w` x `h` points.
fn map_point(x: i32, y: i32, size: Size, ox: f64, oy: f64, w: f64, h: f64) -> CGPoint {
    CGPoint { x: ox + (x as f64 + 0.5) * w / size.width as f64, y: oy + (y as f64 + 0.5) * h / size.height as f64 }
}

const SHIFT: u64 = 0x2_0000;
const CONTROL: u64 = 0x4_0000;
const OPTION: u64 = 0x8_0000;
const COMMAND: u64 = 0x10_0000;

/// A key combination as Quartz key codes with each one's modifier flag (0 for an ordinary key), plus the flags held for
/// the ordinary keys. "Ctrl+a" is the Control key; on a Mac most shortcuts are Cmd ("Cmd+a").
pub(crate) fn combo(keys: &[Key]) -> Result<(u64, Vec<(u16, u64)>), String> {
    let mut flags = 0u64;
    let mut out = Vec::new();
    for k in keys {
        let (code, flag) = keycode(k.as_str()).ok_or_else(|| format!("unknown key \"{}\" (use names like Cmd, Shift, Option, Ctrl, Enter, Tab, Escape, Up, F5, a)", k.as_str()))?;
        flags |= flag;
        out.push((code, flag));
    }
    Ok((flags, out))
}

/// A key name to its Quartz key code (US layout positions) and, for a modifier, its flag.
fn keycode(name: &str) -> Option<(u16, u64)> {
    let low = name.trim().to_ascii_lowercase();
    let m = match low.as_str() {
        "cmd" | "command" | "meta" | "super" | "win" | "⌘" => Some((55, COMMAND)),
        "shift" => Some((56, SHIFT)),
        "option" | "opt" | "alt" | "⌥" => Some((58, OPTION)),
        "ctrl" | "control" | "⌃" => Some((59, CONTROL)),
        _ => None,
    };
    if m.is_some() { return m; }
    let code: u16 = match low.as_str() {
        "enter" | "return" => 36, "tab" => 48, "space" => 49, "backspace" | "delete" => 51, "escape" | "esc" => 53,
        "forwarddelete" | "del" => 117, "home" => 115, "end" => 119, "pageup" | "page_up" => 116, "pagedown" | "page_down" => 121,
        "left" => 123, "right" => 124, "down" => 125, "up" => 126, "capslock" => 57,
        "f1" => 122, "f2" => 120, "f3" => 99, "f4" => 118, "f5" => 96, "f6" => 97, "f7" => 98, "f8" => 100,
        "f9" => 101, "f10" => 109, "f11" => 103, "f12" => 111,
        s if s.chars().count() == 1 => match s.chars().next().unwrap() {
            'a' => 0, 's' => 1, 'd' => 2, 'f' => 3, 'h' => 4, 'g' => 5, 'z' => 6, 'x' => 7, 'c' => 8, 'v' => 9, 'b' => 11,
            'q' => 12, 'w' => 13, 'e' => 14, 'r' => 15, 'y' => 16, 't' => 17, '1' => 18, '2' => 19, '3' => 20, '4' => 21,
            '6' => 22, '5' => 23, '=' => 24, '9' => 25, '7' => 26, '-' => 27, '8' => 28, '0' => 29, ']' => 30, 'o' => 31,
            'u' => 32, '[' => 33, 'i' => 34, 'p' => 35, 'l' => 37, 'j' => 38, '\'' => 39, 'k' => 40, ';' => 41, '\\' => 42,
            ',' => 43, '/' => 44, 'n' => 45, 'm' => 46, '.' => 47, '`' => 50,
            _ => return None,
        },
        _ => return None,
    };
    Some((code, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_screen_is_scaled_to_fit_with_its_aspect() {
        assert_eq!(fit(1920.0, 1080.0), Size { width: 1280, height: 720, depth: 24 });
        assert_eq!(fit(1440.0, 900.0), Size { width: 1280, height: 800, depth: 24 });
        assert_eq!(fit(1024.0, 768.0), Size { width: 1024, height: 768, depth: 24 }); // never scaled up
        assert_eq!(fit(0.0, 0.0), Size::DEFAULT);
    }

    #[test]
    fn a_pixel_lands_on_the_matching_point() {
        let s = Size { width: 1280, height: 720, depth: 24 };
        let p = map_point(640, 360, s, 0.0, 0.0, 1920.0, 1080.0);
        assert!((p.x - 960.75).abs() < 1e-9 && (p.y - 540.75).abs() < 1e-9, "{p:?}");
        let corner = map_point(1279, 719, s, 0.0, 0.0, 1920.0, 1080.0);
        assert!(corner.x < 1920.0 && corner.y < 1080.0);
    }

    #[test]
    fn key_names_become_codes_and_flags() {
        let (flags, codes) = combo(&[Key::new("Cmd"), Key::new("Shift"), Key::new("t")]).unwrap();
        assert_eq!(flags, COMMAND | SHIFT);
        assert_eq!(codes, vec![(55, COMMAND), (56, SHIFT), (17, 0)]);
        assert_eq!(combo(&[Key::new("Enter")]).unwrap().1, vec![(36, 0)]);
        assert_eq!(combo(&[Key::new("ctrl"), Key::new("A")]).unwrap().1, vec![(59, CONTROL), (0, 0)]);
        assert!(combo(&[Key::new("Hyper")]).unwrap_err().contains("unknown key"));
    }

    /// The real desktop: run by hand with `--ignored` on a Mac where am2strata's test binary has Screen Recording.
    #[test]
    #[ignore = "needs a Mac session with Screen Recording allowed for the test binary"]
    fn a_screenshot_of_this_mac() {
        let b = MacBackend::new();
        println!("permissions (accessibility, screen): {:?}", MacBackend::permissions());
        let s = b.size();
        let shot = b.screenshot().unwrap();
        assert_eq!((shot.width, shot.height), (s.width, s.height));
        println!("{} windows", b.list_windows().unwrap().len());
    }
}
