//! The computer-use desktop tools, offered only when `"computer": {"use": true}` (`computer/mod.rs` / server_tools.rs). One
//! name list (for run.rs's dispatch) and one JSON schema per tool, with descriptions that tell the model what it drives —
//! this PC's desktop, input on a display via xdotool/XTest — and that it sees a screenshot after each step.
//!
//! Each result carries a screenshot (when the model reads pictures), else the window list and the focused window title;
//! `computer_open_app` opens only an app in the allow list (the rules engine refuses the rest before the step runs).

/// The desktop computer tools, matched against a request's tool call in run.rs (`crate::computer::TOOL_NAMES`).
pub const TOOL_NAMES: [&str; 16] = [
    "computer_screenshot", "computer_click", "computer_type", "computer_key",
    "computer_scroll", "computer_open_app", "computer_wait",
    "computer_double_click", "computer_drag", "computer_move", "computer_windows", "computer_focus", "computer_clipboard",
    // only in the sandbox VM mode
    "computer_shell", "computer_install", "computer_ads",
];

/// The sandbox VM's own tools: a root shell and installs. The VM is a separate machine, so these run without asking.
pub fn vm_tools() -> Vec<Value> {
    vec![
        json!({"name": "computer_shell", "description": "Run a shell command in the sandbox VM (Debian 13; you are root there with sudo, \
            and may change anything in it; the user's PC is not affected). Returns the exit code and output. The desktop is DISPLAY=:1; \
            start GUI apps with `am2-run <command>`. Use root: true for system changes.",
               "parameters": {"type": "object", "properties": {"command": {"type": "string"},
                   "root": {"type": "boolean", "default": false}, "timeout_s": {"type": "number", "description": "default 300"}},
                   "required": ["command"]}}),
        json!({"name": "computer_install", "description": "Install or remove software in the sandbox VM: a catalog app (chromium, firefox, \
            bitwarden, bitwarden-extension, tailscale, netbird, mullvad, surfshark, vpn (WireGuard and OpenVPN), vscode, libreoffice, thunderbird, keepassxc, gimp, vlc, obsidian, dev-tools, docker), or \
            by kind: apt (Debian package names), flatpak (a Flathub app id), deb (an https URL of a .deb), appimage (an https URL).",
               "parameters": {"type": "object", "properties": {
                   "app": {"type": "string", "description": "a catalog id"},
                   "kind": {"type": "string", "enum": ["apt", "flatpak", "deb", "appimage"]},
                   "target": {"type": "string", "description": "package names, the Flathub id, or the URL"},
                   "remove": {"type": "boolean", "default": false}}}}),
        json!({"name": "computer_ads", "description": "Block or allow ads and trackers in the sandbox VM (every program, and the \
            browsers). For a scraping or checking job that must see a site as it is, block with that site in \"allow\", or allow all.",
               "parameters": {"type": "object", "properties": {"block": {"type": "boolean"},
                   "lists": {"type": "array", "items": {"type": "string", "enum": ["pihole", "adguard"]}, "description": "default: pihole"},
                   "allow": {"type": "array", "items": {"type": "string"}, "description": "domains whose ads stay (while blocking)"}},
                   "required": ["block"]}}),
    ]
}

/// The tools and their parameter schemas for the model (the `strata_computer: true` flag offers them; server_tools.rs).
/// `size` is the screenshot's size (the coordinate system of every x, y); `mac` says the keys are a Mac's (Cmd, Option).
pub fn tools(vision: bool, size: crate::computer::backend::Size, mac: bool) -> Vec<Value> {
    let shot = if vision { " You get a screenshot after each step." } else { "" };
    let (w, h) = (size.width, size.height);
    let px = format!("pixels of the {w}x{h} screenshot");
    let keys = if mac {
        "Press a key or key combination on this Mac (e.g. \"Enter\", \"Cmd+c\", [\"Cmd\", \"Shift\", \"t\"]; Cmd, Option, Ctrl, Shift, Tab, Escape, arrows, F1-F12)."
    } else {
        "Press a key or key combination on the desktop (e.g. \"Enter\", or [\"Ctrl\", \"a\"] / \"Ctrl+a\")."
    };
    let mut t = base_tools(vision, &shot, &px, keys);
    t.extend([
        json!({"name": "computer_double_click", "description": format!("Double-click at x and y ({px}): open a file or select a word."),
               "parameters": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}}),
        json!({"name": "computer_drag", "description": format!("Press at x, y, move to to_x, to_y and let go ({px}): move a file or window, a slider, select text."),
               "parameters": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"},
                                 "to_x": {"type": "number"}, "to_y": {"type": "number"}}, "required": ["x", "y", "to_x", "to_y"]}}),
        json!({"name": "computer_move", "description": format!("Move the pointer to x, y ({px}) without clicking, to open a menu or tooltip that shows on hover."),
               "parameters": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}}),
        json!({"name": "computer_windows", "description": "List the windows on screen: each one's id, title and app. Bring one to the front with computer_focus.",
               "parameters": {"type": "object", "properties": {}}}),
        json!({"name": "computer_focus", "description": "Bring a window to the front by its id from computer_windows.",
               "parameters": {"type": "object", "properties": {"id": {"type": "number"}}, "required": ["id"]}}),
        json!({"name": "computer_clipboard", "description": "With text: put it on the clipboard (then paste it with a key). Without: read what is on the clipboard (asks the owner first).",
               "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}}),
    ]);
    t
}

fn base_tools(_vision: bool, shot: &str, px: &str, keys: &str) -> Vec<Value> {
    let (shot, px) = (shot.to_string(), px.to_string());
    vec![
        json!({"name": "computer_screenshot", "description": format!("Take a screenshot of the desktop.{} What is on screen is shown.", shot),
               "parameters": {"type": "object", "properties": {}}}),
        json!({"name": "computer_click", "description": format!("Click the desktop at x and y ({px}). Both coordinates are required."),
               "parameters": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"},
                                 "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"}},
                              "required": ["x", "y"]}}),
        json!({"name": "computer_type", "description": "Type text at the cursor in the focused element (it appends, it does not replace — click the field first to focus it).",
               "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}),
        json!({"name": "computer_key", "description": keys,
               "parameters": {"type": "object", "properties": {
                   "keys": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
                            "description": "a key or combination, as one string ([\"Ctrl\", \"a\"]) or joined (\"Ctrl+a\")"}}}}),
        json!({"name": "computer_scroll", "description": "Scroll the desktop down/up by dy and right/left by dx (screens).",
               "parameters": {"type": "object", "properties": {"dx": {"type": "number", "description": "positive scrolls right"},
                                 "dy": {"type": "number", "description": "positive scrolls down"}}}}),
        json!({"name": "computer_open_app", "description": "Open an application (only apps in the allowed list). Its windows appear for the next step.",
               "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}),
        json!({"name": "computer_wait", "description": "Wait a number of seconds (default 1), e.g. for an app to open.",
               "parameters": {"type": "object", "properties": {"seconds": {"type": "number", "description": "default 1"}}}}),
    ]
}

use serde_json::{json, Value};
