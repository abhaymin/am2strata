//! The computer-use desktop tools, offered only when `"computer": {"use": true}` (`computer/mod.rs` / server_tools.rs). One
//! name list (for run.rs's dispatch) and one JSON schema per tool, with descriptions that tell the model what it drives —
//! this PC's desktop, input on a display via xdotool/XTest — and that it sees a screenshot after each step.
//!
//! Each result carries a screenshot (when the model reads pictures), else the window list and the focused window title;
//! `computer_open_app` opens only an app in the allow list (the rules engine refuses the rest before the step runs).

/// The desktop computer tools, matched against a request's tool call in run.rs (`crate::computer::TOOL_NAMES`).
pub const TOOL_NAMES: [&str; 7] = [
    "computer_screenshot", "computer_click", "computer_type", "computer_key",
    "computer_scroll", "computer_open_app", "computer_wait",
];

/// The tools and their parameter schemas for the model (the `strata_computer: true` flag offers them; server_tools.rs).
pub fn tools(vision: bool) -> Vec<Value> {
    let shot = if vision { " You get a screenshot after each step." } else { "" };
    vec![
        json!({"name": "computer_screenshot", "description": format!("Take a screenshot of the desktop.{} What is on screen is shown.", shot),
               "parameters": {"type": "object", "properties": {}}}),
        json!({"name": "computer_click", "description": format!("Click the desktop at x and y (pixels of the {W}x{H} screenshot). Both coordinates are required."),
               "parameters": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"},
                                 "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"}},
                              "required": ["x", "y"]}}),
        json!({"name": "computer_type", "description": "Type text at the cursor in the focused element (it appends, it does not replace — click the field first to focus it).",
               "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}),
        json!({"name": "computer_key", "description": "Press a key or key combination on the desktop (e.g. \"Enter\", or [\"Ctrl\", \"a\"] / \"Ctrl+a\").",
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

/// A tool's display width and height, so the model knows the screenshot's coordinate system.
const W: u32 = 1280;
const H: u32 = 800;

use serde_json::{json, Value};
