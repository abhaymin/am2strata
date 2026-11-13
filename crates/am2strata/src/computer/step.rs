//! Turning a raw computer-use tool call (`computer_click`, its arguments) into a validated `ComputerStep`: the normalized
//! shape the tool layer runs through a backend. Pure (no OS, no network), so it is unit-tested directly — this is where
//! an unknown argument is refused up front with a message the model can act on, like browser.rs refuses a click without
//! x and y.

use crate::computer::backend::{Button, Key};
use serde_json::Value;

/// A normalized computer-use step, ready to run through a `Backend` (or a mock in tests). Only the fields for its `tool`
/// are meaningful (`x`/`y` for a click, `keys` for a key, etc.).
#[derive(Clone, Debug, PartialEq)]
pub struct ComputerStep {
    pub tool: String,
    pub x: Option<i32>,
    pub y: Option<i32>,
    pub button: Button,
    pub text: Option<String>,
    pub keys: Vec<Key>,
    pub dx: Option<i32>,
    pub dy: Option<i32>,
    pub name: Option<String>,
    /// seconds to wait (computer_wait); default 1
    pub seconds: u32,
}

impl ComputerStep {
    fn new(tool: &str) -> ComputerStep {
        ComputerStep { tool: tool.to_string(), x: None, y: None, button: Button::Left, text: None, keys: Vec::new(),
                       dx: None, dy: None, name: None, seconds: 1 }
    }
}

/// The tools this classifier recognizes (matches `computer::TOOL_NAMES`); anything else is not a computer step.
const TOOLS: [&str; 7] = [
    "computer_screenshot", "computer_click", "computer_type", "computer_key",
    "computer_scroll", "computer_open_app", "computer_wait",
];

/// A tool call into a validated step, or an error naming the missing/invalid argument. Unknown tools and arguments that
/// do not match any tool are both errors, so run.rs can tell "not ours" apart from "ours but wrong".
pub fn classify_step(tool: &str, args: &Value) -> Result<ComputerStep, String> {
    if !TOOLS.contains(&tool) {
        return Err(format!("'{tool}' is not a computer-use tool"));
    }
    let s = ComputerStep::new(tool);
    match tool {
        "computer_screenshot" => Ok(s),
        "computer_wait" => {
            let secs = number(args.get("seconds")).unwrap_or(1.0).clamp(0.0, 3600.0);
            Ok(ComputerStep { seconds: secs as u32, ..s })
        }
        "computer_click" => {
            match (number(args.get("x")), number(args.get("y"))) {
                (Some(x), Some(y)) => Ok(ComputerStep { x: Some(x as i32), y: Some(y as i32), button: parse_button(args.get("button")), ..s }),
                _ => return Err("give x and y, or a window title to click".into()),
            }
        }
        "computer_type" => {
            let text = args.get("text").and_then(Value::as_str).ok_or("give the text to type")?;
            Ok(ComputerStep { text: Some(text.to_string()), ..s })
        }
        "computer_key" => {
            let keys = keys_from(args.get("keys")).ok_or("press a key, e.g. \"Enter\" or [\"Ctrl\", \"a\"]")?;
            Ok(ComputerStep { keys, ..s })
        }
        "computer_scroll" => Ok(ComputerStep { dx: number(args.get("dx")).map(|v| v as i32), dy: number(args.get("dy")).map(|v| v as i32), ..s }),
        "computer_open_app" => {
            let name = args.get("name").and_then(Value::as_str).ok_or("give the application name")?;
            Ok(ComputerStep { name: Some(name.to_string()), ..s })
        }
        _ => unreachable!("classify_step only accepts tools in TOOLS"),
    }
}

/// A number argument as an f64, or None. NaN/inf are rejected so a click's coordinates are real.
fn number(v: Option<&Value>) -> Option<f64> {
    match v {
        Some(Value::Number(n)) => n.as_f64().filter(|f| f.is_finite()),
        _ => None,
    }
}

/// The click button from its schema name (`"left"` etc.), defaulting to Left.
fn parse_button(v: Option<&Value>) -> Button {
    match v.and_then(Value::as_str).map(str::to_ascii_lowercase).unwrap_or_default().as_str() {
        "right" => Button::Right,
        "middle" => Button::Middle,
        _ => Button::Left,
    }
}

/// `computer_key`'s keys: an array of strings, or a single string split on '+'. None when there are none.
fn keys_from(v: Option<&Value>) -> Option<Vec<Key>> {
    let mut out: Vec<Key> = match v {
        Some(Value::Array(a)) => a.iter().filter_map(|k| k.as_str().map(|s| Key::new(s))).collect(),
        Some(Value::String(s)) if !s.trim().is_empty() => s.split('+').map(|s| Key::new(s.trim())).collect(),
        _ => return None,
    };
    out.retain(|k| !k.as_str().is_empty());
    if out.is_empty() { None } else { Some(out) }
}

// ------------------------------------------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn screenshot_and_wait_neither_argument_nor_restriction() {
        assert_eq!(classify_step("computer_screenshot", &json!({})).unwrap().tool, "computer_screenshot");
        assert_eq!(classify_step("computer_wait", &json!({})).unwrap().seconds, 1);
        assert_eq!(classify_step("computer_wait", &json!({"seconds": -5})).unwrap().seconds, 0);
        assert_eq!(classify_step("computer_wait", &json!({"seconds": 9999})).unwrap().seconds, 3600);
    }

    #[test]
    fn click_requires_coordinates() {
        assert_eq!(classify_step("computer_click", &json!({"x": 10.0, "y": 20.0})).unwrap(),
                   ComputerStep { tool: "computer_click".into(), x: Some(10), y: Some(20), button: Button::Left, ..ComputerStep::new("computer_click") });
        assert_eq!(classify_step("computer_click", &json!({"x": 5.5})).unwrap_err(), "give x and y, or a window title to click");
        // an explicit right button is kept
        assert_eq!(classify_step("computer_click", &json!({"x": 1, "y": 2, "button": "right"})).unwrap().button, Button::Right);
    }

    #[test]
    fn type_requires_text() {
        assert_eq!(classify_step("computer_type", &json!({"text": "hi"})).unwrap().text, Some("hi".into()));
        assert_eq!(classify_step("computer_type", &json!({})).unwrap_err(), "give the text to type");
    }

    #[test]
    fn key_takes_an_array_or_a_joined_string() {
        assert_eq!(classify_step("computer_key", &json!({"keys": ["Ctrl", "a"]})).unwrap().keys, vec![Key::new("Ctrl"), Key::new("a")]);
        assert_eq!(classify_step("computer_key", &json!({"keys": "Enter"})).unwrap().keys, vec![Key::new("Enter")]);
        assert_eq!(classify_step("computer_key", &json!({"keys": "Alt+F4"})).unwrap().keys, vec![Key::new("Alt"), Key::new("F4")]);
        assert_eq!(classify_step("computer_key", &json!({})).unwrap_err(), "press a key, e.g. \"Enter\" or [\"Ctrl\", \"a\"]");
    }

    #[test]
    fn scroll_and_open_app() {
        assert_eq!(classify_step("computer_scroll", &json!({"dy": 2})).unwrap().dy, Some(2));
        assert_eq!(classify_step("computer_scroll", &json!({})).unwrap(), ComputerStep::new("computer_scroll"));
        assert_eq!(classify_step("computer_open_app", &json!({"name": "firefox"})).unwrap().name, Some("firefox".into()));
        assert_eq!(classify_step("computer_open_app", &json!({})).unwrap_err(), "give the application name");
    }

    #[test]
    fn an_unknown_tool_is_not_a_computer_step() {
        assert_eq!(classify_step("browser_click", &json!({})).unwrap_err(), "'browser_click' is not a computer-use tool");
    }
}
