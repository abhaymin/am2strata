//! Computer use: full desktop + browser control switched on in Settings, with server-enforced rule presets (see the plan).
//!
//! - `rules` — the pure rules engine (§2 of the plan): classify a step into an Intent and decide Allowed / Approval /
//!   Refused(rule). Zero OS dependency, unit-tested. Everything here depends on it.
//! - `settings` — the run-config parsing for computer use / browser use / rule presets (off by default), round-trips to
//!   the config value that produced it. Feeds the runtime and the Settings › Rules page.
//! - `backend` — the desktop a step drives, hidden behind one trait (`Backend`) so neither the tool layer nor the tests
//!   see which OS or display it is on (Linux now; macOS/Windows stubs). See §3 of the plan.
//! - `tools` — the desktop computer tools: their names (`TOOL_NAMES`, re-exported for run.rs like the other tools) and
//!   parameter schemas, offered only when `"computer": {"use": true}`.
//! - `step` — turns a raw tool call into a validated `ComputerStep` (the normalized shape the backend runs), pure and
//!   unit-tested.

pub mod backend;
pub mod rules;
pub mod settings;
pub mod step;
pub mod tools;

/// The desktop computer-use tool names (re-exported so run.rs dispatches them like `browser`/`web_tools`, etc.).
pub use tools::TOOL_NAMES;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_lists_the_seven_desktop_tools() {
        assert_eq!(TOOL_NAMES.len(), 7);
        for name in ["computer_screenshot", "computer_click", "computer_type", "computer_key",
                     "computer_scroll", "computer_open_app", "computer_wait"] {
            assert!(TOOL_NAMES.contains(&name), "{name} is a computer-use tool");
        }
    }
}
