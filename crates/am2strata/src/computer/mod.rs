//! Computer use: full desktop + browser control switched on in Settings, with server-enforced rule presets (see the plan).
//!
//! - `rules` — the pure rules engine (§2 of the plan): classify a step into an Intent and decide Allowed / Approval /
//!   Refused(rule). Zero OS dependency, unit-tested. Everything here depends on it.
//! - `settings` — the run-config parsing for computer use / browser use / rule presets (off by default), round-trips to
//!   the config value that produced it. Feeds the runtime and the Settings › Rules page.

pub mod rules;
pub mod settings;
