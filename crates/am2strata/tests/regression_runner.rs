//! `am2strata-run regression` (docs/REGRESSION.md) run against harness servers: every row passes on a healthy
//! server, `--quick` makes at most three model calls and changes nothing, the key is never printed, and a server
//! that is wrong or away makes it exit non-zero. (It is never run against a real server from the tests.)

mod common;

use common::regress::*;
use common::*;
use serde_json::{json, Value};
use std::process::Command;

const ANSWER: &str = "Thinking.\n</think>\n\nok";
/// what a forced tool call's reply continues with (the server writes the call's opening itself)
const CALL_REST: &str = "<parameter=city>\nPune\n</parameter>\n</function>\n</tool_call>";

/// The runner as a user starts it, told everything on its command line; nothing of this PC's own server can reach it
/// (an empty models folder, no AM2STRATA_* in its environment).
fn runner(empty: &TempDir, args: &[&str]) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_am2strata-run"));
    c.arg("regression").args(args).arg("--models-dir").arg(&empty.0).env_remove("AM2STRATA_URL").env_remove("AM2STRATA_API_KEY");
    c
}

fn run(mut c: Command) -> (i32, String, String) {
    let o = c.output().expect("am2strata-run");
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

/// What a caller may see of a server's state: the same before and after a run that changes nothing.
fn state(s: &Server) -> Value {
    let get = |p: &str| s.get(p, true).1;
    let mut settings = get("/settings");
    // (the time it was read is not state)
    if let Some(o) = settings.as_object_mut() { o.remove("now"); }
    json!({"settings": settings, "config": get("/config")["values"], "current": get("/api/models/current")["model"]["id"],
           "memory": get("/api/memory")["facts"], "jobs": get("/v1/workshop/jobs"), "tls": get("/api/tls")["on"],
           "computer": get("/api/computer/status"), "keys": get("/api/keys")["keys"], "mcp": get("/api/mcp/servers")})
}

#[test]
fn quick_passes_on_a_healthy_server_with_three_model_calls_and_no_changes() {
    let s = Server::start(&[client_model(json!({}))], &[ids(ANSWER)], json!({}));
    let empty = TempDir::new("runner");
    let before = state(&s);
    let (code, out, err) = run(runner(&empty, &["--url", &s.base, "--api-key", KEY, "--quick"]));
    assert_eq!(code, 0, "stdout:\n{out}\nstderr:\n{err}");
    assert!(!out.contains("FAIL"), "{out}");
    for row in ["health", "models list", "chat (OpenAI)", "chat, streamed", "messages (Anthropic)", "count_tokens (Anthropic)", "responses: stateless",
                "Gemini: models, countTokens", "metrics", "queue", "HTTPS and HSTS", "Workshop files", "Computer status", "usage", "web app"] {
        let line = out.lines().find(|l| l.contains(row)).unwrap_or_else(|| panic!("no row {row:?} in:\n{out}"));
        assert!(line.trim_start().starts_with("pass"), "{line}");
        assert!(line.contains(" ms"), "a timing in every row: {line}");
    }
    // the VM's status: there on Linux, skipped (said so) elsewhere; never started
    let vm = out.lines().find(|l| l.contains("VM status")).expect("the VM row");
    assert!(vm.trim_start().starts_with(if cfg!(target_os = "linux") { "pass" } else { "skip" }), "{vm}");
    assert!(vm.contains("phase off") || !cfg!(target_os = "linux"), "the VM was not started: {vm}");
    assert!(!out.contains("tool round trip"), "--quick has no tool round trip (it would be more model calls)");
    assert!(out.lines().last().unwrap().contains("0 failed"), "{out}");
    // at most three model calls reached the engine, and nothing on the server changed
    assert_eq!(s.gen_lines().len(), 3, "model calls of --quick: {:?}", s.gen_lines().iter().map(|l| l.chars().take(30).collect::<String>()).collect::<Vec<_>>());
    assert_eq!(state(&s), before, "--quick changed the server's state");
    // the key is never printed
    assert!(!out.contains(KEY) && !err.contains(KEY), "the API key was printed");
}

#[test]
fn full_passes_with_every_api_and_the_tool_round_trip_as_json() {
    let replies = [ids(ANSWER), ids(ANSWER), ids(ANSWER), ids(ANSWER), ids(ANSWER), ids(CALL_REST), ids(ANSWER)];
    let s = Server::start(&[client_model(json!({}))], &replies, json!({}));
    let empty = TempDir::new("runner");
    let before = state(&s);
    // the key from the environment (given to the runner's own process, not to this one)
    let mut c = runner(&empty, &["--url", &s.base, "--full", "--json"]);
    c.env("AM2STRATA_API_KEY", KEY);
    let (code, out, err) = run(c);
    assert_eq!(code, 0, "stdout:\n{out}\nstderr:\n{err}");
    let j: Value = serde_json::from_str(&out).unwrap_or_else(|e| panic!("--json prints JSON only ({e}):\n{out}"));
    assert_eq!((j["failed"].as_u64(), j["mode"].as_str()), (Some(0), Some("full")), "{out}");
    let rows = j["rows"].as_array().unwrap();
    let row = |name: &str| rows.iter().find(|r| r["check"] == name).unwrap_or_else(|| panic!("no row {name:?}:\n{out}"));
    for name in ["responses (OpenAI)", "Gemini: generateContent", "tool round trip", "media, MCP, memory", "chat (OpenAI)", "errors: 404 and 400"] {
        assert_eq!(row(name)["result"], "pass", "{name}: {}", row(name));
        assert!(row(name)["ms"].is_u64());
    }
    assert!(row("tool round trip")["detail"].as_str().unwrap().contains("Pune"), "{}", row("tool round trip"));
    assert_eq!(j["passed"].as_u64().unwrap() + j["skipped"].as_u64().unwrap(), rows.len() as u64);
    assert_eq!(j["model_calls"].as_u64(), Some(s.gen_lines().len() as u64), "it counts its own model calls");
    assert!(s.gen_lines().len() <= 8, "{} model calls", s.gen_lines().len());
    assert_eq!(state(&s), before, "--full changed the server's state");
    assert!(!out.contains(KEY) && !err.contains(KEY), "the API key was printed");
}

#[test]
fn a_wrong_key_a_missing_key_and_a_server_that_is_away_fail_with_a_non_zero_exit() {
    let s = Server::start(&[client_model(json!({}))], &[ids(ANSWER)], json!({}));
    let empty = TempDir::new("runner");
    // a wrong key: the server is up (health passes), everything behind the key fails, and the wrong key is not echoed
    let wrong = "wrong-key-0123456789-regress";
    let (code, out, err) = run(runner(&empty, &["--url", &s.base, "--api-key", wrong]));
    assert_ne!(code, 0, "{out}");
    assert!(out.lines().any(|l| l.contains("health") && l.trim_start().starts_with("pass")), "{out}");
    assert!(out.lines().any(|l| l.contains("models list") && l.trim_start().starts_with("FAIL") && l.contains("401")), "{out}");
    assert!(!out.contains(wrong) && !err.contains(wrong), "the key was printed:\n{out}");
    assert_eq!(s.gen_lines().len(), 0, "no model call got through");
    // no key at all (none given, none in the environment, none in a run config)
    let (code, out, _) = run(runner(&empty, &["--url", &s.base, "--json"]));
    assert_ne!(code, 0, "{out}");
    let j: Value = serde_json::from_str(&out).expect("JSON");
    assert!(j["failed"].as_u64().unwrap() > 5, "{out}");
    assert!(j["rows"].as_array().unwrap().iter().any(|r| r["check"] == "the API key is needed" && r["result"] == "skip"), "{out}");
    // a client profile's key is not the owner's: the model APIs pass, the owner's status rows fail
    let (code, out, _) = run(runner(&empty, &["--url", &s.base, "--api-key", CLIENT_KEY]));
    assert_ne!(code, 0, "{out}");
    assert!(out.lines().any(|l| l.contains("chat (OpenAI)") && l.trim_start().starts_with("pass")), "{out}");
    assert!(out.lines().any(|l| l.contains("Workshop files") && l.trim_start().starts_with("FAIL")), "{out}");
    assert!(!out.contains(CLIENT_KEY));
    // nothing listens there: one failed row, not twenty
    let (code, out, _) = run(runner(&empty, &["--url", &format!("http://127.0.0.1:{}", free_port()), "--api-key", KEY]));
    assert_ne!(code, 0, "{out}");
    assert_eq!(out.lines().filter(|l| l.trim_start().starts_with("FAIL")).count(), 1, "{out}");
    assert!(!out.contains(KEY));
}

#[test]
fn https_and_hsts_are_checked_when_tls_is_on() {
    let port = free_port();
    let s = Server::start(&[client_model(json!({"tls": {"https_port": port}}))], &[ids(ANSWER)], json!({}));
    https_client(&s.dir, port);
    let empty = TempDir::new("runner");
    let (code, out, err) = run(runner(&empty, &["--url", &s.base, "--api-key", KEY]));
    assert_eq!(code, 0, "stdout:\n{out}\nstderr:\n{err}");
    let line = out.lines().find(|l| l.contains("HTTPS and HSTS")).expect("the HTTPS row");
    assert!(line.trim_start().starts_with("pass") && line.contains("self-signed") && line.contains(&format!("port {port}")) && line.contains("max-age=0"), "{line}");
    // and against the HTTPS address itself
    let (code, out, err) = run(runner(&empty, &["--url", &format!("https://localhost:{port}"), "--api-key", KEY]));
    assert_eq!(code, 0, "stdout:\n{out}\nstderr:\n{err}");
    assert!(out.lines().any(|l| l.contains("chat (OpenAI)") && l.trim_start().starts_with("pass")), "{out}");
}
