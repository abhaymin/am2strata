//! Site tests: a list of browser steps and checks, run the same way every time with no model (the Browser tab's Tests,
//! `POST /api/browser/test`, `am2strata-run browser test <file>`; JUnit XML for CI). Saved ones live in
//! `<data>/browser-tests/<name>.json`.
//!
//! ```json
//! {"name": "shop", "target": "server", "base": "https://shop.example",
//!  "steps": [{"open": "/"}, {"expect": [{"status": 200}, {"text": "Welcome"}, {"console_errors": 0}]},
//!            {"fill": [{"field": "Search", "value": "lamp"}]}, {"key": "Enter"}, {"wait": "results"},
//!            {"extract": {"kind": "items", "selector": ".product", "fields": {"name": "h3", "price": ".price"}},
//!             "rows": {"min": 1}},
//!            {"click": "Next"}, {"expect": [{"url": "page=2"}]}]}
//! ```
//!
//! Each step is `{tool, args}` or one short form: open, click (text, or {ref|x,y|text}), type ({text, selector?, enter?}),
//! fill, select ({selector, value}), wait (text), key, mouse, scroll (screens), back, forward, screenshot, extract (with
//! `rows: {min?, max?}`), expect (browser_check's checks). A step failing stops the suite unless `"continue": true`
//! (on the step or the suite). Every run uses a fresh tab, closed at the end.

use crate::browser::Browsers;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn dir(data: &Path) -> PathBuf { data.join("browser-tests") }

pub fn list(data: &Path) -> Vec<Value> {
    let mut out: Vec<Value> = std::fs::read_dir(dir(data)).into_iter().flatten().flatten().filter_map(|e| {
        let n = e.file_name().to_string_lossy().strip_suffix(".json")?.to_string();
        let v: Value = serde_json::from_slice(&std::fs::read(e.path()).ok()?).ok()?;
        Some(json!({"name": n, "steps": v["steps"].as_array().map(|a| a.len()).unwrap_or(0), "base": v["base"], "target": v["target"],
                    "last": v["_last"]}))
    }).collect();
    out.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    out
}

pub fn load(data: &Path, name: &str) -> Result<Value, String> {
    crate::datasets::check_name(name)?;
    let b = std::fs::read(dir(data).join(format!("{name}.json"))).map_err(|_| format!("there is no saved test {name}"))?;
    serde_json::from_slice(&b).map_err(|e| format!("test {name} is not valid JSON: {e}"))
}

pub fn store(data: &Path, name: &str, suite: &Value) -> Result<(), String> {
    crate::datasets::check_name(name)?;
    let steps = suite["steps"].as_array().ok_or("a test needs steps: […]")?;
    for (i, s) in steps.iter().enumerate() { to_call(s, suite["base"].as_str()).map_err(|e| format!("step {}: {e}", i + 1))?; }
    std::fs::create_dir_all(dir(data)).map_err(|e| e.to_string())?;
    let mut v = suite.clone();
    v["name"] = json!(name);
    std::fs::write(dir(data).join(format!("{name}.json")), serde_json::to_vec_pretty(&v).unwrap()).map_err(|e| e.to_string())
}

pub fn remove(data: &Path, name: &str) -> Result<bool, String> {
    crate::datasets::check_name(name)?;
    let p = dir(data).join(format!("{name}.json"));
    let had = p.exists();
    let _ = std::fs::remove_file(p);
    Ok(had)
}

/// A step as a tool call (name, args), or "expect" with its checks.
pub fn to_call(step: &Value, base: Option<&str>) -> Result<(String, Value), String> {
    let o = step.as_object().ok_or("a step is an object")?;
    if let Some(t) = o.get("tool").and_then(Value::as_str) {
        let name = if t.starts_with("browser_") { t.to_string() } else { format!("browser_{t}") };
        if !crate::browser::TOOL_NAMES.contains(&name.as_str()) { return Err(format!("there is no tool {name}")); }
        return Ok((name, o.get("args").cloned().unwrap_or(json!({}))));
    }
    let get = |k: &str| o.get(k);
    let s = |v: &Value| v.as_str().map(str::to_string);
    if let Some(v) = get("open") {
        let url = s(v).ok_or("open takes an address")?;
        let url = match base { Some(b) if !url.contains("://") => join(b, &url), _ => url };
        return Ok(("browser_open".into(), json!({"url": url})));
    }
    if let Some(v) = get("expect") { return Ok(("expect".into(), json!({"checks": v}))); }
    if let Some(v) = get("click") { return Ok(("browser_click".into(), if v.is_object() { v.clone() } else { json!({"text": s(v).ok_or("click takes text or {ref|x,y|text}")?}) })); }
    if let Some(v) = get("type") { return Ok(("browser_type".into(), if v.is_object() { v.clone() } else { json!({"text": s(v).ok_or("type takes text")?}) })); }
    if let Some(v) = get("fill") { return Ok(("browser_fill".into(), if v.is_array() { json!({"fields": v}) } else { v.clone() })); }
    if let Some(v) = get("select") { return Ok(("browser_select".into(), v.clone())); }
    if let Some(v) = get("wait") { return Ok(("browser_wait".into(), json!({"for": s(v).ok_or("wait takes the text to wait for")?}))); }
    if let Some(v) = get("key") { return Ok(("browser_key".into(), if v.is_object() { v.clone() } else { json!({"keys": s(v).ok_or("key takes keys")?}) })); }
    if let Some(v) = get("mouse") { return Ok(("browser_mouse".into(), v.clone())); }
    if let Some(v) = get("scroll") { return Ok(("browser_scroll".into(), json!({"screens": v.as_f64().unwrap_or(1.0)}))); }
    if get("back").is_some() { return Ok(("browser_back".into(), json!({}))); }
    if get("forward").is_some() { return Ok(("browser_forward".into(), json!({}))); }
    if let Some(v) = get("screenshot") { return Ok(("browser_screenshot".into(), if v.is_object() { v.clone() } else { json!({"full": v.as_bool().unwrap_or(false)}) })); }
    if let Some(v) = get("extract") { return Ok(("browser_extract".into(), v.clone())); }
    if let Some(v) = get("elements") { return Ok(("browser_elements".into(), if v.is_object() { v.clone() } else { json!({}) })); }
    Err(format!("unknown step {step}: use tool+args, or open, click, type, fill, select, wait, key, mouse, scroll, back, forward, screenshot, extract, expect"))
}

fn join(base: &str, path: &str) -> String {
    reqwest::Url::parse(base).and_then(|b| b.join(path)).map(|u| u.to_string()).unwrap_or_else(|_| format!("{}{}", base.trim_end_matches('/'), path))
}

/// Run a suite; the report has every step's result, the checks, and pass/fail.
pub async fn run(b: &Browsers, suite: &Value) -> Value {
    let t0 = Instant::now();
    let name = suite["name"].as_str().unwrap_or("test").to_string();
    let base = suite["base"].as_str();
    let go_on = suite["continue"].as_bool().unwrap_or(false);
    let sess = format!("test:{}", uuid::Uuid::new_v4().simple());
    let mut steps = Vec::new();
    let (mut passed, mut failed, mut checks_failed) = (0usize, 0usize, 0usize);
    let empty = vec![];
    let list = suite["steps"].as_array().unwrap_or(&empty);
    let mut stopped = false;
    for (i, st) in list.iter().enumerate() {
        let s0 = Instant::now();
        let mut rec = json!({"n": i + 1, "step": st});
        let ok = match to_call(st, base) {
            Err(e) => { rec["error"] = json!(e); false }
            Ok((tool, args)) if tool == "expect" => {
                match b.current(&sess).await {
                    None => { rec["error"] = json!("nothing is open yet: open a page before expect"); false }
                    Some(page) => {
                        let checks = args["checks"].as_array().cloned().unwrap_or_default();
                        let r = crate::browser::more::run_checks(b, &page, &checks).await;
                        checks_failed += r.failed;
                        rec["checks"] = json!(r.results);
                        rec["text"] = json!(r.lines.join("\n"));
                        r.failed == 0
                    }
                }
            }
            Ok((tool, args)) => match b.call(&sess, &tool, &args).await {
                Ok(out) => {
                    rec["text"] = json!(out.text.chars().take(1500).collect::<String>());
                    let mut ok = true;
                    if tool == "browser_extract" {
                        let n = count_rows(&out.text);
                        rec["rows"] = json!(n);
                        if let Some(min) = st["rows"]["min"].as_u64() { if (n as u64) < min { ok = false; rec["error"] = json!(format!("{n} row(s), wanted at least {min}")); } }
                        if let Some(max) = st["rows"]["max"].as_u64() { if (n as u64) > max { ok = false; rec["error"] = json!(format!("{n} row(s), wanted at most {max}")); } }
                    }
                    if tool == "browser_screenshot" { rec["image"] = json!(out.screenshot.map(|i| format!("data:image/jpeg;base64,{i}"))); }
                    ok
                }
                Err(e) => { rec["error"] = json!(e); false }
            },
        };
        rec["ok"] = json!(ok);
        rec["ms"] = json!(s0.elapsed().as_millis() as u64);
        if ok { passed += 1 } else {
            failed += 1;
            if let Some(page) = b.current(&sess).await {
                rec["url"] = json!(page.url().await.ok().flatten());
                let shot = page.screenshot(chromiumoxide::page::ScreenshotParams::builder()
                    .format(chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat::Jpeg).quality(50).build()).await.ok();
                use base64::Engine as _;
                rec["image"] = json!(shot.map(|b| format!("data:image/jpeg;base64,{}", base64::engine::general_purpose::STANDARD.encode(b))));
            }
        }
        steps.push(rec);
        if !ok && !go_on && st["continue"].as_bool() != Some(true) {
            stopped = true;
            break;
        }
    }
    b.close_session(&sess).await;
    let skipped = list.len() - steps.len();
    tracing::info!(%name, passed, failed, skipped, "browser test run");
    json!({"name": name, "ok": failed == 0 && !list.is_empty(), "passed": passed, "failed": failed, "skipped": skipped,
           "checks_failed": checks_failed, "stopped": stopped, "ms": t0.elapsed().as_millis() as u64, "steps": steps})
}

/// "N row(s) from …" at the start of browser_extract's answer.
fn count_rows(text: &str) -> usize {
    text.split_whitespace().next().and_then(|n| n.parse().ok()).unwrap_or(0)
}

fn xml(s: &str) -> String {
    s.chars().filter(|c| !c.is_control() || *c == '\n' || *c == '\t').collect::<String>()
        .replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// The report as JUnit XML: one testcase per step.
pub fn junit(r: &Value) -> String {
    let name = xml(r["name"].as_str().unwrap_or("test"));
    let steps = r["steps"].as_array().cloned().unwrap_or_default();
    let total = steps.len() + r["skipped"].as_u64().unwrap_or(0) as usize;
    let mut out = format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<testsuite name=\"{name}\" tests=\"{total}\" failures=\"{}\" skipped=\"{}\" time=\"{:.3}\">\n",
                          r["failed"], r["skipped"], r["ms"].as_f64().unwrap_or(0.0) / 1000.0);
    for s in &steps {
        let label = xml(&format!("{} {}", s["n"], s["step"]).chars().take(120).collect::<String>());
        out.push_str(&format!("  <testcase classname=\"{name}\" name=\"{label}\" time=\"{:.3}\">", s["ms"].as_f64().unwrap_or(0.0) / 1000.0));
        if s["ok"] != true {
            let msg = s["error"].as_str().map(str::to_string).or_else(|| s["text"].as_str().map(str::to_string)).unwrap_or_default();
            out.push_str(&format!("<failure message=\"{}\">{}</failure>", xml(&msg.chars().take(300).collect::<String>()), xml(&msg)));
        }
        out.push_str("</testcase>\n");
    }
    for i in 0..r["skipped"].as_u64().unwrap_or(0) {
        out.push_str(&format!("  <testcase classname=\"{name}\" name=\"{} (not run)\"><skipped/></testcase>\n", steps.len() as u64 + i + 1));
    }
    out.push_str("</testsuite>\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_steps_become_tool_calls() {
        let b = Some("https://shop.example/a/");
        assert_eq!(to_call(&json!({"open": "/x?y=1"}), b).unwrap(), ("browser_open".into(), json!({"url": "https://shop.example/x?y=1"})));
        assert_eq!(to_call(&json!({"open": "https://other.example"}), b).unwrap().1["url"], "https://other.example");
        assert_eq!(to_call(&json!({"click": "Next"}), None).unwrap(), ("browser_click".into(), json!({"text": "Next"})));
        assert_eq!(to_call(&json!({"click": {"ref": "e3"}}), None).unwrap().1, json!({"ref": "e3"}));
        assert_eq!(to_call(&json!({"fill": [{"field": "a", "value": "b"}]}), None).unwrap().1, json!({"fields": [{"field": "a", "value": "b"}]}));
        assert_eq!(to_call(&json!({"key": "Enter"}), None).unwrap(), ("browser_key".into(), json!({"keys": "Enter"})));
        assert_eq!(to_call(&json!({"tool": "read"}), None).unwrap().0, "browser_read");
        assert_eq!(to_call(&json!({"expect": [{"text": "x"}]}), None).unwrap(), ("expect".into(), json!({"checks": [{"text": "x"}]})));
        assert!(to_call(&json!({"tool": "browser_nope"}), None).is_err());
        assert!(to_call(&json!({"dance": 1}), None).is_err());
    }

    #[test]
    fn suites_are_checked_when_saved_and_listed() {
        let d = std::env::temp_dir().join(format!("am2-bt-{}", uuid::Uuid::new_v4().simple()));
        assert!(store(&d, "bad", &json!({"steps": [{"dance": 1}]})).unwrap_err().starts_with("step 1"));
        assert!(store(&d, "../x", &json!({"steps": []})).is_err());
        store(&d, "home", &json!({"base": "https://a.example", "steps": [{"open": "/"}, {"expect": [{"status": 200}]}]})).unwrap();
        assert_eq!(list(&d)[0]["steps"], 2);
        assert_eq!(load(&d, "home").unwrap()["name"], "home");
        assert!(remove(&d, "home").unwrap());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn junit_escapes_and_counts() {
        let r = json!({"name": "a<b", "failed": 1, "skipped": 1, "ms": 1500, "steps": [
            {"n": 1, "step": {"open": "/"}, "ok": true, "ms": 10},
            {"n": 2, "step": {"expect": [{"text": "\"x\""}]}, "ok": false, "ms": 5, "error": "not <found>"}]});
        let x = junit(&r);
        assert!(x.contains("name=\"a&lt;b\" tests=\"3\" failures=\"1\" skipped=\"1\""));
        assert!(x.contains("<failure message=\"not &lt;found&gt;\">"));
        assert!(x.contains("<skipped/>"));
    }

    #[test]
    fn extract_answers_start_with_their_row_count() {
        assert_eq!(count_rows("12 row(s) from 1 page(s) of x"), 12);
        assert_eq!(count_rows("nothing"), 0);
    }
}
