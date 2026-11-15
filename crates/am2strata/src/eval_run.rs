//! `am2strata-run eval`: real agent jobs with checkers written and locked before any run (docs/GAPS-PLAN.md A18). A
//! job is a request a person would make (build an app, find a fact, fill a form, collect records, fix a service), sent
//! through the server's own chat API with the server's tools, and a checker that decides pass or fail from what is left
//! afterwards: the answer, a page in a browser, a dataset, or the sandbox VM's state. Nothing the model says about its
//! own success counts.
//!
//! - `evals/jobs/<name>.json`: `{name, about, prompt, request: {extra chat fields}, timeout_s, needs?: "vm", setup?: {vm: script},
//!   check: {kind, …}}`.
//! - **Locked:** `eval lock` writes `evals/LOCK.json`, the SHA-256 of each job's prompt, request, setup and check; `eval run`
//!   refuses a job that changed since (so a checker is never fitted to the results), unless `--unlocked`.
//! - **Checks:** `regex` (`all` / `none` on the answer), `html_browser` (the answer's ```html file opened in the server's
//!   browser and tested with site-test steps, `$PAGE` its address), `browser_test` (site-test steps after the job),
//!   `dataset` (`name`, `min_rows`, `columns`), `vm_shell` (a script in the sandbox VM that exits 0).
//! - **Report:** per job and try: pass, the checker's score, seconds, completion tokens and tool calls; a table, and the
//!   JSON in `evals/results/<time>-<model>.json`. Risky steps a job asks for are allowed by the runner (it is the operator
//!   of a test), and each one is printed.

use serde_json::{json, Value};
use sha2::Digest;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub struct Opts {
    pub dir: PathBuf,
    pub jobs: Vec<String>,
    pub model: Option<String>,
    pub tries: u32,
    pub unlocked: bool,
    pub quiet: bool,
}

pub struct Server {
    pub url: String,
    pub key: Option<String>,
}

impl Server {
    fn client(&self, secs: u64) -> reqwest::blocking::Client {
        reqwest::blocking::Client::builder().timeout(Duration::from_secs(secs)).build().unwrap()
    }
    fn auth(&self, rb: reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder {
        // the server's own page origin, so the browser's approval route and the browser tools are offered
        let rb = rb.header("Origin", &self.url);
        match &self.key { Some(k) => rb.bearer_auth(k), None => rb }
    }
    fn json(&self, method: &str, path: &str, body: Option<&Value>, secs: u64) -> Result<Value, String> {
        let c = self.client(secs);
        let rb = match method { "GET" => c.get(format!("{}{path}", self.url)), "DELETE" => c.delete(format!("{}{path}", self.url)),
                                _ => c.post(format!("{}{path}", self.url)) };
        let rb = match body { Some(b) => rb.json(b), None => rb };
        let r = self.auth(rb).send().map_err(|e| format!("{}: {e}", self.url))?;
        let ok = r.status().is_success();
        let v: Value = r.json().unwrap_or(Value::Null);
        if ok { Ok(v) } else { Err(v["error"]["message"].as_str().unwrap_or("failed").to_string()) }
    }
}

/// What makes a job: the parts the lock covers.
fn locked_part(job: &Value) -> Value {
    json!({"prompt": job["prompt"], "request": job["request"], "setup": job["setup"], "check": job["check"], "needs": job["needs"]})
}

pub fn digest(job: &Value) -> String {
    let mut h = sha2::Sha256::new();
    h.update(serde_json::to_vec(&locked_part(job)).unwrap());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

pub fn load_jobs(dir: &Path) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    let jd = dir.join("jobs");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&jd).map_err(|e| format!("{}: {e}", jd.display()))?
        .flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    files.sort();
    for f in files {
        let v: Value = serde_json::from_slice(&std::fs::read(&f).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", f.display()))?;
        if v["name"].as_str().is_none() || v["prompt"].as_str().is_none() || v["check"]["kind"].as_str().is_none() {
            return Err(format!("{}: a job needs name, prompt and check.kind", f.display()));
        }
        out.push(v);
    }
    Ok(out)
}

pub fn lock(dir: &Path) -> Result<String, String> {
    let jobs = load_jobs(dir)?;
    let m: serde_json::Map<String, Value> = jobs.iter().map(|j| (j["name"].as_str().unwrap().to_string(), json!(digest(j)))).collect();
    std::fs::write(dir.join("LOCK.json"), serde_json::to_vec_pretty(&Value::Object(m)).unwrap()).map_err(|e| e.to_string())?;
    Ok(format!("locked {} job(s) in {}", jobs.len(), dir.join("LOCK.json").display()))
}

/// The jobs whose checker changed since they were locked (or were never locked).
pub fn unlocked(dir: &Path, jobs: &[Value]) -> Vec<String> {
    let lock: Value = std::fs::read(dir.join("LOCK.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!({}));
    jobs.iter().filter(|j| lock[j["name"].as_str().unwrap()].as_str() != Some(digest(j).as_str()))
        .map(|j| j["name"].as_str().unwrap().to_string()).collect()
}

/// The answer's HTML file: the last ```html block, else the whole answer when it is a document.
pub fn html_of(answer: &str) -> Option<String> {
    let mut found = None;
    let mut rest = answer;
    while let Some(i) = rest.find("```html") {
        let after = &rest[i + 7..];
        let body_start = after.find('\n').map(|n| n + 1).unwrap_or(0);
        let end = after[body_start..].find("```").map(|e| body_start + e).unwrap_or(after.len());
        found = Some(after[body_start..end].trim().to_string());
        rest = &after[end.min(after.len())..];
    }
    found.or_else(|| {
        let a = answer.trim();
        (a.to_lowercase().starts_with("<!doctype html") || a.to_lowercase().starts_with("<html")).then(|| a.to_string())
    }).filter(|h| !h.is_empty())
}

/// `regex` check: every `all` pattern matches and no `none` pattern does (case-insensitive); score is the share met.
pub fn regex_check(answer: &str, check: &Value) -> (bool, f64, Vec<String>) {
    let mut notes = Vec::new();
    let (mut met, mut total) = (0usize, 0usize);
    let pat = |p: &str| regex::RegexBuilder::new(p).case_insensitive(true).build();
    for p in check["all"].as_array().into_iter().flatten().filter_map(|p| p.as_str()) {
        total += 1;
        match pat(p) { Ok(r) if r.is_match(answer) => met += 1, Ok(_) => notes.push(format!("missing /{p}/")), Err(e) => notes.push(format!("bad pattern /{p}/: {e}")) }
    }
    for p in check["none"].as_array().into_iter().flatten().filter_map(|p| p.as_str()) {
        total += 1;
        match pat(p) { Ok(r) if !r.is_match(answer) => met += 1, Ok(_) => notes.push(format!("has /{p}/")), Err(e) => notes.push(format!("bad pattern /{p}/: {e}")) }
    }
    let score = if total == 0 { 0.0 } else { met as f64 / total as f64 };
    (total > 0 && met == total, score, notes)
}

/// One chat through the server, streamed: the answer, completion tokens, tool calls, and every approval allowed.
fn chat(srv: &Server, model: &str, job: &Value, quiet: bool) -> Result<(String, u64, u64, Vec<String>), String> {
    let mut body = json!({"model": model, "stream": true, "stream_options": {"include_usage": true}, "max_tokens": job["max_tokens"].as_u64().unwrap_or(16000),
                          "messages": [{"role": "user", "content": job["prompt"]}]});
    if let Some(o) = job["request"].as_object() { for (k, v) in o { body[k] = v.clone(); } }
    let secs = job["timeout_s"].as_u64().unwrap_or(1800);
    let r = srv.auth(srv.client(secs).post(format!("{}/v1/chat/completions", srv.url)).json(&body)).send().map_err(|e| e.to_string())?;
    if !r.status().is_success() {
        let v: Value = r.json().unwrap_or(Value::Null);
        return Err(v["error"]["message"].as_str().unwrap_or("the chat was refused").to_string());
    }
    let (mut answer, mut tokens, mut calls, mut approved) = (String::new(), 0u64, 0u64, Vec::new());
    for line in std::io::BufReader::new(r).lines() {
        let line = line.map_err(|e| format!("the stream broke: {e}"))?;
        let Some(data) = line.strip_prefix("data:").map(str::trim) else { continue };
        if data == "[DONE]" { break; }
        let Ok(c) = serde_json::from_str::<Value>(data) else { continue };
        if let Some(m) = c.get("strata_mcp") {
            match m["event"].as_str() {
                Some("call") => { calls += 1; if !quiet { eprintln!("    · {} {}", m["name"].as_str().unwrap_or(""), m["arguments"].to_string().chars().take(100).collect::<String>()); } }
                Some("approve") => {
                    let what = m["what"].as_str().unwrap_or("").to_string();
                    eprintln!("    ! allowed (test run): {what}");
                    let _ = srv.json("POST", "/api/tools/approve", Some(&json!({"id": m["id"], "ok": true})), 30);
                    approved.push(what);
                }
                _ => {}
            }
            continue;
        }
        for ch in c["choices"].as_array().into_iter().flatten() {
            if let Some(t) = ch["delta"]["content"].as_str() { answer.push_str(t); }
        }
        if let Some(n) = c["usage"]["completion_tokens"].as_u64() { tokens = n; }
    }
    Ok((answer, tokens, calls, approved))
}

/// Decide the job from what is left: (pass, score 0..1, notes).
fn check(srv: &Server, job: &Value, answer: &str) -> (bool, f64, Vec<String>) {
    let c = &job["check"];
    match c["kind"].as_str().unwrap_or("") {
        "regex" => regex_check(answer, c),
        "html_browser" | "browser_test" => {
            let mut steps = c["steps"].clone();
            if c["kind"] == "html_browser" {
                let Some(html) = html_of(answer) else { return (false, 0.0, vec!["the answer has no ```html file".into()]) };
                use base64::Engine as _;
                let page = format!("data:text/html;base64,{}", base64::engine::general_purpose::STANDARD.encode(html.as_bytes()));
                steps = serde_json::from_str(&steps.to_string().replace("$PAGE", &page)).unwrap_or(steps);
            }
            let suite = json!({"name": job["name"], "steps": steps, "continue": true, "base": c["base"]});
            match srv.json("POST", "/api/browser/test?target=server", Some(&json!({"suite": suite})), 600) {
                Ok(r) => {
                    let total = r["steps"].as_array().map(|a| a.len()).unwrap_or(0).max(1) as f64;
                    let notes = r["steps"].as_array().into_iter().flatten().filter(|s| s["ok"] != true)
                        .map(|s| format!("step {} {}: {}", s["n"], s["step"].to_string().chars().take(100).collect::<String>(), s["error"].as_str().or(s["text"].as_str()).unwrap_or("failed").chars().take(160).collect::<String>())).collect();
                    (r["ok"] == true, r["passed"].as_f64().unwrap_or(0.0) / total, notes)
                }
                Err(e) => (false, 0.0, vec![format!("the site test did not run: {e}")]),
            }
        }
        "dataset" => {
            let name = c["name"].as_str().unwrap_or("");
            match srv.json("GET", &format!("/api/datasets/{name}?limit=5"), None, 60) {
                Ok(d) => {
                    let mut notes = Vec::new();
                    let total = d["total"].as_u64().unwrap_or(0);
                    let min = c["min_rows"].as_u64().unwrap_or(1);
                    if total < min { notes.push(format!("{total} rows, wanted at least {min}")); }
                    let cols: Vec<String> = d["columns"].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(str::to_lowercase)).collect();
                    for want in c["columns"].as_array().into_iter().flatten().filter_map(|x| x.as_str()) {
                        if !cols.iter().any(|c| c.contains(&want.to_lowercase())) { notes.push(format!("no column like {want:?} (has {})", cols.join(", "))); }
                    }
                    let parts = 1 + c["columns"].as_array().map(|a| a.len()).unwrap_or(0);
                    (notes.is_empty(), (parts - notes.len().min(parts)) as f64 / parts as f64, notes)
                }
                Err(e) => (false, 0.0, vec![format!("no dataset {name}: {e}")]),
            }
        }
        "vm_shell" => match srv.json("POST", "/api/vm/exec", Some(&json!({"command": c["script"], "root": true, "timeout_s": 120})), 180) {
            Ok(r) if r["code"] == 0 => (true, 1.0, vec![]),
            Ok(r) => (false, 0.0, vec![format!("exit {}: {}", r["code"], r["output"].as_str().unwrap_or("").chars().rev().take(300).collect::<String>().chars().rev().collect::<String>())]),
            Err(e) => (false, 0.0, vec![format!("the VM check did not run: {e}")]),
        },
        other => (false, 0.0, vec![format!("unknown check kind {other:?}")]),
    }
}

/// Run the jobs; returns the report.
pub fn run(srv: &Server, o: &Opts) -> Result<Value, String> {
    let mut jobs = load_jobs(&o.dir)?;
    if !o.jobs.is_empty() { jobs.retain(|j| o.jobs.iter().any(|n| j["name"] == n.as_str())); }
    if jobs.is_empty() { return Err("no jobs to run".into()); }
    let changed = unlocked(&o.dir, &jobs);
    if !changed.is_empty() && !o.unlocked {
        return Err(format!("job(s) changed since they were locked: {} (`eval lock` after review, or --unlocked for a trial run that does not count)", changed.join(", ")));
    }
    let model = match &o.model { Some(m) => m.clone(), None => {
        let v = srv.json("GET", "/v1/models", None, 30)?;
        v["data"][0]["id"].as_str().ok_or("the server lists no model")?.to_string()
    } };
    let vm_ok = srv.json("GET", "/api/vm", None, 30).is_ok_and(|v| v["available"] != false);
    let mut rows = Vec::new();
    for j in &jobs {
        let name = j["name"].as_str().unwrap();
        if j["needs"] == "vm" && !vm_ok { eprintln!("{name}: skipped (needs the sandbox VM)"); rows.push(json!({"job": name, "skipped": "needs the sandbox VM"})); continue; }
        for t in 1..=o.tries {
            eprintln!("{name} (try {t}/{}) on {model}", o.tries);
            if let Some(s) = j["setup"]["vm"].as_str() {
                if let Err(e) = srv.json("POST", "/api/vm/exec", Some(&json!({"command": s, "root": true, "timeout_s": 900})), 960) {
                    rows.push(json!({"job": name, "try": t, "pass": false, "score": 0.0, "notes": [format!("setup failed: {e}")]})); continue;
                }
            }
            if let Some(ds) = j["check"]["kind"].as_str().filter(|k| *k == "dataset").and(j["check"]["name"].as_str()) {
                let _ = srv.json("DELETE", &format!("/api/datasets/{ds}"), None, 30); // each try starts clean
            }
            let t0 = Instant::now();
            let (answer, tokens, calls, approved, err) = match chat(srv, &model, j, o.quiet) {
                Ok((a, tk, c, ap)) => (a, tk, c, ap, None),
                Err(e) => (String::new(), 0, 0, vec![], Some(e)),
            };
            let secs = t0.elapsed().as_secs_f64();
            let (pass, score, mut notes) = if let Some(e) = &err { (false, 0.0, vec![format!("the chat failed: {e}")]) } else { check(srv, j, &answer) };
            if !approved.is_empty() { notes.push(format!("{} risky step(s) allowed", approved.len())); }
            eprintln!("  {} score {:.2}, {:.0} s, {} tokens, {} tool calls{}", if pass { "PASS" } else { "FAIL" }, score, secs, tokens, calls,
                      if notes.is_empty() { String::new() } else { format!("\n    {}", notes.join("\n    ")) });
            rows.push(json!({"job": name, "try": t, "pass": pass, "score": score, "seconds": (secs * 10.0).round() / 10.0, "tokens": tokens,
                             "tool_calls": calls, "notes": notes, "answer_tail": answer.chars().rev().take(400).collect::<String>().chars().rev().collect::<String>()}));
        }
    }
    let ran: Vec<&Value> = rows.iter().filter(|r| r.get("skipped").is_none()).collect();
    let passed = ran.iter().filter(|r| r["pass"] == true).count();
    let report = json!({"model": model, "server": srv.url, "when": chrono_now(), "tries": o.tries, "unlocked": o.unlocked && !changed.is_empty(),
                        "passed": passed, "ran": ran.len(), "rows": rows,
                        "locks": jobs.iter().map(|j| (j["name"].as_str().unwrap().to_string(), json!(digest(j)))).collect::<serde_json::Map<_, _>>()});
    let out = o.dir.join("results");
    let _ = std::fs::create_dir_all(&out);
    let file = out.join(format!("{}-{}.json", report["when"].as_str().unwrap().replace([':', ' '], "-"), model.replace(['/', ':'], "_")));
    let _ = std::fs::write(&file, serde_json::to_vec_pretty(&report).unwrap());
    eprintln!("report: {}", file.display());
    Ok(report)
}

/// Prove the checkers: each job with a known-good answer (`evals/known-good/<name>.md`) must pass on it. A checker that
/// fails on correct work, or a browser that cannot run it (WebGL, a blocked site), is found here before any model is judged.
pub fn selftest(srv: &Server, dir: &Path) -> Result<bool, String> {
    let mut all = true;
    for j in load_jobs(dir)? {
        let name = j["name"].as_str().unwrap();
        let Ok(answer) = std::fs::read_to_string(dir.join("known-good").join(format!("{name}.md"))) else {
            println!("{name:<22} no known-good answer (its checker reads what a job leaves behind)");
            continue;
        };
        let (pass, score, notes) = check(srv, &j, &answer);
        all &= pass;
        println!("{name:<22} {} score {score:.2}{}", if pass { "PASS" } else { "FAIL" },
                 if notes.is_empty() { String::new() } else { format!("\n    {}", notes.join("\n    ")) });
    }
    Ok(all)
}

/// A table of the report: per job, passes over tries, mean score and mean seconds.
pub fn table(report: &Value) -> String {
    let mut by: Vec<(String, Vec<&Value>)> = Vec::new();
    for r in report["rows"].as_array().into_iter().flatten() {
        let n = r["job"].as_str().unwrap_or("").to_string();
        match by.iter_mut().find(|(k, _)| *k == n) { Some((_, v)) => v.push(r), None => by.push((n, vec![r])) }
    }
    let mut t = format!("{:<22} {:>7} {:>6} {:>8} {:>8}\n", "job", "passed", "score", "seconds", "tokens");
    for (n, rs) in by {
        if let Some(s) = rs[0]["skipped"].as_str() { t.push_str(&format!("{n:<22} skipped: {s}\n")); continue; }
        let k = rs.len() as f64;
        let mean = |f: &str| rs.iter().map(|r| r[f].as_f64().unwrap_or(0.0)).sum::<f64>() / k;
        t.push_str(&format!("{n:<22} {:>3}/{:<3} {:>6.2} {:>8.0} {:>8.0}\n", rs.iter().filter(|r| r["pass"] == true).count(), rs.len(),
                            mean("score"), mean("seconds"), mean("tokens")));
    }
    t.push_str(&format!("model {}: {} of {} tries passed{}", report["model"].as_str().unwrap_or(""), report["passed"], report["ran"],
                        if report["unlocked"] == true { " (UNLOCKED jobs: a trial, not a result)" } else { "" }));
    t
}

fn chrono_now() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (d, rem) = (s / 86_400, s % 86_400);
    // civil date from days since 1970 (Howard Hinnant's algorithm)
    let z = d as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z", rem / 3600, (rem % 3600) / 60, rem % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_html_file_is_the_last_html_block() {
        let a = "Here:\n```html\n<p>old</p>\n```\nbetter:\n```html\n<!doctype html><p>new</p>\n```\nDone.";
        assert_eq!(html_of(a).as_deref(), Some("<!doctype html><p>new</p>"));
        assert_eq!(html_of("<!DOCTYPE html><title>x</title>").as_deref(), Some("<!DOCTYPE html><title>x</title>"));
        assert_eq!(html_of("no file here"), None);
        assert_eq!(html_of("```html\n```"), None);
    }

    #[test]
    fn regex_checks_count_what_was_met() {
        let c = json!({"all": ["41[, ]?913[, ]?860", "jakarta"], "none": ["i could not"]});
        assert_eq!(regex_check("Jakarta: 41,913,860.", &c).0, true);
        let (ok, score, notes) = regex_check("Jakarta, I could not find it", &c);
        assert!(!ok && (score - 1.0 / 3.0).abs() < 1e-9 && notes.len() == 2, "{notes:?}");
        assert!(!regex_check("anything", &json!({})).0, "a check with nothing to check never passes");
    }

    #[test]
    fn a_changed_checker_is_caught_by_the_lock() {
        let d = std::env::temp_dir().join(format!("am2-eval-{}", std::process::id()));
        std::fs::create_dir_all(d.join("jobs")).unwrap();
        let job = json!({"name": "j", "prompt": "p", "check": {"kind": "regex", "all": ["x"]}});
        std::fs::write(d.join("jobs/j.json"), job.to_string()).unwrap();
        assert_eq!(unlocked(&d, &load_jobs(&d).unwrap()), vec!["j"], "never locked");
        lock(&d).unwrap();
        assert!(unlocked(&d, &load_jobs(&d).unwrap()).is_empty());
        let mut changed = job.clone();
        changed["check"]["all"] = json!(["y"]);
        std::fs::write(d.join("jobs/j.json"), changed.to_string()).unwrap();
        assert_eq!(unlocked(&d, &load_jobs(&d).unwrap()), vec!["j"], "the checker changed after the lock");
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn dates_are_civil() {
        assert!(chrono_now().starts_with("20") && chrono_now().ends_with('Z'));
    }
}
