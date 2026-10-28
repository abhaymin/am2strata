//! Run a chat's Python code block on this PC, for the web app's canvas (GET/POST /api/run).
//! Only when a person clicks Run, only from this server's own page, and inside a bubblewrap sandbox: no network,
//! the system read-only, home folders hidden, a fresh work folder, a time limit. What the code prints and the
//! files it writes (pictures, HTML, CSV...) come back. "code_run": false in the config switches it off.

use crate::{authorized, error, foreign_origin, not_own_page, AppState};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine as _;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;

const BWRAP: &str = "/usr/bin/bwrap";
const MAX_CODE: usize = 200_000;
const MAX_OUT: usize = 200_000;
const MAX_FILES: usize = 12;
const MAX_FILE_BYTES: u64 = 8 << 20;
const MAX_ALL_FILES: u64 = 24 << 20;

pub struct CodeRun {
    pub enabled: bool,
    /// off because the server listens beyond this PC without an API key
    pub needs_key: bool,
    pub timeout: Duration,
    python: Option<PathBuf>,
}

impl CodeRun {
    pub fn from_config(rest: &std::collections::BTreeMap<String, Value>) -> CodeRun {
        let python = ["/usr/bin/python3", "/usr/local/bin/python3"].iter().map(PathBuf::from).find(|p| p.exists());
        let enabled = rest.get("code_run") != Some(&Value::Bool(false)) && Path::new(BWRAP).exists() && python.is_some();
        let secs = rest.get("code_run_timeout_s").and_then(Value::as_u64).filter(|s| *s > 0).unwrap_or(30).min(600);
        CodeRun { enabled, needs_key: false, timeout: Duration::from_secs(secs), python }
    }

    fn why_off(&self, rest_off: bool) -> &'static str {
        if self.needs_key { "the server listens beyond this PC without an API key (set \"api_key\" in the run config)" }
        else if rest_off { "switched off in the config (\"code_run\": false)" }
        else if !Path::new(BWRAP).exists() { "needs bubblewrap for its sandbox: sudo apt install bubblewrap" }
        else { "needs python3 on this PC" }
    }
}

pub async fn info(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let c = &s.code_run;
    Json(json!({"enabled": c.enabled, "languages": if c.enabled { json!(["python"]) } else { json!([]) },
                "sandbox": "bubblewrap: no network, system read-only, home hidden",
                "timeout_s": c.timeout.as_secs(),
                "reason": if c.enabled { Value::Null } else { json!(c.why_off(!s.code_run_config_on)) }})).into_response()
}

pub async fn run(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if foreign_origin(&s, &headers) {
        return not_own_page("code can be run");
    }
    let c = &s.code_run;
    if !c.enabled {
        return error(StatusCode::NOT_IMPLEMENTED, "not_available", format!("running code is off: {}", c.why_off(!s.code_run_config_on)));
    }
    let lang = req.get("language").and_then(Value::as_str).unwrap_or("python").to_lowercase();
    if !matches!(lang.as_str(), "python" | "py" | "python3") {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("{lang} code cannot be run here (Python only; HTML previews in the browser)"));
    }
    let Some(code) = req.get("code").and_then(Value::as_str).filter(|c| !c.trim().is_empty()) else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "send {\"language\": \"python\", \"code\": \"...\"}");
    };
    if code.len() > MAX_CODE {
        return error(StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error", "the code is longer than 200 KB");
    }
    let stdin = req.get("stdin").and_then(Value::as_str).unwrap_or("").to_string();
    match sandboxed(c, code, &stdin).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "run_error", e),
    }
}

async fn sandboxed(c: &CodeRun, code: &str, stdin: &str) -> Result<Value, String> {
    let work = std::env::temp_dir().join(format!("am2strata-run-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&work).map_err(|e| e.to_string())?;
    let result = run_in(c, &work, code, stdin).await;
    let _ = std::fs::remove_dir_all(&work);
    result
}

async fn run_in(c: &CodeRun, work: &Path, code: &str, stdin: &str) -> Result<Value, String> {
    std::fs::write(work.join("main.py"), code).map_err(|e| e.to_string())?;
    let python = c.python.as_ref().ok_or("no python3")?;
    let mut cmd = tokio::process::Command::new(BWRAP);
    cmd.args(["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/home",
              "--tmpfs", "/root", "--tmpfs", "/run/user"])
        .arg("--bind").arg(work).arg("/tmp/work")
        .args(["--chdir", "/tmp/work", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
               "--setenv", "HOME", "/tmp/work", "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin", "--setenv", "MPLBACKEND", "Agg",
               "--setenv", "PYTHONUNBUFFERED", "1", "--setenv", "LANG", "C.UTF-8", "--setenv", "OPENBLAS_NUM_THREADS", "4"])
        .arg(python).args(["-I", "main.py"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    let t0 = Instant::now();
    let mut child = cmd.spawn().map_err(|e| format!("could not start the sandbox: {e}"))?;
    if let Some(mut si) = child.stdin.take() {
        let data = stdin.as_bytes().to_vec();
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            let _ = si.write_all(&data).await;
        });
    }
    let (mut so, mut se) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
    let read_out = tokio::spawn(async move { capped(&mut so).await });
    let read_err = tokio::spawn(async move { capped(&mut se).await });
    let (status, timed_out) = match tokio::time::timeout(c.timeout, child.wait()).await {
        Ok(s) => (s.ok(), false),
        Err(_) => {
            let _ = child.kill().await;
            (None, true)
        }
    };
    let (stdout, out_cut) = read_out.await.unwrap_or_default();
    let (mut stderr, err_cut) = read_err.await.unwrap_or_default();
    if timed_out {
        stderr.push_str(&format!("\n[stopped: the {} s time limit]", c.timeout.as_secs()));
    }
    Ok(json!({
        "exit_code": status.and_then(|s| s.code()),
        "timed_out": timed_out,
        "duration_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
        "stdout": stdout, "stderr": stderr, "truncated": out_cut || err_cut,
        "files": files(work),
    }))
}

async fn capped<R: tokio::io::AsyncRead + Unpin>(r: &mut R) -> (String, bool) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let mut cut = false;
    loop {
        match r.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if buf.len() < MAX_OUT {
                    buf.extend_from_slice(&chunk[..n.min(MAX_OUT - buf.len())]);
                }
                cut |= buf.len() >= MAX_OUT;
            }
        }
    }
    (String::from_utf8_lossy(&buf).into_owned(), cut)
}

fn mime(name: &str) -> Option<&'static str> {
    let ext = name.rsplit('.').next()?.to_lowercase();
    Some(match ext.as_str() {
        "png" => "image/png", "jpg" | "jpeg" => "image/jpeg", "gif" => "image/gif", "webp" => "image/webp",
        "svg" => "image/svg+xml", "html" | "htm" => "text/html", "csv" => "text/csv", "json" => "application/json",
        "txt" | "md" | "log" => "text/plain", "pdf" => "application/pdf",
        _ => return None,
    })
}

/// The files the code wrote (main.py aside), largest-last, within the caps.
fn files(work: &Path) -> Vec<Value> {
    let mut found = Vec::new();
    let mut stack = vec![work.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            let Ok(m) = std::fs::symlink_metadata(&p) else { continue };
            if m.is_dir() && found.len() < 200 {
                stack.push(p);
            } else if m.is_file() {
                let rel = p.strip_prefix(work).unwrap_or(&p).to_string_lossy().into_owned();
                if rel != "main.py" {
                    found.push((rel, p, m.len()));
                }
            }
        }
    }
    found.sort_by_key(|f| f.2);
    let (mut out, mut total) = (Vec::new(), 0u64);
    for (rel, p, len) in found {
        let Some(mt) = mime(&rel) else {
            out.push(json!({"name": rel, "size": len, "mime": null, "data": null}));
            continue;
        };
        if out.len() >= MAX_FILES || len > MAX_FILE_BYTES || total + len > MAX_ALL_FILES {
            out.push(json!({"name": rel, "size": len, "mime": mt, "data": null, "note": "too large to send back"}));
            continue;
        }
        if let Ok(b) = std::fs::read(&p) {
            total += len;
            out.push(json!({"name": rel, "size": len, "mime": mt, "data": base64::engine::general_purpose::STANDARD.encode(b)}));
        }
    }
    out
}
