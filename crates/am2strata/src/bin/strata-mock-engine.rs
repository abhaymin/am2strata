//! A stand-in for PolyStrata's engines and image encoder, for tests without a GPU.
//!
//! Engine mode (default; started with `--serve` and any engine arguments): prints `INFO` and `READY`, then answers
//! each `GEN`/`GENI` line with the next scripted reply - `PP` progress, one `T <id>` per token, `DONE` - and
//! honours `STOP` (ends the reply with finish `cancel`) and `QUIT`.
//!
//! Vision mode (when started with `--mmproj`): prints `READY`, answers `ENC <image> <out>` by writing a small file and
//! `OK <tokens>`.
//!
//! Environment:
//!   MOCK_SCRIPT   a JSON file: {"replies": [[id, ...], ...], "finish": "stop", "delay_ms": 0}; replies are used in turn
//!                 (the last one repeats); without it every reply is the token 0 once
//!   MOCK_RECORD   a file each received command line is appended to
//!   MOCK_IMAGE_TOKENS  the token count the vision mode reports (default 4)

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

fn record(line: &str) {
    if let Ok(path) = std::env::var("MOCK_RECORD") {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "{line}");
        }
    }
}

fn vision() {
    let n = std::env::var("MOCK_IMAGE_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(4u32);
    let out = std::io::stdout();
    let mut out = out.lock();
    let _ = writeln!(out, "READY");
    let _ = out.flush();
    for line in std::io::stdin().lock().lines().map_while(Result::ok) {
        record(&format!("VISION {line}"));
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.as_slice() {
            ["ENC", img, dst] => {
                let ok = std::fs::metadata(img).is_ok() && std::fs::write(dst, vec![7u8; 16 * n as usize]).is_ok();
                let _ = writeln!(out, "{}", if ok { format!("OK {n}") } else { "ERR cannot read".into() });
            }
            ["QUIT"] => return,
            _ => {
                let _ = writeln!(out, "ERR unknown command");
            }
        }
        let _ = out.flush();
    }
}

struct Script {
    replies: Vec<Vec<u32>>,
    finish: String,
    delay: Duration,
}

fn script() -> Script {
    let mut s = Script { replies: vec![vec![0]], finish: "stop".into(), delay: Duration::ZERO };
    if let Ok(path) = std::env::var("MOCK_SCRIPT") {
        if let Ok(v) = std::fs::read_to_string(path).map(|t| serde_json::from_str::<serde_json::Value>(&t)) {
            let v = v.unwrap_or_default();
            if let Some(r) = v.get("replies").and_then(|r| r.as_array()) {
                s.replies = r
                    .iter()
                    .map(|x| x.as_array().map(|a| a.iter().filter_map(|n| n.as_u64().map(|n| n as u32)).collect()).unwrap_or_default())
                    .collect();
            }
            if let Some(f) = v.get("finish").and_then(|f| f.as_str()) {
                s.finish = f.into();
            }
            if let Some(d) = v.get("delay_ms").and_then(|d| d.as_u64()) {
                s.delay = Duration::from_millis(d);
            }
        }
    }
    s
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--mmproj") {
        return vision();
    }
    let max_context = args.iter().position(|a| a == "--max-context").and_then(|i| args.get(i + 1)).cloned().unwrap_or("8192".into());
    let script = script();
    let stdout = Arc::new(std::sync::Mutex::new(std::io::stdout()));
    let say = |s: &str| {
        let mut o = stdout.lock().unwrap();
        let _ = writeln!(o, "{s}");
        let _ = o.flush();
    };
    say(&format!("INFO context={max_context} mock=1"));
    say("READY mock");
    // stdin is read on its own thread so a STOP arrives while a reply is being written
    let (tx, rx) = mpsc::channel::<String>();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_reader = stop.clone();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines().map_while(Result::ok) {
            record(&line);
            if line.trim() == "STOP" {
                stop_reader.store(true, Ordering::SeqCst);
                continue;
            }
            if tx.send(line).is_err() {
                return;
            }
        }
    });
    let mut turn = 0usize;
    while let Ok(line) = rx.recv() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first().copied() {
            Some("QUIT") => return,
            // disk sessions, as `strata --serve` answers them: a progress line, a blocking step, then the result
            Some("SAVE") => {
                let path = line.splitn(2, ' ').nth(1).unwrap_or("");
                say("SWAIT capture 30");
                match std::fs::write(path, vec![b'x'; 1234]) {
                    Ok(()) => { say("SESSION 1234 1234"); say("SAVED 4096 1234 2.5"); }
                    Err(e) => say(&format!("SERR io 0 {e}")),
                }
            }
            Some("RESTORE") => {
                let path = line.splitn(2, ' ').nth(1).unwrap_or("");
                if std::path::Path::new(path).is_file() { say("SESSION 1234 1234"); say("RESTORED 4096 1234 1.5"); }
                else { say("SERR invalid 0 session file: saved with another model (model fingerprint differs)"); }
            }
            Some("GEN") | Some("GENI") => {
                stop.store(false, Ordering::SeqCst);
                let max_new: usize = f.get(1).and_then(|n| n.parse().ok()).unwrap_or(1);
                let n_prompt = f.last().map(|ids| ids.split(',').count()).unwrap_or(0);
                say(&format!("PP {n_prompt} {n_prompt} 1.0 1000.0"));
                let reply = script.replies.get(turn).or(script.replies.last()).cloned().unwrap_or_default();
                turn += 1;
                let mut sent = 0;
                let mut finish = script.finish.clone();
                for t in reply {
                    if stop.load(Ordering::SeqCst) {
                        finish = "cancel".into();
                        break;
                    }
                    if sent == max_new {
                        finish = "length".into();
                        break;
                    }
                    say(&format!("T {t}"));
                    sent += 1;
                    if !script.delay.is_zero() {
                        std::thread::sleep(script.delay);
                    }
                }
                if stop.load(Ordering::SeqCst) {
                    finish = "cancel".into();
                }
                say(&format!("DONE {sent} {n_prompt} 1.0 {}.0 {finish} 0 0 0", sent.max(1)));
            }
            _ => say("ERR unknown command"),
        }
    }
}
