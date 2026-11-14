//! A small QMP client (QEMU's JSON control protocol) over the VM's private Unix socket: pause, resume, save the
//! running VM to a file (a migration to `file:`), restore, and quit.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

pub struct Qmp {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

impl Qmp {
    /// Connect (waiting up to `wait` for QEMU to create the socket) and leave capability negotiation.
    pub fn connect(sock: &Path, wait: Duration) -> Result<Qmp, String> {
        let t0 = Instant::now();
        let stream = loop {
            match UnixStream::connect(sock) {
                Ok(s) => break s,
                Err(e) if t0.elapsed() < wait => { let _ = e; std::thread::sleep(Duration::from_millis(30)); }
                Err(e) => return Err(format!("the VM's control socket {}: {e}", sock.display())),
            }
        };
        stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
        let writer = stream.try_clone().map_err(|e| e.to_string())?;
        let mut q = Qmp { reader: BufReader::new(stream), writer };
        let mut greeting = String::new();
        q.reader.read_line(&mut greeting).map_err(|e| format!("QMP greeting: {e}"))?;
        q.cmd("qmp_capabilities", None)?;
        Ok(q)
    }

    /// One command; its `return`, or its error's description. Events that arrive meanwhile are skipped.
    pub fn cmd(&mut self, name: &str, args: Option<Value>) -> Result<Value, String> {
        let mut msg = json!({"execute": name});
        if let Some(a) = args {
            msg["arguments"] = a;
        }
        writeln!(self.writer, "{msg}").map_err(|e| format!("QMP {name}: {e}"))?;
        loop {
            let mut line = String::new();
            if self.reader.read_line(&mut line).map_err(|e| format!("QMP {name}: {e}"))? == 0 {
                return Err(format!("QMP {name}: the VM closed its control socket"));
            }
            let v: Value = serde_json::from_str(&line).unwrap_or_default();
            if let Some(r) = v.get("return") {
                return Ok(r.clone());
            }
            if let Some(e) = v.get("error") {
                return Err(format!("QMP {name}: {}", e["desc"].as_str().unwrap_or("failed")));
            }
        }
    }

    /// "running", "paused", "inmigrate", "postmigrate", …
    pub fn status(&mut self) -> Result<String, String> {
        Ok(self.cmd("query-status", None)?["status"].as_str().unwrap_or("").to_string())
    }

    /// Wait for the current migration (a save or a restore) to end; Ok when it completed.
    pub fn wait_migration(&mut self, limit: Duration) -> Result<(), String> {
        let t0 = Instant::now();
        loop {
            let m = self.cmd("query-migrate", None)?;
            match m["status"].as_str() {
                Some("completed") => return Ok(()),
                Some("failed") | Some("cancelled") => {
                    return Err(format!("saving or restoring the VM failed: {}", m["error-desc"].as_str().unwrap_or("no reason given")));
                }
                _ if t0.elapsed() > limit => return Err("saving or restoring the VM took too long".into()),
                _ => std::thread::sleep(Duration::from_millis(40)),
            }
        }
    }

    /// Stop the CPUs and write the whole VM state to `file` (its disk must not change after this: the VM is quit next).
    pub fn save_to(&mut self, file: &Path) -> Result<(), String> {
        self.cmd("stop", None)?;
        self.cmd("migrate", Some(json!({"uri": format!("file:{}", file.display())})))?;
        self.wait_migration(Duration::from_secs(300))
    }
}
