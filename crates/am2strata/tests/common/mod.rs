//! Test harness: the real `am2strata` binary driving `strata-mock-engine` (no GPU). Each test gets its own
//! models folder, port, scripted engine replies and a record of every command the engine and encoder received.

#![allow(dead_code)]

/// the regression suite's shared helpers (docs/REGRESSION.md)
pub mod regress;

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use strata_frontend::{openai_to_messages, PromptBuilder, ToolOrders};
use strata_template::ChatTemplate;
use strata_tokenizer::Tokenizer;

pub const KEY: &str = "test-key";

pub fn testdata() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata")
}

pub fn tokenizer() -> Arc<Tokenizer> {
    Arc::new(Tokenizer::from_dir(testdata().join("tokenizer-qwen35")).expect("test tokenizer"))
}

/// Token ids of `text` as the model would write it (special tags parsed).
pub fn ids(text: &str) -> Vec<u32> {
    tokenizer().encode(text, true).expect("encode")
}

/// The prompt ids the server must send for an OpenAI request (computed with the verified front end).
pub fn expected_prompt(req: &Value) -> Vec<u32> {
    let tok = tokenizer();
    let tmpl = Arc::new(ChatTemplate::from_file(testdata().join("tokenizer-qwen35/chat_template.jinja")).unwrap());
    let p = openai_to_messages(req, &ToolOrders::default()).unwrap();
    PromptBuilder::new(tmpl, tok).build(&p).unwrap().ids
}

pub struct Server {
    child: Child,
    /// how it was started (restart() starts it again the same way, on the same data folder)
    model: String,
    extra: Value,
    pub port: u16,
    pub dir: PathBuf,
    pub record: PathBuf,
    pub base: String,
}

pub struct Model<'a> {
    pub id: &'a str,
    pub vision: bool,
    pub arch: Option<&'a str>,
    pub context: u32,
    /// more run-config keys (merged in)
    pub extra: Value,
}

impl Default for Model<'_> {
    fn default() -> Self {
        Model { id: "mock-a", vision: true, arch: None, context: 8192, extra: Value::Null }
    }
}

/// A port no other test of this run is given (asking the OS for port 0 and letting it go raced between parallel tests:
/// two servers could get the same one, and a test talked to another test's server).
pub fn free_port() -> u16 {
    static NEXT: std::sync::atomic::AtomicU16 = std::sync::atomic::AtomicU16::new(0);
    let _ = NEXT.compare_exchange(0, 20000 + (std::process::id() % 20000) as u16, std::sync::atomic::Ordering::SeqCst,
                                  std::sync::atomic::Ordering::SeqCst);
    loop {
        let p = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if std::net::TcpListener::bind(("127.0.0.1", p)).is_ok() {
            return p;
        }
    }
}

fn write_config(dir: &Path, m: &Model) {
    let mock = env!("CARGO_BIN_EXE_strata-mock-engine");
    let mut cfg = json!({
        "exe": mock,
        "args": ["--model", dir.join(format!("{}.gguf", m.id)), "--max-context", m.context.to_string()],
        "cwd": dir, "tokenizer": testdata().join("tokenizer-qwen35"),
        "model_name": m.id, "display_name": format!("Mock {}", m.id),
        "log": dir.join(format!("{}.log", m.id)), "api_key": KEY, "host": "127.0.0.1",
    });
    if m.vision {
        cfg["vision"] = json!({"exe": mock, "mmproj": dir.join("mmproj.gguf"), "model": dir.join("m.gguf"), "gpu": false});
    }
    if let Some(a) = m.arch {
        cfg["arch"] = json!(a);
    }
    if let (Some(c), Some(e)) = (cfg.as_object_mut(), m.extra.as_object()) {
        c.extend(e.clone());
    }
    std::fs::write(dir.join(format!("strata-{}.json", m.id)), serde_json::to_vec_pretty(&cfg).unwrap()).unwrap();
}

impl Server {
    /// Start a server over `models` (the first is loaded at start) with the engine's scripted `replies`.
    pub fn start(models: &[Model], replies: &[Vec<u32>], extra: Value) -> Server {
        let dir = std::env::temp_dir().join(format!("am2strata-test-{}-{}", std::process::id(), uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        for m in models {
            write_config(&dir, m);
        }
        let mut script = json!({"replies": replies});
        if let (Some(s), Some(e)) = (script.as_object_mut(), extra.as_object()) {
            s.extend(e.clone());
        }
        std::fs::write(dir.join("script.json"), script.to_string()).unwrap();
        // what models took from free memory at earlier loads (mem.rs), for the memory guard's tests
        if let Some(m) = extra.get("model_memory") {
            std::fs::write(dir.join("model-memory.json"), m.to_string()).unwrap();
        }
        // API calls get memory by default; the tests that check exact prompts run without it unless they ask
        if extra.get("memory_api").is_none() {
            std::fs::create_dir_all(dir.join("memory")).unwrap();
            std::fs::write(dir.join("memory/settings.json"), r#"{"api": false}"#).unwrap();
        }
        let record = dir.join("record.txt");
        // a port another program took between `free_port` and the server's own bind stops the server at once: start
        // it again on the next port instead of waiting 30 s for a server that is gone (seen once on a macOS runner)
        let mut gone = Vec::new();
        for _ in 0..3 {
            let port = free_port();
            let child = spawn(&dir, port, models[0].id, &extra, &record);
            let mut s = Server { child, model: models[0].id.to_string(), extra: extra.clone(), port, dir: dir.clone(), record: record.clone(),
                                 base: format!("http://127.0.0.1:{port}") };
            match s.wait_until_up() {
                Ok(()) => return s,
                Err(why) => {
                    gone.push(format!("port {port}: {why}"));
                    // keep the folder for the next try (dropping a Server removes it)
                    stop(&mut s.child);
                    std::mem::forget(s);
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
        panic!("am2strata did not start: {}", gone.join("; "));
    }

    /// Wait for /health; Err says whether the server ended (and how) or is still running but does not answer.
    fn wait_until_up(&mut self) -> Result<(), String> {
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(30) {
            if reqwest::blocking::get(format!("{}/health", self.base)).is_ok_and(|r| r.status().is_success()) {
                return Ok(());
            }
            if let Ok(Some(status)) = self.child.try_wait() {
                let err = std::fs::read_to_string(self.dir.join(format!("stderr-{}.txt", self.port))).unwrap_or_default();
                let tail: Vec<&str> = err.lines().rev().take(5).collect::<Vec<_>>().into_iter().rev().collect();
                return Err(format!("it ended after {:.1} s ({status}): {}", t0.elapsed().as_secs_f32(), tail.join(" | ")));
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        Err("still running after 30 s, /health does not answer".into())
    }

    pub fn post(&self, path: &str, body: &Value) -> (u16, Value) {
        let r = reqwest::blocking::Client::new()
            .post(format!("{}{path}", self.base))
            .bearer_auth(KEY)
            .json(body)
            .timeout(Duration::from_secs(60))
            .send()
            .expect("request");
        let code = r.status().as_u16();
        (code, r.json().unwrap_or(Value::Null))
    }

    /// DELETE with the key.
    pub fn delete(&self, path: &str) -> (u16, Value) {
        let r = reqwest::blocking::Client::new().delete(format!("{}{path}", self.base)).bearer_auth(KEY).send().unwrap();
        let code = r.status().as_u16();
        (code, r.json().unwrap_or(Value::Null))
    }

    pub fn get(&self, path: &str, auth: bool) -> (u16, Value) {
        let mut rb = reqwest::blocking::Client::new().get(format!("{}{path}", self.base));
        if auth {
            rb = rb.bearer_auth(KEY);
        }
        let r = rb.send().expect("request");
        (r.status().as_u16(), r.json().unwrap_or(Value::Null))
    }

    /// A streamed request's SSE events as (event name, data) pairs.
    pub fn stream(&self, path: &str, body: &Value) -> Vec<(String, Value)> {
        let text = reqwest::blocking::Client::new()
            .post(format!("{}{path}", self.base))
            .header("x-api-key", KEY)
            .json(body)
            .timeout(Duration::from_secs(60))
            .send()
            .expect("request")
            .text()
            .unwrap();
        let mut out = Vec::new();
        let mut name = String::new();
        for line in text.lines() {
            if let Some(n) = line.strip_prefix("event:") {
                name = n.trim().to_string();
            } else if let Some(d) = line.strip_prefix("data:") {
                let d = d.trim();
                out.push((name.clone(), if d == "[DONE]" { json!("[DONE]") } else { serde_json::from_str(d).unwrap_or(json!(d)) }));
            }
        }
        out
    }

    /// Every command line the mock engine and encoder received so far.
    pub fn commands(&self) -> Vec<String> {
        std::fs::read_to_string(&self.record).unwrap_or_default().lines().map(str::to_owned).collect()
    }

    pub fn gen_lines(&self) -> Vec<String> {
        self.commands().into_iter().filter(|l| l.starts_with("GEN")).collect()
    }
}

/// The server process over `dir` (its run configs, data and script).
/// Stop a test server and everything it started (its own process group: an engine, a sandbox desktop's Xvfb), so a
/// test never leaves processes behind even though it kills rather than asks.
fn stop(child: &mut Child) {
    #[cfg(unix)]
    unsafe { libc::kill(-(child.id() as i32), libc::SIGKILL); }
    let _ = child.kill();
    let _ = child.wait();
}

fn spawn(dir: &Path, port: u16, model: &str, extra: &Value, record: &Path) -> Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_am2strata"));
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
    cmd
            .args(["--models-dir"]).arg(&dir)
            .args(["--model", model, "--port", &port.to_string(), "--host", "127.0.0.1"])
            .arg("--web-dir").arg(extra.get("web_dir").and_then(|v| v.as_str()).map(PathBuf::from).unwrap_or_else(|| testdata().join("web")))
            .arg("--data-dir").arg(&dir)
            .env("AM2STRATA_MEDIA_EXE", if extra.get("no_media").is_some() { PathBuf::from("/nonexistent/am2strata-media") }
                                        else { PathBuf::from(env!("CARGO_BIN_EXE_strata-mock-media")) })
            .env("MOCK_MEDIA_RECORD", dir.join("media-record.txt"))
            .env("AM2_TEST_MEM_GB", extra.get("mem_gb").map(|v| v.to_string()).unwrap_or_else(|| "64".into()))
            .envs(extra.get("no_multilingual").map(|_| ("MOCK_NO_MULTILINGUAL", "1")))
            .envs(extra.get("no_music").map(|_| ("MOCK_NO_MUSIC", "1")))
            .env("MOCK_MEDIA_DELAY_MS", extra.get("media_delay_ms").map(|v| v.to_string()).unwrap_or_else(|| "0".into()))
            .env("MOCK_SCRIPT", dir.join("script.json"))
            .env("AM2_JOB_IDS_FROM", "1")
            .env("MOCK_RECORD", record)
            // "server_log": the server's own log in <dir>/server.log (info), for tests that read it
            .env("RUST_LOG", if extra.get("server_log").is_some() { "info" } else { "warn" })
            .env("AM2STRATA_NO_MDNS", "1")
            // more environment for the server, from the test (e.g. AM2STRATA_HOME), never changed in this process
            .envs(extra.get("env").and_then(|e| e.as_object()).into_iter().flatten()
                  .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string()))))
            // the server's error output, so a server that does not start says why (wait_until_up reads its tail)
            .stderr(std::fs::File::create(dir.join(format!("stderr-{port}.txt"))).map(Stdio::from).unwrap_or(Stdio::null()))
            .stdout(match extra.get("server_log") {
                Some(_) => std::fs::OpenOptions::new().create(true).append(true).open(dir.join("server.log")).map(Stdio::from).unwrap_or(Stdio::null()),
                None => Stdio::null(),
            })
            .spawn()
            .expect("start am2strata")
}

impl Server {
    /// Stop the server and start it again on the same data folder and port (a restart, as the service does).
    pub fn restart(&mut self) {
        stop(&mut self.child);
        self.child = spawn(&self.dir, self.port, &self.model, &self.extra, &self.record);
        if let Err(why) = self.wait_until_up() {
            panic!("am2strata did not start again: {why}");
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        stop(&mut self.child);
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The ids at the end of a GEN line.
pub fn line_ids(line: &str) -> Vec<u32> {
    line.rsplit(' ').next().unwrap().split(',').map(|x| x.parse().unwrap()).collect()
}

pub fn png() -> String {
    // a 2x1 PNG (red, blue)
    use base64::Engine as _;
    let mut img = image::RgbImage::new(2, 1);
    img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
    img.put_pixel(1, 0, image::Rgb([0, 0, 255]));
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(buf.into_inner()))
}
