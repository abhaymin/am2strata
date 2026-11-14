//! HTTPS end to end against the mock engine: on from the run config, HSTS only over TLS, this PC keeps plain HTTP,
//! and turned on, changed and off at runtime through /api/tls (what Settings › Secure connection and
//! `am2strata-run tls` use). The client trusts only the server's own certificate, never "accept anything".

mod common;

use common::*;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

fn trusting(cert_pem: &[u8]) -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .add_root_certificate(reqwest::Certificate::from_pem(cert_pem).unwrap())
        .timeout(Duration::from_secs(10))
        .build().unwrap()
}

fn wait_https(dir: &std::path::Path, port: u16) -> reqwest::blocking::Client {
    let t0 = Instant::now();
    loop {
        if let Ok(pem) = std::fs::read(dir.join("server-tls/cert.pem")) {
            let c = trusting(&pem);
            if c.get(format!("https://localhost:{port}/health")).send().is_ok_and(|r| r.status().is_success()) {
                return c;
            }
        }
        assert!(t0.elapsed() < Duration::from_secs(30), "HTTPS did not come up on {port}");
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn tls_post(s: &Server, body: Value) -> (u16, Value) {
    s.post("/api/tls", &body)
}

#[test]
fn https_from_the_run_config_with_hsts_only_over_tls() {
    let port = free_port();
    let s = Server::start(&[Model { extra: json!({"tls": {"https_port": port}}), ..Model::default() }], &[ids("hi")], json!({}));
    let c = wait_https(&s.dir, port);
    let r = c.get(format!("https://localhost:{port}/health")).send().unwrap();
    assert_eq!(r.headers()["strict-transport-security"], "max-age=31536000", "HSTS over HTTPS");
    // the API answers over HTTPS with the key, like over HTTP
    let r = c.get(format!("https://localhost:{port}/v1/models")).bearer_auth(KEY).send().unwrap();
    assert_eq!(r.status(), 200);
    // plain HTTP from this PC keeps answering (local-only), and never carries HSTS
    let r = reqwest::blocking::get(format!("{}/health", s.base)).unwrap();
    assert_eq!(r.status(), 200);
    assert!(r.headers().get("strict-transport-security").is_none(), "no HSTS over plain HTTP");
    // the key stays private
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(s.dir.join("server-tls/key.pem")).unwrap().permissions().mode() & 0o777, 0o600);
    }
    // the status says what is in force
    let (code, st) = s.get("/api/tls", true);
    assert_eq!(code, 200);
    assert_eq!((st["on"].as_bool(), st["mode"].as_str(), st["https_port"].as_u64()), (Some(true), Some("self-signed"), Some(port as u64)));
    assert_eq!(st["certificate"]["self_signed"], true);
    let names: Vec<&str> = st["certificate"]["names"].as_array().unwrap().iter().filter_map(|x| x.as_str()).collect();
    assert!(names.contains(&"localhost") && names.contains(&"127.0.0.1"), "names {names:?}");
    assert!(st["certificate"]["sha256"].as_str().unwrap().len() == 95, "AA:BB:… fingerprint");
}

#[test]
fn turned_on_changed_and_off_at_runtime_and_kept_in_the_run_config() {
    let port = free_port();
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({}));
    let (_, st) = s.get("/api/tls", true);
    assert_eq!(st["on"], false, "off unless configured");
    assert!(s.get("/api/tls", false).0 == 401, "the status needs the key");

    // on, self-signed, no restart
    let (code, st) = tls_post(&s, json!({"action": "self-signed", "https_port": port}));
    assert_eq!(code, 200, "{st}");
    assert_eq!(st["on"], true);
    let c = wait_https(&s.dir, port);
    let cfg: Value = serde_json::from_slice(&std::fs::read(s.dir.join("strata-mock-a.json")).unwrap()).unwrap();
    assert_eq!(cfg["tls"]["mode"], "self-signed", "kept for the next start");
    assert_eq!(cfg["tls"]["https_port"], port);

    // load: a certificate with a key that is not its own is refused, and nothing changes
    let cert = std::fs::read_to_string(s.dir.join("server-tls/cert.pem")).unwrap();
    let (code, e) = tls_post(&s, json!({"action": "load", "cert_pem": cert, "key_pem": "-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n"}));
    assert_eq!(code, 400, "{e}");
    assert!(c.get(format!("https://localhost:{port}/health")).send().is_ok(), "still serving the earlier certificate");
    // load the real pair: mode files, the same certificate served
    let key = std::fs::read_to_string(s.dir.join("server-tls/key.pem")).unwrap();
    let (code, st) = tls_post(&s, json!({"action": "load", "cert_pem": cert, "key_pem": key}));
    assert_eq!(code, 200, "{st}");
    assert_eq!(st["mode"], "files");
    assert!(c.get(format!("https://localhost:{port}/health")).send().is_ok());

    // another port while the listener runs takes a restart, said plainly
    let (code, e) = tls_post(&s, json!({"action": "self-signed", "https_port": free_port()}));
    assert_eq!(code, 400);
    assert!(e["error"]["message"].as_str().unwrap().contains("restart"), "{e}");
    // the API port itself is refused
    let (code, _) = tls_post(&s, json!({"action": "self-signed", "https_port": s.port}));
    assert_eq!(code, 400);

    // off: new HTTPS connections are refused, plain HTTP answers, the block leaves the run config
    let (code, st) = tls_post(&s, json!({"action": "off"}));
    assert_eq!(code, 200);
    assert_eq!(st["on"], false);
    let fresh = trusting(cert.as_bytes());
    assert!(fresh.get(format!("https://localhost:{port}/health")).send().is_err(), "HTTPS off");
    assert_eq!(reqwest::blocking::get(format!("{}/health", s.base)).unwrap().status(), 200);
    let cfg: Value = serde_json::from_slice(&std::fs::read(s.dir.join("strata-mock-a.json")).unwrap()).unwrap();
    assert!(cfg.get("tls").is_none());
    // and a wrong action is a 400
    assert_eq!(tls_post(&s, json!({"action": "nope"})).0, 400);
}

#[test]
fn a_wrong_tls_block_stops_the_start_instead_of_running_plain() {
    // a typo must not leave the server on plain HTTP by surprise: the server refuses to start
    let dir = std::env::temp_dir().join(format!("am2-https-bad-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = json!({"exe": env!("CARGO_BIN_EXE_strata-mock-engine"), "args": ["--model", dir.join("m.gguf"), "--max-context", "8192"],
                     "cwd": dir, "tokenizer": testdata().join("tokenizer-qwen35"), "model_name": "m", "api_key": KEY,
                     "tls": {"mode": "selfsigned-typo"}});
    std::fs::write(dir.join("strata-m.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_am2strata"))
        .args(["--models-dir"]).arg(&dir).args(["--port", &free_port().to_string(), "--host", "127.0.0.1", "--data-dir"]).arg(&dir)
        .env("AM2STRATA_NO_MDNS", "1").output().unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("unknown mode"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
