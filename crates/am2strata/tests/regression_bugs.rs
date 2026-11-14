//! One named test per class of bug fixed in this repository's history (docs/REGRESSION.md), each with the pull request
//! it came from. The rule: EVERY BUG FIX ADDS A TEST HERE (or, for a library crate, a `regress_…` unit test beside
//! the fix, listed at the end of this file), so the same bug cannot come back unseen.
//!
//! Pull requests: https://github.com/abhaymin/am2strata/pull/<n>

mod common;

use common::regress::*;
use common::*;
use serde_json::{json, Value};
use std::time::Duration;

fn get(s: &Server, path: &str, key: Option<&str>) -> Answer {
    Req::new("GET", &s.base, path).key(key).send()
}

/// A stand-in for the am2strata-workshop command, written into `dir`: prints the job's start event; for an input
/// whose name holds "slow" it starts a child of its own (as a real job starts threads and helpers), writes that
/// own pid and that child's to `<dir>/grandchild.pid` and waits; anything else "converts" by writing the output file.
#[cfg(unix)]
fn fake_workshop(dir: &std::path::Path) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let p = dir.join("fake-workshop.sh");
    std::fs::write(&p, format!(r#"#!/bin/sh
shift; kind="$1"; input="$2"; out=""
while [ $# -gt 0 ]; do [ "$1" = "--out" ] && out="$2"; shift; done
echo "{{\"event\":\"start\",\"what\":\"$kind $input\"}}"
case "$input" in
  *slow*) sleep 120 & echo "$$ $!" > "{pid}.tmp"; mv "{pid}.tmp" "{pid}"; wait ;;
esac
mkdir -p "$(dirname "$out")" 2>/dev/null; echo data > "$out"
echo "{{\"event\":\"done\",\"output\":\"$out\",\"kind\":\"gguf\",\"bytes\":5,\"seconds\":0.1}}"
"#, pid = dir.join("grandchild.pid").display())).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

/// A Hugging Face model folder (as far as the Workshop's checks look) inside the server's models folder.
fn hf_model(s: &Server, name: &str) -> std::path::PathBuf {
    let d = s.dir.join(name);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("config.json"), "{}").unwrap();
    std::fs::write(d.join("model.safetensors"), "x").unwrap();
    d
}

// =====================================================================================================================
// PR #13 — Model Workshop
// =====================================================================================================================

/// PR #13 (review fixes, commit 46462b1): Workshop paths stay inside the Workshop's folders. `..` out of a folder,
/// an absolute path elsewhere, a relative path, and an output outside are all refused, and a refused output makes no
/// folders anywhere.
#[test]
fn pr13_workshop_paths_cannot_traverse_out_of_their_folders() {
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({}));
    let model = hf_model(&s, "hf-model");
    let outside = TempDir::new("outside");
    let refused = |body: Value, why: &str| {
        let (code, e) = s.post("/v1/workshop/jobs", &body);
        assert_eq!(code, 400, "{why}: {e}");
        assert!(e["error"]["message"].is_string(), "{why}: {e}");
    };
    refused(json!({"kind": "convert", "input": "/etc"}), "an absolute path outside");
    refused(json!({"kind": "convert", "input": model.join("../../..")}), "`..` out of the models folder");
    refused(json!({"kind": "convert", "input": format!("{}/../{}", model.display(), "../../etc")}), "`..` in the middle");
    refused(json!({"kind": "convert", "input": "hf-model"}), "a relative path");
    refused(json!({"kind": "convert", "input": model, "output": outside.0.join("a/b/out.gguf")}), "an output outside");
    assert!(!outside.0.join("a").exists(), "a refused output made a folder outside the Workshop's folders");
    refused(json!({"kind": "convert", "input": model, "output": s.dir.join("workshop/../../escaped.gguf")}), "an output that climbs out");
    refused(json!({"kind": "quantize", "input": "/etc/passwd", "type": "q4_k"}), "quantize reads only its folders");
    refused(json!({"kind": "merge", "base": "/etc", "adapter": model}), "merge reads only its folders");
    // the dataset reader and editor take paths too
    for path in ["/etc/passwd", "/etc/hostname.jsonl", &format!("{}/workshop/datasets/../../../x.jsonl", s.dir.display()), "x.jsonl"] {
        let a = get(&s, &format!("/v1/workshop/dataset?path={}", path.replace('/', "%2F")), Some(KEY));
        assert_eq!(a.status, 400, "dataset {path}: {}", a.text);
        assert_eq!(s.post("/v1/workshop/dataset", &json!({"path": path, "remove": [0]})).0, 400, "dataset edit {path}");
    }
    let (_, list) = s.get("/v1/workshop/jobs", true);
    assert_eq!(list["jobs"].as_array().map(Vec::len), Some(0), "no refused request became a job: {list}");
}

/// PR #13 (commit 46462b1): a symlink inside a Workshop folder that points outside it is outside, as an input, as
/// the folder of an output, and as a dataset.
#[cfg(unix)]
#[test]
fn pr13_workshop_symlinks_cannot_escape_their_folders() {
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({}));
    let model = hf_model(&s, "hf-model");
    let outside = TempDir::new("symlink-target");
    std::fs::write(outside.0.join("config.json"), "{}").unwrap();
    std::fs::write(outside.0.join("secret.jsonl"), "{\"messages\": [{\"role\": \"user\", \"content\": \"secret\"}]}\n").unwrap();
    std::os::unix::fs::symlink(&outside.0, s.dir.join("sneaky")).unwrap();
    let (code, e) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": s.dir.join("sneaky")}));
    assert_eq!(code, 400, "an input through the link: {e}");
    let (code, e) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": model, "output": s.dir.join("sneaky/out.gguf")}));
    assert_eq!(code, 400, "an output through the link: {e}");
    assert!(!outside.0.join("out.gguf").exists());
    // a dataset inside the Workshop is read; a link to one outside is not
    let datasets = s.dir.join("workshop/datasets");
    std::fs::create_dir_all(&datasets).unwrap();
    std::fs::write(datasets.join("ok.jsonl"), "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}\n").unwrap();
    std::os::unix::fs::symlink(outside.0.join("secret.jsonl"), datasets.join("leak.jsonl")).unwrap();
    let q = |name: &str| format!("/v1/workshop/dataset?path={}", datasets.join(name).display().to_string().replace('/', "%2F"));
    let ok = get(&s, &q("ok.jsonl"), Some(KEY));
    assert_eq!((ok.status, ok.json()["total"].as_u64()), (200, Some(1)), "{}", ok.text);
    let leak = get(&s, &q("leak.jsonl"), Some(KEY));
    assert_eq!(leak.status, 400, "{}", leak.text);
    assert!(!leak.text.contains("secret\"}"), "the file outside was not read out");
    assert_eq!(s.post("/v1/workshop/dataset", &json!({"path": datasets.join("leak.jsonl"), "remove": [0]})).0, 400);
    assert!(std::fs::read_to_string(outside.0.join("secret.jsonl")).unwrap().contains("secret"), "the file outside was not edited");
}

/// PR #16: an uploaded file's name is made safe (no folders, no leading dot) and the file lands in the uploads
/// folder, whatever the name asked for.
#[test]
fn pr16_workshop_upload_names_cannot_place_a_file_elsewhere() {
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({}));
    let uploads = s.dir.join("workshop/uploads");
    for name in ["..%2F..%2Fescaped.txt", "%2Fetc%2Fcron.d%2Fx", "..%5C..%5Cwin.txt", ".bashrc", "a%00b.txt"] {
        let a = Req::new("POST", &s.base, &format!("/v1/workshop/uploads?name={name}")).key(Some(KEY)).raw("application/octet-stream", b"data").send();
        assert_eq!(a.status, 200, "{name}: {}", a.text);
        let path = std::path::PathBuf::from(a.json()["path"].as_str().unwrap());
        assert_eq!(path.parent(), Some(uploads.as_path()), "{name} was saved at {}", path.display());
        assert!(!path.file_name().unwrap().to_string_lossy().starts_with('.'), "{name}: no hidden file");
    }
    assert!(!s.dir.join("escaped.txt").exists() && !s.dir.join("workshop/escaped.txt").exists());
    for bad in ["", "...", "%2F"] {
        let a = Req::new("POST", &s.base, &format!("/v1/workshop/uploads?name={bad}")).key(Some(KEY)).raw("application/octet-stream", b"data").send();
        assert_eq!(a.status, 400, "a name with nothing left is refused ({bad:?})");
    }
    let a = Req::new("POST", &s.base, "/v1/workshop/uploads?name=empty.txt").key(Some(KEY)).raw("application/octet-stream", b"").send();
    assert_eq!(a.status, 400, "an empty file is refused");
}

/// PR #13 (commit dfdd000, "the key is checked before the job id is parsed"): a stranger gets 401 from a route with
/// an id in it whatever the id looks like: never the 404 or 400 that would tell them the route read it. With the
/// key, the same ids are a plain 404.
#[test]
fn pr13_an_id_in_the_path_is_read_only_after_the_key() {
    let s = Server::start(&[client_model(json!({}))], &[ids("hi")], json!({}));
    let ids_ = ["1", "0", "-1", "abc", "1e9", "99999999999999999999999999", "%20", "1%2F..%2F2", "null"];
    let routes: &[(&str, &str, &[u16])] = &[
        ("GET", "/v1/workshop/jobs/{}", &[404]), ("DELETE", "/v1/workshop/jobs/{}", &[404, 409]),
        ("GET", "/api/jobs/{}", &[404]), ("GET", "/v1/media/jobs/{}", &[404]), ("GET", "/v1/media/jobs/{}/content", &[404]),
        ("GET", "/v1/media/jobs/{}/captions.srt", &[404]), ("GET", "/api/models/{}/removal", &[404]), ("DELETE", "/api/models/{}", &[404]),
        ("DELETE", "/api/memory/{}", &[404]), ("DELETE", "/v1/memory/{}", &[404]), ("DELETE", "/api/mcp/servers/{}", &[404, 400]),
        ("GET", "/api/monitor/media/{}", &[404]), ("DELETE", "/api/cluster/workers/{}", &[404]), ("GET", "/v1/responses/{}", &[404]),
    ];
    for (method, route, with_key) in routes {
        for id in ids_ {
            let path = route.replace("{}", id);
            for key in [None, Some(WRONG_KEY)] {
                let a = Req::new(method, &s.base, &path).key(key).send();
                assert_eq!(a.status, 401, "{method} {path} for a stranger (key {key:?}): {}", a.text);
            }
            let a = Req::new(method, &s.base, &path).key(Some(KEY)).send();
            assert!(with_key.contains(&a.status), "{method} {path} with the key: {} {}", a.status, a.text);
        }
    }
    // the owner-only ones among them stay closed to a client profile's key, whatever the id
    for (method, path) in [("GET", "/v1/workshop/jobs/abc"), ("DELETE", "/v1/workshop/jobs/1"), ("DELETE", "/api/models/abc"), ("DELETE", "/api/memory/abc")] {
        assert_eq!(Req::new(method, &s.base, path).key(Some(CLIENT_KEY)).send().status, 401, "{method} {path} with a client key");
    }
}

/// PR #13 (commit dfdd000): a Workshop job runs in its own process group, and a cancel stops the whole group: the
/// command AND whatever it started. Nothing is left running.
#[cfg(unix)]
#[test]
fn pr13_a_cancelled_job_leaves_no_orphaned_processes() {
    let tmp = TempDir::new("orphans");
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({"env": {"AM2STRATA_WORKSHOP_EXE": fake_workshop(&tmp.0)}}));
    let slow = hf_model(&s, "slow-model");
    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": slow}));
    assert_eq!(code, 202, "{j}");
    let id = j["id"].as_u64().unwrap();
    // the job's own child is running
    let pid_file = tmp.0.join("grandchild.pid");
    let (job_pid, grandchild): (i32, i32) = wait_for("the job to start its child", Duration::from_secs(20), || {
        let t = std::fs::read_to_string(&pid_file).ok()?;
        let mut it = t.split_whitespace().map(|x| x.parse::<i32>().ok());
        Some((it.next()??, it.next()??))
    });
    let alive = |pid: i32| unsafe { libc::kill(pid, 0) } == 0;
    assert!(alive(grandchild) && alive(job_pid), "the job {job_pid} and its child {grandchild} run");
    wait_for("the job to be running", Duration::from_secs(10), || (s.get(&format!("/v1/workshop/jobs/{id}"), true).1["state"] == "running").then_some(()));
    let (code, c) = s.delete(&format!("/v1/workshop/jobs/{id}"));
    assert_eq!((code, c["state"].as_str()), (200, Some("cancelled")), "{c}");
    // both are gone (the cancel asks first, and kills after 3 s)
    wait_for("the cancelled job and its child to stop", Duration::from_secs(15), || (!alive(grandchild) && !alive(job_pid)).then_some(()));
    assert_eq!(s.get(&format!("/v1/workshop/jobs/{id}"), true).1["state"], "cancelled", "a cancelled job stays cancelled");
    assert!(!s.dir.join("workshop/slow-model-f16.gguf").exists(), "a cancelled job wrote no output");
    assert_eq!(s.delete(&format!("/v1/workshop/jobs/{id}")).0, 409, "it cannot be cancelled twice");
}

// =====================================================================================================================
// PR #12, #17 — HTTPS
// =====================================================================================================================

/// PR #12 (commit c9262b0, Copilot and Kilo): the HTTP -> HTTPS redirect names the request's Host, so the host check
/// runs first. A forged Host gets 421 and no Location (no open redirect); the server's own name is redirected to
/// itself, path and query kept.
#[test]
fn pr12_the_https_redirect_is_never_an_open_redirect() {
    let port = free_port();
    let s = Server::start(&[Model { extra: json!({"tls": {"https_port": port, "http": "redirect"}}), ..Model::default() }], &[ids("hi")], json!({}));
    https_client(&s.dir, port);
    for host in ["evil.example", "evil.example:443", "evil.example:8000", "localhost.evil.example", "127.0.0.1.evil.example", "evil.example#@localhost"] {
        let Some((status, head)) = raw_http(s.port, &format!("GET /v1/models?x=1 HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n")) else { continue };
        assert!(status == 421 || status == 400, "Host {host:?}: {status}\n{head}");
        assert!(!head.contains("location:"), "Host {host:?} got a redirect:\n{head}");
    }
    let (status, head) = raw_http(s.port, &format!("GET /v1/models?x=1 HTTP/1.1\r\nHost: localhost:{}\r\nConnection: close\r\n\r\n", s.port)).expect("an answer");
    assert_eq!(status, 308, "{head}");
    assert!(head.contains(&format!("location: https://localhost:{port}/v1/models?x=1")), "{head}");
    // a path that would read as another host after "https://<host>" is still this host's path
    let (status, head) = raw_http(s.port, &format!("GET //evil.example/x HTTP/1.1\r\nHost: localhost:{}\r\nConnection: close\r\n\r\n", s.port)).expect("an answer");
    assert!(status != 308 || head.contains(&format!("location: https://localhost:{port}/")), "{status}\n{head}");
}

/// PR #12 (commit c9262b0): `strip_port` never panics on a malformed Host. Each of these is refused or answered,
/// never a 500 (a panic is answered as one) and never a redirect to it; the server keeps serving.
#[test]
fn pr12_a_malformed_host_never_panics_the_server() {
    let port = free_port();
    let s = Server::start(&[Model { extra: json!({"tls": {"https_port": port, "http": "redirect"}}), ..Model::default() }], &[ids("hi")], json!({}));
    https_client(&s.dir, port);
    let plain = Server::start(&[Model::default()], &[ids("hi")], json!({}));
    for host in ["[fd7a::1", "[", "]", "[]", "[]:", ":", "::", ":80", "a:b:c", "[::1]:x", "[::1]x:80", "localhost:", "localhost:99999999999999999999",
                 "localhost:-1", "[::1", "::1]", "é:1", "a]b[c:1", "%", " ", "[[::1]]:80"] {
        // (/health is never redirected, so the redirecting server is asked for a path that is)
        for (srv, what, path) in [(&s, "redirecting", "/v1/models"), (&plain, "plain", "/health")] {
            let got = raw_http(srv.port, &format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"));
            // hyper may refuse a Host it cannot parse before am2Strata sees it (400), or close: both are fine
            if let Some((status, head)) = got {
                assert!(status < 500, "Host {host:?} on the {what} server: {status}\n{head}");
                // a redirect, if any, goes to this PC only: localhost or an IP literal, never a name the Host carried in
                let target = head.lines().find_map(|l| l.strip_prefix("location: https://")).map(|t| t.split('/').next().unwrap_or("").to_string());
                assert!(target.as_deref().is_none_or(|t| t.starts_with("localhost") || t.chars().all(|c| c.is_ascii_hexdigit() || ":.[]".contains(c))),
                        "Host {host:?} on the {what} server was redirected to {target:?}:\n{head}");
            }
        }
    }
    assert_eq!(Req::new("GET", &s.base, "/health").send().status, 200, "still serving after every malformed Host");
    assert_eq!(plain.get("/health", false).0, 200);
}

/// PR #12 and PR #17 (commit ee6e717): HSTS is sent only over TLS, never over plain HTTP; with plain HTTP kept (the
/// default) HTTPS sends max-age=0 (no pin, and an old pin cleared); only "http": "redirect" pins.
#[test]
fn pr12_pr17_hsts_only_over_tls_and_only_pinned_when_http_redirects() {
    let (keep_port, redirect_port) = (free_port(), free_port());
    let keep = Server::start(&[Model { extra: json!({"tls": {"https_port": keep_port}}), ..Model::default() }], &[ids("hi")], json!({}));
    let c = https_client(&keep.dir, keep_port);
    let hsts = |r: &reqwest::blocking::Response| r.headers().get("strict-transport-security").map(|v| v.to_str().unwrap().to_string());
    for path in ["/health", "/v1/models", "/", "/nope"] {
        let r = c.get(format!("https://localhost:{keep_port}{path}")).send().unwrap();
        assert_eq!(hsts(&r).as_deref(), Some("max-age=0"), "HTTPS {path} with plain HTTP kept");
        let r = Req::new("GET", &keep.base, path).send();
        assert!(r.headers.get("strict-transport-security").is_none(), "plain HTTP {path} carries no HSTS");
        assert_ne!(r.status, 308, "PR #17: plain HTTP keeps answering beside HTTPS ({path})");
    }
    let redirect = Server::start(&[Model { extra: json!({"tls": {"https_port": redirect_port, "http": "redirect"}}), ..Model::default() }], &[ids("hi")], json!({}));
    let c = https_client(&redirect.dir, redirect_port);
    let r = c.get(format!("https://localhost:{redirect_port}/health")).send().unwrap();
    assert_eq!(hsts(&r).as_deref(), Some("max-age=31536000"), "HTTPS with plain HTTP redirected");
    let r = Req::new("GET", &redirect.base, "/v1/models").header("host", &format!("localhost:{}", redirect.port)).send();
    assert_eq!(r.status, 308);
    assert!(r.headers.get("strict-transport-security").is_none(), "the plain-HTTP redirect itself carries no HSTS");
    // and a server without HTTPS never sends it
    let off = Server::start(&[Model::default()], &[ids("hi")], json!({}));
    assert!(Req::new("GET", &off.base, "/health").send().headers.get("strict-transport-security").is_none());
}

/// PR #12 (commit c9262b0): a certificate upload that fails leaves nothing behind: no temp file, HTTPS as it was.
#[test]
fn pr12_a_refused_certificate_leaves_no_file_and_https_off() {
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({}));
    let files = |dir: &std::path::Path| -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        v.sort();
        v
    };
    let before = files(&s.dir.join("server-tls"));
    for body in [json!({"action": "load", "cert_pem": "not a certificate", "key_pem": "not a key"}),
                 json!({"action": "load", "cert_pem": "-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n", "key_pem": "-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n"}),
                 json!({"action": "load"}), json!({"action": "load", "cert_pem": 5, "key_pem": []})] {
        let (code, e) = s.post("/api/tls", &body);
        assert_eq!(code, 400, "{e}");
    }
    assert_eq!(files(&s.dir.join("server-tls")), before, "a refused upload left a file");
    assert!(!files(&s.dir).iter().any(|f| f.contains(".tmp") || f.ends_with(".pem")), "{:?}", files(&s.dir));
    assert_eq!(s.get("/api/tls", true).1["on"], false, "HTTPS stays off");
}

// =====================================================================================================================
// PR #15 — the Computer tab
// =====================================================================================================================

fn ws(s: &Server, origin: Option<&str>, key_header: Option<&str>)
      -> Result<tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>, tungstenite::Error> {
    use tungstenite::client::IntoClientRequest;
    let mut req = format!("ws://127.0.0.1:{}/api/computer/stream", s.port).into_client_request().unwrap();
    if let Some(o) = origin {
        req.headers_mut().insert("origin", o.parse().unwrap());
    }
    if let Some(k) = key_header {
        req.headers_mut().insert("authorization", format!("Bearer {k}").parse().unwrap());
    }
    tungstenite::connect(req).map(|(w, _)| w)
}

/// The next JSON message, or None when the socket closed (frames, which are binary, are counted in `frames`).
fn next_json(w: &mut tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>, frames: &mut usize) -> Option<Value> {
    loop {
        match w.read() {
            Ok(tungstenite::Message::Text(t)) => return serde_json::from_str(&t).ok(),
            Ok(tungstenite::Message::Binary(_)) => *frames += 1,
            Ok(tungstenite::Message::Close(_)) | Err(_) => return None,
            Ok(_) => {}
        }
    }
}

/// PR #15 (commit 8ac4ad9): the desktop stream is a WebSocket, and "a plain request gets 401 first": without the
/// key 401, with the key 426 (use a WebSocket). A socket from another site's page is refused before it opens, and one
/// from the server's own page gets nothing until its first message carries the right key: a wrong key, or anything
/// that is not the key, is told so once and sees no frame.
#[test]
fn pr15_the_desktop_stream_needs_the_key_before_anything() {
    let s = Server::start(&[client_model(json!({"computer": {"use": true, "mode": "sandbox", "apps": ["true"]}}))], &[ids("hi")], json!({}));
    assert_eq!(get(&s, "/api/computer/stream", None).status, 401, "a plain request without the key");
    assert_eq!(get(&s, "/api/computer/stream", Some(WRONG_KEY)).status, 401);
    assert_eq!(get(&s, "/api/computer/stream", Some(CLIENT_KEY)).status, 401, "a client profile's key is not the owner's");
    assert_eq!(get(&s, "/api/computer/stream", Some(KEY)).status, 426, "a plain request with the key: use a WebSocket");
    // another site's page: refused at the handshake, even holding the key
    for key in [None, Some(KEY)] {
        match ws(&s, Some(FOREIGN_ORIGIN), key) {
            Err(tungstenite::Error::Http(r)) => assert_eq!(r.status(), 403),
            other => panic!("a foreign origin must be refused: {:?}", other.map(|_| ())),
        }
    }
    // the server's own page, each wrong first message: an error naming the key, no state, no frame
    for first in [json!({"type": "auth", "key": "wrong"}), json!({"type": "auth", "key": CLIENT_KEY}), json!({"type": "auth"}),
                  json!({"type": "take"}), json!({"type": "key", "key": KEY}), json!([KEY]), json!("auth")] {
        let mut w = ws(&s, Some(&s.base), None).expect("the server's own page may open the socket");
        w.send(tungstenite::Message::Text(first.to_string().into())).unwrap();
        let mut frames = 0;
        let m = next_json(&mut w, &mut frames);
        assert_eq!(frames, 0, "a frame was sent before the key ({first})");
        if let Some(m) = m {
            assert_eq!(m["type"], "error", "first message {first}: {m}");
            assert!(next_json(&mut w, &mut frames).is_none_or(|n| n["type"] == "error") && frames == 0, "nothing follows a refused key ({first})");
        }
    }
    assert_eq!(s.get("/api/computer/status", true).1["operator_in_control"], false, "no refused socket took control");
}

/// PR #21: the live channel is a WebSocket held to the same rule as the desktop stream: a plain request gets 401
/// without a key and 426 with one, and another site's page is refused at the handshake.
#[test]
fn pr21_the_live_channel_needs_the_key_before_anything() {
    let s = Server::start(&[client_model(json!({}))], &[ids("hi")], json!({}));
    assert_eq!(get(&s, "/v1/live", None).status, 401);
    assert_eq!(get(&s, "/v1/live", Some(WRONG_KEY)).status, 401);
    assert_eq!(get(&s, "/v1/live", Some(KEY)).status, 426);
    assert_eq!(get(&s, "/v1/live", Some(CLIENT_KEY)).status, 426, "a client key may open it (as a client)");
    use tungstenite::client::IntoClientRequest;
    for key in [None, Some(KEY)] {
        let mut req = format!("ws://127.0.0.1:{}/v1/live", s.port).into_client_request().unwrap();
        req.headers_mut().insert("origin", FOREIGN_ORIGIN.parse().unwrap());
        if let Some(k) = key {
            req.headers_mut().insert("authorization", format!("Bearer {k}").parse().unwrap());
        }
        match tungstenite::connect(req) {
            Err(tungstenite::Error::Http(r)) => assert_eq!(r.status(), 403),
            other => panic!("a foreign origin must be refused: {:?}", other.map(|_| ())),
        }
    }
}

/// PR #13 and PR #15: a client profile's key opens the model APIs and nothing of the owner's: not the Workshop
/// (model files), not the Computer tab (the desktop), not settings, and it may not ask for the owner's priority.
#[test]
fn pr13_pr15_a_client_key_is_refused_on_every_owner_route() {
    let s = Server::start(&[client_model(json!({"computer": {"use": true, "mode": "sandbox", "apps": ["true"]}}))], &[ids("hi")], json!({}));
    assert_eq!(get(&s, "/v1/models", Some(CLIENT_KEY)).status, 200, "the client key is a real key");
    for (method, path, body) in [
        ("GET", "/v1/workshop/jobs", None), ("GET", "/v1/workshop/files", None), ("POST", "/v1/workshop/jobs", Some(json!({"kind": "convert", "input": "/x"}))),
        ("POST", "/v1/workshop/uploads?name=x.txt", Some(json!("x"))), ("GET", "/v1/workshop/dataset?path=x.jsonl", None),
        ("POST", "/v1/workshop/dataset", Some(json!({"path": "x.jsonl"}))),
        ("GET", "/api/computer/status", None), ("GET", "/api/computer/screenshot", None), ("GET", "/api/computer/clipboard", None),
        ("POST", "/api/computer/control", Some(json!({"action": "take"}))), ("POST", "/api/computer/clipboard", Some(json!({"text": "x"}))),
        ("POST", "/api/computer/type", Some(json!({"text": "x"}))), ("POST", "/api/computer/open", Some(json!({"app": "true"}))),
        ("GET", "/settings", None), ("POST", "/settings", Some(json!({"defaults": null}))), ("GET", "/config", None),
        ("POST", "/config", Some(json!({"set": {"api_monitor": true}}))), ("GET", "/api/clients", None), ("GET", "/api/tls", None),
        ("POST", "/api/tls", Some(json!({"action": "off"}))), ("POST", "/api/models/unload", None), ("POST", "/unload", None),
        ("GET", "/api/mcp/servers", None), ("POST", "/api/run", Some(json!({"code": "print(1)"}))), ("GET", "/metrics", None),
        ("GET", "/api/keys", None), ("POST", "/api/keys", Some(json!({"name": "mine"}))), ("GET", "/api/usage", None),
        #[cfg(target_os = "linux")]
        ("GET", "/api/vm", None),
        #[cfg(target_os = "linux")]
        ("POST", "/api/vm", Some(json!({"action": "start"}))),
        #[cfg(target_os = "linux")]
        ("POST", "/api/vm/exec", Some(json!({"command": "id"}))),
    ] {
        let mut q = Req::new(method, &s.base, path).key(Some(CLIENT_KEY));
        if let Some(b) = &body {
            q = q.json(b);
        }
        let a = q.send();
        assert_eq!(a.status, 401, "{method} {path} with a client key: {}", a.text);
        // and as the x-api-key header, the other way a client sends it
        let mut q = Req::new(method, &s.base, path).header("x-api-key", CLIENT_KEY);
        if let Some(b) = &body {
            q = q.json(b);
        }
        assert_eq!(q.send().status, 401, "{method} {path} with a client key in x-api-key");
    }
    assert_eq!(s.get("/api/computer/status", true).1["operator_in_control"], false, "the client took no control");
    assert_eq!(s.get("/api/models/current", true).1["loaded"], true, "the client unloaded nothing");
    // the owner's key next to a client's in the other header does not turn the client into the owner by confusion:
    // either the request is the owner's (200) or it is refused, never half of each
    let a = Req::new("GET", &s.base, "/settings").key(Some(CLIENT_KEY)).header("x-api-key", WRONG_KEY).send();
    assert_eq!(a.status, 401);
}

// =====================================================================================================================
// PR #18, #19 — the sandbox VM, API keys
// =====================================================================================================================

/// PR #19: a key made at run time is a client profile's key: it opens the model APIs and nothing of the owner's (not
/// the keys themselves, not the VM, not settings); it is told once and never listed; revoked, it opens nothing at all.
#[test]
fn pr19_a_generated_key_is_a_client_key_and_a_revoked_one_opens_nothing() {
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({}));
    let (code, made) = s.post("/api/keys", &json!({"name": "regress"}));
    assert_eq!(code, 201, "{made}");
    let (key, id) = (made["key"].as_str().unwrap().to_string(), made["id"].as_str().unwrap().to_string());
    assert_eq!(get(&s, "/v1/models", Some(&key)).status, 200, "the new key opens the model APIs");
    assert_eq!(get(&s, "/v1/usage", Some(&key)).status, 200, "and its own usage");
    for path in ["/api/keys", "/api/usage", "/settings", "/config", "/v1/workshop/files", "/api/computer/status", "/api/clients", "/api/vm"] {
        assert_eq!(get(&s, path, Some(&key)).status, 401, "{path} with a generated key");
    }
    for (method, path) in [("POST", "/api/keys"), ("POST", &*format!("/api/keys/{id}")), ("DELETE", &*format!("/api/keys/{id}"))] {
        let a = Req::new(method, &s.base, path).key(Some(&key)).json(&json!({"name": "more"})).send();
        assert_eq!(a.status, 401, "{method} {path}: a key may not make, change or revoke keys (itself included)");
    }
    // the key is told once: no listing holds it, nor anything it could be rebuilt from
    let list = get(&s, "/api/keys", Some(KEY));
    assert_eq!(list.status, 200);
    assert!(!list.text.contains(&key) && !list.text.contains(&key[8..]) && !list.text.contains("\"hash\""), "the listing holds a key or its hash");
    let on_disk = std::fs::read_to_string(s.dir.join("api-keys.json")).unwrap_or_default();
    assert!(!on_disk.contains(&key[8..]), "the key itself is not kept on disk");
    // revoked: refused everywhere, at once
    let (code, r) = s.delete(&format!("/api/keys/{id}"));
    assert_eq!(code, 200, "{r}");
    for path in ["/v1/models", "/v1/usage", "/api/keys"] {
        assert_eq!(get(&s, path, Some(&key)).status, 401, "{path} with a revoked key");
    }
    let a = Req::new("POST", &s.base, "/v1/chat/completions").key(Some(&key)).json(&json!({"messages": [{"role": "user", "content": "hi"}]})).send();
    assert_eq!(a.status, 401, "a chat with a revoked key: {}", a.text);
    assert_eq!(get(&s, "/v1/models", Some(KEY)).status, 200, "the owner's key is untouched");
}

// =====================================================================================================================
// Earlier fixes (before pull requests were used: commits on main)
// =====================================================================================================================

/// Commit cf65b1a ("Security: DNS-rebinding guard…"): a request for a Host this server does not answer to is refused
/// with 421 whoever sends it, so a web page that points its own domain at this PC is never "this server's own page".
#[test]
fn security_a_rebound_host_is_refused_before_anything_else() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    for (path, key) in [("/health", None), ("/v1/models", Some(KEY)), ("/", None), ("/settings", Some(KEY))] {
        let a = Req::new("GET", &s.base, path).key(key).header("host", "evil.example:8000").header("origin", "http://evil.example:8000").send();
        assert_eq!(a.status, 421, "{path}");
    }
    // a state change from a rebound page, with the key and an Origin that matches its forged Host
    let a = Req::new("POST", &s.base, "/api/models/unload").key(Some(KEY)).header("host", "evil.example").header("origin", "http://evil.example").send();
    assert_eq!(a.status, 421);
    assert_eq!(s.get("/api/models/current", true).1["loaded"], true);
    for host in ["localhost:1", "127.0.0.1:1", "[::1]:1", "app.localhost"] {
        assert_eq!(Req::new("GET", &s.base, "/health").header("host", host).send().status, 200, "{host} is this PC");
    }
}

/// Numbers a request names are clamped to what the server allows, never trusted and never an error or a huge
/// allocation: page sizes (PR #16's dataset reader, the Monitor), and a job's time budget. The config's
/// `computer.max_steps` is clamped the same way (commit dae5323; unit test
/// `computer::settings::tests::max_steps_is_clamped_not_truncated`).
#[test]
fn numbers_from_requests_are_clamped_not_trusted() {
    let s = Server::start(&[Model { extra: json!({"computer": {"use": false, "max_steps": 5_000_000_000u64}}), ..Model::default() }], &[ids("hi")], json!({}));
    assert_eq!(s.get("/api/computer/status", true).0, 200, "a max_steps over u32 starts the server (clamped)");
    let datasets = s.dir.join("workshop/datasets");
    std::fs::create_dir_all(&datasets).unwrap();
    let line = "{\"messages\": [{\"role\": \"user\", \"content\": \"q\"}, {\"role\": \"assistant\", \"content\": \"a\"}]}\n";
    std::fs::write(datasets.join("big.jsonl"), line.repeat(1200)).unwrap();
    let page = |q: &str| {
        let a = get(&s, &format!("/v1/workshop/dataset?path={}{q}", datasets.join("big.jsonl").display().to_string().replace('/', "%2F")), Some(KEY));
        assert_eq!(a.status, 200, "{q}: {}", a.text);
        a.json()
    };
    assert_eq!(page("&limit=999999999999")["records"].as_array().unwrap().len(), 500, "a page is at most 500 records");
    assert_eq!(page("&limit=18446744073709551615")["records"].as_array().unwrap().len(), 500);
    assert_eq!(page("&limit=0")["records"].as_array().unwrap().len(), 1, "and at least one");
    assert_eq!(page("&limit=-5")["records"].as_array().unwrap().len(), 50, "a number that is none is the default");
    assert_eq!(page("&offset=18446744073709551615&limit=10")["records"].as_array().unwrap().len(), 0, "an offset past the end is an empty page");
    assert_eq!(page("")["total"], 1200);
    for q in ["?page=0&per=0", "?page=18446744073709551615&per=18446744073709551615", "?page=-1&per=x", "?requests=all", "?per=99999999"] {
        let a = get(&s, &format!("/metrics{q}"), Some(KEY));
        assert_eq!(a.status, 200, "/metrics{q}: {}", a.text);
    }
    // usage counts in an answer are what was generated, whatever the request asked for
    let (code, r) = s.post("/v1/chat/completions", &json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}));
    assert_eq!(code, 200, "{r}");
    assert!(r["usage"]["completion_tokens"].as_u64().unwrap() <= 1, "{r}");
    for bad in [json!(-1), json!(1e30), json!("many"), json!(18446744073709551615u64)] {
        let (code, r) = s.post("/v1/chat/completions", &json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": bad}));
        assert!(code == 400 || code == 200, "max_tokens {bad}: {code} {r}");
    }
}

/// PR #14 (commits 87d401e, c978932): tests never change this process's environment (`std::env::set_var` while
/// other test threads read it is undefined behaviour, and was the flaky agents test); a server is given its
/// environment through the harness (`extra.env`).
#[test]
fn pr14_no_test_changes_the_process_environment() {
    let needles = [format!("{}_var(", "set"), format!("{}_var(", "remove")];
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    let mut stack = vec![root.join("tests"), root.join("src"), root.join("../strata-workshop/src"), root.join("../strata-workshop/tests")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.file_name().and_then(|n| n.to_str()) != Some("fixtures") {
                    stack.push(p);
                }
            } else if p.extension().and_then(|x| x.to_str()) == Some("rs") {
                files.push(p);
            }
        }
    }
    assert!(files.len() > 40, "only {} source files found", files.len());
    let mut found = Vec::new();
    for f in files {
        for (n, line) in std::fs::read_to_string(&f).unwrap_or_default().lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            if needles.iter().any(|needle| code.contains(needle.as_str())) {
                found.push(format!("{}:{}: {}", f.strip_prefix(&root).unwrap_or(&f).display(), n + 1, line.trim()));
            }
        }
    }
    assert!(found.is_empty(), "the process environment is changed (pass it to the server with the harness's extra.env instead):\n  {}", found.join("\n  "));
}

/// PR #15 (commit c3bbfd4): tests never write into the repository (a test rewrote a committed fixture while another
/// read it). The golden shapes are compared, not written; `UPDATE_GOLDEN` writes to a temp folder only.
#[test]
fn pr15_the_golden_files_are_not_rewritten_by_a_test_run() {
    let shapes = include_str!("regression_shapes.rs");
    let writes: Vec<&str> = shapes.lines().filter(|l| l.contains("fs::write(") || l.contains("File::create(")).collect();
    assert_eq!(writes.len(), 1, "regression_shapes.rs writes in one place only (the UPDATE_GOLDEN temp folder): {writes:?}");
    assert!(shapes.contains("GOLDEN_OUT is inside the repository"), "the guard that refuses an output folder inside the repository is gone");
}

/// Found by this suite: POST /v1/chat/completions answered 500 (a handler panic) for a JSON body that is not an
/// object, or whose messages are not objects (media_jobs::expand_audio indexed them for writing); and, with shared
/// defaults set in Settings, /v1/chat/completions and /v1/messages did the same in settings::with_shared.
#[test]
fn a_json_body_that_is_no_object_is_a_400_not_a_panic() {
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({}));
    let paths = ["/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/messages/count_tokens"];
    let bodies = [json!([]), json!("x"), json!(7), json!(true), json!(null), json!([{"role": "user", "content": "hi"}]),
                  json!({"messages": [5, "x", null, [], {"role": "user", "content": [5, "x", null, {"type": "input_audio"}]}]})];
    for round in ["no shared defaults", "shared defaults set"] {
        for path in paths {
            for b in &bodies {
                let a = Req::new("POST", &s.base, path).key(Some(KEY)).json(b).send();
                assert!((400..500).contains(&a.status), "{path} with {b} ({round}): {} {}", a.status, a.text);
                assert!(a.json()["error"]["message"].is_string(), "{path} with {b} ({round}): {}", a.text);
            }
        }
        let (code, r) = s.post("/settings", &json!({"defaults": {"max_tokens": 7, "reasoning_effort": "low"}}));
        assert_eq!(code, 200, "{r}");
    }
    let (code, r) = s.post("/v1/chat/completions", &json!({"messages": [{"role": "user", "content": "hi"}]}));
    assert_eq!(code, 200, "a good request still works: {r}");
}

/// Found by this suite: a body axum could not read (broken JSON, no JSON content type) was refused in plain text,
/// and before the key was looked at. Now a stranger gets 401 whatever the body, and the owner gets the refusal as JSON
/// (`error.message`), on every route that takes a body.
#[test]
fn an_unreadable_body_is_401_for_a_stranger_and_a_json_400_for_the_owner() {
    let s = Server::start(&[client_model(json!({}))], &[ids("hi")], json!({}));
    for path in ["/v1/chat/completions", "/v1/messages", "/v1/responses", "/settings", "/config", "/v1/workshop/jobs", "/api/tls",
                 "/api/computer/control", "/api/models/load", "/api/memory"] {
        for (ct, body) in [("application/json", &b"{\"broken\": "[..]), ("text/plain", &b"hello"[..]), ("application/json", &b""[..])] {
            for key in [None, Some(WRONG_KEY)] {
                let a = Req::new("POST", &s.base, path).key(key).raw(ct, body).send();
                assert_eq!(a.status, 401, "{path} ({ct}) for a stranger: {}", a.text);
                assert_eq!(a.json()["error"]["type"], "authentication_error", "{}", a.text);
            }
            let a = Req::new("POST", &s.base, path).key(Some(KEY)).raw(ct, body).send();
            assert!(matches!(a.status, 400 | 415 | 422), "{path} ({ct}) for the owner: {} {}", a.status, a.text);
            assert_eq!(a.json()["error"]["type"], "invalid_request_error", "{path} ({ct}): {}", a.text);
            assert!(a.headers.get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|c| c.starts_with("application/json")), "{path}: a JSON answer");
        }
    }
    // a client profile's key is a key: on a model API it gets the JSON 400, on an owner route the 401
    assert_eq!(Req::new("POST", &s.base, "/v1/chat/completions").key(Some(CLIENT_KEY)).raw("application/json", b"{").send().status, 400);
    assert_eq!(Req::new("POST", &s.base, "/settings").key(Some(CLIENT_KEY)).raw("application/json", b"{\"defaults\": null}").send().status, 401);
    // the two routes whose proof is not a key header keep their own answers: a pairing code, and Gemini's ?key=
    let a = Req::new("POST", &s.base, "/api/cluster/join").raw("application/json", b"{").send();
    assert_eq!(a.status, 400, "{}", a.text);
    let a = Req::new("POST", &s.base, &format!("/v1beta/models/x:generateContent?key={KEY}")).raw("application/json", b"{").send();
    assert_eq!((a.status, a.json()["error"]["status"].as_str()), (400, Some("INVALID_ARGUMENT")), "{}", a.text);
}

/// Found by this suite: seven state-changing owner routes did not refuse another site's page (a foreign `Origin`):
/// loading a model, measuring the disk, starting and switching a tuning, downloading and choosing a media model, and
/// cancelling a Workshop job. On a server without an API key, a web page open in the owner's browser could call the
/// ones that need no JSON body. (The table in regression_api.rs holds every route to this; these are the seven by name.)
#[test]
fn state_changing_owner_routes_refuse_another_sites_page() {
    let s = Server::start(&[Model::default(), Model { id: "mock-b", ..Model::default() }], &[ids("hi")], json!({}));
    for (method, path, body) in [
        ("POST", "/api/models/load", Some(json!({"model": "mock-b"}))), ("POST", "/api/machine/measure", None),
        ("POST", "/api/models/mock-a/optimise", None), ("POST", "/api/models/mock-a/tuned", Some(json!({"enabled": true}))),
        ("POST", "/api/media/models/pull", Some(json!({"model": "z-image-turbo:q8"}))), ("POST", "/api/media/models/default", Some(json!({"model": "x"}))),
        ("DELETE", "/v1/workshop/jobs/1", None),
    ] {
        let mut q = Req::new(method, &s.base, path).key(Some(KEY)).header("origin", FOREIGN_ORIGIN);
        if let Some(b) = &body {
            q = q.json(b);
        }
        let a = q.send();
        assert_eq!(a.status, 403, "{method} {path} from another site's page: {}", a.text);
        assert_eq!(a.json()["error"]["type"], "permission_error");
    }
    let (_, cur) = s.get("/api/models/current", true);
    assert_eq!(cur["model"]["id"], "mock-a", "the refused load changed nothing: {cur}");
    assert_eq!(s.get("/api/optimise", true).1["state"], "idle", "no tuning was started");
    // the server's own page (its Origin is its address) is let through
    let a = Req::new("POST", &s.base, "/api/models/mock-a/tuned").key(Some(KEY)).header("origin", &s.base).json(&json!({"enabled": false})).send();
    assert_ne!(a.status, 403, "{}", a.text);
}

// =====================================================================================================================
// Fixes whose regression tests live beside the code (referenced here, not repeated)
// =====================================================================================================================

/// The byte-for-byte gates against llama.cpp's own output (PR #13 and the commits before it) already exist as tests
/// in their crates. This test only makes sure nobody deletes or renames them unnoticed; run them with
/// `cargo test -p strata-workshop -p strata-pack`.
#[test]
fn the_fixture_gates_and_unit_regressions_still_exist() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    for (file, tests) in [
        // quantize: our GGUF -> GGUF quantizer against llama-quantize --pure, tensor for tensor
        ("strata-workshop/tests/quantize_pure.rs", &["fn tiny_matches_pure_reference", "fn wide_matches_pure_reference", "fn gqa_matches_llama_f32_reference"][..]),
        ("strata-pack/tests/quantize.rs", &["fn quantized_bytes_match_llama_quantize"][..]),
        // convert: HF -> GGUF against llama.cpp's convert_hf_to_gguf.py (F32, F16, and a BF16 checkpoint)
        ("strata-workshop/tests/convert_reference.rs", &["fn converted_matches_llama_convert_hf_to_gguf"][..]),
        ("strata-workshop/tests/convert_f16.rs", &["fn f16_conversion_matches_llama_cpp", "fn a_bf16_checkpoint_converts_as_llama_cpp_does"][..]),
        // PR #13 (Kilo): chunk never loops; MinHash compares hashes exactly. And this suite's own chunk fix
        ("strata-workshop/src/dataset.rs", &["fn regress_pr13_chunk_never_loops_whatever_the_overlap", "fn regress_pr13_chunk_moves_on_after_a_boundary_in_the_first_chunk",
                                              "fn regress_pr13_near_duplicates_compare_hashes_exactly", "fn regress_chunk_counts_the_whitespace_it_keeps",
                                              "fn an_overlap_as_long_as_a_chunk_still_finishes"][..]),
        // commit dae5323: computer.max_steps clamped, not truncated; PR #12: the plain-HTTP policy and HSTS values
        ("am2strata/src/computer/settings.rs", &["fn max_steps_is_clamped_not_truncated"][..]),
        ("am2strata/src/server_tls.rs", &["fn plain_http_policy", "fn hsts_value"][..]),
        // PR #16: upload names
        ("am2strata/src/workshop.rs", &["fn upload_names_are_made_safe"][..]),
        // PR #18 (commit 9812a60): the VM's RFB client refuses an oversized screen, at the handshake and at a resize
        ("am2strata/src/computer/vm/rfb.rs", &["fn sizes_from_the_guest_are_bounded", "fn regress_pr18_an_oversized_screen_is_refused_at_the_handshake",
                                               "fn regress_pr18_an_oversized_resize_or_rectangle_stops_the_reader"][..]),
        // PR #19: a key is told once and only its hash is kept
        ("am2strata/src/api_keys.rs", &["fn a_key_is_told_once_and_only_its_hash_is_kept"][..]),
    ] {
        let text = std::fs::read_to_string(root.join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
        for t in tests {
            assert!(text.contains(t), "{file} no longer has `{t}`: a regression gate was removed or renamed (update this list if it moved)");
        }
    }
}
