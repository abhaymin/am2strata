//! The live channel (GET /v1/live) end to end against the mock engine: the key (header, or the first message), the
//! page's Origin, snapshots and pushed updates, requests through the same router as HTTP (a JSON answer, a stream,
//! cancel, several at once, a client profile's limits), ping/pong, and all of it again over wss.

mod common;

use common::*;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::time::{Duration, Instant};

trait Io: Read + Write {}
impl<T: Read + Write> Io for T {}
type Ws = tungstenite::WebSocket<Box<dyn Io>>;

const LAPTOP: &str = "laptop-key-0123456789abcdef";
const PHONE: &str = "phone-key-0123456789abcdef0";

fn user(t: &str) -> Value {
    json!({"role": "user", "content": t})
}

fn request(url: &str, key: Option<&str>, origin: Option<&str>) -> tungstenite::handshake::client::Request {
    use tungstenite::client::IntoClientRequest;
    let mut req = url.into_client_request().unwrap();
    if let Some(k) = key {
        req.headers_mut().insert("authorization", format!("Bearer {k}").parse().unwrap());
    }
    if let Some(o) = origin {
        req.headers_mut().insert("origin", o.parse().unwrap());
    }
    req
}

fn tcp(port: u16) -> std::net::TcpStream {
    let t = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    // a test that waits for a message that never comes fails instead of hanging
    t.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    t
}

/// ws:// with the key in the header (a CLI, a node), or none.
fn open(s: &Server, key: Option<&str>, origin: Option<&str>) -> Result<Ws, tungstenite::Error> {
    let io: Box<dyn Io> = Box::new(tcp(s.port));
    tungstenite::client(request(&format!("ws://127.0.0.1:{}/v1/live", s.port), key, origin), io).map(|(w, _)| w)
        .map_err(|e| match e {
            tungstenite::HandshakeError::Failure(e) => e,
            tungstenite::HandshakeError::Interrupted(_) => panic!("handshake interrupted"),
        })
}

fn say(w: &mut Ws, v: Value) {
    w.send(tungstenite::Message::Text(v.to_string().into())).unwrap();
}

fn next(w: &mut Ws) -> Value {
    loop {
        match w.read().expect("a message") {
            tungstenite::Message::Text(t) => return serde_json::from_str(t.as_str()).unwrap(),
            tungstenite::Message::Close(c) => return json!({"type": "closed", "code": c.map(|c| u16::from(c.code))}),
            _ => continue,
        }
    }
}

/// The next message that `want` accepts (others — updates of topics, events of other requests — are skipped).
fn until(w: &mut Ws, what: &str, want: impl Fn(&Value) -> bool) -> Value {
    let t0 = Instant::now();
    loop {
        let m = next(w);
        if std::env::var_os("LIVE_TEST_TRACE").is_some() {
            eprintln!("waiting for {what}: {}", m.to_string().chars().take(300).collect::<String>());
        }
        if want(&m) {
            return m;
        }
        assert!(m["type"] != "closed", "closed while waiting for {what}");
        assert!(t0.elapsed() < Duration::from_secs(30), "no {what}; last: {m}");
    }
}

/// Connected with the server's key in the header: past `ready`.
fn owner(s: &Server) -> Ws {
    let mut w = open(s, Some(KEY), None).expect("connect");
    let m = next(&mut w);
    assert_eq!((m["type"].as_str(), m["role"].as_str()), (Some("ready"), Some("owner")), "{m}");
    w
}

/// Everything a request sends back until it ends: its events, and the message that ended it.
fn answer(w: &mut Ws, id: &str) -> (Vec<Value>, Value) {
    let mut events = Vec::new();
    loop {
        let m = until(w, "the request's answer", |m| m["id"] == id);
        match m["type"].as_str() {
            Some("event") => events.push(m),
            _ => return (events, m),
        }
    }
}

fn text_of(events: &[Value]) -> String {
    events.iter().filter_map(|e| e["data"]["choices"][0]["delta"]["content"].as_str()).collect()
}

fn chat(id: &str, q: &str, stream: bool) -> Value {
    json!({"type": "request", "id": id, "method": "POST", "path": "/v1/chat/completions",
           "body": {"messages": [user(q)], "stream": stream, "chat_template_kwargs": {"enable_thinking": false}}})
}

#[test]
fn no_key_is_refused_and_another_sites_page_too() {
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({}));
    // a plain GET: the key first, then "this is a WebSocket"
    assert_eq!(s.get("/v1/live", false).0, 401);
    assert_eq!(s.get("/v1/live", true).0, 426);
    // another site's page is refused before the socket opens, even with the key
    match open(&s, Some(KEY), Some("https://evil.example")) {
        Err(tungstenite::Error::Http(r)) => assert_eq!(r.status(), 403),
        other => panic!("a foreign origin must be refused: {:?}", other.map(|_| ())),
    }
    // a wrong key in the first message: told so, then closed with 4401; nothing else is answered
    let mut w = open(&s, None, Some(&s.base)).expect("the server's own page");
    say(&mut w, json!({"type": "auth", "key": "wrong"}));
    let m = next(&mut w);
    assert_eq!((m["type"].as_str(), m["code"].as_u64()), (Some("error"), Some(401)), "{m}");
    assert_eq!(next(&mut w), json!({"type": "closed", "code": 4401}));
    // no key, and something else than auth first: the same
    let mut w = open(&s, None, None).expect("connect");
    say(&mut w, json!({"type": "subscribe", "topics": ["metrics"]}));
    assert_eq!(next(&mut w)["type"], "error");
    assert_eq!(next(&mut w)["type"], "closed");
    // a wrong key in the header: the first message may still bring the right one (the browser's way)
    let mut w = open(&s, Some("wrong"), None).expect("connect");
    say(&mut w, json!({"type": "auth", "key": KEY}));
    let m = next(&mut w);
    assert_eq!((m["type"].as_str(), m["role"].as_str()), (Some("ready"), Some("owner")), "{m}");
    assert_eq!(m["topics"], json!(["status", "queue", "jobs", "workshop", "metrics"]));
}

#[test]
fn a_subscribe_gets_a_snapshot_and_then_what_changes() {
    let long: Vec<u32> = std::iter::repeat_n(ids("word ")[0], 150).collect();
    let s = Server::start(&[Model::default()], &[long], json!({"delay_ms": 10}));
    let mut w = owner(&s);
    say(&mut w, json!({"type": "subscribe", "topics": ["status", "queue", "jobs", "workshop", "metrics", "nope"]}));
    let m = next(&mut w);
    assert_eq!(m["type"], "error", "the unknown topic is named: {m}");
    assert!(m["message"].as_str().unwrap().contains("nope"), "{m}");
    assert_eq!(next(&mut w), json!({"type": "subscribed", "topics": ["status", "queue", "jobs", "workshop", "metrics"]}));
    let mut seen = std::collections::BTreeMap::new();
    for _ in 0..5 {
        let m = next(&mut w);
        assert_eq!((m["type"].as_str(), m["snapshot"].as_bool()), (Some("update"), Some(true)), "{m}");
        seen.insert(m["topic"].as_str().unwrap().to_string(), m["data"].clone());
    }
    let st = &seen["status"];
    assert_eq!((st["state"].as_str(), st["loaded"].as_bool(), st["model"]["id"].as_str(), st["model"]["name"].as_str()),
               (Some("idle"), Some(true), Some("mock-a"), Some("Mock mock-a")), "{st}");
    assert_eq!(st["queue"], json!({"running": null, "waiting": 0}));
    assert_eq!(seen["queue"]["running"], Value::Null);
    assert_eq!(seen["queue"]["resident"], json!({"text_model": "mock-a"}));
    assert_eq!(seen["jobs"], json!({"jobs": []}));
    assert_eq!(seen["workshop"], json!({"jobs": []}));
    // the metrics topic is what GET /metrics answers
    let (_, http) = s.get("/metrics", true);
    let pushed = &seen["metrics"];
    assert_eq!(pushed["live"]["state"], "idle");
    assert_eq!(pushed["engine"], http["engine"]);
    assert_eq!(pushed.as_object().unwrap().keys().collect::<Vec<_>>(), http.as_object().unwrap().keys().collect::<Vec<_>>());

    // the Monitor's figures only from here on, to follow the others
    say(&mut w, json!({"type": "unsubscribe", "topics": ["metrics"]}));
    until(&mut w, "unsubscribed", |m| m["type"] == "unsubscribed");
    // a chat over plain HTTP from elsewhere: this connection hears the queue and the status change, unasked
    let base = s.base.clone();
    let http = std::thread::spawn(move || {
        reqwest::blocking::Client::new().post(format!("{base}/v1/chat/completions")).bearer_auth(KEY)
            .json(&json!({"messages": [user("essay")], "chat_template_kwargs": {"enable_thinking": false}})).send().unwrap().status().as_u16()
    });
    let mut heard = Vec::new();
    loop {
        let m = until(&mut w, "the chat's updates", |m| m["type"] == "update");
        assert!(m.get("snapshot").is_none(), "a pushed change, not a snapshot: {m}");
        heard.push(m);
        let saw = |topic: &str, ok: &dyn Fn(&Value) -> bool| heard.iter().position(|m| m["topic"] == topic && ok(&m["data"]));
        let ran = saw("queue", &|d| d["running"]["kind"] == "chat");
        let answering = saw("status", &|d| (d["state"] == "generating" || d["state"] == "reading") && d["queue"]["running"] == "chat");
        if let (Some(ran), Some(answering)) = (ran, answering) {
            // and, after those, the queue empty and the status idle again
            let empty = heard.iter().skip(ran).any(|m| m["topic"] == "queue" && m["data"]["running"].is_null());
            let idle = heard.iter().skip(answering).any(|m| m["topic"] == "status" && m["data"]["state"] == "idle");
            if empty && idle {
                break;
            }
        }
    }
    assert_eq!(http.join().unwrap(), 200);
    assert!(heard.len() <= 12, "only changes are sent, not a stream of repeats: {} updates", heard.len());
    // idle: nothing more is sent (no clock ticks, no repeats) — only the pong asked for arrives
    std::thread::sleep(Duration::from_millis(2500));
    say(&mut w, json!({"type": "ping"}));
    assert_eq!(next(&mut w)["type"], "pong", "an idle server pushes nothing");
}

#[test]
fn a_chat_over_the_socket_answers_and_streams() {
    let s = Server::start(&[Model::default()], &[ids("Hello there."), ids("One two three.")], json!({}));
    let mut w = owner(&s);
    // a JSON answer: one response, the body as HTTP gives it
    say(&mut w, chat("a", "hi", false));
    let (events, m) = answer(&mut w, "a");
    assert!(events.is_empty());
    assert_eq!((m["type"].as_str(), m["status"].as_u64()), (Some("response"), Some(200)), "{m}");
    assert_eq!(m["body"]["choices"][0]["message"]["content"], "Hello there.");
    assert_eq!(m["headers"]["content-type"], "application/json");
    // a stream: each SSE event as an event, then done
    say(&mut w, chat("b", "count", true));
    let (events, m) = answer(&mut w, "b");
    assert_eq!(m, json!({"type": "done", "id": "b", "status": 200}));
    assert_eq!(text_of(&events), "One two three.");
    assert_eq!(events.last().unwrap()["data"], "[DONE]");
    assert!(events.iter().any(|e| e["data"]["choices"][0]["finish_reason"] == "stop"), "{events:?}");
    // it ran through the same router as HTTP: the Monitor counts both, and the same GET answers the same
    say(&mut w, json!({"type": "request", "id": 7, "path": "/v1/models"}));
    let (_, m) = answer(&mut w, "7");
    assert_eq!(m["body"], s.get("/v1/models", true).1);
    assert_eq!(s.get("/metrics", true).1["totals"]["requests"], 2);
    // an error is the HTTP error, with its status
    say(&mut w, json!({"type": "request", "id": "bad", "method": "POST", "path": "/v1/chat/completions", "body": {"messages": "not a list"}}));
    let (_, m) = answer(&mut w, "bad");
    assert_eq!((m["type"].as_str(), m["status"].as_u64()), (Some("response"), Some(400)), "{m}");
    assert!(m["body"]["error"]["message"].is_string(), "{m}");
    say(&mut w, json!({"type": "request", "id": "none", "path": "/v1/no-such-route"}));
    assert_eq!(answer(&mut w, "none").1["status"], 404);
}

#[test]
fn malformed_messages_get_a_clear_error_and_the_connection_stays() {
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({}));
    let mut w = owner(&s);
    let mut err = |v: tungstenite::Message| {
        w.send(v).unwrap();
        let m = next(&mut w);
        assert_eq!(m["type"], "error", "{m}");
        m
    };
    let text = |v: Value| tungstenite::Message::Text(v.to_string().into());
    assert!(err(tungstenite::Message::Text("{not json".into()))["message"].as_str().unwrap().contains("not JSON"));
    err(tungstenite::Message::Text("[1,2]".into()));
    err(tungstenite::Message::Binary(vec![1u8, 2, 3].into()));
    assert!(err(text(json!({"type": "dance"})))["message"].as_str().unwrap().contains("dance"));
    err(text(json!({"no": "type"})));
    err(text(json!({"type": "subscribe"})));
    err(text(json!({"type": "request", "path": "/v1/models"})));
    // only /v1/ and /api/, never the channel itself, never a header that would change who is asking
    for (path, headers) in [("/settings", json!({})), ("/config", json!({})), ("/metrics", json!({})), ("/v1/live", json!({})),
                            ("http://example.com/v1/models", json!({})), ("/v1/models", json!({"authorization": "Bearer other"})),
                            ("/v1/models", json!({"origin": "https://evil.example"}))] {
        let m = err(text(json!({"type": "request", "id": "x", "path": path, "headers": headers})));
        assert_eq!(m["id"], "x", "the error names the request: {m}");
    }
    assert_eq!(err(text(json!({"type": "request", "id": "x", "method": "TRACE", "path": "/v1/models"})))["id"], "x");
    assert_eq!(err(text(json!({"type": "cancel", "id": "never-sent"})))["id"], "never-sent");
    // and it still works
    say(&mut w, json!({"type": "request", "id": "ok", "path": "/v1/models"}));
    assert_eq!(answer(&mut w, "ok").1["status"], 200);
}

#[test]
fn cancel_stops_the_model_and_the_next_request_runs() {
    let long: Vec<u32> = std::iter::repeat_n(ids("word ")[0], 400).collect();
    let s = Server::start(&[Model::default()], &[long, ids("next")], json!({"delay_ms": 10}));
    let mut w = owner(&s);
    say(&mut w, chat("long", "essay", true));
    until(&mut w, "the first words", |m| m["id"] == "long" && m["data"]["choices"][0]["delta"]["content"].is_string());
    say(&mut w, json!({"type": "cancel", "id": "long"}));
    until(&mut w, "cancelled", |m| m == &json!({"type": "cancelled", "id": "long"}));
    let t0 = Instant::now();
    while !s.commands().iter().any(|c| c == "STOP") {
        assert!(t0.elapsed() < Duration::from_secs(3), "STOP after the cancel, as when an HTTP client leaves");
        std::thread::sleep(Duration::from_millis(50));
    }
    // the next one does not wait for the abandoned answer, and nothing more of the cancelled one arrives
    let t0 = Instant::now();
    say(&mut w, chat("after", "again", false));
    let m = until(&mut w, "the next answer", |m| {
        assert!(m["id"] != "long", "nothing of a cancelled request after its cancelled: {m}");
        m["id"] == "after"
    });
    assert_eq!(m["body"]["choices"][0]["message"]["content"], "next", "{m}");
    assert!(t0.elapsed().as_secs() < 3);
    // closing the socket mid-answer stops the model too
    let long: Vec<u32> = std::iter::repeat_n(ids("word ")[0], 400).collect();
    let s = Server::start(&[Model::default()], &[long], json!({"delay_ms": 10}));
    let mut w = owner(&s);
    say(&mut w, chat("long", "essay", true));
    until(&mut w, "the first words", |m| m["id"] == "long" && m["data"]["choices"][0]["delta"]["content"].is_string());
    drop(w);
    let t0 = Instant::now();
    while !s.commands().iter().any(|c| c == "STOP") {
        assert!(t0.elapsed() < Duration::from_secs(3), "STOP after the socket closed");
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn two_requests_in_flight_and_ping_pong() {
    let slow: Vec<u32> = std::iter::repeat_n(ids("word ")[0], 60).collect();
    let s = Server::start(&[Model::default()], &[slow, ids("second")], json!({"delay_ms": 15}));
    let mut w = owner(&s);
    say(&mut w, chat("one", "essay", true));
    until(&mut w, "the first words", |m| m["id"] == "one" && m["data"]["choices"][0]["delta"]["content"].is_string());
    // while the first streams: a second chat (it waits its turn), a GET (answered at once), a ping, a repeated id
    say(&mut w, chat("two", "and you", true));
    say(&mut w, json!({"type": "request", "id": "list", "path": "/v1/models"}));
    say(&mut w, json!({"type": "ping"}));
    say(&mut w, chat("one", "same id", false));
    let (mut one, mut two) = (Vec::new(), Vec::new());
    let (mut done, mut order) = (Vec::new(), Vec::new());
    let (mut pong, mut dup) = (false, false);
    while done.len() < 3 {
        let m = next(&mut w);
        assert!(m["type"] != "closed", "closed early");
        match (m["type"].as_str().unwrap(), m["id"].as_str().unwrap_or("")) {
            ("pong", _) => { assert!(m["t"].as_u64().unwrap() > 1_600_000_000_000, "{m}"); pong = true; }
            ("error", "one") => dup = true,
            ("event", "one") => one.push(m),
            ("event", "two") => two.push(m),
            ("done", id) | ("response", id) => { order.push(id.to_string()); done.push(m); }
            other => panic!("unexpected {other:?}: {m}"),
        }
    }
    assert!(pong && dup, "the ping is answered and the repeated id refused while requests run");
    assert_eq!(order, ["list", "one", "two"], "the GET does not wait for the chats; the chats run in turn");
    assert_eq!(text_of(&one), "word".repeat(60));
    assert_eq!(text_of(&two), "second");
    assert_eq!(done[0]["body"]["data"][0]["id"], "mock-a", "{}", done[0]);
    // the one that waited was told its place, as an SSE comment says it over HTTP
    assert!(two.iter().any(|e| e["comment"].as_str().is_some_and(|c| c.starts_with("queued:"))), "{two:?}");
}

#[test]
fn a_client_key_gets_its_chat_and_the_public_topics_only() {
    let clients = json!({"clients": {"laptop": {"key": LAPTOP, "tools": []}, "phone": {"key": PHONE, "tools": []}}});
    let long: Vec<u32> = std::iter::repeat_n(ids("word ")[0], 120).collect();
    let s = Server::start(&[Model { extra: clients, ..Model::default() }], &[ids("for the laptop"), long, ids("laptop again")], json!({"delay_ms": 10}));
    // the key in the first message, as a browser sends it
    let mut w = open(&s, None, Some(&s.base)).expect("connect");
    say(&mut w, json!({"type": "auth", "key": LAPTOP}));
    let m = next(&mut w);
    assert_eq!(m["type"], "ready", "{m}");
    assert_eq!((m["role"].as_str(), m["client"].as_str()), (Some("client"), Some("laptop")));
    assert_eq!(m["topics"], json!(["status", "queue"]));
    // its own chat
    say(&mut w, chat("c", "hi", false));
    let (_, m) = answer(&mut w, "c");
    assert_eq!(m["body"]["choices"][0]["message"]["content"], "for the laptop", "{m}");
    // owner routes: the same refusal as over HTTP, for each
    let http = |path: &str| reqwest::blocking::Client::new().get(format!("{}{path}", s.base)).bearer_auth(LAPTOP).send().unwrap().status().as_u16();
    for path in ["/v1/workshop/jobs", "/v1/workshop/files", "/api/computer/status", "/api/queue", "/api/clients", "/api/mcp/servers", "/api/tls", "/v1/status"] {
        say(&mut w, json!({"type": "request", "id": path, "path": path}));
        let (_, m) = answer(&mut w, path);
        assert_eq!(m["type"], "response", "{m}");
        assert_eq!(m["status"].as_u64().unwrap() as u16, http(path), "{path} over the socket as over HTTP");
        assert!(matches!(m["status"].as_u64(), Some(401 | 403)), "{path}: {m}");
    }
    say(&mut w, json!({"type": "request", "id": "w", "method": "POST", "path": "/v1/workshop/jobs", "body": {"kind": "quantize"}}));
    assert_eq!(answer(&mut w, "w").1["status"], 401);
    // settings and the run config are not reachable at all (outside /v1/ and /api/)
    say(&mut w, json!({"type": "request", "id": "s", "path": "/settings"}));
    assert_eq!(answer(&mut w, "s").1["type"], "error");
    // topics: the owner's are refused by name; the public ones answer, the queue without anyone else's jobs
    say(&mut w, json!({"type": "subscribe", "topics": ["metrics", "jobs", "workshop", "status", "queue"]}));
    for t in ["metrics", "jobs", "workshop"] {
        let m = next(&mut w);
        assert_eq!(m["type"], "error", "{m}");
        assert!(m["message"].as_str().unwrap().contains(t) && m["message"].as_str().unwrap().contains("own API key"), "{m}");
    }
    assert_eq!(next(&mut w), json!({"type": "subscribed", "topics": ["status", "queue"]}));
    assert_eq!(next(&mut w)["data"]["model"]["id"], "mock-a");
    assert_eq!(next(&mut w)["data"], json!({"running": null, "waiting": 0, "mine": []}));
    // the phone's chat runs, the laptop's waits behind it: the laptop sees one job running and its own waiting, no
    // label, prompt or client of the other
    let base = s.base.clone();
    let phone = std::thread::spawn(move || {
        reqwest::blocking::Client::new().post(format!("{base}/v1/chat/completions")).bearer_auth(PHONE)
            .json(&json!({"messages": [user("the phone's private question")], "chat_template_kwargs": {"enable_thinking": false}})).send().unwrap().status().as_u16()
    });
    until(&mut w, "the phone's chat running", |m| m["topic"] == "queue" && m["data"]["running"]["kind"] == "chat");
    say(&mut w, chat("mine", "and mine", false));
    let m = until(&mut w, "its own job waiting", |m| m["topic"] == "queue" && m["data"]["waiting"] == 1);
    assert_eq!(m["data"]["mine"][0]["client"], "laptop", "{m}");
    assert_eq!(m["data"]["running"], json!({"kind": "chat"}), "{m}");
    let (_, m) = answer(&mut w, "mine");
    assert_eq!(m["body"]["choices"][0]["message"]["content"], "laptop again", "{m}");
    assert_eq!(phone.join().unwrap(), 200);
    // an unknown key opens nothing
    let mut w = open(&s, Some("not-a-key-at-all-00000"), None).expect("connect");
    assert_eq!(next(&mut w)["type"], "error");
}

// --------------------------------------------------------------------------------------------------------------- wss

fn wait_https(dir: &std::path::Path, port: u16) -> Vec<u8> {
    let t0 = Instant::now();
    loop {
        if let Ok(pem) = std::fs::read(dir.join("server-tls/cert.pem")) {
            let c = reqwest::blocking::Client::builder().add_root_certificate(reqwest::Certificate::from_pem(&pem).unwrap())
                .timeout(Duration::from_secs(10)).build().unwrap();
            if c.get(format!("https://localhost:{port}/health")).send().is_ok_and(|r| r.status().is_success()) {
                return pem;
            }
        }
        assert!(t0.elapsed() < Duration::from_secs(30), "HTTPS did not come up on {port}");
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// wss:// trusting only the server's own certificate (never "accept anything").
fn open_tls(port: u16, cert_pem: &[u8], key: Option<&str>, origin: Option<&str>) -> Result<Ws, tungstenite::Error> {
    use rustls::pki_types::pem::PemObject;
    let mut roots = rustls::RootCertStore::empty();
    for c in rustls::pki_types::CertificateDer::pem_slice_iter(cert_pem) {
        roots.add(c.unwrap()).unwrap();
    }
    let cfg = rustls::ClientConfig::builder_with_provider(std::sync::Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions().unwrap().with_root_certificates(roots).with_no_client_auth();
    let conn = rustls::ClientConnection::new(std::sync::Arc::new(cfg), "localhost".try_into().unwrap()).unwrap();
    let io: Box<dyn Io> = Box::new(rustls::StreamOwned::new(conn, tcp(port)));
    tungstenite::client(request(&format!("wss://localhost:{port}/v1/live"), key, origin), io).map(|(w, _)| w)
        .map_err(|e| match e {
            tungstenite::HandshakeError::Failure(e) => e,
            tungstenite::HandshakeError::Interrupted(_) => panic!("handshake interrupted"),
        })
}

#[test]
fn the_same_over_wss() {
    let port = free_port();
    let long: Vec<u32> = std::iter::repeat_n(ids("word ")[0], 400).collect();
    let s = Server::start(&[Model { extra: json!({"tls": {"https_port": port}}), ..Model::default() }],
                          &[ids("Hello over TLS."), ids("Streamed over TLS."), long, ids("next")], json!({"delay_ms": 10}));
    let pem = wait_https(&s.dir, port);
    // another site's page, and no key: refused as over ws://
    match open_tls(port, &pem, Some(KEY), Some("https://evil.example")) {
        Err(tungstenite::Error::Http(r)) => assert_eq!(r.status(), 403),
        other => panic!("a foreign origin must be refused: {:?}", other.map(|_| ())),
    }
    let mut w = open_tls(port, &pem, None, None).expect("connect");
    say(&mut w, json!({"type": "auth", "key": "wrong"}));
    assert_eq!(next(&mut w)["type"], "error");
    assert_eq!(next(&mut w), json!({"type": "closed", "code": 4401}));
    // the page at https://localhost:<port>, the key in its first message
    let origin = format!("https://localhost:{port}");
    let mut w = open_tls(port, &pem, None, Some(&origin)).expect("the server's own page over HTTPS");
    say(&mut w, json!({"type": "auth", "key": KEY}));
    assert_eq!(next(&mut w)["role"], "owner");
    say(&mut w, json!({"type": "ping"}));
    assert_eq!(next(&mut w)["type"], "pong");
    say(&mut w, json!({"type": "subscribe", "topics": ["status"]}));
    assert_eq!(next(&mut w)["type"], "subscribed");
    let m = next(&mut w);
    assert_eq!((m["topic"].as_str(), m["snapshot"].as_bool(), m["data"]["state"].as_str()), (Some("status"), Some(true), Some("idle")), "{m}");
    say(&mut w, json!({"type": "unsubscribe", "topics": ["status"]}));
    assert_eq!(next(&mut w)["type"], "unsubscribed");
    // requests: a JSON answer, a stream, and cancel
    say(&mut w, chat("a", "hi", false));
    assert_eq!(answer(&mut w, "a").1["body"]["choices"][0]["message"]["content"], "Hello over TLS.");
    say(&mut w, chat("b", "more", true));
    let (events, m) = answer(&mut w, "b");
    assert_eq!(m["type"], "done");
    assert_eq!(text_of(&events), "Streamed over TLS.");
    say(&mut w, chat("long", "essay", true));
    until(&mut w, "the first words", |m| m["id"] == "long" && m["data"]["choices"][0]["delta"]["content"].is_string());
    say(&mut w, json!({"type": "cancel", "id": "long"}));
    until(&mut w, "cancelled", |m| m["type"] == "cancelled");
    let t0 = Instant::now();
    while !s.commands().iter().any(|c| c == "STOP") {
        assert!(t0.elapsed() < Duration::from_secs(3), "STOP after the cancel");
        std::thread::sleep(Duration::from_millis(50));
    }
    say(&mut w, chat("after", "again", false));
    assert_eq!(answer(&mut w, "after").1["body"]["choices"][0]["message"]["content"], "next");
    // a request over wss is a request over TLS to the router: HSTS policy and all (an owner route answers)
    say(&mut w, json!({"type": "request", "id": "tls", "path": "/api/tls"}));
    assert_eq!(answer(&mut w, "tls").1["body"]["on"], true);
    // plain ws:// from this PC keeps working beside it
    let mut plain = owner(&s);
    say(&mut plain, json!({"type": "ping"}));
    assert_eq!(next(&mut plain)["type"], "pong");
}

// --------------------------------------------------------------------------------------------------------------- CLI

/// `am2strata-run live`: its first `n` lines of output (each one pushed update), then it is stopped.
fn cli_lines(args: &[&str], n: usize) -> Vec<Value> {
    use std::io::BufRead;
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_am2strata-run")).arg("live").args(args)
        .env_remove("AM2STRATA_URL").env_remove("AM2STRATA_API_KEY")
        .stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null()).spawn().expect("am2strata-run");
    let out = std::io::BufReader::new(child.stdout.take().unwrap());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in out.lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let lines: Vec<Value> = (0..n).map_while(|_| rx.recv_timeout(Duration::from_secs(20)).ok()).map(|l| serde_json::from_str(&l).expect("one JSON per line")).collect();
    let _ = child.kill();
    let _ = child.wait();
    lines
}

#[test]
fn the_cli_prints_pushed_updates_over_ws_and_wss() {
    let port = free_port();
    let s = Server::start(&[Model { extra: json!({"tls": {"https_port": port}}), ..Model::default() }], &[ids("hi")], json!({}));
    wait_https(&s.dir, port);
    let lines = cli_lines(&["--url", &s.base, "--api-key", KEY, "--topics", "status,queue"], 2);
    assert_eq!(lines.len(), 2, "a snapshot of each topic");
    assert!(lines.iter().all(|l| l["type"] == "update"), "{lines:?}");
    let topics: Vec<&str> = lines.iter().filter_map(|l| l["topic"].as_str()).collect();
    assert_eq!(topics, ["status", "queue"]);
    assert_eq!(lines[0]["data"]["model"]["id"], "mock-a");
    // wss, trusting the server's own certificate
    let cert = s.dir.join("server-tls/cert.pem");
    let lines = cli_lines(&["--url", &format!("https://localhost:{port}"), "--api-key", KEY, "--topics", "status", "--cert", cert.to_str().unwrap()], 1);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["data"]["state"], "idle");
    // without that certificate a self-signed server is not trusted; a wrong key prints nothing
    assert!(cli_lines(&["--url", &format!("wss://localhost:{port}"), "--api-key", KEY], 1).is_empty());
    assert!(cli_lines(&["--url", &s.base, "--api-key", "wrong"], 1).is_empty());
}
