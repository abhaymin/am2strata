//! The Computer tab's server side end to end against the mock engine: its routes need the server's own key, the
//! operator's control is taken and handed back, input needs control, and the WebSocket refuses another site's page
//! and a wrong key in its first message. (The desktop itself needs Xvfb; on a machine without it the stream says so.)

mod common;

use common::*;
use serde_json::{json, Value};

fn ws(s: &Server, origin: Option<&str>) -> Result<tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>, tungstenite::Error> {
    use tungstenite::client::IntoClientRequest;
    let mut req = format!("ws://127.0.0.1:{}/api/computer/stream", s.port).into_client_request().unwrap();
    if let Some(o) = origin {
        req.headers_mut().insert("origin", o.parse().unwrap());
    }
    tungstenite::connect(req).map(|(w, _)| w)
}

fn next_json(w: &mut tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>) -> Value {
    loop {
        match w.read().expect("a message") {
            tungstenite::Message::Text(t) => return serde_json::from_str(&t).unwrap(),
            tungstenite::Message::Binary(_) => continue, // a frame
            tungstenite::Message::Close(_) => return json!({"type": "closed"}),
            _ => continue,
        }
    }
}

#[test]
fn the_computer_tab_needs_the_key_control_and_its_own_page() {
    let s = Server::start(&[Model { extra: json!({"computer": {"use": true, "mode": "sandbox", "apps": ["true"]}}), ..Model::default() }],
                          &[ids("hi")], json!({}));
    assert_eq!(s.get("/api/computer/status", false).0, 401);
    let (code, st) = s.get("/api/computer/status", true);
    assert_eq!(code, 200);
    assert_eq!((st["on"].as_bool(), st["operator_in_control"].as_bool()), (Some(true), Some(false)));

    // typing needs control; take it, then hand it back
    assert_eq!(s.post("/api/computer/type", &json!({"text": "x"})).0, 409);
    let (_, st) = s.post("/api/computer/control", &json!({"action": "take"}));
    assert_eq!(st["operator_in_control"], true);
    let (_, st) = s.post("/api/computer/control", &json!({"action": "give"}));
    assert_eq!(st["operator_in_control"], false);
    assert_eq!(s.post("/api/computer/control", &json!({"action": "steal"})).0, 400);
    // only allowed apps
    assert_eq!(s.post("/api/computer/open", &json!({"app": "bash"})).0, 403);

    // the stream: another site's page is refused before it opens
    match ws(&s, Some("https://evil.example")) {
        Err(tungstenite::Error::Http(r)) => assert_eq!(r.status(), 403),
        other => panic!("a foreign origin must be refused: {:?}", other.map(|_| ())),
    }
    // a wrong key in the first message: told so, nothing else
    let mut w = ws(&s, Some(&s.base)).expect("own page");
    w.send(tungstenite::Message::Text(json!({"type": "auth", "key": "wrong"}).to_string().into())).unwrap();
    let m = next_json(&mut w);
    assert_eq!(m["type"], "error");
    assert!(m["message"].as_str().unwrap().contains("API key"), "{m}");
    // the right key: the state (with Xvfb), or a plain error saying what is missing (without)
    let mut w = ws(&s, Some(&s.base)).expect("own page");
    w.send(tungstenite::Message::Text(json!({"type": "auth", "key": KEY}).to_string().into())).unwrap();
    let m = next_json(&mut w);
    match m["type"].as_str() {
        Some("state") => {
            assert_eq!(m["operator_in_control"], false);
            assert!(m["width"].as_u64().unwrap() > 0);
            // closing the tab hands control back
            w.send(tungstenite::Message::Text(json!({"type": "take"}).to_string().into())).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(300));
            assert_eq!(s.get("/api/computer/status", true).1["operator_in_control"], true);
            drop(w);
            std::thread::sleep(std::time::Duration::from_millis(800));
            assert_eq!(s.get("/api/computer/status", true).1["operator_in_control"], false, "a closed tab hands control back");
        }
        Some("error") => assert!(m["message"].as_str().unwrap().contains("Xvfb") || m["message"].as_str().unwrap().contains("sandbox"), "{m}"),
        _ => panic!("unexpected {m}"),
    }
}
