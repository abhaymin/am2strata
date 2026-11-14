//! The route and contract inventory (docs/REGRESSION.md): every route main.rs registers has a row here saying who
//! may call it and what each kind of caller gets. The walk runs every row against a harness server; the guard reads
//! main.rs and fails when a route has no row (or a row has no route), so an endpoint cannot ship without a contract.
//!
//! Each row is called six ways:
//!   1. with no key                              -> `no_key`
//!   2. with a wrong key of the right length     -> `wrong_key`
//!   3. with a client profile's key              -> `client`
//!   4. with the owner's key and a minimal body  -> `owner`
//!   5. with the owner's key and a broken body   -> `malformed`  (never a 5xx)
//!   6. with the owner's key from another site's page (a foreign `Origin`) -> 403 on every state-changing owner route
//! and, for routes that take JSON, with JSON of the wrong type throughout (never a 5xx).
//!
//! "Minimal" means the smallest body the route accepts that stays on this PC: where a real call would download
//! from Hugging Face, start a program or delete a model, the row names something that does not exist and expects the
//! refusal that proves the key was accepted and the body read (the row's note says so).

mod common;

use common::regress::*;
use common::*;
use serde_json::json;
use std::collections::BTreeSet;

/// Who may call a route.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Who {
    /// no key needed
    Public,
    /// the owner's key or a client profile's key (the model APIs)
    Client,
    /// the owner's key only (the server's settings, files and desktop)
    Owner,
    /// its own proof (a pairing code, a master's key): every status is spelled out in the row
    Special,
}
use Who::*;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Body {
    None,
    Json(&'static str),
    /// (content type, body)
    Raw(&'static str, &'static str),
}

/// What a foreign `Origin` gets with the owner's key.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Foreign {
    /// refused with 403 (every state-changing owner route)
    Refused,
    /// not a state-changing owner route: the foreign Origin changes nothing (the owner status is expected)
    Same,
    /// answered differently, for the reason given
    Status(u16, &'static str),
}

#[derive(Clone, Copy, Debug)]
struct Row {
    method: &'static str,
    /// the route as main.rs registers it
    route: &'static str,
    /// the path called (the route with its parameters filled in)
    path: &'static str,
    who: Who,
    body: Body,
    no_key: u16,
    wrong_key: u16,
    client: u16,
    owner: u16,
    malformed: u16,
    foreign: Foreign,
    /// further statuses the wrong-type sweep may get besides 4xx (e.g. 200 where every field is optional)
    note: &'static str,
}

const fn row(method: &'static str, route: &'static str, who: Who) -> Row {
    let (no_key, client) = match who {
        Public => (200, 200),
        Client => (401, 200),
        Owner | Special => (401, 401),
    };
    let state_changing = !matches!(method.as_bytes()[0], b'G');
    Row { method, route, path: route, who, body: Body::None, no_key, wrong_key: no_key, client, owner: 200, malformed: 200,
          foreign: if state_changing && matches!(who, Owner) { Foreign::Refused } else { Foreign::Same }, note: "" }
}

impl Row {
    /// The path to call when the route has parameters.
    const fn at(mut self, path: &'static str) -> Row {
        self.path = path;
        self
    }

    /// The minimal JSON body; a broken one is then a 400 unless `.malformed` says otherwise.
    const fn json(mut self, body: &'static str) -> Row {
        self.body = Body::Json(body);
        self.malformed = 400;
        self
    }

    const fn raw(mut self, content_type: &'static str, body: &'static str) -> Row {
        self.body = Body::Raw(content_type, body);
        self
    }

    /// The status for everyone the route lets in.
    const fn ok(mut self, status: u16) -> Row {
        if matches!(self.who, Public) {
            self.no_key = status;
            self.wrong_key = status;
        }
        if matches!(self.who, Public | Client) {
            self.client = status;
        }
        self.owner = status;
        if matches!(self.body, Body::None) {
            self.malformed = status;
        }
        self
    }

    const fn client(mut self, status: u16) -> Row {
        self.client = status;
        self
    }

    const fn malformed(mut self, status: u16) -> Row {
        self.malformed = status;
        self
    }

    const fn statuses(mut self, no_key: u16, wrong_key: u16, client: u16, owner: u16, malformed: u16) -> Row {
        self.no_key = no_key;
        self.wrong_key = wrong_key;
        self.client = client;
        self.owner = owner;
        self.malformed = malformed;
        self
    }

    const fn foreign(mut self, f: Foreign) -> Row {
        self.foreign = f;
        self
    }

    const fn note(mut self, note: &'static str) -> Row {
        self.note = note;
        self
    }
}

const CHAT: &str = r#"{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}"#;
const GEMINI: &str = r#"{"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": {"maxOutputTokens": 8}}"#;

/// THE TABLE. One row per method of every route in main.rs, in main.rs's order. Adding a route there without a row
/// here fails `every_route_in_main_rs_has_a_contract_row`.
const ROUTES: &[Row] = &[
    // ---------------------------------------------------------------------------------- health and the model APIs
    row("GET", "/health", Public),
    row("GET", "/api/health", Public),
    row("GET", "/api/inference/loaded-models", Client),
    row("GET", "/api/inference/status", Client),
    row("POST", "/api/inference/load", Client).json(r#"{"model_path": "mock-a"}"#),
    row("GET", "/v1/models", Client),
    row("POST", "/v1/chat/completions", Client).json(CHAT),
    row("POST", "/v1/responses", Client).json(r#"{"input": "hi", "max_output_tokens": 8}"#),
    row("GET", "/v1/responses/{*rest}", Client).at("/v1/responses/resp_1").ok(404).note("stateless: nothing is stored"),
    row("POST", "/v1/responses/{*rest}", Client).at("/v1/responses/resp_1/cancel").ok(404).note("stateless"),
    row("DELETE", "/v1/responses/{*rest}", Client).at("/v1/responses/resp_1").ok(404).note("stateless"),
    row("POST", "/v1/messages", Client).json(r#"{"model": "x", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}"#),
    row("POST", "/v1/messages/count_tokens", Client).json(r#"{"model": "x", "messages": [{"role": "user", "content": "hi"}]}"#),
    // ------------------------------------------------------------------------------------- the server's own state
    row("GET", "/api/system", Owner),
    row("GET", "/api/models", Owner),
    row("GET", "/api/models/current", Owner),
    row("POST", "/api/models/load", Owner).json(r#"{"model": "mock-a"}"#),
    row("POST", "/api/models/unload", Owner),
    row("GET", "/api/queue", Owner),
    row("GET", "/api/gpus", Owner),
    row("GET", "/api/machine", Owner),
    row("GET", "/api/cluster", Owner),
    row("POST", "/api/cluster/pair", Owner).json(r#"{"url": "", "code": ""}"#).ok(400).note("an empty address: refused before anything is contacted"),
    row("POST", "/api/cluster/join", Special).json(r#"{"code": "123456", "master_id": "m"}"#).statuses(404, 404, 404, 404, 400)
        .note("a worker's pairing code is the proof, not a key; this server is no worker, so there is nothing to join"),
    row("POST", "/api/cluster/forget", Special).statuses(401, 401, 401, 401, 401)
        .note("only a key this worker gave a master: not even the owner's"),
    row("DELETE", "/api/cluster/workers/{name}", Owner).at("/api/cluster/workers/nope").ok(404),
    row("POST", "/api/machine/measure", Owner),
    row("GET", "/api/optimise", Owner),
    row("POST", "/api/models/{id}/optimise", Owner).at("/api/models/nope/optimise").json(r#"{"objective": "decode"}"#).ok(404)
        .note("no such model (a real one would tune for minutes)"),
    row("POST", "/api/models/{id}/tuned", Owner).at("/api/models/nope/tuned").json(r#"{"enabled": false}"#).ok(404),
    row("POST", "/api/tools/approve", Owner).json(r#"{"id": "nope", "ok": false}"#).ok(404).note("no step waits"),
    row("POST", "/api/cache/snapshots/clear", Owner),
    row("DELETE", "/api/models/{id}", Owner).at("/api/models/nope").ok(404).note("no such model: nothing is removed"),
    row("GET", "/api/models/{id}/removal", Owner).at("/api/models/nope/removal").ok(404),
    row("DELETE", "/api/media/models/{name}", Owner).at("/api/media/models/nope").ok(400).note("not downloaded: nothing is removed"),
    row("GET", "/api/media/models/{name}/removal", Owner).at("/api/media/models/nope/removal").ok(400),
    // ------------------------------------------------------------------------------------------------------ media
    row("GET", "/api/media/capabilities", Client),
    row("GET", "/api/media/genres", Client),
    row("POST", "/api/jobs", Client).json(r#"{"tool": "nope", "arguments": {}}"#).ok(400).note("an unknown tool: nothing is queued"),
    row("POST", "/v1/media/productions", Client).json(r#"{}"#).ok(400).client(403)
        .note("no parts: nothing is queued; the client profile of this test has neither music nor speech, and is told so"),
    row("POST", "/v1/media/videos/compose", Client).json(r#"{}"#).ok(400).note("nothing to join"),
    row("POST", "/v1/media/thumbnails", Client).json(r#"{}"#).ok(400),
    row("POST", "/v1/media/check", Client).json(r#"{}"#).note("nothing to check is a report saying so"),
    row("POST", "/v1/media/dub", Client).json(r#"{}"#).ok(400),
    row("GET", "/api/jobs/{id}", Client).at("/api/jobs/mj-999").ok(404),
    row("GET", "/v1/media/jobs/{id}", Client).at("/v1/media/jobs/mj-999").ok(404),
    row("GET", "/v1/media/jobs/{id}/content", Client).at("/v1/media/jobs/mj-999/content").ok(404),
    row("GET", "/v1/media/jobs/{id}/captions.srt", Client).at("/v1/media/jobs/mj-999/captions.srt").ok(404),
    // ------------------------------------------------------------------------------- downloads, monitor, the canvas
    row("GET", "/api/models/updates", Owner).at("/api/models/updates?id=nope").ok(404).note("no such model: Hugging Face is not asked"),
    row("POST", "/api/models/update", Owner).json(r#"{"id": "nope"}"#).ok(404).note("no such model: nothing is fetched"),
    row("GET", "/api/requests", Owner),
    row("POST", "/api/requests/clear", Owner),
    row("POST", "/api/hub/quants", Owner).json(r#"{}"#).ok(400).note("no repository: nothing is fetched"),
    row("GET", "/api/hub/pull", Owner),
    row("POST", "/api/hub/pull", Owner).json(r#"{"repo": "owner/name"}"#).ok(400).note("no quantization: nothing is fetched"),
    row("GET", "/api/run", Owner),
    row("POST", "/api/run", Owner).json(r#"{"language": "ruby", "code": "puts 1"}"#).ok(400).note("not Python: nothing runs (501 without bubblewrap)"),
    row("POST", "/v1/images/generations", Client).json(r#"{"prompt": "a red dot", "size": "256x256"}"#),
    row("POST", "/v1/images/edits", Client).raw("application/json", r#"{}"#).ok(400).malformed(400),
    row("POST", "/v1/videos/generations", Client).json(r#"{}"#).ok(400),
    row("POST", "/v1/audio/transcriptions", Client).raw("application/json", r#"{}"#).ok(400).malformed(400),
    row("POST", "/v1/audio/translations", Client).raw("application/json", r#"{}"#).ok(400).malformed(400),
    row("POST", "/v1/audio/speech", Client).json(r#"{}"#).ok(400),
    row("POST", "/v1/audio/music", Client).json(r#"{}"#).ok(400),
    row("POST", "/api/media/audio/device", Owner).json(r#"{"device": "nope"}"#).ok(400),
    // ----------------------------------------------------------------------------------------------------- memory
    row("GET", "/api/memory", Owner),
    row("POST", "/api/memory", Owner).json(r#"{"text": "the regression suite was here"}"#),
    row("GET", "/v1/memory", Client),
    row("POST", "/v1/memory", Client).json(r#"{"text": "a client's fact"}"#),
    row("DELETE", "/v1/memory/{id}", Client).at("/v1/memory/nope").ok(404),
    row("POST", "/api/memory/settings", Owner).json(r#"{}"#),
    row("DELETE", "/api/memory/{id}", Owner).at("/api/memory/nope").ok(404),
    // ------------------------------------------------------------------------------------------------ media models
    row("GET", "/api/media/models", Owner),
    row("GET", "/api/media/models/pull", Owner),
    row("POST", "/api/media/models/pull", Owner).json(r#"{"model": "nope"}"#).ok(400).note("not in the catalog: nothing is fetched"),
    row("POST", "/api/media/models/default", Owner).json(r#"{"model": "nope"}"#).ok(400),
    row("POST", "/api/media/licenses/accept", Owner).json(r#"{"model": "nope"}"#).ok(400),
    row("POST", "/api/preview", Owner).json(r#"{"html": "<p>hi</p>"}"#),
    row("GET", "/preview/{id}", Public).at("/preview/nope").ok(404).note("the unguessable address is the capability; an unknown one is a 404"),
    row("POST", "/load", Owner),
    row("POST", "/unload", Owner),
    row("GET", "/settings", Owner),
    row("POST", "/settings", Owner).json(r#"{}"#),
    row("GET", "/config", Owner),
    row("POST", "/config", Owner).json(r#"{"set": {"api_monitor": true}}"#).note("the value it has already: nothing changes"),
    // --------------------------------------------------------------------------------------------------- Workshop
    row("GET", "/v1/workshop/jobs", Owner),
    row("POST", "/v1/workshop/jobs", Owner).json(r#"{"kind": "nope"}"#).ok(400).note("an unknown kind: nothing starts"),
    row("GET", "/v1/workshop/jobs/{id}", Owner).at("/v1/workshop/jobs/999").ok(404),
    row("DELETE", "/v1/workshop/jobs/{id}", Owner).at("/v1/workshop/jobs/999").ok(409).note("no such job to cancel"),
    row("GET", "/v1/workshop/files", Owner),
    row("POST", "/v1/workshop/uploads", Owner).at("/v1/workshop/uploads?name=regress.txt").raw("application/octet-stream", "hello"),
    row("GET", "/v1/workshop/dataset", Owner).at("/v1/workshop/dataset?path=/nope/x.jsonl").ok(400).note("outside the Workshop's folders"),
    row("POST", "/v1/workshop/dataset", Owner).json(r#"{"path": "/nope/x.jsonl"}"#).ok(400),
    // ------------------------------------------------------------------------------------------- HTTPS, Computer
    row("GET", "/api/tls", Owner),
    row("POST", "/api/tls", Owner).json(r#"{"action": "nope"}"#).ok(400).note("an unknown action: HTTPS stays as it is"),
    row("GET", "/api/computer/status", Owner),
    row("GET", "/api/computer/stream", Owner).ok(426).note("a WebSocket: a plain request gets 401 without the key, 426 with it"),
    row("GET", "/api/computer/screenshot", Owner).ok(409).note("computer use is off in this run config"),
    row("POST", "/api/computer/control", Owner).json(r#"{"action": "give"}"#),
    row("GET", "/api/computer/clipboard", Owner).ok(409),
    row("POST", "/api/computer/clipboard", Owner).json(r#"{"text": "x"}"#).ok(409),
    row("POST", "/api/computer/type", Owner).json(r#"{"text": "x"}"#).ok(409),
    row("POST", "/api/computer/open", Owner).json(r#"{"app": "nope"}"#).ok(403),
    // -------------------------------------------------------------------------------------------- MCP and the rest
    row("GET", "/mcp", Owner),
    row("GET", "/api/mcp/servers", Owner).foreign(Foreign::Status(403, "the list shows each server's command and address: this server's own page only")),
    row("POST", "/api/mcp/servers", Owner).json(r#"{}"#).ok(400).note("no name: nothing is added"),
    row("POST", "/api/mcp/servers/test", Owner).json(r#"{}"#).note("a server that cannot start is a report {ok: false}, not an error"),
    row("PUT", "/api/mcp/servers/{name}", Owner).at("/api/mcp/servers/nope").json(r#"{}"#).ok(404),
    row("DELETE", "/api/mcp/servers/{name}", Owner).at("/api/mcp/servers/nope").ok(404),
    row("POST", "/api/mcp/servers/{name}/tools", Owner).at("/api/mcp/servers/nope/tools").json(r#"{"search": false}"#).ok(404),
    row("GET", "/metrics", Owner),
    row("POST", "/api/monitor/clear", Owner),
    row("GET", "/api/agents", Owner),
    row("GET", "/api/clients", Owner),
    row("GET", "/v1beta/models", Client),
    row("GET", "/v1beta/models/{target}", Client).at("/v1beta/models/mock-a"),
    row("POST", "/v1beta/models/{target}", Client).at("/v1beta/models/gemini-2.5-pro:generateContent").json(GEMINI),
    row("GET", "/api/monitor/media/{name}", Owner).at("/api/monitor/media/nope").ok(404),
    row("GET", "/metrics/prometheus", Owner),
    row("GET", "/v1/status", Owner),
    // ------------------------------------------------------------------------------------------------ the web app
    row("GET", "/", Public),
    row("GET", "/api-monitor", Public),
    row("GET", "/web", Public).at("/web/index.html"),
    row("GET", "/fonts", Public).at("/fonts/nope.woff2").ok(404).note("the test web folder has no fonts"),
];

fn table() -> BTreeSet<(String, String)> {
    ROUTES.iter().map(|r| (r.method.to_string(), r.route.to_string())).collect()
}

/// THE GUARD: main.rs and the table list the same routes. A new endpoint needs its contract row before it can merge;
/// a row whose route is gone is removed with it.
#[test]
fn every_route_in_main_rs_has_a_contract_row() {
    let (main, table) = (routes_in_main(), table());
    let missing: Vec<String> = main.difference(&table).map(|(m, p)| format!("{m} {p}")).collect();
    let stale: Vec<String> = table.difference(&main).map(|(m, p)| format!("{m} {p}")).collect();
    assert!(missing.is_empty(), "routes in main.rs with no row in tests/regression_api.rs (add one: who may call it, and what \
                                 each caller gets; docs/REGRESSION.md says how):\n  {}", missing.join("\n  "));
    assert!(stale.is_empty(), "rows in tests/regression_api.rs whose route is no longer in main.rs:\n  {}", stale.join("\n  "));
    assert_eq!(ROUTES.len(), table.len(), "a route is listed twice in the table");
    assert!(main.len() > 100, "only {} routes read from main.rs: did the router move?", main.len());
    let elsewhere = routers_outside_main();
    assert!(elsewhere.is_empty(), "a Router is built outside main.rs ({}): its routes need rows too, and \
                                   tests/common/regress.rs::routes_in_main must learn to read them", elsewhere.join(", "));
}

/// The guard's own reader, checked against routes written every way main.rs writes them.
#[test]
fn the_guard_reads_every_method_of_a_route() {
    let main = routes_in_main();
    for (m, p) in [("GET", "/v1/responses/{*rest}"), ("POST", "/v1/responses/{*rest}"), ("DELETE", "/v1/responses/{*rest}"),
                   ("PUT", "/api/mcp/servers/{name}"), ("DELETE", "/api/mcp/servers/{name}"),
                   ("DELETE", "/api/cluster/workers/{name}"), ("POST", "/v1/workshop/uploads"), ("GET", "/"), ("GET", "/web")] {
        assert!(main.contains(&(m.to_string(), p.to_string())), "{m} {p} was not read from main.rs");
    }
}

/// The rows agree with themselves: who may call a route decides what a stranger and a client get.
#[test]
fn the_table_is_consistent() {
    for r in ROUTES {
        let name = format!("{} {}", r.method, r.route);
        match r.who {
            Public => assert_eq!((r.no_key, r.wrong_key, r.client), (r.owner, r.owner, r.owner), "{name}: a public route answers everyone alike"),
            Client => {
                assert_eq!((r.no_key, r.wrong_key), (401, 401), "{name}: a stranger gets 401");
                assert_ne!(r.client, 401, "{name}: a client profile's key is let in");
            }
            Owner => {
                assert_eq!((r.no_key, r.wrong_key, r.client), (401, 401, 401), "{name}: only the owner's key");
                if r.method != "GET" {
                    assert_eq!(r.foreign, Foreign::Refused, "{name}: a state-changing owner route refuses another site's page");
                }
            }
            Special => assert!(!r.note.is_empty(), "{name}: a special route says why in its note"),
        }
        assert!(r.malformed < 500 && r.owner < 500, "{name}: a 5xx is never the contract");
        // the path called matches the route's pattern
        let (pat, path): (Vec<&str>, Vec<&str>) = (r.route.split('/').collect(), r.path.split('?').next().unwrap().split('/').collect());
        let fits = pat.iter().enumerate().all(|(i, seg)| seg.starts_with("{*") || path.get(i).is_some_and(|p| seg.starts_with('{') || p == seg))
            && (path.len() == pat.len() || pat.last().is_some_and(|s| s.starts_with("{*")) || matches!(r.route, "/web" | "/fonts"));
        assert!(fits, "{name}: the path {} does not fit the route", r.path);
    }
}

fn request<'a>(s: &Server, r: &Row, key: Option<&'a str>, body: Body) -> Req<'a> {
    let q = Req::new(r.method, &s.base, r.path).key(key);
    match body {
        Body::None => q,
        Body::Json(b) => q.raw("application/json", b.as_bytes()),
        Body::Raw(ct, b) => q.raw(ct, b.as_bytes()),
    }
}

fn excerpt(a: &Answer) -> String {
    a.text.chars().take(160).collect::<String>().replace('\n', " ")
}

/// THE WALK: every row, every kind of caller. All mismatches are reported together.
#[test]
fn every_route_answers_each_caller_as_its_row_says() {
    let s = Server::start(&[client_model(json!({"api_monitor": true}))], &[ids("Thinking.\n</think>\n\nhello")], json!({}));
    let wrong: std::cell::RefCell<Vec<String>> = Default::default();
    let mut checks = 0;
    for r in ROUTES {
        let name = format!("{} {}", r.method, r.path);
        let mut check = |what: &str, want: u16, a: Answer| {
            checks += 1;
            if a.status != want {
                wrong.borrow_mut().push(format!("{name} [{what}]: expected {want}, got {} ({})", a.status, excerpt(&a)));
            }
            a
        };
        check("no key", r.no_key, request(&s, r, None, r.body).send());
        check("wrong key", r.wrong_key, request(&s, r, Some(WRONG_KEY), r.body).send());
        // a wrong key in the other header a client may use
        check("wrong x-api-key", r.wrong_key, request(&s, r, None, r.body).header("x-api-key", WRONG_KEY).send());
        check("client key", r.client, request(&s, r, Some(CLIENT_KEY), r.body).send());
        // another site's page with the owner's key, before the owner's own call (a refused call changed nothing)
        match r.foreign {
            Foreign::Refused => {
                let a = check("foreign origin", 403, request(&s, r, Some(KEY), r.body).header("origin", FOREIGN_ORIGIN).send());
                if a.status == 403 && a.json()["error"]["type"] != "permission_error" {
                    wrong.borrow_mut().push(format!("{name} [foreign origin]: a 403 without the permission_error body ({})", excerpt(&a)));
                }
            }
            Foreign::Same => { check("foreign origin", r.owner, request(&s, r, Some(KEY), r.body).header("origin", FOREIGN_ORIGIN).send()); }
            Foreign::Status(code, _) => { check("foreign origin", code, request(&s, r, Some(KEY), r.body).header("origin", FOREIGN_ORIGIN).send()); }
        }
        let a = check("owner key", r.owner, request(&s, r, Some(KEY), r.body).send());
        // every refusal the server words itself is JSON with error.message (the shape all three API families share)
        if (400..500).contains(&a.status) && r.who != Public && !a.json()["error"]["message"].is_string() {
            wrong.borrow_mut().push(format!("{name} [owner key]: a {} without a JSON error.message ({})", a.status, excerpt(&a)));
        }
        let broken = match r.body {
            Body::Raw(ct, _) => Body::Raw(ct, "{\"broken\": "),
            _ => Body::Json("{\"broken\": "),
        };
        let a = check("malformed body", r.malformed, request(&s, r, Some(KEY), broken).send());
        if a.status == 400 && !a.json()["error"]["message"].is_string() {
            wrong.borrow_mut().push(format!("{name} [malformed body]: a 400 without a JSON error.message ({})", excerpt(&a)));
        }
    }
    let wrong = wrong.into_inner();
    assert!(wrong.is_empty(), "{} of {checks} contract checks failed:\n  {}", wrong.len(), wrong.join("\n  "));
    assert!(checks >= ROUTES.len() * 7);
    // the server is still up and still answers a chat after all of it
    let a = Req::new("POST", &s.base, "/v1/chat/completions").key(Some(KEY)).raw("application/json", CHAT.as_bytes()).send();
    assert_eq!(a.status, 200, "the server after the walk: {}", excerpt(&a));
}

/// JSON of the wrong type everywhere (an array, a string, a number where an object is read; every usual field with
/// the wrong type): a refusal, never a 5xx, and the server keeps running.
#[test]
fn wrong_typed_json_is_refused_never_a_500() {
    let s = Server::start(&[client_model(json!({}))], &[ids("Thinking.\n</think>\n\nhello")], json!({}));
    let junk = [
        json!([]), json!("x"), json!(7), json!(null), json!({}),
        json!({"model": 5, "messages": "x", "input": 7, "contents": 3, "prompt": [], "text": {}, "tools": "x", "tool": 4,
               "arguments": "x", "stream": "yes", "max_tokens": "many", "path": [], "name": {}, "action": 1, "kind": 9,
               "id": [], "ok": "x", "code": 3, "language": 2, "url": 5, "repo": [], "quant": {}, "enabled": "x", "html": 4,
               "device": [], "app": 3, "parts": "x", "voice": 1, "size": 9, "n": "x", "scope": 4, "remove": "x", "replace": 3}),
        json!({"messages": [5, null, {"role": 3, "content": {"x": 1}}], "input": [1, {"type": 9}], "contents": [1, {"parts": 4}],
               "tools": [1, {"function": 2}], "system": 9, "max_tokens": -1, "temperature": "hot"}),
    ];
    let mut wrong: Vec<String> = Vec::new();
    let mut checks = 0;
    for r in ROUTES.iter().filter(|r| r.method != "GET" && r.who != Public) {
        for j in &junk {
            let a = Req::new(r.method, &s.base, r.path).key(Some(KEY)).json(j).send();
            checks += 1;
            if a.status >= 500 && !ALLOWED_5XX.iter().any(|(m, p, code, _)| *m == r.method && *p == r.route && *code == a.status) {
                wrong.push(format!("{} {} with {}: {} ({})", r.method, r.path, j.to_string().chars().take(60).collect::<String>(), a.status, excerpt(&a)));
            }
        }
    }
    // again for the model APIs with shared defaults set (Settings): they are written into each request, and writing
    // into a body that is no object must not panic either
    let (code, r) = s.post("/settings", &json!({"defaults": {"max_tokens": 7, "reasoning_effort": "low"}}));
    assert_eq!(code, 200, "{r}");
    for path in ["/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/messages/count_tokens",
                 "/v1beta/models/gemini-2.5-pro:generateContent", "/v1beta/models/x:countTokens"] {
        for j in &junk {
            let a = Req::new("POST", &s.base, path).key(Some(KEY)).json(j).send();
            checks += 1;
            if a.status >= 500 {
                wrong.push(format!("POST {path} (shared defaults set) with {}: {} ({})", j.to_string().chars().take(60).collect::<String>(), a.status, excerpt(&a)));
            }
        }
    }
    assert!(wrong.is_empty(), "{} of {checks} wrong-type requests got a 5xx:\n  {}", wrong.len(), wrong.join("\n  "));
    assert_eq!(s.get("/health", false).0, 200, "the server is still up");
}

/// A stranger learns nothing from a route, whatever the body: 401 comes before any reading of it (the bug class of
/// PR #13's "the key is checked before the job id is parsed"). Before this suite, the 45 routes that take JSON
/// answered a stranger's broken body with axum's own 400/415; main.rs's `body_rejections` now puts the key first for
/// every route at once. A route that needs an exception is listed here with its reason, never silently.
const KNOWN_BUG_BODY_READ_BEFORE_KEY: &[(&str, &str)] = &[];

#[test]
fn a_stranger_with_a_broken_body_gets_401_not_a_parse_error() {
    let s = Server::start(&[client_model(json!({}))], &[ids("hi")], json!({}));
    let (mut wrong, mut fixed) = (Vec::new(), Vec::new());
    for r in ROUTES.iter().filter(|r| r.no_key == 401) {
        let listed = KNOWN_BUG_BODY_READ_BEFORE_KEY.contains(&(r.method, r.route));
        let mut codes = Vec::new();
        for key in [None, Some(WRONG_KEY)] {
            codes.push(Req::new(r.method, &s.base, r.path).key(key).raw("application/json", b"{\"broken\": ").send().status);
            // and with no content type at all
            let mut q = Req::new(r.method, &s.base, r.path).key(key);
            q.body = Some(("text/plain", b"x".to_vec()));
            codes.push(q.send().status);
        }
        let all_401 = codes.iter().all(|c| *c == 401);
        if listed && all_401 {
            fixed.push(format!("(\"{}\", \"{}\")", r.method, r.route));
        } else if listed {
            assert!(codes.iter().all(|c| matches!(c, 400 | 401 | 415 | 422)), "{} {}: a stranger got {codes:?}", r.method, r.route);
        } else if !all_401 {
            wrong.push(format!("(\"{}\", \"{}\"),   // got {codes:?}", r.method, r.route));
        }
    }
    assert!(wrong.is_empty(), "routes that answer a stranger's broken body with something other than 401 (check the key before \
                               reading the body):\n    {}", wrong.join("\n    "));
    assert!(fixed.is_empty(), "fixed: remove from KNOWN_BUG_BODY_READ_BEFORE_KEY:\n    {}", fixed.join("\n    "));
}

/// 5xx answers that are the contract for a feature this PC (or the test harness) does not have: (method, route,
/// status, why). Anything else of 500 and above on wrong-typed input fails the sweep.
const ALLOWED_5XX: &[(&str, &str, u16, &str)] = &[];

/// The security headers every answer carries, whoever asked.
#[test]
fn every_answer_carries_the_security_headers() {
    let s = Server::start(&[client_model(json!({}))], &[ids("hi")], json!({}));
    for (path, key) in [("/", None), ("/health", None), ("/v1/models", None), ("/v1/models", Some(KEY)), ("/web/index.html", None)] {
        let a = Req::new("GET", &s.base, path).key(key).send();
        assert_eq!(a.headers.get("x-frame-options").and_then(|v| v.to_str().ok()), Some("SAMEORIGIN"), "{path}");
        assert_eq!(a.headers.get("x-content-type-options").and_then(|v| v.to_str().ok()), Some("nosniff"), "{path}");
        assert_eq!(a.headers.get("referrer-policy").and_then(|v| v.to_str().ok()), Some("no-referrer"), "{path}");
        assert!(a.headers.get("strict-transport-security").is_none(), "{path}: no HSTS over plain HTTP");
        assert!(a.headers.get("access-control-allow-origin").is_none(), "{path}: no CORS grant to other sites");
    }
}

/// An unknown path or a method a route does not take is a plain 404/405, with or without the key (never a 500, never
/// the web app's page).
#[test]
fn unknown_paths_and_methods_are_404_and_405() {
    let s = Server::start(&[client_model(json!({}))], &[ids("hi")], json!({}));
    for key in [None, Some(KEY)] {
        assert_eq!(Req::new("GET", &s.base, "/v1/nope").key(key).send().status, 404);
        assert_eq!(Req::new("GET", &s.base, "/v1/chat/completions").key(key).send().status, 405);
        assert_eq!(Req::new("DELETE", &s.base, "/v1/models").key(key).send().status, 405);
        assert_eq!(Req::new("PATCH", &s.base, "/settings").key(key).send().status, 405);
    }
}
