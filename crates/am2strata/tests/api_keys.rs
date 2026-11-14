//! Generated API keys end to end against the mock engine: a key is made and told once, opens the model APIs and
//! nothing of the owner's, is counted (per key and per model, streams too), limited, revoked and expired; the file
//! keeps no key.

mod common;

use common::*;
use serde_json::{json, Value};
use std::time::Duration;

fn send(s: &Server, method: &str, key: Option<&str>, path: &str, body: Option<&Value>, extra: &[(&str, &str)]) -> (u16, Value) {
    let c = reqwest::blocking::Client::new();
    let url = format!("{}{path}", s.base);
    let mut r = match method {
        "POST" => c.post(url),
        "DELETE" => c.delete(url),
        _ => c.get(url),
    };
    if let Some(k) = key {
        r = r.bearer_auth(k);
    }
    if let Some(b) = body {
        r = r.json(b);
    }
    for (k, v) in extra {
        r = r.header(*k, *v);
    }
    let r = r.timeout(Duration::from_secs(60)).send().expect("request");
    (r.status().as_u16(), r.json().unwrap_or(Value::Null))
}

fn chat(s: &Server, key: &str, body: &Value) -> (u16, Value) {
    send(s, "POST", Some(key), "/v1/chat/completions", Some(body), &[])
}

fn hi() -> Value {
    json!({"messages": [{"role": "user", "content": "hi"}]})
}

/// The text of a streamed answer (every event read to the end).
fn stream(s: &Server, key: &str, path: &str, body: &Value) -> String {
    reqwest::blocking::Client::new().post(format!("{}{path}", s.base)).header("x-api-key", key).json(body)
        .timeout(Duration::from_secs(60)).send().expect("request").text().unwrap()
}

fn new_key(s: &Server, body: Value) -> (String, String) {
    let (c, v) = s.post("/api/keys", &body);
    assert_eq!(c, 201, "{v}");
    (v["key"].as_str().unwrap().to_string(), v["id"].as_str().unwrap().to_string())
}

fn reply() -> Vec<u32> {
    ids("Hi.\n</think>\n\nhello")
}

#[test]
fn a_generated_key_opens_the_model_apis_is_counted_and_can_be_revoked() {
    let s = Server::start(&[Model::default()], &[reply()], json!({}));
    assert_eq!(s.get("/api/keys", false).0, 401, "needs the key");
    assert_eq!(s.get("/api/keys", true).1["keys"], json!([]));

    // made: the key is told once; the list and the file never have it
    let (c, made) = s.post("/api/keys", &json!({"name": "laptop", "tools": ["web", "memory"], "expires_in": "30d"}));
    assert_eq!(c, 201, "{made}");
    let key = made["key"].as_str().unwrap().to_string();
    let id = made["id"].as_str().unwrap().to_string();
    assert!(key.starts_with("am2_") && key.len() == 44, "the key's shape");
    assert_eq!(made["prefix"], key[..8]);
    assert_eq!(made["status"], "active");
    assert_eq!(made["tools"], json!(["memory", "web"]));
    let (_, list) = s.get("/api/keys", true);
    assert_eq!(list["keys"][0]["name"], "laptop");
    assert!(!list.to_string().contains(&key[8..]), "the list has no key");
    let file = std::fs::read_to_string(s.dir.join("api-keys.json")).unwrap();
    assert!(!file.contains(&key) && !file.contains(&key[8..]), "the file holds no key in plain text");
    assert!(file.contains("\"hash\""));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(s.dir.join("api-keys.json")).unwrap().permissions().mode() & 0o777, 0o600);
    }
    // bad requests say why; another web page may not make keys
    assert_eq!(s.post("/api/keys", &json!({"name": "laptop"})).0, 400, "the name is taken");
    assert_eq!(s.post("/api/keys", &json!({"name": "x", "tools": ["shell"]})).0, 400);
    assert_eq!(send(&s, "POST", Some(KEY), "/api/keys", Some(&json!({"name": "evil"})), &[("origin", "http://evil.example")]).0, 403);

    // it works at once, as a client profile: the model APIs with its tools
    let (c, r) = chat(&s, &key, &hi());
    assert_eq!(c, 200, "{r}");
    let prompt = tokenizer().decode(&line_ids(s.gen_lines().last().unwrap())).unwrap_or_default();
    assert!(prompt.contains("web_search") && prompt.contains("remember"), "the key's tools: {prompt}");
    let used = (r["usage"]["prompt_tokens"].as_u64().unwrap(), r["usage"]["completion_tokens"].as_u64().unwrap());
    assert!(used.0 > 0 && used.1 > 0, "{r}");
    assert_eq!(send(&s, "GET", Some(&key), "/v1/models", None, &[]).0, 200);
    // x-api-key works as Bearer does
    assert_eq!(send(&s, "GET", None, "/v1/models", None, &[("x-api-key", &key)]).0, 200);

    // never the owner's routes: settings, the Workshop, the Computer tab, the keys and everyone's usage
    for path in ["/settings", "/config", "/v1/workshop/jobs", "/api/computer/status", "/api/keys", "/api/usage", "/api/clients", "/api/mcp/servers"] {
        assert_eq!(send(&s, "GET", Some(&key), path, None, &[]).0, 401, "{path}");
    }
    assert_eq!(send(&s, "POST", Some(&key), "/api/keys", Some(&json!({"name": "more"})), &[]).0, 401);
    assert_eq!(send(&s, "DELETE", Some(&key), &format!("/api/keys/{id}"), None, &[]).0, 401);
    assert_eq!(send(&s, "POST", Some(&key), &format!("/api/keys/{id}"), Some(&json!({"limits": null})), &[]).0, 401);

    // streams are counted too: OpenAI's, Anthropic's and Responses'
    let text = stream(&s, &key, "/v1/chat/completions", &json!({"messages": [{"role": "user", "content": "hi"}], "stream": true}));
    assert!(text.contains("[DONE]"), "{text}");
    let text = stream(&s, &key, "/v1/messages", &json!({"max_tokens": 50, "messages": [{"role": "user", "content": "hi"}], "stream": true}));
    assert!(text.contains("message_stop"), "{text}");
    let text = stream(&s, &key, "/v1/responses", &json!({"input": "hi", "stream": true}));
    assert!(text.contains("response.completed"), "{text}");
    // the owner's own call is counted as the owner's
    assert_eq!(s.post("/v1/chat/completions", &hi()).0, 200);
    // an error is counted as one
    assert_eq!(chat(&s, &key, &json!({"messages": "not a list"})).0, 400);

    // the owner's view: per key, per model, per day
    let (c, u) = s.get(&format!("/api/usage?days=7&key={id}"), true);
    assert_eq!(c, 200);
    assert_eq!(u["by_day"].as_array().unwrap().len(), 7);
    let row = &u["by_key"][0];
    assert_eq!((row["key"].as_str(), row["name"].as_str(), row["kind"].as_str()), (Some(id.as_str()), Some("laptop"), Some("key")));
    assert_eq!(row["requests"], 5, "{u}");
    assert_eq!(row["errors"], 1, "{u}");
    let m = u["by_model"].as_array().unwrap().iter().find(|m| m["model"] == "mock-a").unwrap_or_else(|| panic!("the model: {u}"));
    assert_eq!(m["requests"], 5, "{u}");
    assert!(m["prompt_tokens"].as_u64().unwrap() >= 4 * used.0, "four answers' prompts (the streams' too): {u}");
    assert_eq!(m["completion_tokens"].as_u64().unwrap(), 4 * used.1, "four answers, three of them streamed: {u}");
    assert!(row["last_used"].as_u64().unwrap() > 0);
    assert_eq!(u["by_day"][6]["requests"], 5, "today is the last day: {u}");
    let (_, all) = s.get("/api/usage", true);
    assert_eq!(all["by_day"].as_array().unwrap().len(), 30, "30 days by default");
    assert_eq!(all["totals"]["requests"], 6);
    let owner = all["by_key"].as_array().unwrap().iter().find(|k| k["key"] == "owner").unwrap();
    assert_eq!((owner["requests"].as_u64(), owner["kind"].as_str()), (Some(1), Some("owner")));
    // the list shows each key's use
    let (_, list) = s.get("/api/keys", true);
    let k = &list["keys"][0];
    assert_eq!((k["usage"]["today"]["requests"].as_u64(), k["usage"]["days_30"]["requests"].as_u64()), (Some(5), Some(5)));
    assert_eq!(k["usage"]["models"], json!(["mock-a"]));
    assert!(k["last_used"].as_u64().is_some());

    // a key reads its own usage, and only its own
    let (c, own) = send(&s, "GET", Some(&key), "/v1/usage?days=1", None, &[]);
    assert_eq!(c, 200);
    assert_eq!((own["key"]["name"].as_str(), own["totals"]["requests"].as_u64(), own["today"]["requests"].as_u64()), (Some("laptop"), Some(5), Some(5)));
    assert_eq!(own["by_key"].as_array().unwrap().len(), 1, "{own}");
    assert_eq!(send(&s, "GET", Some(KEY), "/v1/usage", None, &[]).1["key"]["kind"], "owner");
    assert_eq!(send(&s, "GET", None, "/v1/usage", None, &[]).0, 401);
    assert_eq!(send(&s, "GET", Some("am2_notakeynotakeynotakeynotakeynotakeynota"), "/v1/usage", None, &[]).0, 401);

    // changed while it runs: renamed, its tools taken away
    let (c, v) = s.post(&format!("/api/keys/{id}"), &json!({"name": "old laptop", "tools": []}));
    assert_eq!((c, v["name"].as_str()), (200, Some("old laptop")), "{v}");
    assert_eq!(chat(&s, &key, &hi()).0, 200);
    let prompt = tokenizer().decode(&line_ids(s.gen_lines().last().unwrap())).unwrap_or_default();
    assert!(!prompt.contains("web_search"), "no tools now");
    assert_eq!(s.post("/api/keys/nope", &json!({"name": "x"})).0, 404);

    // revoked: refused at once, with the reason; its usage stays; it cannot be revoked by another page
    assert_eq!(send(&s, "DELETE", Some(KEY), &format!("/api/keys/{id}"), None, &[("origin", "http://evil.example")]).0, 403);
    let (c, v) = s.delete(&format!("/api/keys/{id}"));
    assert_eq!((c, v["status"].as_str()), (200, Some("revoked")), "{v}");
    let (c, r) = chat(&s, &key, &hi());
    assert_eq!(c, 401, "{r}");
    assert!(r["error"]["message"].as_str().unwrap().contains("revoked"), "{r}");
    assert_eq!(send(&s, "GET", Some(&key), "/v1/models", None, &[]).0, 401);
    assert_eq!(s.delete("/api/keys/nope").0, 404);
    assert_eq!(s.get(&format!("/api/usage?key={id}"), true).1["totals"]["requests"], 6);
}

#[test]
fn keys_expire_keep_to_their_limits_and_models_and_outlive_a_restart() {
    let mut s = Server::start(&[Model::default(), Model { id: "mock-b", ..Model::default() }], &[reply()], json!({}));

    // a requests limit: the request past it is refused with 429 and the reason
    let (limited, limited_id) = new_key(&s, json!({"name": "two a day", "limits": {"requests_per_day": 2}}));
    assert_eq!(chat(&s, &limited, &hi()).0, 200);
    assert_eq!(chat(&s, &limited, &hi()).0, 200);
    let r = reqwest::blocking::Client::new().post(format!("{}/v1/chat/completions", s.base)).bearer_auth(&limited).json(&hi()).send().unwrap();
    assert_eq!(r.status().as_u16(), 429);
    assert!(r.headers().get("retry-after").is_some(), "says when to come back");
    let v: Value = r.json().unwrap();
    assert!(v["error"]["message"].as_str().unwrap().contains("2 requests for today"), "{v}");
    // what is not a model call is not limited; the refusal is not counted as a request
    assert_eq!(send(&s, "GET", Some(&limited), "/v1/models", None, &[]).0, 200);
    assert_eq!(send(&s, "GET", Some(&limited), "/v1/usage", None, &[]).1["today"]["requests"], 2);
    // the owner raises the limit: it works again without a restart
    assert_eq!(s.post(&format!("/api/keys/{limited_id}"), &json!({"limits": {"requests_per_day": 3}})).0, 200);
    assert_eq!(chat(&s, &limited, &hi()).0, 200);
    assert_eq!(chat(&s, &limited, &hi()).0, 429);

    // a tokens limit
    let (tokens, _) = new_key(&s, json!({"name": "few tokens", "limits": {"tokens_per_day": 5}}));
    assert_eq!(chat(&s, &tokens, &hi()).0, 200, "under the limit when it started");
    let (c, v) = chat(&s, &tokens, &hi());
    assert_eq!(c, 429, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("5 tokens for today"), "{v}");

    // a list of models: another model is refused, with no model the loaded one is checked
    let (only_a, _) = new_key(&s, json!({"name": "only a", "models": ["mock-a"]}));
    assert_eq!(chat(&s, &only_a, &hi()).0, 200);
    assert_eq!(chat(&s, &only_a, &json!({"model": "mock-a", "messages": [{"role": "user", "content": "hi"}]})).0, 200);
    let (c, v) = chat(&s, &only_a, &json!({"model": "mock-b", "messages": [{"role": "user", "content": "hi"}]}));
    assert_eq!(c, 403, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("mock-b"), "{v}");

    // expired: refused with the reason
    let (short, short_id) = new_key(&s, json!({"name": "short", "expires_in": "1s"}));
    assert_eq!(s.post("/api/keys", &json!({"name": "past", "expires_at": 1})).0, 400, "no key is born expired");
    std::thread::sleep(Duration::from_millis(2100));
    let (c, v) = chat(&s, &short, &hi());
    assert_eq!(c, 401, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("expired"), "{v}");
    let (_, list) = s.get("/api/keys", true);
    let row = list["keys"].as_array().unwrap().iter().find(|k| k["id"] == short_id.as_str()).unwrap();
    assert_eq!(row["status"], "expired");
    // the owner gives it more time: it works again
    assert_eq!(s.post(&format!("/api/keys/{short_id}"), &json!({"expires_at": null})).0, 200);
    assert_eq!(chat(&s, &short, &hi()).0, 200);

    // the keys are there after a restart (the limit's count of today comes back with the usage file, written every
    // 20 s and at shutdown; this test stops the server by killing it, so only the keys are checked)
    s.restart();
    assert_eq!(s.get("/api/keys", true).1["keys"].as_array().unwrap().len(), 4);
    assert_eq!(send(&s, "GET", Some(&short), "/v1/models", None, &[]).0, 200);
    assert_eq!(send(&s, "GET", Some(&only_a), "/v1/usage", None, &[]).1["key"]["models"], json!(["mock-a"]));
}

/// A run config's client profile is counted by its name, and its key cannot manage keys either.
#[test]
fn client_profiles_are_counted_by_name_and_cannot_manage_keys() {
    const LAPTOP: &str = "laptop-key-0123456789";
    let m = Model { extra: json!({"clients": {"laptop": {"key": LAPTOP, "tools": []}}}), ..Model::default() };
    let s = Server::start(&[m], &[reply()], json!({}));
    assert_eq!(chat(&s, LAPTOP, &hi()).0, 200);
    assert_eq!(send(&s, "GET", Some(LAPTOP), "/api/keys", None, &[]).0, 401);
    assert_eq!(send(&s, "POST", Some(LAPTOP), "/api/keys", Some(&json!({"name": "mine"})), &[]).0, 401);
    assert_eq!(send(&s, "GET", Some(LAPTOP), "/api/usage", None, &[]).0, 401);
    let (c, own) = send(&s, "GET", Some(LAPTOP), "/v1/usage", None, &[]);
    assert_eq!((c, own["key"]["id"].as_str(), own["totals"]["requests"].as_u64()), (200, Some("client:laptop"), Some(1)), "{own}");
    let (_, all) = s.get("/api/usage", true);
    assert_eq!((all["by_key"][0]["kind"].as_str(), all["by_key"][0]["name"].as_str()), (Some("client"), Some("laptop")), "{all}");
    // a generated key may not take the profile's name
    assert_eq!(s.post("/api/keys", &json!({"name": "laptop"})).0, 400);
}

/// `am2strata-run keys ...` against the running server: create (the key on stdout, once), list, usage, revoke.
#[test]
fn the_command_line_makes_lists_and_revokes_keys() {
    let s = Server::start(&[Model::default()], &[reply()], json!({}));
    let run = |args: &[&str]| {
        let o = std::process::Command::new(env!("CARGO_BIN_EXE_am2strata-run")).arg("keys").args(args)
            .args(["--url", &s.base, "--api-key", KEY, "--models-dir"]).arg(&s.dir).output().expect("am2strata-run");
        (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
    };
    let (ok, out, err) = run(&["create", "--name", "cli key", "--tools", "web,memory", "--expires", "30d", "--requests-per-day", "9"]);
    assert!(ok, "{err}");
    let key = out.trim().to_string();
    assert!(key.starts_with("am2_") && key.len() == 44, "only the key on stdout");
    assert!(err.contains("shown this once") && !err.contains(&key), "{err}");
    assert_eq!(chat(&s, &key, &hi()).0, 200);

    let (ok, out, err) = run(&["list"]);
    assert!(ok, "{err}");
    assert!(out.contains("cli key") && out.contains("active") && out.contains("today 1 req") && out.contains("memory,web") && out.contains("mock-a"), "{out}");
    assert!(!out.contains(&key[8..]), "the list never shows a key");
    let id = out.split_whitespace().next().unwrap().to_string();

    let (ok, out, err) = run(&["usage", "--days", "7", "--key", &id]);
    assert!(ok, "{err}");
    assert!(out.contains("cli key") && out.contains("mock-a") && out.contains("by day:"), "{out}");

    let (ok, out, err) = run(&["revoke", &id]);
    assert!(ok && out.contains("revoked"), "{out} {err}");
    assert_eq!(chat(&s, &key, &hi()).0, 401);
    // mistakes say what to do, and fail
    assert!(!run(&["create"]).0 && !run(&["revoke"]).0 && !run(&["revoke", "nope"]).0 && !run(&["frobnicate"]).0);
    // a key of the wrong kind cannot manage keys
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_am2strata-run")).args(["keys", "list", "--url", &s.base, "--api-key", &key, "--models-dir"])
        .arg(&s.dir).output().unwrap();
    assert!(!o.status.success());
}
