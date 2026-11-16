//! `am2strata-run cluster | tune | clients | memory …` against a running test server (docs/GAPS-PLAN B). The CLI is an
//! exact copy of what a user runs: the binary run with --url/--api-key, one test Server per case. It covers the four
//! CLI columns' contract: memory add/list/forget round-trip; clients new shows a key once and never its value, list
//! shows the name, edit changes a running key's tools (and that change takes effect for the key), and revoke marks it revoked; cluster status says "no workers" on a single machine and `pair` refuses
//! a wrong code (and takes a worker for the right one); tune reports an unknown model as the server's error.

mod common;

use common::*;
use serde_json::json;

/// Run `am2strata-run <cmd> …` exactly as a user would: the --url/--api-key go first (the per-action values follow),
/// which clap needs because each command's trailing values are read after a literal `--`. Returns success, stdout and
/// stderr. The server's own models folder is passed as --models-dir like media_cli.rs; --url/--api-key win over it.
fn run(server: &Server, cmd: &str, args: &[&str]) -> (bool, String, String) {
    let mut full: Vec<String> = vec![
        cmd.into(), "--models-dir".into(), server.dir.to_string_lossy().into_owned(),
        "--url".into(), server.base.clone(), "--api-key".into(), KEY.into(),
    ];
    for a in args { full.push(a.to_string()); }
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_am2strata-run")).args(&full).output().expect("am2strata-run");
    (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn find_line(out: &str, needle: &str) -> Option<String> {
    out.lines().find(|l| l.contains(needle)).map(|l| l.trim().to_string())
}

/// memory add, then list shows it, then forget removes it (memory.rs /v1/memory).
#[test]
fn memory_add_list_and_forget_round_trip() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));

    // add: the fact is saved and an id comes back.
    let text = "the am2strata sky is blue";
    let (ok, out, err) = run(&s, "memory", &["add", "--", text]);
    assert!(ok && !out.is_empty(), "add printed nothing: {out:?} stderr {err:?}");
    let id = find_line(&out, "saved").map(|l| l.trim_start_matches("saved ").trim().to_string()).unwrap_or_default();
    assert!(!id.is_empty(), "save reports an id, not {out:?}");

    // list: the fact is there now.
    let (_, lout, _) = run(&s, "memory", &["list"]);
    assert!(lout.contains(text), "list shows the added fact: {lout:?}");

    // forget it, then list no longer carries it.
    let (ok2, fout, _) = run(&s, "memory", &["forget", "--", &id]);
    assert!(ok2 && find_line(&fout, "forgotten").is_some(), "forget confirms: {fout:?}");
    let (_, lout2, _) = run(&s, "memory", &["list"]);
    assert!(!lout2.contains(text), "the fact is gone after forget: {lout2:?}");
}

/// clients new prints a key once; list shows the name and never the value; revoke marks it revoked. (api_keys.rs)
#[test]
fn clients_new_shows_a_key_once_then_list_and_revoke() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));

    // new: the key is printed once (stdout), a line says so (stderr), and it is never kept beyond this.
    let (_, nout, nerr) = run(&s, "clients", &["new", "--tools", "web,image", "--expires", "30d", "--", "laptop"]);
    assert!(!nout.trim().is_empty() && !nout.starts_with("[am2strata]"), "the new key is printed on stdout, not {nout:?}");
    assert!(nerr.contains("this once"), "it is told to be copied now: {nerr:?}");

    // list shows the name and its tools/expiry, but never the key's value.
    let (_, lout, _) = run(&s, "clients", &["list"]);
    assert!(lout.contains("laptop"), "the new key is listed by name: {lout:?}");
    assert!(!lout.contains(nout.trim()), "its value is never shown: {lout:?}");
    // the id in the list row (first column) is what revoke needs.
    let id = find_line(&lout, "laptop").map(|l| l.split_whitespace().next().unwrap_or("").to_string()).unwrap_or_default();
    assert!(!id.is_empty(), "a key id is printed before its name: {lout:?}");

    // revoke: it stops working; the next list shows it revoked.
    let (_, rout, _) = run(&s, "clients", &["revoke", "--", &id]);
    assert!(rout.trim().contains("revoked"), "revoke confirms: {rout:?}");
    let (_, lout2, _) = run(&s, "clients", &["list"]);
    assert!(find_line(&lout2, "laptop").unwrap_or_default().split_whitespace().any(|w| w == "revoked"),
        "the key is now revoked: {lout2:?}");
}

/// clients edit changes a running key's tools with the CLI and the change takes effect for that key (api_keys.rs).
#[test]
fn clients_edit_changes_the_tools_and_uses_them() {
    let s = Server::start(&[Model::default()], &[ids("Hi.\n</think>\n\nhello")], json!({}));

    // make a key with one tool set (memory); keep its value and id for the steps that follow.
    let (c, made) = s.post("/api/keys", &json!({"name": "editor", "tools": ["memory"]}));
    assert_eq!(c, 201, "{made}");
    let key = made["key"].as_str().unwrap().to_string();
    let id = made["id"].as_str().unwrap().to_string();

    // edit its tools with the CLI: web instead of memory. It prints the updated record (a URL banner goes to stderr, as always);
    // stdout never carries the key's value.
    let (ok, out, _) = run(&s, "clients", &["edit", "--tools", "web", "--", &id]);
    assert!(ok && !out.is_empty(), "edit printed the updated record: {out:?}");
    assert!(out.contains("editor") && out.contains("web") && !out.contains("memory"), "the CLI shows the updated tools: {out:?}");
    assert!(!out.contains(&key), "the printed record never carries a key's value: {out:?}");

    // the list shows the new set now and no longer the old one (first column is the id, like the existing clients test).
    let (_, lout, _) = run(&s, "clients", &["list"]);
    assert!(find_line(&lout, "editor").unwrap_or_default().contains("web"), "the list shows the new tools: {lout:?}");

    // use the key on a chat with strata_web left out; its profile now offers web_search (and no longer remember).
    let r = reqwest::blocking::Client::new().post(format!("{}/v1/chat/completions", s.base))
        .bearer_auth(&key).json(&json!({"messages": [{"role": "user", "content": "hi"}]})).send().unwrap();
    assert_eq!(r.status().as_u16(), 200, "the edited key still answers: {}", r.text().unwrap());
    let prompt = tokenizer().decode(&line_ids(s.gen_lines().last().unwrap())).unwrap_or_default();
    assert!(prompt.contains("web_search"), "the profile offers the new set (web): {prompt}");
    assert!(!prompt.contains("remember"), "the old tool set is gone: {prompt}");
}

/// clients new/edit set the same fields as the web form (daily limits, allowed models), clear them with none/all, and
/// refuse an empty name or tool list before sending anything (Kilo on #72).
#[test]
fn clients_set_limits_and_models_and_refuse_empty_values() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    // new by --name, with limits and models
    let (ok, _, nerr) = run(&s, "clients", &["new", "--name", "limited", "--requests-per-day", "50", "--tokens-per-day", "9000", "--models", "a,b"]);
    assert!(ok, "new with --name and limits: {nerr}");
    let (_, keys) = s.get("/api/keys", true);
    let rec = keys["keys"].as_array().unwrap().iter().find(|k| k["name"] == "limited").cloned().unwrap_or_default();
    assert_eq!(rec["limits"]["requests_per_day"], 50, "{rec}");
    assert_eq!(rec["limits"]["tokens_per_day"], 9000, "{rec}");
    assert_eq!(rec["models"], json!(["a", "b"]), "{rec}");
    let id = rec["id"].as_str().unwrap().to_string();

    // edit clears a limit and the model list
    let (ok, _, eerr) = run(&s, "clients", &["edit", "--requests-per-day", "none", "--models", "all", "--", &id]);
    assert!(ok, "edit clearing limits: {eerr}");
    let (_, keys) = s.get("/api/keys", true);
    let rec = keys["keys"].as_array().unwrap().iter().find(|k| k["id"] == id.as_str()).cloned().unwrap_or_default();
    assert!(rec["limits"]["requests_per_day"].is_null() && rec["limits"]["tokens_per_day"] == 9000, "{rec}");
    assert!(rec["models"].is_null(), "all models again: {rec}");

    // refused before sending: an empty tool list, an empty name, a zero limit, two different names, nothing to change
    for (args, want) in [(vec!["edit", "--tools", " , ", "--", id.as_str()], "at least one tool"),
                         (vec!["edit", "--name", " ", "--", id.as_str()], "name cannot be empty"),
                         (vec!["edit", "--tokens-per-day", "0", "--", id.as_str()], "positive whole number"),
                         (vec!["new", "--name", "one", "--", "two"], "two names"),
                         (vec!["edit", "--", id.as_str()], "nothing to change")] {
        let (ok, _, err) = run(&s, "clients", &args);
        assert!(!ok && err.contains(want), "{args:?} should be refused with {want:?}: {err}");
    }
}

/// cluster status on a single machine says "no workers"; pair refuses a wrong code and exits non-zero. (cluster.rs)
#[test]
fn cluster_status_has_no_workers_and_pair_refuses_a_wrong_code() {
    // single machine: this node, no paired worker.
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    let (ok, out, _) = run(&s, "cluster", &["status"]);
    assert!(ok && find_line(&out, "no workers").is_some(), "a single machine has no worker: {out:?}");

    // a worker and a master (as server.rs's a_worker_pairs_by_its_code_and_is_removed_cleanly set up): the master
    // refuses to pair for a wrong code, with the server's own message and a non-zero exit.
    let worker = Server::start(&[Model { id: "mock-b", extra: json!({"cluster": {"role": "worker", "allow_plain": true, "tls_port": free_port()}}), ..Model::default() }],
                               &[ids("Ok.\n</think>\n\nFrom the paired worker.")], json!({}));
    let master = Server::start(&[Model::default()], &[ids("Ok.\n</think>\n\nFrom the master.")], json!({}));

    // the worker's pairing code, read back through its own `cluster status`.
    let (wok, wout, _) = run(&worker, "cluster", &["status"]);
    assert!(wok, "the worker's status works: {wout:?}");
    let code_line = find_line(&wout, "pairing code");
    let code = code_line.as_deref().and_then(|s| s.split(':').nth(1)).map(|c| c.trim().to_string()).unwrap_or_default();
    assert_eq!(code.len(), 6, "the worker reports a 6-digit code, not {code:?}");

    // a wrong code: the server refuses and the CLI exits non-zero with that refusal on stderr.
    let wrong = if code == "000000" { "000001" } else { "000000" };
    let (rok, rout, rerr) = run(&master, "cluster", &["pair", "--", &worker.base, wrong]);
    assert!(!rok && !rerr.is_empty(), "wrong code exits non-zero: out {rout:?} err {rerr:?}");
    assert!(rerr.to_lowercase().contains("worker"), "the refusal names the worker's side: {rerr:?}");

    // the right code pairs: the master keeps the worker and reports it.
    let (ok3, pout, _) = run(&master, "cluster", &["pair", "--", &worker.base, &code]);
    assert!(ok3 && find_line(&pout, "paired").is_some(), "the correct code pairs: {pout:?}");
}

/// tune on an unknown model is the server's error and a non-zero exit (web.rs optimise). (a successful tune would be
/// slow to load the model in a test, so the failure path is what is checked here.)
#[test]
fn tune_unknown_model_reports_the_servers_error() {
    let s = Server::start(&[Model::default()], &[ids("ok")], json!({}));
    // POST /api/models/no-such-model/optimise is refused before any background search starts, so the CLI exits at once.
    let (ok, out, err) = run(&s, "tune", &["no-such-model"]);
    assert!(!ok && !err.is_empty(), "unknown model exits non-zero: {out:?} err {err:?}");
    assert!(err.contains("no installed model"), "the server's refusal is shown verbatim: {err:?}");
}
