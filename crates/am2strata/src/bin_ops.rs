//! `am2strata-run cluster | tune | clients | memory …`: thin HTTP clients of the server's own API (docs/GAPS-PLAN B).
//! Every big feature is a web app screen and a CLI command; these wrap four the web app never got a card for:
//! managing a cluster, tuning a model with the optimizer, editing client profiles/keys and the memory facts. The
//! request building here reads each server handler to send exactly what it expects (cluster.rs, web.rs optimise,
//! api_keys.rs, clients.rs, memory.rs) and prints the answer; the tests exercise every one against a running test
//! server just like media_cli.rs does.

use std::process::ExitCode;
use std::time::{Duration, SystemTime};

use serde_json::{json, Value};

use clap::Args;

use crate::local_server;

type RB = reqwest::blocking::RequestBuilder;

/// `cluster | tune | clients | memory`: the server (its --url/--api-key, else $AM2STRATA_URL / AM2STRATA_API_KEY,
/// then the local run config's), a banner on stderr (stdout stays the answer) and an HTTP client. A request may set
/// `origin` so the server sees its own page (settings endpoints refuse foreign origins).
fn connect(action: &str, url: Option<String>, api_key: Option<String>, models_dir: &str) -> (reqwest::blocking::Client, String, Option<String>) {
    let (local_url, local_key) = local_server(models_dir);
    let url = url.or_else(|| std::env::var("AM2STRATA_URL").ok()).unwrap_or(local_url).trim_end_matches('/').to_string();
    let key = api_key.or_else(|| std::env::var("AM2STRATA_API_KEY").ok()).or(local_key);
    eprintln!("[am2strata] {action}: {url}"); // on stderr: stdout is the answer, so it can be piped
    (reqwest::blocking::Client::builder().timeout(Duration::from_secs(60)).build().unwrap(), url, key)
}

/// Send `rb` (bearer authed; optionally stamped with `Origin: <url>` so a settings route sees its own page), turning
/// the server's error into its message so it prints plainly and exits non-zero. The builder already carries its client.
fn call(key: &Option<String>, url: &str, rb: RB, origin: bool) -> Result<Value, String> {
    let mut rb = rb;
    if let Some(k) = key { rb = rb.bearer_auth(k); }
    if origin { rb = rb.header("Origin", url); }
    let r = rb.send().map_err(|e| format!("{url} does not answer ({e}); is the server running?"))?;
    if r.status().is_success() { return Ok(r.json().unwrap_or_default()); }
    let j: Value = r.json().unwrap_or_default();
    Err(j["error"]["message"].as_str().unwrap_or("failed").to_string())
}

// --------------------------------------------------------------------------- cluster (cluster.rs)

/// `am2strata-run cluster ...`: the server's /api/cluster as Settings › Cluster uses it. The actions:
///   status  this node, each paired worker (ready, transport, models, loaded, error), or its pairing code on a worker
///   pair <url> <code>  POST /api/cluster/pair {"url","code"} (the master takes the worker; prints paired, node_id)
///   forget <name>  DELETE /api/cluster/workers/{name} (stop using it and tell it to forget this master)
#[derive(Args)]
pub struct ClusterOpts {
    /// status | pair <url> <code> | forget <name>
    pub action: String,
    /// the rest for pair (the worker's url and code) or forget (its name)
    #[arg(last = true, trailing_var_arg = true, allow_hyphen_values = true)]
    pub values: Vec<String>,
    /// print the raw server answer as JSON instead of a friendly rendering
    #[arg(long)]
    pub json: bool,
}

pub fn cluster_cmd(o: ClusterOpts, url: Option<String>, api_key: Option<String>, models_dir: &str) -> ExitCode {
    let (client, base, key) = connect("cluster", url, api_key, models_dir);
    // cluster's /api endpoints refuse foreign origins like the settings ones, so a real client stamps its own Origin.
    let send = |rb: RB| call(&key, &base, rb, true);
    let r = (|| -> Result<(), String> {
        match o.action.as_str() {
            "status" => cmd_cluster_status(&client, &send, &base, o.json)?,
            "pair" => cmd_cluster_pair(&client, &send, &base, &o.values)?,
            "forget" => cmd_cluster_forget(&client, &send, &base, &o.values)?,
            other => return Err(format!("unknown action {other:?}: status, pair or forget")),
        }
        Ok(())
    })();
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("{e}"); ExitCode::from(1) }
    }
}

fn cmd_cluster_status(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<Value, String>, base: &str, json: bool) -> Result<(), String> {
    let j = send(client.get(format!("{base}/api/cluster")))?;
    if json { println!("{}", serde_json::to_string_pretty(&j).map_err(|e| e.to_string())?); return Ok(()); }
    println!("{} ({})", j["node"]["name"].as_str().unwrap_or("?"), j["role"].as_str().unwrap_or("?"));
    // a master with none, or a single machine: nothing to pair with here
    if j["workers"].as_array().map(|a| a.len()).unwrap_or(0) == 0 { println!("no workers"); }
    for w in j["workers"].as_array().cloned().unwrap_or_default() {
        let models = w["models"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default();
        let mut line = format!("- {}: ready={} {} :: models[{}] loaded={}",
            w["name"].as_str().unwrap_or("?"), w["ready"], w["transport"].as_str().unwrap_or(""), models, w["loaded"].as_str().unwrap_or(""));
        if let Some(e) = w["error"].as_str() { line.push_str(&format!(" error: {e}")); }
        println!("{line}");
    }
    // on a worker the pairing code and its masters are what Settings › Cluster shows; print them here too.
    if let Some(code) = j["pairing_code"].as_str() { println!("pairing code: {code}"); }
    for m in j["masters"].as_array().cloned().unwrap_or_default() {
        println!("  master {}: {}", m["id"].as_str().unwrap_or("?"), m["name"].as_str().unwrap_or("?"));
    }
    Ok(())
}

fn cmd_cluster_pair(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<Value, String>, base: &str, values: &[String]) -> Result<(), String> {
    let url = values.first().ok_or("pair needs the worker's url (http://host:port)")?.clone();
    let code = values.get(1).ok_or("pair needs the worker's 6-digit code")?.clone();
    let j = send(client.post(format!("{base}/api/cluster/pair")).json(&json!({"url": url, "code": code})))?;
    println!("paired {}", j["paired"].as_str().unwrap_or("?"));
    println!("node_id {}{}", j["node_id"].as_str().unwrap_or(""), if j["encrypted"] == json!(true) {", encrypted" } else {", plain HTTP"});
    Ok(())
}

fn cmd_cluster_forget(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<Value, String>, base: &str, values: &[String]) -> Result<(), String> {
    let name = values.first().ok_or("forget needs a worker name (from `cluster status`)")?.clone();
    let j = send(client.delete(format!("{base}/api/cluster/workers/{name}")))?;
    if j.get("worker_told") == Some(&json!(true)) { println!("{} forgotten, told it to forget us", name); }
    else { println!("{} forgotten", name); }
    Ok(())
}

// --------------------------------------------------------------------------- tune (web.rs optimise)

/// `am2strata-run tune <model>`: the optimizer's Tune as Settings › Models uses it — POST /api/models/{id}/optimise,
/// follow GET /api/optimise until it ends (the way the app polls), then print the result. --objective is decode
/// (default) or prefill; the run config is never changed.
#[derive(Args)]
pub struct TuneOpts {
    /// a model installed on this server (from Settings › Models)
    pub model: String,
    /// what to make faster: decode (default) or prefill
    #[arg(long, default_value = "decode")]
    pub objective: String,
    /// print the raw server answers as JSON instead of a friendly rendering
    #[arg(long)]
    pub json: bool,
}

pub fn tune_cmd(o: TuneOpts, url: Option<String>, api_key: Option<String>, models_dir: &str) -> ExitCode {
    let (client, base, key) = connect("tune", url, api_key, models_dir);
    // the route refuses foreign origins (web.rs optimise), so stamp its own Origin like the web app does.
    let send = |rb: RB| call(&key, &base, rb, true);
    let r = (|| -> Result<(), String> {
        if o.objective != "decode" && o.objective != "prefill" {
            return Err("tune --objective must be decode or prefill".into());
        }
        // start it: POST accepts immediately and runs the search in the background (chats wait meanwhile).
        let started = send(client.post(format!("{base}/api/models/{}/optimise", o.model)).json(&json!({"objective": o.objective})))?;
        println!("tuning {} ({}): {}", o.model, o.objective, started["state"].as_str().unwrap_or("waiting"));
        // follow its progress until it ends (the web app polls every few seconds), printing only what changes.
        let mut last: Option<String> = None;
        loop {
            std::thread::sleep(Duration::from_secs(2));
            let j = send(client.get(format!("{base}/api/optimise")))?;
            match j["state"].as_str() {
                Some("done") => return print_tune_result(&j),
                Some("error") => return Err(format!("tuning {} stopped: {}", j["model"].as_str().unwrap_or(""), j["error"].as_str().unwrap_or("unknown"))),
                _ => {}
            }
            if let Some(line) = tune_progress(&j, o.json) {
                if last.as_deref() != Some(line.as_str()) { println!("{line}"); last = Some(line); }
            }
        }
    })();
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("{e}"); ExitCode::from(1) }
    }
}

/// What to print while running: the current trial and its last score, once each (the web app's "trial N · last X").
fn tune_progress(j: &Value, json: bool) -> Option<String> {
    if json { return None; } // the raw answer already shows every field; printing it again per poll is noise
    let trials = j["trials"].as_u64().unwrap_or(0);
    Some(format!("trial {} · last {}", trials, tune_tok(&j["last"])))
}

/// The finished tune: the tuned/baseline speed and whether a setting gained enough to be kept (web.rs optimise_status).
fn print_tune_result(j: &Value) -> Result<(), String> {
    let model = j["model"].as_str().unwrap_or("");
    if j["enabled"] == json!(true) {
        println!("{model}: {t} tok/s, was {b} — a setting gained enough, used from the next load", t = tune_tok(&j["tuned"]), b = tune_tok(&j["baseline"]));
    } else if j.get("gain").is_some() {
        println!("{model}: no setting gained 3% (was {}); the run config stays in use", tune_tok(&j["baseline"]));
    } else if let Some(e) = j["error"].as_str() {
        return Err(format!("tuning stopped: {e}"));
    }
    Ok(())
}

/// The tuned model's speed for the objective, whichever of decode/prefill tok/s is present.
fn tune_tok(s: &Value) -> String {
    match (s["decode_tok_s"].as_f64(), s["prefill_tok_s"].as_f64()) {
        (Some(t), _) => format!("{:.0}", t),
        (_, Some(t)) => format!("{:.0}", t),
        (None, None) => "…".to_string(),
    }
}

// --------------------------------------------------------------------------- clients + keys (api_keys.rs, clients.rs)

/// `am2strata-run clients ...`: Settings › Clients through the running server. The actions:
///   list  every client profile (GET /api/clients) and every key with its tools, expiry and status (GET /api/keys)
///   new <name> [--tools a,b] [--expires 30d]  POST /api/keys — the new key is shown once and kept nowhere
///   edit <id> [--name n] [--tools a,b] [--expires 30d|never]  POST /api/keys/{id} — change name, tools or expiry
///   revoke <id>  DELETE /api/keys/{id} (the key stops working at once; its usage stays)
#[derive(Args)]
pub struct ClientsOpts {
    /// list | new <name> | edit <id> | revoke <id>
    pub action: String,
    /// the rest for new (the name), edit (the key's id) or revoke (its id, from `clients list`)
    #[arg(last = true, trailing_var_arg = true, allow_hyphen_values = true)]
    pub values: Vec<String>,
    /// new, edit: the new name
    #[arg(long)]
    pub name: Option<String>,
    /// new, edit: the tool sets, comma separated (memory, web, image, video, speech, music, browser, code)
    #[arg(long)]
    pub tools: Option<String>,
    /// new, edit: how long it works (30d, 12h, 90m); "never" to clear an expiry when editing; until revoked when omitted
    #[arg(long)]
    pub expires: Option<String>,
    /// new, edit: requests a day this key may make; "none" for no limit
    #[arg(long)]
    pub requests_per_day: Option<String>,
    /// new, edit: tokens a day this key may use; "none" for no limit
    #[arg(long)]
    pub tokens_per_day: Option<String>,
    /// new, edit: the models this key may use, comma separated; "all" for every model
    #[arg(long)]
    pub models: Option<String>,
    /// print the raw server answers as JSON instead of a friendly rendering
    #[arg(long)]
    pub json: bool,
}

pub fn clients_cmd(o: ClientsOpts, url: Option<String>, api_key: Option<String>, models_dir: &str) -> ExitCode {
    let (client, base, key) = connect("clients", url, api_key, models_dir);
    // keys are made and revoked over POST/DELETE, which refuse foreign origins; stamp the own page for them.
    let send = |rb: RB| call(&key, &base, rb, true);
    let r = (|| -> Result<(), String> {
        match o.action.as_str() {
            "list" => cmd_clients_list(&client, &send, &base, o.json)?,
            "new" => cmd_clients_new(&client, &send, &base, &o)?,
            "edit" => cmd_clients_edit(&client, &send, &base, &o)?,
            "revoke" => cmd_clients_revoke(&client, &send, &base, &o.values)?,
            other => return Err(format!("unknown action {other:?}: list, new, edit or revoke")),
        }
        Ok(())
    })();
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("{e}"); ExitCode::from(1) }
    }
}

fn cmd_clients_list(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<Value, String>, base: &str, json: bool) -> Result<(), String> {
    let profiles = send(client.get(format!("{base}/api/clients")))?;
    if json { println!("{}", serde_json::to_string_pretty(&profiles).map_err(|e| e.to_string())?); }
    else {
        println!("client profiles:");
        for p in profiles["profiles"].as_array().cloned().unwrap_or_default() {
            println!("  - {} : tools[{}]", p["name"].as_str().unwrap_or("?"),
                p["tools"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default());
        }
    }
    let j = send(client.get(format!("{base}/api/keys")))?;
    if json { println!("{}", serde_json::to_string_pretty(&j).map_err(|e| e.to_string())?); return Ok(()); }
    println!("keys:");
    for k in j["keys"].as_array().cloned().unwrap_or_default() {
        // name, tools, expiry and status — never a key's value (the list holds none).
        let tools = k["tools"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default();
        let expiry = if let Some(e) = k["expires_at"].as_u64() { expires_at(e) } else { "until revoked".to_string() };
        println!("  {}  {}  tools[{}]  {:<8} expires {}", k["id"].as_str().unwrap_or(""), k["name"].as_str().unwrap_or(""), tools, k["status"].as_str().unwrap_or(""), expiry);
    }
    Ok(())
}

/// The fields `new` and `edit` share, checked before anything is sent (Kilo on #72): a name and a tool list are never
/// empty; limits are whole numbers above zero or "none"; models a list or "all". The same rules and fields as the web form.
fn key_spec(o: &ClientsOpts, name: Option<&str>) -> Result<Value, String> {
    let mut spec = json!({});
    if let Some(n) = name {
        let n = n.trim();
        if n.is_empty() { return Err("the name cannot be empty".into()); }
        spec["name"] = json!(n);
    }
    if let Some(t) = &o.tools {
        let list: Vec<&str> = t.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
        if list.is_empty() { return Err("--tools needs at least one tool set (memory, web, image, video, speech, music, browser, code)".into()); }
        spec["tools"] = json!(list);
    }
    let limit = |v: &str, what: &str| -> Result<Value, String> {
        if v.eq_ignore_ascii_case("none") { return Ok(Value::Null); }
        match v.trim().parse::<u64>() { Ok(n) if n > 0 => Ok(json!(n)), _ => Err(format!("--{what}: a whole number above zero, or none")) }
    };
    let mut limits = serde_json::Map::new();
    if let Some(v) = &o.requests_per_day { limits.insert("requests_per_day".into(), limit(v, "requests-per-day")?); }
    if let Some(v) = &o.tokens_per_day { limits.insert("tokens_per_day".into(), limit(v, "tokens-per-day")?); }
    if !limits.is_empty() { spec["limits"] = Value::Object(limits); }
    if let Some(m) = &o.models {
        let list: Vec<&str> = m.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
        spec["models"] = if list.is_empty() || (list.len() == 1 && list[0].eq_ignore_ascii_case("all")) { Value::Null } else { json!(list) };
    }
    Ok(spec)
}

fn cmd_clients_new(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<Value, String>, base: &str, o: &ClientsOpts) -> Result<(), String> {
    // the name: the word after `new`, or --name (not both, unless they agree)
    let name = match (o.values.first(), o.name.as_deref()) {
        (Some(a), Some(b)) if a != b => return Err(format!("two names given ({a:?} and --name {b:?}): give one")),
        (Some(a), _) => a.clone(),
        (None, Some(b)) => b.to_string(),
        (None, None) => return Err("new needs a name: clients new <name> (or --name <name>)".into()),
    };
    // send only the fields given; the server fills defaults.
    let mut spec = key_spec(o, Some(&name))?;
    if let Some(e) = &o.expires { spec["expires_in"] = json!(e); }
    let j = send(client.post(format!("{base}/api/keys")).json(&spec))?;
    println!("{}", j["key"].as_str().unwrap_or("")); // the new key: stdout, so it can be copied
    eprintln!("key {} ({}) made — shown only this once and kept nowhere; copy it now", j["id"].as_str().unwrap_or(""), j["name"].as_str().unwrap_or("?"));
    Ok(())
}

fn cmd_clients_revoke(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<Value, String>, base: &str, values: &[String]) -> Result<(), String> {
    let id = values.first().ok_or("revoke needs a key id (from `clients list`)")?.clone();
    let j = send(client.delete(format!("{base}/api/keys/{id}")))?;
    println!("key {} ({}) is revoked: it stopped working now", j["id"].as_str().unwrap_or(&id), j["name"].as_str().unwrap_or("?"));
    Ok(())
}

fn cmd_clients_edit(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<Value, String>, base: &str, o: &ClientsOpts) -> Result<(), String> {
    let id = o.values.first().ok_or("edit needs a key id (from `clients list`)")?.clone();
    let json = o.json;
    // send only the fields given (a null clears), like new; "never" clears an expiry. The change takes effect at once.
    let mut spec = key_spec(o, o.name.as_deref())?;
    if spec.as_object().is_some_and(|m| m.is_empty()) && o.expires.is_none() {
        return Err("nothing to change: give --name, --tools, --expires, --requests-per-day, --tokens-per-day or --models".into());
    }
    if let Some(e) = o.expires.as_deref() {
        if e.eq_ignore_ascii_case("never") { spec["expires_at"] = json!(null); } else { spec["expires_in"] = json!(e); }
    }
    let j = send(client.post(format!("{base}/api/keys/{id}")).json(&spec))?;
    if json { println!("{}", serde_json::to_string_pretty(&j).map_err(|e| e.to_string())?); return Ok(()); }
    // the updated record: id, name, tools and status — never a key's value (the store holds none of it).
    let tools = j["tools"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default();
    let expiry = if let Some(e) = j["expires_at"].as_u64() { expires_at(e) } else { "until revoked".to_string() };
    println!("  {}  {}  tools[{}]  {:<8} expires {}", j["id"].as_str().unwrap_or(&id), j["name"].as_str().unwrap_or("?"), tools, j["status"].as_str().unwrap_or(""), expiry);
    Ok(())
}

/// `expires_at` (unix seconds) from now, in whole days when far off — the "30d"-style the list shows.
fn expires_at(unix: u64) -> String {
    let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    if unix <= now { return "expired".to_string() }
    format!("{} days", (unix.saturating_sub(now) + 43_200) / 86_400) // round to the nearest day
}

// --------------------------------------------------------------------------- memory (memory.rs)

/// `am2strata-run memory ...`: the stored facts as chat uses /v1/memory. The actions:
///   list [--query q]  GET /v1/memory (its own space's facts, and the ones kept for everyone)
///   add "<text>"  POST /v1/memory {"fact": "..."}
///   forget <id>  DELETE /v1/memory/{id}
#[derive(Args)]
pub struct MemoryOpts {
    /// list | add "<text>" | forget <id>
    pub action: String,
    /// add: the fact's text; forget: the id from `memory list`
    #[arg(last = true, trailing_var_arg = true, allow_hyphen_values = true)]
    pub values: Vec<String>,
    /// list: only facts whose text matches (server-side)
    #[arg(long)]
    pub query: Option<String>,
    /// print the raw server answer as JSON instead of a friendly rendering
    #[arg(long)]
    pub json: bool,
}

pub fn memory_cmd(o: MemoryOpts, url: Option<String>, api_key: Option<String>, models_dir: &str) -> ExitCode {
    let (client, base, key) = connect("memory", url, api_key, models_dir);
    // /v1/memory needs no own page (it is a client's own data), so Origin is left off like other GETs.
    let send = |rb: RB| call(&key, &base, rb, false);
    let r = (|| -> Result<(), String> {
        match o.action.as_str() {
            "list" => cmd_memory_list(&client, &send, &base, o.query.as_deref(), o.json)?,
            "add" => cmd_memory_add(&client, &send, &base, &o.values)?,
            "forget" => cmd_memory_forget(&client, &send, &base, &o.values)?,
            other => return Err(format!("unknown action {other:?}: list, add or forget")),
        }
        Ok(())
    })();
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("{e}"); ExitCode::from(1) }
    }
}

fn cmd_memory_list(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<Value, String>, base: &str, query: Option<&str>, json: bool) -> Result<(), String> {
    let mut rb = client.get(format!("{base}/v1/memory"));
    if let Some(q) = query { rb = rb.query(&[("query", q)]); }
    let j = send(rb)?;
    if json { println!("{}", serde_json::to_string_pretty(&j).map_err(|e| e.to_string())?); return Ok(()); }
    for f in j["data"].as_array().cloned().unwrap_or_default() {
        println!("{}  {}", f["id"].as_str().unwrap_or(""), f["text"].as_str().unwrap_or(""));
    }
    Ok(())
}

fn cmd_memory_add(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<Value, String>, base: &str, values: &[String]) -> Result<(), String> {
    let text = values.join(" ");
    if text.trim().is_empty() { return Err("add needs the fact's text (quote it)".into()); }
    let j = send(client.post(format!("{base}/v1/memory")).json(&json!({"fact": text})))?;
    println!("saved {}", j["id"].as_str().unwrap_or("?"));
    Ok(())
}

fn cmd_memory_forget(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<Value, String>, base: &str, values: &[String]) -> Result<(), String> {
    let id = values.first().ok_or("forget needs a fact id (from `memory list`)")?.clone();
    send(client.delete(format!("{base}/v1/memory/{id}")))?;
    println!("{} forgotten", id);
    Ok(())
}
