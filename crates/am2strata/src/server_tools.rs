//! The tools this server runs itself, for every API (OpenAI chat, Anthropic messages, Responses): the config's MCP
//! servers (`"strata_mcp": true`, this server's own page only), the memory tools (a request with memory, when the
//! settings allow them), the media tools (`"strata_media": true`) and web search (`"strata_web": true`), or, with no
//! flag at all, the sets of the key's client profile (clients.rs; `x-strata-tools` narrows them for one request).
//! They are added to the request's tool list;
//! the model's calls to them are run in run::start_mcp's loop and the rest of the turn goes on. The request's own
//! tools win a name clash, and a forced tool_choice is left alone.

use crate::run::{self, Refusal, RunOpts, Started};
use crate::{error, AppState};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use strata_frontend::request::forced_call;
use strata_frontend::Prepared;

/// The server's tools a request was given: the MCP hub (when MCP tools were) and the names of every tool run here.
pub type Offered = Option<(Option<Arc<crate::mcp::McpHub>>, HashSet<String>)>;

fn own_names(prep: &Prepared) -> Vec<String> {
    prep.tools.iter().flatten().filter_map(|t| t.get("name").and_then(Value::as_str).map(str::to_owned)).collect()
}

fn add(prep: &mut Prepared, offered: &mut Offered, extra: Vec<Value>) {
    let (hub, mut names) = offered.take().unwrap_or((None, HashSet::new()));
    names.extend(extra.iter().filter_map(|t| t["name"].as_str().map(str::to_owned)));
    prep.tools.get_or_insert_with(Vec::new).extend(extra);
    *offered = Some((hub, names));
}

/// Add the server's tools this request opted in to. `structured`: a checked response_format (no tools then).
pub async fn offer(s: &AppState, headers: &HeaderMap, req: &Value, prep: &mut Prepared, opts: &mut RunOpts, structured: bool)
                   -> Result<Offered, Response> {
    let mut offered: Offered = None;
    // the key's client profile (the main key's "default" one, if the run config has it), narrowed by x-strata-tools
    let mut profile = crate::api_access(s, headers).flatten();
    if let Some(h) = headers.get("x-strata-tools").and_then(|v| v.to_str().ok()) {
        profile = match profile {
            Some(p) => Some(p.narrowed(h).map_err(|e| error(StatusCode::FORBIDDEN, "permission_error", e))?),
            None => return Err(error(StatusCode::FORBIDDEN, "permission_error",
                                     "x-strata-tools: this key has no client profile (the run config's \"clients\" gives keys their tools)")),
        };
    }
    let has = |set: &str| profile.as_ref().is_some_and(|p| p.has(set));
    let client = profile.as_ref().filter(|p| p.name != "default");
    // memory: a client profile's requests use its own space, and only when the profile has memory
    if let Some(p) = client {
        opts.memory = p.has("memory").then(|| p.memory_space());
    } else if opts.memory.is_none() && has("memory") {
        opts.memory = Some(profile.as_ref().map(|p| p.memory_space()).filter(|m| m != "default").unwrap_or_else(|| "api".into()));
    } else if opts.memory.is_none() && !headers.contains_key("origin") && req.get("strata_memory") != Some(&Value::Bool(false))
        && s.mgr.memory.settings.read().unwrap().api {
        // an API client with the main key (no web page): memory in the "api" space unless it opts out
        opts.memory = Some("api".into());
    }
    // MCP tools run with the user's rights on this PC: every server for this server's own page (strata_mcp), or the
    // servers a profile names (address servers with mcp:*; a program server only by its name); never another web page
    let flag_mcp = req.get("strata_mcp").and_then(Value::as_bool) == Some(true);
    let (mcp_names, mcp_all) = profile.as_ref().map(|p| p.mcp()).unwrap_or_default();
    if let (true, Some(hub)) = (flag_mcp || !mcp_names.is_empty() || mcp_all, s.mcp.clone()) {
        if crate::foreign_origin(s, headers) {
            return Err(crate::not_own_page("MCP tools can be used"));
        }
        hub.wait(Duration::from_secs(10)).await; // servers still starting (only right after the start)
        let allowed = |srv: &crate::mcp::McpServer| flag_mcp || mcp_names.contains(&srv.name) || (mcp_all && !srv.is_command());
        let extra = hub.template_tools_of(&own_names(prep), Some(&allowed));
        if !extra.is_empty() {
            if forced_call(opts.tool_choice.as_ref(), prep.tools.as_deref()).is_some() {
                return Err(error(StatusCode::BAD_REQUEST, "invalid_request_error", "a forced tool_choice with MCP tools is not supported"));
            }
            let names = extra.iter().filter_map(|t| t["name"].as_str().map(str::to_owned)).collect();
            prep.tools.get_or_insert_with(Vec::new).extend(extra);
            offered = Some((Some(hub), names));
        }
    }
    let forced = forced_call(opts.tool_choice.as_ref(), prep.tools.as_deref()).is_some();
    // remember / recall / forget, for a request with memory
    if opts.memory.is_some() && s.mgr.memory.settings.read().unwrap().tools && !structured && !forced {
        let own = own_names(prep);
        let extra: Vec<Value> = crate::memory::tools().into_iter().filter(|t| !own.iter().any(|o| t["name"] == o.as_str())).collect();
        if !extra.is_empty() {
            add(prep, &mut offered, extra);
        }
    }
    // generate_image / generate_video / speak / make_song, for the kinds this PC can make (all with strata_media,
    // else the profile's); never for another web page
    let flag_media = req.get("strata_media") == Some(&Value::Bool(true));
    let kinds: Vec<&str> = [("image", "image"), ("video", "video"), ("speech", "tts"), ("music", "music")].iter()
        .filter(|(set, _)| flag_media || has(set)).map(|(_, k)| *k).collect();
    if !kinds.is_empty() && !structured && !forced && s.media.available() && !crate::foreign_origin(s, headers) {
        let mut caps = s.media_jobs.capabilities().await;
        if let Some(all) = caps["kinds"].as_object_mut() {
            for (k, v) in all.iter_mut() {
                if !kinds.contains(&k.as_str()) {
                    v["usable"] = Value::Bool(false);
                }
            }
        }
        let own = own_names(prep);
        let extra: Vec<Value> = crate::media_jobs::Jobs::tools(&caps).into_iter().filter(|t| !own.iter().any(|o| t["name"] == o.as_str())).collect();
        if !extra.is_empty() {
            add(prep, &mut offered, extra);
            // defer (the default) or swap: the request's own strata_media_in_turn, else the key's profile's
            let mode = req.get("strata_media_in_turn").and_then(Value::as_str)
                .or(profile.as_ref().and_then(|p| p.media_in_turn.as_deref())).unwrap_or("defer");
            caps["in_turn"] = serde_json::json!(if mode == "swap" { "swap" } else { "defer" });
            opts.media = Some(caps);
        }
    }
    // the browser (computer use): only for a request that asks, never for another web page
    if (req.get("strata_browser") == Some(&Value::Bool(true)) || has("browser")) && !structured && !forced && !crate::foreign_origin(s, headers) {
        if let Some(b) = s.mgr.browser.get().filter(|b| b.available()) {
            let vision = match s.mgr.current().await { Some(l) => l.vision.is_some(), None => false };
            let own = own_names(prep);
            let extra: Vec<Value> = crate::browser::Browsers::tools(vision, b.user_session()).into_iter().filter(|t| !own.iter().any(|o| t["name"] == o.as_str())).collect();
            if !extra.is_empty() {
                add(prep, &mut offered, extra);
            }
        }
    }
    // run_python in the sandbox: only for a profile with "code" (a person on the page runs code with its Run button)
    if has("code") && !structured && !forced && s.code_run.enabled && !crate::foreign_origin(s, headers) {
        let own = own_names(prep);
        let extra: Vec<Value> = crate::code_run::tools(&s.code_run).into_iter().filter(|t| !own.iter().any(|o| t["name"] == o.as_str())).collect();
        if !extra.is_empty() {
            add(prep, &mut offered, extra);
        }
    }
    // Chat Completions' web_search_options: web search for this request, which a key's profile must allow
    if req.get("web_search_options").is_some_and(|o| !o.is_null()) && !has("web") && req.get("strata_web") != Some(&Value::Bool(true)) {
        return Err(error(StatusCode::BAD_REQUEST, "invalid_request_error",
                         "web_search_options: web search is answered for a key whose client profile has \"web\" (the run config's \"clients\")"));
    }
    // web_search / fetch_url: the queries leave this PC, so only for a request that asks (or a profile with "web")
    if (req.get("strata_web") == Some(&Value::Bool(true)) || has("web")) && !structured && !forced {
        if let Some(w) = s.mgr.web.get() {
            let own = own_names(prep);
            let extra: Vec<Value> = w.tools().into_iter().filter(|t| !own.iter().any(|o| t["name"] == o.as_str())).collect();
            if !extra.is_empty() {
                add(prep, &mut offered, extra);
            }
        }
    }
    Ok(offered)
}

/// Start the request: with the server's tools, in the tool loop; else as it is.
pub async fn start(mgr: Arc<crate::manager::Manager>, loaded: Arc<crate::manager::Loaded>, prep: Prepared, req: Value, opts: RunOpts,
                   offered: Offered) -> Result<Started, Refusal> {
    match offered {
        Some((hub, names)) => run::start_mcp(mgr, loaded, prep, req, opts, hub, names).await,
        None => run::start(mgr, loaded, prep, req, opts).await,
    }
}
