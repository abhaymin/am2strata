//! The tools this server runs itself, for every API (OpenAI chat, Anthropic messages, Responses): the config's MCP
//! servers (`"strata_mcp": true`, this server's own page only), the memory tools (a request with memory, when the
//! settings allow them) and the media tools (`"strata_media": true`). They are added to the request's tool list;
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
    // MCP tools run with the user's rights on this PC, so only for this server's own page
    if let (Some(true), Some(hub)) = (req.get("strata_mcp").and_then(Value::as_bool), s.mcp.clone()) {
        if crate::foreign_origin(s, headers) {
            return Err(crate::not_own_page("MCP tools can be used"));
        }
        hub.wait(Duration::from_secs(10)).await; // servers still starting (only right after the start)
        let extra = hub.template_tools(&own_names(prep));
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
    // generate_image / generate_video / speak, for the kinds this PC can make; never for another web page
    if req.get("strata_media") == Some(&Value::Bool(true)) && !structured && !forced && s.media.available()
        && !crate::foreign_origin(s, headers) {
        let caps = s.media_jobs.capabilities().await;
        let own = own_names(prep);
        let extra: Vec<Value> = crate::media_jobs::Jobs::tools(&caps).into_iter().filter(|t| !own.iter().any(|o| t["name"] == o.as_str())).collect();
        if !extra.is_empty() {
            add(prep, &mut offered, extra);
            opts.media = Some(caps);
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
