//! Settings › Tools: the MCP servers added, changed, tested and removed from the web app (docs/MCP-UI-PLAN.md).
//!
//! - GET    /api/mcp/servers               every server: source (app / run config), state, tools, masked secrets
//! - POST   /api/mcp/servers               add one ({name, url, headers} or {name, command, args, env, cwd}), or a
//!                                         pasted Claude Desktop / Cursor / VS Code block ({"mcpServers": {...}})
//! - PUT    /api/mcp/servers/{name}        change one of the app's (masked values keep their stored values)
//! - DELETE /api/mcp/servers/{name}        remove one of the app's
//! - POST   /api/mcp/servers/test          start one and list its tools, nothing kept
//! - POST   /api/mcp/servers/{name}/tools  {tool: true/false}
//!
//! Every route needs the API key and the server's own page. A command server runs a program on this PC with the
//! user's rights: it is added, changed or tested only from this PC (a loopback address), unless the run config sets
//! `"mcp_remote_commands": true`. Each change is logged with the address it came from (and is in API calls).

use crate::mcp::{self, McpHub};
use crate::{authorized, error, AppState};
use axum::extract::{ConnectInfo, Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Map, Value};
use std::net::SocketAddr;
use std::sync::Arc;

fn guard(s: &AppState, headers: &HeaderMap) -> Result<Arc<McpHub>, Response> {
    if !authorized(s, headers) {
        return Err(error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required"));
    }
    if crate::foreign_origin(s, headers) {
        return Err(crate::not_own_page("MCP servers can be changed"));
    }
    s.mcp.clone().ok_or_else(|| error(StatusCode::SERVICE_UNAVAILABLE, "unavailable", "MCP is not set up"))
}

fn is_command(cfg: &Value) -> bool {
    !cfg.get("url").is_some_and(|u| !u.is_null())
}

/// May a request from `addr` add, change or test a command server?
pub fn commands_allowed(addr: &SocketAddr, remote_commands: bool) -> bool {
    addr.ip().to_canonical().is_loopback() || remote_commands
}

fn no_remote_commands() -> Response {
    error(StatusCode::FORBIDDEN, "permission_error",
          "a command server runs a program on this PC, so it can be added or changed only from this PC itself (or with \
           \"mcp_remote_commands\": true in the run config); add the server by its address (a URL) instead")
}

/// A body's server configuration: the body without its "name" (and the app's own fields).
fn server_cfg(body: &Value) -> Value {
    let mut c = body.as_object().cloned().unwrap_or_default();
    for k in ["name", "replace"] {
        c.remove(k);
    }
    Value::Object(c)
}

fn bad(msg: impl Into<String>) -> Response {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
}

pub async fn list(State(s): State<AppState>, headers: HeaderMap) -> Response {
    match guard(&s, &headers) {
        Ok(hub) => Json(hub.status()).into_response(),
        Err(r) => r,
    }
}

pub async fn add(State(s): State<AppState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    let hub = match guard(&s, &headers) {
        Ok(h) => h,
        Err(r) => return r,
    };
    // one server, or a pasted block of several
    let block = ["mcpServers", "mcp_servers", "servers"].iter().find_map(|k| body.get(*k).and_then(Value::as_object)).cloned();
    let replace = body["replace"] == json!(true);
    let entries: Vec<(String, Value)> = match block {
        Some(b) => b.into_iter().map(|(n, c)| (n, c)).collect(),
        None => match body["name"].as_str().map(str::trim) {
            Some(n) if !n.is_empty() => vec![(n.to_string(), server_cfg(&body))],
            _ => return bad("name: give the server a name"),
        },
    };
    if entries.is_empty() {
        return bad("no servers in the block");
    }
    // every entry is checked before any is kept
    for (name, cfg) in &entries {
        if !cfg.is_object() {
            return bad(format!("{name}: a server is an object with \"url\" or \"command\""));
        }
        if is_command(cfg) && !commands_allowed(&addr, hub.remote_commands) {
            return no_remote_commands();
        }
        match hub.source_of(name) {
            Some("config") => return error(StatusCode::CONFLICT, "conflict", format!("{name:?} is a server of the run config")),
            Some(_) if !replace => return error(StatusCode::CONFLICT, "conflict", format!("there is already a server {name:?} (change it instead)")),
            _ => {}
        }
    }
    let mut added = Vec::new();
    for (name, cfg) in entries {
        if let Err(e) = hub.put_app(&name, cfg.clone()).await {
            return bad(e);
        }
        tracing::info!("MCP servers: {} added {name:?} ({})", addr.ip(), mcp::command_line(&cfg).unwrap_or_else(|| cfg["url"].as_str().unwrap_or("").to_string()));
        added.push(name);
    }
    Json(json!({"added": added, "mcp": hub.status()})).into_response()
}

pub async fn update(State(s): State<AppState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Path(name): Path<String>,
                    Json(body): Json<Value>) -> Response {
    let hub = match guard(&s, &headers) {
        Ok(h) => h,
        Err(r) => return r,
    };
    let old = match hub.source_of(&name) {
        None => return error(StatusCode::NOT_FOUND, "not_found", format!("there is no MCP server {name:?}")),
        Some("config") => return error(StatusCode::CONFLICT, "conflict", format!("{name:?} is a server of the run config: change it there")),
        Some(_) => hub.app_config(&name),
    };
    let cfg = mcp::unmask(server_cfg(&body), old.as_ref());
    if (is_command(&cfg) || old.as_ref().is_some_and(is_command)) && !commands_allowed(&addr, hub.remote_commands) {
        return no_remote_commands();
    }
    if let Err(e) = hub.put_app(&name, cfg.clone()).await {
        return bad(e);
    }
    tracing::info!("MCP servers: {} changed {name:?}{}", addr.ip(), if cfg["disabled"] == json!(true) { " (turned off)" } else { "" });
    Json(json!({"changed": name, "mcp": hub.status()})).into_response()
}

pub async fn remove(State(s): State<AppState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Path(name): Path<String>) -> Response {
    let hub = match guard(&s, &headers) {
        Ok(h) => h,
        Err(r) => return r,
    };
    match hub.source_of(&name) {
        None => return error(StatusCode::NOT_FOUND, "not_found", format!("there is no MCP server {name:?}")),
        Some("config") => return error(StatusCode::CONFLICT, "conflict", format!("{name:?} is a server of the run config: remove it there")),
        _ => {}
    }
    if let Err(e) = hub.remove_app(&name).await {
        return bad(e);
    }
    tracing::info!("MCP servers: {} removed {name:?}", addr.ip());
    Json(json!({"removed": name, "mcp": hub.status()})).into_response()
}

pub async fn test(State(s): State<AppState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    let hub = match guard(&s, &headers) {
        Ok(h) => h,
        Err(r) => return r,
    };
    // testing an edit of a stored server: its masked values are the stored ones
    let old = body["name"].as_str().and_then(|n| hub.app_config(n));
    let cfg = mcp::unmask(server_cfg(&body), old.as_ref());
    if is_command(&cfg) && !commands_allowed(&addr, hub.remote_commands) {
        return no_remote_commands();
    }
    Json(hub.test(cfg).await).into_response()
}

pub async fn tools(State(s): State<AppState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Path(name): Path<String>,
                   Json(body): Json<Value>) -> Response {
    let hub = match guard(&s, &headers) {
        Ok(h) => h,
        Err(r) => return r,
    };
    let Some(sw) = body.as_object().filter(|o| !o.is_empty()) else {
        return bad("expected {\"tool\": true or false, ...}");
    };
    let sw: Map<String, Value> = sw.clone();
    match hub.set_tools(&name, &sw) {
        Ok(()) => {
            let what = serde_json::to_string(&sw).unwrap_or_default();
            tracing::info!("MCP servers: {} switched tools of {name:?}: {what}", addr.ip());
            Json(json!({"mcp": hub.status()})).into_response()
        }
        Err(e) if e.starts_with("there is no") => error(StatusCode::NOT_FOUND, "not_found", e),
        Err(e) => bad(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_servers_only_from_this_pc_unless_allowed() {
        let local: SocketAddr = "127.0.0.1:5000".parse().unwrap();
        let mapped: SocketAddr = "[::ffff:127.0.0.1]:5000".parse().unwrap();
        let lan: SocketAddr = "192.168.1.20:5000".parse().unwrap();
        assert!(commands_allowed(&local, false));
        assert!(commands_allowed(&mapped, false));
        assert!(!commands_allowed(&lan, false));
        assert!(commands_allowed(&lan, true));
        assert!(is_command(&json!({"command": "x"})) && !is_command(&json!({"url": "http://h/mcp"})));
        assert_eq!(server_cfg(&json!({"name": "a", "url": "u", "replace": true})), json!({"url": "u"}));
    }
}
