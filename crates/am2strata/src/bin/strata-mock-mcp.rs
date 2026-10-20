//! A tiny MCP server for the tests (stdio, newline-delimited JSON-RPC): one tool, `add`, that sums `a` and `b`.

use serde_json::{json, Value};
use std::io::{BufRead, Write};

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let Ok(m) = serde_json::from_str::<Value>(&line) else { continue };
        let Some(id) = m.get("id").cloned() else { continue }; // notifications need no answer
        let result = match m["method"].as_str().unwrap_or("") {
            "initialize" => json!({"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                                   "serverInfo": {"name": "mock-calc", "version": "1"}}),
            "tools/list" => json!({"tools": [{"name": "add", "description": "Add two numbers.",
                "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}}]}),
            "tools/call" => {
                let a = &m["params"]["arguments"];
                let sum = a["a"].as_f64().unwrap_or(0.0) + a["b"].as_f64().unwrap_or(0.0);
                json!({"content": [{"type": "text", "text": format!("{sum}")}]})
            }
            other => {
                let e = json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": format!("no {other}")}});
                let _ = writeln!(out, "{e}");
                let _ = out.flush();
                continue;
            }
        };
        let _ = writeln!(out, "{}", json!({"jsonrpc": "2.0", "id": id, "result": result}));
        let _ = out.flush();
    }
}
