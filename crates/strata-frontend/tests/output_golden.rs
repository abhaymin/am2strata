//! The Rust OutputParser against events recorded from PolyStrata's Python OutputParser (tools/make_golden_output.py).
//! Text events are compared merged by kind (what a client sees); tool events exactly (name, arguments, and the
//! concatenated streamed argument pieces).

use serde_json::{json, Value};
use strata_frontend::output::{Event, OutputParser};

/// Merge adjacent text events of the same kind; tool events stay as they are (ids left out).
fn normalize(events: &[Value]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    for e in events {
        let kind = e["kind"].as_str().unwrap();
        match kind {
            "reasoning" | "content" => {
                let text = e["text"].as_str().unwrap();
                if text.is_empty() {
                    continue;
                }
                if let Some(last) = out.last_mut() {
                    if last["kind"] == kind {
                        let joined = format!("{}{}", last["text"].as_str().unwrap(), text);
                        last["text"] = json!(joined);
                        continue;
                    }
                }
                out.push(json!({"kind": kind, "text": text}));
            }
            "tool_args" => {
                if let Some(last) = out.last_mut() {
                    if last["kind"] == "tool_args" {
                        let joined = format!("{}{}", last["text"].as_str().unwrap(), e["text"].as_str().unwrap());
                        last["text"] = json!(joined);
                        continue;
                    }
                }
                out.push(json!({"kind": "tool_args", "text": e["text"]}));
            }
            "tool_start" => out.push(json!({"kind": "tool_start", "name": e["name"]})),
            _ => out.push(json!({"kind": kind, "name": e["name"], "arguments": e["arguments"]})),
        }
    }
    out
}

fn rust_events(text: &str, thinking: bool, tools: Option<&[Value]>, chunks: &Value, stream: bool) -> Value {
    let mut p = OutputParser::new(thinking, tools, stream);
    let pieces: Vec<String> = match chunks.as_u64() {
        None => vec![text.to_string()],
        Some(n) => {
            let cs: Vec<char> = text.chars().collect();
            cs.chunks(n as usize).map(|c| c.iter().collect()).collect()
        }
    };
    let mut events = Vec::new();
    for piece in &pieces {
        match p.feed(piece) {
            Ok(evs) => events.extend(evs),
            Err(e) => return json!({"error": e.0}),
        }
    }
    events.extend(p.finish(None));
    Value::Array(
        events
            .into_iter()
            .map(|e| match e {
                Event::Reasoning(t) => json!({"kind": "reasoning", "text": t}),
                Event::Content(t) => json!({"kind": "content", "text": t}),
                Event::ToolStart { name, .. } => json!({"kind": "tool_start", "text": "", "name": name}),
                Event::ToolArgs { text, .. } => json!({"kind": "tool_args", "text": text}),
                Event::ToolCall(c) => json!({"kind": "tool_call", "text": "", "name": c.name, "arguments": c.arguments}),
            })
            .collect(),
    )
}

#[test]
fn output_parser_matches_python() {
    let path = format!("{}/../../golden/output.json", env!("CARGO_MANIFEST_DIR"));
    let g: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut failures = Vec::new();
    for c in g["cases"].as_array().unwrap() {
        let tools: Option<Vec<Value>> = c["tools"].as_array().cloned();
        let got = rust_events(c["text"].as_str().unwrap(), c["thinking"].as_bool().unwrap(), tools.as_deref(),
                              &c["chunks"], c["stream_tools"].as_bool().unwrap());
        let want = &c["events"];
        let ok = match (want.get("error"), got.get("error")) {
            (Some(_), Some(_)) => true,
            (None, None) => normalize(want.as_array().unwrap()) == normalize(got.as_array().unwrap()),
            _ => false,
        };
        if !ok {
            failures.push(format!("{} chunks={} stream={}\n   got:  {}\n   want: {}", c["name"], c["chunks"],
                                  c["stream_tools"], serde_json::to_string(&got).unwrap(), serde_json::to_string(want).unwrap()));
        }
    }
    assert!(failures.is_empty(), "{} of {} cases differ:\n{}", failures.len(), g["cases"].as_array().unwrap().len(),
            failures.join("\n"));
}
