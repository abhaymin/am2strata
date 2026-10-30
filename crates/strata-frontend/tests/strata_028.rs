//! Strata 0.2.8's output parser changes: serve/test_quoted_call_values.py, test_tool_routes.py and the recoveries.

use serde_json::{json, Value};
use strata_frontend::output::{lenient_json, Event, OutputParser};

fn tools() -> Vec<Value> {
    vec![json!({"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "text": {"type": "string"}}}}),
         json!({"name": "todo", "parameters": {"properties": {"items": {"type": "array"}, "meta": {"type": "object"}}}})]
}

fn run(text: &str, thinking: bool, recover: bool) -> (Vec<Event>, OutputParser) {
    let t = tools();
    let mut p = OutputParser::new(thinking, Some(&t), false);
    p.set_recover(recover);
    let mut ev = p.feed(text).unwrap();
    ev.extend(p.finish(None));
    (ev, p)
}

fn calls(ev: &[Event]) -> Vec<(String, Value)> {
    ev.iter().filter_map(|e| if let Event::ToolCall(c) = e { Some((c.name.clone(), Value::Object(c.arguments.clone()))) } else { None }).collect()
}

fn content(ev: &[Event]) -> String {
    ev.iter().filter_map(|e| if let Event::Content(t) = e { Some(t.as_str()) } else { None }).collect()
}

#[test]
fn a_value_that_quotes_a_call_stays_in_the_value() {
    let quoted = "see <function=write>\n<parameter=path>\nx\n</parameter>\n</function> done";
    let out = format!("<tool_call>\n<function=write>\n<parameter=path>\na.md\n</parameter>\n<parameter=text>\n{quoted}\n</parameter>\n</function>\n</tool_call>");
    let (ev, _) = run(&out, false, false);
    assert_eq!(calls(&ev), [("write".to_string(), json!({"path": "a.md", "text": quoted}))]);
}

#[test]
fn a_left_out_function_close_ends_at_the_wrapper() {
    let out = "<tool_call>\n<function=write>\n<parameter=path>\na.md\n</parameter>\n</tool_call>";
    let (ev, _) = run(out, false, false);
    assert_eq!(calls(&ev), [("write".to_string(), json!({"path": "a.md"}))]);
}

#[test]
fn damaged_json_arguments_are_read_and_noted() {
    let out = "<tool_call>\n<function=todo>\n<parameter=items>\n[{'id': 1, \"done\": False,},]\n</parameter>\n</function>\n</tool_call>";
    let (ev, p) = run(out, false, false);
    assert_eq!(calls(&ev), [("todo".to_string(), json!({"items": [{"id": 1, "done": false}]}))]);
    assert_eq!(p.notes.len(), 1);
    // a bare string between two members, and brackets left open at the end
    assert_eq!(lenient_json(r#"{"status": "pending", "in_progress", "n": 2"#).unwrap(), json!({"status": "pending", "n": 2}));
    // it never guesses at a value
    assert!(lenient_json(r#"["cut short"#).is_err());
    assert!(lenient_json("[nonsense]").is_err());
}

#[test]
fn a_stray_think_close_is_dropped_and_a_call_after_it_counts() {
    let out = "thinking\n</think>\nmore thinking with `code\n</think>\n\nThe answer.";
    let (ev, p) = run(out, true, false);
    assert_eq!(content(&ev), "more thinking with `code\nThe answer.");
    assert_eq!(p.stray_think, 1);
    let out = "t\n</think>\nmore `open\n</think>\n<tool_call>\n<function=write>\n<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>";
    assert_eq!(calls(&run(out, true, false).0).len(), 1);
}

#[test]
fn recover_reads_the_forms_next_to_the_template() {
    // a bare <function= at the start of a line
    let (ev, _) = run("Writing it.\n<function=write>\n<parameter=path>\na\n</parameter>\n</function>", false, true);
    assert_eq!(calls(&ev), [("write".to_string(), json!({"path": "a"}))]);
    assert_eq!(content(&ev), "Writing it.");
    // a tool's name written as the first parameter
    let (ev, _) = run("<tool_call>\n<parameter=write>\n<parameter=path>\na\n</parameter>\n</function>\n</tool_call>", false, true);
    assert_eq!(calls(&ev), [("write".to_string(), json!({"path": "a"}))]);
    // two calls in one wrapper
    let two = "<tool_call>\n<function=write>\n<parameter=path>\na\n</parameter>\n</function>\n<function=write>\n<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>";
    assert_eq!(calls(&run(two, false, true).0).len(), 2);
    // in code, or an undeclared name, it is text
    let (ev, _) = run("```\n<function=write>\n```\n", false, true);
    assert!(calls(&ev).is_empty() && content(&ev).contains("<function=write>"));
    let (ev, _) = run("<function=nope>\nhello", false, true);
    assert!(calls(&ev).is_empty());
}

#[test]
fn streaming_a_call_piece_by_piece_gives_the_same_call() {
    let t = tools();
    let out = "Writing.\n<function=write>\n<parameter=path>\na\n</parameter>\n</function>\n";
    let mut p = OutputParser::new(false, Some(&t), true);
    p.set_recover(true);
    let mut ev = Vec::new();
    for c in out.chars() {
        ev.extend(p.feed(&c.to_string()).unwrap());
    }
    ev.extend(p.finish(None));
    assert_eq!(calls(&ev), [("write".to_string(), json!({"path": "a"}))]);
}
