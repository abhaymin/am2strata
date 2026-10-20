//! Rendered prompts against PolyStrata's Python ChatTemplate (tools/make_golden.py).

use serde_json::Value;
use strata_template::ChatTemplate;

fn check(model: &str) {
    let path = format!("{}/../../golden/{model}.json", env!("CARGO_MANIFEST_DIR"));
    let g: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let dir = g["tokenizer_dir"].as_str().unwrap();
    let t = ChatTemplate::from_file(format!("{dir}/chat_template.jinja")).expect("load template");
    let mut failures = Vec::new();
    for c in g["cases"].as_array().unwrap() {
        let messages = c["messages"].as_array().unwrap();
        let tools = if c["tools"].is_null() { None } else { Some(&c["tools"]) };
        let extra = c["kwargs"].as_object().cloned().unwrap_or_default();
        let want = c["text"].as_str().unwrap();
        match t.render(messages, tools, c["add_generation_prompt"].as_bool().unwrap(), &extra) {
            Ok(got) if got == want => {}
            Ok(got) => {
                let at: usize = got.chars().zip(want.chars()).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
                let ctx = |s: &str| s.chars().skip(at.saturating_sub(40)).take(90).collect::<String>();
                failures.push(format!("{}: differs at char {at}\n   got:  {:?}\n   want: {:?}", c["name"], ctx(&got), ctx(want)));
            }
            Err(e) => failures.push(format!("{}: error {e}", c["name"])),
        }
    }
    assert!(failures.is_empty(), "{model}: {} mismatches:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn tiel_matches_python() {
    check("tiel");
}

#[test]
fn orca_matches_python() {
    check("orca");
}
