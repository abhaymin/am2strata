//! Whole requests against PolyStrata's Python front end (tools/reference-polystrata/make_golden_requests.py): normalized messages,
//! tools and kwargs, the literal tags, the prompt text and the token ids sent to the engine.

use serde_json::Value;
use strata_frontend::{openai_to_messages, PromptBuilder, ToolOrders};
use strata_template::ChatTemplate;
use strata_tokenizer::Tokenizer;

fn check(model: &str) {
    let path = format!("{}/../../golden/{model}-requests.json", env!("CARGO_MANIFEST_DIR"));
    let g: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let dir = g["tokenizer_dir"].as_str().unwrap();
    let tok = std::sync::Arc::new(Tokenizer::from_dir(dir).unwrap());
    let tmpl = std::sync::Arc::new(ChatTemplate::from_file(format!("{dir}/chat_template.jinja")).unwrap());
    let builder = PromptBuilder::new(tmpl, tok);

    // literal tags: same tags, same marks, same order
    let want: Vec<(String, String)> =
        g["literals"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string())).collect();
    let got: Vec<(String, String)> = builder.literals().tags.iter().map(|(t, c)| (t.clone(), c.to_string())).collect();
    assert_eq!(got, want, "{model}: literal tags");

    let mut failures = Vec::new();
    for c in g["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let orders = ToolOrders::default();
        let converted = if c["api"] == "anthropic" {
            strata_frontend::request::anthropic_to_messages(&c["request"], &orders, true).map(|(p, b)| (p, b))
        } else {
            openai_to_messages(&c["request"], &orders).map(|p| (p, None))
        };
        let p = match converted {
            Ok((p, budget)) => {
                if c["api"] == "anthropic" && budget != c["budget"].as_u64() {
                    failures.push(format!("{name}: budget {budget:?} want {}", c["budget"]));
                }
                p
            }
            Err(e) => {
                failures.push(format!("{name}: refused: {e}"));
                continue;
            }
        };
        if Value::Array(p.messages.clone()) != c["messages"] {
            failures.push(format!("{name}: messages differ\n   got:  {}\n   want: {}",
                                  Value::Array(p.messages.clone()), c["messages"]));
            continue;
        }
        let tools = p.tools.clone().map(Value::Array).unwrap_or(Value::Null);
        if tools != c["tools"] {
            failures.push(format!("{name}: tools differ"));
            continue;
        }
        if Value::Object(p.kwargs.clone()) != c["kwargs"] {
            failures.push(format!("{name}: kwargs differ: got {:?} want {}", p.kwargs, c["kwargs"]));
            continue;
        }
        let prompt = match builder.build(&p) {
            Ok(x) => x,
            Err(e) => {
                failures.push(format!("{name}: build failed: {e}"));
                continue;
            }
        };
        if prompt.text != c["prompt"].as_str().unwrap() {
            failures.push(format!("{name}: prompt text differs"));
            continue;
        }
        let want_ids: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        if prompt.ids != want_ids {
            let at = prompt.ids.iter().zip(&want_ids).position(|(a, b)| a != b).unwrap_or(0);
            failures.push(format!("{name}: ids differ at {at} (got {} want {})", prompt.ids.len(), want_ids.len()));
        }
    }
    assert!(failures.is_empty(), "{model}: {} mismatches:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn tiel_requests_match_python() {
    check("tiel");
}

#[test]
fn orca_requests_match_python() {
    check("orca");
}

#[test]
fn tool_order_is_stable() {
    let orders = ToolOrders::default();
    let w = serde_json::json!({"type": "function", "function": {"name": "a", "parameters": {}}});
    let t = serde_json::json!({"type": "function", "function": {"name": "b", "parameters": {}}});
    let first = openai_to_messages(&serde_json::json!({"messages": [{"role": "user", "content": "x"}], "tools": [w, t]}), &orders).unwrap();
    let again = openai_to_messages(&serde_json::json!({"messages": [{"role": "user", "content": "x"}], "tools": [t, w]}), &orders).unwrap();
    assert_eq!(first.tools, again.tools);
}
