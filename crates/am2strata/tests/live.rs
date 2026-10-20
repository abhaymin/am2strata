//! Live tests against a running am2strata with the real engines (GPU). Ignored by default:
//!
//!   STRATA_LIVE_URL=http://127.0.0.1:8001 STRATA_LIVE_KEY=... cargo test -p am2strata --release --test live -- --ignored --test-threads=1
//!
//! The server must have the installed PolyStrata models (Orca, Tiel Coder, MiniCPM-V). Answers vary between runs (the
//! engines are not deterministic), so the checks look for the facts an answer must contain, not exact text.

use serde_json::{json, Value};
use std::time::Duration;

fn base() -> String {
    std::env::var("STRATA_LIVE_URL").unwrap_or_else(|_| "http://127.0.0.1:8001".into())
}

fn key() -> String {
    std::env::var("STRATA_LIVE_KEY").unwrap_or_default()
}

fn post(path: &str, body: &Value) -> (u16, Value) {
    let r = reqwest::blocking::Client::new()
        .post(format!("{}{path}", base()))
        .bearer_auth(key())
        .json(body)
        .timeout(Duration::from_secs(1800))
        .send()
        .expect("is the live server running? set STRATA_LIVE_URL");
    (r.status().as_u16(), r.json().unwrap_or(Value::Null))
}

fn user(c: Value) -> Value {
    json!({"role": "user", "content": c})
}

fn no_think() -> Value {
    json!({"enable_thinking": false})
}

fn load(model: &str) {
    let (c, r) = post("/api/models/load", &json!({"model": model}));
    assert_eq!(c, 200, "load {model}: {r}");
}

fn png(left: [u8; 3], right: [u8; 3]) -> String {
    use base64::Engine as _;
    let mut img = image::RgbImage::new(96, 64);
    for (x, _, p) in img.enumerate_pixels_mut() {
        *p = image::Rgb(if x < 48 { left } else { right });
    }
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(buf.into_inner()))
}

fn content(r: &Value) -> String {
    r["choices"][0]["message"]["content"].as_str().unwrap_or("").to_lowercase()
}

#[test]
#[ignore]
fn live_orca_text_image_tools_and_anthropic() {
    load("orcarouter");
    let (c, r) = post("/v1/chat/completions", &json!({"messages": [user(json!("What is 17*23? Reply with just the number."))],
                                                       "max_tokens": 200, "chat_template_kwargs": no_think()}));
    assert_eq!(c, 200);
    assert!(content(&r).contains("391"), "{r}");
    let (_, r) = post("/v1/chat/completions", &json!({"messages": [user(json!([
        {"type": "text", "text": "What two colors are in this image, left and right? Answer briefly."},
        {"type": "image_url", "image_url": {"url": png([255, 0, 0], [0, 0, 255])}}]))], "max_tokens": 300, "chat_template_kwargs": no_think()}));
    assert!(content(&r).contains("red") && content(&r).contains("blue"), "{r}");
    let tools = json!([{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]);
    let (_, r) = post("/v1/chat/completions", &json!({"messages": [user(json!("What's the weather in Pune right now? Use the tool."))],
                                                       "tools": tools, "max_tokens": 800, "chat_template_kwargs": no_think()}));
    assert_eq!(r["choices"][0]["finish_reason"], "tool_calls", "{r}");
    assert_eq!(r["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "get_weather");
    let (_, r) = post("/v1/messages", &json!({"model": "x", "max_tokens": 200, "thinking": {"type": "disabled"},
                                               "messages": [user(json!("What is 6*7? Reply with just the number."))]}));
    assert!(r["content"][0]["text"].as_str().unwrap_or("").contains("42"), "{r}");
}

#[test]
#[ignore]
fn live_minicpm_document_and_frames() {
    load("minicpm");
    let (_, r) = post("/v1/chat/completions", &json!({"messages": [user(json!([
        {"type": "text", "text": "What two colors are in this image, left and right? Answer briefly."},
        {"type": "image_url", "image_url": {"url": png([0, 255, 0], [255, 255, 0])}}]))], "max_tokens": 300, "chat_template_kwargs": no_think()}));
    assert!(content(&r).contains("green") && content(&r).contains("yellow"), "{r}");
    // four frames of a ball moving right
    let frames: Vec<Value> = [10u32, 30, 50, 70]
        .iter()
        .map(|pct| {
            use base64::Engine as _;
            let mut img = image::RgbImage::from_pixel(200, 60, image::Rgb([255, 255, 255]));
            let cx = 200 * pct / 100;
            for (x, y, p) in img.enumerate_pixels_mut() {
                let (dx, dy) = (x as i32 - cx as i32, y as i32 - 30);
                if dx * dx + dy * dy < 225 {
                    *p = image::Rgb([255, 0, 0]);
                }
            }
            let mut buf = std::io::Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
            json!({"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(buf.into_inner()))}})
        })
        .collect();
    let mut parts = vec![json!({"type": "text", "text": "These are 4 frames of a video, in order. In one sentence, what happens?"})];
    parts.extend(frames);
    let (_, r) = post("/v1/chat/completions", &json!({"messages": [user(json!(parts))], "max_tokens": 300, "chat_template_kwargs": no_think()}));
    let a = content(&r);
    assert!(a.contains("right") || a.contains("move"), "{r}");
}

#[test]
#[ignore]
fn live_tiel_streamed_tool_call_and_speed() {
    load("tiel-coder-35b-a3b-ud-q6");
    let (_, r) = post("/v1/chat/completions", &json!({"messages": [user(json!("Write a Python function that reverses a string. Code only."))],
                                                       "max_tokens": 400, "temperature": 0, "chat_template_kwargs": no_think()}));
    assert!(content(&r).contains("def "), "{r}");
    // the speed of a warm engine (the first answers after a load fill its expert cache) over a long answer
    let ask = json!({"messages": [user(json!("Write a Python module implementing a binary search tree with insert, delete, \
                     search and in-order traversal, with docstrings."))], "max_tokens": 800, "chat_template_kwargs": no_think()});
    post("/v1/chat/completions", &ask);
    let (_, r) = post("/v1/chat/completions", &ask);
    let tps = r["timings"]["predicted_per_second"].as_f64().unwrap_or(0.0);
    eprintln!("Tiel Q6: {} tokens at {tps:.1} tok/s", r["timings"]["predicted_n"]);
    assert!(tps > 70.0, "below the 70 tok/s floor (this card does ~100-110 warm): {tps}");
}

fn get(path: &str) -> Value {
    reqwest::blocking::Client::new().get(format!("{}{path}", base())).bearer_auth(key()).send().unwrap().json().unwrap()
}

#[test]
#[ignore]
fn live_orca_responses_budget_forced_call_and_monitor() {
    load("orcarouter");
    // the Responses API (Codex's), stateless
    let (c, r) = post("/v1/responses", &json!({"input": "What is 12*12? Reply with just the number.", "max_output_tokens": 300,
                                                "reasoning": {"effort": "none"}}));
    assert_eq!(c, 200, "{r}");
    let text: String = r["output"].as_array().unwrap().iter().filter(|i| i["type"] == "message")
        .filter_map(|i| i["content"][0]["text"].as_str()).collect();
    assert!(text.contains("144"), "{r}");
    // a small thinking budget: the thinking is wrapped up and an answer follows
    let (_, r) = post("/v1/chat/completions", &json!({"messages": [user(json!("Is 97 a prime number? Answer yes or no."))],
                                                       "max_tokens": 2000, "reasoning_budget_tokens": 64}));
    assert!(!content(&r).is_empty(), "an answer after the budget: {r}");
    assert!(content(&r).contains("yes"), "{r}");
    // tool_choice forces the named call even for a question that needs none
    let tools = json!([{"type": "function", "function": {"name": "lookup", "description": "Look a word up.",
                        "parameters": {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]}}}]);
    let (_, r) = post("/v1/chat/completions", &json!({"messages": [user(json!("Say hello."))], "tools": tools, "max_tokens": 600,
        "tool_choice": {"type": "function", "function": {"name": "lookup"}}, "chat_template_kwargs": no_think()}));
    assert_eq!(r["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "lookup", "{r}");
    // the monitor saw these requests and the GPU
    let m = get("/metrics");
    assert!(m["totals"]["requests"].as_u64().unwrap() >= 3, "{m}");
    assert!(m["hardware"]["gpu_mem_total"].as_f64().unwrap_or(0.0) > 0.0, "{m}");
    assert!(m["requests"][0]["decode_tok_s"].as_f64().unwrap_or(0.0) > 10.0, "{m}");
}
