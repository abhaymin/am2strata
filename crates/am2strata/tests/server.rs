//! The server end to end against the mock engine: both APIs, streaming, reasoning, tool calls, stop strings,
//! pictures, model switching, client disconnects, refusals and the web app. No GPU needed.

mod common;

use common::*;
use serde_json::{json, Value};

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

#[test]
fn health_models_and_auth() {
    let s = Server::start(&[Model::default(), Model { id: "mock-b", ..Model::default() }], &[ids("hi")], json!({}));
    let (c, h) = s.get("/health", false);
    assert_eq!(c, 200);
    assert_eq!(h["model"], "Mock mock-a");
    assert_eq!(h["loaded"], true);
    assert_eq!(h["images"], true);
    assert_eq!(s.get("/v1/models", false).0, 401, "the API needs the key");
    let (c, m) = s.get("/v1/models", true);
    assert_eq!(c, 200);
    let names: Vec<&str> = m["data"].as_array().unwrap().iter().map(|x| x["id"].as_str().unwrap()).collect();
    assert_eq!(names, ["mock-a", "mock-b"]);
    let (_, l) = s.get("/api/models", true);
    assert_eq!(l["models"][0]["current"], true);
    assert_eq!(l["models"][1]["current"], false);
}

#[test]
fn openai_sends_the_verified_prompt_and_returns_reasoning_and_content() {
    let s = Server::start(&[Model::default()], &[ids("I add them.\n</think>\n\nThe answer is 4.")], json!({}));
    let req = json!({"messages": [user("What is 2+2?")], "max_tokens": 100});
    let (c, r) = s.post("/v1/chat/completions", &req);
    assert_eq!(c, 200, "{r}");
    let m = &r["choices"][0]["message"];
    assert_eq!(m["reasoning_content"], "I add them.\n");
    assert_eq!(m["content"], "The answer is 4.");
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
    assert_eq!(r["usage"]["completion_tokens"], ids("I add them.\n</think>\n\nThe answer is 4.").len());
    let gen = s.gen_lines();
    assert_eq!(gen.len(), 1);
    assert!(gen[0].starts_with("GEN 100 "), "{}", gen[0]);
    assert_eq!(line_ids(&gen[0]), expected_prompt(&req), "the token ids sent to the engine");
}

#[test]
fn model_card_sampling_applies_only_when_the_request_sets_none() {
    let m = Model { arch: Some("qwen35moe"), ..Model::default() };
    let s = Server::start(&[m], &[ids("ok")], json!({}));
    s.post("/v1/chat/completions", &json!({"messages": [user("x")], "max_tokens": 5}));
    s.post("/v1/chat/completions", &json!({"messages": [user("x")], "max_tokens": 5, "chat_template_kwargs": {"enable_thinking": false}}));
    s.post("/v1/chat/completions", &json!({"messages": [user("x")], "max_tokens": 5, "temperature": 0}));
    let g = s.gen_lines();
    assert!(g[0].contains(" temperature=1.0 top_p=0.95 top_k=20 penalty_present=1.5 penalty_last_n=64 "), "thinking: {}", g[0]);
    assert!(g[1].contains(" temperature=0.7 top_p=0.8 top_k=20 penalty_present=1.5 "), "instruct: {}", g[1]);
    assert!(!g[2].contains("temperature="), "temperature 0 is greedy: {}", g[2]);
}

#[test]
fn openai_stream_chunks_and_usage() {
    let s = Server::start(&[Model::default()], &[ids("Thinking.\n</think>\n\nred, green, blue")], json!({}));
    let ev = s.stream("/v1/chat/completions", &json!({"messages": [user("Three colors?")], "stream": true}));
    assert_eq!(ev.last().unwrap().1, "[DONE]");
    let deltas: Vec<&Value> = ev.iter().filter_map(|(_, v)| v.get("choices").map(|c| &c[0]["delta"])).collect();
    let content: String = deltas.iter().filter_map(|d| d["content"].as_str()).collect();
    let reasoning: String = deltas.iter().filter_map(|d| d["reasoning_content"].as_str()).collect();
    assert_eq!(content, "red, green, blue");
    assert_eq!(reasoning, "Thinking.\n");
    let last = &ev[ev.len() - 2].1;
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert!(last["usage"]["completion_tokens"].as_u64().unwrap() > 0);
    assert!(last.get("timings").is_some());
}

const CALL: &str = "\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nPune\n</parameter>\n</function>\n</tool_call>";

fn weather_openai() -> Value {
    json!({"type": "function", "function": {"name": "get_weather", "description": "Weather.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})
}

#[test]
fn openai_tool_calls_whole_and_streamed() {
    let reply = ids(&format!("Need the tool.\n</think>{CALL}"));
    let s = Server::start(&[Model::default()], &[reply.clone(), reply], json!({}));
    let req = json!({"messages": [user("Weather in Pune?")], "tools": [weather_openai()]});
    let (c, r) = s.post("/v1/chat/completions", &req);
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["choices"][0]["finish_reason"], "tool_calls");
    let call = &r["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "get_weather");
    assert_eq!(serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(), json!({"city": "Pune"}));

    let mut sreq = req.clone();
    sreq["stream"] = json!(true);
    let ev = s.stream("/v1/chat/completions", &sreq);
    let pieces = ev.iter().filter_map(|(_, v)| v.get("choices")).filter(|c| c[0]["delta"].get("tool_calls").is_some()).count();
    assert!(pieces > 2, "the call streamed in pieces ({pieces})");
    let args: String = ev
        .iter()
        .filter_map(|(_, v)| v.get("choices"))
        .flat_map(|c| c[0]["delta"]["tool_calls"].as_array().cloned().unwrap_or_default())
        .filter_map(|t| t["function"]["arguments"].as_str().map(str::to_owned))
        .collect();
    assert_eq!(serde_json::from_str::<Value>(&args).unwrap(), json!({"city": "Pune"}), "pieces join to the arguments");
    let fin: Vec<&Value> = ev.iter().filter_map(|(_, v)| v.get("choices")).map(|c| &c[0]["finish_reason"]).filter(|f| !f.is_null()).collect();
    assert_eq!(fin, [&json!("tool_calls")]);
}

#[test]
fn anthropic_text_tool_use_and_stream_events() {
    let s = Server::start(
        &[Model::default()],
        &[ids("Bonjour!"), ids(&format!("Checking.{CALL}")), ids("Salut!")],
        json!({}),
    );
    let (c, r) = s.post("/v1/messages", &json!({"model": "x", "max_tokens": 50, "thinking": {"type": "disabled"},
                                                 "messages": [user("Hello in French?")]}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["type"], "message");
    assert_eq!(r["content"][0], json!({"type": "text", "text": "Bonjour!"}));
    assert_eq!(r["stop_reason"], "end_turn");

    let tools = json!([{"name": "get_weather", "description": "Weather.",
                        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]);
    let (_, r) = s.post("/v1/messages", &json!({"model": "x", "max_tokens": 200, "thinking": {"type": "disabled"},
                                                 "messages": [user("Weather?")], "tools": tools}));
    assert_eq!(r["stop_reason"], "tool_use", "{r}");
    let tu: Vec<&Value> = r["content"].as_array().unwrap().iter().filter(|b| b["type"] == "tool_use").collect();
    assert_eq!(tu[0]["name"], "get_weather");
    assert_eq!(tu[0]["input"], json!({"city": "Pune"}));

    let ev = s.stream("/v1/messages", &json!({"model": "x", "max_tokens": 50, "stream": true, "thinking": {"type": "disabled"},
                                               "messages": [user("Hi")]}));
    let names: Vec<&str> = ev.iter().map(|(n, _)| n.as_str()).filter(|n| *n != "ping").collect();
    assert_eq!(names.first(), Some(&"message_start"));
    assert_eq!(names.last(), Some(&"message_stop"));
    assert!(names.contains(&"content_block_start") && names.contains(&"content_block_delta") && names.contains(&"message_delta"));
    let text: String = ev.iter().filter_map(|(_, v)| v["delta"]["text"].as_str()).collect();
    assert_eq!(text, "Salut!");
}

#[test]
fn count_tokens_matches_the_prompt() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    let (c, r) = s.post("/v1/messages/count_tokens", &json!({"model": "x", "messages": [user("How many tokens is this?")]}));
    assert_eq!(c, 200);
    assert!(r["input_tokens"].as_u64().unwrap() > 5);
}

#[test]
fn stop_strings_cut_the_answer_and_stop_the_engine() {
    let s = Server::start(&[Model::default()], &[ids("1 2 3 4 5 6 7 8 9")], json!({"delay_ms": 20}));
    let (_, r) = s.post("/v1/chat/completions", &json!({"messages": [user("count")], "stop": [" 5"],
                                                         "chat_template_kwargs": {"enable_thinking": false}}));
    assert_eq!(r["choices"][0]["message"]["content"], "1 2 3 4");
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert!(s.commands().iter().any(|c| c == "STOP"), "the engine was told to stop: {:?}", s.commands());
}

#[test]
fn pictures_go_through_the_encoder_and_geni() {
    let s = Server::start(&[Model::default()], &[ids("A red and a blue square.")], json!({}));
    let req = json!({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is this?"}, {"type": "image_url", "image_url": {"url": png()}}]}],
        "chat_template_kwargs": {"enable_thinking": false}});
    let (c, r) = s.post("/v1/chat/completions", &req);
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["choices"][0]["message"]["content"], "A red and a blue square.");
    let cmds = s.commands();
    assert!(cmds.iter().any(|c| c.starts_with("VISION ENC ")), "the picture was encoded: {cmds:?}");
    let geni = s.gen_lines().into_iter().find(|l| l.starts_with("GENI")).expect("a GENI line");
    let pad = common::tokenizer().token_id("<|image_pad|>").unwrap();
    assert!(geni.contains(&format!(" image_pad={pad} ")), "{geni}");
    assert!(geni.contains(" img_keys="), "{geni}");
    let n_pads = line_ids(&geni).iter().filter(|t| **t == pad).count();
    assert_eq!(n_pads, 4, "one placeholder per image token (the mock encoder reports 4)");
}

#[test]
fn switching_models_stops_one_engine_and_starts_the_other() {
    let s = Server::start(&[Model::default(), Model { id: "mock-b", vision: false, ..Model::default() }], &[ids("a")], json!({}));
    let (c, r) = s.post("/api/models/load", &json!({"model": "mock-b"}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["status"], "loaded");
    assert_eq!(s.get("/health", false).1["model"], "Mock mock-b");
    assert!(s.commands().iter().any(|c| c == "QUIT"), "the first engine was asked to quit");
    let (c, _) = s.post("/v1/chat/completions", &json!({"messages": [user("hi")]}));
    assert_eq!(c, 200);
    let (c, r) = s.post("/api/models/load", &json!({"model": "no-such-model"}));
    assert_eq!(c, 400);
    assert!(r["error"]["message"].as_str().unwrap().contains("no installed model"));
}

#[test]
fn a_client_that_leaves_stops_the_engine_and_the_next_request_runs() {
    let long: Vec<u32> = std::iter::repeat_n(ids("word ")[0], 400).collect();
    let s = Server::start(&[Model::default()], &[long, ids("next")], json!({"delay_ms": 10}));
    let resp = reqwest::blocking::Client::new()
        .post(format!("{}/v1/chat/completions", s.base))
        .bearer_auth(KEY)
        .json(&json!({"messages": [user("essay")], "stream": true, "chat_template_kwargs": {"enable_thinking": false}}))
        .send()
        .unwrap();
    let mut buf = [0u8; 512];
    use std::io::Read;
    let mut r = resp;
    let _ = r.read(&mut buf);
    drop(r); // hang up mid-answer
    std::thread::sleep(std::time::Duration::from_millis(500));
    assert!(s.commands().iter().any(|c| c == "STOP"), "STOP after the client left");
    let t0 = std::time::Instant::now();
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("again")], "chat_template_kwargs": {"enable_thinking": false}}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["choices"][0]["message"]["content"], "next");
    assert!(t0.elapsed().as_secs() < 3, "the next request did not wait for the abandoned one");
}

#[test]
fn refusals_are_400s() {
    let s = Server::start(&[Model { context: 64, ..Model::default() }], &[ids("x")], json!({}));
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("hi")], "max_tokens": 5000}));
    assert_eq!(c, 400);
    assert!(r["error"]["message"].as_str().unwrap().contains("exceeds the context"), "{r}");
    let (c, _) = s.post("/v1/chat/completions", &json!({"messages": "not json"}));
    assert_eq!(c, 400);
    let (c, _) = s.post("/v1/chat/completions", &json!({"messages": [user("x")], "reasoning_effort": "turbo"}));
    assert_eq!(c, 400);
}

#[test]
fn video_without_ffmpeg_says_how_to_fix_it() {
    if strata_serve_has_ffmpeg() {
        return; // the frames path is covered by the live tests
    }
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe"}, {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}]}]}));
    assert_eq!(c, 400);
    assert!(r["error"]["message"].as_str().unwrap().contains("ffmpeg"));
}

fn strata_serve_has_ffmpeg() -> bool {
    std::process::Command::new("ffmpeg").arg("-version").output().is_ok_and(|o| o.status.success())
}

#[test]
fn web_app_is_served() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    let page = reqwest::blocking::get(format!("{}/", s.base)).unwrap().text().unwrap();
    assert!(page.contains("PolyStrata test page"));
    let (c, m) = s.get("/mcp", true);
    assert_eq!((c, m["tools"].as_u64()), (200, Some(0)));
}

#[test]
fn a_thinking_budget_wraps_up_and_the_answer_follows() {
    let long: Vec<u32> = std::iter::repeat_n(ids(" hmm")[0], 60).collect();
    // the engine still writing when the budget is reached (no delay: it may have finished, with nothing to STOP)
    let s = Server::start(&[Model::default()], &[long, ids("Answer: 4.")], json!({"delay_ms": 5}));
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("2+2?")], "max_tokens": 200,
                                                         "reasoning_budget_tokens": 10}));
    assert_eq!(c, 200, "{r}");
    let m = &r["choices"][0]["message"];
    assert!(m["reasoning_content"].as_str().unwrap().contains("I have thought about this long enough"), "{m}");
    assert_eq!(m["content"], "Answer: 4.");
    let gens = s.gen_lines();
    assert_eq!(gens.len(), 2, "a second pass after the wrap-up: {gens:?}");
    let first = line_ids(&gens[0]);
    let second = line_ids(&gens[1]);
    assert!(second.starts_with(&first), "the second pass continues the first one's prefix");
    let wrap = ids("\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n");
    assert!(second.ends_with(&wrap), "the wrap-up ends the second pass's prompt");
    assert_eq!(second.len(), first.len() + 10 + wrap.len(), "ten thought tokens, then the wrap-up");
    assert!(s.commands().iter().any(|c| c == "STOP"), "the first pass was stopped at the budget");
}

#[test]
fn tool_choice_forces_the_call() {
    let rest = "<parameter=city>\nPune\n</parameter>\n</function>\n</tool_call>";
    let s = Server::start(&[Model::default()], &[ids(rest), ids("Let me check.\n</think>\n\n"), ids(rest)], json!({}));
    // without thinking: the call's opening ends the prompt
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("Weather?")], "tools": [weather_openai()],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, "chat_template_kwargs": {"enable_thinking": false}}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["choices"][0]["finish_reason"], "tool_calls");
    let call = &r["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "get_weather");
    assert_eq!(serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(), json!({"city": "Pune"}));
    let g = s.gen_lines();
    assert!(line_ids(&g[0]).ends_with(&ids("<tool_call>\n<function=get_weather>\n")), "the opening ends the prompt");
    // with thinking: the opening is written once the thinking ends (Anthropic's "any" with one tool names it)
    let (c, r) = s.post("/v1/messages", &json!({"model": "x", "max_tokens": 300, "messages": [user("Weather?")],
        "tool_choice": {"type": "any"},
        "tools": [{"name": "get_weather", "description": "Weather.", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["stop_reason"], "tool_use", "{r}");
    let tu: Vec<&Value> = r["content"].as_array().unwrap().iter().filter(|b| b["type"] == "tool_use").collect();
    assert_eq!(tu[0]["input"], json!({"city": "Pune"}));
    let g = s.gen_lines();
    assert_eq!(g.len(), 3, "two passes for the thinking request: {g:?}");
    assert!(line_ids(&g[2]).ends_with(&ids("<tool_call>\n<function=get_weather>\n")));
}

#[test]
fn responses_api_text_reasoning_and_the_same_prompt_as_chat() {
    let s = Server::start(&[Model::default()], &[ids("I add them.\n</think>\n\nThe answer is 4.")], json!({}));
    let (c, r) = s.post("/v1/responses", &json!({"instructions": "Be brief.", "input": "What is 2+2?",
                                                  "max_output_tokens": 100, "include": ["reasoning.encrypted_content"]}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["object"], "response");
    assert_eq!(r["status"], "completed");
    let out = r["output"].as_array().unwrap();
    assert_eq!(out[0]["type"], "reasoning");
    assert_eq!(out[0]["content"][0]["text"], "I add them.\n");
    assert!(out[0]["encrypted_content"].as_str().unwrap().starts_with("strata.r1:"));
    assert_eq!(out[1]["type"], "message");
    assert_eq!(out[1]["content"][0]["text"], "The answer is 4.");
    assert!(r["usage"]["output_tokens"].as_u64().unwrap() > 0);
    let chat = json!({"messages": [{"role": "system", "content": "Be brief."}, user("What is 2+2?")]});
    let gen = s.gen_lines();
    assert!(gen[0].starts_with("GEN 100 "), "{}", gen[0]);
    assert_eq!(line_ids(&gen[0]), expected_prompt(&chat), "the same prompt as the chat request");
}

#[test]
fn responses_api_streams_typed_events_and_function_calls() {
    let reply = ids(&format!("Need the tool.\n</think>{CALL}"));
    let s = Server::start(&[Model::default()], &[reply], json!({}));
    let ev = s.stream("/v1/responses", &json!({"input": [{"role": "user", "content": "Weather in Pune?"}], "stream": true,
        "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]}));
    let kinds: Vec<&str> = ev.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(&kinds[..2], ["response.created", "response.in_progress"]);
    assert_eq!(*kinds.last().unwrap(), "response.completed");
    for (k, v) in &ev {
        assert_eq!(v["type"], k.as_str(), "the event name is the data's type");
    }
    let seq: Vec<u64> = ev.iter().map(|(_, v)| v["sequence_number"].as_u64().unwrap()).collect();
    assert!(seq.windows(2).all(|w| w[1] == w[0] + 1), "sequence numbers count up: {seq:?}");
    let args: String = ev.iter().filter(|(k, _)| k == "response.function_call_arguments.delta")
        .map(|(_, v)| v["delta"].as_str().unwrap().to_string()).collect();
    assert_eq!(serde_json::from_str::<Value>(&args).unwrap(), json!({"city": "Pune"}));
    let done = &ev.last().unwrap().1["response"]["output"];
    let call = done.as_array().unwrap().iter().find(|i| i["type"] == "function_call").unwrap();
    assert_eq!(call["name"], "get_weather");
    assert_eq!(call["status"], "completed");
    assert!(call["call_id"].as_str().unwrap().starts_with("call_"));
}

#[test]
fn responses_api_is_stateless() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    let (c, r) = s.post("/v1/responses", &json!({"input": "hi", "previous_response_id": "resp_1"}));
    assert_eq!(c, 400);
    assert_eq!(r["error"]["code"], "unsupported_parameter");
    assert_eq!(s.get("/v1/responses/resp_1", true).0, 404);
}

#[test]
fn monitor_records_requests_totals_and_the_machine() {
    let s = Server::start(&[Model::default()], &[ids("Thinking.\n</think>\n\nDone.")], json!({}));
    assert_eq!(s.get("/metrics", false).0, 401, "the monitor needs the key");
    let (_, m) = s.get("/metrics", true);
    assert_eq!(m["live"]["state"], "idle");
    assert_eq!(m["requests"], json!([]));
    s.post("/v1/chat/completions", &json!({"messages": [user("x")], "max_tokens": 50}));
    let (_, m) = s.get("/metrics", true);
    let r = &m["requests"][0];
    assert_eq!(r["finish"], "stop");
    assert_eq!(r["output_tokens"], ids("Thinking.\n</think>\n\nDone.").len());
    assert!(r["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(m["totals"]["requests"], 1);
    assert_eq!(m["engine"]["max_context"], 8192);
    assert!(m["hardware_static"]["threads"].as_u64().unwrap() > 0);
    let text = reqwest::blocking::Client::new().get(format!("{}/metrics/prometheus", s.base)).bearer_auth(KEY).send().unwrap().text().unwrap();
    assert!(text.contains("strata:requests_total 1"), "{text}");
    let (c, st) = s.get("/v1/status", true);
    assert_eq!(c, 200);
    assert_eq!(st["activity"]["requests"], 1);
    assert_eq!(st["dialects"][2], "/v1/responses");
}

#[test]
fn shared_settings_fill_what_requests_leave_out_and_persist() {
    let s = Server::start(&[Model::default()], &[ids("ok")], json!({}));
    let (c, r) = s.post("/settings", &json!({"defaults": {"temperature": 0.3, "max_tokens": 7, "reasoning_effort": "none"}}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["shared"], true);
    assert!(s.dir.join("strata-mock-a.shared-settings.json").exists(), "kept next to the run config");
    assert_eq!(s.post("/settings", &json!({"defaults": {"top_k": 99}})).0, 400);
    s.post("/v1/chat/completions", &json!({"messages": [user("x")]}));
    s.post("/v1/chat/completions", &json!({"messages": [user("x")], "max_tokens": 3, "temperature": 0.9}));
    let g = s.gen_lines();
    assert!(g[0].starts_with("GEN 7 ") && g[0].contains(" temperature=0.3"), "{}", g[0]);
    assert!(g[1].starts_with("GEN 3 ") && g[1].contains(" temperature=0.9"), "{}", g[1]);
    let chat = json!({"messages": [user("x")], "reasoning_effort": "none"});
    assert_eq!(line_ids(&g[0]), expected_prompt(&chat), "the shared thinking level is in the prompt");
    let (_, r) = s.post("/settings", &json!({"defaults": null}));
    assert_eq!(r["shared"], false);
    assert!(!s.dir.join("strata-mock-a.shared-settings.json").exists());
}

#[test]
fn sampling_merges_the_model_card_key_by_key() {
    let m = Model { arch: Some("qwen35moe"), ..Model::default() };
    let s = Server::start(&[m], &[ids("ok")], json!({}));
    s.post("/v1/chat/completions", &json!({"messages": [user("x")], "max_tokens": 5, "temperature": 0.5}));
    let g = s.gen_lines();
    assert!(g[0].contains(" temperature=0.5 top_p=0.95 top_k=20 penalty_present=1.5"), "the card fills the rest: {}", g[0]);
}

#[test]
fn config_view_changes_only_its_keys() {
    let s = Server::start(&[Model::default()], &[ids("ok")], json!({}));
    let (c, v) = s.get("/config", true);
    assert_eq!(c, 200);
    assert_eq!(v["file"], "strata-mock-a.json");
    assert!(v["keys"].as_array().unwrap().iter().any(|k| k["key"] == "sampling.temperature"));
    let (c, r) = s.post("/config", &json!({"set": {"sampling.temperature": 0.6, "fit_max_tokens": true}}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["changed"], json!(["sampling.temperature", "fit_max_tokens"]));
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(s.dir.join("strata-mock-a.json")).unwrap()).unwrap();
    assert_eq!(cfg["sampling"]["temperature"], 0.6);
    assert_eq!(cfg["api_key"], KEY, "the other keys stay");
    assert!(s.dir.join("strata-mock-a.json.bak").exists());
    assert_eq!(s.post("/config", &json!({"set": {"api_key": "x"}})).0, 400);
}

#[test]
fn the_reply_ends_at_the_models_stop_token() {
    let s = Server::start(&[Model::default()], &[ids("Done.<|im_end|>never sent")], json!({"delay_ms": 5}));
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("x")], "chat_template_kwargs": {"enable_thinking": false}}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["choices"][0]["message"]["content"], "Done.");
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
}

#[test]
fn preserve_thinking_puts_the_servers_own_thinking_back() {
    let m = Model { extra: json!({"preserve_thinking": true}), ..Model::default() };
    let s = Server::start(&[m], &[ids("Thought A.\n</think>\n\nAnswer A."), ids("ok")], json!({}));
    s.post("/v1/chat/completions", &json!({"messages": [user("Q1")], "max_tokens": 50}));
    let follow = json!({"messages": [user("Q1"), {"role": "assistant", "content": "Answer A."}, user("Q2")], "max_tokens": 50});
    s.post("/v1/chat/completions", &follow);
    let want = json!({"messages": [user("Q1"), {"role": "assistant", "content": "Answer A.", "reasoning_content": "Thought A."}, user("Q2")],
                      "chat_template_kwargs": {"preserve_thinking": true}});
    assert_eq!(line_ids(&s.gen_lines()[1]), expected_prompt(&want), "the answer's thinking is back in the prompt");
}

#[test]
fn reasoning_close_retry_closes_the_thinking_once() {
    let m = Model { extra: json!({"reasoning_close_retry": true}), ..Model::default() };
    let s = Server::start(&[m], &[ids("Only thinking.<|im_end|>"), ids("Final.")], json!({"delay_ms": 2}));
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("x")], "max_tokens": 100}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["choices"][0]["message"]["reasoning_content"], "Only thinking.\n");
    assert_eq!(r["choices"][0]["message"]["content"], "Final.");
    let g = s.gen_lines();
    assert_eq!(g.len(), 2);
    let close = ids("\n</think>\n\n");
    assert!(line_ids(&g[1]).ends_with(&close), "the second pass goes on after the closed thinking");
}

#[test]
fn reasoning_loop_recovery_stop_ends_a_looping_reply() {
    let m = Model { extra: json!({"reasoning_loop_recovery": "stop"}), context: 65536, ..Model::default() };
    let looping = "I will check the sum again to be sure that it is right. ".repeat(400);
    let s = Server::start(&[m], &[ids(&looping)], json!({}));
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("x")], "max_tokens": 20000}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["choices"][0]["finish_reason"], "length");
    assert!(r["usage"]["completion_tokens"].as_u64().unwrap() < ids(&looping).len() as u64, "ended early");
}

#[test]
fn mcp_tools_run_on_the_server_and_the_model_continues() {
    let m = Model { extra: json!({"mcp_servers": {"calc": {"command": env!("CARGO_BIN_EXE_strata-mock-mcp")}}}), ..Model::default() };
    let call = "\n\n<tool_call>\n<function=calc__add>\n<parameter=a>\n2\n</parameter>\n<parameter=b>\n3\n</parameter>\n</function>\n</tool_call>";
    let s = Server::start(&[m], &[ids(&format!("Use the tool.\n</think>{call}")), ids("Adding.\n</think>\n\nThe sum is 5.")], json!({}));
    let (_, st) = s.get("/mcp", true);
    let t0 = std::time::Instant::now();
    let mut st = st;
    while st["tools"] != 1 && t0.elapsed() < std::time::Duration::from_secs(10) {
        std::thread::sleep(std::time::Duration::from_millis(100));
        st = s.get("/mcp", true).1;
    }
    assert_eq!(st["servers"][0]["status"], "ready", "{st}");
    assert_eq!(st["servers"][0]["tools"][0]["name"], "calc__add");

    let ev = s.stream("/v1/chat/completions", &json!({"messages": [user("2+3?")], "stream": true, "strata_mcp": true}));
    let mcp: Vec<&Value> = ev.iter().filter_map(|(_, v)| v.get("strata_mcp")).collect();
    let kinds: Vec<&str> = mcp.iter().map(|e| e["event"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["start", "call", "result"], "{mcp:?}");
    assert_eq!(mcp[2]["text"], "5");
    assert_eq!(mcp[2]["ok"], true);
    let content: String = ev.iter().filter_map(|(_, v)| v.get("choices")).filter_map(|c| c[0]["delta"]["content"].as_str()).collect();
    assert_eq!(content, "The sum is 5.");
    let calls = ev.iter().filter_map(|(_, v)| v.get("choices")).filter(|c| c[0]["delta"].get("tool_calls").is_some()).count();
    assert_eq!(calls, 0, "the MCP call is not the client's to answer");
    let g = s.gen_lines();
    assert_eq!(g.len(), 2, "a second round after the tool's result");

    // without the opt-in the MCP tools are not offered; from a foreign web page they are refused
    let r = reqwest::blocking::Client::new().post(format!("{}/v1/chat/completions", s.base)).bearer_auth(KEY)
        .header("Origin", "https://evil.example").json(&json!({"messages": [user("x")], "strata_mcp": true})).send().unwrap();
    assert_eq!(r.status().as_u16(), 403);
}

#[test]
fn the_engine_gets_its_eight_token_margin() {
    let s = Server::start(&[Model::default()], &[ids("ok")], json!({}));
    let req = json!({"messages": [user("hi")]});
    let (c, _) = s.post("/v1/chat/completions", &req);
    assert_eq!(c, 200);
    let max_new: usize = s.gen_lines()[0].split(' ').nth(1).unwrap().parse().unwrap();
    assert_eq!(max_new + expected_prompt(&req).len() + 8, 8192, "prompt + max_new + 8 = context");
}

#[test]
fn structured_output_is_checked_before_delivery() {
    let schema = json!({"type": "json_schema", "json_schema": {"name": "p", "schema": {"type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}}});
    let s = Server::start(&[Model::default()],
        &[ids("```json\n{\"name\": \"Ada\", \"age\": 36}\n```"), ids("{\"name\": \"Ada\"}"), ids("{\"name\": \"Bo\", \"age\": 7}")],
        json!({}));
    let base = json!({"messages": [user("x")], "chat_template_kwargs": {"enable_thinking": false}, "response_format": schema});
    let (c, r) = s.post("/v1/chat/completions", &base);
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["choices"][0]["message"]["content"], r#"{"name":"Ada","age":36}"#);
    let (c, r) = s.post("/v1/chat/completions", &base);
    assert_eq!(c, 502, "a missing field fails the schema: {r}");
    assert_eq!(r["error"]["code"], "structured_output_failed");
    let mut st = base.clone();
    st["stream"] = json!(true);
    let ev = s.stream("/v1/chat/completions", &st);
    let content: String = ev.iter().filter_map(|(_, v)| v.get("choices")).filter_map(|c| c[0]["delta"]["content"].as_str()).collect();
    assert_eq!(content, r#"{"name":"Bo","age":7}"#, "streamed only once checked");
    assert!(s.gen_lines()[0].len() > 0);
    let tools = json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]);
    let mut bad = base.clone();
    bad["tools"] = tools;
    assert_eq!(s.post("/v1/chat/completions", &bad).0, 400, "JSON format with tools is refused");
}

#[test]
fn responses_text_format_is_checked() {
    let s = Server::start(&[Model::default()], &[ids("{\"ok\": true}")], json!({}));
    let (c, r) = s.post("/v1/responses", &json!({"input": "x", "reasoning": {"effort": "none"},
        "text": {"format": {"type": "json_object"}}}));
    assert_eq!(c, 200, "{r}");
    let msg = r["output"].as_array().unwrap().iter().find(|o| o["type"] == "message").unwrap();
    assert_eq!(msg["content"][0]["text"], r#"{"ok":true}"#);
}

#[test]
fn api_monitor_keeps_the_calls() {
    let m = Model { extra: json!({"api_monitor": true}), ..Model::default() };
    let s = Server::start(&[m], &[ids("Thought.\n</think>\n\nHello there."), ids("Streamed answer.")], json!({}));
    s.post("/v1/chat/completions", &json!({"messages": [user("hi")], "max_tokens": 50}));
    s.stream("/v1/chat/completions", &json!({"messages": [user("again")], "stream": true, "chat_template_kwargs": {"enable_thinking": false}}));
    let (c, l) = s.get("/api/requests", true);
    assert_eq!(c, 200, "{l}");
    let reqs = l["requests"].as_array().unwrap();
    assert_eq!(reqs.len(), 2);
    assert!(reqs.iter().all(|r| r["state"] == "completed" && r["http_status"] == 200), "{l}");
    assert!(reqs[0].get("input").is_none(), "the list leaves the bodies out");
    let older = reqs[1]["id"].as_str().unwrap();
    let (_, d) = s.get(&format!("/api/requests?id={older}"), true);
    assert_eq!(d["output"], "Hello there.");
    assert_eq!(d["reasoning"], "Thought.\n");
    assert!(d["input"].as_str().unwrap().contains("\"hi\""));
    assert!(d["usage"]["completion_tokens"].as_u64().unwrap() > 0);
    let (_, d) = s.get(&format!("/api/requests?id={}", reqs[0]["id"].as_str().unwrap()), true);
    assert_eq!(d["output"], "Streamed answer.");
    assert_eq!(d["stream"], true);
    let page = reqwest::blocking::get(format!("{}/api-monitor", s.base)).unwrap();
    assert_eq!(page.status().as_u16(), 200);
}

#[test]
fn api_monitor_is_off_by_default() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    assert_eq!(s.get("/api/requests", true).0, 404);
}

#[test]
fn a_model_added_while_running_is_listed_and_loads() {
    let s = Server::start(&[Model::default()], &[ids("hello from b")], json!({}));
    let (_, l) = s.get("/api/models", true);
    assert_eq!(l["models"].as_array().unwrap().len(), 1);
    // a pulled model's config appears beside the others
    let a = std::fs::read_to_string(s.dir.join("strata-mock-a.json")).unwrap().replace("mock-a", "mock-new");
    std::fs::write(s.dir.join("strata-mock-new.json"), a).unwrap();
    let (_, l) = s.get("/api/models", true);
    let ids: Vec<&str> = l["models"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["mock-a", "mock-new"]);
    let (c, r) = s.post("/api/models/load", &json!({"model": "mock-new"}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(s.get("/v1/models", true).1["data"].as_array().unwrap().len(), 2);
}

/// A one-page PDF with `text` on it (Helvetica), xref offsets computed.
fn tiny_pdf(text: &str) -> Vec<u8> {
    let content = format!("BT /F1 18 Tf 72 720 Td ({text}) Tj ET");
    let objs = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".to_string(),
        format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len()),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
    ];
    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (i, o) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend(format!("{} 0 obj\n{o}\nendobj\n", i + 1).bytes());
    }
    let xref = out.len();
    out.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).bytes());
    for off in offsets {
        out.extend(format!("{off:010} 00000 n \n").bytes());
    }
    out.extend(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objs.len() + 1).bytes());
    out
}

fn pdftools() -> bool {
    std::process::Command::new("pdftotext").arg("-v").output().is_ok()
}

#[test]
fn pdf_documents_become_text_and_page_pictures() {
    if !pdftools() {
        eprintln!("skipped: poppler-utils not installed");
        return;
    }
    use base64::Engine as _;
    let pdf = base64::engine::general_purpose::STANDARD.encode(tiny_pdf("Invoice total 4217 EUR"));
    let s = Server::start(&[Model::default()], &[ids("The total is 4217 EUR."), ids("4217.")], json!({"image_tokens": 4}));
    let (c, r) = s.post("/v1/chat/completions", &json!({"max_tokens": 50, "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": [{"type": "text", "text": "What is the total?"},
            {"type": "file", "file": {"filename": "invoice.pdf", "file_data": format!("data:application/pdf;base64,{pdf}")}}]}]}));
    assert_eq!(c, 200, "{r}");
    let g = s.gen_lines();
    assert!(g[0].starts_with("GENI"), "the page goes to the encoder as a picture: {}", &g[0][..40]);
    let tok = tokenizer();
    let prompt = tok.decode(&line_ids(&g[0])).unwrap();
    assert!(prompt.contains("[Document: invoice.pdf, 1 page") && prompt.contains("Invoice total 4217 EUR"), "{prompt}");
    // Anthropic's document block
    let (c, r) = s.post("/v1/messages", &json!({"model": "x", "max_tokens": 50, "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": [{"type": "document", "title": "inv.pdf",
            "source": {"type": "base64", "media_type": "application/pdf", "data": pdf}}, {"type": "text", "text": "Total?"}]}]}));
    assert_eq!(c, 200, "{r}");
    assert!(tok.decode(&line_ids(&s.gen_lines()[1])).unwrap().contains("Invoice total 4217 EUR"));
}

#[test]
fn hugging_face_downloads_check_what_they_are_sent() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    // nothing running yet; the status needs the key
    assert_eq!(s.get("/api/hub/pull", true), (200, json!({"state": "idle"})));
    assert_eq!(s.get("/api/hub/pull", false).0, 401);
    // a repository and a quantization are required, before anything is fetched
    let (c, r) = s.post("/api/hub/quants", &json!({}));
    assert_eq!(c, 400, "{r}");
    let (c, r) = s.post("/api/hub/pull", &json!({"repo": "owner/name"}));
    assert_eq!(c, 400, "{r}");
    assert!(r["error"]["message"].as_str().unwrap().contains("quant"));
    assert_eq!(s.get("/api/hub/pull", true).1["state"], "idle", "a refused request starts nothing");
    // they write files on this PC: another web page may not start one
    for path in ["/api/hub/quants", "/api/hub/pull"] {
        let r = reqwest::blocking::Client::new().post(format!("{}{path}", s.base)).bearer_auth(KEY)
            .header("Origin", "https://evil.example").json(&json!({"repo": "owner/name", "quant": "Q4_K_M"})).send().unwrap();
        assert_eq!(r.status().as_u16(), 403, "{path}");
    }
    assert_eq!(s.get("/api/hub/pull", true).1["state"], "idle");
}

#[test]
fn python_runs_in_a_sandbox_only_when_asked_from_this_page() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    let (c, info) = s.get("/api/run", true);
    assert_eq!(c, 200);
    assert_eq!(s.get("/api/run", false).0, 401);
    let (c, r) = s.post("/api/run", &json!({"language": "ruby", "code": "puts 1"}));
    assert_eq!(c, if info["enabled"] == true { 400 } else { 501 }, "{r}");
    let r = reqwest::blocking::Client::new().post(format!("{}/api/run", s.base)).bearer_auth(KEY)
        .header("Origin", "https://evil.example").json(&json!({"code": "print(1)"})).send().unwrap();
    assert_eq!(r.status().as_u16(), 403, "another web page may not run code here");
    if info["enabled"] != true {
        return;                                       // no bubblewrap on this machine
    }
    let code = "import os, socket\nprint(sorted(os.listdir('/home')))\nopen('out.txt', 'w').write('hi')\n\
                try:\n    socket.create_connection(('1.1.1.1', 80), timeout=2); print('net')\nexcept OSError:\n    print('no net')\n";
    let (c, r) = s.post("/api/run", &json!({"language": "python", "code": code}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["exit_code"], 0, "{r}");
    assert_eq!(r["stdout"], "[]\nno net\n", "home hidden, no network");
    assert_eq!(r["files"][0]["name"], "out.txt");
    assert_eq!(r["files"][0]["data"], "aGk=");
}

#[test]
fn canvas_previews_get_a_sandboxed_address() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    let (c, r) = s.post("/api/preview", &json!({"html": "<h1>hi</h1>"}));
    assert_eq!(c, 200, "{r}");
    let url = r["url"].as_str().unwrap().to_string();
    assert!(url.starts_with("/preview/") && url.len() > 30, "an unguessable address: {url}");
    // the page itself needs no key (a frame cannot send one): its address is the capability
    let page = reqwest::blocking::get(format!("{}{url}", s.base)).unwrap();
    assert_eq!(page.status().as_u16(), 200);
    let csp = page.headers().get("content-security-policy").unwrap().to_str().unwrap().to_string();
    assert!(csp.starts_with("sandbox ") && !csp.contains("allow-same-origin"), "{csp}");
    assert_eq!(page.text().unwrap(), "<h1>hi</h1>");
    assert_eq!(reqwest::blocking::get(format!("{}/preview/nope", s.base)).unwrap().status().as_u16(), 404);
    let r = reqwest::blocking::Client::new().post(format!("{}/api/preview", s.base)).bearer_auth(KEY)
        .header("Origin", "https://evil.example").json(&json!({"html": "x"})).send().unwrap();
    assert_eq!(r.status().as_u16(), 403);
}

#[test]
fn images_are_drawn_by_the_media_engine_and_the_text_model_comes_back() {
    let s = Server::start(&[Model::default()], &[ids("hi"), ids("again")], json!({}));
    assert_eq!(s.get("/health", false).1["image_generation"], true);
    let (c, r) = s.post("/v1/images/generations", &json!({"prompt": "a lighthouse", "size": "512x512", "n": 2, "seed": 7, "steps": 4}));
    assert_eq!(c, 200, "{r}");
    let data = r["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!((data[0]["seed"].as_u64(), data[1]["seed"].as_u64()), (Some(7), Some(8)), "image i uses seed + i");
    assert!(data[0]["b64_json"].as_str().unwrap().starts_with("iVBORw0KGgo"), "a PNG");
    assert_eq!((r["size"].as_str(), r["strata"]["steps"].as_u64()), (Some("512x512"), Some(4)));
    // two generations, then the media engine frees its VRAM; the text model was stopped for the job
    let rec = std::fs::read_to_string(s.dir.join("media-record.txt")).unwrap();
    let ops: Vec<String> = rec.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()["op"].as_str().unwrap().to_string()).collect();
    assert_eq!(ops, ["generate", "generate", "unload"]);
    assert_eq!(s.get("/health", false).1["loaded"], false, "the text model gave the GPU to the image job");
    // the next chat request loads it again
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("hello")]}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(s.get("/health", false).1["loaded"], true);
}

#[test]
fn image_requests_are_checked_and_failures_reported() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    for (body, what) in [(json!({"size": "512x512"}), "prompt"), (json!({"prompt": "p", "size": "300x300"}), "multiple of 16"),
                         (json!({"prompt": "p", "n": 5}), "n must"), (json!({"prompt": "p", "response_format": "url"}), "b64_json")] {
        let (c, r) = s.post("/v1/images/generations", &body);
        assert_eq!(c, 400, "{body}: {r}");
        assert!(r["error"]["message"].as_str().unwrap().contains(what), "{r}");
    }
    let (c, r) = s.post("/v1/images/generations", &json!({"prompt": "FAIL please"}));
    assert_eq!(c, 502, "{r}");
    assert!(r["error"]["message"].as_str().unwrap().contains("mock failure"));
    let n = Server::start(&[Model::default()], &[ids("x")], json!({"no_media": true}));
    assert_eq!(n.get("/health", false).1["image_generation"], false);
    let (c, r) = n.post("/v1/images/generations", &json!({"prompt": "p"}));
    assert_eq!(c, 501, "{r}");
}

#[test]
fn media_models_are_listed_pulled_and_chosen() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    let (c, r) = s.get("/api/media/models", true);
    assert_eq!(c, 200, "{r}");
    let kinds: Vec<&str> = r["models"].as_array().unwrap().iter().map(|m| m["kind"].as_str().unwrap()).collect();
    assert!(kinds.contains(&"image") && kinds.contains(&"video") && kinds.contains(&"audio"), "{kinds:?}");
    assert_eq!(r["image_model"], "z-image-turbo:q8");
    assert_eq!(s.get("/api/media/models", false).0, 401);
    // a downloaded video model becomes the video model; one not downloaded cannot be chosen
    let (c, r) = s.post("/api/media/models/default", &json!({"model": "wan22-ti2v-5b:q8"}));
    assert_eq!((c, r["kind"].as_str(), r["video_model"].as_str()), (200, Some("video"), Some("wan22-ti2v-5b:q8")), "{r}");
    let (c, r) = s.post("/api/media/models/default", &json!({"model": "flux2-klein:q4"}));
    assert!(c == 400 && r["error"]["message"].as_str().unwrap().contains("not downloaded"), "{r}");
    let (c, r) = s.post("/api/media/models/default", &json!({"model": "z-image-turbo:q8"}));
    assert_eq!(c, 200, "{r}");
    // a download runs in the background; its state is polled
    let (c, r) = s.post("/api/media/models/pull", &json!({"model": "flux2-klein:q4"}));
    assert_eq!(c, 202, "{r}");
    let mut state = Value::Null;
    for _ in 0..50 {
        state = s.get("/api/media/models/pull", true).1;
        if state["state"] == "done" {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert_eq!((state["state"].as_str(), state["model"].as_str()), (Some("done"), Some("flux2-klein:q4")), "{state}");
}

#[test]
fn videos_are_made_with_the_models_defaults_and_checked() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    let (c, r) = s.post("/v1/videos/generations", &json!({"prompt": "waves on a beach", "frames": 17, "seed": 3}));
    assert_eq!(c, 200, "{r}");
    assert_eq!((r["frames"].as_u64(), r["fps"].as_u64(), r["size"].as_str()), (Some(17), Some(24), Some("1280x704")),
               "frames from the request, fps and size from the model's defaults: {r}");
    assert_eq!((r["data"][0]["format"].as_str(), r["data"][0]["seed"].as_u64()), (Some("mp4"), Some(3)));
    let rec = std::fs::read_to_string(s.dir.join("media-record.txt")).unwrap();
    let gen: Vec<Value> = rec.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()).filter(|v| v["op"] == "generate").collect();
    assert_eq!((gen[0]["request"]["steps"].as_u64(), gen[0]["request"]["guidance"].as_f64()), (Some(30), Some(5.0)));
    // Wan takes 4n+1 frames: 18 is rounded up to 21, and said so
    let (c, r) = s.post("/v1/videos/generations", &json!({"prompt": "p", "frames": 18}));
    assert_eq!((c, r["frames"].as_u64()), (200, Some(21)), "{r}");
    assert!(r["strata"]["frames_note"].as_str().unwrap().contains("4n+1"), "{r}");
    // a format this engine does not make is refused by name
    let (c, r) = s.post("/v1/videos/generations", &json!({"prompt": "p", "format": "webp"}));
    assert!(c == 400 && r["error"]["message"].as_str().unwrap().contains("mp4, apng, gif"), "{r}");
    for (body, what) in [(json!({"prompt": "p", "model": "z-image-turbo:q8"}), "not a video model"),
                         (json!({"prompt": "p", "size": "100x100"}), "multiple of 16"), (json!({"prompt": "p", "format": "avi"}), "makes")] {
        let (c, r) = s.post("/v1/videos/generations", &body);
        assert_eq!(c, 400, "{body}: {r}");
        assert!(r["error"]["message"].as_str().unwrap().contains(what), "{r}");
    }
}

#[test]
fn media_jobs_show_in_api_calls_and_the_monitor() {
    let m = Model { extra: json!({"api_monitor": true}), ..Model::default() };
    let s = Server::start(&[m], &[ids("x")], json!({}));
    let (c, r) = s.post("/v1/images/generations", &json!({"prompt": "a lighthouse", "size": "512x512", "steps": 4}));
    assert_eq!(c, 200, "{r}");
    let calls = s.get("/api/requests", true).1;
    let rec = calls["requests"].as_array().unwrap().iter().find(|r| r["path"] == "/v1/images/generations").cloned()
        .unwrap_or_else(|| panic!("{calls}"));
    assert_eq!(rec["model"], "z-image-turbo:q8", "{rec}");
    let d = s.get(&format!("/api/requests?id={}", rec["id"].as_str().unwrap()), true).1;
    let resp = d["response"].as_str().unwrap_or_else(|| panic!("{d}"));
    assert!(resp.contains("<PNG, ") && !resp.contains("iVBORw0KGgo"), "the record keeps a note, not the image: {resp}");
    let m = s.get("/metrics", true).1;
    let job = m["requests"].as_array().unwrap().iter().find(|r| r["kind"] == "image").cloned().unwrap();
    assert_eq!((job["finish"].as_str(), job["output"].as_str(), job["steps"].as_u64()), (Some("stop"), Some("1 image, 512x512"), Some(4)), "{job}");
}

fn prompt_text(line: &str) -> String {
    common::tokenizer().decode(&common::line_ids(line)).unwrap()
}

#[test]
fn memory_carries_a_conversation_past_the_context() {
    let m = Model { context: 1024, ..Model::default() };
    let s = Server::start(&[m], &[ids("It is Biscuit."), ids("x")], json!({}));
    let filler = "The weather report said rain and wind and more rain over the hills for the whole week. ".repeat(12);
    let mut messages = vec![user("My dog is called Biscuit."), json!({"role": "assistant", "content": "A fine name."})];
    for i in 0..4 {
        messages.push(user(&format!("Note {i}: {filler}")));
        messages.push(json!({"role": "assistant", "content": "Noted."}));
    }
    messages.push(user("What is my dog called?"));
    // without memory the request does not fit and is refused, as before
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": messages, "max_tokens": 64}));
    assert_eq!(c, 400, "{r}");
    // with it, the oldest turns go to the archive and the one that matches comes back in the <memory> block
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": messages, "max_tokens": 64, "strata_memory": true}));
    assert_eq!(c, 200, "{r}");
    let p = prompt_text(s.gen_lines().last().unwrap());
    assert!(p.contains("<memory>") && p.contains("Biscuit"), "{p}");
    assert!(p.contains("What is my dog called?"));
    assert!(p.matches("Note ").count() < 4, "the oldest notes were moved out: {p}");
    let st = s.get("/api/memory", true).1;
    assert_eq!(st["conversations_archived"], 1, "{st}");
}

#[test]
fn the_model_remembers_with_its_memory_tool() {
    let call = "\n\n<tool_call>\n<function=remember>\n<parameter=fact>\nThe user's favourite colour is teal\n</parameter>\n</function>\n</tool_call>";
    let s = Server::start(&[Model::default()], &[ids(&format!("Save it.\n</think>{call}")), ids("Saved.\n</think>\n\nI will remember that.")], json!({}));
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("Remember: my favourite colour is teal.")], "strata_memory": true}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["choices"][0]["message"]["content"], "I will remember that.");
    let st = s.get("/api/memory", true).1;
    let f = &st["facts"][0];
    assert_eq!((f["text"].as_str(), f["source"].as_str(), f["scope"].as_str()), (Some("The user's favourite colour is teal"), Some("model"), Some("api")), "{st}");
    // facts are added and forgotten through the API too; a foreign page cannot
    let (c, f2) = s.post("/api/memory", &json!({"text": "Deploys go to the 5060 Ti box", "scope": "all"}));
    assert_eq!(c, 200, "{f2}");
    let r = reqwest::blocking::Client::new().delete(format!("{}/api/memory/{}", s.base, f2["id"].as_str().unwrap())).bearer_auth(KEY).send().unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let r = reqwest::blocking::Client::new().post(format!("{}/api/memory", s.base)).bearer_auth(KEY)
        .header("Origin", "https://evil.example").json(&json!({"text": "x"})).send().unwrap();
    assert_eq!(r.status().as_u16(), 403);
}

#[test]
fn speech_is_transcribed_and_spoken_in_openais_shapes() {
    let m = Model { extra: json!({"api_monitor": true}), ..Model::default() };
    let s = Server::start(&[m], &[ids("x")], json!({}));
    let c = reqwest::blocking::Client::new();
    // choose the small Whisper (the default, large-v3-turbo, is not downloaded here)
    let (code, r) = s.post("/api/media/models/default", &json!({"model": "whisper-base"}));
    assert_eq!((code, r["kind"].as_str()), (200, Some("stt")), "{r}");
    let (code, r) = s.post("/api/media/models/default", &json!({"model": "parler-tts-mini-v1"}));
    assert_eq!((code, r["kind"].as_str()), (200, Some("tts")), "{r}");
    // multipart, as OpenAI's clients send it; whisper-1 means the chosen model
    let form = reqwest::blocking::multipart::Form::new().text("model", "whisper-1").text("language", "en")
        .part("file", reqwest::blocking::multipart::Part::bytes(vec![1u8, 2, 3, 4]).file_name("a.wav"));
    let r: Value = c.post(format!("{}/v1/audio/transcriptions", s.base)).bearer_auth(KEY).multipart(form).send().unwrap().json().unwrap();
    assert_eq!(r["text"], "Hello world.", "{r}");
    // JSON with a data URL, and the subtitle formats
    let body = json!({"file": "data:audio/wav;base64,AQIDBA==", "response_format": "srt"});
    let srt = c.post(format!("{}/v1/audio/transcriptions", s.base)).bearer_auth(KEY).json(&body).send().unwrap().text().unwrap();
    assert_eq!(srt, "1\n00:00:00,000 --> 00:00:01,500\nHello world.\n\n");
    let body = json!({"file": "AQIDBA==", "response_format": "verbose_json"});
    let (code, r) = s.post("/v1/audio/translations", &body);
    assert_eq!((code, r["task"].as_str(), r["text"].as_str()), (200, Some("translate"), Some("Hello in English.")), "{r}");
    let (code, r) = s.post("/v1/audio/transcriptions", &json!({"file": "AQIDBA==", "model": "whisper-large-v3-turbo"}));
    assert!(code == 400 && r["error"]["message"].as_str().unwrap().contains("not downloaded"), "{r}");
    // speech: WAV bytes; mp3 is refused by name
    let w = c.post(format!("{}/v1/audio/speech", s.base)).bearer_auth(KEY).json(&json!({"input": "Hello there.", "voice": "alloy"})).send().unwrap();
    assert_eq!(w.headers()["content-type"], "audio/wav");
    assert!(w.bytes().unwrap().starts_with(b"RIFF"));
    let (code, r) = s.post("/v1/audio/speech", &json!({"input": "x", "response_format": "mp3"}));
    assert!(code == 400 && r["error"]["message"].as_str().unwrap().contains("wav"), "{r}");
    let rec = std::fs::read_to_string(s.dir.join("media-record.txt")).unwrap();
    let speak = rec.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()).find(|v| v["op"] == "speak").unwrap();
    assert_eq!((speak["voice"].as_str(), speak["model"].as_str()), (Some("laura"), Some("parler-tts-mini-v1")), "OpenAI's voice maps to one of ours");
    // Monitor and API calls keep both
    let m = s.get("/metrics", true).1;
    let kinds: Vec<&str> = m["requests"].as_array().unwrap().iter().filter_map(|r| r["kind"].as_str()).collect();
    assert!(kinds.contains(&"speech-to-text") && kinds.contains(&"text-to-speech") && kinds.contains(&"translation"), "{kinds:?}");
    let calls = s.get("/api/requests", true).1;
    let sp = calls["requests"].as_array().unwrap().iter().find(|r| r["path"] == "/v1/audio/speech" && r["http_status"] == 200).cloned().unwrap();
    let d = s.get(&format!("/api/requests?id={}", sp["id"].as_str().unwrap()), true).1;
    assert!(d["response"].as_str().unwrap().starts_with("<WAV audio"), "{d}");
    // OpenAI's names (whisper-1, tts-1) are recorded as the models that ran; a multipart form's model is read too
    assert!(calls["requests"].as_array().unwrap().iter().filter(|r| r["path"].as_str().unwrap().starts_with("/v1/audio"))
        .all(|r| r["model"] != "whisper-1" && r["model"] != "tts-1"), "{calls}");
    let form = reqwest::blocking::multipart::Form::new().text("model", "whisper-base")
        .part("file", reqwest::blocking::multipart::Part::bytes(vec![1u8, 2, 3, 4]).file_name("a.wav"));
    assert_eq!(c.post(format!("{}/v1/audio/transcriptions", s.base)).bearer_auth(KEY).multipart(form).send().unwrap().status().as_u16(), 200);
    let calls = s.get("/api/requests", true).1;
    assert_eq!(calls["requests"][0]["model"], "whisper-base", "{}", calls["requests"][0]);
}

#[test]
fn speech_models_leave_the_ram_when_it_is_short_and_stay_warm_when_not() {
    for (mem, unloads) in [(1, true), (64, false)] {
        let s = Server::start(&[Model::default()], &[ids("x")], json!({"mem_gb": mem}));
        s.post("/api/media/models/default", &json!({"model": "whisper-base"}));
        let (c, r) = s.post("/v1/audio/transcriptions", &json!({"file": "AQIDBA=="}));
        assert_eq!(c, 200, "{r}");
        let rec = std::fs::read_to_string(s.dir.join("media-record.txt")).unwrap();
        let ops: Vec<String> = rec.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()["op"].as_str().unwrap().to_string()).collect();
        assert_eq!(ops.contains(&"unload_audio".to_string()), unloads, "{mem} GB free: {ops:?}");
        let t = rec.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()).find(|v| v["op"] == "transcribe").unwrap();
        assert_eq!(t["device"], "cpu", "an engine without CUDA computes on the CPU");
    }
}

#[test]
fn a_chat_waits_behind_a_media_job_in_the_queue_and_says_its_place() {
    let s = Server::start(&[Model::default()], &[ids("Queued answer.")], json!({"media_delay_ms": 1500}));
    let base = s.base.clone();
    let img = std::thread::spawn(move || {
        reqwest::blocking::Client::new().post(format!("{base}/v1/images/generations")).bearer_auth(KEY)
            .json(&json!({"prompt": "a lighthouse", "size": "512x512", "steps": 2})).send().unwrap().status().as_u16()
    });
    // the image job holds the turn
    let t0 = std::time::Instant::now();
    loop {
        let q = s.get("/api/queue", true).1;
        if q["running"]["kind"] == "media" { break; }
        assert!(t0.elapsed().as_secs() < 10, "{q}");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    // a streamed chat now waits; it starts at once and tells its place in SSE comments
    let r = reqwest::blocking::Client::new().post(format!("{}/v1/chat/completions", s.base)).bearer_auth(KEY)
        .json(&json!({"messages": [user("hi")], "stream": true})).send().unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let q = s.get("/api/queue", true).1;
    assert_eq!(q["waiting"][0]["kind"], "chat", "{q}");
    let body = r.text().unwrap();
    assert!(body.contains(": queued: 0 ahead"), "{body}");
    let content: String = body.lines().filter_map(|l| l.strip_prefix("data: ")).filter(|d| *d != "[DONE]")
        .filter_map(|d| serde_json::from_str::<Value>(d).ok()).map(|v| { let d = &v["choices"][0]["delta"];
            format!("{}{}", d["reasoning_content"].as_str().unwrap_or(""), d["content"].as_str().unwrap_or("")) }).collect();
    assert_eq!(content, "Queued answer.", "{body}");
    assert_eq!(img.join().unwrap(), 200);
    let q = s.get("/api/queue", true).1;
    assert!(q["running"].is_null() && q["waiting"].as_array().unwrap().is_empty(), "{q}");
}

#[test]
fn a_chat_queues_an_image_with_its_media_tool_and_the_job_runs_after_the_answer() {
    let call = "\n\n<tool_call>\n<function=generate_image>\n<parameter=prompt>\na lighthouse at dusk\n</parameter>\n</function>\n</tool_call>";
    let s = Server::start(&[Model::default()], &[ids(&format!("Draw it.\n</think>{call}")), ids("Queued.\n</think>\n\nYour lighthouse is on its way."),
                                                 ids("Plain.\n</think>\n\nNo tools here.")], json!({}));
    let (c, caps) = s.get("/api/media/capabilities", true);
    assert_eq!(c, 200);
    assert_eq!((caps["kinds"]["image"]["usable"].as_bool(), caps["kinds"]["video"]["usable"].as_bool()), (Some(true), Some(true)), "{caps}");
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("Draw a lighthouse at dusk.")], "strata_media": true}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["choices"][0]["message"]["content"], "Your lighthouse is on its way.");
    let lines = s.gen_lines();
    let first = prompt_text(&lines[0]);
    assert!(first.contains("generate_image") && first.contains("generate_video") && first.contains("speak"), "{first}");
    assert!(prompt_text(&lines[1]).contains("image job mj-1 queued"), "the tool result went back to the model");
    let t0 = std::time::Instant::now();
    let job = loop {
        let (c, j) = s.get("/api/jobs/mj-1", true);
        assert_eq!(c, 200, "{j}");
        if j["state"] == "done" || j["state"] == "error" { break j; }
        assert!(t0.elapsed().as_secs() < 10, "{j}");
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    assert_eq!((job["state"].as_str(), job["kind"].as_str(), job["mime"].as_str()), (Some("done"), Some("image"), Some("image/png")), "{job}");
    let r = reqwest::blocking::Client::new().get(format!("{}/v1/media/jobs/mj-1/content", s.base)).bearer_auth(KEY).send().unwrap();
    assert_eq!(r.status().as_u16(), 200);
    assert_eq!(&r.bytes().unwrap()[1..4], b"PNG");
    // without the opt-in, no media tools
    let (c, _) = s.post("/v1/chat/completions", &json!({"messages": [user("Hi")]}));
    assert_eq!(c, 200);
    assert!(!prompt_text(s.gen_lines().last().unwrap()).contains("generate_image"));
    assert_eq!(s.get("/api/jobs/mj-9", true).0, 404);
}

#[test]
fn an_audio_attachment_in_a_chat_is_transcribed_into_the_prompt() {
    let s = Server::start(&[Model::default()], &[ids("Heard.\n</think>\n\nYou said hello world.")], json!({}));
    assert_eq!(s.post("/api/media/models/default", &json!({"model": "whisper-base"})).0, 200);
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "What did I say?"},
        {"type": "input_audio", "input_audio": {"data": "UklGRiYAAABXQVZF", "format": "wav", "name": "note.wav"}}]}]}));
    assert_eq!(c, 200, "{r}");
    let p = prompt_text(s.gen_lines().last().unwrap());
    assert!(p.contains("[Transcript of note.wav, 2 s: Hello world.]") && p.contains("What did I say?"), "{p}");
}

/// Not a test: a server with the real web app over mock engines, for checking the page by hand or in a browser
/// (AM2_UI_DEMO_SECS, default 900). cargo test --release --test server web_ui_demo -- --ignored --nocapture
#[test]
#[ignore]
fn web_ui_demo() {
    let img = "\n\n<tool_call>\n<function=generate_image>\n<parameter=prompt>\na lighthouse at dusk\n</parameter>\n</function>\n</tool_call>";
    let spk = "\n\n<tool_call>\n<function=speak>\n<parameter=text>\nGood evening.\n</parameter>\n</function>\n</tool_call>";
    let web = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web");
    let s = Server::start(&[Model::default(), Model { id: "mock-b", ..Model::default() }], &[ids(&format!("Draw.\n</think>{img}")), ids("Ok.\n</think>\n\nYour lighthouse is on its way."),
                                                 ids(&format!("Say it.\n</think>{spk}")), ids("Ok.\n</think>\n\nHere is the greeting, read aloud."),
                                                 ids("Heard.\n</think>\n\nThe recording says: hello world.")],
                          json!({"web_dir": web.to_str().unwrap(), "media_delay_ms": 4000}));
    assert_eq!(s.post("/api/media/models/default", &json!({"model": "whisper-base"})).0, 200);
    println!("UI demo at {} (key {KEY})", s.base);
    let secs = std::env::var("AM2_UI_DEMO_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(900);
    std::thread::sleep(std::time::Duration::from_secs(secs));
}

#[test]
fn one_answer_queues_at_most_three_media_jobs() {
    let call = "\n\n<tool_call>\n<function=generate_image>\n<parameter=prompt>\na fox\n</parameter>\n</function>\n</tool_call>";
    let again = ids(&format!("Again.\n</think>{call}"));
    let s = Server::start(&[Model::default()], &[again.clone(), again.clone(), again.clone(), again, ids("Done.\n</think>\n\nFour foxes coming.")], json!({}));
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("Draw foxes.")], "strata_media": true}));
    assert_eq!(c, 200, "{r}");
    let lines = s.gen_lines();
    assert!(prompt_text(lines.last().unwrap()).contains("already queued 3 media jobs"), "the fourth call was refused");
    assert_eq!(s.get("/api/jobs/mj-3", true).0, 200);
    assert_eq!(s.get("/api/jobs/mj-4", true).0, 404);
}

/// Every route in main.rs but the public ones (health, the web app's files, sandboxed previews) refuses a request
/// without the key, or with a wrong key of the same length.
#[test]
fn security_every_route_needs_the_key() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    let main = include_str!("../src/main.rs");
    let routes = main.split("let app = Router::new()").nth(1).unwrap().split(".with_state(").next().unwrap();
    let caps: Vec<(String, String)> = routes.split(".route(\"").skip(1)
        .map(|r| { let (p, rest) = r.split_once('"').unwrap(); (p.to_string(), rest.lines().next().unwrap().to_string()) }).collect();
    let c = reqwest::blocking::Client::new();
    let mut checked = 0;
    for cap in &caps {
        let cap = [String::new(), cap.0.clone(), cap.1.clone()];
        let path = cap[1].replace("{*rest}", "x").replace("{id}", "mj-1");
        if matches!(path.as_str(), "/health" | "/preview/mj-1") {
            continue;
        }
        for (m, method) in [("get(", reqwest::Method::GET), ("post(", reqwest::Method::POST), ("delete(", reqwest::Method::DELETE)] {
            if !cap[2].contains(m) {
                continue;
            }
            for key in [None, Some("test-kez")] {
                let mut r = c.request(method.clone(), format!("{}{path}", s.base)).header("content-type", "application/json").body("{}");
                if let Some(k) = key {
                    r = r.bearer_auth(k);
                }
                let code = r.send().unwrap().status().as_u16();
                assert_eq!(code, 401, "{method} {path} with key {key:?}");
                checked += 1;
            }
        }
    }
    assert!(checked > 90, "only {checked} checks: did the route list move?");
}

#[test]
fn security_a_rebound_host_and_a_foreign_page_are_refused() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    let c = reqwest::blocking::Client::new();
    // DNS rebinding: another domain pointed at this PC, its page claiming to be this server's own
    let r = c.get(format!("{}/api/models", s.base)).bearer_auth(KEY).header("host", "evil.example:8000")
        .header("origin", "http://evil.example:8000").send().unwrap();
    assert_eq!(r.status().as_u16(), 421);
    for host in ["localhost:1", "127.0.0.1:1", "[::1]:1", "app.localhost"] {
        let r = c.get(format!("{}/api/models", s.base)).bearer_auth(KEY).header("host", host).send().unwrap();
        assert_eq!(r.status().as_u16(), 200, "{host}");
    }
    // a body-less unload from another site
    let r = c.post(format!("{}/api/models/unload", s.base)).bearer_auth(KEY).header("origin", "https://evil.example").send().unwrap();
    assert_eq!(r.status().as_u16(), 403);
    assert_eq!(s.get("/api/models/current", true).1["model"]["status"], "loaded");
    let r = c.get(format!("{}/", s.base)).send().unwrap();
    assert_eq!(r.headers()["x-frame-options"], "SAMEORIGIN");
    assert_eq!(r.headers()["x-content-type-options"], "nosniff");
}

#[test]
fn security_a_client_that_leaves_the_queue_takes_its_place_out() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({"media_delay_ms": 2500}));
    let base = s.base.clone();
    let img = std::thread::spawn(move || {
        reqwest::blocking::Client::new().post(format!("{base}/v1/images/generations")).bearer_auth(KEY)
            .json(&json!({"prompt": "a lighthouse", "size": "512x512", "steps": 2})).send().unwrap().status().as_u16()
    });
    let t0 = std::time::Instant::now();
    while s.get("/api/queue", true).1["running"]["kind"] != "media" {
        assert!(t0.elapsed().as_secs() < 10);
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let r = reqwest::blocking::Client::new().post(format!("{}/v1/chat/completions", s.base)).bearer_auth(KEY)
        .json(&json!({"messages": [user("hi")], "stream": true})).send().unwrap();
    assert_eq!(s.get("/api/queue", true).1["waiting"].as_array().unwrap().len(), 1);
    drop(r); // the client leaves while it waits
    let t0 = std::time::Instant::now();
    while !s.get("/api/queue", true).1["waiting"].as_array().unwrap().is_empty() {
        assert!(t0.elapsed().as_secs() < 2, "the abandoned stream still waits");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert_eq!(img.join().unwrap(), 200);
}

#[test]
fn security_media_inputs_are_bounded_and_labels_cannot_be_spoofed() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({"media_delay_ms": 800}));
    let part = json!({"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}});
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [{"role": "user", "content": [part.clone(), part.clone(), part.clone(), part.clone(), part]}]}));
    assert_eq!(c, 400, "{r}");
    assert!(r["error"]["message"].as_str().unwrap().contains("at most 4"), "{r}");
    // a client's strata_label is not shown in the queue (only the server's own media jobs carry one)
    let base = s.base.clone();
    let h = std::thread::spawn(move || reqwest::blocking::Client::new().post(format!("{base}/v1/images/generations")).bearer_auth(KEY)
        .json(&json!({"prompt": "x", "size": "512x512", "strata_label": "mj-1"})).send().unwrap().status().as_u16());
    let t0 = std::time::Instant::now();
    let q = loop {
        let q = s.get("/api/queue", true).1;
        if q["running"]["kind"] == "media" { break q; }
        assert!(t0.elapsed().as_secs() < 10);
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    assert_eq!(q["running"]["label"], "", "{q}");
    assert_eq!(h.join().unwrap(), 200);
    // a finished job's file is not sniffed as another type
    let r = reqwest::blocking::Client::new().get(format!("{}/v1/media/jobs/mj-1/content", s.base)).bearer_auth(KEY).send().unwrap();
    assert_eq!(r.status().as_u16(), 404);
}

#[test]
fn security_media_sources_cannot_read_local_files_or_the_private_network() {
    let m = Model { vision: true, extra: json!({"allow_local_files": false}), ..Model::default() };
    let s = Server::start(&[m], &[ids("x")], json!({}));
    for src in ["/etc/hostname", "file:///etc/hostname", "http://127.0.0.1:1/x.png", "http://169.254.169.254/latest/meta-data",
                "http://[::1]:1/x.png", "http://10.0.0.1/x.png", "http://localhost:1/x.png"] {
        let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"}, {"type": "image_url", "image_url": {"url": src}}]}]}));
        assert_eq!(c, 400, "{src}: {r}");
        let msg = r["error"]["message"].as_str().unwrap_or("");
        assert!(msg.contains("not read") || msg.contains("private network"), "{src}: {msg}");
    }
}

#[test]
fn anthropic_and_responses_clients_get_the_media_tools_too() {
    let call = "\n\n<tool_call>\n<function=generate_image>\n<parameter=prompt>\na fox\n</parameter>\n</function>\n</tool_call>";
    let s = Server::start(&[Model::default()], &[ids(&format!("Draw.\n</think>{call}")), ids("Ok.\n</think>\n\nThe fox is on its way.")], json!({}));
    let (c, r) = s.post("/v1/messages", &json!({"max_tokens": 500, "messages": [user("Draw a fox.")], "strata_media": true}));
    assert_eq!(c, 200, "{r}");
    let text: String = r["content"].as_array().unwrap().iter().filter_map(|b| b["text"].as_str()).collect();
    assert_eq!(text, "The fox is on its way.", "{r}");
    assert_eq!(r["stop_reason"], "end_turn", "the server's own tool call does not end the turn: {r}");
    assert_eq!(s.get("/api/jobs/mj-1", true).0, 200);
    let s = Server::start(&[Model::default()], &[ids(&format!("Draw.\n</think>{call}")), ids("Ok.\n</think>\n\nA second fox is coming.")], json!({}));
    let (c, r) = s.post("/v1/responses", &json!({"input": "Draw another fox.", "strata_media": true}));
    assert_eq!(c, 200, "{r}");
    let out = r["output"].to_string();
    assert!(out.contains("A second fox is coming."), "{r}");
    assert!(!out.contains("function_call"), "{r}");
    assert_eq!(s.get("/api/jobs/mj-1", true).0, 200);
}

#[test]
fn a_downloaded_model_is_removed_from_the_disk_keeping_what_others_use() {
    let b = Model { id: "mock-b", ..Model::default() };
    let s = Server::start(&[Model::default(), b], &[ids("x")], json!({}));
    let d = s.dir.clone();
    let mk = |p: &std::path::Path| { std::fs::create_dir_all(p.parent().unwrap()).unwrap(); std::fs::write(p, vec![7u8; 1000]).unwrap(); };
    let (own, pack, shared) = (d.join("models/B-dl/b.gguf"), d.join("packs/mock-b/profile.txt"), d.join("models/shared-dl/s.bin"));
    for p in [&own, &pack, &shared] { mk(p); }
    // a download folder that is a link to a cache elsewhere: the link goes, the cache stays
    let outside = std::env::temp_dir().join(format!("am2strata-outside-{}", uuid::Uuid::new_v4().simple()));
    mk(&outside.join("cached.gguf"));
    std::os::unix::fs::symlink(&outside, d.join("models/linked")).unwrap();
    let edit = |id: &str, k: &str, v: Value| {
        let p = d.join(format!("strata-{id}.json"));
        let mut c: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        c[k] = v;
        std::fs::write(&p, serde_json::to_vec_pretty(&c).unwrap()).unwrap();
    };
    edit("mock-b", "files", json!([own, pack, shared, d.join("models/linked/cached.gguf")]));
    edit("mock-a", "files", json!([shared]));
    // the loaded (and start-up) model cannot go
    let (c, p) = s.get("/api/models/mock-a/removal", true);
    assert_eq!(c, 200);
    assert!(p["blocked"].as_str().unwrap().contains("loaded"), "{p}");
    let del = |id: &str, origin: Option<&str>| {
        let mut r = reqwest::blocking::Client::new().delete(format!("{}/api/models/{id}", s.base)).bearer_auth(KEY);
        if let Some(o) = origin { r = r.header("origin", o); }
        let r = r.send().unwrap();
        (r.status().as_u16(), r.json::<Value>().unwrap_or(Value::Null))
    };
    assert_eq!(del("mock-a", None).0, 409);
    assert_eq!(del("mock-b", Some("https://evil.example")).0, 403);
    // the plan, then the removal
    let (c, p) = s.get("/api/models/mock-b/removal", true);
    assert_eq!(c, 200, "{p}");
    assert!(p["blocked"].is_null(), "{p}");
    let paths: Vec<&str> = p["files"].as_array().unwrap().iter().map(|f| f["path"].as_str().unwrap()).collect();
    assert!(paths.iter().any(|x| x.ends_with("models/B-dl")) && paths.iter().any(|x| x.ends_with("packs/mock-b"))
            && paths.iter().any(|x| x.ends_with("models/linked")) && !paths.iter().any(|x| x.contains("shared-dl")), "{p}");
    assert!(p["kept"].to_string().contains("also used by mock-a"), "{p}");
    let (c, r) = del("mock-b", None);
    assert_eq!(c, 200, "{r}");
    assert!(r["freed_bytes"].as_u64().unwrap() >= 2000, "{r}");
    assert!(!d.join("models/B-dl").exists() && !d.join("packs/mock-b").exists() && !d.join("strata-mock-b.json").exists());
    assert!(std::fs::symlink_metadata(d.join("models/linked")).is_err(), "the link is gone");
    assert!(outside.join("cached.gguf").exists(), "what it pointed at is not touched");
    assert!(shared.exists(), "mock-a still uses it");
    let list = s.get("/api/models", true).1.to_string();
    assert!(!list.contains("mock-b"), "{list}");
    let _ = std::fs::remove_dir_all(&outside);
}

#[test]
fn a_media_model_is_removed_through_the_media_engine() {
    let s = Server::start(&[Model::default()], &[ids("x")], json!({}));
    let (c, p) = s.get("/api/media/models/z-image-turbo:q8/removal", true);
    assert_eq!(c, 200, "{p}");
    assert_eq!(p["bytes"], 6_580_000_000u64);
    let r = reqwest::blocking::Client::new().delete(format!("{}/api/media/models/z-image-turbo:q8", s.base)).bearer_auth(KEY).send().unwrap();
    assert_eq!(r.status().as_u16(), 200);
    assert_eq!(r.json::<Value>().unwrap()["freed_bytes"], 6_580_000_000u64);
    let r = reqwest::blocking::Client::new().delete(format!("{}/api/media/models/flux2-klein:q4", s.base)).bearer_auth(KEY).send().unwrap();
    assert_eq!(r.status().as_u16(), 400);
}

/// A tiny HTTP server for the web tools' tests: SearXNG's JSON at /search, a page at /page.
fn fake_web() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for c in l.incoming().flatten() {
            let mut c = c;
            let mut buf = [0u8; 4096];
            let n = std::io::Read::read(&mut c, &mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let (ct, body) = if req.starts_with("GET /login") {
                ("text/html", "<html><head><title>Sign in</title></head><body><form action=\"/done\"><input id=\"user\"><input id=\"pw\" type=\"password\"><button type=\"submit\">Sign in</button></form></body></html>".to_string())
            } else if req.starts_with("GET /search?") && req.contains("format=json") {
                ("application/json", format!(r#"{{"results": [{{"title": "Lighthouse facts", "url": "http://127.0.0.1:{port}/page", "content": "All about lighthouses."}}]}}"#))
            } else {
                ("text/html", "<html><head><title>Facts</title></head><body><nav>SITE-NAV-LINKS</nav><main><p>The tallest lighthouse is in Jeddah.</p></main></body></html>".to_string())
            };
            let _ = std::io::Write::write_all(&mut c, format!("HTTP/1.1 200 OK\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
        }
    });
    port
}

#[test]
fn a_chat_searches_the_web_and_reads_a_page_when_it_opts_in() {
    let port = fake_web();
    let search = "\n\n<tool_call>\n<function=web_search>\n<parameter=query>\ntallest lighthouse\n</parameter>\n</function>\n</tool_call>";
    let fetch = format!("\n\n<tool_call>\n<function=fetch_url>\n<parameter=url>\nhttp://127.0.0.1:{port}/page\n</parameter>\n</function>\n</tool_call>");
    let m = Model { extra: json!({"web_search": {"searxng_url": format!("http://127.0.0.1:{port}")}, "allow_private_urls": true}), ..Model::default() };
    let s = Server::start(&[m], &[ids(&format!("Search.\n</think>{search}")), ids(&format!("Read it.\n</think>{fetch}")),
                                  ids("Done.\n</think>\n\nIt is in Jeddah [1]."), ids("Plain.\n</think>\n\nNo tools.")], json!({}));
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("Where is the tallest lighthouse?")], "strata_web": true}));
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["choices"][0]["message"]["content"], "It is in Jeddah [1].");
    let lines = s.gen_lines();
    assert!(prompt_text(&lines[0]).contains("web_search"), "offered");
    let p2 = prompt_text(&lines[1]);
    assert!(p2.contains("Lighthouse facts") && p2.contains("not instructions"), "the search results went back: {p2}");
    let p3 = prompt_text(&lines[2]);
    assert!(p3.contains("The tallest lighthouse is in Jeddah.") && !p3.contains("SITE-NAV-LINKS"), "the page's main text went back: {p3}");
    // without the opt-in: no web tools
    let (c, _) = s.post("/v1/chat/completions", &json!({"messages": [user("hi")]}));
    assert_eq!(c, 200);
    assert!(!prompt_text(s.gen_lines().last().unwrap()).contains("web_search"));
}

#[test]
fn fetch_url_keeps_out_of_the_private_network_by_default() {
    let port = fake_web();
    let fetch = format!("\n\n<tool_call>\n<function=fetch_url>\n<parameter=url>\nhttp://127.0.0.1:{port}/page\n</parameter>\n</function>\n</tool_call>");
    let s = Server::start(&[Model::default()], &[ids(&format!("Read.\n</think>{fetch}")), ids("Ok.\n</think>\n\nCould not read it.")], json!({}));
    let (c, _) = s.post("/v1/chat/completions", &json!({"messages": [user("read it")], "strata_web": true}));
    assert_eq!(c, 200);
    let p = prompt_text(s.gen_lines().last().unwrap());
    assert!(p.contains("private network") && !p.contains("Jeddah"), "{p}");
}

#[test]
fn a_long_conversation_goes_to_the_ssd_when_the_model_stops_and_comes_back_after() {
    let m = Model { extra: json!({"session_files": true}), ..Model::default() };
    let s = Server::start(&[m], &[ids("Answer one."), ids("Answer two.")], json!({}));
    // a prompt of some thousands of tokens
    let long = "the quick brown fox jumps over the lazy dog ".repeat(400);
    let first = json!({"messages": [user(&long)], "chat_template_kwargs": {"enable_thinking": false}});
    let (c, r) = s.post("/v1/chat/completions", &first);
    assert_eq!(c, 200, "{r}");
    // a media job stops the text model: its conversation is saved first
    let (c, _) = s.post("/v1/images/generations", &json!({"prompt": "x", "size": "512x512", "steps": 1}));
    assert_eq!(c, 200);
    let cmds = s.commands();
    let save = cmds.iter().position(|c| c.starts_with("SAVE ")).expect("SAVE before the stop");
    let file = cmds[save]["SAVE ".len()..].to_string();
    assert!(std::path::Path::new(&file).is_file() && file.contains("/cache/kv/mock-a/"), "{file}");
    assert!(std::path::Path::new(&file.replace(".sess", ".ids")).is_file());
    // the next turn of the same conversation: the model loads again, the snapshot is read back, then GEN
    let mut msgs = first["messages"].as_array().unwrap().clone();
    msgs.push(json!({"role": "assistant", "content": "Answer one."}));
    msgs.push(user("and then?"));
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": msgs, "chat_template_kwargs": {"enable_thinking": false}}));
    assert_eq!(c, 200, "{r}");
    let cmds = s.commands();
    let restore = cmds.iter().rposition(|c| c.starts_with("RESTORE ")).expect("RESTORE after the reload");
    let gen = cmds.iter().rposition(|c| c.starts_with("GEN ")).unwrap();
    assert!(restore < gen && cmds[restore].ends_with(&file), "{:?}", &cmds[save..]);
}

#[test]
fn a_snapshot_the_engine_refuses_is_dropped_and_the_request_goes_on() {
    let m = Model { extra: json!({"session_files": true}), ..Model::default() };
    let s = Server::start(&[m], &[ids("One."), ids("Two.")], json!({}));
    let long = "a b c d e f g h ".repeat(800);
    let first = json!({"messages": [user(&long)], "chat_template_kwargs": {"enable_thinking": false}});
    assert_eq!(s.post("/v1/chat/completions", &first).0, 200);
    assert_eq!(s.post("/v1/images/generations", &json!({"prompt": "x", "size": "512x512", "steps": 1})).0, 200);
    let cmds = s.commands();
    let file = cmds.iter().find(|c| c.starts_with("SAVE ")).unwrap()["SAVE ".len()..].to_string();
    std::fs::remove_file(&file).unwrap(); // gone: the engine refuses it
    let mut msgs = first["messages"].as_array().unwrap().clone();
    msgs.push(json!({"role": "assistant", "content": "One."}));
    msgs.push(user("more"));
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": msgs, "chat_template_kwargs": {"enable_thinking": false}}));
    assert_eq!(c, 200, "{r}");
    assert!(!r["choices"][0]["message"]["content"].as_str().unwrap_or("").is_empty(), "{r}");
    assert!(s.commands().iter().any(|c| c.starts_with("RESTORE ")), "it was tried");
    assert!(!std::path::Path::new(&file.replace(".sess", ".ids")).exists(), "the refused snapshot was dropped");
}

/// The headless Chrome these tests drive (skipped where there is none).
fn chrome() -> Option<String> {
    let p = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("Development/Strata-data/browser/chrome-headless-shell-linux64/chrome-headless-shell");
    p.is_file().then(|| p.display().to_string())
}

#[test]
fn the_browser_opens_a_page_and_keeps_out_of_the_private_network_by_default() {
    let Some(exe) = chrome() else { eprintln!("no chrome-headless-shell: skipped"); return };
    let port = fake_web();
    let open = format!("\n\n<tool_call>\n<function=browser_open>\n<parameter=url>\nhttp://127.0.0.1:{port}/page\n</parameter>\n</function>\n</tool_call>");
    // allowed to the private network (the test page is on this PC)
    let m = Model { extra: json!({"browser": {"exe": exe}, "allow_private_urls": true}), ..Model::default() };
    let s = Server::start(&[m], &[ids(&format!("Open.\n</think>{open}")), ids("Ok.\n</think>\n\nIt says Jeddah.")], json!({}));
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("open the page")], "strata_browser": true}));
    assert_eq!(c, 200, "{r}");
    let p = prompt_text(s.gen_lines().last().unwrap());
    assert!(p.contains("The tallest lighthouse is in Jeddah.") && p.contains("\"Facts\""), "the page's text went back: {p}");
    // the default: this PC's pages are not opened
    let m = Model { extra: json!({"browser": {"exe": exe}}), ..Model::default() };
    let s = Server::start(&[m], &[ids(&format!("Open.\n</think>{open}")), ids("Ok.\n</think>\n\nBlocked.")], json!({}));
    let (c, _) = s.post("/v1/chat/completions", &json!({"messages": [user("open the page")], "strata_browser": true}));
    assert_eq!(c, 200);
    let p = prompt_text(s.gen_lines().last().unwrap());
    assert!(p.contains("not a public web address") && !p.contains("Jeddah"), "{p}");
}

#[test]
fn a_password_in_the_browser_waits_for_the_users_yes() {
    let Some(exe) = chrome() else { eprintln!("no chrome-headless-shell: skipped"); return };
    let port = fake_web();
    let open = format!("\n\n<tool_call>\n<function=browser_open>\n<parameter=url>\nhttp://127.0.0.1:{port}/login\n</parameter>\n</function>\n</tool_call>");
    let typ = "\n\n<tool_call>\n<function=browser_type>\n<parameter=selector>\n#pw\n</parameter>\n<parameter=text>\nhunter2\n</parameter>\n</function>\n</tool_call>";
    let m = Model { extra: json!({"browser": {"exe": exe}, "allow_private_urls": true}), ..Model::default() };
    let s = Server::start(&[m], &[ids(&format!("Open.\n</think>{open}")), ids(&format!("Type.\n</think>{typ}")),
                                  ids("Ok.\n</think>\n\nYou did not allow it.")], json!({}));
    let r = reqwest::blocking::Client::new().post(format!("{}/v1/chat/completions", s.base)).bearer_auth(KEY)
        .json(&json!({"messages": [user("sign in")], "strata_browser": true, "stream": true})).send().unwrap();
    let mut lines = std::io::BufRead::lines(std::io::BufReader::new(r));
    let mut denied = false;
    let mut body = String::new();
    while let Some(Ok(l)) = lines.next() {
        body.push_str(&l);
        body.push('\n');
        let Some(d) = l.strip_prefix("data: ") else { continue };
        let Ok(v) = serde_json::from_str::<Value>(d) else { continue };
        if v["strata_mcp"]["event"] == "approve" && !denied {
            assert!(v["strata_mcp"]["what"].as_str().unwrap().contains("password"), "{v}");
            let id = v["strata_mcp"]["id"].as_str().unwrap().to_string();
            let (c, a) = s.post("/api/tools/approve", &json!({"id": id, "ok": false}));
            assert_eq!(c, 200, "{a}");
            denied = true;
        }
    }
    assert!(denied, "an approval was asked: {body}");
    assert!(body.contains("did not allow this step"), "{body}");
}

/// Not a test: the web app over mock engines with the real browser on a sign-in page (AM2_UI_DEMO_SECS).
#[test]
#[ignore]
fn web_ui_demo_browser() {
    let exe = chrome().expect("chrome-headless-shell");
    let port = fake_web();
    let open = format!("\n\n<tool_call>\n<function=browser_open>\n<parameter=url>\nhttp://127.0.0.1:{port}/login\n</parameter>\n</function>\n</tool_call>");
    let typ = "\n\n<tool_call>\n<function=browser_type>\n<parameter=selector>\n#pw\n</parameter>\n<parameter=text>\nhunter2\n</parameter>\n</function>\n</tool_call>";
    let web = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web");
    let m = Model { extra: json!({"browser": {"exe": exe}, "allow_private_urls": true}), ..Model::default() };
    let s = Server::start(&[m], &[ids(&format!("Open.\n</think>{open}")), ids(&format!("Type.\n</think>{typ}")),
                                  ids("Ok.\n</think>\n\nDone with the sign-in page.")], json!({"web_dir": web.to_str().unwrap()}));
    println!("UI demo at {} (key {KEY})", s.base);
    let secs = std::env::var("AM2_UI_DEMO_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(900);
    std::thread::sleep(std::time::Duration::from_secs(secs));
}
