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
    let (c, m) = s.get("/mcp", false);
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
