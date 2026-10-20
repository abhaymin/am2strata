//! One request on the loaded model: prompt, pictures, sampling, generation, output parsing and stop strings.
//! Both APIs (OpenAI and Anthropic) format the events this produces.

use std::collections::HashMap;
use crate::manager::{Loaded, Manager};
use crate::media;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use strata_engine::vision::normalize_image;
use strata_engine::GenEvent;
use strata_frontend::images::{expand_images, images_of};
use strata_frontend::output::{Event, OutputParser};
use strata_frontend::request::forced_call;
use strata_frontend::Prepared;
use strata_protocol::{Images, Sampling};
use strata_tokenizer::Detokenizer;
use tokio::sync::mpsc;

/// The engine keeps a few tokens of slack at the end of the context (serve/server.py CTX_SLACK).
const CTX_SLACK: usize = 1;
/// #1053: what closes the thinking when a reply stopped inside it (reasoning_close_retry)
const REASONING_CLOSE: &str = "\n</think>\n\n";
/// #728: the effort sentences the loop recovery swaps (the template's own xhigh one, and low)
const HIGH_EFFORT: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, \
                           consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const LOW_EFFORT: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion \
                          without unnecessary elaboration.";
const LOOP_CHECK_EVERY: usize = 512;
const LOOP_COVERAGE: f64 = 0.25;
const LOOP_HISTORY_WORDS: usize = 30000;
const LOOP_TAIL_CHARS: usize = 2048;
const LOOP_TAIL_PERIOD: usize = 64;

/// #728: the share of the last 2,000 words inside 12-word passages seen at least three times; a tail of the last
/// 2,048 characters with a period of at most 64 reads as full coverage (#1753). Port of reasoning_repeat_coverage.
pub fn reasoning_repeat_coverage(text: &str) -> f64 {
    let lower: Vec<char> = text.to_lowercase().chars().collect();
    if lower.len() >= LOOP_TAIL_CHARS {
        let tail = &lower[lower.len() - LOOP_TAIL_CHARS..];
        if (1..=LOOP_TAIL_PERIOD).any(|p| tail[p..] == tail[..tail.len() - p]) {
            return 1.0;
        }
    }
    let mut words: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in lower {
        if c.is_alphanumeric() || c == '_' {
            cur.push(c);
            continue;
        }
        if !cur.is_empty() {
            words.push(std::mem::take(&mut cur));
        }
        if !c.is_whitespace() {
            words.push(c.to_string());
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    if words.len() > LOOP_HISTORY_WORDS {
        words.drain(..words.len() - LOOP_HISTORY_WORDS);
    }
    let (width, window) = (12usize, 2000usize);
    if words.len() < window {
        return 0.0;
    }
    let grams: Vec<&[String]> = words.windows(width).collect();
    let mut counts: HashMap<&[String], usize> = HashMap::new();
    for g in &grams {
        *counts.entry(*g).or_default() += 1;
    }
    let start = words.len() - window;
    let mut covered = vec![false; words.len()];
    for i in start..grams.len() {
        if counts[grams[i]] >= 3 {
            covered[i..i + width].iter_mut().for_each(|c| *c = true);
        }
    }
    covered[start..].iter().filter(|c| **c).count() as f64 / (words.len() - start) as f64
}

/// #728: the prompt's ids with the xhigh effort sentence of the first system message replaced by the low one, then
/// every token generated so far (a splice of ids: the prompt is never decoded and encoded again).
fn focused_recovery_prompt(tok: &strata_tokenizer::Tokenizer, ids: &[u32], generated: &[u32]) -> Option<Vec<u32>> {
    let end = tok.encode("<|im_end|>", true).ok().filter(|v| v.len() == 1)?[0];
    let head = &ids[..ids.iter().position(|t| *t == end)?];
    if !tok.decode(head).ok()?.starts_with("<|im_start|>system\n") {
        return None;
    }
    let (old, new) = (tok.encode(HIGH_EFFORT, false).ok()?, tok.encode(LOW_EFFORT, false).ok()?);
    let i = head.windows(old.len()).position(|w| w == old.as_slice())?;
    Some([&ids[..i], &new, &ids[i + old.len()..], generated].concat())
}
/// #606: a reply that repeats one token this many times in a row is ended there ("length"); the config's
/// "repeat_stop_tokens" sets it, 0 turns it off.
const REPEAT_STOP_TOKENS: usize = 256;

pub enum RunEvent {
    /// the prompt is being read: nothing to say yet (a keep-alive)
    Ping,
    Event(Event),
    Done(Summary),
    Error(String),
    /// MCP tool activity (the web app's `strata_mcp` chunks)
    Mcp(Value),
}

#[derive(Debug, Clone)]
pub struct Summary {
    /// "stop", "length" or "cancel"
    pub finish: String,
    pub completion_tokens: u64,
    pub reused: u64,
    pub stop_sequence: Option<String>,
    pub timings: Value,
}

/// A request refused before anything is generated: the status and the sentence.
pub struct Refusal(pub u16, pub String);

impl Refusal {
    fn bad(msg: impl Into<String>) -> Self {
        Refusal(400, msg.into())
    }
}

/// The model card's sampling ("Best Practices") for the family and thinking state (PolyStrata 0.1.45).
pub fn model_card_defaults(family: Option<&str>, thinking: bool) -> crate::settings::Defaults {
    let (t, p, pp) = match (family, thinking) {
        (Some("qwen4exp"), true) => (1.0, 0.95, None),
        (Some("qwen4exp"), false) => (0.7, 0.8, Some(1.5)),
        (Some("qwen35moe"), true) => (1.0, 0.95, Some(1.5)),
        (Some("qwen35moe"), false) => (0.7, 0.8, Some(1.5)),
        _ => return Default::default(),
    };
    let mut d = crate::settings::Defaults::new();
    d.insert("temperature".into(), json!(t));
    d.insert("top_p".into(), json!(p));
    d.insert("top_k".into(), json!(20));
    if let Some(pp) = pp {
        d.insert("presence_penalty".into(), json!(pp));
    }
    d
}

/// The request's sampling over the defaults (model card, config, shared settings), key by key: a field the request
/// sets wins (an explicit temperature 0 stays greedy), a field set to null falls back to the default.
pub fn sampling_of(req: &Value, defaults: &crate::settings::Defaults) -> Sampling {
    let get = |k: &str| req.get(k).filter(|v| !v.is_null()).or_else(|| defaults.get(k)).filter(|v| !v.is_null());
    let f = |k: &str| get(k).and_then(Value::as_f64);
    let i = |k: &str| get(k).and_then(|v| v.as_i64().or_else(|| v.as_f64().filter(|x| x.fract() == 0.0).map(|x| x as i64)));
    Sampling {
        temperature: f("temperature"),
        top_p: f("top_p"),
        top_k: i("top_k"),
        min_p: f("min_p"),
        repetition_penalty: f("repetition_penalty"),
        frequency_penalty: f("frequency_penalty"),
        presence_penalty: f("presence_penalty"),
        penalty_last_n: i("penalty_last_n"),
        seed: i("seed"),
    }
}

/// The request's stop strings: OpenAI's `stop` (a string or up to 4) or Anthropic's `stop_sequences`.
pub fn stop_strings(req: &Value) -> Result<Vec<String>, Refusal> {
    let v = req.get("stop").filter(|v| !v.is_null()).or_else(|| req.get("stop_sequences").filter(|v| !v.is_null()));
    let list = match v {
        None => return Ok(Vec::new()),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) if a.iter().all(Value::is_string) => a.iter().map(|s| s.as_str().unwrap().to_string()).collect(),
        Some(_) => return Err(Refusal::bad("stop must be a string or a list of strings")),
    };
    if req.get("stop").is_some_and(|v| !v.is_null()) && list.len() > 4 {
        return Err(Refusal::bad("stop takes at most 4 strings"));
    }
    Ok(list.into_iter().filter(|s| !s.is_empty()).collect())
}

/// Cuts the answer text at the first stop string, holding back what could be the start of one.
struct StopCutter {
    stops: Vec<String>,
    held: String,
    hit: Option<String>,
}

impl StopCutter {
    fn push(&mut self, text: &str) -> String {
        if self.stops.is_empty() {
            return text.to_string();
        }
        self.held.push_str(text);
        if let Some((at, s)) = self.stops.iter().filter_map(|s| self.held.find(s.as_str()).map(|i| (i, s))).min_by_key(|(i, _)| *i) {
            let out = self.held[..at].to_string();
            self.hit = Some(s.clone());
            self.held.clear();
            return out;
        }
        let keep = self.stops.iter().map(|s| {
            (1..s.len().min(self.held.len() + 1)).rev().find(|&n| {
                self.held.is_char_boundary(self.held.len() - n) && s.starts_with(&self.held[self.held.len() - n..])
            }).unwrap_or(0)
        }).max().unwrap_or(0);
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held.drain(..cut);
        out
    }

    fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

fn timings(d: &strata_protocol::Done) -> Value {
    let per_s = |n: u64, ms: f64| if ms > 0.0 { n as f64 * 1000.0 / ms } else { 0.0 };
    let read = d.prompt_read.unwrap_or(d.prompt_tokens.saturating_sub(d.reused.unwrap_or(0)));
    json!({
        "cache_n": d.reused.unwrap_or(0), "prompt_n": read, "prompt_ms": d.prompt_ms,
        "prompt_per_second": per_s(read, d.prompt_ms), "predicted_n": d.generated, "predicted_ms": d.decode_ms,
        "predicted_per_second": per_s(d.generated, d.decode_ms),
        "draft_n": d.drafts_offered, "draft_n_accepted": d.drafts_accepted,
    })
}

pub struct Started {
    pub events: mpsc::Receiver<RunEvent>,
    pub prompt_tokens: usize,
    pub model: String,
}

/// What a request asks beyond its messages.
#[derive(Default, Clone)]
pub struct RunOpts {
    /// report tool calls while they are being written
    pub stream_tools: bool,
    /// the most tokens to think (Anthropic's budget_tokens, or reasoning_budget_tokens)
    pub budget: Option<u64>,
    /// the request's tool_choice, in either API's shape
    pub tool_choice: Option<Value>,
}

/// From this max_tokens on, thinking without a budget is closed early enough to leave the answer room (0.1.45).
const ANSWER_ROOM_MIN_CAP: usize = 1024;
/// The tokens kept for the answer when a thinking budget is cut down to fit max_tokens (#984).
const ANSWER_RESERVE_MIN: usize = 512;
/// What closes the thinking at its budget (#123): the model's own end-of-thinking tag after a short sentence.
const REASONING_WRAP_UP: &str = "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n";

/// The thinking budget for this request: its own, cut to leave the answer room, or (without one, from
/// ANSWER_ROOM_MIN_CAP) three quarters of max_tokens at most.
pub fn thinking_budget(asked: Option<u64>, max_new: usize, answer_room: bool) -> Option<usize> {
    let reserve = (max_new / 4).max(ANSWER_RESERVE_MIN).min(max_new / 2);
    let mut budget = asked.filter(|b| *b > 0).map(|b| b as usize);
    if budget.is_none() && answer_room && max_new >= ANSWER_ROOM_MIN_CAP {
        budget = Some(max_new - reserve);
    }
    budget.map(|b| if b > max_new - reserve { (max_new - reserve).max(1) } else { b })
}

/// Start a request. `prep`: the normalized messages, tools and template kwargs; `req`: the raw request (sampling,
/// max tokens, stop strings).
// (the emit! macro also runs after the last check of `answered`)
#[allow(unused_assignments)]
pub async fn start(mgr: Arc<Manager>, loaded: Arc<Loaded>, prep: Prepared, req: Value, opts: RunOpts)
    -> Result<Started, Refusal> {
    let stops = stop_strings(&req)?;
    // one request on the GPU at a time: image encoding and generation both take the turn
    let waiting = mgr.monitor.waiting();
    let turn = mgr.take_turn().await;
    drop(waiting);
    let mut prep = prep;
    // #25: preserve_thinking is the config's default; a request's chat_template_kwargs can still turn it off
    let preserve = loaded.preserve_thinking() && prep.kwargs.get("preserve_thinking") != Some(&Value::Bool(false));
    if preserve {
        prep.kwargs.entry("preserve_thinking").or_insert(Value::Bool(true));
        loaded.restore_reasoning(&mut prep.messages);
    }
    let prompt = loaded.prompts.build(&prep).map_err(|e| Refusal::bad(e.0))?;
    let mut ids = prompt.ids.clone();
    let sources = images_of(&prep.messages);
    let mut images = None;
    let mut temp: Option<PathBuf> = None;
    if !sources.is_empty() {
        let (Some(vision), Some(pad)) = (&loaded.vision, loaded.pad) else {
            return Err(Refusal::bad(format!("{} was started without its image encoder, so it cannot read images", loaded.entry.id)));
        };
        let mut encoded: Vec<(PathBuf, usize)> = Vec::new();
        for src in &sources {
            let data = media::load_source(src).await.map_err(|e| Refusal::bad(e.to_string()))?;
            let data = normalize_image(&data).map_err(|e| Refusal::bad(e.to_string()))?;
            let keep: Vec<PathBuf> = encoded.iter().map(|(p, _)| p.clone()).collect();
            encoded.push(vision.encode(&data, &keep).await.map_err(|e| Refusal::bad(e.to_string()))?);
        }
        let counts: Vec<usize> = encoded.iter().map(|(_, n)| *n).collect();
        ids = expand_images(&ids, pad, loaded.vision_start, &loaded.pad_literal, &counts).map_err(Refusal::bad)?;
        let files: Vec<PathBuf> = encoded.iter().map(|(p, _)| p.clone()).collect();
        let combined = vision.combine(&files).map_err(|e| Refusal(500, e.to_string()))?;
        let keys = files.iter().map(|p| p.file_stem().and_then(|s| s.to_str()).unwrap_or("").chars().take(16).collect()).collect();
        images = Some(Images { embeddings: combined.display().to_string(), pad, keys });
        temp = Some(combined);
    }
    let ctx = loaded.max_context;
    let room = ctx.saturating_sub(CTX_SLACK + ids.len());
    let asked = req.get("max_tokens").or_else(|| req.get("max_completion_tokens")).and_then(Value::as_i64).filter(|n| *n > 0);
    let max_new = match asked {
        None if room >= 1 => room,
        Some(n) if (n as usize) <= room => n as usize,
        Some(_) if loaded.entry.cfg.fit_max_tokens && room >= 1 => room,
        _ => {
            if let Some(t) = &temp {
                let _ = std::fs::remove_file(t);
            }
            return Err(Refusal::bad(match asked {
                Some(n) => format!("request exceeds the context window: prompt ({} tokens) + max tokens ({n}) exceeds the \
                                    context ({ctx}); requests are never truncated. Send a smaller max_tokens (at most {room} here)",
                                   ids.len()),
                None => format!("request exceeds the context window: prompt ({} tokens) leaves no room to answer in the \
                                 context ({ctx}); requests are never truncated", ids.len()),
            }));
        }
    };

    let thinking = prep.kwargs.get("enable_thinking") != Some(&Value::Bool(false));
    let defaults = loaded.defaults(thinking);
    let sampling = sampling_of(&req, &defaults);
    let answer_room = loaded.entry.cfg.rest.get("answer_room") != Some(&Value::Bool(false));
    let asked_budget = opts.budget.or_else(|| {
        req.get("reasoning_budget_tokens")
            .or_else(|| defaults.get("reasoning_budget_tokens"))
            .or_else(|| loaded.entry.cfg.rest.get("reasoning_budget_tokens"))
            .and_then(Value::as_u64)
    });
    let mut budget = if thinking { thinking_budget(asked_budget, max_new, answer_room) } else { None };
    let mut force = forced_call(opts.tool_choice.as_ref(), prep.tools.as_deref());
    let mut parser = if loaded.call_format == strata_frontend::formats::CallFormat::Gemma {
        let opened = !loaded.think_open_ids.is_empty() && ids.ends_with(&loaded.think_open_ids);
        OutputParser::with_format(thinking, prep.tools.as_deref(), opts.stream_tools, loaded.call_format, opened)
    } else {
        OutputParser::with_format(prompt.opens_reasoning, prep.tools.as_deref(), opts.stream_tools, loaded.call_format, false)
    };
    let repeat_stop = loaded.entry.cfg.rest.get("repeat_stop_tokens").and_then(Value::as_u64).unwrap_or(REPEAT_STOP_TOKENS as u64) as usize;
    let mut opening = Vec::new();
    if let (Some(f), false) = (&force, thinking) {
        // without thinking the prompt ends with the forced opening, read as if the model had written it
        ids.extend(loaded.tok.encode(f, true).map_err(|e| Refusal(500, e.to_string()))?);
        opening = parser.feed(f).map_err(|e| Refusal(500, e.0))?;
        force = None;
    }
    let turn_at = strata_protocol::turn_at(&ids, &loaded.turn_ids);
    let prompt_tokens = ids.len();
    let mut record = mgr.monitor.begin(prompt_tokens, max_new);
    let (tx, events) = mpsc::channel(256);
    let model = loaded.entry.id.clone();
    let lc = loaded.clone();
    tokio::spawn(async move {
        let _turn = turn; // held until this request ends
        let mut detok = Detokenizer::new();
        let mut cut = StopCutter { stops, held: String::new(), hit: None };
        let mut prompt_ids = ids;
        let (mut n, mut thought) = (0usize, 0usize);
        let (mut run_tok, mut run_len): (Option<u32>, usize) = (None, 0);
        let mut sampling = sampling;
        let orig_ids = prompt_ids.clone();
        let mut generated: Vec<u32> = Vec::new(); // every token of the reply so far, the wrap-ups included
        let (mut reasoning_text, mut answer_text, mut answered) = (String::new(), String::new(), false);
        let (mut next_loop_check, mut recovered, mut close_retried) = (LOOP_CHECK_EVERY, false, false);
        let loop_recovery = lc.loop_recovery();
        let close_retry = lc.reasoning_close_retry();
        let has_images = images.is_some();
        let mut tail = String::new(); // the last characters written (the newlines before a forced call)
        let mut last_done: Option<strata_protocol::Done> = None;
        // events through the stop strings to the client; false when the client has gone
        macro_rules! emit {
            ($evs:expr) => {{
                let mut alive = true;
                for e in $evs {
                    let out = match e {
                        Event::Content(t) if cut.hit.is_none() => {
                            let t = cut.push(&t);
                            if t.is_empty() { continue; }
                            Event::Content(t)
                        }
                        Event::Content(_) => continue,
                        other => other,
                    };
                    if let Event::Content(t) | Event::Reasoning(t) = &out {
                        tail = format!("{tail}{t}").chars().rev().take(2).collect::<Vec<_>>().into_iter().rev().collect();
                    }
                    match &out {
                        Event::Reasoning(t) => reasoning_text.push_str(t),
                        Event::Content(t) => {
                            answer_text.push_str(t);
                            answered = true;
                        }
                        Event::ToolStart { .. } | Event::ToolCall(_) => answered = true,
                        _ => {}
                    }
                    if tx.send(RunEvent::Event(out)).await.is_err() { alive = false; break; }
                }
                alive
            }};
        }
        if !emit!(opening) {
            return;
        }
        let summary = loop {
            let left = max_new.saturating_sub(n);
            let mut rx = lc.engine.generate(left as u32, sampling.clone(), turn_at, prompt_ids.clone(), images.clone()).await;
            let (mut seg, mut wrap, mut opens) = (Vec::new(), false, false);
            let (mut stopped, mut repeated, mut looped) = (false, false, false);
            let mut recover_prompt: Option<Vec<u32>> = None;
            let mut ended: Option<strata_protocol::Done> = None;
            while let Some(ev) = rx.recv().await {
                match ev {
                    Ok(GenEvent::Token(_)) if stopped => break, // the engine went on past its stop token: STOP it
                    Ok(GenEvent::Token(t)) => {
                        n += 1;
                        record.token(parser.in_reasoning());
                        if lc.stop_ids.contains(&t) {
                            // the end of the turn (<|im_end|>), not text
                            if force.is_some() && parser.holds_think_end() {
                                opens = true; // the held </think> was the end: the call opens there
                                break;
                            }
                            // its DONE (the timings, the reused prefix) normally follows at once
                            stopped = true;
                            continue;
                        }
                        run_len = if Some(t) == run_tok { run_len + 1 } else { 1 };
                        run_tok = Some(t);
                        if repeat_stop > 0 && run_len >= repeat_stop {
                            repeated = true; // #606: a model in a loop, not an answer: end it here
                            break;
                        }
                        seg.push(t);
                        generated.push(t);
                        let text = lc.tok.token_bytes(t).map(|b| detok.push(&b)).unwrap_or_default();
                        let evs = match parser.feed(&text) {
                            Ok(e) => e,
                            Err(e) => {
                                record.fail();
                                let _ = tx.send(RunEvent::Error(e.0)).await;
                                return;
                            }
                        };
                        let writes_call = evs.iter().any(|e| matches!(e, Event::ToolStart { .. } | Event::ToolCall(_)));
                        if !emit!(evs) {
                            return; // the client left: dropping rx makes the engine STOP
                        }
                        if cut.hit.is_some() {
                            break;
                        }
                        if let Some(recover) = loop_recovery {
                            if !recovered && parser.in_reasoning() && n >= next_loop_check && parser.is_clean() && !detok.has_pending() {
                                next_loop_check = n + LOOP_CHECK_EVERY;
                                let cov = reasoning_repeat_coverage(&reasoning_text);
                                if cov >= LOOP_COVERAGE {
                                    if !recover {
                                        tracing::warn!("the thinking repeated the same passages (coverage={cov:.3}) at {n} tokens: \
                                                        ended as \"length\" (reasoning_loop_recovery: \"stop\")");
                                        looped = true;
                                        break;
                                    }
                                    if !has_images {
                                        recover_prompt = focused_recovery_prompt(&lc.tok, &orig_ids, &generated);
                                        if recover_prompt.is_some() {
                                            tracing::warn!("repeated reasoning detected at {n} tokens (coverage={cov:.3}); resuming \
                                                            the same output with the low-effort instruction");
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(b) = budget {
                            if parser.in_reasoning() {
                                thought += 1;
                                if thought >= b && parser.is_clean() && !detok.has_pending() {
                                    wrap = true;
                                    break;
                                }
                            }
                        }
                        if force.is_some() {
                            if writes_call {
                                force = None; // the model is writing a call of its own
                            } else if parser.at_clean_content() && !detok.has_pending() {
                                opens = true;
                                break;
                            }
                        }
                    }
                    Ok(GenEvent::Progress { done, total }) => {
                        record.progress(done, total);
                        if tx.send(RunEvent::Ping).await.is_err() {
                            return;
                        }
                    }
                    Ok(GenEvent::Done(d)) => {
                        ended = Some(d);
                        break;
                    }
                    Err(e) => {
                        record.fail();
                        let _ = tx.send(RunEvent::Error(e.to_string())).await;
                        return;
                    }
                }
            }
            drop(rx); // a pass ended early: the engine is told to STOP and drained before the next one
            if let Some(d) = &ended {
                last_done = Some(d.clone());
            }
            if let Some(p) = recover_prompt.take().filter(|_| n < max_new) {
                // #728: once, from the same output with the low-effort sentence; only temperature and presence penalty
                // are raised (to at least 1.0 and 1.5)
                recovered = true;
                sampling.temperature = Some(sampling.temperature.unwrap_or(0.0).max(1.0));
                sampling.presence_penalty = Some(sampling.presence_penalty.unwrap_or(0.0).max(1.5));
                prompt_ids = p;
                continue;
            }
            if cut.hit.is_some() {
                let mut evs = parser.finish(Some("stop"));
                evs.retain(|e| !matches!(e, Event::Content(_)));
                let _ = emit!(evs);
                break ("stop", cut.hit.clone());
            }
            let natural = !repeated && !looped && (stopped || ended.as_ref().is_none_or(|d| d.finish == "stop"));
            if close_retry && thinking && natural && !close_retried && !answered && !wrap && !opens
                && parser.in_reasoning() && !parser.holds_think_end()
            {
                // #1053: the model wrote its reasoning and stopped before </think>: close it once and let it answer
                close_retried = true;
                let extra = lc.tok.encode(REASONING_CLOSE, true).unwrap_or_default();
                if max_new.saturating_sub(n + extra.len()) >= 1 {
                    tracing::info!("the reply ended inside its thinking with no answer: closing the thinking once and continuing");
                    for t in &extra {
                        n += 1;
                        generated.push(*t);
                        let piece = lc.tok.token_bytes(*t).map(|b| detok.push(&b)).unwrap_or_default();
                        match parser.feed(&piece) {
                            Ok(evs) => {
                                if !emit!(evs) {
                                    return;
                                }
                            }
                            Err(e) => {
                                record.fail();
                                let _ = tx.send(RunEvent::Error(e.0)).await;
                                return;
                            }
                        }
                    }
                    prompt_ids.extend(seg);
                    prompt_ids.extend(extra);
                    continue;
                }
            }
            if !(wrap || opens) {
                let finish = match ended.as_ref().map(|d| d.finish.as_str()) {
                    _ if repeated || looped => "length",
                    _ if stopped => "stop",
                    Some("length") => "length",
                    Some("cancel") => "cancel",
                    _ => "stop",
                };
                if !emit!(parser.finish(Some(finish))) {
                    return;
                }
                if cut.hit.is_none() {
                    let rest = cut.finish();
                    if !rest.is_empty() {
                        let _ = tx.send(RunEvent::Event(Event::Content(rest))).await;
                    }
                }
                break (if cut.hit.is_some() { "stop" } else { finish }, cut.hit.clone());
            }
            // the thinking reached its budget, or a forced call opens: write the wrap-up / the call's opening and go on
            // from the prefix the engine already holds
            let text = if wrap {
                budget = None;
                format!("{REASONING_WRAP_UP}{}", force.take().unwrap_or_default())
            } else {
                let have = tail.len() - tail.trim_end_matches('\n').len();
                format!("{}{}", "\n".repeat(2usize.saturating_sub(have)), force.take().unwrap_or_default())
            };
            let extra = lc.tok.encode(&text, true).unwrap_or_default();
            if max_new < n + extra.len() + 1 {
                if !emit!(parser.finish(Some("length"))) {
                    return;
                }
                break ("length", None);
            }
            for t in &extra {
                n += 1;
                generated.push(*t);
                let piece = lc.tok.token_bytes(*t).map(|b| detok.push(&b)).unwrap_or_default();
                match parser.feed(&piece) {
                    Ok(evs) => {
                        if !emit!(evs) {
                            return;
                        }
                    }
                    Err(e) => {
                        record.fail();
                        let _ = tx.send(RunEvent::Error(e.0)).await;
                        return;
                    }
                }
            }
            prompt_ids.extend(seg);
            prompt_ids.extend(extra);
        };
        let (finish, stop) = summary;
        if preserve {
            lc.remember_reasoning(&answer_text, &reasoning_text);
        }
        let t = last_done.as_ref().map(timings).unwrap_or(Value::Null);
        record.finish(finish, n as u64, last_done.as_ref(), t.clone());
        let _ = tx
            .send(RunEvent::Done(Summary {
                finish: finish.to_string(),
                completion_tokens: n as u64,
                reused: last_done.as_ref().and_then(|d| d.reused).unwrap_or(0),
                stop_sequence: stop,
                timings: t,
            }))
            .await;
        if let Some(t) = temp {
            let _ = std::fs::remove_file(t);
        }
    });
    Ok(Started { events, prompt_tokens, model })
}

/// `start` with the MCP tools run here: the model writes a call to an MCP tool, the server runs it, adds the call
/// and its result to the conversation and lets the model continue, up to `max_rounds` times. `names`: the MCP tools
/// this request offered; any other call is the request's own and ends the turn as always (the client answers it),
/// and MCP calls written beside it are then not run. Port of serve/server.py run_with_mcp.
pub async fn start_mcp(mgr: Arc<Manager>, loaded: Arc<Loaded>, prep: Prepared, req: Value, opts: RunOpts,
                       hub: Arc<crate::mcp::McpHub>, names: std::collections::HashSet<String>) -> Result<Started, Refusal> {
    let first = start(mgr.clone(), loaded.clone(), prep.clone(), req.clone(), opts.clone()).await?;
    let (model, prompt_tokens) = (first.model.clone(), first.prompt_tokens);
    let (tx, events) = mpsc::channel(256);
    tokio::spawn(async move {
        let (mut cur, mut prep, mut total, mut rounds) = (first, prep, 0u64, 0usize);
        loop {
            let (mut text, mut reasoning, mut calls, mut own) = (String::new(), String::new(), Vec::new(), 0usize);
            let mut mcp_ids = std::collections::HashSet::new();
            let mut done = None;
            while let Some(ev) = cur.events.recv().await {
                let out = match ev {
                    RunEvent::Event(Event::ToolStart { id, name }) if names.contains(&name) => {
                        mcp_ids.insert(id.clone());
                        RunEvent::Mcp(json!({"event": "start", "id": id, "name": name}))
                    }
                    RunEvent::Event(Event::ToolArgs { id, .. }) if mcp_ids.contains(&id) => continue,
                    RunEvent::Event(Event::ToolCall(c)) if names.contains(&c.name) => {
                        calls.push(c);
                        continue;
                    }
                    RunEvent::Done(s) => {
                        done = Some(s);
                        break;
                    }
                    RunEvent::Event(e) => {
                        match &e {
                            Event::ToolCall(_) => own += 1,
                            Event::Content(t) => text.push_str(t),
                            Event::Reasoning(t) => reasoning.push_str(t),
                            _ => {}
                        }
                        RunEvent::Event(e)
                    }
                    other => other,
                };
                let is_error = matches!(out, RunEvent::Error(_));
                if tx.send(out).await.is_err() || is_error {
                    return;
                }
            }
            let Some(mut sum) = done else { return };
            total += sum.completion_tokens;
            let mut run_them = !calls.is_empty() && own == 0 && sum.finish == "stop";
            if run_them && rounds >= hub.settings.max_rounds {
                let _ = tx.send(RunEvent::Mcp(json!({"event": "limit", "max_rounds": hub.settings.max_rounds}))).await;
                run_them = false;
            }
            if !run_them {
                for c in &calls {
                    // announced, never run: closed in the client's view
                    let _ = tx.send(RunEvent::Mcp(json!({"event": "result", "id": c.id, "ok": false, "skipped": true,
                        "text": "not run", "chars": 0, "truncated": false, "ms": 0}))).await;
                }
                sum.completion_tokens = total;
                let _ = tx.send(RunEvent::Done(sum)).await;
                return;
            }
            rounds += 1;
            let mut results = Vec::new();
            for c in &calls {
                let route = hub.routes().into_iter().find(|(n, _, _)| *n == c.name);
                let _ = tx.send(RunEvent::Mcp(json!({"event": "call", "id": c.id, "name": c.name,
                    "server": route.as_ref().map(|(_, s, _)| s.name.clone()),
                    "tool": route.as_ref().map(|(_, _, t)| t["name"].clone()).unwrap_or(json!(c.name)),
                    "arguments": c.arguments, "round": rounds}))).await;
                let args = Value::Object(c.arguments.clone());
                // the call runs while the client is watched: one that leaves (the web app's Stop) ends it
                let r = tokio::select! {
                    r = hub.call(&c.name, &args) => r,
                    _ = tx.closed() => return,
                };
                tracing::info!("tool {}: {}, {} characters in {:.1} s{}", c.name, if r["ok"] == true { "ok" } else { "error" },
                               r["chars"], r["ms"].as_f64().unwrap_or(0.0) / 1000.0,
                               if r["truncated"] == true { " (truncated for the model)" } else { "" });
                results.push(r["text"].as_str().unwrap_or("").to_string());
                let mut ev = json!({"event": "result", "id": c.id});
                for k in ["ok", "text", "chars", "truncated", "ms"] {
                    ev[k] = r[k].clone();
                }
                if tx.send(RunEvent::Mcp(ev)).await.is_err() {
                    return;
                }
            }
            let mut msg = json!({"role": "assistant", "content": text.trim(),
                "tool_calls": calls.iter().map(|c| json!({"function": {"name": c.name, "arguments": c.arguments}})).collect::<Vec<_>>()});
            if !reasoning.is_empty() {
                msg["reasoning_content"] = json!(reasoning.trim());
            }
            prep.messages.push(msg);
            prep.messages.extend(results.into_iter().map(|r| json!({"role": "tool", "content": r})));
            cur = match start(mgr.clone(), loaded.clone(), prep.clone(), req.clone(), opts.clone()).await {
                Ok(s) => s,
                Err(Refusal(_, m)) => {
                    let _ = tx.send(RunEvent::Error(m)).await;
                    return;
                }
            };
        }
    });
    Ok(Started { events, prompt_tokens, model })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_leaves_the_answer_room() {
        // no budget asked: from 1024 max_tokens on, the answer keeps a quarter (at least 512, at most half)
        assert_eq!(thinking_budget(None, 8192, true), Some(6144));
        assert_eq!(thinking_budget(None, 1024, true), Some(512));
        assert_eq!(thinking_budget(None, 1000, true), None);
        assert_eq!(thinking_budget(None, 8192, false), None, "answer_room off");
        // a budget of its own is kept unless it leaves no room
        assert_eq!(thinking_budget(Some(100), 8192, true), Some(100));
        assert_eq!(thinking_budget(Some(8000), 8192, true), Some(6144));
        assert_eq!(thinking_budget(Some(0), 500, true), None);
    }
}

#[cfg(test)]
mod loop_tests {
    use super::reasoning_repeat_coverage;

    #[test]
    fn repeated_passages_and_periodic_tails_are_loops() {
        assert_eq!(reasoning_repeat_coverage("short text"), 0.0);
        let looping = "check the sum again to be sure it is right now ok ".repeat(300);
        assert!(reasoning_repeat_coverage(&looping) > 0.9);
        assert_eq!(reasoning_repeat_coverage(&"12345678901".repeat(300)), 1.0, "#1753: one long periodic word");
        let varied: String = (0..3000).map(|i| format!("w{i} ")).collect();
        assert_eq!(reasoning_repeat_coverage(&varied), 0.0);
    }
}
