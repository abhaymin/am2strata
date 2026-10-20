//! One request on the loaded model: prompt, pictures, sampling, generation, output parsing and stop strings.
//! Both APIs (OpenAI and Anthropic) format the events this produces.

use crate::manager::{Loaded, Manager};
use crate::media;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use strata_engine::vision::normalize_image;
use strata_engine::GenEvent;
use strata_frontend::images::{expand_images, images_of};
use strata_frontend::output::{Event, OutputParser};
use strata_frontend::Prepared;
use strata_protocol::{Images, Sampling};
use strata_tokenizer::Detokenizer;
use tokio::sync::mpsc;

/// The engine keeps a few tokens of slack at the end of the context (serve/server.py CTX_SLACK).
const CTX_SLACK: usize = 1;

pub enum RunEvent {
    /// the prompt is being read: nothing to say yet (a keep-alive)
    Ping,
    Event(Event),
    Done(Summary),
    Error(String),
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

/// The model card's sampling ("Best Practices"), used when the request does not say (PolyStrata 0.1.45).
fn model_card_sampling(family: Option<&str>, thinking: bool) -> Sampling {
    let (t, p, pp) = match (family, thinking) {
        (Some("qwen4exp"), true) => (1.0, 0.95, None),
        (Some("qwen4exp"), false) => (0.7, 0.8, Some(1.5)),
        (Some("qwen35moe"), true) => (1.0, 0.95, Some(1.5)),
        (Some("qwen35moe"), false) => (0.7, 0.8, Some(1.5)),
        _ => return Sampling::default(),
    };
    Sampling { temperature: Some(t), top_p: Some(p), top_k: Some(20), presence_penalty: pp, ..Default::default() }
}

/// The request's sampling over the model card's: a field the request sets wins; temperature 0 asks for greedy.
pub fn sampling_of(req: &Value, family: Option<&str>, thinking: bool) -> Sampling {
    let f = |k: &str| req.get(k).and_then(Value::as_f64);
    let i = |k: &str| req.get(k).and_then(Value::as_i64);
    let card = model_card_sampling(family, thinking);
    let any_given = ["temperature", "top_p", "top_k", "min_p"].iter().any(|k| req.get(*k).is_some_and(|v| !v.is_null()));
    let base = if any_given { Sampling::default() } else { card.clone() };
    Sampling {
        temperature: f("temperature").or(base.temperature),
        top_p: f("top_p").or(base.top_p),
        top_k: i("top_k").or(base.top_k),
        min_p: f("min_p").or(base.min_p),
        repetition_penalty: f("repetition_penalty").or(base.repetition_penalty),
        frequency_penalty: f("frequency_penalty").or(base.frequency_penalty),
        presence_penalty: f("presence_penalty").or(if any_given { None } else { card.presence_penalty }),
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

/// Start a request. `prep`: the normalized messages, tools and template kwargs; `req`: the raw request (sampling,
/// max tokens, stop strings); `stream_tools`: report tool calls while they are being written.
pub async fn start(mgr: Arc<Manager>, loaded: Arc<Loaded>, prep: Prepared, req: Value, stream_tools: bool)
    -> Result<Started, Refusal> {
    let stops = stop_strings(&req)?;
    // one request on the GPU at a time: image encoding and generation both take the turn
    let turn = mgr.take_turn().await;
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
    let sampling = sampling_of(&req, loaded.family.as_deref(), thinking);
    let turn_at = strata_protocol::turn_at(&ids, &loaded.turn_ids);
    let prompt_tokens = ids.len();
    let mut rx = loaded.engine.generate(max_new as u32, sampling, turn_at, ids, images).await;
    let (tx, events) = mpsc::channel(256);
    let tools = prep.tools.clone();
    let opens = prompt.opens_reasoning;
    let model = loaded.entry.id.clone();
    let lc = loaded.clone();
    tokio::spawn(async move {
        let _turn = turn; // held until this request ends
        let mut parser = OutputParser::new(opens, tools.as_deref(), stream_tools);
        let mut detok = Detokenizer::new();
        let mut cut = StopCutter { stops, held: String::new(), hit: None };
        let mut generated = 0u64;
        let send_events = |evs: Vec<Event>, cut: &mut StopCutter| -> Vec<RunEvent> {
            let mut out = Vec::new();
            for e in evs {
                match e {
                    Event::Content(t) if cut.hit.is_none() => {
                        let t = cut.push(&t);
                        if !t.is_empty() {
                            out.push(RunEvent::Event(Event::Content(t)));
                        }
                    }
                    Event::Content(_) => {}
                    other => out.push(RunEvent::Event(other)),
                }
            }
            out
        };
        let finish_with = |done: Option<strata_protocol::Done>, finish: &str, stop: Option<String>, generated: u64| Summary {
            finish: finish.to_string(),
            completion_tokens: done.as_ref().map_or(generated, |d| d.generated),
            reused: done.as_ref().and_then(|d| d.reused).unwrap_or(0),
            stop_sequence: stop,
            timings: done.as_ref().map(timings).unwrap_or(Value::Null),
        };
        loop {
            match rx.recv().await {
                Some(Ok(GenEvent::Token(t))) => {
                    generated += 1;
                    if lc.tok.token_type(t) == 3 {
                        continue; // CONTROL tokens (<|im_end|>) are not text
                    }
                    let text = lc.tok.token_bytes(t).map(|b| detok.push(&b)).unwrap_or_default();
                    match parser.feed(&text) {
                        Ok(evs) => {
                            for e in send_events(evs, &mut cut) {
                                if tx.send(e).await.is_err() {
                                    return; // the client left: dropping rx makes the engine STOP
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(RunEvent::Error(e.0)).await;
                            return;
                        }
                    }
                    if cut.hit.is_some() {
                        // a stop string: stop the engine (dropping rx sends STOP) and finish here
                        drop(rx);
                        let mut tail = parser.finish(Some("stop"));
                        tail.retain(|e| !matches!(e, Event::Content(_)));
                        for e in tail {
                            let _ = tx.send(RunEvent::Event(e)).await;
                        }
                        let stop = cut.hit.clone();
                        let _ = tx.send(RunEvent::Done(finish_with(None, "stop", stop, generated))).await;
                        break;
                    }
                }
                Some(Ok(GenEvent::Progress { .. })) => {
                    if tx.send(RunEvent::Ping).await.is_err() {
                        return;
                    }
                }
                Some(Ok(GenEvent::Done(d))) => {
                    let finish = match d.finish.as_str() {
                        "length" => "length",
                        "stop" => "cancel",
                        _ => "stop",
                    };
                    // the parser's held text first goes through the stop strings, then what they held back
                    for e in send_events(parser.finish(Some(finish)), &mut cut) {
                        let _ = tx.send(e).await;
                    }
                    if cut.hit.is_none() {
                        let tail = cut.finish();
                        if !tail.is_empty() {
                            let _ = tx.send(RunEvent::Event(Event::Content(tail))).await;
                        }
                    }
                    let stop = cut.hit.clone();
                    let finish = if stop.is_some() { "stop" } else { finish };
                    let _ = tx.send(RunEvent::Done(finish_with(Some(d), finish, stop, generated))).await;
                    break;
                }
                Some(Err(e)) => {
                    let _ = tx.send(RunEvent::Error(e.to_string())).await;
                    break;
                }
                None => break,
            }
        }
        if let Some(t) = temp {
            let _ = std::fs::remove_file(t);
        }
    });
    Ok(Started { events, prompt_tokens, model })
}
