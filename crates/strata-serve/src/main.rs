//! strata-serve: an OpenAI-compatible HTTP server for one PolyStrata engine.
//!
//! Phase 1 slice: /health, /v1/models and /v1/chat/completions (streaming and not), driving the existing C++
//! engines over their line protocol. The prompt text and token ids match PolyStrata's Python server exactly
//! (checked against golden files recorded from it).

mod output;

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use output::OutputParser;
use serde_json::{json, Map, Value};
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use strata_engine::{Engine, GenEvent, RunConfig};
use strata_protocol::Sampling;
use strata_template::ChatTemplate;
use strata_tokenizer::{Detokenizer, Tokenizer};

#[derive(Parser)]
#[command(about = "OpenAI-compatible server for a PolyStrata engine")]
struct Args {
    /// PolyStrata run config (strata-<model>.json)
    #[arg(long)]
    config: PathBuf,
    /// port (default: the config's)
    #[arg(long)]
    port: Option<u16>,
    /// address to listen on (default: the config's host, else 127.0.0.1)
    #[arg(long)]
    host: Option<String>,
    /// seconds to wait for the engine to load
    #[arg(long, default_value_t = 900)]
    ready_timeout: u64,
}

struct AppState {
    cfg: RunConfig,
    tok: Tokenizer,
    template: ChatTemplate,
    engine: Engine,
    turn_ids: Vec<u32>,
    max_context: usize,
}

type Shared = Arc<AppState>;

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn error(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    (status, Json(json!({"error": {"type": kind, "message": message.into()}}))).into_response()
}

fn authorized(state: &AppState, headers: &HeaderMap) -> bool {
    let Some(key) = state.cfg.api_key.as_deref().filter(|k| !k.is_empty()) else { return true };
    let bearer = headers.get("authorization").and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer "));
    let x_key = headers.get("x-api-key").and_then(|v| v.to_str().ok());
    bearer == Some(key) || x_key == Some(key)
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn models(State(s): State<Shared>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    Json(json!({"object": "list", "data": [{
        "id": s.cfg.model_name, "object": "model", "created": now(), "owned_by": "polystrata",
        "meta": {"n_ctx": s.max_context}
    }]}))
    .into_response()
}

fn sampling_of(req: &Value) -> Sampling {
    let f = |k: &str| req.get(k).and_then(Value::as_f64);
    let i = |k: &str| req.get(k).and_then(Value::as_i64);
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

/// The rendered prompt left a `<think>` open: the reply starts inside the reasoning.
fn opens_reasoning(prompt: &str) -> bool {
    match (prompt.rfind("<think>"), prompt.rfind("</think>")) {
        (Some(o), Some(c)) => o > c,
        (Some(_), None) => true,
        _ => false,
    }
}

fn finish_reason(engine: &str) -> &'static str {
    match engine {
        "length" => "length",
        _ => "stop",
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

async fn chat(State(s): State<Shared>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let Some(messages) = req.get("messages").and_then(Value::as_array) else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "messages is required");
    };
    let tools = req.get("tools").filter(|t| t.as_array().is_some_and(|a| !a.is_empty()));
    let mut extra: Map<String, Value> =
        req.get("chat_template_kwargs").and_then(Value::as_object).cloned().unwrap_or_default();
    if let Some(e) = req.get("reasoning_effort").cloned() {
        extra.entry("reasoning_effort").or_insert(e);
    }
    let prompt = match s.template.render(messages, tools, true, &extra) {
        Ok(p) => p,
        Err(strata_template::Error::Request(m)) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", m),
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string()),
    };
    let ids = match s.tok.encode(&prompt, true) {
        Ok(ids) => ids,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string()),
    };
    // requests are never truncated; max_tokens is clamped to the room left only with fit_max_tokens
    let room = s.max_context.saturating_sub(ids.len() + 1);
    let asked = req.get("max_tokens").or_else(|| req.get("max_completion_tokens")).and_then(Value::as_u64);
    let max_new = match asked {
        None => room,
        Some(n) if (n as usize) <= room => n as usize,
        Some(_) if s.cfg.fit_max_tokens => room,
        Some(n) => {
            return error(StatusCode::BAD_REQUEST, "invalid_request_error", format!(
                "prompt ({} tokens) + max tokens ({n}) exceeds the context ({}); requests are never truncated",
                ids.len(), s.max_context))
        }
    };
    if max_new == 0 {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error",
                     format!("prompt ({} tokens) leaves no room to answer in the context ({})", ids.len(), s.max_context));
    }
    let turn_at = strata_protocol::turn_at(&ids, &s.turn_ids);
    let prompt_tokens = ids.len();
    let in_reasoning = opens_reasoning(&prompt);
    let rx = s.engine.generate(max_new as u32, sampling_of(&req), turn_at, ids).await;
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    if req.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        stream_reply(s, rx, id, in_reasoning).await
    } else {
        whole_reply(s, rx, id, in_reasoning, prompt_tokens).await
    }
}

/// Text of one generated token, CONTROL tokens (`<|im_end|>`) left out.
fn token_text(s: &AppState, d: &mut Detokenizer, t: u32) -> String {
    if s.tok.token_type(t) == 3 {
        return String::new();
    }
    s.tok.token_bytes(t).map(|b| d.push(&b)).unwrap_or_default()
}

async fn whole_reply(s: Shared, mut rx: tokio::sync::mpsc::Receiver<strata_engine::Result<GenEvent>>, id: String,
                     in_reasoning: bool, prompt_tokens: usize) -> Response {
    let mut detok = Detokenizer::new();
    let mut parser = OutputParser::new(in_reasoning);
    let (mut reasoning, mut content) = (String::new(), String::new());
    let mut done = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            Ok(GenEvent::Token(t)) => {
                let d = parser.push(&token_text(&s, &mut detok, t));
                reasoning += &d.reasoning;
                content += &d.content;
            }
            Ok(GenEvent::Progress { .. }) => {}
            Ok(GenEvent::Done(d)) => done = Some(d),
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string()),
        }
    }
    let d = parser.finish();
    reasoning += &d.reasoning;
    content += &d.content;
    let Some(done) = done else {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "the engine ended the request without DONE");
    };
    let mut message = json!({"role": "assistant", "content": content});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    Json(json!({
        "id": id, "object": "chat.completion", "created": now(), "model": s.cfg.model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(&done.finish)}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": done.generated,
                  "total_tokens": prompt_tokens as u64 + done.generated},
        "timings": timings(&done),
    }))
    .into_response()
}

async fn stream_reply(s: Shared, mut rx: tokio::sync::mpsc::Receiver<strata_engine::Result<GenEvent>>, id: String,
                      in_reasoning: bool) -> Response {
    let created = now();
    let model = s.cfg.model_name.clone();
    let chunk = move |delta: Value, finish: Option<&str>, extra: Option<Value>| {
        let mut c = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        if let Some(x) = extra {
            c["timings"] = x;
        }
        Ok::<_, Infallible>(SseEvent::default().data(c.to_string()))
    };
    let stream = async_stream(move |tx| async move {
        let mut detok = Detokenizer::new();
        let mut parser = OutputParser::new(in_reasoning);
        let _ = tx.send(chunk(json!({"role": "assistant", "content": ""}), None, None)).await;
        let send_delta = |d: output::Delta| {
            let mut v = Map::new();
            if !d.reasoning.is_empty() {
                v.insert("reasoning_content".into(), json!(d.reasoning));
            }
            if !d.content.is_empty() {
                v.insert("content".into(), json!(d.content));
            }
            Value::Object(v)
        };
        while let Some(ev) = rx.recv().await {
            match ev {
                Ok(GenEvent::Token(t)) => {
                    let d = parser.push(&token_text(&s, &mut detok, t));
                    if !d.is_empty() && tx.send(chunk(send_delta(d), None, None)).await.is_err() {
                        return; // client gone: dropping rx makes the engine STOP
                    }
                }
                Ok(GenEvent::Progress { .. }) => {}
                Ok(GenEvent::Done(done)) => {
                    let d = parser.finish();
                    if !d.is_empty() {
                        let _ = tx.send(chunk(send_delta(d), None, None)).await;
                    }
                    let _ = tx.send(chunk(json!({}), Some(finish_reason(&done.finish)), Some(timings(&done)))).await;
                    let _ = tx.send(Ok(SseEvent::default().data("[DONE]"))).await;
                    return;
                }
                Err(e) => {
                    let _ = tx.send(Ok(SseEvent::default().data(json!({"error": {"message": e.to_string()}}).to_string()))).await;
                    return;
                }
            }
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10))).into_response()
}

/// A stream fed by a task: the task gets the sender, the response gets the receiver.
fn async_stream<F, Fut>(f: F) -> tokio_stream::wrappers::ReceiverStream<Result<SseEvent, Infallible>>
where
    F: FnOnce(tokio::sync::mpsc::Sender<Result<SseEvent, Infallible>>) -> Fut,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    tokio::spawn(f(tx));
    tokio_stream::wrappers::ReceiverStream::new(rx)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter(
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into())).init();
    let args = Args::parse();
    let cfg = RunConfig::load(&args.config).context("run config")?;
    let started = Instant::now();
    let tok = Tokenizer::from_dir(&cfg.tokenizer).context("tokenizer")?;
    let template = ChatTemplate::from_file(cfg.tokenizer.join("chat_template.jinja")).context("chat template")?;
    let turn_ids: Vec<u32> = tok.token_id("<|im_start|>").into_iter().collect();
    tracing::info!(model = %cfg.model_name, vocab = tok.vocab_size(), "tokenizer and template loaded in {:?}", started.elapsed());
    let max_context = cfg.max_context();
    let engine = Engine::start(&cfg, Duration::from_secs(args.ready_timeout)).await.context("engine")?;
    tracing::info!("engine ready after {:?}", started.elapsed());
    let host = args.host.or_else(|| cfg.host.clone()).unwrap_or_else(|| "127.0.0.1".into());
    let port = args.port.or(cfg.port).unwrap_or(8080);
    let state = Arc::new(AppState { cfg, tok, template, engine, turn_ids, max_context });
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await.with_context(|| format!("bind {host}:{port}"))?;
    tracing::info!("listening on http://{host}:{port}/v1");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    state.engine.shutdown().await;
    Ok(())
}
