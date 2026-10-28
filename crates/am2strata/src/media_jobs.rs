//! Media from a normal chat: the model gets `generate_image`, `generate_video` and `speak` as built-in tools (run here
//! in the tool loop, as the memory tools are) when the chat opts in with `"strata_media": true` and the capability
//! check passes for that kind. A call does not block the answer: it queues a media job and returns its id at once;
//! the job waits its place in the queue (behind the answer, so the text model is not stopped in the middle of it) and
//! runs after. GET /api/jobs/{id} (or /v1/media/jobs/{id}) follows it; /v1/media/jobs/{id}/content is the file.

use crate::{authorized, error, AppState};
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

pub const TOOL_NAMES: [&str; 3] = ["generate_image", "generate_video", "speak"];

/// Jobs kept (the oldest finished ones go first).
const KEEP: usize = 64;

struct Job {
    id: String,
    kind: &'static str,
    model: String,
    prompt: String,
    created: u64,
    /// "queued" until the job's request returns, then "done" or "error"
    state: &'static str,
    error: Option<String>,
    mime: Option<String>,
    data: Option<Vec<u8>>,
    meta: Value,
    seconds: Option<f64>,
}

pub struct Jobs {
    app: OnceLock<AppState>,
    jobs: Mutex<BTreeMap<u64, Job>>,
    next: Mutex<u64>,
    /// the capability check, kept for a minute (and used as it is while the media engine runs a job)
    caps: Mutex<Option<(Instant, Value)>>,
    /// seconds the last finished job of each model took, for the estimates
    speeds: Mutex<HashMap<String, f64>>,
}

impl Jobs {
    pub fn new() -> Jobs {
        Jobs { app: OnceLock::new(), jobs: Mutex::new(BTreeMap::new()), next: Mutex::new(1), caps: Mutex::new(None),
               speeds: Mutex::new(HashMap::new()) }
    }

    /// The server the jobs run on (set once it is built).
    pub fn bind(&self, s: AppState) {
        let _ = self.app.set(s);
    }

    fn app(&self) -> &AppState {
        self.app.get().expect("media jobs bound to the server")
    }

    /// The tools for the kinds the capability check passed.
    pub fn tools(caps: &Value) -> Vec<Value> {
        let ok = |k: &str| caps["kinds"][k]["usable"] == true;
        let mut t = Vec::new();
        if ok("image") {
            let sizes = caps["kinds"]["image"]["sizes"].clone();
            t.push(json!({"name": "generate_image", "description": format!(
                "Draw an image on this PC from a text description. It is queued and appears in the chat after your answer: \
                 say that it is on its way, do not describe it as done. Sizes: {sizes}."),
                "parameters": {"type": "object", "properties": {
                    "prompt": {"type": "string", "description": "a detailed description of the image"},
                    "size": {"type": "string", "description": "WIDTHxHEIGHT, default 1024x1024"}}, "required": ["prompt"]}}));
        }
        if ok("video") {
            let v = &caps["kinds"]["video"];
            t.push(json!({"name": "generate_video", "description": format!(
                "Make a short video clip on this PC from a text description. It is queued and appears in the chat after \
                 your answer (it takes minutes): say that it is on its way. Sizes: {}; frames up to {}.", v["sizes"], v["max_frames"]),
                "parameters": {"type": "object", "properties": {
                    "prompt": {"type": "string", "description": "a detailed description of the scene and its motion"},
                    "frames": {"type": "integer", "description": "number of frames (default the model's)"},
                    "size": {"type": "string", "description": "WIDTHxHEIGHT"}}, "required": ["prompt"]}}));
        }
        if ok("tts") {
            t.push(json!({"name": "speak", "description":
                "Read a text aloud on this PC (text to speech, a WAV file). It is queued and appears in the chat as an \
                 audio player after your answer.",
                "parameters": {"type": "object", "properties": {
                    "text": {"type": "string", "description": "what to say, at most 4096 characters"},
                    "voice": {"type": "string", "description": "a voice name (laura, jon, lea...) or a sentence describing the voice"}},
                    "required": ["text"]}}));
        }
        t
    }

    /// GET /api/media/capabilities: per kind, the model in use, whether it is downloaded and fits this PC, its sizes
    /// and limits, and the time the last job of it took.
    pub async fn capabilities(&self) -> Value {
        if let Some((at, v)) = self.caps.lock().unwrap().clone() {
            if at.elapsed() < Duration::from_secs(60) || self.app().media.busy() {
                return self.with_speeds(v);
            }
        }
        let s = self.app();
        if !s.media.available() {
            return json!({"available": false, "kinds": {}, "note": "the media engine is not built"});
        }
        let list = match s.media.call(json!({"op": "list"}), |_| {}).await {
            Ok(l) => l,
            Err(e) => return json!({"available": false, "kinds": {}, "note": format!("{e:#}")}),
        };
        let models = list["models"].as_array().cloned().unwrap_or_default();
        let find = |name: &str| models.iter().find(|m| m["name"] == name).cloned();
        // the text model is stopped for a GPU job, so the whole VRAM is the media model's; mold offloads past it
        let vram_gb: f64 = s.mgr.hw.gpus.iter().map(|g| g.vram_mib as f64 / 1024.0).sum();
        let mut kinds = serde_json::Map::new();
        for (kind, model) in [("image", s.media.model()), ("video", s.media.video_model()), ("tts", s.media.tts_model()),
                              ("stt", s.media.stt_model())] {
            let m = find(&model);
            let size = m.as_ref().and_then(|m| m["size_gb"].as_f64());
            let downloaded = m.as_ref().is_some_and(|m| m["downloaded"] == true);
            // the speech models run on the CPU when the RAM has room, else as a GPU job: they fit either way
            let fits = match kind {
                "tts" | "stt" => true,
                _ => size.is_none_or(|g| vram_gb == 0.0 || g <= vram_gb * 1.5),
            };
            let mut k = json!({"model": model, "downloaded": downloaded, "size_gb": size, "fits": fits,
                               "usable": m.is_some() && downloaded && fits});
            let d = m.as_ref().map(|m| m["defaults"].clone()).unwrap_or(Value::Null);
            match kind {
                "image" => k["sizes"] = json!(["1024x1024", "1024x768", "768x1024", "512x512"]),
                "video" => {
                    let (w, h) = (d["width"].as_u64().unwrap_or(832), d["height"].as_u64().unwrap_or(480));
                    k["sizes"] = json!([format!("{w}x{h}"), format!("{h}x{w}")]);
                    k["max_frames"] = json!(if model.starts_with("wan21") { 81 } else { 121 });
                    k["frames"] = d["frames"].clone();
                }
                "tts" => k["max_chars"] = json!(4096),
                _ => {}
            }
            if !downloaded {
                k["note"] = json!(format!("{model} is not downloaded (Settings > Models > Media models)"));
            } else if !fits {
                k["note"] = json!(format!("{model} needs {:.1} GB; this PC's GPUs have {vram_gb:.1} GB", size.unwrap_or(0.0)));
            }
            kinds.insert(kind.to_string(), k);
        }
        let v = json!({"available": true, "kinds": kinds});
        *self.caps.lock().unwrap() = Some((Instant::now(), v.clone()));
        self.with_speeds(v)
    }

    fn with_speeds(&self, mut v: Value) -> Value {
        let speeds = self.speeds.lock().unwrap();
        if let Some(kinds) = v["kinds"].as_object_mut() {
            for k in kinds.values_mut() {
                if let Some(s) = k["model"].as_str().and_then(|m| speeds.get(m)) {
                    k["last_seconds"] = json!(s);
                }
            }
        }
        v
    }

    /// A tool call: the job queued, and the tool result for the model and the client (the job in `job`).
    pub fn submit(&self, name: &str, args: &Value, caps: &Value) -> Value {
        let (kind, key) = match name {
            "generate_image" => ("image", "prompt"),
            "generate_video" => ("video", "prompt"),
            _ => ("tts", "text"),
        };
        let fail = |t: String| json!({"ok": false, "text": t, "chars": t.len(), "truncated": false, "ms": 0});
        let Some(prompt) = args[key].as_str().map(str::trim).filter(|p| !p.is_empty()) else {
            return fail(format!("error: {name} needs \"{key}\""));
        };
        let k = &caps["kinds"][kind];
        if k["usable"] != true {
            return fail(format!("error: {} cannot be made here: {}", kind, k["note"].as_str().unwrap_or("not available")));
        }
        let model = k["model"].as_str().unwrap_or("").to_string();
        let mut req = match kind {
            "image" => {
                let size = args["size"].as_str().unwrap_or("1024x1024");
                let ok = size.split_once('x').and_then(|(a, b)| Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?)))
                    .is_some_and(|(w, h)| (256..=2048).contains(&w) && (256..=2048).contains(&h) && w % 16 == 0 && h % 16 == 0);
                if !ok {
                    return fail(format!("error: size {size:?} does not work; use one of {}", k["sizes"]));
                }
                json!({"prompt": prompt, "size": size})
            }
            "video" => {
                let mut r = json!({"prompt": prompt});
                if let Some(f) = args["frames"].as_u64() {
                    let max = k["max_frames"].as_u64().unwrap_or(81);
                    if f > max {
                        return fail(format!("error: {f} frames is more than this model makes ({max}); ask for {max} or fewer"));
                    }
                    r["frames"] = json!(f);
                }
                if let Some(z) = args["size"].as_str() {
                    r["size"] = json!(z);
                }
                r
            }
            _ => {
                if prompt.chars().count() > 4096 {
                    return fail("error: at most 4096 characters can be read aloud at once".into());
                }
                let mut r = json!({"input": prompt});
                if let Some(v) = args["voice"].as_str() {
                    r["voice"] = json!(v);
                }
                r
            }
        };
        let n = {
            let mut g = self.next.lock().unwrap();
            *g += 1;
            *g - 1
        };
        let id = format!("mj-{n}");
        req["strata_label"] = json!(id);
        let job = Job { id: id.clone(), kind, model: model.clone(), prompt: prompt.chars().take(300).collect(), created: crate::now(),
                        state: "queued", error: None, mime: None, data: None, meta: Value::Null, seconds: None };
        {
            let mut jobs = self.jobs.lock().unwrap();
            jobs.insert(n, job);
            while jobs.len() > KEEP {
                let Some(old) = jobs.iter().find(|(_, j)| j.state != "queued").map(|(k, _)| *k) else { break };
                jobs.remove(&old);
            }
        }
        let s = self.app().clone();
        tokio::spawn(async move { run(s, n, kind, req).await });
        let what = match kind { "image" => "image", "video" => "video", _ => "speech" };
        let eta = self.speeds.lock().unwrap().get(&model).map(|t| format!(", about {}", human(*t))).unwrap_or_default();
        let text = format!("{what} job {id} queued ({model}{eta}). It runs after this answer and appears in the chat by itself; \
                            tell the user it is on its way (do not describe the result, you will not see it).");
        let mut r = json!({"ok": true, "text": text, "chars": text.len(), "truncated": false, "ms": 0});
        r["job"] = json!({"id": id, "kind": kind, "model": model, "url": format!("/api/jobs/{id}")});
        r
    }

    fn finish(&self, n: u64, r: Result<(String, Vec<u8>, Value), String>) {
        let mut jobs = self.jobs.lock().unwrap();
        let Some(j) = jobs.get_mut(&n) else { return };
        let secs = (crate::now().saturating_sub(j.created)) as f64;
        match r {
            Ok((mime, data, meta)) => {
                j.state = "done";
                j.mime = Some(mime);
                j.data = Some(data);
                j.seconds = meta["seconds"].as_f64().or(Some(secs));
                j.meta = meta;
                if let Some(t) = j.seconds {
                    self.speeds.lock().unwrap().insert(j.model.clone(), t);
                }
            }
            Err(e) => {
                j.state = "error";
                j.error = Some(e);
            }
        }
    }

    fn status(&self, id: &str) -> Option<Value> {
        let n: u64 = id.strip_prefix("mj-")?.parse().ok()?;
        let jobs = self.jobs.lock().unwrap();
        let j = jobs.get(&n)?;
        let mut v = json!({"id": j.id, "object": "media.job", "kind": j.kind, "model": j.model, "prompt": j.prompt,
                           "created": j.created, "state": j.state, "error": j.error, "mime": j.mime, "meta": j.meta,
                           "seconds": j.seconds});
        if j.state == "done" {
            v["url"] = json!(format!("/v1/media/jobs/{}/content", j.id));
            v["bytes"] = json!(j.data.as_ref().map(|d| d.len()));
        }
        if j.state == "queued" {
            // its place in the queue, or running with its step
            let q = self.app().mgr.queue.snapshot();
            if q["running"]["label"] == j.id.as_str() {
                v["state"] = json!("running");
                if let Some((st, tot)) = self.app().mgr.monitor.media_progress() {
                    v["step"] = json!(st);
                    v["steps"] = json!(tot);
                }
            } else if let Some(w) = q["waiting"].as_array().and_then(|a| a.iter().find(|w| w["label"] == j.id.as_str())) {
                v["position"] = w["position"].clone();
                v["ahead"] = json!(q["waiting"].as_array().map(|a| a.iter().take_while(|x| x["label"] != j.id.as_str()).count()
                    + q["running"].is_object() as usize));
            }
        }
        Some(v)
    }
}

fn human(s: f64) -> String {
    if s < 90.0 { format!("{} s", s.round()) } else { format!("{} min", (s / 60.0).round()) }
}

/// The job, run through the same handler as the API (with the server's own key), its answer kept.
async fn run(s: AppState, n: u64, kind: &'static str, req: Value) {
    let mut headers = HeaderMap::new();
    if let Some(k) = &s.api_key {
        if let Ok(v) = HeaderValue::from_str(&format!("Bearer {k}")) {
            headers.insert(header::AUTHORIZATION, v);
        }
    }
    let r = match kind {
        "image" => crate::image_gen::generations(State(s.clone()), headers, Json(req)).await,
        "video" => crate::image_gen::video_generations(State(s.clone()), headers, Json(req)).await,
        _ => crate::audio_api::speech(State(s.clone()), headers, Json(req)).await,
    };
    let status = r.status();
    let ct = r.headers().get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let secs = r.headers().get("x-strata-seconds").and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<f64>().ok());
    let body = axum::body::to_bytes(r.into_body(), 512 << 20).await.map(|b| b.to_vec()).unwrap_or_default();
    let out = if !status.is_success() {
        let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        Err(v["error"]["message"].as_str().map(str::to_string).unwrap_or_else(|| String::from_utf8_lossy(&body).into_owned()))
    } else if ct.starts_with("audio/") {
        Ok((ct, body, json!({"seconds_of_audio": secs})))
    } else {
        let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        let d = &v["data"][0];
        let b64 = d["b64_json"].as_str().or(d["b64"].as_str()).unwrap_or("");
        match base64::engine::general_purpose::STANDARD.decode(b64) {
            Ok(bytes) if !bytes.is_empty() => {
                let mime = match d["format"].as_str() {
                    None => "image/png",
                    Some("mp4") => "video/mp4",
                    Some("gif") => "image/gif",
                    Some("webp") => "image/webp",
                    Some(_) => "image/apng",
                };
                Ok((mime.to_string(), bytes, json!({"size": v["size"], "frames": v["frames"], "fps": v["fps"], "seed": d["seed"],
                                                    "seconds": v["strata"]["seconds"], "frames_note": v["strata"]["frames_note"]})))
            }
            _ => Err("the media engine returned nothing".into()),
        }
    };
    s.media_jobs.finish(n, out);
}

pub async fn capabilities(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    Json(s.media_jobs.capabilities().await).into_response()
}

/// GET /api/jobs/{id}, /v1/media/jobs/{id}
pub async fn status(State(s): State<AppState>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    match s.media_jobs.status(&id) {
        Some(v) => Json(v).into_response(),
        None => error(StatusCode::NOT_FOUND, "not_found", format!("no media job {id} (jobs are kept in memory, the last {KEEP})")),
    }
}

/// GET /v1/media/jobs/{id}/content: the finished file. The web app's <img>/<video>/<audio> elements cannot send the
/// key, so a page of this server's own origin may fetch it as it fetches everything else (with the key) and use a blob.
pub async fn content(State(s): State<AppState>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let Some(n) = id.strip_prefix("mj-").and_then(|n| n.parse::<u64>().ok()) else {
        return error(StatusCode::NOT_FOUND, "not_found", format!("no media job {id}"));
    };
    let jobs = s.media_jobs.jobs.lock().unwrap();
    match jobs.get(&n) {
        Some(Job { data: Some(d), mime: Some(m), .. }) => {
            Response::builder().header(header::CONTENT_TYPE, m.as_str()).body(Body::from(d.clone())).unwrap()
        }
        Some(j) => error(StatusCode::CONFLICT, "not_ready", format!("media job {id} is {}", j.state)),
        None => error(StatusCode::NOT_FOUND, "not_found", format!("no media job {id}")),
    }
}

/// The chat's audio attachments (OpenAI's `input_audio` parts) transcribed on the CPU into text parts, before the
/// prompt is built, the way documents become text.
pub async fn expand_audio(s: &AppState, req: &mut Value) -> Result<(), String> {
    let Some(msgs) = req["messages"].as_array_mut() else { return Ok(()) };
    for m in msgs.iter_mut() {
        let Some(parts) = m["content"].as_array_mut() else { continue };
        for p in parts.iter_mut() {
            if p["type"] != "input_audio" {
                continue;
            }
            let data = p["input_audio"]["data"].as_str().unwrap_or("");
            let data = data.split_once(";base64,").map(|(_, b)| b).unwrap_or(data).to_string();
            let name = p["input_audio"]["name"].as_str().map(str::to_string)
                .unwrap_or_else(|| format!("audio.{}", p["input_audio"]["format"].as_str().unwrap_or("wav")));
            let text = crate::audio_api::transcribe_for_chat(s, &data).await
                .map_err(|e| format!("the audio attachment {name} could not be transcribed: {e}"))?;
            let (t, secs) = text;
            *p = json!({"type": "text", "text": format!("[Transcript of {name}, {secs:.0} s: {}]", t.trim())});
        }
    }
    Ok(())
}
