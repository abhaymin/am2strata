//! What the web app's Monitor tab shows (GET /metrics): the running request, the last requests, totals since the
//! start, and the machine (GPU, CPU, RAM, disk) with a minute of history per series. Also /metrics/prometheus (the
//! llama-server metric names) and /v1/status. Port of serve/server.py metrics() and serve/telemetry.py.
//!
//! The request history is kept on disk (<data>/monitor/history.jsonl, one finished request a line) until it is
//! cleared (POST /api/monitor/clear), so it outlives a restart. Each row names the model that ran it. What a media
//! job made (its images, video or speech) is saved beside it (<data>/monitor/media/) and listed in the row's `media`,
//! for the Monitor to show; clearing removes those files too.

use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Requests per page of /metrics (`?page=N&per=M`, newest first); `?requests=all` lists every one kept.
pub const PAGE_DEFAULT: usize = 50;
pub const PAGE_MAX: usize = 500;

/// Requests kept for the table until cleared: past this the oldest go, from memory (the file is rewritten without
/// them at the next start).
const HISTORY_KEEP: usize = 5000;
/// Samples per hardware series: one a second, a minute.
const SERIES_KEEP: usize = 60;
/// The live decode speed is measured over this many seconds.
const RATE_WINDOW_S: f64 = 3.0;
const SERIES: [&str; 10] =
    ["gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used", "disk_read_mb", "tok_s", "prefill_tok_s_mean"];

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

#[derive(Default, Clone)]
struct Totals {
    since: f64,
    requests: u64,
    prompt_tokens: u64,
    reused: u64,
    output_tokens: u64,
    prompt_ms: f64,
    decode_ms: f64,
    drafts_offered: u64,
    drafts_accepted: u64,
    tool_args_repaired: u64,
    stray_think_closers: u64,
}

struct Live {
    id: u64,
    model: String,
    started: f64,
    first_token: Option<f64>,
    prompt_tokens: usize,
    generated: u64,
    max_tokens: usize,
    reasoning: bool,
    progress: Option<(u64, u64)>,
    read_started: f64,
    recent: VecDeque<f64>,
}

/// Seconds per step from the first to the latest step reported; None before two steps.
fn s_per_step(m: &MediaLive) -> Option<f64> {
    match (m.first_step, m.last_step) {
        (Some((s0, t0)), Some((s1, t1))) if s1 > s0 => Some(((t1 - t0) / (s1 - s0) as f64 * 100.0).round() / 100.0),
        _ => None,
    }
}

/// A media job (image or video) on the GPU: shown live and kept in the request history like a chat request.
struct MediaLive {
    id: u64,
    kind: &'static str,
    model: String,
    started: f64,
    step: Option<(u64, u64)>,
    /// the seed of the image or clip being made now (each image of a batch has its own)
    seed: Option<u64>,
    /// what it was asked to make
    prompt: Option<String>,
    /// a long job's stage now (a video's clip k of n), and a line for each stage finished, oldest first
    stage: Option<(u64, u64)>,
    stage_log: Vec<String>,
    /// (step, time) of the first and the latest step reported: the time per step
    first_step: Option<(u64, f64)>,
    last_step: Option<(u64, f64)>,
    /// what the job made, saved with its row: (file extension, bytes)
    files: Vec<(String, Vec<u8>)>,
}

#[derive(Default)]
struct Inner {
    next_id: u64,
    live: Option<Live>,
    media: Option<MediaLive>,
    queued: usize,
    history: VecDeque<Value>,
    totals: Totals,
    last_timings: Option<Value>,
    last_at: Option<f64>,
}

#[derive(Default)]
struct Telemetry {
    now: serde_json::Map<String, Value>,
    hist: HashMap<&'static str, VecDeque<Value>>,
    stat: serde_json::Map<String, Value>,
}

/// how long a notice stays among the alerts
pub const NOTICE_S: f64 = 900.0;

pub struct Monitor {
    /// the health watch: cards the driver cannot reach, driver errors since the start, the last one
    health: Mutex<(Vec<String>, u64, Option<String>)>,
    /// events the user should hear about (an engine that stopped and was restarted, low memory, a long queue):
    /// (when, text), shown with the alerts for NOTICE_S
    notices: Mutex<std::collections::VecDeque<(f64, String)>>,
    inner: Mutex<Inner>,
    tel: Mutex<Telemetry>,
    pub started: f64,
    /// <data>/monitor: the history file and the media it lists (none: kept in memory only)
    dir: Option<PathBuf>,
}

/// A saved media file's name: digits, letters, '-' and one '.' (what `save_media` writes), so a request cannot reach
/// outside the folder.
pub fn media_name_ok(name: &str) -> bool {
    !name.is_empty() && name.len() < 64 && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.')
        && name.matches('.').count() == 1 && !name.starts_with('.')
}

impl Monitor {
    pub fn new(dir: Option<PathBuf>) -> Arc<Self> {
        let history = dir.as_ref().map(|d| Self::load(d)).unwrap_or_default();
        let next_id = history.iter().filter_map(|r| r["id"].as_u64()).max().unwrap_or(0);
        let m = Arc::new(Monitor {
            health: Mutex::new((Vec::new(), 0, None)),
            notices: Mutex::new(std::collections::VecDeque::new()),
            inner: Mutex::new(Inner { totals: Totals { since: now(), ..Default::default() }, history, next_id, ..Default::default() }),
            tel: Mutex::new(Telemetry::default()),
            started: now(),
            dir,
        });
        sampler::start(m.clone());
        m
    }

    /// The history file's rows, the newest HISTORY_KEEP; a damaged line (a crash mid-write) is skipped.
    fn load(dir: &std::path::Path) -> VecDeque<Value> {
        let text = std::fs::read_to_string(dir.join("history.jsonl")).unwrap_or_default();
        let mut rows: VecDeque<Value> = text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
        if rows.len() > HISTORY_KEEP {
            rows.drain(..rows.len() - HISTORY_KEEP);
            // the file keeps only what is shown: the rows past the limit (and their media) go
            let keep: String = rows.iter().map(|r| format!("{r}\n")).collect();
            let _ = std::fs::write(dir.join("history.jsonl"), keep);
        }
        rows
    }

    /// A finished row: into the table, and onto the end of the history file.
    fn push(&self, g: &mut Inner, row: Value) {
        if let Some(dir) = &self.dir {
            let line = format!("{row}\n");
            let ok = std::fs::create_dir_all(dir).and_then(|_| {
                std::fs::OpenOptions::new().create(true).append(true).open(dir.join("history.jsonl"))?.write_all(line.as_bytes())
            });
            if let Err(e) = ok {
                tracing::warn!("the Monitor's history file: {e}");
            }
        }
        g.history.push_back(row);
        while g.history.len() > HISTORY_KEEP {
            g.history.pop_front();
        }
    }

    /// Saves a media job's files; their names for the row ("/api/monitor/media/<name>" serves each).
    fn save_media(&self, id: u64, started: f64, files: &[(String, Vec<u8>)]) -> Vec<Value> {
        let Some(dir) = &self.dir else { return Vec::new() };
        let media = dir.join("media");
        if std::fs::create_dir_all(&media).is_err() {
            return Vec::new();
        }
        files.iter().enumerate().filter_map(|(i, (ext, bytes))| {
            let ext: String = ext.chars().filter(|c| c.is_ascii_alphanumeric()).take(5).collect();
            let name = format!("{}-{id}-{i}.{ext}", (started * 1000.0) as u64);
            std::fs::write(media.join(&name), bytes).ok()?;
            Some(json!({"name": name, "type": media_type(&ext), "bytes": bytes.len()}))
        }).collect()
    }

    /// POST /api/monitor/clear: the history, its file and its media go; the totals since the start stay.
    pub fn clear(&self) -> usize {
        let mut g = self.inner.lock().unwrap();
        let n = g.history.len();
        g.history.clear();
        if let Some(dir) = &self.dir {
            let _ = std::fs::remove_file(dir.join("history.jsonl"));
            let _ = std::fs::remove_dir_all(dir.join("media"));
        }
        n
    }

    /// A saved media file (GET /api/monitor/media/<name>): its type and bytes.
    pub fn media_file(&self, name: &str) -> Option<(&'static str, Vec<u8>)> {
        let dir = self.dir.as_ref().filter(|_| media_name_ok(name))?;
        let bytes = std::fs::read(dir.join("media").join(name)).ok()?;
        Some((media_type(name.rsplit('.').next().unwrap_or("")), bytes))
    }

    /// A request is waiting for the GPU; the returned guard counts it until dropped.
    pub fn waiting(self: &Arc<Self>) -> Waiting {
        self.inner.lock().unwrap().queued += 1;
        Waiting(self.clone())
    }

    /// A request starts on the engine: the guard records it in the history when it ends (or is dropped: the client
    /// left).
    pub fn begin(self: &Arc<Self>, model: &str, prompt_tokens: usize, max_tokens: usize) -> Record {
        let mut g = self.inner.lock().unwrap();
        g.next_id += 1;
        let id = g.next_id;
        let t = now();
        g.live = Some(Live { id, model: model.to_string(), started: t, first_token: None, prompt_tokens, generated: 0, max_tokens, reasoning: false,
                             progress: None, read_started: t, recent: VecDeque::new() });
        g.last_at = Some(t);
        Record { mon: self.clone(), id, prompt_tokens, done: false, outcome: "disconnect" }
    }

    fn set_health(&self, lost: Vec<String>, errors: u64, last: Option<String>) {
        let mut h = self.health.lock().unwrap();
        *h = (lost, errors, last);
    }

    /// Plain warnings for /health, /api/system and the web app: nothing means healthy.
    pub fn alerts(&self) -> Vec<String> {
        let h = self.health.lock().unwrap();
        let mut out: Vec<String> = h.0.iter().map(|c| format!("GPU {c} is on the bus but the driver cannot reach it")).collect();
        if h.1 > 0 {
            out.push(format!("{} GPU driver error(s) since this server started; the last: {}", h.1, h.2.as_deref().unwrap_or("")));
        }
        let t = now();
        out.extend(self.notices.lock().unwrap().iter().filter(|(at, _)| t - at < NOTICE_S).map(|(_, n)| n.clone()));
        out
    }

    /// Tell the user about an event (with /health's and the web app's alerts, for NOTICE_S); the same text again
    /// replaces the older one.
    pub fn notice(&self, text: &str) {
        let mut n = self.notices.lock().unwrap();
        n.retain(|(at, x)| x != text && now() - at < NOTICE_S);
        n.push_back((now(), text.to_string()));
        while n.len() > 20 {
            n.pop_front();
        }
    }

    /// What the output parser repaired in a request (Strata 0.2.8 #4427, #4077).
    pub fn parser_repairs(&self, args: usize, stray: usize) {
        let mut g = self.inner.lock().unwrap();
        g.totals.tool_args_repaired += args as u64;
        g.totals.stray_think_closers += stray as u64;
    }

    /// A media job starts; `media_step` reports its progress and `media_end` records it.
    pub fn media_begin(&self, kind: &'static str, model: &str) -> u64 {
        let mut g = self.inner.lock().unwrap();
        g.next_id += 1;
        let id = g.next_id;
        g.media = Some(MediaLive { id, kind, model: model.to_string(), started: now(), step: None, seed: None, prompt: None, stage: None, stage_log: Vec::new(), first_step: None,
                                   last_step: None, files: Vec::new() });
        g.last_at = Some(now());
        id
    }

    pub fn media_step(&self, id: u64, step: u64, total: u64) {
        if let Some(m) = self.inner.lock().unwrap().media.as_mut().filter(|m| m.id == id) {
            m.step = Some((step, total));
            let t = now();
            if m.first_step.is_none_or(|(s0, _)| step < s0) {
                m.first_step = Some((step, t));
            }
            m.last_step = Some((step, t));
        }
    }

    /// The seed of the image or clip a media job is making now.
    pub fn media_seed(&self, id: u64, seed: u64) {
        if let Some(m) = self.inner.lock().unwrap().media.as_mut().filter(|m| m.id == id) {
            m.seed = Some(seed);
        }
    }

    /// The media job running now: its step of steps, when it said.
    pub fn media_progress(&self) -> Option<(u64, u64)> {
        self.inner.lock().unwrap().media.as_ref().and_then(|m| m.step)
    }

    /// What a media job made, to be saved with its row when it ends: `ext` names the type ("png", "mp4", "wav").
    /// It also goes into the media library (media_files.rs), with the job's kind, model, seed and prompt. -> its id there.
    pub fn media_file_add(&self, id: u64, ext: &str, bytes: Vec<u8>) -> Option<String> {
        let mut g = self.inner.lock().unwrap();
        let m = g.media.as_mut().filter(|m| m.id == id)?;
        let meta = json!({"job": m.kind, "model": m.model, "seed": m.seed, "prompt": m.prompt});
        m.files.push((ext.to_string(), bytes.clone()));
        drop(g);
        crate::media_files::keep_generated(&bytes, meta)
    }

    /// A long job's stage (a video's clip `stage` of `stages`, of the scene `what`): when it moves on, the stage before
    /// is logged as finished, with the time it took so far.
    pub fn media_stage(&self, id: u64, stage: u64, stages: u64, what: &str) {
        let mut g = self.inner.lock().unwrap();
        let Some(m) = g.media.as_mut().filter(|m| m.id == id) else { return };
        if m.stage.is_some_and(|(s, _)| s == stage) {
            return;
        }
        if let Some((prev, n)) = m.stage {
            let t = now() - m.started;
            m.stage_log.push(format!("clip {prev} of {n} done at {}:{:02}", (t / 60.0) as u64, (t % 60.0) as u64));
        }
        if !what.is_empty() && (m.stage_log.is_empty() || m.stage.is_none_or(|(s, _)| s + 1 == stage)) {
            // the scene a clip belongs to, said once when it starts
            let scene: String = what.chars().take(80).collect();
            if !m.stage_log.last().is_some_and(|l| l.ends_with(&format!("\"{scene}\""))) {
                m.stage_log.push(format!("clip {stage} of {stages}: \"{scene}\""));
            }
        }
        m.stage = Some((stage, stages));
    }

    /// The media job running now: its step, stage and the stages finished (for the job's status and a waiting chat).
    pub fn media_detail(&self) -> Option<Value> {
        let g = self.inner.lock().unwrap();
        let m = g.media.as_ref()?;
        Some(json!({"step": m.step.map(|s| s.0), "steps": m.step.map(|s| s.1), "stage": m.stage.map(|s| s.0),
                    "stages": m.stage.map(|s| s.1), "log": m.stage_log, "elapsed_s": (now() - m.started).round()}))
    }

    /// What the media job was asked to make (kept with its files in the library).
    pub fn media_prompt(&self, id: u64, prompt: &str) {
        if let Some(m) = self.inner.lock().unwrap().media.as_mut().filter(|m| m.id == id) {
            m.prompt = Some(prompt.chars().take(2000).collect());
        }
    }

    /// `finish`: "stop" (made) or "error"; `output`: what was made ("2 images, 1024x1024").
    pub fn media_end(&self, id: u64, finish: &str, output: &str) {
        let mut g = self.inner.lock().unwrap();
        let Some(m) = g.media.take_if(|m| m.id == id) else { return };
        let t = now();
        let media = self.save_media(m.id, m.started, &m.files);
        let row = json!({"id": m.id, "time": m.started, "duration_s": ((t - m.started) * 10.0).round() / 10.0, "finish": finish,
                         "kind": m.kind, "model": m.model, "output": output, "steps": m.step.map(|s| s.1), "media": media,
                         "seed": m.seed, "s_per_step": s_per_step(&m)});
        self.push(&mut g, row);
    }

    fn with_live(&self, id: u64, f: impl FnOnce(&mut Live)) {
        let mut g = self.inner.lock().unwrap();
        if let Some(l) = g.live.as_mut().filter(|l| l.id == id) {
            f(l);
        }
    }

    fn end(&self, id: u64, prompt_tokens: usize, finish: &str, n: u64, done: Option<&strata_protocol::Done>, timings: Value) {
        let mut g = self.inner.lock().unwrap();
        let Some(live) = g.live.take_if(|l| l.id == id) else { return };
        let t = now();
        let d = done.cloned().unwrap_or_default();
        let read = done.and_then(|d| d.prompt_read);
        let seen = match (finish, read) {
            ("cancel" | "disconnect", Some(r)) => (r + d.reused.unwrap_or(0)) as usize,
            _ => prompt_tokens,
        };
        let decode_tok_s = (n > 0 && d.generated > 0 && d.decode_ms > 0.0)
            .then(|| (d.generated as f64 / (d.decode_ms / 1000.0) * 10.0).round() / 10.0);
        let hit_rate = d.lookups.filter(|l| *l > 0).map(|l| ((d.hits.unwrap_or(0) as f64 / l as f64) * 1000.0).round() / 1000.0);
        let row = json!({
            "id": live.id, "model": live.model, "time": live.started, "duration_s": ((t - live.started) * 10.0).round() / 10.0, "finish": finish,
            "prompt_tokens": seen, "reused": d.reused, "output_tokens": n,
            "prompt_total": prompt_tokens, "prompt_read": read, "engine_generated": done.map(|d| d.generated),
            "prompt_ms": done.map(|d| d.prompt_ms), "decode_ms": done.map(|d| d.decode_ms), "decode_tok_s": decode_tok_s,
            "hit_rate": hit_rate, "pcie_share": null, "projection": null,
            "drafts_offered": d.drafts_offered, "drafts_accepted": d.drafts_accepted,
        });
        self.push(&mut g, row);
        let tt = &mut g.totals;
        tt.requests += 1;
        tt.prompt_tokens += seen as u64;
        tt.reused += d.reused.unwrap_or(0);
        tt.output_tokens += n;
        tt.prompt_ms += d.prompt_ms;
        tt.decode_ms += d.decode_ms;
        tt.drafts_offered += d.drafts_offered.unwrap_or(0);
        tt.drafts_accepted += d.drafts_accepted.unwrap_or(0);
        if !timings.is_null() {
            g.last_timings = Some(timings);
        }
    }

    fn live_rates(l: &Live, t: f64) -> (Option<f64>, Option<f64>) {
        let recent = l.recent.iter().filter(|x| t - **x <= RATE_WINDOW_S).count();
        let window = l.first_token.map(|f| (t - f).clamp(0.001, RATE_WINDOW_S)).unwrap_or(RATE_WINDOW_S);
        let tok_s = l.first_token.map(|_| recent as f64 / window);
        let mean = l.first_token.map(|f| l.generated as f64 / (t - f).max(0.001));
        (tok_s, mean)
    }

    fn prefill_rate(l: &Live, t: f64) -> Option<f64> {
        let (read, _) = l.progress?;
        let end = l.first_token.unwrap_or(t);
        (end > l.read_started).then(|| read as f64 / (end - l.read_started))
    }

    /// The decode speed now and the prefill speed of the running request (for the hardware series).
    fn speeds(&self) -> (Option<f64>, Option<f64>) {
        let g = self.inner.lock().unwrap();
        let t = now();
        g.live.as_ref().map(|l| (Self::live_rates(l, t).0, Self::prefill_rate(l, t))).unwrap_or((None, None))
    }

    /// GET /metrics.
    /// `page`: (page from 1, rows per page) of the request history, newest first; None: all of it.
    pub fn metrics(&self, engine: Value, loaded: bool, page: Option<(usize, usize)>) -> Value {
        let g = self.inner.lock().unwrap();
        let t = now();
        let r1 = |v: f64| (v * 10.0).round() / 10.0;
        let live = match g.live.as_ref() {
            Some(l) => {
                let reading = l.first_token.is_none();
                let (tok_s, mean) = Self::live_rates(l, t);
                json!({
                    "state": if reading { "reading" } else { "generating" }, "queued": g.queued, "model": l.model,
                    "phase": if reading { Value::Null } else if l.reasoning { json!("thinking") } else { json!("answering") },
                    "prompt_tokens": l.prompt_tokens,
                    "prompt_read": if reading { l.progress.map(|p| p.0) } else { None },
                    "prompt_total": if reading { l.progress.map(|p| p.1) } else { None },
                    "generated": l.generated, "max_tokens": l.max_tokens, "elapsed_s": r1(t - l.started),
                    "tok_s": if reading { None } else { tok_s.map(r1) },
                    "tok_s_mean": if reading { None } else { mean.map(r1) },
                    "prefill_tok_s_mean": Self::prefill_rate(l, t).map(r1),
                    "tok_s_window_s": if reading { None } else { Some(RATE_WINDOW_S) },
                })
            }
            None if g.media.is_some() => {
                let m = g.media.as_ref().unwrap();
                json!({"state": "media", "kind": m.kind, "model": m.model, "step": m.step.map(|s| s.0), "steps": m.step.map(|s| s.1),
                       "stage": m.stage.map(|s| s.0), "stages": m.stage.map(|s| s.1), "stage_log": m.stage_log,
                       "seed": m.seed, "s_per_step": s_per_step(m),
                       "elapsed_s": r1(t - m.started), "queued": g.queued, "phase": null, "generated": null, "max_tokens": null,
                       "tok_s": null, "tok_s_mean": null, "prompt_tokens": null, "prompt_read": null, "prompt_total": null,
                       "prefill_tok_s_mean": null, "tok_s_window_s": null})
            }
            None => json!({"state": if loaded { "idle" } else { "unloaded" }, "queued": g.queued, "phase": null,
                           "prompt_tokens": null, "prompt_read": null, "prompt_total": null, "generated": null,
                           "max_tokens": null, "elapsed_s": null, "tok_s": null, "tok_s_mean": null,
                           "prefill_tok_s_mean": null, "tok_s_window_s": null}),
        };
        let total = g.history.len();
        let (page, per) = page.map(|(p, n)| (p.max(1), n.clamp(1, PAGE_MAX))).unwrap_or((1, total.max(1)));
        let pages = total.div_ceil(per).max(1);
        let page = page.min(pages);
        let hist: Vec<Value> = g.history.iter().rev().skip((page - 1) * per).take(per).cloned().collect();
        let tt = &g.totals;
        let totals = json!({"since": tt.since, "requests": tt.requests, "prompt_tokens": tt.prompt_tokens,
                            "reused": tt.reused, "output_tokens": tt.output_tokens, "prompt_ms": tt.prompt_ms,
                            "decode_ms": tt.decode_ms, "drafts_offered": tt.drafts_offered,
                            "drafts_accepted": tt.drafts_accepted});
        let last = g.history.back();
        let reused_requests = g.history.iter().filter(|r| r["reused"].as_u64().unwrap_or(0) > 0).count();
        let cache = json!({
            "enabled": false, "requests": g.history.len(), "requests_reused": reused_requests,
            "prompt_tokens": tt.prompt_tokens, "reused_tokens": tt.reused,
            "last_prompt": last.map(|r| r["prompt_tokens"].clone()), "last_reused": last.map(|r| r["reused"].clone()),
        });
        let tel = self.tel.lock().unwrap();
        let history: serde_json::Map<String, Value> =
            tel.hist.iter().map(|(k, v)| (k.to_string(), Value::Array(v.iter().cloned().collect()))).collect();
        json!({"engine": engine, "live": live, "requests": hist, "conversation_cache": cache,
               "requests_kept": total, "requests_page": {"page": page, "per": per, "pages": pages, "total": total}, "totals": totals, "hardware": tel.now, "hardware_static": tel.stat,
               "history": history, "time": t, "alerts": self.alerts()})
    }

    /// GET /metrics/prometheus: llama-server's metric names, so its dashboards work unchanged.
    pub fn prometheus(&self, loaded: bool) -> String {
        // read first: a lock taken in the table below would be held to the end of that statement
        let (unreachable, driver_errors) = {
            let h = self.health.lock().unwrap();
            (h.0.len() as f64, h.1 as f64)
        };
        let g = self.inner.lock().unwrap();
        let t = &g.totals;
        let read = t.prompt_tokens.saturating_sub(t.reused);
        let (ps, ds) = (t.prompt_ms / 1000.0, t.decode_ms / 1000.0);
        let rows: [(&str, &str, &str, f64); 15] = [
            ("llamacpp:prompt_tokens_total", "counter", "Prompt tokens read", read as f64),
            ("llamacpp:prompt_tokens_cached_total", "counter", "Prompt tokens the conversation cache already held", t.reused as f64),
            ("llamacpp:prompt_seconds_total", "counter", "Prompt read time", ps),
            ("llamacpp:tokens_predicted_total", "counter", "Tokens generated", t.output_tokens as f64),
            ("llamacpp:tokens_predicted_seconds_total", "counter", "Generation time", ds),
            ("llamacpp:requests_processing", "gauge", "Requests running", g.live.is_some() as u8 as f64),
            ("llamacpp:requests_deferred", "gauge", "Requests waiting their turn", g.queued as f64),
            ("strata:requests_total", "counter", "Requests finished", t.requests as f64),
            ("strata:drafts_offered_total", "counter", "Speculative draft tokens offered", t.drafts_offered as f64),
            ("strata:drafts_accepted_total", "counter", "Speculative draft tokens accepted", t.drafts_accepted as f64),
            ("strata:tool_args_repaired_total", "counter", "Tool call arguments (a JSON array or object) that were damaged and read as written", t.tool_args_repaired as f64),
            ("strata:stray_think_closers_total", "counter", "Extra closing think tags dropped from answers", t.stray_think_closers as f64),
            ("strata:gpus_unreachable", "gauge", "NVIDIA cards on the bus that the driver cannot reach", unreachable),
            ("strata:gpu_driver_errors_total", "counter", "GPU driver errors in the kernel log since this server started", driver_errors),
            ("strata:model_loaded", "gauge", "1 while the model is loaded", loaded as u8 as f64),
        ];
        rows.iter().map(|(n, k, h, v)| format!("# HELP {n} {h}\n# TYPE {n} {k}\n{n} {v}\n")).collect()
    }

    /// GET /v1/status: what this server is and does, for a client that would rather ask than guess.
    pub fn status(&self, model: Value, loaded: bool, ctx: Option<usize>, images: bool) -> Value {
        let g = self.inner.lock().unwrap();
        let tel = self.tel.lock().unwrap();
        let mib = |k: &str| tel.now.get(k).and_then(Value::as_f64).map(|v| (v / 1048576.0).round());
        let t = now();
        json!({
            "service": "strata", "server": "am2Strata", "model": model, "loaded": loaded, "auto_load": true,
            "started": self.started as u64, "uptime_s": (t - self.started) as u64,
            "cache_max_tokens": ctx, "context": {"native": ctx, "max_positions": ctx},
            "concurrency": {"serving": 1, "requested": 1},
            "dialects": ["/v1/chat/completions", "/v1/messages", "/v1/responses"],
            "vision": {"enabled": images, "available": images, "error": null},
            "image_generation": {"enabled": false},
            "activity": {"requests": g.totals.requests + g.live.is_some() as u64,
                         "in_flight": g.live.is_some() as usize + g.queued, "last_request_at": g.last_at.map(|x| x as u64)},
            "last_timings": g.last_timings,
            "machine": {"at": t as u64,
                        "gpu": tel.stat.get("gpu_name").map(|n| json!({"name": n, "used_mib": mib("gpu_mem_used"),
                            "total_mib": mib("gpu_mem_total"), "util_pct": tel.now.get("gpu_util"),
                            "temp_c": tel.now.get("gpu_temp"), "power_w": tel.now.get("gpu_power")})),
                        "gpus": tel.now.get("gpus"),
                        "ram": {"used_mib": mib("ram_used"), "total_mib": mib("ram_total")},
                        "cpu_pct": tel.now.get("cpu")},
        })
    }

    /// This machine for GET /api/system and Settings: every GPU, the CPU, the memory, each with its load now.
    pub fn system(&self) -> Value {
        let tel = self.tel.lock().unwrap();
        let mib = |k: &str| tel.now.get(k).and_then(Value::as_f64).map(|v| (v / 1048576.0).round());
        let gpus: Vec<Value> = tel.now.get("gpus").and_then(Value::as_array).into_iter().flatten().map(|g| {
            let m = |k: &str| g[k].as_f64().map(|v| (v / 1048576.0).round());
            json!({"index": g["index"], "name": g["name"], "util_pct": g["util"], "memory_used_mib": m("mem_used"), "memory_total_mib": m("mem_total"),
                   "temperature_c": g["temp"], "power_w": g["power"], "power_limit_w": g["power_limit"],
                   "pcie": {"gen": g["pcie_gen"], "gen_max": g["pcie_gen_max"], "width": g["pcie_width"]}})
        }).collect();
        let mut cpu = json!({"name": tel.stat.get("cpu_name"), "cores": tel.stat.get("cores"), "threads": tel.stat.get("threads"), "util_pct": tel.now.get("cpu")});
        // Apple silicon's performance and efficiency cores (macOS only: absent elsewhere, so Linux's answer is unchanged)
        for k in ["performance_cores", "efficiency_cores"] {
            if let Some(v) = tel.stat.get(k) {
                cpu[k] = v.clone();
            }
        }
        json!({"alerts": self.alerts(), "gpus": gpus, "gpu_driver": tel.stat.get("gpu_driver"), "gpu_note": tel.stat.get("gpu_note"),
               "cpu": cpu,
               "memory": {"used_mib": mib("ram_used"), "total_mib": mib("ram_total")},
               "accelerators": accelerators()})
    }
}

/// NPUs, TPUs and other AI accelerators on this machine: PCI devices of the accelerator classes, or known to be one, and
/// Coral's USB sticks. Most report no load to the OS: `util_pct` is then null and `state` says what can be seen.
pub fn accelerators() -> Vec<Value> {
    let mut out = Vec::new();
    // Apple silicon's Neural Engine: on every M-series chip, and macOS reports no load for it
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    out.push(json!({"kind": "NPU", "name": "Apple Neural Engine", "bus": "on-chip", "address": null, "driver": null, "device": null,
                    "state": "ready", "util_pct": null, "temperature_c": null, "busy_time_us": null, "note": "load not reported"}));
    let read = |p: std::path::PathBuf| std::fs::read_to_string(p).ok().map(|t| t.trim().to_string());
    let lspci = std::process::Command::new("lspci").args(["-D", "-nn"]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    for line in lspci.lines() {
        let l = line.to_lowercase();
        let by_class = l.contains("processing accelerators") || l.contains("signal processing controller");
        let by_name = l.contains("edge tpu") || l.contains(" npu") || l.contains(" ipu device") || l.contains("neural") || l.contains("ai boost") || l.contains("hailo");
        if !by_class && !by_name {
            continue;
        }
        let Some((addr, rest)) = line.split_once(' ') else { continue };
        let dev = std::path::Path::new("/sys/bus/pci/devices").join(addr);
        let driver = std::fs::read_link(dev.join("driver")).ok().and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));
        let name = rest.split_once(": ").map_or(rest, |x| x.1).trim();
        let kind = if l.contains("tpu") { "TPU" } else if l.contains("npu") || l.contains("ipu") || l.contains("neural") || l.contains("ai boost") { "NPU" } else { "accelerator" };
        // the device file the driver made: /dev/accel/accelN (NPUs), /dev/apex_N (Coral)
        let node = std::fs::read_dir(dev.join("accel")).ok().and_then(|mut d| d.next()).and_then(|e| e.ok())
            .map(|e| format!("/dev/accel/{}", e.file_name().to_string_lossy()))
            .or_else(|| std::fs::read_dir(dev.join("apex")).ok().and_then(|mut d| d.next()).and_then(|e| e.ok()).map(|e| format!("/dev/{}", e.file_name().to_string_lossy())));
        let power = read(dev.join("power/runtime_status"));
        let busy_us = read(dev.join("npu_busy_time_us")); // Intel's NPU counts its busy time
        let temp = read(dev.join("apex").join("apex_0").join("temp")).and_then(|t| t.parse::<f64>().ok()).map(|t| t / 1000.0);
        let state = match (&driver, &power) {
            (None, _) => "no driver loaded".to_string(),
            (Some(_), Some(p)) if p == "suspended" => "idle (powered down)".to_string(),
            (Some(_), Some(p)) if p == "active" => "ready".to_string(),
            _ => "ready".to_string(),
        };
        out.push(json!({"kind": kind, "name": name, "bus": "PCIe", "address": addr, "driver": driver, "device": node, "state": state,
                        "util_pct": null, "temperature_c": temp, "busy_time_us": busy_us.and_then(|v| v.parse::<u64>().ok())}));
    }
    for d in std::fs::read_dir("/sys/bus/usb/devices").into_iter().flatten().flatten() {
        let id = (read(d.path().join("idVendor")), read(d.path().join("idProduct")));
        if matches!((id.0.as_deref(), id.1.as_deref()), (Some("1a6e"), Some("089a")) | (Some("18d1"), Some("9302"))) {
            out.push(json!({"kind": "TPU", "name": "Coral USB Accelerator", "bus": "USB", "address": d.file_name().to_string_lossy(), "driver": null,
                            "device": null, "state": "connected", "util_pct": null, "temperature_c": null, "busy_time_us": null}));
        }
    }
    out
}

fn media_type(ext: &str) -> &'static str {
    match ext {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "apng" => "image/apng",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        _ => "application/octet-stream",
    }
}

/// Counts a request waiting for the GPU while it lives.
pub struct Waiting(Arc<Monitor>);

impl Drop for Waiting {
    fn drop(&mut self) {
        let mut g = self.0.inner.lock().unwrap();
        g.queued = g.queued.saturating_sub(1);
    }
}

/// A running media job: recorded as "disconnect" when dropped before `media_end` (its client went away), so the
/// live view does not show it running for ever.
pub struct MediaRow {
    mon: Arc<Monitor>,
    pub id: u64,
}

impl MediaRow {
    pub fn new(mon: &Arc<Monitor>, id: u64) -> Self {
        Self { mon: mon.clone(), id }
    }
}

impl Drop for MediaRow {
    fn drop(&mut self) {
        // a no-op when the job already ended
        self.mon.media_end(self.id, "disconnect", "the client went away before it ended");
    }
}

/// One running request; recorded in the history by `finish`, or as "disconnect" when dropped before.
pub struct Record {
    mon: Arc<Monitor>,
    id: u64,
    prompt_tokens: usize,
    done: bool,
    /// what the history says when the record is dropped unfinished
    outcome: &'static str,
}

impl Record {
    pub fn token(&self, reasoning: bool) {
        self.mon.with_live(self.id, |l| {
            let t = now();
            l.first_token.get_or_insert(t);
            l.generated += 1;
            l.reasoning = reasoning;
            l.recent.push_back(t);
            while l.recent.front().is_some_and(|x| t - x > RATE_WINDOW_S) {
                l.recent.pop_front();
            }
        });
    }

    /// The request ends with an error: so the history says, when the record is dropped.
    pub fn fail(&mut self) {
        self.outcome = "error";
    }

    pub fn progress(&self, done: u64, total: u64) {
        self.mon.with_live(self.id, |l| l.progress = Some((done, total)));
    }

    /// What the output parser repaired in this request.
    pub fn parser_repairs(&self, args: usize, stray: usize) {
        self.mon.parser_repairs(args, stray);
    }

    pub fn finish(mut self, finish: &str, n: u64, done: Option<&strata_protocol::Done>, timings: Value) {
        self.done = true;
        self.mon.end(self.id, self.prompt_tokens, finish, n, done, timings);
    }
}

impl Drop for Record {
    fn drop(&mut self) {
        if !self.done {
            let n = {
                let g = self.mon.inner.lock().unwrap();
                g.live.as_ref().filter(|l| l.id == self.id).map(|l| l.generated).unwrap_or(0)
            };
            self.mon.end(self.id, self.prompt_tokens, self.outcome, n, None, Value::Null);
        }
    }
}

/// The hardware sampler: nvidia-smi in its own loop mode (one process, a line per GPU a second), /proc for the CPU,
/// RAM and disks. Missing tools leave their fields out, as PolyStrata's telemetry does.
mod sampler {
    use super::*;
    use tokio::io::{AsyncBufReadExt, BufReader};

    const GPU_FIELDS: &str = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,\
                              pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current";

    /// NVIDIA display cards on the PCI bus, as `0000:08:00.0`.
    fn cards_on_bus() -> Vec<String> {
        let mut out = Vec::new();
        for d in std::fs::read_dir("/sys/bus/pci/devices").into_iter().flatten().flatten() {
            let read = |n: &str| std::fs::read_to_string(d.path().join(n)).unwrap_or_default();
            if read("vendor").trim() == "0x10de" && read("class").starts_with("0x03") {
                out.push(d.file_name().to_string_lossy().into_owned());
            }
        }
        out.sort();
        out
    }

    /// What the health watch reports now: cards the driver cannot reach, and new driver errors since `since`.
    pub(super) fn check(since: &str) -> (Vec<String>, u64, Vec<String>) {
        let reach = std::process::Command::new("nvidia-smi").args(["--query-gpu=pci.bus_id", "--format=csv,noheader"]).output()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_lowercase()).unwrap_or_default();
        let lost: Vec<String> = cards_on_bus().into_iter().filter(|c| !reach.lines().any(|l| l.trim().ends_with(c.as_str()))).collect();
        let log = std::process::Command::new("journalctl").args(["-k", "--no-pager", "-q", "--since", since]).output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
        let errors: Vec<String> = log.lines().filter(|l| l.contains("NVRM: Xid") || l.contains("fallen off the bus"))
            .map(|l| l.chars().take(200).collect()).collect();
        (lost, errors.len() as u64, errors)
    }

    pub fn start(m: Arc<Monitor>) {
        // the health watch: a card lost, or a driver error, is seen within 30 s
        let watch = m.clone();
        tokio::spawn(async move {
            let since = format!("@{}", now() as u64);
            let mut known = (Vec::<String>::new(), 0u64);
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                let s = since.clone();
                let Ok((lost, n, errors)) = tokio::task::spawn_blocking(move || check(&s)).await else { continue };
                if lost != known.0 {
                    if lost.is_empty() {
                        tracing::info!("health: every GPU is reachable again");
                    } else {
                        tracing::warn!("health: GPU(s) on the bus that the driver cannot reach: {}", lost.join(", "));
                    }
                }
                if n > known.1 {
                    tracing::warn!("health: {} new GPU driver error(s): {}", n - known.1, errors.last().map(String::as_str).unwrap_or(""));
                }
                known = (lost.clone(), n);
                watch.set_health(lost, n, errors.last().cloned());
            }
        });
        static_facts(&m);
        let gpus: Arc<Mutex<HashMap<u32, Vec<String>>>> = Arc::default();
        if let Ok(mut child) = tokio::process::Command::new("nvidia-smi")
            .args([&format!("--query-gpu={GPU_FIELDS}"), "--format=csv,noheader,nounits", "-lms", "1000"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
        {
            let stdout = child.stdout.take();
            let g = gpus.clone();
            tokio::spawn(async move {
                let _child = child;
                let Some(out) = stdout else { return };
                let mut lines = BufReader::new(out).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let f: Vec<String> = line.split(',').map(|x| x.trim().to_string()).collect();
                    if let Some(i) = f.first().and_then(|x| x.parse().ok()) {
                        g.lock().unwrap().insert(i, f);
                    }
                }
            });
        }
        tokio::spawn(async move {
            let mut cpu_prev = cpu_times();
            let mut disk_prev = (now(), disk_sectors());
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                let mut s = serde_json::Map::new();
                gpu_fields(&gpus.lock().unwrap(), &mut s);
                #[cfg(target_os = "macos")]
                if let Some(g) = mac::gpu_now().await {
                    s.insert("gpu_util".into(), g["util"].clone());
                    s.insert("gpus".into(), json!([g]));
                }
                let cpu = cpu_times();
                if let (Some((b0, i0)), Some((b1, i1))) = (cpu_prev, cpu) {
                    let (db, di) = (b1.saturating_sub(b0) as f64, i1.saturating_sub(i0) as f64);
                    if db + di > 0.0 {
                        s.insert("cpu".into(), json!((db / (db + di) * 1000.0).round() / 10.0));
                    }
                }
                cpu_prev = cpu;
                if let Some((used, total)) = ram() {
                    s.insert("ram_used".into(), json!(used));
                    s.insert("ram_total".into(), json!(total));
                }
                let disk = (now(), disk_sectors());
                if let (Some((r0, w0)), Some((r1, w1))) = (disk_prev.1, disk.1) {
                    let dt = (disk.0 - disk_prev.0).max(0.001);
                    let mb = |a: u64, b: u64| ((b.saturating_sub(a) as f64 * 512.0 / 1048576.0 / dt) * 100.0).round() / 100.0;
                    s.insert("disk_read_mb".into(), json!(mb(r0, r1)));
                    s.insert("disk_write_mb".into(), json!(mb(w0, w1)));
                }
                disk_prev = disk;
                let (tok_s, prefill) = m.speeds();
                s.insert("tok_s".into(), json!(tok_s.map(|v| (v * 100.0).round() / 100.0)));
                s.insert("prefill_tok_s_mean".into(), json!(prefill.map(|v| (v * 100.0).round() / 100.0)));
                let mut tel = m.tel.lock().unwrap();
                for k in SERIES {
                    let q = tel.hist.entry(k).or_default();
                    q.push_back(s.get(k).cloned().unwrap_or(Value::Null));
                    while q.len() > SERIES_KEEP {
                        q.pop_front();
                    }
                }
                s.remove("tok_s");
                s.remove("prefill_tok_s_mean");
                tel.now = s;
            }
        });
    }

    fn num(f: &[String], i: usize) -> Option<f64> {
        f.get(i).and_then(|x| x.parse::<f64>().ok())
    }

    fn gpu_fields(gpus: &HashMap<u32, Vec<String>>, s: &mut serde_json::Map<String, Value>) {
        if gpus.is_empty() {
            return;
        }
        let mut idx: Vec<&u32> = gpus.keys().collect();
        idx.sort();
        let rows: Vec<&Vec<String>> = idx.iter().map(|i| &gpus[i]).collect();
        let mib = |v: Option<f64>| v.map(|x| x * 1048576.0);
        let sum = |i: usize| -> Option<f64> {
            let v: Vec<f64> = rows.iter().filter_map(|r| num(r, i)).collect();
            (!v.is_empty()).then(|| v.iter().sum())
        };
        let utils: Vec<f64> = rows.iter().filter_map(|r| num(r, 2)).collect();
        let temps: Vec<f64> = rows.iter().filter_map(|r| num(r, 5)).collect();
        s.insert("gpu_util".into(), json!((!utils.is_empty()).then(|| utils.iter().sum::<f64>() / utils.len() as f64)));
        s.insert("gpu_mem_used".into(), json!(mib(sum(3))));
        s.insert("gpu_mem_total".into(), json!(mib(sum(4))));
        s.insert("gpu_temp".into(), json!(temps.iter().cloned().fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.max(b))))));
        s.insert("gpu_power".into(), json!(sum(6)));
        s.insert("gpu_power_limit".into(), json!(sum(7)));
        s.insert("gpu_pcie_gen".into(), json!(num(rows[0], 8)));
        s.insert("gpu_pcie_gen_max".into(), json!(num(rows[0], 9)));
        s.insert("gpu_pcie_width".into(), json!(num(rows[0], 10)));
        let per: Vec<Value> = rows
            .iter()
            .map(|r| json!({"index": num(r, 0), "name": r.get(1), "util": num(r, 2), "mem_used": mib(num(r, 3)), "mem_total": mib(num(r, 4)),
                            "temp": num(r, 5), "power": num(r, 6), "power_limit": num(r, 7), "pcie_gen": num(r, 8), "pcie_gen_max": num(r, 9),
                            "pcie_width": num(r, 10)}))
            .collect();
        s.insert("gpus".into(), json!(per));
    }

    fn static_facts(m: &Monitor) {
        let mut st = serde_json::Map::new();
        if let Ok(o) = std::process::Command::new("nvidia-smi").args(["--query-gpu=name,driver_version", "--format=csv,noheader"]).output() {
            let text = String::from_utf8_lossy(&o.stdout);
            if let Some(first) = text.lines().next().filter(|l| !l.is_empty()) {
                let mut f = first.split(',').map(str::trim);
                st.insert("gpu_name".into(), json!(f.next()));
                st.insert("gpu_driver".into(), json!(f.next()));
            }
        }
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        if let Some(name) = cpuinfo.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split(':').nth(1)) {
            st.insert("cpu_name".into(), json!(name.trim()));
        }
        st.insert("threads".into(), json!(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)));
        let cores: std::collections::HashSet<(String, String)> = cpuinfo
            .split("\n\n")
            .filter_map(|b| {
                let get = |k: &str| b.lines().find(|l| l.starts_with(k)).and_then(|l| l.split(':').nth(1)).map(|v| v.trim().to_string());
                Some((get("physical id")?, get("core id")?))
            })
            .collect();
        if !cores.is_empty() {
            st.insert("cores".into(), json!(cores.len()));
        }
        #[cfg(target_os = "macos")]
        mac::static_facts(&mut st);
        st.insert("psutil".into(), json!(true)); // the web app's "needs psutil" hint does not apply
        m.tel.lock().unwrap().stat = st;
    }

    /// (busy, idle) ticks of all CPUs: the kernel's counters on macOS (no process started).
    #[cfg(target_os = "macos")]
    fn cpu_times() -> Option<(u64, u64)> {
        mac::cpu_times()
    }

    /// (busy, idle) jiffies of all CPUs.
    #[cfg(not(target_os = "macos"))]
    fn cpu_times() -> Option<(u64, u64)> {
        let stat = std::fs::read_to_string("/proc/stat").ok()?;
        let v: Vec<u64> = stat.lines().next()?.split_whitespace().skip(1).filter_map(|x| x.parse().ok()).collect();
        let idle = v.get(3)? + v.get(4).unwrap_or(&0);
        Some((v.iter().take(8).sum::<u64>() - idle, idle))
    }

    /// macOS: the CPU from sysctl, its load from the kernel's tick counters, the GPU (Apple silicon's, on the same memory
    /// as the CPU) from the IOAccelerator entry in the I/O registry.
    #[cfg(target_os = "macos")]
    pub(super) mod mac {
        use serde_json::{json, Map, Value};

        fn sysctl_string(name: &str) -> Option<String> {
            let c = std::ffi::CString::new(name).ok()?;
            let mut len: libc::size_t = 0;
            unsafe {
                if libc::sysctlbyname(c.as_ptr(), std::ptr::null_mut(), &mut len, std::ptr::null_mut(), 0) != 0 || len == 0 {
                    return None;
                }
                let mut buf = vec![0u8; len];
                if libc::sysctlbyname(c.as_ptr(), buf.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0) != 0 {
                    return None;
                }
                buf.truncate(len);
                let s = String::from_utf8_lossy(&buf).trim_end_matches('\0').trim().to_string();
                (!s.is_empty()).then_some(s)
            }
        }

        fn sysctl_u32(name: &str) -> Option<u32> {
            let c = std::ffi::CString::new(name).ok()?;
            let mut v: u32 = 0;
            let mut len = std::mem::size_of::<u32>();
            let ok = unsafe { libc::sysctlbyname(c.as_ptr(), (&mut v as *mut u32).cast(), &mut len, std::ptr::null_mut(), 0) } == 0;
            (ok && v > 0).then_some(v)
        }

        /// The chip's name ("Apple M4"), its cores (and Apple silicon's performance / efficiency split), and the GPU.
        pub fn static_facts(st: &mut Map<String, Value>) {
            if let Some(n) = sysctl_string("machdep.cpu.brand_string") {
                st.insert("cpu_name".into(), json!(n));
            }
            if let Some(c) = sysctl_u32("hw.physicalcpu") {
                st.insert("cores".into(), json!(c));
            }
            // perflevel0 is the performance cluster, perflevel1 the efficiency one (absent on Intel Macs)
            if let (Some(p), Some(e)) = (sysctl_u32("hw.perflevel0.physicalcpu"), sysctl_u32("hw.perflevel1.physicalcpu")) {
                st.insert("performance_cores".into(), json!(p));
                st.insert("efficiency_cores".into(), json!(e));
            }
            if let Some(name) = gpu_name() {
                st.insert("gpu_name".into(), json!(name));
                st.insert("gpu_note".into(), json!("unified memory: the GPU uses the system RAM; its memory is what it has allocated now"));
            }
        }

        /// (busy, idle) ticks of all CPUs since boot: host_statistics64(HOST_CPU_LOAD_INFO).
        pub fn cpu_times() -> Option<(u64, u64)> {
            let mut info: libc::host_cpu_load_info = unsafe { std::mem::zeroed() };
            let mut count = libc::HOST_CPU_LOAD_INFO_COUNT;
            #[allow(deprecated)] // mach_host_self: libc points to the mach2 crate, which this crate does not use
            let r = unsafe { libc::host_statistics64(libc::mach_host_self(), libc::HOST_CPU_LOAD_INFO, (&mut info as *mut libc::host_cpu_load_info).cast(), &mut count) };
            if r != libc::KERN_SUCCESS {
                return None;
            }
            let t = info.cpu_ticks.map(|x| x as u64);
            let idle = t[libc::CPU_STATE_IDLE as usize];
            Some((t.iter().sum::<u64>() - idle, idle))
        }

        /// A value of the IOAccelerator entry's text as `ioreg -r -d 1 -c IOAccelerator` prints it: `"key" = value`.
        fn field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
            let at = text.find(&format!("\"{key}\""))? + key.len() + 2;
            let rest = text[at..].trim_start().strip_prefix('=')?.trim_start();
            Some(rest.split(['\n', ',', '}']).next()?.trim())
        }

        fn ioreg() -> Option<String> {
            let o = std::process::Command::new("ioreg").args(["-r", "-d", "1", "-c", "IOAccelerator"]).output().ok()?;
            o.status.success().then(|| String::from_utf8_lossy(&o.stdout).into_owned())
        }

        /// "Apple M4 (10-core GPU)": the accelerator's model and core count, read once.
        fn gpu_name() -> Option<String> {
            let text = ioreg()?;
            let model = field(&text, "model")?.trim_matches('"').to_string();
            Some(match field(&text, "gpu-core-count").and_then(|c| c.parse::<u32>().ok()) {
                Some(n) => format!("{model} ({n}-core GPU)"),
                None => model,
            })
        }

        /// The GPU now: its utilization and the memory it has in use (one `ioreg` a second, about 10 ms; not on the
        /// sampler's thread).
        pub async fn gpu_now() -> Option<Value> {
            let o = tokio::process::Command::new("ioreg").args(["-r", "-d", "1", "-c", "IOAccelerator"]).output().await.ok()?;
            let text = String::from_utf8_lossy(&o.stdout);
            let util = field(&text, "Device Utilization %").and_then(|v| v.parse::<f64>().ok())?;
            let used = field(&text, "In use system memory").and_then(|v| v.parse::<f64>().ok());
            // the name static_facts read ("Apple M4 (10-core GPU)"), else the model
            static NAME: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
            let name = NAME.get_or_init(gpu_name).clone().or_else(|| field(&text, "model").map(|m| m.trim_matches('"').to_string()));
            Some(json!({"index": 0, "name": name, "util": util, "mem_used": used, "mem_total": null, "temp": null, "power": null,
                        "power_limit": null, "pcie_gen": null, "pcie_gen_max": null, "pcie_width": null}))
        }

        #[cfg(test)]
        mod tests {
            #[test]
            fn this_mac_has_a_named_cpu_with_cores() {
                let mut st = serde_json::Map::new();
                super::static_facts(&mut st);
                assert!(st.get("cpu_name").and_then(|v| v.as_str()).is_some_and(|n| !n.is_empty()), "{st:?}");
                assert!(st.get("cores").and_then(|v| v.as_u64()).is_some_and(|c| c > 0), "{st:?}");
                let (b, i) = super::cpu_times().expect("cpu ticks");
                assert!(b + i > 0);
            }

            #[test]
            fn the_io_registry_fields_are_read() {
                let text = r#"  "model" = "Apple M4"
      "gpu-core-count" = 10
      "PerformanceStatistics" = {"Tiler Utilization %"=11,"Device Utilization %"=7,"In use system memory"=287227904}"#;
                assert_eq!(super::field(text, "model"), Some("\"Apple M4\""));
                assert_eq!(super::field(text, "gpu-core-count"), Some("10"));
                assert_eq!(super::field(text, "Device Utilization %"), Some("7"));
                assert_eq!(super::field(text, "In use system memory"), Some("287227904"));
            }
        }
    }

    /// (used, total) bytes of RAM, as psutil counts them (total - available). From mem.rs on every OS.
    fn ram() -> Option<(u64, u64)> {
        let total = crate::mem::total_mib();
        if total == 0 { return None; }   // could not tell (was: /proc/meminfo missing off Linux)
        let avail = crate::mem::available_mib();
        Some((total.saturating_sub(avail) * 1_048_576, total * 1_048_576))
    }

    /// Sectors read and written by the whole disks (not their partitions).
    fn disk_sectors() -> Option<(u64, u64)> {
        let stats = std::fs::read_to_string("/proc/diskstats").ok()?;
        let (mut r, mut w) = (0u64, 0u64);
        for line in stats.lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            let Some(name) = f.get(2) else { continue };
            let whole = std::path::Path::new(&format!("/sys/block/{name}/device")).exists();
            if whole {
                r += f.get(5).and_then(|x| x.parse::<u64>().ok()).unwrap_or(0);
                w += f.get(9).and_then(|x| x.parse::<u64>().ok()).unwrap_or(0);
            }
        }
        Some((r, w))
    }
}

#[cfg(test)]
mod history_tests {
    use super::*;

    #[tokio::test]
    async fn a_media_job_whose_client_went_away_ends_as_a_disconnect() {
        let m = Arc::new(Monitor::new(None));
        let id = m.media_begin("text-to-speech", "parler-tts-mini-v1");
        drop(MediaRow::new(&m, id));
        let v = m.metrics(Value::Null, false, None);
        assert_ne!(v["live"]["state"], "media", "{v}");
        assert_eq!(v["requests"][0]["finish"], "disconnect");
        // a job that ended first is not ended twice
        let id = m.media_begin("image", "z-image-turbo:q8");
        let row = MediaRow::new(&m, id);
        m.media_end(id, "stop", "1 image, 512x512");
        drop(row);
        assert_eq!(m.metrics(Value::Null, false, None)["requests"].as_array().map(Vec::len), Some(2));
    }

    #[tokio::test]
    async fn the_history_comes_in_pages_newest_first() {
        let m = Arc::new(Monitor::new(None));
        for k in 0..7 {
            let id = m.media_begin("image", "z-image-turbo:q8");
            m.media_end(id, "stop", &format!("image {k}"));
        }
        let page = |p, n| m.metrics(Value::Null, false, Some((p, n)));
        let v = page(1, 3);
        assert_eq!(v["requests_page"], json!({"page": 1, "per": 3, "pages": 3, "total": 7}));
        let outs = |v: &Value| v["requests"].as_array().unwrap().iter().map(|r| r["output"].as_str().unwrap_or("").to_string()).collect::<Vec<_>>();
        assert_eq!(outs(&v), ["image 6", "image 5", "image 4"]);
        assert_eq!(outs(&page(3, 3)), ["image 0"]);
        // a page past the end is the last page; all of them with None
        assert_eq!(page(9, 3)["requests_page"]["page"], 3);
        assert_eq!(m.metrics(Value::Null, false, None)["requests"].as_array().map(Vec::len), Some(7));
    }

    #[tokio::test]
    async fn the_history_file_is_read_back_at_the_next_start() {
        let dir = std::env::temp_dir().join(format!("am2-monitor-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let m = Monitor::new(Some(dir.clone()));
        let id = m.media_begin("image", "z-image-turbo:q8");
        m.media_file_add(id, "png", b"\x89PNG..".to_vec());
        m.media_end(id, "stop", "1 image, 512x512");
        std::fs::OpenOptions::new().append(true).open(dir.join("history.jsonl")).unwrap().write_all(b"{\"time\": 1, broken\n").unwrap();
        // a new start: the row is back (the damaged line skipped), its file is served, and ids go on after it
        let m = Monitor::new(Some(dir.clone()));
        let rows = m.metrics(Value::Null, false, None)["requests"].clone();
        assert_eq!(rows.as_array().map(Vec::len), Some(1), "{rows}");
        assert_eq!(rows[0]["model"], "z-image-turbo:q8");
        let name = rows[0]["media"][0]["name"].as_str().unwrap();
        assert_eq!(m.media_file(name).map(|f| f.0), Some("image/png"));
        assert!(m.media_begin("video", "wan") > rows[0]["id"].as_u64().unwrap());
        assert!(m.media_file("../history.jsonl").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
