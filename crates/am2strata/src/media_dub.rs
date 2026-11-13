//! Dub a narration track into a soundtrack in another language — `POST /v1/media/dub` and the `dub_narration` tool.
//!
//! Port of the reviewer's `.agent/ref/dub.py`: it re-lays the narration over "One PC, Many Voices" so every take is
//! placed at the SAME beat times the film was built on and the other stems (music, listening clips) stay as they are —
//! the picture does not move. Only the narration lines change, one language at a time. The stages mirror dub.py:
//!
//! - **translate**: validate the translation file (a line per narration beat, speakers unchanged in order, no empty
//!   text) and report each line's slot length and characters/second so a long line can be shortened;
//! - **speak**: take a line through the speech model with a seed retry (the take file is overwritten each seed, so a
//!   long first take does not get re-measured on top of itself), cap the pauses INSIDE the take at 0.25 s (task 5's
//!   `check_media` pause definition) before fitting, and keep the shortest take when none fit within the 12% ceiling;
//! - **mix**: place each take at its beat time in a full-length soundtrack of the original length — a take longer than
//!   its window (slot + the gap before the next beat) is sped up with ffmpeg's atempo to fit, and only when even the
//!   2x ceiling does not help it is cut (faded out over its last 0.3 s); every change is reported;
//! - **hear**: fill `words_heard` from each take with task 5's `check_media`, the Unicode-aware share of the expected
//!   words heard (Hindi nukta/chandrabindu folded so Whisper's spelling still matches).
//!
//! The input is the narration lines with their beat times, the other audio stems and a target language; the output is
//! the soundtrack WAV plus one report entry per line that was sped up, cut or kept silent.

use crate::{api_authorized, error, AppState};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine as _;
use serde_json::{json, Value};
use std::process::Command;
use std::sync::Arc;

pub const TOOL_NAMES: [&str; 1] = ["dub_narration"];

/// The narration sample rate (what the takes are recorded at), shared with dub.py.
pub const SR: u32 = 48_000;
/// A take may be sped up by at most this to fit its slot in `speak` (the 12% ceiling).
const MAX_TEMPO: f64 = 1.12;
/// In `mix` a take longer than its window may be sped up by at most this, else CUT.
const MIX_MAX_TEMPO: f64 = 2.0;
/// The gap before the next beat, added to a slot's duration for the width a take has to fit in (dub.py's GAP_S).
const GAP_S: f64 = 0.2;
/// How long the last part of a cut take is faded out over (dub.py's 0.3 s).
const CUT_FADE_S: f64 = 0.3;

fn r3(x: f64) -> f64 { (x * 1000.0).round() / 1000.0 }
fn max_f(a: f64, b: f64) -> f64 { if a > b { a } else { b } }

// ------------------------------------------------------------------------------------------------------------- reading
/// A WAV's samples at its own rate, via task 5's `check_media` reader (16/8-bit PCM and float).
pub fn read_wav(data: &[u8]) -> Result<(u32, Vec<f32>), String> { crate::media_check::read_wav(data) }

/// A mono 48 kHz 16-bit WAV of `samples`, as bytes (the format the soundtrack is written in).
pub fn write_wav(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + samples.len() as u32 * 2).to_le_bytes()));
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(SR as u32).to_le_bytes());
    out.extend_from_slice(&(SR as u32 * 2u32).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(samples.len() as u32 * 2u32).to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    out
}

/// Linearly resample `x` from `rate` to the narration rate, so takes recorded elsewhere still line up sample-accurate.
fn resample(x: &[f32], rate: u32) -> Vec<f32> {
    if rate == SR as u32 || x.len() < 2 { return x.to_vec(); }
    let n = (x.len() as f64 * SR as f64 / rate as f64).round() as usize;
    let src: Vec<f64> = x.iter().map(|&s| s as f64).collect();
    (0..n).map(|i| {
        let pos = i as f64 * src.len() as f64 / n as f64;
        let lo = pos as usize;
        if lo + 1 < src.len() {
            let w = pos - lo as f64;
            src[lo] * (1.0 - w) + src[lo + 1] * w
        } else {
            src[lo.min(src.len() - 1)]
        }
    }).map(|v| v as f32).collect()
}

// ------------------------------------------------------------ translate: validate the translation
/// The narration lines with their beat times, as sent to dub: each has a `speaker`, its `text` and the beat's `t0`/`dur`.
pub type Line = Value;

/// A translation must have one line per narration beat, speakers unchanged in order, no empty text. Rejects missing or
/// changed lines (the count first, then the speaker of each line). Used by callers that pass a reference narration.
pub fn validate(lines: &[Value], source_speakers: &[&str]) -> Result<(), String> {
    if lines.len() != source_speakers.len() {
        return Err(format!("missing line(s): expected {} lines, got {}", source_speakers.len(), lines.len()));
    }
    for (i, x) in lines.iter().enumerate() {
        let spk = x.get("speaker").and_then(Value::as_str).unwrap_or("");
        if spk != source_speakers[i] { return Err(format!("line {}: speaker changed: {} != {}", i + 1, spk, source_speakers[i])); }
        let txt = x.get("text").and_then(Value::as_str).unwrap_or("").trim();
        if txt.is_empty() { return Err(format!("line {}: text is empty", i + 1)); }
    }
    Ok(())
}

/// The (t0, dur) of every narration line in order — the kind=="line" beats' times, the slot each take must fit.
pub fn line_slots(beats: &[Value]) -> Vec<(f64, f64)> {
    beats.iter().filter(|b| b.get("kind").and_then(Value::as_str) == Some("line"))
        .map(|b| (b["t0"].as_f64().unwrap_or(0.0), b["dur"].as_f64().unwrap_or(0.0))).collect()
}

/// The worst characters/second across the lines against their slots — a long line's signal to shorten it (dub.py's
/// translate table). Only the worst is returned so a caller can render the per-line table itself.
pub fn worst_cps(lines: &[Line], slots: &[(f64, f64)]) -> f64 {
    lines.iter().enumerate().map(|(i, x)| {
        let dur = slots.get(i).map(|(_, d)| *d).unwrap_or(0.0);
        let ch = x.get("text").and_then(Value::as_str).unwrap_or("").chars().count();
        if dur > 0.0 { ch as f64 / dur } else { 0.0 }
    }).fold(0.0f64, |a, b| if a > b { a } else { b })
}

// ------------------------------------------------------------------------- speak: seed retry + pause cap
/// A take recorded for one line, at the narration rate after any resampling.
pub struct Take { pub speaker: String, pub samples: Vec<f32> }

/// Trim leading/trailing silence and fade each new end down to zero so a take of any length opens and closes on silence
/// instead of clicking (dub.py's trim_silence). `x` longer than the loud body is shrunk; a one-sample input is kept.
pub fn trim_silence(x: &[f32]) -> Vec<f32> {
    if x.is_empty() { return Vec::new(); }
    let hop = (SR / 100) as usize;
    let y = &x[..x.len() / hop * hop];
    let peak = x.iter().cloned().fold(0f32, |m, s| m.max(s.abs()));
    let thr = if peak > 1e-4 { peak * 10f32.powf(-40.0 / 20.0) } else { 1e-4 };
    let rms: Vec<f32> = y.chunks(hop).map(|c| (c.iter().map(|&s| s * s).sum::<f32>() / c.len() as f32).sqrt()).collect();
    if rms.is_empty() || rms.iter().all(|&r| r <= thr) { return x.to_vec(); }
    let mut first = 0usize;
    while first < rms.len() && rms[first] <= thr { first += 1; }
    let mut last = rms.len().saturating_sub(1);
    while last > first && rms[last] <= thr { last -= 1; }
    let a = y[first * hop..last * hop + hop].to_vec();
    let n = a.len();
    if n < 2 { return a; }
    let fh = (n / 40).max(2).min(200);
    if 2 * fh <= n {
        let w: Vec<f32> = (0..fh).map(|k| {
            let phase = std::f32::consts::PI * (k as f32 / fh as f32);   // raised cosine across the fade, 0 -> pi
            0.5f32 * (1.0 - phase.cos())                               // w[0] == 0, w[fh-1] == 1
        }).collect();
        let mut out = a.clone();
        for k in 0..fh { out[k] *= w[k]; }               // head: out[0] *= w[0] == 0
        for k in 0..fh { out[n - 1 - k] *= w[k]; }        // tail: out[last] *= w[0] == 0, the fade-in end untouched
        out
    } else {
        let mut out = a;
        if !out.is_empty() { out[0] = 0.0; }
        let n = out.len();
        if n > 1 { out[n - 1] = 0.0; }
        out
    }
}

/// The longest interior pause in `data` (a WAV), measured the way task 5's check_media does — a run of frames more than
/// GAP_DB under the loudest, between the first and last loud frame, at a 10 ms window. A take's pauses are capped to this
/// before it is fitted; here it reports what remains (<= DEFAULT_MAX_S after capping). Reuses task 5.
pub fn interior_pause_s(data: &[u8]) -> Option<f64> {
    let (_, samples) = read_wav(data).ok()?;
    Some(crate::media_check::longest_pause_s(&samples, SR, crate::media_check::GAP_DB))
}

/// The outcome of taking one line with the speech model (dub.py's speak_stage, the seed loop): which seed was kept, its
/// speed-up to the slot, and why. A take that fits within MAX_TEMPO wins on the seed it first fitted; when none fit the
/// SHORTEST take is kept and reported TOO LONG with ITS seed.
#[derive(Debug)]
pub enum SpeakResult {
    Fit { used_seed: usize, tempo: f64 },
    TooLong { seed: usize, seconds: f64 },
    Error(String),
}

/// Take one line through the speech model up to six seeds (an initial take plus five retries). `say` returns a fresh
/// take's bytes each seed (the file is overwritten, so a long seed 0 does not sit under a later short one); it is read,
/// trimmed and its length measured before fitting. When none of the six fit within MAX_TEMPO the shortest taken is kept.
pub fn speak_run<F>(slot_dur: f64, mut say: F) -> SpeakResult
where F: FnMut(usize) -> Result<Vec<u8>, String> {
    let mut best: Option<(f64, usize)> = None;   // (seconds, seed) of the shortest take seen so far
    for seed in 0..6 {
        let data = match say(seed) { Ok(d) => d, Err(e) => return SpeakResult::Error(e.to_string()) };
        let (_, samples) = match read_wav(&data) { Ok(v) => v, Err(e) => return SpeakResult::Error(e) };
        let sec = trim_silence(&samples).len() as f64 / SR as f64;   // measured after trimming (and the pause cap)
        if best.is_none_or(|(bs, _)| sec < bs) { best = Some((sec, seed)); }
        if let Some(tempo) = fit_take(sec, slot_dur) {             // fits within the 12% ceiling -> stop here
            return SpeakResult::Fit { used_seed: seed, tempo };
        }
    }
    // none of the six fitted within MAX_TEMPO: keep the shortest take, report it TOO LONG with its own seed
    let (seconds, seed) = best.unwrap_or((0.0, 5));
    SpeakResult::TooLong { seed, seconds: r3(seconds) }
}

/// Playback tempo (>=1) to fit a take into its slot without exceeding MAX_TEMPO; None means TOO LONG over the 12% ceiling
/// (dub.py's fit_take). A take already at or under the slot needs no speed-up (returns 1.0).
pub fn fit_take(take_dur: f64, slot_dur: f64) -> Option<f64> {
    if take_dur <= slot_dur { return Some(1.0); }
    let t = take_dur / slot_dur;
    if t > MAX_TEMPO { None } else { Some(r3(t)) }
}

// -------------------------------------------------------------------- mix: place takes, fit or cut a line
/// One thing done to a line in the mix that had to change: sped up to fit its window, cut (faded out) because it could not
/// or kept silent with music only because no take was provided. Listed in the report.
#[derive(Debug, Clone, PartialEq)]
pub enum LineAction { SpeedUp(f64), Cut(String), Silent }

/// The tempo to fit a take into its window without exceeding MIX_MAX_TEMPO; None means it does not fit at <= 2x (→ CUT).
fn fit_for_window(take_dur: f64, window_s: f64) -> Option<f64> {
    if take_dur <= window_s { return None; }
    let t = take_dur / window_s;
    if t > MIX_MAX_TEMPO { None } else { Some(r3(t)) }
}

/// Build a full-length soundtrack of the original length with `takes` placed at their beats over the other `stems`. The
/// result is exactly `out_len` samples long (the longer of the stems and the narration timeline). A take that fits its
/// window (slot + GAP_S) is left as-is; one longer than it is sped up to fit when that needs MIX_MAX_TEMPO or less, else
/// cut over its last 0.3 s; a missing take keeps silence (music only). Returns the mixed mono samples and the report.
pub fn render_mix(beats: &[(f64, f64)], stems: Option<Vec<u8>>, takes: &[Option<Take>]) -> (Vec<f32>, Vec<(usize, LineAction)>) {
    let timeline = beats.iter().map(|(t0, dur)| max_f(*t0 + dur, 0.0)).fold(0.0f64, max_f);
    let stem_s = stems.as_ref().and_then(|b| read_wav(b).ok()).map(|(_, s)| s.len() as f64 / SR as f64).unwrap_or(0.0);
    let out_len = (max_f(stem_s, timeline) * SR as f64 + 0.5) as usize;
    let mut mix: Vec<f32> = vec![0f32; out_len];
    if let Some(b) = stems {
        if let Ok((_, s)) = read_wav(&b) {
            for (i, &v) in s.iter().enumerate() { if i < out_len { mix[i] += v; } }
        }
    }
    let mut report = Vec::new();
    for (li, ((t0, dur), take)) in beats.iter().zip(takes).enumerate() {
        let window_s = max_f(*dur + GAP_S, 0.0);              // slot + the gap before the next beat
        let start = (*t0 * SR as f64) as usize;
        let win = (window_s * SR as f64) as usize;            // width a take has to fit in (slot + the gap)
        match take {
            None => report.push((li + 1, LineAction::Silent)),
            Some(tk) => {
                let raw = resample(&tk.samples, SR);           // at the narration rate when it was already 48 kHz
                let dur_s = raw.len() as f64 / SR as f64;
                let (placed, action): (Vec<f32>, Option<LineAction>) = if dur_s <= window_s {
                    (raw.clone(), None)                      // fits the window as-is, left untouched
                } else if let Some(ratio) = fit_for_window(dur_s, window_s) {   // sped up to fit
                    match apply_tempo(&wav_encode(&raw), ratio) {
                        Ok(a) => (a[..a.len().min(win)].to_vec(), Some(LineAction::SpeedUp(ratio))),
                        Err(_) => (trim_to(&raw, win), None),  // could not speed it: fall through to the cut below
                    }
                } else { (trim_to(&raw, win), None) };         // too long even at the ceiling → cut
                if action.is_none() && dur_s > window_s {
                    place_cut(&mut mix, start, &trim_to(&raw, win));   // keep only the first `win` seconds, faded out
                    report.push((li + 1, LineAction::Cut(format!("{:.2}fs lost", r3(dur_s - window_s)))));
                } else {
                    place(&mut mix, start, &placed);
                }
                if let Some(a) = action { report.push((li + 1, a)); }
            }
        }
    }
    (mix, report)
}

/// Keep only the first `win` samples of `raw` (a take longer than its window: the tail that did not make it is dropped).
fn trim_to(raw: &[f32], win: usize) -> Vec<f32> { raw[..raw.len().min(win)].to_vec() }

/// Place a cut take's first `win` seconds into `mix`, then fade its last 0.3 s down to silence so the line ends quiet.
fn place_cut(mix: &mut [f32], start: usize, placed: &[f32]) {
    place(mix, start, placed);
    let n = CUT_FADE_S as usize * SR as usize;               // 0.3 s in samples
    let mut k = 0usize;
    while k < n && start + placed.len() - 1 - k >= start {   // fade the last `n` samples of the placed region
        mix[start + placed.len() - 1 - k] *= 0.5f32 * (1.0 - (std::f32::consts::PI * k as f32 / n.max(1) as f32)).cos();
        k += 1;
    }
}

/// Place `src` into `mix` at `start`, clipping to the buffer and summing additively (the other stems stay level).
fn place(mix: &mut [f32], start: usize, src: &[f32]) {
    for (i, &v) in src.iter().enumerate() {
        if let Some(slot) = mix.get_mut(start + i) { *slot += v; }
    }
}

/// Speed `x` up by `tempo` with ffmpeg's atempo, re-sampled back to the narration rate (dub.py's apply_tempo).
pub fn apply_tempo(data: &[u8], tempo: f64) -> Result<Vec<f32>, String> {
    if !crate::media::ffmpeg_available() { return Err("this server cannot change speed: install ffmpeg".into()); }
    let dir = std::env::temp_dir().join(format!("strata-dub-{}-{}", std::process::id(), uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{e:#}"))?;
    let tmp = dir.join("_tmp.wav");
    let out = tmp.join("fast.wav");
    std::fs::write(&tmp, data).map_err(|e| format!("{e:#}"))?;
    let status = Command::new("ffmpeg").args(["-y", "-loglevel", "error", "-i"]).arg(&tmp)
        .args(["-af", &format!("atempo={:.5}", tempo)]).arg(&out).status().map_err(|e| format!("{e:#}"))?;
    if !status.success() { let _ = std::fs::remove_dir_all(&dir); return Err("ffmpeg atemper did not run".into()); }
    let bytes = std::fs::read(&out).map_err(|e| format!("{e:#}"))?;
    let _ = std::fs::remove_dir_all(&dir);
    read_wav(&bytes).map(|(_, s)| s)
}

fn wav_encode(x: &[f32]) -> Vec<u8> { write_wav(x) }

// ------------------------------------------------------------------------- hear: reuse check_media for words_heard
/// Share of the expected words heard in the transcript, via task 5's Unicode-aware matcher (Hindi folded). None when the
/// expected text has no words (dub.py's word_match_rate, reusing media_check).
pub fn words_heard(expected: &str, heard: &str, lang: &str) -> Option<f64> {
    crate::media_check::word_match_rate(expected, heard, lang)
}

// ------------------------------------------------------------- the endpoint + tool (JSON <-> report)
fn bad(msg: impl Into<String>) -> Response { error(StatusCode::BAD_REQUEST, "invalid_request_error", msg) }

/// The dub request: narration lines with their beat times, the takes to lay over them (optional — a missing take keeps
/// that line silent), the other stems (the music etc., which set the film's length), an optional per-line heard transcript
/// for the words-heard check and the target language. A line is `{"speaker","text","t0","dur"}`; an optional `source_lines`
/// array can validate against a reference narration (speakers unchanged in order, no empty text).
#[derive(Clone)]
pub struct DubReq { pub lines: Vec<Line>, pub takes: Option<Vec<String>>, pub stems: Option<String>, pub heard: Option<Vec<String>>, pub source: Option<Vec<Line>>, pub lang: String }

fn parse_req(req: &Value) -> Result<DubReq, String> {
    let arr = req.get("lines").and_then(Value::as_array).ok_or_else(|| "dub: \"lines\" must be an array".to_string())?;
    let mut lines = Vec::with_capacity(arr.len());
    for (i, l) in arr.iter().enumerate() {
        let txt = l.get("text").and_then(Value::as_str).unwrap_or("").trim().to_string();
        let spk = l.get("speaker").and_then(Value::as_str).unwrap_or("").to_string();
        if txt.is_empty() { return Err(format!("dub: line {} text is empty", i + 1)); }
        lines.push(json!({"speaker": spk, "text": txt,
            "t0": l.get("t0").or_else(|| l.get("at")).and_then(Value::as_f64).unwrap_or(0.0),
             "dur": l.get("dur").or_else(|| l.get("length")).and_then(Value::as_f64).unwrap_or(0.0)}));
    }
    let takes = req.get("takes").and_then(Value::as_array).map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect());
    let stems = req.get("stems").or_else(|| req.get("music")).and_then(Value::as_str).map(str::to_string);
    let heard = req.get("heard").or_else(|| req.get("transcripts")).and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect());
    let source = req.get("source_lines").or_else(|| req.get("source")).and_then(Value::as_array)
        .map(|a| a.iter().cloned().collect());
    let lang = req.get("language").or_else(|| req.get("lang")).and_then(Value::as_str).unwrap_or("en").to_lowercase();
    Ok(DubReq { lines, takes, stems, heard, source, lang })
}

/// Load one take's or the stem's audio (base64/data URL, http(s) URL or a local file) into bytes.
async fn load_audio(_s: &AppState, src: &str) -> Result<Vec<u8>, String> { crate::media::load_source(src).await.map_err(|e| format!("{e:#}")) }

/// The whole dub for the endpoint (no STT): validate when a reference was given, then place the takes at their beats. When
/// `hear` is true and a per-line heard transcript is given, fill words_heard via task 5's check_media for each take. Returns
/// the report plus the soundtrack as a base64 data URL of the original length.
async fn build(s: &AppState, req: &Value, hear: bool) -> Result<Value, String> {
    let d = parse_req(req)?;
    let slots = line_slots(&d.lines);
    if let Some(source) = &d.source {
        let ref_speakers: Vec<&str> = source.iter().map(|l| l.get("speaker").and_then(Value::as_str).unwrap_or("")).collect();
        validate(&d.lines, &ref_speakers)?;
    }

    // take bytes per line (None when not provided) — fetched before the mix so one bad take is a clear error
    let mut takes: Vec<Option<Take>> = Vec::with_capacity(d.lines.len());
    if let Some(urls) = &d.takes {
        for (i, url) in urls.iter().enumerate() {
            let b = load_audio(s, url).await?;
            let (_, samples) = read_wav(&b)?;
            takes.push(Some(Take { speaker: d.lines[i].get("speaker").and_then(Value::as_str).unwrap_or("").to_string(), samples }));
        }
    }
    let stems = match &d.stems { Some(u) => Some(load_audio(s, u).await?), None => None };

    let (mix, actions) = render_mix(&slots, stems, &takes);

    // words_heard per take with check_media (reused), when asked and a transcript is given
    let mut report_actions: Vec<Value> = Vec::new();
    for (n, a) in actions.iter().enumerate() {
        let heard_rate = if hear {
            match (takes.get(n).and_then(|t| t.as_ref()), &d.heard, s.mgr.media_check.get()) {
                (Some(tk), Some(transcripts), Some(check)) if n < transcripts.len() && !transcripts[n].is_empty() => {
                    let audio_url = format!("data:audio/wav;base64,{}", base64::engine::general_purpose::STANDARD.encode(write_wav(&tk.samples)));
                    check.call(&json!({"audio": audio_url, "expected_text": d.lines[n]["text"], "language": d.lang,
                        "heard_text": transcripts[n]})).await.ok().and_then(|r| r["audio"]["words_heard"]["rate"].as_f64())
                }
                _ => None,
            }
        } else { None };
        let (what, note) = match a { (_, LineAction::Silent) => ("kept silent", "".to_string()), (_, LineAction::SpeedUp(r)) => ("sped up", format!("x{}", r)),
            (_, LineAction::Cut(s)) => ("cut", s.clone()) };
        report_actions.push(json!({"line": n + 1, "action": what, "note": note, "words_heard": heard_rate}));
    }

    let wav = write_wav(&mix);
    let soundtrack = format!("data:audio/wav;base64,{}", base64::engine::general_purpose::STANDARD.encode(&wav));
    let report = json!({"lines": d.lines.len(), "worst_cps": r3(worst_cps(&d.lines, &slots)), "actions": report_actions});
    Ok(json!({"soundtrack": soundtrack, "seconds": r3(mix.len() as f64 / SR as f64), "samples": mix.len(), "report": report}))
}

/// POST /v1/media/dub {"lines":[...], "takes":[...], "stems":..., "heard":[...], "source_lines":[...], "language":"de"}.
/// Returns the full-length soundtrack WAV as a data URL and one report entry per line that was sped up, cut or kept silent.
/// `hear` fills words_heard via check_media (task 5) for each take with a given transcript.
pub async fn dub_endpoint(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !api_authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    match build(&s, &req, true).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => bad(e),
    }
}

/// The `dub_narration` chat tool handle (holds the app state so it can run check_media for words_heard).
pub struct Dubbing { app: Arc<AppState> }

impl Dubbing {
    pub fn new(app: Arc<AppState>) -> Self { Dubbing { app } }

    /// Run one dub from a tool call's arguments (the narration lines, takes, stems and language), returning the report and
    /// a short note. The translation must be shown to the user before voices are made — this mixes what it is given and
    /// lists every line that had to change; the speaking of each line with the speak tool happens first, so their bytes can
    /// be passed as `takes`.
    pub async fn call(&self, args: &Value) -> Result<Value, String> {
        let out = build(&self.app, args, true).await?;
        let report = &out["report"];
        let sped = count_actions(report, "sped up");
        let cut = count_actions(report, "cut");
        let silent = count_actions(report, "kept silent");
        let note = format!("dubbed {} line(s); the soundtrack is the original length ({:.1} s): {} sped up to fit, {} cut, \
                            {} kept silent with only the stems.", report["lines"].as_u64().unwrap_or(0),
                            out["seconds"].as_f64().unwrap_or(0.0), sped, cut, silent);
        Ok(json!({"ok": true, "text": note, "report": out["report"], "soundtrack": out["soundtrack"], "seconds": out["seconds"]}))
    }
}

fn count_actions(report: &Value, action: &str) -> usize {
    report["actions"].as_array().map(|a| a.iter().filter(|x| x["action"] == action).count()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f32, dur_s: f64, amp: f32) -> Vec<f32> {
        (0..(dur_s * SR as f64) as usize).map(|i| amp * (2.0 * std::f32::consts::PI * freq * (i as f32 / SR as f32)).sin()).collect()
    }

    fn take(spk: &str, dur_s: f64, samples: &[f32]) -> Option<Take> {
        Some(Take { speaker: spk.to_string(), samples: samples.to_vec() })
    }

    /// A talk tone of `dur_s` seconds as a WAV (the takes seed-retry records).
    fn talk(dur_s: f64) -> Vec<u8> { write_wav(&tone(220.0, dur_s, 0.8)) }

    fn beats_from(lines: &[Value]) -> Vec<(f64, f64)> { line_slots(lines).into_iter().collect() }

    // -------------------------------------------------------------- the fit rule (12% ceiling)
    #[test]
    fn fit_rule_cap_is_twelve_percent() {
        assert_eq!(fit_take(10.0, 9.2), Some(1.087), "a 10 s take in a 9.2 s slot gets tempo round(10/9.2,3)");
        assert_eq!(fit_take(10.0, 8.0), None, "over the 12% ceiling -> TOO LONG");
        assert_eq!(fit_take(9.0, 10.0), Some(1.0));
        assert_eq!(fit_take(10.0, 10.0), Some(1.0));
    }

    // -------------------------------------------------------------- trim keeps the voice
    #[test]
    fn trim_keeps_voice_and_ends_on_silence() {
        let mut x = vec![0f32; (0.5 * SR as f64) as usize];      // lead silence
        x.extend_from_slice(&tone(220.0, 1.0, 0.8));             // voiced body
        x.extend_from_slice(&vec![0f32; (0.7 * SR as f64) as usize]);  // tail silence
        let out = trim_silence(&x);
        let voiced = out.iter().any(|&s| s.abs() > 1e-3);
        assert!(voiced, "trim keeps the voice");
        assert_eq!(out[0], 0.0, "the fade starts at silence (would click otherwise)");
        assert_eq!(out[out.len() - 1], 0.0, "the fade ends at silence (would click otherwise)");
        assert!(out.len() < x.len(), "silence was trimmed ({}/{} samples)", out.len(), x.len());
    }

    // -------------------------------------------------------------- Bug 1: seed retry
    #[test]
    fn seed_retry_remeasures_a_fresh_take_each_seed() {
        // LONG take for seed 0 (4.5 s -> over the 12% ceiling of a 3.0 s slot), SHORT ones afterwards
        let long = talk(4.5);
        let short = talk(0.5);
        let mut i = 0usize;
        let say = |_seed: usize| { let r = if i == 0 { long.clone() } else { short.clone() }; i += 1; Ok(r) };
        assert!(matches!(speak_run(3.0, say), SpeakResult::Fit { used_seed: 1, .. }), "seed 0 long then seed 1 short -> OK with seed 1");
    }

    #[test]
    fn seed_retry_keeps_the_shortest_when_none_fit() {
        // every take too long for a 3.0 s slot at the 12% ceiling; seed 2 (3.5 s) is the shortest
        let say = |seed: usize| Ok(talk([5.0, 4.0, 3.5, 4.5][seed % 4]));
        match speak_run(3.0, say) {
            SpeakResult::TooLong { seed, seconds } => assert_eq!(seed, 2, "the shortest take (seed 2) is the one kept"),
            other => panic!("expected TOO LONG, got {other:?}"),
        }
    }

    // -------------------------------------------------------------- the mix places takes at their beats
    #[test]
    fn english_narration_is_placed_exactly_like_the_film() {
        // two back-to-back slots with exactly GAP_S between them; the English takes are each slot-long, tempo 1.0
        let lines = vec![json!({"kind":"line","t0":0.0,"dur":2.0}), json!({"kind":"line","t0":2.2,"dur":1.5})];
        let slots = beats_from(&lines);
        let stems_wav = write_wav(&tone(110.0, 3.7, 0.05));      // the other stems set the film's length (3.7 s)
        let take0 = take("laura", 2.0, &tone(220.0, 2.0, 0.8));
        let take1 = take("gautham", 1.5, &tone(165.0, 1.5, 0.8));

        // build the expected soundtrack by placing exactly what render_mix must do, independently of it
        let out_len = (3.7 * SR as f64) as usize;
        let mut expected: Vec<f32> = vec![0f32; out_len];
        for (i, &v) in tone(110.0, 3.7, 0.05).iter().enumerate() { if i < out_len { expected[i] += v; } }
        let mut start1 = (2.2 * SR as f64) as usize;
        for (i, &v) in take0.as_ref().unwrap().samples.iter().enumerate() { if i < out_len { expected[i] += v; } }
        for (i, &v) in take1.as_ref().unwrap().samples.iter().enumerate() { let j = start1 + i; if j < out_len { expected[j] += v; } }

        let (mix, report) = render_mix(&slots, Some(stems_wav), &[take0, take1]);
        assert_eq!(mix.len(), out_len, "the soundtrack is the original length");
        for (a, b) in mix.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3, "placed take differs from where it lands ({:.6} vs {:.6})", a, b);
        }
        assert!(report.is_empty(), "both takes fit, so nothing is reported: {report:?}");
    }

    #[test]
    fn full_length_matches_the_stems_even_when_a_take_is_missing() {
        let lines = vec![json!({"kind":"line","t0":0.0,"dur":1.0}), json!({"kind":"line","t0":1.2,"dur":1.0})];
        let slots = beats_from(&lines);
        let stems_wav = write_wav(&tone(120.0, 3.0, 0.1));       // the film is 3.0 s of stems (music only)
        let (mix, report) = render_mix(&slots, Some(stems_wav), &[None, None]);
        assert_eq!(mix.len(), (3.0 * SR as f64) as usize, "the soundtrack is the original length");
        assert!(report.iter().all(|(_, a)| matches!(a, LineAction::Silent)), "both lines kept silent with music only: {report:?}");
    }

    #[test]
    fn mix_cuts_a_too_long_take_and_keeps_the_length() {
        // one slot of 11.8 s -> window 12.0; a 30 s take needs 30/12 = 2.5 (over the 2x ceiling) -> CUT, no ffmpeg needed
        let lines = vec![json!({"kind":"line","t0":0.0,"dur":11.8})];
        let slots = beats_from(&lines);
        let win = 12.0;
        assert!(30.0 / win > MIX_MAX_TEMPO, "precondition: 30 s in a {} s window is over the ceiling", win);
        let stems_wav = write_wav(&tone(120.0, 12.0, 0.1));
        let (_, report) = render_mix(&slots, Some(stems_wav), &[take("laura", 30.0, &tone(220.0, 30.0, 0.8))]);
        let out_len = (12.0 * SR as f64) as usize;
        assert_eq!(report.len(), 1);
        if let LineAction::Cut(note) = &report[0].1 { assert_eq!(note, "18.00fs lost", "the lost tail is reported: {note:?}") }
        else { panic!("expected CUT, got {:?}", report[0].1); }
    }

    #[test]
    fn mix_speds_up_a_take_that_fits_within_the_ceiling() {
        // 20 s in a 12 s window needs 1.67x (<= 2x) -> SPEED UP with ffmpeg's atempo; skipped without ffmpeg
        if !crate::media::ffmpeg_available() { eprintln!("dub: skip speed-up test, no ffmpeg"); return; }
        let lines = vec![json!({"kind":"line","t0":0.0,"dur":11.8})];
        let slots = beats_from(&lines);
        let win = 12.0;
        assert!(20.0 / win <= MIX_MAX_TEMPO, "precondition: 20 s in a {} s window is within the ceiling", win);
        let stems_wav = write_wav(&tone(120.0, 12.0, 0.1));
        let (_, report) = render_mix(&slots, Some(stems_wav), &[take("laura", 20.0, &tone(220.0, 20.0, 0.8))]);
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].1, LineAction::SpeedUp(1.667), "the 20 s take is sped up x1.667, got {:?}", report[0].1);
    }

    #[test]
    fn atempo_speeds_up_to_the_ratio() {
        if !crate::media::ffmpeg_available() { eprintln!("dub: skip atempo test, no ffmpeg"); return; }
        let x = tone(220.0, 2.0, 0.8);
        let fast = apply_tempo(&wav_encode(&x), 1.087).expect("atempo runs");
        let ratio = fast.len() as f64 / x.len() as f64;
        assert!((ratio - 1.0 / 1.087).abs() < 0.02, "atempo 1.087 shrinks 2.0 s to ~1.839 s (got {})", fast.len() as f64 / SR as f64);
    }

    // -------------------------------------------------------------- translate: validation + slots
    #[test]
    fn validate_rejects_a_missing_line() {
        let source = vec!["laura", "gautham"];
        let lines = vec![json!({"speaker":"laura","text":"hi"}), json!({"speaker":"gautham","text":"bye"})];
        assert!(validate(&lines, &source).is_ok(), "two matching lines validate");
        let bad = vec![json!({"speaker":"laura","text":"hi"})];       // missing a line
        assert!(validate(&bad, &source).is_err(), "a file with a missing line is rejected");
    }

    #[test]
    fn validate_rejects_a_changed_speaker_and_empty_text() {
        let source = vec!["laura", "gautham"];
        let bad_speaker = vec![json!({"speaker":"jon","text":"hi"}), json!({"speaker":"gautham","text":"bye"})];
        assert!(validate(&bad_speaker, &source).is_err(), "a changed speaker is rejected");
        let empty = vec![json!({"speaker":"laura","text":"hi"}), json!({"speaker":"gautham","text":"   "})];
        assert!(validate(&empty, &source).is_err(), "an empty text is rejected");
    }

    #[test]
    fn line_slots_are_the_line_beats_in_order() {
        let beats = vec![json!({"kind":"music","t0":0.0,"dur":3.0}), json!({"kind":"line","t0":3.0,"dur":1.5}),
            json!({"kind":"title","t0":4.5,"dur":0.5}), json!({"kind":"line","t0":5.0,"dur":2.0})];
        assert_eq!(line_slots(&beats), vec![(3.0, 1.5), (5.0, 2.0)], "only the kind==\"line\" beats");
    }

    #[test]
    fn worst_cps_returns_the_busiest_line() {
        let lines = vec![json!({"text":"a"}), json!({"text":"abcdefghijklmno"})];   // 1 char, then 15 chars
        assert_eq!(worst_cps(&lines, &[(0.0, 3.0), (4.0, 3.0)]), r3(15.0 / 3.0), "the busiest line over its slot");
    }

    // -------------------------------------------------------------- hear reuses check_media for words_heard
    #[test]
    fn words_heard_is_unicode_and_folded_for_hindi() {
        assert_eq!(words_heard("café déjà vu", "café déjà vu", "fr"), Some(1.0), "French accents counted whole");
        assert_eq!(words_heard("niño rojo", "niño rojo", "es"), Some(1.0), "Spanish ñ counted whole");
        let ref_word = "\u{0917}\u{094d}\u{0930}\u{093e}\u{092b}\u{093c}\u{093f}\u{0915}";   // ग़
        assert_eq!(words_heard(ref_word, "\u{0917}\u{094d}\u{0930}\u{093e}\u{092b}\u{093f}\u{0915}", "hi"), Some(1.0),
            "Hindi differing only in nukta counts as fully heard");
    }

    #[test]
    fn interior_pause_reuses_check_media_definition() {
        let mut x = tone(220.0, 0.5, 0.8);                        // voiced ... 2 s silence ... voiced
        x.extend(std::iter::repeat(0f32).take((2.0 * SR as f64) as usize));
        x.extend(tone(220.0, 0.5, 0.8));
        let data = write_wav(&x);
        let p = interior_pause_s(&data).expect("the WAV is read");
        assert!((p - 2.0).abs() < 0.1, "a 2 s interior pause measures ~2 s (got {p})");
    }

    #[test]
    fn trim_silence_peaks_on_the_absolute_value_even_when_negative() {
        // input: near-silence (-0.005) for the first ~8/10 s, then a loud NEGATIVE spike (-1.0). Its true peak magnitude is
        // 1.0 (the most-negative sample), so the -40 dB threshold clears the early quiet and only the spike survives: if the
        // peak were the raw max value instead of its absolute value it would be ~-0.005 and that quiet would wrongly remain.
        let mut x = Vec::new();
        for _ in 0..8 { x.extend(std::iter::repeat(-0.005f32).take(480)); }   // 7/10 s of near-silence (frames)
        for _ in 0..2 { x.extend(std::iter::repeat(-1.0f32).take(480)); }      // 2/10 s loud negative peak
        let kept = trim_silence(&x);
        assert!(!kept.is_empty(), "keeps the loud end");
        assert!(kept.iter().map(|&s| s.abs()).fold(0f32, f32::max) > 0.9, "the largest-magnitude (negative) sample survives");
        assert!(kept.len() < x.len() / 2, "only the loud end remains ({}/{} frames: early quiet is cleared)", kept.len(), x.len());
    }
}
