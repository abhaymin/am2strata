//! Media quality checks the chat can run — `POST /v1/media/check` and the `check_media` tool.
//!
//! Ported from the reviewer's Python in `.agent/ref/` (std-only, unit-tested on synthetic data):
//! - **audio** (a WAV; 16-bit PCM is what narration takes are, 8-bit PCM and float are read too):
//!   - `longest_pause` — the longest run of frames more than GAP_DB under the loudest frame, between the first and
//!     last such frame, at a 10 ms frame (capgaps.py's `_longest_gap`, the reviewer's pause definition);
//!   - `clipped` — the fraction of samples pinned near full scale;
//!   - `ending` — how loud the final window still is vs the whole clip (cut off mid-voice stays loud, a fade ends quiet);
//!   - `words_heard` — the share of the expected text's words the speech model heard (dub.py's `word_match_rate`,
//!     Unicode `\w+` on both sides and Hindi nukta/chandrabindu folded so Whisper's spelling still matches).
//! - **video** (each frame decoded to 160x90 grey with ffmpeg):
//!   - `jumps` — an isolated single-frame spike over JUMP_FLOOR grey levels and this x taller than both neighbours
//!     (scan_jumps.py, the reviewer's broken-crossfade rule);
//!   - `near_static` — a frozen/near-static hold longer than NEAR_MAX_S;
//!   - `length` — the clip's length from its frame count and fps.
//!
//! Every check returns its number plus an ok/boolean and the threshold used, so a line can see *why* it passed.

use crate::{api_authorized, error, AppState};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine as _;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::Read;
use std::sync::Arc;

pub const TOOL_NAMES: [&str; 1] = ["check_media"];

/// A frame is "loud" when it is more than this many dB under the loudest frame (capgaps.py's GAP_DB).
const GAP_DB: f32 = 30.0;
/// The analysis frame length: a 10 ms window (capgaps.py's FS), used for pause and near-static seconds.
const FRAME_S: f64 = 0.01;
/// The longest interior pause allowed, in seconds (dub.py caps pauses inside a take at this).
const DEFAULT_MAX_S: f64 = 0.25;
/// A clip with more than this fraction of samples pinned near full scale is flagged as clipped.
const CLIP_TOL: f64 = 0.01;
/// The window still louder than this x the clip's peak means it ends mid-voice (cut off) rather than fading out.
const ENDING_FLOOR: f32 = 0.2;
/// A share of the expected words heard at or above this passes the word check (dub.py marks below it "LOW WORDS").
const WORDS_OK: f64 = 0.6;
/// The reviewer's scan resolution for video (scan_jumps.py).
const VIDEO_W: usize = 160;
const VIDEO_H: usize = 90;
/// A frame jump is more than this many grey levels (scan_jumps.py's ABS_FLOOR, the reviewer said >8).
const JUMP_FLOOR: f64 = 8.0;
/// A jump must be at least this x taller than BOTH neighbours to be isolated (the task text says 2.5x).
const JUMP_ISO: f64 = 2.5;
/// Per-frame change under this many grey levels reads as near-static / frozen.
const NEAR_FLOOR: f64 = 0.5;
/// A frozen hold longer than this is flagged (a genuine static shot may be longer).
const NEAR_MAX_S: f64 = 3.0;

fn r3(x: f64) -> f64 { (x * 1000.0).round() / 1000.0 }
fn r2(x: f64) -> f64 { (x * 100.0).round() / 100.0 }

/// Per-check overrides from a request (`POST /v1/media/check`), all optional, the defaults above used otherwise.
#[derive(Clone)]
pub struct Opts {
    pub max_s: f64,
    pub gap_db: f32,
    pub clip_tol: f64,
    pub ending_floor: f32,
    pub words_ok: f64,
    /// per-frame grey-level floor a jump must exceed (scan_jumps)
    pub jump_floor: f64,
    /// isolation: how x taller than both neighbours a jump must be
    pub jump_iso: f64,
    /// per-frame change below which frames read as near-static
    pub near_floor: f64,
    /// longest near-static hold allowed in seconds before it is flagged
    pub near_max_s: f64,
    /// video fps for the length/near-static seconds (probed from the video when known; 24 otherwise)
    pub fps: f64,
}

impl Default for Opts {
    fn default() -> Self {
        Opts { max_s: DEFAULT_MAX_S, gap_db: GAP_DB, clip_tol: CLIP_TOL, ending_floor: ENDING_FLOOR, words_ok: WORDS_OK,
               jump_floor: JUMP_FLOOR, jump_iso: JUMP_ISO, near_floor: NEAR_FLOOR, near_max_s: NEAR_MAX_S, fps: 24.0 }
    }
}

impl Opts {
    fn from_value(v: &Value) -> Self {
        let mut o = Opts::default();
        if let Some(x) = v["max_s"].as_f64() { o.max_s = x; }
        if let Some(x) = v["gap_db"].as_f64() { o.gap_db = x as f32; }
        if let Some(x) = v["clip_tol"].as_f64() { o.clip_tol = x; }
        if let Some(x) = v["ending_floor"].as_f64() { o.ending_floor = x as f32; }
        if let Some(x) = v["words_ok"].as_f64() { o.words_ok = x; }
        if let Some(x) = v["jump_floor"].as_f64() { o.jump_floor = x; }
        if let Some(x) = v["isolation"].as_f64() { o.jump_iso = x; }
        if let Some(x) = v["near_floor"].as_f64() { o.near_floor = x; }
        if let Some(x) = v["near_max_s"].as_f64() { o.near_max_s = x; }
        if let Some(x) = v["fps"].as_f64().filter(|&x| x > 0.0) { o.fps = x; }
        o
    }
}

// ---------------------------------------------------------------------------------------------------------- audio WAV

/// A WAV's sample rate and its downmixed mono samples in [-1, 1] (16-bit PCM, 8-bit PCM and float are read).
pub fn read_wav(data: &[u8]) -> Result<(u32, Vec<f32>), String> {
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("the audio is not a WAV file".into());
    }
    let mut fmt = (0u16, 0u16, 0u32, 0u16); // format tag, channels, rate, bits
    let mut data_start = 0usize;
    let mut data_len = 0usize;
    let mut i = 12;
    while i + 8 <= data.len() {
        let cid = &data[i..i + 4];
        let len = u32::from_le_bytes(data[i + 4..i + 8].try_into().unwrap()) as usize;
        let b = i + 8;
        let e = (b + len).min(data.len());
        if cid == b"fmt " && e - b >= 16 {
            fmt = (u16::from_le_bytes(data[b..b + 2].try_into().unwrap()), u16::from_le_bytes(data[b + 2..b + 4].try_into().unwrap()),
                    u32::from_le_bytes(data[b + 4..b + 8].try_into().unwrap()), u16::from_le_bytes(data[b + 14..b + 16].try_into().unwrap()));
        } else if cid == b"data" {
            data_start = b; data_len = e - b;
        }
        i += 8 + len + (len & 1); // chunks are word-aligned
    }
    let (tag, ch, rate, bits) = fmt;
    if ch == 0 || rate == 0 { return Err("the WAV has no readable format".into()); }
    let ch_u = ch.max(1) as usize;
    let bpp = bits as usize / 8; // bytes per interleaved channel sample
    let bytes_per_frame = ch_u * bpp;
    let frames = data_len.min(data.len() - data_start) / bytes_per_frame;
    let chan = frames * ch_u; // total interleaved channel samples
    let mut flat = Vec::with_capacity(chan);
    match (tag, bits) {
        (1, 16) => {
            for k in 0..chan {
                let o = data_start + k * bpp;
                flat.push(i16::from_le_bytes(data[o..o + 2].try_into().unwrap()) as f32 / 32768.0);
            }
        }
        (1, 8) => {
            for k in 0..chan {
                let o = data_start + k * bpp;
                flat.push((data[o] as i32 - 128) as f32 / 128.0);
            }
        }
        (3, 32) => {
            for k in 0..chan {
                let o = data_start + k * bpp;
                flat.push(f32::from_le_bytes(data[o..o + 4].try_into().unwrap()).clamp(-1.0, 1.0));
            }
        }
        _ => return Err(format!("unsupported WAV format tag {tag}, {bits}-bit (need 16/8-bit PCM or float)")),
    }
    // average the channels of each frame into mono
    let mut out = Vec::with_capacity(frames / ch_u);
    for frame in 0..(flat.len() / ch_u) {
        let base = frame * ch_u;
        let s: f32 = flat[base..base + ch_u].iter().sum::<f32>() / ch_u as f32;
        out.push(s);
    }
    Ok((rate, out))
}

/// Longest run of quiet frames (>GAP_DB under the loudest), between the first and last loud frame, in seconds.
pub fn longest_pause_s(samples: &[f32], rate: u32, gap_db: f32) -> f64 {
    if rate == 0 || samples.is_empty() { return 0.0; }
    let h = (FRAME_S * rate as f64).round() as usize;
    if h == 0 { return 0.0; }
    let n = samples.len() / h;
    if n == 0 { return 0.0; }
    let mut rms = vec![0f32; n];
    for (i, f) in rms.iter_mut().enumerate() {
        let s = &samples[i * h..(i + 1) * h];
        *f = (s.iter().map(|&x| x * x).sum::<f32>() / s.len() as f32).sqrt();
    }
    let peak = rms.iter().cloned().fold(0.0f32, f32::max);
    if peak <= 0.0 { return 0.0; }
    let thr = peak * 10f32.powf(-gap_db / 20.0);
    let loud: Vec<bool> = rms.iter().map(|&r| r > thr).collect();
    let first = loud.iter().position(|&b| b).unwrap_or(0);
    let last = loud.iter().rposition(|&b| b).unwrap_or(0);
    if last <= first { return 0.0; }
    let mut run = 0usize;
    let mut longest = 0usize;
    for i in first..=last {
        if loud[i] { run = 0; } else { run += 1; longest = longest.max(run); }
    }
    longest as f64 * FRAME_S
}

/// Fraction of samples pinned near full scale (clipping).
pub fn clipped_ratio(samples: &[f32]) -> f64 {
    if samples.is_empty() { return 0.0; }
    let c = samples.iter().filter(|&&x| x.abs() > 0.999).count();
    c as f64 / samples.len() as f64
}

/// How loud the last ~1/10 is, as a fraction of the clip's peak (1.0 = cut off mid-voice, <floor = faded out).
pub fn ending_peak_ratio(samples: &[f32]) -> f32 {
    if samples.is_empty() { return 0.0; }
    let overall = samples.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    if overall <= 0.0 { return 0.0; }
    let win = samples.len().max(1) / 10; // last ~1/10 of the clip as the window
    let tail = samples[samples.len().saturating_sub(win)..].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    tail / overall
}

// -------------------------------------------------------------- words heard (Unicode-aware)

/// A char counts as part of a word if Python's `\w+` would: alphanumeric, an underscore, or a combining mark (so a
/// Devanagari conjunct stays one token instead of splitting at the halant/nukta that Whisper writes differently).
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || is_mark(c as u32)
}

/// The Unicode general-category marks (Mn, Md, Me), i.e. combining accents and the halant/nukta of Indic scripts.
fn is_mark(cp: u32) -> bool {
    const RANGES: &[(u32, u32)] = &[
        // Latin combining accents
        (0x0300, 0x036F),
        // Devanagari (Hindi): matra, halant and nukta
        (0x0901, 0x0902), (0x093C, 0x093C), (0x0941, 0x0948), (0x094D, 0x094D),
        (0x0951, 0x095F), (0x0962, 0x0963), (0x09BC, 0x09CD),
        // other Indic scripts' marks
        (0x0981, 0x0981), (0x09BC, 0x09CD), (0x09E2, 0x09E3), (0x0A01, 0x0A02), (0x0A3C, 0x0A3C),
        (0x0A41, 0x0A42), (0x0A47, 0x0A4D), (0x0A70, 0x0A71), (0x0AED, 0x0AEE),
        (0x0B01, 0x0B02), (0x0B3C, 0x0B3C), (0x0B41, 0x0B43), (0x0B4D, 0x0B4D),
        (0x0B82, 0x0B83), (0x0BC0, 0x0BCD), (0x0CE2, 0x0CE3), (0x0D01, 0x0D03),
        (0x0D3C, 0x0D3D), (0x0D4E, 0x0D4E), (0x0DC0, 0x0DCD), (0x0DD2, 0x0DD3),
        // Tibetan / Arabic combining marks
        (0x0F18, 0x0F19), (0x0F35, 0x0F37), (0x0F39, 0x0F39), (0x0670, 0x0670),
        (0x064B, 0x065F), (0x06D6, 0x06DC), (0x06DF, 0x06E4), (0x06E7, 0x06E8), (0x06EA, 0x06ED),
        (0x07EB, 0x07F3), (0x0591, 0x05BD), (0x0610, 0x061A),
    ];
    RANGES.iter().any(|&(s, e)| cp >= s && cp <= e)
}

/// Unicode word tokens: maximal runs of `\w+` chars on both sides, lower-cased (keeps Devanagari and accents whole).
pub fn tokenize_words(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if is_word_char(c) { cur.extend(c.to_lowercase()); }
        else if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}

/// A word compared after this rewrite (Hindi drops the nukta and folds chandrabindu to anusvara; others as-is).
pub fn normalize_word(w: &str, lang: &str) -> String {
    if lang == "hi" { w.replace('\u{093c}', "").replace('\u{0901}', "\u{0902}") } else { w.to_string() }
}

/// Share of `expected`'s words the speech model heard in `heard` (None if `expected` has no words).
pub fn word_match_rate(expected: &str, heard: &str, lang: &str) -> Option<f64> {
    let ref_words = tokenize_words(expected).into_iter().filter(|w| !w.is_empty())
        .map(|w| normalize_word(&w, lang)).collect::<Vec<_>>();
    if ref_words.is_empty() { return None; }
    let heard_set: HashSet<String> = tokenize_words(heard).into_iter().map(|w| normalize_word(&w, lang)).collect();
    let hit = ref_words.iter().filter(|w| heard_set.contains(*w)).count();
    Some(r3(hit as f64 / ref_words.len() as f64))
}

// ----------------------------------------------------------------------------- video frames

/// Mean absolute change in grey levels between consecutive frames (each frame is VIDEO_W*VIDEO_H bytes).
pub fn grey_frame_deltas(frames: &[&[u8]]) -> Vec<f64> {
    let mut d = Vec::new();
    if frames.len() < 2 { return d; }
    let w = VIDEO_W * VIDEO_H;
    for pair in frames.windows(2) {
        let n = w.min(pair[0].len()).min(pair[1].len());
        let sum: f64 = (0..n).map(|k| (pair[0][k] as f64 - pair[1][k] as f64).abs()).sum();
        d.push(sum / n.max(1) as f64);
    }
    d
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

/// A single-frame jump the reviewer flags: an isolated spike over `floor` grey levels and `iso` x taller than both
/// neighbours (not a smooth crossfade ramp). Times it at frame/fps.
pub struct Jump { pub t: f64, pub levels: f64 }

pub fn scan_jumps(deltas: &[f64], fps: f64, floor: f64, iso: f64) -> Vec<Jump> {
    let mut out = Vec::new();
    if deltas.len() < 3 || fps <= 0.0 { return out; }
    let med = median(deltas);
    let thr = floor.max(iso * med);
    for i in 1..deltas.len().saturating_sub(1) {
        let d = deltas[i];
        if d > thr && d >= iso * deltas[i - 1].max(deltas[i + 1]) {
            out.push(Jump { t: (i + 1) as f64 / fps, levels: r3(d) });
        }
    }
    out
}

/// The longest run of near-static frames, as (frame count, seconds).
pub fn near_static_run(deltas: &[f64], fps: f64) -> (usize, f64) {
    let mut run = 0usize;
    let mut longest = 0usize;
    for &d in deltas {
        if d < NEAR_FLOOR { run += 1; longest = longest.max(run); } else { run = 0; }
    }
    // `run` zero-deltas between them means run+1 identical frames were held, capped at the frame count.
    let frames = (longest + 1).min(deltas.len() + 1);
    (frames, frames as f64 / fps.max(0.0001))
}

/// Decode a video's frames to 160x90 grey raw bytes with ffmpeg; returns the frames and their fps.
pub fn video_grey_frames(data: &[u8], fps_override: Option<f64>) -> Result<(Vec<Vec<u8>>, f64), String> {
    if !crate::media::ffmpeg_available() { return Err("this server cannot read video: install ffmpeg (sudo apt install ffmpeg)".into()); }
    let dir = std::env::temp_dir().join(format!("strata-check-{}-{}", std::process::id(), uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{e:#}"))?;
    let in_file = dir.join("in.video");
    std::fs::write(&in_file, data).map_err(|e| format!("{e:#}"))?;
    // probe fps from ffprobe (fall back to the override, then 24)
    let mut fps = fps_override.unwrap_or(24.0);
    if let Ok(p) = std::process::Command::new("ffprobe").args(["-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate"]).arg(&in_file).output() {
        if let Some(s) = String::from_utf8_lossy(&p.stdout).trim().to_string().split('/').next() {
            if let Ok(d) = s.trim().parse::<f64>() { if d > 0.0 { fps = d; } }
        }
    }
    let cmd = std::process::Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(&in_file)
        .args(["-vf", &format!("format=gray,scale={}:{}", VIDEO_W, VIDEO_H), "-f", "rawvideo", "-"])
        .stdout(std::process::Stdio::piped()).spawn()
        .map_err(|e| format!("ffmpeg could not run: {e:#}"))?;
    let mut out = cmd.stdout.expect("capture ffmpeg stdout");
    let nbytes = VIDEO_W * VIDEO_H;
    let mut gray = Vec::new();
    let mut buf = vec![0u8; nbytes];
    loop {
        let n = out.read(&mut buf).map_err(|e| format!("reading ffmpeg frames: {e:#}"))?;
        if n == 0 || n < nbytes { break; }
        gray.push(buf[..nbytes].to_vec());
    }
    let _ = std::fs::remove_dir_all(&dir);
    if gray.is_empty() { return Err("the video gave no frames".into()); }
    Ok((gray, fps))
}

// ------------------------------------------------------------ the check result (no STT needed)

/// Every media check as numbers plus ok/boolean and the threshold used.
pub fn check(audio: Option<&[u8]>, gray: Option<&[Vec<u8>]>, expected: &str, heard: &str, lang: &str, o: &Opts) -> Value {
    let mut audio_obj = json!({});
    if let Some(a) = audio {
        match read_wav(a) {
            Ok((rate, samples)) => {
                let pause = longest_pause_s(&samples, rate, o.gap_db);
                audio_obj["longest_pause"] = json!({"seconds": r3(pause), "ok": pause <= o.max_s, "max_s": r3(o.max_s),
                    "note": format!("frames {} dB under the loudest at {:.0} ms", o.gap_db, FRAME_S * 1000.0)});
                let clip = clipped_ratio(&samples);
                audio_obj["clipped"] = json!({"ratio": r3(clip), "ok": clip <= o.clip_tol, "tol": r3(o.clip_tol)});
                let end = ending_peak_ratio(&samples);
                audio_obj["ending"] = json!({"peak_ratio": r2(end as f64), "ok": (end as f64) <= o.ending_floor as f64,
                    "threshold": r2(o.ending_floor as f64), "note": if end <= o.ending_floor { "ends quiet" } else { "still loud at the end" }});
            }
            Err(e) => audio_obj["error"] = json!(e),
        }
    } else {
        audio_obj["note"] = json!("no audio");
    }
    // words_heard: null when no transcript was provided; otherwise the Unicode-aware match
    if heard.trim().is_empty() {
        audio_obj["words_heard"] = json!({"rate": null, "ok": null, "words_ok": r3(o.words_ok),
            "note": "no transcript (send \"heard_text\", or run /v1/media/check with the audio so Whisper can transcribe)"});
    } else {
        let rate = word_match_rate(expected, heard, lang);
        let n_exp = tokenize_words(expected).len();
        let n_hit = rate.map(|r| (r * n_exp as f64).round() as usize).unwrap_or(0);
        audio_obj["words_heard"] = json!({"rate": rate, "ok": rate.map(|r| r >= o.words_ok), "words_ok": r3(o.words_ok),
            "expected_words": n_exp, "heard_words": n_hit});
    }

    let mut video_obj = json!({});
    if let Some(frames) = gray {
        let fps = o.fps;
        video_obj["length_seconds"] = json!(r3(frames.len() as f64 / fps.max(0.0001)));
        let deltas = grey_frame_deltas(&frames.iter().map(|f| f.as_slice()).collect::<Vec<_>>());
        let jumps = scan_jumps(&deltas, fps, o.jump_floor, o.jump_iso);
        video_obj["jumps"] = json!({"count": jumps.len(), "ok": jumps.is_empty(), "floor": r2(o.jump_floor),
            "isolation": r2(o.jump_iso),
            "list": jumps.iter().map(|j| json!({"seconds": j.t, "grey_levels": j.levels})).collect::<Vec<_>>()});
        let (nf, ns) = near_static_run(&deltas, fps);
        video_obj["near_static"] = json!({"frames": nf, "seconds": r2(ns), "ok": ns <= o.near_max_s, "max_seconds": r3(o.near_max_s),
            "floor": r2(NEAR_FLOOR)});
    } else {
        video_obj["note"] = json!("no video frames (send the video to /v1/media/check for ffmpeg to decode)");
    }

    json!({"audio": audio_obj, "video": video_obj})
}

// --------------------------------------------------------------------------------------------- resolve + HTTP + tool

/// Decode a request's `audio` (base64/data URL, an http(s) URL or a local file) into bytes, when given.
async fn load_audio(req: &Value) -> Result<Option<Vec<u8>>, String> {
    if let Some(b64) = req["audio"].as_str().filter(|s| !s.is_empty()) {
        let raw = b64.split_once(";base64,").map(|(_, b)| b).unwrap_or(b64);
        return Ok(Some(base64::engine::general_purpose::STANDARD.decode(raw.trim()).map_err(|_| "\"audio\": base64 or a data: URL".to_string())?));
    }
    if let Some(src) = req.get("url").or_else(|| req.get("path")).and_then(Value::as_str).filter(|s| !s.is_empty()) {
        Ok(Some(crate::media::load_source(src).await.map_err(|e| format!("{e:#}"))?))
    } else {
        Ok(None)
    }
}

async fn video_source(req: &Value) -> Option<Result<Vec<u8>, String>> {
    let src = req["video"].as_str().or_else(|| req["url"].as_str()).or_else(|| req["path"].as_str()).filter(|s| !s.is_empty())?;
    Some(crate::media::load_source(src).await.map_err(|e| format!("{e:#}")))
}

/// A chat's audio attachment or the request's `audio` → its Whisper transcript, when no `heard_text` was given.
async fn words_from(s: &AppState, audio: Option<&[u8]>, heard: &str) -> String {
    if !heard.trim().is_empty() { return heard.to_string(); }
    let Some(a) = audio else { return String::new() };
    match crate::audio_api::transcribe_for_chat(s, &base64::engine::general_purpose::STANDARD.encode(a)).await {
        Ok((text, _)) => text,
        Err(e) => { tracing::info!("check_media: Whisper did not transcribe the audio: {e}"); String::new() }
    }
}

/// The whole check from a request's bytes + language + heard text (the endpoint decodes video here; the tool passes None).
async fn run_check(s: &AppState, audio: Option<Vec<u8>>, gray: Option<&[Vec<u8>]>, req: &Value) -> Result<Value, String> {
    let expected = req["expected_text"].as_str().unwrap_or("").to_string();
    let lang = req.get("language").or_else(|| req.get("lang")).and_then(Value::as_str).unwrap_or("en").to_lowercase();
    let heard = words_from(s, audio.as_deref(), req["heard_text"].as_str().unwrap_or("")).await;
    let o = Opts::from_value(req);
    Ok(check(audio.as_deref(), gray, &expected, &heard, &lang, &o))
}

fn bad(msg: impl Into<String>) -> Response { error(StatusCode::BAD_REQUEST, "invalid_request_error", msg) }

/// POST /v1/media/check {"audio": <base64|data URL>, "expected_text": ..., "language": ... and optional threshold
/// overrides] and optionally a video to scan for frame jumps. Returns each check's numbers plus ok/boolean and the
/// thresholds used.
pub async fn check_endpoint(State(s): State<AppState>, headers: HeaderMap, Json(mut req): Json<Value>) -> Response {
    if !api_authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let audio = match load_audio(&req).await { Ok(x) => x, Err(e) => return bad(e) };
    let mut gray: Option<Vec<Vec<u8>>> = None;
    if let Some(src) = video_source(&req).await {
        match src {
            Ok(b) => match video_grey_frames(&b, req["fps"].as_f64()) {
                Ok((g, fps)) => { gray = Some(g); req["fps"] = json!(fps); }
                Err(e) => return Json(json!({"audio": null, "video": {"note": e}})).into_response(),
            },
            Err(e) => return bad(e),
        }
    }
    match run_check(&s, audio, gray.as_deref(), &req).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => bad(e),
    }
}

/// The `check_media` chat tool handle (holds the app state for Whisper transcription when no transcript is given).
pub struct Checker { app: Arc<AppState> }

impl Checker {
    pub fn new(app: Arc<AppState>) -> Self { Checker { app } }
    /// Run one check from a tool call's arguments (`audio`, `expected_text`, `language`, `heard_text`, overrides).
    pub async fn call(&self, args: &Value) -> Result<Value, String> {
        let audio = match load_audio(args).await { Ok(x) => x, Err(e) => return Err(e) };
        run_check(&self.app, audio, None, args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 16_000;

    fn tone(freq: f32, dur_s: f32, rate: u32, amp: f32) -> Vec<f32> {
        let n = (dur_s * rate as f32) as usize;
        (0..n).map(|i| amp * (2.0 * std::f32::consts::PI * freq * (i as f32 / rate as f32)).sin()).collect()
    }

    /// A WAV file from a flat, already-interleaved list of [-1, 1] samples.
    fn wav_flat(rate: u32, channels: u16, bits: u16, flat: &[f32]) -> Vec<u8> {
        let bpf = channels.max(1) as usize * (bits as usize / 8);
        let mut data = Vec::with_capacity(flat.len() * bpf);
        for &s in flat {
            match bits {
                16 => data.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes()),
                8 => data.push(((s.clamp(-1.0, 1.0) * 32767.0).round() as i32 / 32767 * 128 + 128) as u8),
                _ => {}
            }
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((44 + data.len()) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate as u32 * channels.max(1) as u32 * (bits as u32 / 8)).to_le_bytes());
        out.extend_from_slice(&(channels.max(1) as u16 * (bits as u16 / 8)).to_le_bytes());
        out.extend_from_slice(&(bits as u16).to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&data);
        out
    }

    fn grey(v: u8) -> Vec<u8> { vec![v; VIDEO_W * VIDEO_H] }

    #[test]
    fn read_wav_downmixes_stereo_to_mono() {
        // two stereo frames, channels averaged: (0.4+0.6)/2 == 0.5 each
        let flat = [0.4f32, 0.6, 0.4, 0.6];
        let (rate, mono) = read_wav(&wav_flat(8_000, 2, 16, &flat)).unwrap();
        assert_eq!(rate, 8_000);
        assert_eq!(mono.len(), 2);
        for m in &mono { assert!((m - 0.5).abs() < 1e-3, "stereo averaged to mono ({m})"); }
    }

    #[test]
    fn read_wav_rejects_non_wav() {
        assert!(read_wav(b"NOTAWAV!!").is_err());
    }

    #[test]
    fn longest_pause_is_the_interior_silence_gap() {
        let rate = SR;
        let mut s = tone(220.0, 0.5, rate, 0.8);       // voiced
        s.extend(std::iter::repeat(0f32).take((2.0 * rate as f32) as usize)); // 2 s of silence
        s.extend(tone(220.0, 0.5, rate, 0.8));
        let o = Opts::default();                        // gap_db 30
        let p = longest_pause_s(&s, rate, o.gap_db);
        assert!((p - 2.0).abs() < 0.1, "a 2 s interior pause should measure ~2 s (got {p})");
        assert!(!(p <= o.max_s), "2 s exceeds the 0.25 s cap -> not ok");
    }

    #[test]
    fn longest_pause_is_nearly_zero_for_a_continuous_tone() {
        let rate = SR;
        let p = longest_pause_s(&tone(220.0, 2.0, rate, 0.8), rate, GAP_DB);
        assert!(p < FRAME_S * 2.0, "a continuous tone has no interior pause (got {p})");
    }

    #[test]
    fn clipped_and_ending_flag_the_bad_ends() {
        let rate = SR;
        // a full-scale square wave is clipped everywhere
        let square: Vec<f32> = (0..rate as usize).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert!((clipped_ratio(&square) - 1.0).abs() < 0.01, "a full-scale wave is clipped (got {}", clipped_ratio(&square));
        // a soft tone is not clipped
        assert_eq!(clipped_ratio(&tone(220.0, 1.0, rate, 0.3)), 0.0);

        // cut off mid-voice: the clip stays loud at the very end
        assert!(ending_peak_ratio(&tone(220.0, 1.0, rate, 0.8)) > 0.9, "an abrupt end is still loud");
        // a take that ends in silence is quiet at the very end
        let mut quiet_end = tone(220.0, 0.6, rate, 0.8);
        quiet_end.extend(std::iter::repeat(0f32).take((0.4 * rate as f32) as usize));
        assert!(ending_peak_ratio(&quiet_end) < ENDING_FLOOR, "a take ending in silence is not a loud end");
    }

    #[test]
    fn words_heard_is_unicode_and_folded_for_hindi() {
        // Devanagari / French accents / Spanish ñ survive tokenization (the old [a-z]+ rule dropped them)
        let hi = "ग्राफ़िक पर्पोज़";
        assert!(!tokenize_words(hi).is_empty(), r"\w+ keeps the Devanagari words");
        assert!(tokenize_words("café à û").iter().any(|w| w.contains('é')), "French accents survive as words");
        assert!(tokenize_words("niño").iter().any(|w| w.contains('ñ')), "Spanish ñ survives as a letter");

        // Hindi: nukta + chandrabindu folded, so Whisper's spelling still matches the source
        let folded = normalize_word("ग्राफ़िकसँवद", "hi");
        assert!(!folded.contains('\u{093c}'), "nukta folded away for Hindi");
        assert!(!folded.contains('\u{0901}'), "chandrabindu folded away for Hindi");

        // differing only in nukta/chandrabindu -> 1.0 heard
        let ref_word = "\u{0917}\u{094d}\u{0930}\u{093e}\u{092b}\u{093c}\u{093f}\u{0915}";
        assert_eq!(word_match_rate(ref_word, "\u{0917}\u{094d}\u{0930}\u{093e}\u{092b}\u{093f}\u{0915}", "hi"), Some(1.0));
        // accents counted whole: an identical French line is fully heard
        assert_eq!(word_match_rate("café déjà vu", "café déjà vu", "fr"), Some(1.0));
        assert_eq!(word_match_rate("niño rojo", "niño rojo", "es"), Some(1.0));
        // missing words below the pass rate
        assert_eq!(word_match_rate("alpha beta gamma delta", "alpha delta", "en"), Some(0.5));
        // no expected words -> None (nothing to check)
        assert_eq!(word_match_rate("", "anything here", "en"), None);
    }

    #[test]
    fn scan_jumps_flags_an_isolated_step_but_not_a_crossfade_ramp() {
        let mut step = Vec::new();
        for _ in 0..10 { step.push(grey(0)); }      // dark hold
        for _ in 0..3 { step.push(grey(40)); }       // a single step up held afterwards (one big delta)
        let deltas = grey_frame_deltas(&step.iter().map(|f| f.as_slice()).collect::<Vec<_>>());
        let jumps = scan_jumps(&deltas, 24.0, JUMP_FLOOR, JUMP_ISO);
        assert_eq!(jumps.len(), 1, "one isolated step is one jump");
        assert!((jumps[0].t - 10.0 / 24.0).abs() < 1e-6, "the jump is timed at frame/fps (got {}", jumps[0].t);
        assert!((jumps[0].levels - 40.0).abs() < 1e-6, "the spike is ~40 grey levels (got {}", jumps[0].levels);

        // a smooth few-levels-per-frame ramp is not an isolated spike (floor dominates)
        let mut ramp = Vec::new();
        for v in 0u8..15 { ramp.push(grey(v)); }
        for _ in 0..5 { ramp.push(grey(14)); }
        let rd = grey_frame_deltas(&ramp.iter().map(|f| f.as_slice()).collect::<Vec<_>>());
        assert!(scan_jumps(&rd, 24.0, JUMP_FLOOR, JUMP_ISO).is_empty(), "a crossfade ramp is not flagged");
    }

    #[test]
    fn near_static_reports_a_frozen_hold() {
        let frames: Vec<_> = (0..100u8).map(|_| grey(10)).collect(); // 100 identical frames
        let deltas = grey_frame_deltas(&frames.iter().map(|f| f.as_slice()).collect::<Vec<_>>());
        let (held, secs) = near_static_run(&deltas, 24.0);
        assert_eq!(held, 100, "100 identical frames are a 100-frame frozen hold");
        assert!((secs - 100.0 / 24.0).abs() < 1e-6, "the hold's seconds is frames/fps (got {secs}");
    }

    #[test]
    fn check_assembles_audio_and_video_with_ok_flags() {
        // audio: voiced ... 2 s silence ... voiced -> interior pause too long, not clipped
        let rate = SR;
        let mut s = tone(220.0, 0.5, rate, 0.8);
        s.extend(std::iter::repeat(0f32).take((2.0 * rate as f32) as usize));
        s.extend(tone(220.0, 0.5, rate, 0.8));
        let wav = wav_flat(rate, 1, 16, &s);

        // video: a dark hold then one step up (one jump), plus short holds (near-static under the cap)
        let mut frames = Vec::new();
        for _ in 0..5 { frames.push(grey(0)); }
        for _ in 0..4 { frames.push(grey(40)); }
        let o = Opts::default();
        let hi_exp = "\u{0906}\u{0935}\u{093e}\u{091c} \u{0917}\u{094d}\u{0930}\u{093e}\u{092b}\u{093c}\u{093f}\u{0915} \u{092c}\u{091f}\u{0928}";
        let v = check(Some(&wav), Some(&frames), hi_exp, &normalize_word(hi_exp, "hi"), "hi", &o);

        assert_eq!(v["audio"]["longest_pause"]["ok"], json!(false), "a 2 s pause is not ok");
        assert!((v["audio"]["longest_pause"]["seconds"].as_f64().unwrap() - 2.0).abs() < 0.1, "pause ~2 s (got {}", v["audio"]["longest_pause"]["seconds"]);
        assert_eq!(v["audio"]["clipped"]["ok"], json!(true), "the tone is not clipped");
        let wh = &v["audio"]["words_heard"];
        assert_eq!(wh["rate"].as_f64().unwrap(), 1.0, "Hindi nukta-folding matches -> fully heard");
        assert_eq!(wh["ok"], json!(true));

        assert_eq!(v["video"]["jumps"]["count"].as_u64().unwrap(), 1, "one single-frame jump");
        assert_eq!(v["video"]["jumps"]["ok"], json!(false), "a jump present is not ok");
        assert_eq!(v["video"]["jumps"]["floor"].as_f64().unwrap(), JUMP_FLOOR);
        assert_eq!(v["video"]["near_static"]["ok"], json!(true), "the short hold is within the cap");
    }

    #[test]
    fn words_heard_is_null_without_a_transcript() {
        let rate = SR;
        let wav = wav_flat(rate, 1, 16, &tone(220.0, 1.0, rate, 0.8));
        let v = check(Some(&wav), None, "hello world", "", "en", &Opts::default());
        assert!(v["audio"]["words_heard"]["rate"].is_null(), "no transcript -> words_heard null");
    }
}

