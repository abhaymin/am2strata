//! Compose a finished video from shots and audio (`POST /v1/media/videos/compose`), and a thumbnail poster image
//! (`POST /v1/media/thumbnails`). The chat's `compose_video` queues the one, `make_thumbnail` the other (media_jobs.rs);
//! both are queued media jobs like productions: a job is made behind the answer, its file at `/v1/media/jobs/{id}/content`.
//!
//! ffmpeg is the media engine, located on PATH exactly as media.rs does (`ffmpeg_available`, `ffprobe`): it renders each
//! shot (a centred sub-pixel zoom for an image from a 2× picture, a fitted clip held on its last frame if short), joins
//! them with crossfades (no dip to black), burns the captions and any title/end card in with libass, and muxes the
//! original audio track to one MP4 (H.264 + AAC, faststart). A thumbnail is one big-worded JPEG on a dark gradient, the
//! Outfit font the web app ships drawn by ffmpeg's drawtext. The video and image layout are pure here, unit-tested
//! without ffmpeg; the real encoding runs only when ffmpeg is installed.

use crate::{error, AppState};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::process::Stdio;
use tokio::process::Command as TokioCommand;

/// The frames a composed video plays per second.
const FPS: u32 = 30;
/// A shot without its own length is spread to fill the audio, this much overlapping the next (no dip to black).
const CROSSFADE: f64 = 0.8;
/// The thumbnail's size.
pub const THUMB_W: u32 = 1280;
pub const THUMB_H: u32 = 720;
/// The gentle centred zoom an image plays over its whole shot (from a picture twice the frame size, so it is sub-pixel).
const ZOOM_END: f64 = 1.12;

// ----------------------------------------------------------------------------------------------------------- layout

/// How long each shot plays when none is given a length: spread evenly so they span `total_s` with `crossfade` between
/// them (N shots have N-1 joins, so their durations add to `total_s + crossfade*(N-1)`). The last shot's full length
/// counts, as it has no join after it. Empty when there are no shots.
pub fn spread_over(total_s: f64, crossfade: f64, n_shots: usize) -> Vec<f64> {
    if n_shots == 0 || total_s.is_finite() && total_s > 0.0 {
        let xf = crossfade.clamp(0.0, 8.0);
        let budget = (total_s + xf * n_shots.max(1) as f64 - xf).max(1.0);
        vec![budget / n_shots.max(1) as f64; n_shots]
    } else {
        vec![5.0; n_shots]
    }
}

/// A shot's frames from its length in seconds, at the video's rate.
fn frames_for(seconds: f64) -> usize {
    ((seconds.max(0.0) * FPS as f64).round() as usize).max(1)
}

/// The offsets each shot starts at once the shots are joined with a crossfade between neighbours: shot k begins one
/// crossfade before shot k-1 ends, so the joins overlap by `crossfade` (no dip to black).
pub fn join_offsets(durations: &[f64], crossfade: f64) -> Vec<f64> {
    let xf = crossfade.clamp(0.0, 8.0);
    let mut offs = Vec::with_capacity(durations.len());
    let mut t = 0.0;
    for d in durations {
        offs.push(t);
        t += d - xf;
    }
    offs
}

/// The length the joined video plays: the shots' durations minus one crossfade per join between neighbours.
pub fn total_length(durations: &[f64], crossfade: f64) -> f64 {
    if durations.is_empty() {
        return 0.0;
    }
    let xf = crossfade.clamp(0.0, 8.0);
    durations.iter().sum::<f64>() - xf * (durations.len() as f64 - 1.0)
}

/// A size like "1920x1080", as (width, height). Each side between 128 and 3840 pixels and even (H.264's yuv420p needs
/// even dimensions); when the request leaves it out or asks for something that will not fit, landscape 1920×1080 is used.
pub fn parse_size(s: &str) -> Option<(u32, u32)> {
    let mut parts = s.split('x');
    let w = parts.next()?.trim().parse::<u32>().ok()?;
    let h = parts.next()?.trim().parse::<u32>().ok()?;
    if parts.next().is_some() {
        return None; // more than WIDTHxHEIGHT
    }
    if (128..=3840).contains(&w) && (128..=3840).contains(&h) && w % 2 == 0 && h % 2 == 0 {
        Some((w, h))
    } else {
        None
    }
}

// --------------------------------------------------------------------------------------------------------------- ASS

/// A caption or card's text for one ASS line: its lines joined with ASS's own line break (\\N), and braces (which
/// would start an override block) and backslashes made literal.
fn ass_text(t: &str) -> String {
    t.lines().map(str::trim).collect::<Vec<_>>().join("\n")
        .replace('\\', "\u{29F5}").replace('{', "(").replace('}', ")").replace('\n', "\\N")
}

/// An Ass time like "0:01:23.45".
fn ass_time(t: f64) -> String {
    let t = t.max(0.0);
    let secs = t.fract();
    let whole = t as u64;
    format!("{}:{:02}:{:02}.{:02}", whole / 3600, (whole % 3600) / 60, whole % 60, (secs * 100.0).round() as u64 % 100)
}

/// The captions and the title/end cards as an ASS script burned in with libass: the captions centred low (each fading
/// in and out over a quarter second, like song_video.py), the title at the start and the end card at the end. Empty when
/// there is nothing to show.
pub fn ass_script(w: u32, h: u32, captions: &[(f64, f64, String)], open: Option<&str>, end: Option<&str>, total: f64) -> String {
    let mut out = format!(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: {}\nPlayResY: {}\n\n[V4+ Styles]\nFormat: Name, Fontname, \
         Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, \
         ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n\
         Style: Cap,DejaVu Sans,58,&H00FAF8F5,&H00FFFFFF,&H00180A06,&H96180A06,1,0,0,0,100,100,0,0,3,14,0,2,80,80,120,1\n\
         Style: Ttl,DejaVu Sans,96,&H00FAF8F5,&H00FFFFFF,&H00180A06,&H96180A06,1,0,0,0,100,100,0,0,3,18,0,5,80,80,0,1\n\n\
         [Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
        w, h,
    );
    for (a, b, text) in captions {
        if *b <= *a {
            continue;
        }
        // each caption fades in and out over a quarter second so it does not snap on its change
        let t = format!("{{\\fad(250,250)}}{}", ass_text(text));
        out.push_str(&format!("Dialogue: 0,{},{},Cap,,0,0,120,,{}\n", ass_time(*a), ass_time(*b), t));
    }
    if let Some(t) = open {
        out.push_str(&format!("Dialogue: 0,{},{},Ttl,,0,0,0,,{{\\fad(400,400)}}{}\n", ass_time(0.5), ass_time(9.0_f64.min(total)), ass_text(t)));
    }
    if let Some(t) = end {
        out.push_str(&format!("Dialogue: 0,{},{},Ttl,,0,0,0,,{{\\fad(400,400)}}{}\n", ass_time((total - 10.0).max(0.0)), ass_time(total), ass_text(t)));
    }
    out
}

// ---------------------------------------------------------------------------------------------------------- thumbnail

/// The drawtext option for one line (big words, a dark stroke for contrast on any picture): the colour and, when there
/// is more than one line, its vertical place. The first line sits low, the rest cascade up over it.
pub fn thumbnail_lines(lines: &[(String, String)], tag: Option<&str>) -> BTreeMap<u32, (String, String)> {
    // maps a baseline row (0 = lowest) to (text, colour)
    let mut rows: BTreeMap<u32, (String, String)> = BTreeMap::new();
    for (i, (text, colour)) in lines.iter().enumerate() {
        let base = (lines.len() as u32 - 1) - i as u32;
        rows.insert(base, (text.clone(), colour.clone()));
    }
    if let Some(t) = tag {
        rows.insert(lines.len() as u32, (t.to_string(), "white".to_string()));
    }
    rows
}

// -------------------------------------------------------------------------------------------------------------- run

/// Run ffmpeg once with the given arguments; why it failed, in words (its stderr trimmed).
async fn run_ffmpeg(args: &[&str]) -> Result<(), String> {
    let out = TokioCommand::new(crate::server_tls::program("ffmpeg")).args(args).stdout(Stdio::null()).stderr(Stdio::piped())
        .kill_on_drop(true).output().await.map_err(|e| format!("ffmpeg could not start: {e}"))?;
    if !out.status.success() {
        return Err(format!("ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr).trim().replace('\n', " ")));
    }
    Ok(())
}

/// The length a media source plays, seconds (ffprobe's format=duration), or None when it could not be read.
async fn duration_of(dir: &std::path::PathBuf, name: &str, bytes: &[u8]) -> Result<f64, String> {
    let in_ = dir.join(name);
    std::fs::write(&in_, bytes).map_err(|e| format!("could not write the input: {e}"))?;
    let probe = TokioCommand::new(crate::server_tls::program("ffprobe"))
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1"]).arg(&in_).output().await
        .map_err(|e| format!("ffprobe could not start: {e}"))?;
    String::from_utf8_lossy(&probe.stdout).trim().parse::<f64>().ok().filter(|d| *d > 0.0)
        .ok_or_else(|| format!("could not read the length of the {name}"))
}

/// A media job's bytes (by id) or a source the request carries directly (`data:` URL, http(s), file path).
async fn source_bytes(s: &AppState, what: &str, src: &Value) -> Result<Vec<u8>, String> {
    if let Some(id) = src.as_str().and_then(|s| s.trim().strip_prefix("mj-")) {
        if let Some(b) = s.media_jobs.bytes(&format!("mj-{id}")) {
            return Ok(b);
        }
        return Err(format!("{what}: no media job mj-{id} (make it or send the file itself)"));
    }
    match src.as_str() {
        Some(raw) if !raw.trim().is_empty() => crate::media::load_source(raw.trim()).await.map_err(|e| format!("{what}: {e}")),
        _ => Err(format!("{what}: send the file itself, a data: URL, or one of your media job ids")),
    }
}

/// Is this source an image, a clip (mp4/webm/ogg video) or audio (wav/mp3)? Its kind by its magic bytes.
fn kind_of_bytes(bytes: &[u8]) -> &'static str {
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return "audio"; // a WAV
    }
    if bytes.starts_with(b"\xef\xbb\xbfID3") || (bytes.len() >= 3 && &bytes[0..3] == b"ID3") {
        return "audio"; // an MP3 with an ID3 tag
    }
    if bytes.len() >= 4 && bytes[0] == 0xFF && (bytes[1] & 0xF0) == 0xF0 {
        return "audio"; // a raw MP3 frame stream (no ID3 tag): an MPEG-1 audio frame sync, whose FF F8–FF start JPEG's SOI FF D8 avoids
    }
    if bytes.len() >= 8 && (&bytes[4..8] == b"ftyp" || &bytes[0..4] == b"\x1a\x45\xda\xa3" || &bytes[0..4] == b"OggS") {
        return "clip"; // an mp4, a webm/matroska or an ogg video
    }
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) || bytes.starts_with(&[0xFF, 0xD8]) || bytes.starts_with(&[0x47, b'I', b'F'])
        || bytes.starts_with(&[0x89, b'W', b'P', b'8']) {
        return "image";
    }
    "image"
}

/// POST /v1/media/videos/compose {"audio", "shots": [{"image"|"clip"}, ...], "captions"?: [[s, s, text]],
/// "title_card"?, "end_card"?, "size"?: "1920x1080", "crossfade_s"?: 0.8}: one job for the whole video (GET
/// /api/jobs/{id} follows it; /v1/media/jobs/{id}/content is the MP4). Queued like a production; its meta holds the
/// size, length and how many shots were joined.
pub async fn compose_videos(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !crate::api_authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    match s.media_jobs.submit_compose(&req) {
        Ok(job) => Json(job).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    }
}

/// POST /v1/media/thumbnails {"image", "lines": [[text, "white"|"gold"]], "tag"}: one JPEG, 1280×720, the image with a
/// dark gradient over the words and big Outfit type. Queued like a production; its file is at /v1/media/jobs/{id}/content.
pub async fn thumbnails(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !crate::api_authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    match s.media_jobs.submit_thumbnail(&req) {
        Ok(job) => Json(job).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    }
}

/// Make the video: resolve the audio and shots, lay the shots out over the audio, render each with ffmpeg, join them
/// with crossfades, burn in the captions and cards, and mux the audio to one MP4. Returns the file bytes and its meta.
pub async fn make(s: &AppState, n: u64, req: &Value) -> Result<(String, Vec<u8>, Value), String> {
    if !crate::media::ffmpeg_available() {
        return Err(format!("this server cannot make a video: install ffmpeg ({})",
            crate::doctor::install_hint(std::env::consts::OS, "ffmpeg", "ffmpeg", "Gyan.FFmpeg")).into());
    }
    let (w, h) = parse_size(req["size"].as_str().unwrap_or("1920x1080")).ok_or_else(|| {
        format!("error: size {} does not work; use 1920x1080 or 1080x1920", req["size"].as_str().unwrap_or(""))
    })?;
    let cross = req["crossfade_s"].as_f64().unwrap_or(CROSSFADE).clamp(0.0, 4.0);
    let shots = req["shots"].as_array().ok_or("shots: a list of {\"image\"|\"clip\": ...}")?;
    if shots.is_empty() {
        return Err("shots: at least one shot is needed".into());
    }
    // the audio: required, its length sets the canvas and where the shots land
    let audio = source_bytes(s, "audio", &req["audio"]).await?;
    if kind_of_bytes(&audio) == "image" {
        return Err("audio: send an audio file (WAV or MP3), not an image".into());
    }
    let dir = crate::media::tempdir().map_err(|e| format!("could not make a temp folder: {e}"))?;
    let id = format!("mj-{n}");
    // the caption events, and the title/end cards (optional)
    let captions: Vec<(f64, f64, String)> = req["captions"].as_array().map(|a| {
        a.iter().filter_map(|c| {
            let arr = c.as_array()?;
            let a = arr.first()?.as_f64()?;
            let b = arr.get(1)?.as_f64()?;
            let text = arr.get(2).and_then(|t| t.as_str()).unwrap_or("").to_string();
            if text.is_empty() || b <= a { None } else { Some((a, b, text)) }
        }).collect::<Vec<_>>()
    }).unwrap_or_default();
    // the shots' lengths: their own when given, else spread over the audio (with crossfades)
    let whole_s = duration_of(&dir, "audio.wav", &audio).await?;
    // a part of the audio (a Short from the chorus): audio_from seconds in, for audio_seconds; captions are timed in it
    let (from, total_s) = audio_part(whole_s, req["audio_from"].as_f64(), req["audio_seconds"].as_f64())?;
    let ass = ass_script(w, h, &captions, req["title_card"].as_str(), req["end_card"].as_str(), total_s);
    let subs = dir.join("subs.ass");
    std::fs::write(&subs, &ass).map_err(|e| format!("could not write the captions: {e}"))?;
    // each shot's file, its kind, and a clip's own length (a clip plays its motion, never a frozen last frame)
    let mut srcs: Vec<(std::path::PathBuf, &'static str, Option<f64>)> = Vec::new();
    for (i, sh) in shots.iter().enumerate() {
        let raw = sh.get("image").or_else(|| sh.get("clip")).and_then(Value::as_str).unwrap_or("");
        let bytes = source_bytes(s, if sh.get("clip").is_some() { "clip" } else { "image" }, &Value::from(raw)).await?;
        let kind = kind_of_bytes(&bytes);
        let ext = if kind == "clip" { "mp4" } else { "png" };
        let name = format!("src-{i}.{ext}");
        let len = if kind == "clip" { duration_of(&dir, &name, &bytes).await.ok() } else {
            std::fs::write(dir.join(&name), &bytes).map_err(|e| format!("could not write the {kind}: {e}"))?;
            None
        };
        srcs.push((dir.join(name), kind, len));
    }
    let plan: Vec<(Option<f64>, Option<f64>)> = shots.iter().zip(&srcs).map(|(sh, (_, _, len))| (sh["seconds"].as_f64(), *len)).collect();
    let durations = layout(total_s, cross, &plan);
    // a shot over its given length: images still play their whole zoom (clips are held on the last frame when short)
    let played = total_length(&durations, cross); // what the joined footage actually plays; cap to it
    let _ = join_offsets(&durations, cross);

    s.media_jobs.set_meta(n, json!({"part": 0, "parts": shots.len(), "now": "compose"}));
    // make each input a temp file (a data URL is decoded to bytes above; the rest are paths / job ids already resolved)
    let mut inputs = Vec::new();
    for (i, (src, kind, clip_len)) in srcs.iter().enumerate() {
        let (src, kind) = (src.clone(), *kind);
        let out = dir.join(format!("shot-{:02}.mp4", i));
        let frames = frames_for(durations[i]);
        let (w2, h2) = (w * 2, h * 2);
        if kind == "clip" {
            // fitted to the frame, held on its last frame if shorter than the shot (so a short clip still fills it)
            let fit = clip_filter(w, h, durations[i], *clip_len);
            let srcp = src.display().to_string();
            let outp = out.display().to_string();
            run_ffmpeg(&["-i", srcp.as_str(), "-vf", fit.as_str(), "-frames:v", frames.to_string().as_str(), "-an", "-c:v", "libx264", "-crf", "18", outp.as_str()]).await?;
        } else {
            // a centred zoom spread over the whole shot, from a picture twice the frame size so the steps are sub-pixel
            let zfactor = ZOOM_END - 1.0;
            let zn = (frames.saturating_sub(1)).max(1);
            let vf = format!("scale={w2}:{h2}:force_original_aspect_ratio=increase,crop={w2}:{h2},zoompan=z='1+{}*on/{}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={FPS},format=yuv420p", zfactor, zn);
            let fps_s = FPS.to_string();
            let srcp = src.display().to_string();
            let outp = out.display().to_string();
            run_ffmpeg(&["-loop", "1", "-framerate", fps_s.as_str(), "-i", srcp.as_str(), "-vf", vf.as_str(), "-frames:v", frames.to_string().as_str(), "-an", "-c:v", "libx264", "-crf", "18", outp.as_str()]).await?;
        }
        inputs.push(out);
    }
    // join the shots with a crossfade between each (no dip to black), then hold for the audio's length
    let chain = xfade_chain(&durations, cross);
    let pictures = dir.join("pictures.mp4");
    // the shot files are inputs; collect them as owned Strings so nothing is borrowed out of this block
    let mut inps: Vec<String> = Vec::new();
    for inp in &inputs {
        inps.push("-i".to_string());
        inps.push(inp.display().to_string());
    }
    inps.push("-filter_complex".to_string());
    inps.push(format!("{}fade=t=in:st=0:d=1.0,format=yuv420p[out]", chain));
    inps.push("-map".to_string());
    inps.push("[out]".to_string());
    inps.push("-t".to_string());
    inps.push(played.to_string());
    inps.push("-c:v".to_string());
    inps.push("libx264".to_string());
    inps.push("-crf".to_string());
    inps.push("17".to_string());
    inps.push(pictures.display().to_string()); // the output file (position matters after -map)
    let args: Vec<&str> = inps.iter().map(String::as_str).collect();
    run_ffmpeg(&args).await?;

    // the joined pictures must play as long as planned (a chain cut short would leave a long video with a still end)
    if let Some(got) = probe_seconds(&pictures).await {
        if got + 1.0 < played {
            return Err(format!("the joined pictures play {got:.1} s of the {played:.1} s planned; not finished"));
        }
    }
    // mux the picture chain with the audio and the captions and cards burned in, one MP4 (H.264 + AAC, faststart).
    // The captions are drawn by am2Strata (captions.rs) and laid on with ffmpeg's overlay, which every ffmpeg has;
    // "captions_engine": "libass" uses the subtitles filter instead
    let out = dir.join(format!("{id}.mp4"));
    let picp = pictures.display().to_string();
    let audiop = dir.join("audio.wav").display().to_string();
    let mut args: Vec<String> = vec!["-i".into(), picp, "-i".into(), audiop];
    let fc = if req["captions_engine"].as_str() == Some("libass") {
        let subsp = subs.display().to_string().replace('\\', "\\\\").replace(':', "\\:");
        final_filter(&subsp, played, from)
    } else {
        let ovs = crate::captions::overlays(w, h, &captions, req["title_card"].as_str(), req["end_card"].as_str(), total_s)?;
        for (k, o) in ovs.iter().enumerate() {
            let f = dir.join(format!("cap-{k:03}.png"));
            std::fs::write(&f, &o.png).map_err(|e| format!("could not write a caption: {e}"))?;
            args.extend(["-loop", "1", "-framerate", "30", "-t"].map(String::from));
            args.push(format!("{:.3}", (o.b - o.a).max(0.3)));
            args.push("-i".into());
            args.push(f.display().to_string());
        }
        crate::captions::overlay_filter(&ovs, 2, played, from)
    };
    args.extend(["-filter_complex".to_string(), fc, "-map".into(), "[v]".into(), "-map".into(), "[a]".into(), "-t".into(), played.to_string(),
                 "-r".into(), FPS.to_string(), "-c:v".into(), "libx264".into(), "-preset".into(), "medium".into(), "-crf".into(), "18".into(),
                 "-pix_fmt".into(), "yuv420p".into(), "-c:a".into(), "aac".into(), "-b:a".into(), "192k".into(), "-movflags".into(), "+faststart".into(),
                 out.display().to_string()]);
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    run_ffmpeg(&refs).await?;

    // read the finished video while its temp folder still exists, then free the temporary clips and the whole dir (a restart keeps it)
    let bytes = std::fs::read(&out).map_err(|e| format!("could not read the finished video: {e}"))?;
    let _ = std::fs::remove_dir_all(&dir);
    Ok(("video/mp4".into(), bytes, json!({"width": w, "height": h, "seconds": (played * 10.0).round() / 10.0, "shots": inputs.len(), "captions": captions.len()})))
}

/// A clip fitted to the frame and held on its last frame for as long as its shot plays (a 5 s clip in a 14 s shot
/// fills all 14 s; held only briefly, the crossfade chain ended early and the video stopped long before the audio).
pub fn clip_filter(w: u32, h: u32, shot_s: f64, clip_s: Option<f64>) -> String {
    // slowed to fill its shot, by up to MAX_SLOW (calm motion slowed reads as calm; frozen reads as broken); evened
    // out first, as generated clips flicker from frame to frame
    let slow = clip_s.filter(|c| *c > 0.1).map(|c| (shot_s / c).clamp(1.0, MAX_SLOW)).unwrap_or(1.0);
    // the new frames between a slowed clip's own are blended (a 16 fps Wan clip slowed 1.6x at 30 fps would show each
    // picture three frames in a row: a judder the check sees as jumps)
    format!("deflicker=size=5,setpts={slow:.4}*PTS,framerate=fps={FPS}:interp_start=0:interp_end=255:scene=100,\
             scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},tpad=stop_mode=clone:stop_duration={:.3},format=yuv420p",
            shot_s.max(0.5) + 1.0)
}

/// The most a clip is slowed to fill its shot.
pub const MAX_SLOW: f64 = 1.6;

/// Each shot's length. Shots given "seconds" keep them. A clip plays its own length, slowed by up to MAX_SLOW, but never
/// more than an even share. The images take the rest of the audio, evenly, so no clip is held on a frozen frame. When
/// every shot is a clip, they share the audio evenly (and the last frames hold).
pub fn layout(total_s: f64, crossfade: f64, shots: &[(Option<f64>, Option<f64>)]) -> Vec<f64> {
    let n = shots.len();
    if n == 0 { return vec![]; }
    if shots.iter().all(|(given, _)| given.is_some()) {
        return shots.iter().map(|(g, _)| g.unwrap_or(5.0)).collect();
    }
    let xf = crossfade.clamp(0.0, 8.0);
    let budget = (total_s + xf * (n as f64 - 1.0)).max(1.0);
    let even = budget / n as f64;
    let mut out: Vec<Option<f64>> = shots.iter().map(|(given, clip)| given.or_else(|| clip.map(|c| (c * MAX_SLOW).min(even).max(xf + 0.5)))).collect();
    let free: Vec<usize> = (0..n).filter(|&i| out[i].is_none()).collect();
    if free.is_empty() {
        // only clips (and given lengths): spread what is left over the clips
        return spread_over(total_s, crossfade, n);
    }
    let used: f64 = out.iter().flatten().sum();
    let each = ((budget - used) / free.len() as f64).max(xf + 1.0);
    for i in free { out[i] = Some(each); }
    out.into_iter().map(|d| d.unwrap_or(each)).collect()
}

/// A video file's video-stream length in seconds, by ffprobe.
async fn probe_seconds(f: &std::path::Path) -> Option<f64> {
    let o = TokioCommand::new(crate::server_tls::program("ffprobe"))
        .args(["-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "default=nw=1:nk=1"])
        .arg(f).output().await.ok()?;
    let frames: f64 = String::from_utf8_lossy(&o.stdout).trim().parse().ok()?;
    Some(frames / FPS as f64)
}

/// The last pass's filtergraph: the captions burned into the joined pictures, and the audio from its start with a 4 s
/// fade out at the end. `subs` is the ASS file's path already escaped for a filter option.
pub fn final_filter(subs: &str, played: f64, audio_from: f64) -> String {
    format!("[0:v]subtitles=filename='{}'[v];[1:a]atrim=start={:.3},asetpts=N/SR/TB,afade=t=out:st={:.3}:d=4[a]", subs, audio_from.max(0.0),
            played.max(4.0) - 4.0)
}

/// The part of the audio a video plays: (start, length) from `audio_from` and `audio_seconds` within the whole length.
pub fn audio_part(whole: f64, from: Option<f64>, seconds: Option<f64>) -> Result<(f64, f64), String> {
    let from = from.unwrap_or(0.0);
    if !(0.0..whole).contains(&from) {
        return Err(format!("audio_from: {from} s is not inside the audio ({whole:.1} s)"));
    }
    let len = seconds.map(|s| s.min(whole - from)).unwrap_or(whole - from);
    if len < 1.0 {
        return Err("audio_seconds: at least 1 s of audio is needed".into());
    }
    Ok((from, len))
}

/// The xfade chain that joins the shots in order, each beginning one crossfade before the previous ends (so their joins
/// overlap and there is no dip to black). Shot k starts at the sum of the earlier shots' lengths minus a crossfade each,
/// and the chain's final output is labelled `[v]` (the caller burns the opening fade into it); a single shot is just
/// `[0:v]`.
pub fn xfade_chain(durations: &[f64], crossfade: f64) -> String {
    if durations.is_empty() {
        return String::new();
    }
    let xf = crossfade.clamp(0.0, 8.0);
    // each shot's start in the final timeline: sum of the earlier lengths, minus a crossfade per join so far
    let mut starts = Vec::with_capacity(durations.len());
    let mut acc = 0.0f64;
    for (k, &d) in durations.iter().enumerate() {
        starts.push(acc - (k as f64) * xf);
        acc += d;
    }
    let n = durations.len();
    let mut parts: Vec<String> = Vec::new();
    let mut last = String::from("[0:v]");
    for k in 1..n {
        let out = if k == n - 1 { "[v]".to_string() } else { format!("[v{}]", k) };
        parts.push(format!("{}[{}:v]xfade=transition=fade:duration={:.3}:offset={:.3}{}", last, k, xf, starts[k], out));
        last = out;
    }
    // filter chains are separated by ';' (ffmpeg reads two chains run together as trailing garbage), and the caller's
    // next filter reads the joined stream as its own chain from the [v] label
    if parts.is_empty() { String::from("[0:v]") } else { format!("{};[v]", parts.join(";")) }
}

/// ffmpeg reads a drawtext option between its quotes (e.g. `text='…'`), where inside those quotes a backslash still opens an
/// escape and a single quote closes the value: unescaped, request text could end the quotes and append another option such as
/// `textfile=/etc/passwd`. Escape just those two characters (double the backslash, quote the apostrophe) so the whole value
/// stays inside its own quotes and round-trips to exactly the caption requested — a colon or `%{pts}` passes through as text.
fn drawtext_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            _ => out.push(c),
        }
    }
    out
}

/// Make the thumbnail: one 1280×720 JPEG, the image with a left-dark gradient and big Outfit words drawn over it by
/// ffmpeg's drawtext (the heavy type with a dark stroke, like thumbs.py). Returns the JPEG bytes.
pub async fn make_thumbnail(s: &AppState, req: &Value) -> Result<(String, Vec<u8>, Value), String> {
    if !crate::media::ffmpeg_available() {
        return Err(format!("this server cannot make a thumbnail: install ffmpeg ({})",
            crate::doctor::install_hint(std::env::consts::OS, "ffmpeg", "ffmpeg", "Gyan.FFmpeg")).into());
    }
    let (w, h) = (THUMB_W, THUMB_H);
    let image = source_bytes(s, "image", &req["image"]).await?;
    if kind_of_bytes(&image) != "image" {
        return Err("image: send a picture (PNG or JPEG), not a video".into());
    }
    // the lines of words and the optional tag line beneath them
    let mut lines: Vec<(String, String)> = Vec::new();
    if let Some(arr) = req["lines"].as_array() {
        for l in arr {
            let text = l.get(0).and_then(Value::as_str).unwrap_or("").trim().to_string();
            if text.is_empty() { continue; }
            let colour = l.get(1).and_then(Value::as_str).unwrap_or("white");
            lines.push((text, colour.to_string()));
        }
    }
    if lines.is_empty() {
        return Err("lines: give the words to show, as [[text, \"white\"|\"gold\"]]".into());
    }
    let tag = req["tag"].as_str().filter(|t| !t.trim().is_empty());

    let dir = crate::media::tempdir().map_err(|e| format!("could not make a temp folder: {e}"))?;
    let id = format!("mj-{}", std::process::id()); // thumbnails keep nothing, so a plain id is fine here
    let _ = &id;
    let image_ = dir.join("image.jpg");
    std::fs::write(&image_, &image).map_err(|e| format!("could not write the image: {e}"))?;
    let font = outfit_font();
    let out = dir.join("thumb.jpg");

    // big words, centred, one row below the next: drawtext for each, with a dark shadow for the stroke look
    let rows = thumbnail_lines(&lines, tag);
    let base_y = (h as f64 * 0.72) as i32; // the lowest line's baseline
    let mut draws = Vec::new();
    for (row, (text, colour)) in rows {
        let y = base_y - (row as i32) * 150;
        let color = match colour.as_str() { "gold" => "ffd65a", _ => "ffffff" };
        // each line is centred and placed a row below the last, with a dark shadow for the stroke look on any picture. The
        // options are built as a list and joined with ':' so an absent fontfile leaves no empty field (drawtext=text=...,
        // not drawtext=:text=...); a found Outfit font only leads with its fontfile
        draws.push(thumbnail_drawtext(&text, &font, color, y));
    }
    // the whole thing: fit the picture, darken it so the words read, then stack the words on top; one 1280×720 JPEG
    let _ = run_ffmpeg(&[
        "-y", "-i", image_.display().to_string().as_str(),
        "-vf", &format!("scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},eq=brightness=-0.15,{}", draws.join(",")).as_str(),
        "-vframes", "1", "-c:v", "mjpeg", "-q:v", "3", out.display().to_string().as_str(),
    ]).await?;

    let bytes = std::fs::read(&out).map_err(|e| format!("could not read the thumbnail: {e}"))?;
    std::fs::remove_dir_all(&dir).ok();
    Ok(("image/jpeg".into(), bytes, json!({"width": w, "height": h, "lines": lines.len()})))
}

/// The Outfit font the web app ships (web/fonts/outfit-latin-wght.woff2), found beside this binary or in a checked-out
/// web/ folder; empty when it is not there, so drawtext falls back to whatever system font remains.
fn outfit_font() -> String {
    let candidates = [
        std::env::current_exe().ok().and_then(|e| e.parent()?.parent()?.parent()?.parent().map(|d| d.join("web/fonts/outfit-latin-wght.woff2"))),
        std::env::current_exe().ok().and_then(|e| e.parent()?.parent().map(|d| d.join("web")).map(|d| d.join("fonts/outfit-latin-wght.woff2"))),
        std::env::current_dir().ok().map(|d| d.join("web/fonts/outfit-latin-wght.woff2")),
    ];
    candidates.iter().flatten().find(|p| p.is_file()).map(|p| p.display().to_string()).unwrap_or_default()
}

/// One thumbnail row's drawtext filter, built as a list of options joined with ':' so an absent fontfile leaves no empty
/// field (the filter starts `drawtext=text=`, not `drawtext=:text=`). A found Outfit font only leads with its fontfile.
fn thumbnail_drawtext(text: &str, font: &str, color: &str, y: i32) -> String {
    let mut opts = Vec::new();
    if !font.is_empty() { opts.push(format!("fontfile='{}'", drawtext_escape(font))); }
    opts.push(format!("text='{}'", drawtext_escape(text)));
    opts.extend([
        format!("fontsize=120"),
        format!("fontcolor={color}"),
        "x=(w-text_w)/2".to_string(),
        format!("y={y}"),
        "shadowcolor=0x180A06@0.6".to_string(),
        "shadowx=3".to_string(),
        "shadowy=3".to_string(),
    ]);
    format!("drawtext={}", opts.join(":"))
}

// ----------------------------------------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {

    /// A short clip in a long shot is held to the shot's whole length (Peace and Harmony's 5 s clips in 14 s shots
    /// once made a 15 s picture track under 200 s of audio). Only runs where ffmpeg is installed.
    #[test]
    fn clips_play_their_motion_and_the_pictures_take_the_rest() {
        // 200 s of song, 14 shots, 5 clips of 5.06 s: each clip plays 1.6x slowed (8.1 s), the 9 pictures share the rest
        let mut plan = vec![(None, None); 14];
        for i in [0usize, 3, 5, 9, 13] { plan[i] = (None, Some(5.0625)); }
        let d = layout(200.0, 0.8, &plan);
        assert!((d[0] - 8.1).abs() < 0.01, "{d:?}");
        let pic = d[1];
        assert!(pic > 15.0 && (d.iter().sum::<f64>() - (200.0 + 0.8 * 13.0)).abs() < 1e-6, "fills the audio: {d:?}");
        assert!(d.iter().enumerate().all(|(i, x)| [0, 3, 5, 9, 13].contains(&i) || (*x - pic).abs() < 1e-9));
        // a given length is kept; only clips share the audio evenly
        assert_eq!(layout(20.0, 0.0, &[(Some(3.0), None), (None, None)]), [3.0, 17.0]);
        assert_eq!(layout(10.0, 0.0, &[(None, Some(2.0)), (None, Some(2.0))]), [5.0, 5.0]);
    }

    #[test]
    fn a_short_plays_a_part_of_the_song() {
        assert_eq!(audio_part(200.0, Some(35.9), Some(58.0)).unwrap(), (35.9, 58.0));
        assert_eq!(audio_part(200.0, Some(180.0), Some(58.0)).unwrap(), (180.0, 20.0), "to the end at most");
        assert_eq!(audio_part(45.0, None, None).unwrap(), (0.0, 45.0));
        assert!(audio_part(45.0, Some(50.0), None).is_err());
        assert!(final_filter("s.ass", 58.0, 35.9).contains("atrim=start=35.900"));
    }

    #[test]
    fn compose_needs_no_libass_only_stock_filters() {
        // the listing's second column is the name (Homebrew's ffmpeg 9 lists no subtitles filter)
        // Homebrew's ffmpeg 9 has no libass (no subtitles filter): with captions drawn by am2Strata it composes all the same
        let filters = " T.. xfade             VV->V      Cross fade\n ... zoompan  V->V  zoom\n ... afade A->A fade\n ... tpad V->V pad\n ... fade V->V fade\n ... deflicker V->V even\n ... framerate V->V blend";
        let encoders = " V....D libx264  H.264\n A....D aac   AAC";
        assert_eq!(crate::media::missing_for_compose(filters, encoders), ["overlay"]);
        assert!(crate::media::missing_for_compose(&format!("{filters}\n ... overlay VV->V over"), encoders).is_empty());
    }

    /// The native captions burn in with the stock overlay filter: a held picture, a tone, a caption and a title card
    /// make a video of the right length whose caption frame has the white type in the low box.
    #[tokio::test]
    async fn native_captions_burn_in_without_libass() {
        if !crate::media::ffmpeg_available() { return; }
        let dir = std::env::temp_dir().join(format!("strata-natcap-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let (pic, wav, out) = (dir.join("p.mp4"), dir.join("t.wav"), dir.join("o.mp4"));
        run_ffmpeg(&["-y", "-f", "lavfi", "-i", "color=c=0x406080:s=640x360:r=30:d=4", "-c:v", "libx264", "-pix_fmt", "yuv420p", &pic.display().to_string()]).await.unwrap();
        run_ffmpeg(&["-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=4", &wav.display().to_string()]).await.unwrap();
        let ovs = crate::captions::overlays(640, 360, &[(1.0, 3.5, "Peace in our hands".into())], Some("Title"), None, 4.0).unwrap();
        let mut args: Vec<String> = vec!["-y".into(), "-i".into(), pic.display().to_string(), "-i".into(), wav.display().to_string()];
        for (k, o) in ovs.iter().enumerate() {
            let f = dir.join(format!("c{k}.png"));
            std::fs::write(&f, &o.png).unwrap();
            args.extend(["-loop", "1", "-framerate", "30", "-t"].map(String::from));
            args.push(format!("{:.3}", o.b - o.a));
            args.extend(["-i".to_string(), f.display().to_string()]);
        }
        args.extend(["-filter_complex".into(), crate::captions::overlay_filter(&ovs, 2, 4.0, 0.0), "-map".into(), "[v]".into(), "-map".into(), "[a]".into(),
                     "-t".into(), "4".into(), "-c:v".into(), "libx264".into(), "-c:a".into(), "aac".into(), out.display().to_string()]);
        run_ffmpeg(&args.iter().map(String::as_str).collect::<Vec<_>>()).await.unwrap();
        let len = probe_seconds(&out).await.unwrap();
        assert!((len - 4.0).abs() < 0.1, "{len}");
        // the frame at 2 s: the caption box low in the picture, with white type in it
        let shot = dir.join("f.png");
        run_ffmpeg(&["-y", "-ss", "2", "-i", &out.display().to_string(), "-frames:v", "1", "-update", "1", &shot.display().to_string()]).await.unwrap();
        let img = image::open(&shot).unwrap().to_rgb8();
        let white_low = (250..340).flat_map(|y| (0..640).map(move |x| (x, y))).filter(|&(x, y)| img.get_pixel(x, y)[0] > 220).count();
        let white_high = (0..100).flat_map(|y| (0..640).map(move |x| (x, y))).filter(|&(x, y)| img.get_pixel(x, y)[0] > 220).count();
        assert!(white_low > 100 && white_high == 0, "type low: {white_low}, high: {white_high}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_short_clip_fills_its_whole_shot() {
        if !crate::media::ffmpeg_available() { return; }
        let dir = std::env::temp_dir().join(format!("strata-clip-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let (clip, out) = (dir.join("c.mp4"), dir.join("o.mp4"));
        run_ffmpeg(&["-y", "-f", "lavfi", "-i", "testsrc=s=320x240:r=16:d=1", "-c:v", "libx264", &clip.display().to_string()]).await.unwrap();
        let frames = frames_for(4.0);
        run_ffmpeg(&["-y", "-i", &clip.display().to_string(), "-vf", &clip_filter(160, 90, 4.0, Some(1.0)), "-frames:v", &frames.to_string(),
                     "-an", "-c:v", "libx264", &out.display().to_string()]).await.unwrap();
        let got = probe_seconds(&out).await.unwrap();
        assert!((got - 4.0).abs() < 0.05, "a 1 s clip plays its whole 4 s shot: {got}");
        // slowed to 1.6 s of motion, then held only for the rest
        assert!(clip_filter(160, 90, 4.0, Some(1.0)).contains("setpts=1.6000*PTS"));
        assert!(clip_filter(160, 90, 4.0, Some(5.0)).contains("setpts=1.0000*PTS"), "a long clip is not slowed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The crossfade chain is a filtergraph ffmpeg accepts: three shots joined then faded in (the chains were once run
    /// together without ';', which ffmpeg refused as "trailing garbage"). Only runs where ffmpeg is installed.
    #[test]
    fn the_crossfade_chain_parses_in_ffmpeg() {
        let chain = xfade_chain(&[2.0, 2.0, 2.0], 0.5);
        assert_eq!(chain.matches(';').count(), 2, "{chain}");
        if !crate::media::ffmpeg_available() { return; }
        let mut args: Vec<String> = Vec::new();
        for _ in 0..3 { args.extend(["-f", "lavfi", "-i", "color=c=black:s=64x64:r=30:d=2"].map(String::from)); }
        args.extend(["-filter_complex".into(), format!("{chain}fade=t=in:st=0:d=1.0,format=yuv420p[out]"),
                     "-map".into(), "[out]".into(), "-f".into(), "null".into(), "-".into()]);
        let out = std::process::Command::new(crate::server_tls::program("ffmpeg")).args(["-hide_banner", "-loglevel", "error"]).args(&args).output().unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    }
    use super::*;

    #[test]
    fn shots_are_spread_over_the_audio_length_with_crossfades() {
        // three shots, 9 s of audio, 0.8 s crossfade: their durations add to 9 + 2×0.8 (two joins), each equal
        let d = spread_over(9.0, 0.8, 3);
        assert_eq!(d.len(), 3);
        assert!((d[0] - (9.0 + 2.0 * 0.8) / 3.0).abs() < 1e-6, "{d:?}");
        // joining them back with a crossfade between neighbours plays the whole audio
        assert!((total_length(&d, 0.8) - 9.0).abs() < 1e-6, "spreads to fill the audio: {}", total_length(&d, 0.8));
        // the offsets start at 0 and each moves one crossfade before the previous shot ends
        let off = join_offsets(&d, 0.8);
        assert_eq!(off[0], 0.0);
        assert!((off[1] - (d[0] - 0.8)).abs() < 1e-6 && (off[2] - (d[0] + d[1] - 1.6)).abs() < 1e-6, "{off:?}");
        // more shots of the whole length are each shorter; none is zero even for a tiny clip
        let short = spread_over(2.0, 0.8, 4);
        assert!(short.iter().all(|s| *s > 0.0));
    }

    #[test]
    fn the_size_is_parsed_and_1080x1920_is_vertical() {
        assert_eq!(parse_size("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_size("1080x1920"), Some((1080, 1920)), "a Short is taller than wide");
        assert_eq!(parse_size("   720 x 1280 "), Some((720, 1280)));
        assert!(parse_size("1024x7").is_none()); // a side below the limit is refused; make() defaults/refuses it
        assert!(parse_size("99999x512").is_none()); // a side over the limit is refused by parse_size
        assert_eq!(parse_size("square"), None); // not WIDTHxHEIGHT is refused; make() then defaults it
    }

    #[test]
    fn the_captions_are_written_as_ass_for_libass() {
        let captions = vec![(0.5, 2.5, "Hello world".to_string()), (3.0, 5.0, "made by am2Strata".to_string())];
        let ass = ass_script(1920, 1080, &captions, Some("Peace and Harmony"), Some("am2Strata · free and open source"), 60.0);
        // ASS's own line format: "Dialogue: Layer,Start,End,Style,...", so libass shows them (a "0=" was skipped)
        assert!(ass.contains("Dialogue: 0,0:00:00.50,0:00:02.50,Cap,"), "{ass}");
        assert!(ass.contains("Dialogue: 0,0:00:50.00,0:01:00.00,Ttl,"), "the end card is the last 10 s: {ass}");
        assert!(!ass.contains("Dialogue: 0="), "{ass}");
        let two = ass_script(1280, 720, &[(0.0, 1.0, "line one\nline {two}".to_string())], None, None, 5.0);
        assert!(two.contains("line one\\Nline (two)"), "newlines become \\N and braces cannot start an override: {two}");
        assert!(ass.starts_with("[Script Info]"), "a valid ASS starts with the script header");
        assert!(ass.contains("Style: Cap,DejaVu Sans,"), "the caption style is centred low");
        assert!(ass.contains("0:00:00.50") && ass.contains("0:00:02.50"), "each caption carries its start and end");
        assert!(ass.contains("\\fad(250,250)"), "a caption fades in and out so it does not snap");
        assert!(ass.contains("Peace and Harmony") && ass.contains("am2Strata · free and open source"), "the cards are burned in too");
    }

    #[test]
    fn a_missing_input_is_refused_with_a_clear_message() {
        // no audio at all: the request cannot be understood without it
        assert!(compose_request(&json!({"shots": [{"image": "i.png"}]}))
            .unwrap_err().contains("audio"), "audio is required");
        // an audio but a shot with neither image nor clip is refused plainly
        let bad = compose_request(&json!({"audio": "mj-1", "shots": [json!({"foo": "bar"})]}))
            .unwrap_err();
        assert!(bad.contains("image") || bad.contains("clip"), "a shot needs image or clip: {bad}");
    }

    /// Parse and validate a compose request the way the endpoint does, without touching ffmpeg (returns Err with a clear
    /// message when something is missing).
    fn compose_request(req: &Value) -> Result<(u32, u32, Vec<f64>, usize), String> {
        // the audio is required; without it there is nothing to compose to (mirrors make())
        if req.get("audio").and_then(Value::as_str).is_none_or(|s| s.trim().is_empty()) {
            return Err("error: audio is required".into());
        }
        let (w, h) = parse_size(req["size"].as_str().unwrap_or("1920x1080"))
            .ok_or_else(|| format!("error: size {} does not work", req["size"].as_str().unwrap_or("")))?;
        let shots = req["shots"].as_array().ok_or("shots: a list of {\"image\"|\"clip\": ...}")?;
        if shots.is_empty() { return Err("shots: at least one shot is needed".into()); }
        for (i, sh) in shots.iter().enumerate() {
            if sh.get("image").or_else(|| sh.get("clip")).and_then(Value::as_str).is_none_or(|s| s.trim().is_empty()) {
                return Err(format!("shots[{i}]: give an \"image\" or a \"clip\""));
            }
        }
        // all shots given a length use it, else they are spread over the (here unknown) audio; assume even here for parse
        let durations = if shots.iter().all(|s| s["seconds"].as_f64().is_some()) {
            shots.iter().map(|s| s["seconds"].as_f64().unwrap_or(5.0)).collect()
        } else {
            vec![5.0; shots.len()]
        };
        Ok((w, h, durations, shots.len()))
    }

    /// A real 3-second video from one picture and a tone, made the way `make()` composes it: one image played with its
    /// centred zoom, captions burned in, the audio muxed under it. Only runs with ffmpeg present (hidden behind
    /// `--ignored`, since `cargo test` has no ffmpeg here), but otherwise exercises the real filter chain end to end.
    #[tokio::test]
    #[ignore = "needs ffmpeg and ffprobe"]
    async fn a_picture_and_a_tone_become_a_three_second_video() {
        if !crate::media::ffmpeg_available() { return; }
        let dir = std::env::temp_dir().join(format!("strata-compose-{}-{}", std::process::id(), uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        // a still picture and a 3 s tone, built with ffmpeg itself so the test needs no image library
        let png = dir.join("pic.png");
        let wav = dir.join("tone.wav");
        let out = dir.join("out.mp4");
        run_ffmpeg(&["-y", "-lavfi", "color=c=lightblue:s=1600x1200:r=1", "-frames:v", "1", &png.display().to_string()]).await.unwrap();
        run_ffmpeg(&["-y", "-lavfi", "sine=frequency=440:duration=3.0", "-t", "3.0", &wav.display().to_string()]).await.unwrap();

        // pass 1: the image as one zooming shot (2.7 s, then held by the join) — the same zoom filter make() uses
        let shot = dir.join("shot-00.mp4");
        run_ffmpeg(&[
            "-y", "-loop", "1", "-framerate", "30", "-i", &png.display().to_string(),
            "-vf", "scale=3200:2400:force_original_aspect_ratio=increase,crop=3200:2400,zoompan=z='1.12*on/79':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080:fps=30,format=yuv420p",
            "-frames:v", "81", "-an", "-c:v", "libx264", "-crf", "18", &shot.display().to_string(),
        ]).await.unwrap();

        // pass 2: hold it to 3 s, burn in a caption with make()'s own ASS script and filter, and mux the tone (faststart)
        let subs = dir.join("subs.ass");
        std::fs::write(&subs, ass_script(1920, 1080, &[(0.2, 2.5, "Hello there".into())], Some("Title"), None, 3.0)).unwrap();
        // the held picture first (make() joins its shots the same way), then make()'s own final filter
        let held = dir.join("held.mp4");
        run_ffmpeg(&["-y", "-i", &shot.display().to_string(), "-vf", "tpad=stop_mode=clone:stop_duration=1", "-c:v", "libx264", &held.display().to_string()]).await.unwrap();
        let fc = final_filter(&subs.display().to_string().replace('\\', "\\\\").replace(':', "\\:"), 3.0, 0.0);
        run_ffmpeg(&[
            "-y", "-i", &held.display().to_string(), "-i", &wav.display().to_string(),
            "-filter_complex", &fc, "-map", "[v]", "-map", "[a]",
            "-t", "3.0", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            &out.display().to_string(),
        ]).await.unwrap();

        assert!(out.is_file(), "the MP4 was written");
        let probe = TokioCommand::new(crate::server_tls::program("ffprobe")).args(["-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1"])
            .arg(&out).output().await.unwrap();
        let len: f64 = String::from_utf8_lossy(&probe.stdout).trim().parse().unwrap_or(0.0);
        assert!((len - 3.0).abs() < 0.25, "the video plays for about three seconds: {len}");
    }

    #[test]
    fn request_text_that_broke_the_filter_now_stays_inside_it() {
        // the reviewer's test vectors round-trip with only the two ffmpeg-special characters escaped, so an apostrophe can't
        // close the quoted value and a backslash can't escape it, which is how `textfile=/etc/passwd` injection worked before
        assert_eq!(drawtext_escape("It's"), "It\\'s", "an apostrophe is quoted");
        assert_eq!(drawtext_escape("\\"), "\\\\", "a backslash is doubled so it cannot open a new escape");
        assert_eq!(drawtext_escape("\\'"), "\\\\\\'", "a backslash then an apostrophe keeps its shape without breaking out");
        // these render as plain caption text: ffmpeg never treats them as filter options or escapes inside the quotes
        assert_eq!(drawtext_escape("a:b"), "a:b");
        assert_eq!(drawtext_escape("%{pts}"), "%{pts}");
    }

    #[test]
    fn drawtext_leaves_no_empty_field_when_the_font_is_absent() {
        // no usable Outfit font: the first option is text, so the filter starts drawtext=text= (no leading colon after =)
        assert!(thumbnail_drawtext("Hi", "", "ffffff", 100).starts_with("drawtext=text="),
                "absent font -> the filter starts drawtext=text=: {}", thumbnail_drawtext("Hi", "", "ffffff", 100));
        // a found Outfit font leads with its (escaped, quoted) fontfile then text; backslashes/paths stay valid inside quotes
        let f = thumbnail_drawtext("Hi", "C:\\a b\\outfit.woff2", "ffd65a", 100);
        assert!(f.starts_with("drawtext=fontfile='C:\\\\a b\\\\outfit.woff2':text='Hi'"),
                "found font -> drawtext=fontfile=...:text=: {f}");
    }

    #[test]
    fn an_untagged_raw_mp3_stream_is_audio() {
        // a raw MPEG audio frame stream is recognised by its sync: FF followed by a header byte in 0xF0-0xFF (LAME's VBR tag
        // starts FF FB; a CBR MPEG-1 Layer III frame starts FF F3). JPEG's SOI, FF D8, keeps top nibble 0xD and stays an image.
        assert_eq!(kind_of_bytes(&[0xFF, 0xFB, 0x90, 0x00]), "audio");
        assert_eq!(kind_of_bytes(&[0xFF, 0xF3, 0xFF, 0x90]), "audio");
        assert_eq!(kind_of_bytes(&[0xFF, 0xD8, 0xFF, 0xE0]), "image", "a JPEG starts FF D8, not an MP3");
    }
}
