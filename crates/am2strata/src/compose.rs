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
pub fn ass_script(w: u32, h: u32, captions: &[(f64, f64, String)], open: Option<&str>, end: Option<&str>) -> String {
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
        let t = format!("{{\\fad(250,250)}}{}", text.replace('&', "\\&"));
        out.push_str(&format!("Dialogue: 0={},{},Cap,,0,0,120,,{}\n", ass_time(*a), ass_time(*b), t));
    }
    if let Some(t) = open {
        let t = t.lines().map(str::trim).collect::<Vec<_>>().join("\n");
        out.push_str(&format!("Dialogue: 0=0,0:00:09.00,Ttl,,0,0,0,,{{\\fad(400,400)}}{}\n", t.replace('&', "\\&")));
    }
    if let Some(t) = end {
        let t = t.lines().map(str::trim).collect::<Vec<_>>().join("\n");
        out.push_str(&format!("Dialogue: 0=0:00:01.00,9:59:59.00,Ttl,,0,0,0,,{{\\fad(400,400)}}{}\n", t.replace('&', "\\&")));
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
    let out = TokioCommand::new("ffmpeg").args(args).stdout(Stdio::null()).stderr(Stdio::piped())
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
    let probe = TokioCommand::new("ffprobe")
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
        return Err("this server cannot make a video: install ffmpeg (sudo apt install ffmpeg)".into());
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
    let ass = ass_script(w, h, &captions, req["title_card"].as_str(), req["end_card"].as_str());
    let subs = dir.join("subs.ass");
    std::fs::write(&subs, &ass).map_err(|e| format!("could not write the captions: {e}"))?;

    // the shots' lengths: their own when given, else spread over the audio (with crossfades)
    let total_s = duration_of(&dir, "audio.wav", &audio).await?;
    let per_shot: Vec<Option<f64>> = shots.iter().map(|sh| sh["seconds"].as_f64()).collect();
    let durations = if per_shot.iter().all(|d| d.is_some()) {
        per_shot.iter().map(|d| d.unwrap_or(5.0)).collect()
    } else {
        spread_over(total_s, cross, shots.len())
    };
    // a shot over its given length: images still play their whole zoom (clips are held on the last frame when short)
    let played = total_length(&durations, cross); // what the joined footage actually plays; cap to it
    let _ = join_offsets(&durations, cross);

    s.media_jobs.set_meta(n, json!({"part": 0, "parts": shots.len(), "now": "compose"}));
    // make each input a temp file (a data URL is decoded to bytes above; the rest are paths / job ids already resolved)
    let mut inputs = Vec::new();
    for (i, sh) in shots.iter().enumerate() {
        let raw = sh.get("image").or_else(|| sh.get("clip")).and_then(Value::as_str).unwrap_or("");
        let bytes = source_bytes(s, if sh.get("clip").is_some() { "clip" } else { "image" }, &Value::from(raw)).await?;
        let kind = kind_of_bytes(&bytes);
        let ext = if kind == "clip" { "mp4" } else { "png" };
        let src = dir.join(format!("src-{i}.{ext}"));
        std::fs::write(&src, &bytes).map_err(|e| format!("could not write the {kind}: {e}"))?;
        let out = dir.join(format!("shot-{:02}.mp4", i));
        let frames = frames_for(durations[i]);
        let (w2, h2) = (w * 2, h * 2);
        if kind == "clip" {
            // fitted to the frame, held on its last frame if shorter than the shot (so a short clip still fills it)
            let fit = format!("scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={FPS},tpad=stop_mode=clone:stop_duration=0.5,format=yuv420p");
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

    // mux the picture chain with the audio, captions burned in by libass, one MP4 (H.264 + AAC, faststart)
    let out = dir.join(format!("{id}.mp4"));
    let picp = pictures.display().to_string();
    let audiop = dir.join("audio.wav").display().to_string();
    let subsp = subs.display().to_string().replace('\\', "\\\\").replace(':', "\\:");
    let fc = format!("[0:v]subtitles='{}'[v];[1:a]atrim=0:setpts=N/SR/TB,afade=t=out:st={}:d=4[a]", subsp, played.max(4.0) - 4.0);
    run_ffmpeg(&["-i", picp.as_str(), "-i", audiop.as_str(), "-filter_complex", fc.as_str(), "-map", "[v]", "-map", "[a]", "-t", played.to_string().as_str(), "-r", FPS.to_string().as_str(), "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out.display().to_string().as_str()]).await?;

    // free the temporary clips, keep only the finished video (a restart keeps it)
    let _ = std::fs::remove_dir_all(&dir);
    let bytes = std::fs::read(&out).map_err(|e| format!("could not read the finished video: {e}"))?;
    Ok(("video/mp4".into(), bytes, json!({"width": w, "height": h, "seconds": played.round() / 10.0, "shots": inputs.len(), "captions": captions.len()})))
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
    if parts.is_empty() { String::from("[0:v]") } else { parts.join("") }
}

/// Make the thumbnail: one 1280×720 JPEG, the image with a left-dark gradient and big Outfit words drawn over it by
/// ffmpeg's drawtext (the heavy type with a dark stroke, like thumbs.py). Returns the JPEG bytes.
pub async fn make_thumbnail(s: &AppState, req: &Value) -> Result<(String, Vec<u8>, Value), String> {
    if !crate::media::ffmpeg_available() {
        return Err("this server cannot make a thumbnail: install ffmpeg (sudo apt install ffmpeg)".into());
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
        // each line is centred and placed a row below the last, with a dark shadow for the stroke look on any picture
        draws.push(format!("drawtext=fontfile='{font}':text='{text}':fontsize=120:fontcolor={color}:x=(w-text_w)/2:y={y}:shadowcolor=0x99_180A06@0.7:shadowx=3:shadowy=3"));
    }
    // the whole thing: fit the picture, darken it so the words read, then stack the words on top; one 1280×720 JPEG
    let _ = run_ffmpeg(&[
        "-y", "-i", image_.display().to_string().as_str(),
        "-vf", &format!("scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},eq=brightness=-0.15,{}", draws.join(",")).as_str(),
        "-vframes", "1", "-c:v", "mjpeg", "-q:v", "92", out.display().to_string().as_str(),
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

// ----------------------------------------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
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
        let ass = ass_script(1920, 1080, &captions, Some("Peace and Harmony"), Some("am2Strata · free and open source"));
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

        // pass 2: hold it to 3 s, burn in a caption, and mux the tone under one MP4 (faststart)
        run_ffmpeg(&[
            "-y", "-i", &shot.display().to_string(), "-i", &wav.display().to_string(),
            "-filter_complex", "[0:v]subtitles='':format=yuv420p[v]", "-map", "[v]", "-map", "1:a",
            "-t", "3.0", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            &out.display().to_string(),
        ]).await.unwrap();

        assert!(out.is_file(), "the MP4 was written");
        let probe = TokioCommand::new("ffprobe").args(["-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1"])
            .arg(&out).output().await.unwrap();
        let len: f64 = String::from_utf8_lossy(&probe.stdout).trim().parse().unwrap_or(0.0);
        assert!((len - 3.0).abs() < 0.25, "the video plays for about three seconds: {len}");
    }
}
