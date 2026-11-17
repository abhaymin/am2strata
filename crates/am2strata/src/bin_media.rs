//! `am2strata-run media <action> ...`: a thin HTTP client of the server's own media API (docs/MEDIA.md). Every big
//! feature is a web app screen and a CLI command; these are the ones the web app never got a card for. The request
//! building here is kept in small pure functions so it can be unit-tested without a server, and `media_cmd` reads each
//! handler (image_gen.rs, audio_api.rs, music.rs, media_check.rs, media_dub.rs, media_jobs.rs) to send the fields it
//! expects and write or print the answer.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use serde_json::{json, Value};

use crate::{expand_path, local_server};
use clap::Args;

type RB = reqwest::blocking::RequestBuilder;

// ----------------------------------------------------------------------- request builders (pure)

/// `media image "<prompt>"`: OpenAI's /v1/images/generations body. The model's defaults fill size, steps, seed …; only
/// the fields given here are sent, so a request without --size/--model/--n is just {prompt} and the server fills the
/// rest (default 1024x1024, one image).
pub fn image_request(prompt: &str, size: Option<&str>, model: Option<&str>, n: u32) -> Value {
    let mut r = json!({"prompt": prompt});
    if let Some(s) = size { r["size"] = json!(s); }
    if let Some(m) = model { r["model"] = json!(m); }
    if n > 1 { r["n"] = json!(n); }
    r
}

/// `media video "<prompt>"`: /v1/videos/generations. frames, size, format and model are all optional (the model's own
/// defaults fill them; format defaults to mp4 when this engine makes it). Only what is given is sent.
pub fn video_request(prompt: &str, frames: Option<u32>, size: Option<&str>, format: Option<&str>, model: Option<&str>) -> Value {
    let mut r = json!({"prompt": prompt});
    if let Some(f) = frames { r["frames"] = json!(f); }
    if let Some(s) = size { r["size"] = json!(s); }
    if let Some(f) = format { r["format"] = json!(f); }
    if let Some(m) = model { r["model"] = json!(m); }
    r
}

/// `media speak "<text>"`: /v1/audio/speech. response_format is wav only here, so it is left out (the server makes wav).
pub fn speak_request(text: &str, voice: Option<&str>, model: Option<&str>) -> Value {
    let mut r = json!({"input": text});
    if let Some(v) = voice { r["voice"] = json!(v); }
    if let Some(m) = model { r["model"] = json!(m); }
    r
}

/// `media song "<prompt>"`: /v1/audio/music, which ACE-Step reads as a caption/style (OpenAI has no such endpoint). The
/// prompt is the style; lyrics and duration come from --lyrics / --seconds when given.
pub fn song_request(prompt: &str, lyrics: Option<&str>, seconds: Option<u32>) -> Value {
    let mut r = json!({"style": prompt});
    if let Some(l) = lyrics { r["lyrics"] = json!(l); }
    if let Some(s) = seconds { r["duration"] = json!(s); }
    r
}

/// The non-file fields of a transcription (OpenAI's multipart: `file`, `model`, `response_format`, `language`).
/// `translate` does not change a field: it changes the endpoint (/v1/audio/transcriptions vs /v1/audio/translations).
pub fn transcribe_fields(model: Option<&str>, response_format: Option<&str>, language: Option<&str>) -> Vec<(&'static str, String)> {
    let mut f = vec![("model", model.unwrap_or("whisper-1").to_string())];
    if let Some(r) = response_format { f.push(("response_format", r.to_string())); }
    if let Some(l) = language { f.push(("language", l.to_lowercase())); }
    f
}

/// `media check <file>`: the JSON body for /v1/media/check. The file goes in as a base64 data URL; expected_text and
/// language are optional (without heard_text the take is transcribed on the server).
pub fn check_body(audio_data_url: &str, expected: Option<&str>, language: Option<&str>) -> Value {
    let mut r = json!({"audio": audio_data_url});
    if let Some(e) = expected { r["expected_text"] = json!(e); }
    if let Some(l) = language { r["language"] = json!(l); }
    r
}

/// Where the images go: `out` for the first, `out-2.<ext>`, `out-3.<ext>` … (file.png, file-2.png, …). For n == 1 it is
/// just [out], and when out has no extension each later one keeps the original name.
pub fn out_names(out: &str, n: u32) -> Vec<String> {
    if n <= 1 { return vec![out.to_string()]; }
    let base = Path::new(out);
    let stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or(out).to_string();
    let ext = base.extension().and_then(|e| e.to_str()).map(str::to_string);
    let mut v = vec![out.to_string()];
    for i in 2..=n {
        match &ext {
            // in the same folder as `out` (pics/cat.png -> pics/cat-2.png), not the current one
            Some(e) => v.push(base.with_file_name(format!("{stem}-{i}.{e}")).to_string_lossy().into_owned()),
            None => v.push(format!("{out}-{i}")),
        }
    }
    v
}

/// A small file's bytes (error names the file), and its file name for a multipart upload.
fn read_bytes(path: &Path) -> Result<(Vec<u8>, String), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let name = Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or("upload").to_string();
    Ok((bytes, name))
}

/// A file's bytes as a base64 data URL for a JSON body (the check endpoint reads it this way), with the mime guessed
/// from the extension.
fn data_url(path: &Path) -> Result<String, String> {
    let (bytes, _) = read_bytes(path)?;
    use base64::Engine as _;
    let mime = match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "wav" => "audio/wav", "png" => "image/png", "jpg" | "jpeg" => "image/jpeg", "webp" => "image/webp",
        "mp4" => "video/mp4", "mov" => "video/quicktime", _ => "application/octet-stream",
    };
    Ok(format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(&bytes)))
}

/// Write `bytes` to `path`, expanding a leading ~.
fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// Parse a request from `first` as JSON, whether it is a file on disk or inline JSON (like browser_cmd's parse).
fn parse_request(first: &Option<String>, what: &str) -> Result<Value, String> {
    let t = first.as_ref().ok_or_else(|| format!("{what} needs a request (a file or inline JSON)"))?;
    let text = if Path::new(t).is_file() { std::fs::read_to_string(t).map_err(|e| e.to_string())? } else { t.clone() };
    serde_json::from_str::<Value>(&text).map_err(|e| format!("{what} is not valid JSON: {e}"))
}

/// `am2strata-run media <action> ...`: /v1/images/generations, /v1/videos/generations, /v1/audio/speech and
/// transcriptions/translations, /v1/audio/music, /v1/media/check (a take or video), /v1/media/dub and
/// /v1/media/productions, plus GET /v1/media/jobs/{id}. The actions:
///   image "<prompt>" [--size WxH] [--model M] [--n N] --out file.png
///   video "<prompt>" [--frames N] [--size WxH] [--format mp4|gif|apng] [--model M] --out file
///   speak "<text>" [--voice V] [--model M] --out file.wav
///   transcribe <file> [--language xx] [--format json|text|srt|vtt|verbose_json] [--translate] [--out file]
///   song "<prompt>" [--lyrics file.txt] [--seconds N] --out file.wav
///   check <file> [--expect "text"] [--language xx]       (prints each finding; exit 1 when any is not ok)
///   dub <request.json> [--out file.wav]
///   produce <plan.json> [--out file.wav]                  (posts, polls every 3 s for step N of M, then downloads the content)
///   job <id> [--out file]                                 (the job status; with --out also its /content)
///   files [image|video|audio]                             (the media library: every file made or sent, newest first)
///   add <file>                                            (puts a file in the library; prints its id)
///   get <id> --out file                                   (a library file)
///   rm <id>                                               (removes a library file)
#[derive(Args)]
pub struct MediaOpts {
    /// the answer's file (image, video, speech, song, dub and job --out all need it; refuse a missing one)
    #[arg(long)]
    out: Option<String>,
    /// widthxheight (image, video); e.g. 1024x1024
    #[arg(long)]
    size: Option<String>,
    /// the model's id (image, video, speech)
    #[arg(long)]
    model: Option<String>,
    /// how many images to draw (image), 1 to 4
    #[arg(long, default_value_t = 1)]
    n: u32,
    /// frames (video), rounded to what the model takes
    #[arg(long, default_value_t = 33)]
    frames: u32,
    /// video format: mp4 (default), gif or apng
    #[arg(long)]
    format: Option<String>,
    /// voice name (speech)
    #[arg(long)]
    voice: Option<String>,
    /// transcription format: json, text, srt, vtt or verbose_json (default json)
    #[arg(long)]
    response_format: Option<String>,
    /// language as a code (transcribe, speech, check): ISO 639-1
    #[arg(long)]
    language: Option<String>,
    /// translate the transcription to English (uses /v1/audio/translations)
    #[arg(long)]
    translate: bool,
    /// lyrics file for a song; its text is read from it
    #[arg(long)]
    lyrics: Option<String>,
    /// song length in seconds (10 to 600)
    #[arg(long, default_value_t = 60)]
    seconds: u32,
    /// expected words for the check, so words_heard can be measured
    #[arg(long)]
    expect: Option<String>,
    /// print the raw server answer as JSON instead of a friendly rendering
    #[arg(long)]
    json: bool,
}

/// `am2strata-run media <action> ...`: /v1/images/generations, /v1/videos/generations, /v1/audio/speech and
/// transcriptions/translations, /v1/audio/music, /v1/media/check (a take or video), /v1/media/dub and
/// /v1/media/productions, plus GET /v1/media/jobs/{id}. The actions:
///   image "<prompt>" [--size WxH] [--model M] [--n N] --out file.png
///   video "<prompt>" [--frames N] [--size WxH] [--format mp4|gif|apng] [--model M] --out file
///   speak "<text>" [--voice V] [--model M] --out file.wav
///   transcribe <file> [--language xx] [--format json|text|srt|vtt|verbose_json] [--translate] [--out file]
///   song "<prompt>" [--lyrics file.txt] [--seconds N] --out file.wav
///   check <file> [--expect "text"] [--language xx]       (prints each finding; exit 1 when any is not ok)
///   dub <request.json> [--out file.wav]
///   produce <plan.json> [--out file.wav]                  (posts the plan, polls step N of M every 3 s, then downloads the content)
///   job <id> [--out file]                                 (the job status; with --out also its /content)
///   files [image|video|audio]                             (the media library: every file made or sent, newest first)
///   add <file>                                            (puts a file in the library; prints its id)
///   get <id> --out file                                   (a library file)
///   rm <id>                                               (removes a library file)
#[allow(clippy::too_many_arguments)]
pub fn media_cmd(action: &str, first: Option<String>, o: &MediaOpts, url: Option<String>, api_key: Option<String>, models_dir: &str) -> ExitCode {
    let (local_url, local_key) = local_server(models_dir);
    let url = url.or_else(|| std::env::var("AM2STRATA_URL").ok()).unwrap_or(local_url).trim_end_matches('/').to_string();
    let key = api_key.or_else(|| std::env::var("AM2STRATA_API_KEY").ok()).or(local_key);
    eprintln!("[am2strata] media {action}: {url}"); // on stderr: stdout is the answer, so it can be piped
    let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(1800)).build().unwrap();

    // a server error is turned into its message (the same send every other command uses)
    let send = |rb: RB| -> Result<reqwest::blocking::Response, String> {
        let key = &key;
        let rb = match key { Some(k) => rb.bearer_auth(k), None => rb };
        let r = rb.send().map_err(|e| format!("{url} does not answer ({e}); is the server running?"))?;
        if r.status().is_success() { return Ok(r); }
        let j: Value = r.json().unwrap_or_default();
        Err(j["error"]["message"].as_str().unwrap_or("failed").to_string())
    };

    let r = (|| -> Result<(), String> {
        match action {
            "image" => cmd_image(&client, &send, first.as_deref(), o, &url)?,
            "video" => cmd_video(&client, &send, first.as_deref(), o, &url)?,
            "speak" => cmd_speak(&client, &send, first.as_deref(), o, &url)?,
            "song"  => cmd_song(&client, &send, first.as_deref(), o, &url)?,
            "transcribe" => cmd_transcribe(&client, &send, first.as_ref(), o, &url)?,
            "check" => cmd_check(&client, &send, first.as_ref(), o, &url)?,
            "dub" => cmd_dub(&client, &send, &first, o, &url)?,
            "produce" => cmd_produce(&client, &send, &first, o, &url)?,
            "job" => cmd_job(&client, &send, first.as_deref(), o, &url)?,
            "files" | "add" | "get" | "rm" => cmd_files(&client, &send, action, first.as_deref(), o, &url)?,
            other => return Err(format!("unknown action {other:?}: image, video, speak, transcribe, song, check, dub, produce, job, files, add, get, rm")),
        }
        Ok(())
    })();

    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("{e}"); ExitCode::from(1) }
    }
}

// ----------------------------------------------------------------------------------------------------- image / video

fn cmd_image(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<reqwest::blocking::Response, String>,
             prompt: Option<&str>, o: &MediaOpts, url: &str) -> Result<(), String> {
    let out = require_out(&o.out)?;
    let body = image_request(prompt.ok_or("image needs a prompt")?, o.size.as_deref(), o.model.as_deref(), o.n);
    let r = send(client.post(format!("{url}/v1/images/generations")).json(&body))?;
    let j: Value = r.json().map_err(|e| e.to_string())?;
    let data = j["data"].as_array().ok_or_else(|| "no images in the answer".to_string())?;
    let names = out_names(out.to_str().unwrap(), data.len().max(1) as u32);
    if names.len() < data.len() { return Err("more images came back than --out can hold".into()); }
    use base64::Engine as _;
    for (i, item) in data.iter().enumerate() {
        let raw = item["b64_json"].as_str().unwrap_or("");
        let b64 = raw.split_once(";base64,").map(|(_, b)| b).unwrap_or(raw);
        let bytes = base64::engine::general_purpose::STANDARD.decode(b64.trim()).map_err(|e| e.to_string())?;
        write_file(Path::new(&names[i]), &bytes)?;
    }
    let ids: Vec<&str> = data.iter().filter_map(|d| d["file_id"].as_str()).collect();
    println!("saved {}{}", names[0].clone(), if ids.is_empty() { String::new() } else { format!(" (library {})", ids.join(", ")) });
    Ok(())
}

fn cmd_video(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<reqwest::blocking::Response, String>,
             prompt: Option<&str>, o: &MediaOpts, url: &str) -> Result<(), String> {
    let out = require_out(&o.out)?;
    let body = video_request(prompt.ok_or("video needs a prompt")?, Some(o.frames), o.size.as_deref(), o.format.as_deref(), o.model.as_deref());
    let r = send(client.post(format!("{url}/v1/videos/generations")).json(&body))?;
    let j: Value = r.json().map_err(|e| e.to_string())?;
    let raw = j["data"][0]["b64"].as_str().unwrap_or("");
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD.decode(raw.trim_end_matches(';')).map_err(|e| e.to_string())?;
    write_file(&out, &bytes)?;
    println!("saved {} ({} KB)", out.display(), bytes.len() / 1024);
    Ok(())
}

// ----------------------------------------------------------------------------------------------------- speech / song

fn cmd_speak(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<reqwest::blocking::Response, String>,
             prompt: Option<&str>, o: &MediaOpts, url: &str) -> Result<(), String> {
    let out = require_out(&o.out)?;
    let body = speak_request(prompt.ok_or("speak needs text")?, o.voice.as_deref(), o.model.as_deref());
    let r = send(client.post(format!("{url}/v1/audio/speech")).json(&body))?;
    let bytes = r.bytes().map_err(|e| e.to_string())?;
    write_file(&out, &bytes)?;
    println!("saved {} ({} KB)", out.display(), bytes.len() / 1024);
    Ok(())
}

fn cmd_song(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<reqwest::blocking::Response, String>,
            prompt: Option<&str>, o: &MediaOpts, url: &str) -> Result<(), String> {
    let out = require_out(&o.out)?;
    let lyrics: Option<String> = match &o.lyrics {
        Some(path) => Some(std::fs::read_to_string(expand_path(Path::new(path))).map_err(|e| format!("{path}: {e}"))?),
        None => None,
    };
    let body = song_request(prompt.ok_or("song needs a prompt")?, lyrics.as_deref(), Some(o.seconds));
    let r = send(client.post(format!("{url}/v1/audio/music")).json(&body))?;
    let bytes = r.bytes().map_err(|e| e.to_string())?;
    write_file(&out, &bytes)?;
    println!("saved {} ({} KB)", out.display(), bytes.len() / 1024);
    Ok(())
}

// ------------------------------------------------------------------------------------------------------ transcribe

fn cmd_transcribe(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<reqwest::blocking::Response, String>,
                  first: Option<&String>, o: &MediaOpts, url: &str) -> Result<(), String> {
    use reqwest::blocking::multipart::{Form, Part};

    let (bytes, name) = read_bytes(&expand_path(Path::new(first.ok_or("transcribe needs a file")?)))?;
    let mut form = Form::new().part("file", Part::bytes(bytes).file_name(name));
    for (k, v) in transcribe_fields(o.model.as_deref(), o.response_format.as_deref(), o.language.as_deref()) {
        form = form.text(k, v);
    }
    let rb = client.post(if o.translate { format!("{url}/v1/audio/translations") } else { format!("{url}/v1/audio/transcriptions") }).multipart(form);
    print_transcript_text(send(rb)?, o.out.as_deref())
}

/// json/verbose_json answers are {"text": "…"}; text/srt/vtt answers are the body itself.
fn print_transcript_text(r: reqwest::blocking::Response, out: Option<&str>) -> Result<(), String> {
    let text = r.text().map_err(|e| e.to_string())?;
    let transcript = serde_json::from_str::<Value>(&text).ok()
        .and_then(|v| v["text"].as_str().map(str::to_string))
        .unwrap_or_else(|| text.trim_end_matches('\n').to_string());
    match out { Some(p) => write_file(&expand_path(Path::new(p)), transcript.as_bytes()).map_err(|e| e.to_string())?, None => println!("{transcript}") }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------- check

fn cmd_check(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<reqwest::blocking::Response, String>,
             first: Option<&String>, o: &MediaOpts, url: &str) -> Result<(), String> {
    let file = expand_path(Path::new(first.ok_or("check needs a file")?));
    let body = check_body(&data_url(&file)?, o.expect.as_deref(), o.language.as_deref());
    let r = send(client.post(format!("{url}/v1/media/check")).json(&body))?;
    let j: Value = r.json().map_err(|e| e.to_string())?;
    let mut problems = 0usize;
    for (family, obj) in [("audio", &j["audio"]), ("video", &j["video"])] {
        if let Some(checks) = obj.as_object() {
            for (name, check) in checks.iter() {
                if name == "note" || name == "error" { println!("{family}.{name}: {}", check.as_str().unwrap_or("")); continue; }
                let ok = check["ok"].as_bool();
                if matches!(ok, Some(false)) { problems += 1; }
                println!("{family}.{name}: {}", match ok { Some(true) => "pass", Some(false) => "FAIL", None => "n/a" });
            }
        }
    }
    if j["audio"].get("error").is_some() || j["video"].get("error").is_some() { problems = problems.max(1); }
    if problems > 0 { return Err(format!("{problems} check(s) not ok")); }
    Ok(())
}

// ----------------------------------------------------------------------------------------------------- dub / produce / job

fn cmd_dub(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<reqwest::blocking::Response, String>,
           first: &Option<String>, o: &MediaOpts, url: &str) -> Result<(), String> {
    let out = require_out(&o.out)?;
    let req = parse_request(first, "dub")?;
    let r = send(client.post(format!("{url}/v1/media/dub")).json(&req))?;
    let j: Value = r.json().map_err(|e| e.to_string())?;
    if o.json { println!("{}", serde_json::to_string_pretty(&j).map_err(|e| e.to_string())?); }
    // the soundtrack is a base64 WAV data URL in the answer
    let raw = j["soundtrack"].as_str().unwrap_or("").split_once(";base64,").map(|(_, b)| b).unwrap_or("");
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD.decode(raw.trim_end_matches(';')).map_err(|e| e.to_string())?;
    write_file(&out, &bytes)?;
    if !o.json {
        println!("saved {} ({} KB)", out.display(), bytes.len() / 1024);
        if let Some(report) = j["report"].as_object() {
            for a in report.get("actions").and_then(|x| x.as_array()).into_iter().flatten() {
                println!("line {}: {}", a["line"].as_u64().unwrap_or(0), a["action"].as_str().unwrap_or(""));
            }
        }
    }
    Ok(())
}

fn cmd_job(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<reqwest::blocking::Response, String>,
           id: Option<&str>, o: &MediaOpts, url: &str) -> Result<(), String> {
    let id = id.ok_or("job needs an id")?;
    let r = send(client.get(format!("{url}/v1/media/jobs/{id}")))?;
    let j: Value = r.json().map_err(|e| e.to_string())?;
    if o.json { println!("{}", serde_json::to_string_pretty(&j).map_err(|e| e.to_string())?); }
    else {
        println!("job {}: {} ({})", id, j["state"].as_str().unwrap_or("?"), j["kind"].as_str().unwrap_or(""));
        if let Some(e) = j["error"].as_str() { println!("  error: {e}"); }
    }
    if let Some(p) = &o.out {
        let c = send(client.get(format!("{url}/v1/media/jobs/{id}/content")))?;
        let bytes = c.bytes().map_err(|e| e.to_string())?;
        write_file(&expand_path(Path::new(p)), &bytes)?;
        println!("saved {} ({} KB)", Path::new(p).display(), bytes.len() / 1024);
    }
    Ok(())
}

/// The media library (GET/POST/DELETE /v1/media/files).
fn cmd_files(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<reqwest::blocking::Response, String>,
             action: &str, first: Option<&str>, o: &MediaOpts, url: &str) -> Result<(), String> {
    match action {
        "files" => {
            let q = first.map(|k| format!("?kind={k}")).unwrap_or_default();
            let j: Value = send(client.get(format!("{url}/v1/media/files{q}")))?.json().map_err(|e| e.to_string())?;
            if o.json {
                println!("{}", serde_json::to_string_pretty(&j).map_err(|e| e.to_string())?);
                return Ok(());
            }
            for f in j["data"].as_array().into_iter().flatten() {
                let what = f["prompt"].as_str().or(f["name"].as_str()).or(f["used_for"].as_str()).unwrap_or("");
                println!("{}  {:5} {:9} {:>8} KB  {}", f["id"].as_str().unwrap_or("?"), f["kind"].as_str().unwrap_or(""),
                         f["source"].as_str().unwrap_or(""), f["bytes"].as_u64().unwrap_or(0) / 1024, what.chars().take(60).collect::<String>());
            }
            println!("{} file(s)", j["total"]);
        }
        "add" => {
            let p = expand_path(Path::new(first.ok_or("add needs a file")?));
            let name = p.file_name().map(|n| n.to_string_lossy().into_owned());
            let j: Value = send(client.post(format!("{url}/v1/media/files")).json(&json!({"data": data_url(&p)?, "name": name})))?
                .json().map_err(|e| e.to_string())?;
            if o.json { println!("{j}"); } else { println!("{} ({}, {} KB)", j["id"].as_str().unwrap_or("?"), j["kind"].as_str().unwrap_or(""), j["bytes"].as_u64().unwrap_or(0) / 1024); }
        }
        "get" => {
            let id = first.ok_or("get needs an id")?;
            let out = require_out(&o.out)?;
            let bytes = send(client.get(format!("{url}/v1/media/files/{id}/content")))?.bytes().map_err(|e| e.to_string())?;
            write_file(&out, &bytes)?;
            println!("saved {} ({} KB)", out.display(), bytes.len() / 1024);
        }
        _ => {
            let id = first.ok_or("rm needs an id")?;
            send(client.delete(format!("{url}/v1/media/files/{id}")))?;
            println!("removed {id}");
        }
    }
    Ok(())
}

fn cmd_produce(client: &reqwest::blocking::Client, send: &dyn Fn(RB) -> Result<reqwest::blocking::Response, String>,
               first: &Option<String>, o: &MediaOpts, url: &str) -> Result<(), String> {
    let out = require_out(&o.out)?;
    let req = parse_request(first, "produce")?;
    let r = send(client.post(format!("{url}/v1/media/productions")).json(&req))?;
    let j: Value = r.json().map_err(|e| e.to_string())?;
    let id = j["id"].as_str().ok_or("no job id in the answer")?.to_string();
    println!("production queued: {id}");
    // follow it part by part, printing step N of M whenever it changes
    let mut last_part = String::new();
    loop {
        std::thread::sleep(Duration::from_secs(3));
        let s = send(client.get(format!("{url}/v1/media/jobs/{id}")))?;
        let j: Value = s.json().map_err(|e| e.to_string())?;
        match j["state"].as_str() {
            Some("done") => break,
            // a failed production has nothing to download: say why and exit non-zero
            Some("error") => return Err(format!("production {id} failed: {}", j["error"].as_str().unwrap_or("no reason given"))),
            _ => {}
        }
        if let Some(meta) = j["meta"].as_object() {
            if let (Some(p), Some(t)) = (meta.get("part").and_then(|x| x.as_u64()), meta.get("parts").and_then(|x| x.as_u64())) {
                let here = format!("{p} of {t}");
                if here != last_part { println!("step {here}"); last_part = here; }
            }
        }
    }
    // the finished job: download /content to --out
    let c = send(client.get(format!("{url}/v1/media/jobs/{id}/content")))?;
    let bytes = c.bytes().map_err(|e| e.to_string())?;
    write_file(&out, &bytes)?;
    println!("saved {} ({} KB)", out.display(), bytes.len() / 1024);
    Ok(())
}

fn require_out(out: &Option<String>) -> Result<PathBuf, String> {
    match out { Some(o) => Ok(expand_path(Path::new(o))), None => Err("this answer is a file: give --out <path>".into()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn later_images_go_in_the_same_folder_as_the_first() {
        assert_eq!(out_names("pics/cat.png", 3), vec!["pics/cat.png", "pics/cat-2.png", "pics/cat-3.png"]);
        assert_eq!(out_names("cat.png", 2), vec!["cat.png", "cat-2.png"]);
        assert_eq!(out_names("pics/cat", 2), vec!["pics/cat", "pics/cat-2"]);
    }

    #[test]
    fn image_request_only_sends_what_is_given() {
        assert_eq!(image_request("a cat", None, None, 1), json!({"prompt": "a cat"}));
        assert_eq!(image_request("a cat", Some("512x512"), Some("m:q8"), 3),
            json!({"prompt": "a cat", "size": "512x512", "model": "m:q8", "n": 3}));
    }

    #[test]
    fn video_request_only_sends_what_is_given() {
        assert_eq!(video_request("waves", Some(49), Some("1280x704"), None, None), json!({"prompt": "waves", "frames": 49, "size": "1280x704"}));
        assert_eq!(video_request("waves", None, None, Some("gif"), Some("wan")), json!({"prompt": "waves", "format": "gif", "model": "wan"}));
    }

    #[test]
    fn speech_and_song_send_only_the_fields() {
        assert_eq!(speak_request("hello world", None, None), json!({"input": "hello world"}));
        assert_eq!(speak_request("hi", Some("laura"), Some("m")), json!({"input": "hi", "voice": "laura", "model": "m"}));
        assert_eq!(song_request("a ballad", None, None), json!({"style": "a ballad"}));
        assert_eq!(song_request("sad", Some("[verse]\n…"), Some(90)),
            json!({"style": "sad", "lyrics": "[verse]\n…", "duration": 90}));
    }

    #[test]
    fn transcription_fields_carry_the_model_and_the_optional_ones() {
        assert_eq!(transcribe_fields(None, None, None), vec![("model", "whisper-1".to_string())]);
        assert_eq!(transcribe_fields(Some("whisper-base"), Some("text"), Some("de")),
            vec![("model", "whisper-base".into()), ("response_format", "text".into()), ("language", "de".into())]);
    }

    #[test]
    fn check_body_sends_the_file_as_a_data_url() {
        assert_eq!(check_body("data:audio/wav;base64,AAA", Some("hello"), None), json!({"audio": "data:audio/wav;base64,AAA", "expected_text": "hello"}));
        assert_eq!(check_body("d", None, Some("de")), json!({"audio": "d", "language": "de"}));
    }

    #[test]
    fn out_names_names_the_images_like_the_command_shows() {
        assert_eq!(out_names("file.png", 1), vec!["file.png".to_string()]);
        assert_eq!(out_names("file.png", 3), vec!["file.png".to_string(), "file-2.png".to_string(), "file-3.png".to_string()]);
        // an out without an extension keeps its name, no new suffix on the first
        assert_eq!(out_names("capture", 2), vec!["capture".to_string(), "capture-2".to_string()]);
    }
}
