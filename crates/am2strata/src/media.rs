//! Pictures and video for the image encoder.
//!
//! A picture's source is a data: URL, an http(s) URL or a local file path (never a network share), as PolyStrata
//! reads them. A video is turned into frames before the model sees it: MiniCPM-V and the other models read a video
//! as an ordered series of pictures, and neither llama.cpp's encoder nor the engines decode video themselves. Frames
//! are cut with ffmpeg when it is installed; a client can always send the frames as images instead.

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine as _;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

const URL_MAX: usize = 64 << 20;
pub const DEFAULT_FRAMES: usize = 16;
pub const MAX_FRAMES: usize = 64;

/// What a request's media sources may reach (main.rs sets it from the config at start). Local paths: only for a
/// server that listens on this PC alone, or with "allow_local_files": true. URLs: public addresses only, unless
/// "allow_private_urls": true (else an API client could read the LAN, the router or this PC's own services).
#[derive(Clone, Copy, Debug, Default)]
pub struct SourcePolicy {
    pub local_files: bool,
    pub private_urls: bool,
}

pub static POLICY: std::sync::OnceLock<SourcePolicy> = std::sync::OnceLock::new();

fn policy() -> SourcePolicy {
    POLICY.get().copied().unwrap_or_default()
}

/// Loopback, private, link-local, CGNAT, unique-local, multicast, unspecified or documentation addresses.
pub fn non_public(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(a) => {
            let o = a.octets();
            a.is_loopback() || a.is_private() || a.is_link_local() || a.is_unspecified() || a.is_broadcast() || a.is_multicast()
                || o[0] == 0 || (o[0] == 100 && (64..128).contains(&o[1])) || (o[0] == 192 && o[1] == 0 && o[2] == 0)
                || (o[0] == 198 && (18..20).contains(&o[1])) || o[0] >= 240
        }
        IpAddr::V6(a) => {
            if let Some(v4) = a.to_ipv4_mapped() {
                return non_public(IpAddr::V4(v4));
            }
            let s = a.segments();
            a.is_loopback() || a.is_unspecified() || a.is_multicast() || (s[0] & 0xfe00) == 0xfc00 || (s[0] & 0xffc0) == 0xfe80
        }
    }
}

/// GET of a public URL: each hop's address checked and pinned (no DNS rebinding between the check and the
/// connection), up to 5 redirects, each checked again.
pub async fn fetch_public(src: &str, allow_private: bool) -> Result<reqwest::Response> {
    let mut url = reqwest::Url::parse(src).map_err(|e| anyhow!("the URL could not be read: {e}"))?;
    for _ in 0..6 {
        let host = url.host_str().ok_or_else(|| anyhow!("the URL has no host"))?.to_string();
        let port = url.port_or_known_default().unwrap_or(443);
        let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host((host.trim_start_matches('[').trim_end_matches(']'), port)).await
            .map_err(|e| anyhow!("the URL's host could not be found: {e}"))?.collect();
        let Some(addr) = addrs.first().copied() else { bail!("the URL's host could not be found") };
        if !allow_private && addrs.iter().any(|a| non_public(a.ip())) {
            bail!("the URL points into a private network or this PC ({host}); such URLs are not fetched (\"allow_private_urls\" in the run config)");
        }
        let resp = reqwest::Client::builder()
            .user_agent("strata")
            .timeout(std::time::Duration::from_secs(60))
            .redirect(reqwest::redirect::Policy::none())
            .resolve(&host, addr)
            .build()?
            .get(url.clone())
            .send()
            .await
            .map_err(|e| anyhow!("the URL could not be read: {e}"))?;
        if resp.status().is_redirection() {
            let to = resp.headers().get(reqwest::header::LOCATION).and_then(|v| v.to_str().ok())
                .ok_or_else(|| anyhow!("the URL redirects nowhere"))?;
            url = url.join(to).map_err(|e| anyhow!("the URL redirects to a bad address: {e}"))?;
            if !matches!(url.scheme(), "http" | "https") {
                bail!("the URL redirects to a {} address", url.scheme());
            }
            continue;
        }
        return Ok(resp);
    }
    bail!("the URL redirects too many times")
}

pub async fn load_source(src: &str) -> Result<Vec<u8>> {
    // a media library id ("mf_..."): a file the server made or was sent before. The id is the file's content hash, so
    // holding it is holding the file; listing the library stays per key (media_files.rs)
    if let Some(f) = crate::media_files::resolve(src, crate::media_files::MAIN) {
        return f.map_err(|e| anyhow!(e));
    }
    if let Some(rest) = src.strip_prefix("data:") {
        let (_, b64) = rest.split_once(',').ok_or_else(|| anyhow!("the data: URL could not be read (no comma)"))?;
        return base64::engine::general_purpose::STANDARD
            .decode(b64.trim())
            .map_err(|e| anyhow!("the data: URL could not be read ({e})"));
    }
    if src.starts_with("http://") || src.starts_with("https://") {
        let resp = fetch_public(src, policy().private_urls).await?;
        if resp.content_length().is_some_and(|n| n as usize > URL_MAX) {
            bail!("the URL's file is over {} MiB", URL_MAX >> 20);
        }
        let bytes = resp.bytes().await.map_err(|e| anyhow!("the URL could not be read: {e}"))?;
        if bytes.len() > URL_MAX {
            bail!("the URL's file is over {} MiB", URL_MAX >> 20);
        }
        return Ok(bytes.to_vec());
    }
    let path = src.strip_prefix("file://").unwrap_or(src);
    if path.starts_with("//") || path.starts_with("\\\\") {
        bail!("a file must be on this computer: network paths are not read");
    }
    if !path.is_empty() && !policy().local_files {
        bail!("local file paths are not read by a server that listens beyond this PC: send the file as a data: URL \
               (or set \"allow_local_files\": true in the run config)");
    }
    if !path.is_empty() && Path::new(path).is_file() {
        return std::fs::read(path).with_context(|| format!("the file could not be read from {path}"));
    }
    bail!("an image must be a data: URL, an http(s) URL or a local file path")
}

/// The media engine, for reading MP4/H.264 frames in Rust when ffmpeg is not installed (main.rs sets it at start).
pub static ENGINE: std::sync::OnceLock<std::sync::Arc<crate::image_gen::Media>> = std::sync::OnceLock::new();

/// Frames from the media engine's own MP4/H.264 decoder (`video_frames`), no ffmpeg: `max` frames spread evenly
/// (every_s 0) or one every `every_s` seconds, each scaled to `width`. -> (PNG bytes per frame, fps, duration in s).
/// Other formats come back as the engine's error, which names them and says they need ffmpeg.
pub async fn rust_frames(video: &[u8], every_s: f64, max: usize, width: usize) -> Result<(Vec<Vec<u8>>, f64, f64, Option<String>)> {
    let Some(engine) = ENGINE.get() else { bail!("this server cannot decode video: no media engine and no ffmpeg") };
    let dir = tempdir()?;
    let input = dir.join("in.mp4");
    std::fs::write(&input, video)?;
    let r = engine.call(json!({"op": "video_frames", "file": input.to_string_lossy(), "every_s": every_s, "max": max, "width": width}),
                        |_| {}).await;
    let _ = std::fs::remove_dir_all(&dir);
    let v = r?;
    let mut out = Vec::new();
    for f in v["frames"].as_array().into_iter().flatten() {
        let png = base64::engine::general_purpose::STANDARD.decode(f["png"].as_str().unwrap_or("")).map_err(|e| anyhow::anyhow!("a frame: {e}"))?;
        out.push(png);
    }
    if out.is_empty() {
        bail!("the video gave no frames");
    }
    Ok((out, v["fps"].as_f64().unwrap_or(0.0), v["duration_s"].as_f64().unwrap_or(0.0), v["note"].as_str().map(str::to_owned)))
}

pub fn ffmpeg_available() -> bool {
    // tests on a machine that has ffmpeg check the path without it (set only through the test server's environment)
    if std::env::var_os("AM2_TEST_NO_FFMPEG").is_some() {
        return false;
    }
    Command::new(crate::server_tls::program("ffmpeg")).arg("-version").output().is_ok_and(|o| o.status.success())
}

/// `ffmpeg_available`, asked at most once a minute (a cluster master reads it with each worker's capabilities).
pub fn ffmpeg_cached() -> bool {
    static AT: std::sync::Mutex<Option<(std::time::Instant, bool)>> = std::sync::Mutex::new(None);
    if let Some((t, v)) = *AT.lock().unwrap() {
        if t.elapsed() < std::time::Duration::from_secs(60) {
            return v;
        }
    }
    let v = ffmpeg_available();
    *AT.lock().unwrap() = Some((std::time::Instant::now(), v));
    v
}

/// What compose needs of ffmpeg, missing from this one: the filters (xfade, zoompan, subtitles from libass, afade) and
/// the H.264 and AAC encoders. Empty when it can compose. Homebrew's ffmpeg 9, for one, is built without libass, so it
/// has no `subtitles` filter. Asked at most once a minute.
pub fn compose_missing() -> Vec<&'static str> {
    static AT: std::sync::Mutex<Option<(std::time::Instant, Vec<&'static str>)>> = std::sync::Mutex::new(None);
    if let Some((t, v)) = AT.lock().unwrap().clone() {
        if t.elapsed() < std::time::Duration::from_secs(60) {
            return v;
        }
    }
    let v = if !ffmpeg_available() { vec!["ffmpeg"] } else {
        let run = |a: &str| Command::new(crate::server_tls::program("ffmpeg")).args(["-hide_banner", a]).output()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
        missing_for_compose(&run("-filters"), &run("-encoders"))
    };
    *AT.lock().unwrap() = Some((std::time::Instant::now(), v.clone()));
    v
}

/// The compose needs not in ffmpeg's `-filters` and `-encoders` listings.
pub fn missing_for_compose(filters: &str, encoders: &str) -> Vec<&'static str> {
    let has = |list: &str, name: &str| list.lines().any(|l| l.split_whitespace().nth(1) == Some(name));
    let mut out: Vec<&'static str> = ["xfade", "zoompan", "subtitles", "afade", "tpad"].into_iter().filter(|f| !has(filters, f)).collect();
    out.extend(["libx264", "aac"].into_iter().filter(|e| !has(encoders, e)));
    out
}

fn video_source(part: &Value) -> Option<String> {
    let kind = part.get("type").and_then(Value::as_str)?;
    if !matches!(kind, "video_url" | "video" | "input_video") {
        return None;
    }
    let v = part.get("video_url").or_else(|| part.get("video")).or_else(|| part.get("url")).or_else(|| part.get("source"));
    match v {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(o)) => o.get("url").or_else(|| o.get("path")).and_then(Value::as_str).map(str::to_owned).or_else(|| {
            // Anthropic-style {"type": "base64", "media_type": ..., "data": ...}
            o.get("data").and_then(Value::as_str).map(|d| format!("data:video/mp4;base64,{d}"))
        }),
        _ => None,
    }
}

/// `n` frames evenly spread over the video, as JPEG bytes.
async fn frames(video: &[u8], n: usize) -> Result<Vec<Vec<u8>>> {
    let dir = tempdir()?;
    let input = dir.join("in.video");
    std::fs::write(&input, video)?;
    let probe = Command::new(crate::server_tls::program("ffprobe"))
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1"])
        .arg(&input)
        .output()?;
    let duration: f64 = String::from_utf8_lossy(&probe.stdout).trim().parse().unwrap_or(0.0);
    let fps = if duration > 0.0 { n as f64 / duration } else { 1.0 };
    let status = Command::new(crate::server_tls::program("ffmpeg"))
        .args(["-v", "error", "-i"])
        .arg(&input)
        .args(["-vf", &format!("fps={fps:.6},scale='min(1280,iw)':-2"), "-frames:v", &n.to_string(), "-q:v", "3"])
        .arg(dir.join("f%04d.jpg"))
        .status()?;
    if !status.success() {
        bail!("ffmpeg could not read the video");
    }
    let mut files: Vec<PathBuf> =
        std::fs::read_dir(&dir)?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "jpg")).collect();
    files.sort();
    let out = files.iter().map(std::fs::read).collect::<std::io::Result<Vec<_>>>()?;
    let _ = std::fs::remove_dir_all(&dir);
    if out.is_empty() {
        bail!("the video gave no frames");
    }
    Ok(out)
}

pub fn tempdir() -> Result<PathBuf> {
    let d = std::env::temp_dir().join(format!("strata-video-{}-{}", std::process::id(), uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

/// Replace every video part in an OpenAI request's messages with its frames as image parts, preceded by a short note
/// so the model reads them as one video in order. `n`: frames per video (the request's `video_frames`, default 16).
pub async fn expand_videos(req: &mut Value) -> Result<usize> {
    let n = req.get("video_frames").and_then(Value::as_u64).map(|n| n as usize).unwrap_or(DEFAULT_FRAMES).clamp(1, MAX_FRAMES);
    let Some(messages) = req.get_mut("messages").and_then(Value::as_array_mut) else { return Ok(0) };
    let mut videos = 0;
    for m in messages.iter_mut() {
        let Some(parts) = m.get_mut("content").and_then(Value::as_array_mut) else { continue };
        if !parts.iter().any(|p| video_source(p).is_some()) {
            continue;
        }
        let ffmpeg = ffmpeg_available();
        let mut out = Vec::with_capacity(parts.len() + n);
        for p in parts.drain(..) {
            match video_source(&p) {
                Some(src) => {
                    let data = load_source(&src).await?;
                    // ffmpeg when installed (any format); else the media engine's own MP4/H.264 decoder
                    let (fs, mime, note) = if ffmpeg {
                        (frames(&data, n).await?, "image/jpeg", None)
                    } else {
                        let r = rust_frames(&data, 0.0, n, 1280).await
                            .map_err(|e| anyhow::anyhow!("this server cannot decode this video without ffmpeg: {e:#}. Install ffmpeg ({}), \
                                                          or send the video's frames as images in one message",
                                crate::doctor::install_hint(std::env::consts::OS, "ffmpeg", "ffmpeg", "Gyan.FFmpeg")))?;
                        (r.0, "image/png", r.3)
                    };
                    videos += 1;
                    let how = note.map(|n| format!(" ({n})")).unwrap_or_default();
                    out.push(json!({"type": "text", "text": format!("[A video, as {} frames in order{how}:]\n", fs.len())}));
                    for f in fs {
                        let url = format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(f));
                        out.push(json!({"type": "image_url", "image_url": {"url": url}}));
                    }
                }
                None => out.push(p),
            }
        }
        *parts = out;
    }
    Ok(videos)
}

// ------------------------------------------------------------------------------------------------ documents (PDF)

/// Page images sent with a document's text (the rest of the pages go as text only).
pub const MAX_PAGE_IMAGES: usize = 20;

/// Poppler's `name`: the copy a release bundles (poppler/bin beside or above this binary's folder: the Windows
/// download carries one), else the one on PATH.
fn pdf_tool(name: &str) -> PathBuf {
    let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    std::env::current_exe().ok().and_then(|exe| {
        let dir = exe.parent()?.to_path_buf();
        [dir.join("poppler/bin"), dir.join("../poppler/bin")].into_iter().map(|d| d.join(&file)).find(|p| p.is_file())
    }).unwrap_or_else(|| PathBuf::from(name))
}

pub fn pdftools_available() -> bool {
    Command::new(pdf_tool("pdftotext")).arg("-v").output().is_ok() && Command::new(pdf_tool("pdftoppm")).arg("-v").output().is_ok()
}

/// A PDF -> (its text, page by page, and up to `max_images` page images as JPEG; none when `images` is false).
pub async fn pdf_pages(pdf: &[u8], images: bool, max_images: usize) -> Result<(Vec<String>, Vec<Vec<u8>>)> {
    if !pdftools_available() {
        bail!("this server cannot read PDF files: {}", crate::doctor::install_hint(std::env::consts::OS, "poppler-utils", "poppler", "oschwartz10612.Poppler"));
    }
    if !pdf.starts_with(b"%PDF") {
        bail!("the document is not a PDF file");
    }
    let dir = tempdir()?;
    let input = dir.join("in.pdf");
    std::fs::write(&input, pdf)?;
    let out = Command::new(pdf_tool("pdftotext")).args(["-layout", "-enc", "UTF-8"]).arg(&input).arg("-").output()?;
    if !out.status.success() {
        let _ = std::fs::remove_dir_all(&dir);
        bail!("the PDF could not be read ({})", String::from_utf8_lossy(&out.stderr).trim());
    }
    // pdftotext ends each page with a form feed
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let mut pages: Vec<String> = text.split('\u{c}').map(|p| p.trim_end().to_string()).collect();
    while pages.last().is_some_and(|p| p.is_empty()) {
        pages.pop();
    }
    let mut imgs = Vec::new();
    if images && max_images > 0 {
        let status = Command::new(pdf_tool("pdftoppm"))
            .args(["-jpeg", "-jpegopt", "quality=80", "-r", "110", "-l", &max_images.to_string()])
            .arg(&input)
            .arg(dir.join("p"))
            .status()?;
        if status.success() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)?.flatten().map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "jpg")).collect();
            files.sort();
            imgs = files.iter().take(max_images).map(std::fs::read).collect::<std::io::Result<Vec<_>>>()?;
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok((pages, imgs))
}

/// The text the model reads for a document: its name, then each page's text under a page heading.
fn document_text(name: &str, pages: &[String], images: usize) -> String {
    let mut s = format!("[Document: {name}, {} page{}{}]\n", pages.len(), if pages.len() == 1 { "" } else { "s" },
                        if images > 0 { format!("; the first {images} page{} also as pictures after the text", if images == 1 { "" } else { "s" }) } else { String::new() });
    for (i, p) in pages.iter().enumerate() {
        s.push_str(&format!("\n--- page {} ---\n{}\n", i + 1, if p.trim().is_empty() { "(no text on this page: see its picture)" } else { p }));
    }
    s
}

/// (data, file name) of a document part in any of the three APIs' shapes, or None when the part is not one.
fn document_source(part: &Value) -> Option<(String, String)> {
    let kind = part.get("type").and_then(Value::as_str)?;
    match kind {
        // OpenAI Chat: {"type": "file", "file": {"file_data": "data:application/pdf;base64,...", "filename": ...}}
        "file" => {
            let f = part.get("file")?;
            Some((f.get("file_data")?.as_str()?.to_string(), f.get("filename").and_then(Value::as_str).unwrap_or("document.pdf").to_string()))
        }
        // OpenAI Responses: {"type": "input_file", "file_data": ..., "filename": ...}
        "input_file" => Some((part.get("file_data")?.as_str()?.to_string(),
                              part.get("filename").and_then(Value::as_str).unwrap_or("document.pdf").to_string())),
        // Anthropic: {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": ...}}
        "document" => {
            let s = part.get("source")?;
            let name = part.get("title").and_then(Value::as_str).unwrap_or("document.pdf").to_string();
            match s.get("type").and_then(Value::as_str) {
                Some("base64") => Some((format!("data:application/pdf;base64,{}", s.get("data")?.as_str()?), name)),
                Some("url") => Some((s.get("url")?.as_str()?.to_string(), name)),
                Some("text") => Some((format!("text:{}", s.get("data")?.as_str()?), name)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Replace every document part (PDF) in a request's messages with its text and, for a model that reads pictures,
/// its first pages as images, in the request's own API shape (`anthropic`: image blocks, else image_url parts).
pub async fn expand_documents(req: &mut Value, images: bool, anthropic: bool) -> Result<usize> {
    let key = if req.get("input").is_some_and(Value::is_array) { "input" } else { "messages" };
    let Some(messages) = req.get_mut(key).and_then(Value::as_array_mut) else { return Ok(0) };
    let mut docs = 0;
    let mut budget = MAX_PAGE_IMAGES;
    for m in messages.iter_mut() {
        let Some(parts) = m.get_mut("content").and_then(Value::as_array_mut) else { continue };
        if !parts.iter().any(|p| document_source(p).is_some()) {
            continue;
        }
        let mut out = Vec::with_capacity(parts.len() + 4);
        for p in parts.drain(..) {
            let Some((src, name)) = document_source(&p) else {
                out.push(p);
                continue;
            };
            docs += 1;
            let text_kind = if key == "input" { "input_text" } else { "text" };
            if let Some(plain) = src.strip_prefix("text:") {
                out.push(json!({"type": text_kind, "text": format!("[Document: {name}]\n{plain}")}));
                continue;
            }
            let data = load_source(&src).await?;
            let (pages, imgs) = pdf_pages(&data, images, budget).await?;
            budget -= imgs.len();
            out.push(json!({"type": text_kind, "text": document_text(&name, &pages, imgs.len())}));
            for img in imgs {
                let b64 = base64::engine::general_purpose::STANDARD.encode(img);
                out.push(if anthropic {
                    json!({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
                } else if key == "input" {
                    json!({"type": "input_image", "image_url": format!("data:image/jpeg;base64,{b64}")})
                } else {
                    json!({"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{b64}")}})
                });
            }
        }
        *parts = out;
    }
    Ok(docs)
}
