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

pub async fn load_source(src: &str) -> Result<Vec<u8>> {
    if let Some(rest) = src.strip_prefix("data:") {
        let (_, b64) = rest.split_once(',').ok_or_else(|| anyhow!("the data: URL could not be read (no comma)"))?;
        return base64::engine::general_purpose::STANDARD
            .decode(b64.trim())
            .map_err(|e| anyhow!("the data: URL could not be read ({e})"));
    }
    if src.starts_with("http://") || src.starts_with("https://") {
        let resp = reqwest::Client::builder()
            .user_agent("strata")
            .timeout(std::time::Duration::from_secs(60))
            .build()?
            .get(src)
            .send()
            .await
            .map_err(|e| anyhow!("the URL could not be read: {e}"))?;
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
    if !path.is_empty() && Path::new(path).is_file() {
        return std::fs::read(path).with_context(|| format!("the file could not be read from {path}"));
    }
    bail!("an image must be a data: URL, an http(s) URL or a local file path")
}

pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok_and(|o| o.status.success())
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
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1"])
        .arg(&input)
        .output()?;
    let duration: f64 = String::from_utf8_lossy(&probe.stdout).trim().parse().unwrap_or(0.0);
    let fps = if duration > 0.0 { n as f64 / duration } else { 1.0 };
    let status = Command::new("ffmpeg")
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

fn tempdir() -> Result<PathBuf> {
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
        if !ffmpeg_available() {
            bail!("this server cannot decode video: install ffmpeg (sudo apt install ffmpeg), or send the video's \
                   frames as images in one message");
        }
        let mut out = Vec::with_capacity(parts.len() + n);
        for p in parts.drain(..) {
            match video_source(&p) {
                Some(src) => {
                    let data = load_source(&src).await?;
                    let fs = frames(&data, n).await?;
                    videos += 1;
                    out.push(json!({"type": "text", "text": format!("[A video, as {} frames in order:]\n", fs.len())}));
                    for f in fs {
                        let url = format!("data:image/jpeg;base64,{}", base64::engine::general_purpose::STANDARD.encode(f));
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

pub fn pdftools_available() -> bool {
    Command::new("pdftotext").arg("-v").output().is_ok() && Command::new("pdftoppm").arg("-v").output().is_ok()
}

/// A PDF -> (its text, page by page, and up to `max_images` page images as JPEG; none when `images` is false).
pub async fn pdf_pages(pdf: &[u8], images: bool, max_images: usize) -> Result<(Vec<String>, Vec<Vec<u8>>)> {
    if !pdftools_available() {
        bail!("this server cannot read PDF files: install poppler-utils (sudo apt install poppler-utils)");
    }
    if !pdf.starts_with(b"%PDF") {
        bail!("the document is not a PDF file");
    }
    let dir = tempdir()?;
    let input = dir.join("in.pdf");
    std::fs::write(&input, pdf)?;
    let out = Command::new("pdftotext").args(["-layout", "-enc", "UTF-8"]).arg(&input).arg("-").output()?;
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
        let status = Command::new("pdftoppm")
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
