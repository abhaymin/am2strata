//! The image encoder: `strata-vision` (llama.cpp mtmd + the model's mmproj) reads `ENC <image> <out>` lines and
//! answers `OK <tokens>` after writing the image's embeddings. Results are cached by the image's hash, so a picture a
//! conversation sends every turn is encoded once. Port of serve/server.py `Vision`.

use crate::{Error, Result};
use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, Mutex};

const READY_TIMEOUT: Duration = Duration::from_secs(300);
const ENCODE_TIMEOUT: Duration = Duration::from_secs(600);
const CACHE_MAX: usize = 64;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    pub exe: PathBuf,
    pub mmproj: PathBuf,
    pub model: PathBuf,
    #[serde(default)]
    pub gpu: bool,
    pub threads: Option<u32>,
    pub max_tokens: Option<u32>,
    pub min_tokens: Option<u32>,
}

struct Proc {
    child: Child,
    stdin: ChildStdin,
    lines: mpsc::UnboundedReceiver<Option<String>>,
}

pub struct Vision {
    proc: Mutex<Option<Proc>>,
    dir: PathBuf,
    cache: Mutex<VecDeque<(String, PathBuf, usize)>>,
    pub on_gpu: bool,
}

/// The encoder's work directory: no space in its path (the ENC line is split at spaces, PolyStrata #480).
fn work_dir() -> std::io::Result<PathBuf> {
    let base = std::env::temp_dir();
    for i in 0..1000u32 {
        let d = base.join(format!("strata-vision-rs-{}-{i}", std::process::id()));
        if std::fs::create_dir(&d).is_ok() && !d.display().to_string().contains(' ') {
            return Ok(d);
        }
    }
    Err(std::io::Error::other("no work directory for the image encoder"))
}

impl Vision {
    /// Start the encoder. `env`: the engine's environment (GPU selection, library paths).
    pub async fn start(cfg: &VisionConfig, gpu: bool, env: &[(String, String)], log: Option<&Path>) -> Result<Self> {
        let dir = work_dir()?;
        let mut cmd = Command::new(std::path::absolute(&cfg.exe)?);
        cmd.arg("--mmproj").arg(std::path::absolute(&cfg.mmproj)?).arg("--model").arg(std::path::absolute(&cfg.model)?);
        if gpu {
            cmd.arg("--gpu");
        }
        let threads = cfg.threads.or((!gpu).then(|| (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8) / 2).max(1) as u32));
        if let Some(t) = threads {
            cmd.arg("--threads").arg(t.to_string());
        }
        if let Some(t) = cfg.max_tokens {
            cmd.arg("--max-tokens").arg(t.to_string());
        }
        if let Some(t) = cfg.min_tokens {
            cmd.arg("--min-tokens").arg(t.to_string());
        }
        cmd.current_dir(&dir).stdin(Stdio::piped()).stdout(Stdio::piped()).kill_on_drop(true);
        for (k, v) in env {
            cmd.env(k, v);
        }
        match log {
            Some(p) => {
                cmd.stderr(Stdio::from(std::fs::OpenOptions::new().create(true).append(true).open(p)?));
            }
            None => {
                cmd.stderr(Stdio::null());
            }
        }
        let mut child = cmd.spawn().map_err(|e| Error::Spawn { exe: cfg.exe.display().to_string(), source: e })?;
        let stdin = child.stdin.take().expect("piped");
        let stdout = child.stdout.take().expect("piped");
        let (tx, mut lines) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let mut r = BufReader::new(stdout).lines();
            while let Ok(Some(l)) = r.next_line().await {
                if tx.send(Some(l)).is_err() {
                    return;
                }
            }
            let _ = tx.send(None);
        });
        let first = tokio::time::timeout(READY_TIMEOUT, lines.recv())
            .await
            .map_err(|_| Error::StartTimeout(READY_TIMEOUT))?
            .flatten()
            .ok_or(Error::Died)?;
        if !first.starts_with("READY") {
            return Err(Error::Refused(format!("the image encoder did not start: {first}")));
        }
        Ok(Self { proc: Mutex::new(Some(Proc { child, stdin, lines })), dir, cache: Mutex::new(VecDeque::new()), on_gpu: gpu })
    }

    /// Encode one image (already in a format the encoder reads) -> (embeddings file, image tokens). `keep`: files
    /// the request still needs; they are never evicted by this call.
    pub async fn encode(&self, data: &[u8], keep: &[PathBuf]) -> Result<(PathBuf, usize)> {
        let key = sha256_hex(data)[..32].to_string();
        {
            let mut cache = self.cache.lock().await;
            if let Some(pos) = cache.iter().position(|(k, _, _)| *k == key) {
                let hit = cache.remove(pos).expect("present");
                let out = (hit.1.clone(), hit.2);
                cache.push_back(hit);
                return Ok(out);
            }
        }
        let img = self.dir.join(format!("{key}.img"));
        let out = self.dir.join(format!("{key}.sve"));
        std::fs::File::create(&img)?.write_all(data)?;
        let reply = {
            let mut guard = self.proc.lock().await;
            let p = guard.as_mut().ok_or(Error::Died)?;
            let line = format!("ENC {key}.img {key}.sve\n");
            let sent = p.stdin.write_all(line.as_bytes()).await.and(p.stdin.flush().await);
            let reply = match sent {
                Ok(()) => tokio::time::timeout(ENCODE_TIMEOUT, p.lines.recv()).await,
                Err(_) => Ok(None),
            };
            let _ = std::fs::remove_file(&img);
            match reply {
                Ok(Some(Some(l))) => l,
                Ok(_) => return Err(Error::Died),
                Err(_) => {
                    let _ = p.child.start_kill();
                    *guard = None;
                    return Err(Error::Refused("the image encoder said nothing for 600 s".into()));
                }
            }
        };
        let reply = reply.trim();
        let Some(n) = reply.strip_prefix("OK").and_then(|r| r.split_whitespace().next()).and_then(|n| n.parse().ok()) else {
            let why = reply.strip_prefix("ERR").map(str::trim).unwrap_or("the image encoder stopped");
            return Err(Error::Refused(format!("the image could not be read: {why}")));
        };
        let mut cache = self.cache.lock().await;
        cache.push_back((key, out.clone(), n));
        let mut i = 0;
        while cache.len() > CACHE_MAX && i < cache.len() {
            let held = keep.contains(&cache[i].1) || cache[i].1 == out;
            if held {
                i += 1;
            } else if let Some(old) = cache.remove(i) {
                let _ = std::fs::remove_file(old.1);
            }
        }
        Ok((out, n))
    }

    /// One file for GENI: the request's images' embeddings in prompt order. /dev/shm when it has room (PolyStrata
    /// #874), else the encoder's directory. The caller deletes it after the request.
    pub fn combine(&self, parts: &[PathBuf]) -> Result<PathBuf> {
        let total: u64 = parts.iter().filter_map(|p| std::fs::metadata(p).ok()).map(|m| m.len()).sum();
        let name = format!("req-{}-{}.sve", std::process::id(), sha256_hex(format!("{parts:?}{:?}", std::time::SystemTime::now()).as_bytes()).get(..12).unwrap_or("x"));
        let shm = Path::new("/dev/shm");
        let path = if shm.is_dir() && free_bytes(shm).is_some_and(|f| f > 2 * total + (16 << 20)) {
            shm.join(format!("strata-vision-rs-{name}"))
        } else {
            self.dir.join(name)
        };
        let mut f = std::fs::File::create(&path)?;
        for p in parts {
            std::io::copy(&mut std::fs::File::open(p)?, &mut f)?;
        }
        Ok(path)
    }

    pub async fn shutdown(&self) {
        if let Some(mut p) = self.proc.lock().await.take() {
            let _ = p.child.start_kill();
            let _ = p.child.wait().await;
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn free_bytes(p: &Path) -> Option<u64> {
    let out = std::process::Command::new("df").args(["-B1", "--output=avail"]).arg(p).output().ok()?;
    String::from_utf8_lossy(&out.stdout).lines().nth(1)?.trim().parse().ok()
}

pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

/// The formats the encoder's decoder (stb_image) reads pass through; anything else (WebP, TIFF, ...) is converted
/// to PNG, transparent areas on white.
pub fn normalize_image(data: &[u8]) -> Result<Vec<u8>> {
    let passthrough = data.starts_with(b"\xff\xd8\xff")
        || data.starts_with(b"\x89PNG\r\n\x1a\n")
        || data.starts_with(b"BM")
        || data.starts_with(b"GIF87a")
        || data.starts_with(b"GIF89a");
    if passthrough {
        return Ok(data.to_vec());
    }
    let img = image::load_from_memory(data).map_err(|e| Error::Refused(format!("the image could not be read ({e})")))?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let mut rgb = image::RgbImage::new(w, h);
    for (x, y, p) in rgba.enumerate_pixels() {
        let a = p[3] as u32;
        let mix = |c: u8| ((c as u32 * a + 255 * (255 - a)) / 255) as u8;
        rgb.put_pixel(x, y, image::Rgb([mix(p[0]), mix(p[1]), mix(p[2])]));
    }
    let mut out = std::io::Cursor::new(Vec::new());
    rgb.write_to(&mut out, image::ImageFormat::Png).map_err(|e| Error::Refused(format!("PNG: {e}")))?;
    Ok(out.into_inner())
}
