//! Models from Hugging Face, as llama.cpp's `-hf` and PolyStrata's setup fetch them: a repository's GGUF files for one
//! quantization (all shards of a split file, and its mmproj for pictures), with or without an access token
//! (`--token`, `HF_TOKEN`, or the token file `huggingface-cli login` writes), resumed after an interruption and checked
//! against the SHA-256 the Hub publishes. `HF_ENDPOINT` points at a mirror.

pub mod config;
pub mod pull;

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Msg(String),
    #[error("{path}: {source}")]
    Io { path: String, source: std::io::Error },
    #[error("network: {0}")]
    Http(#[from] reqwest::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

fn msg(s: impl Into<String>) -> Error {
    Error::Msg(s.into())
}

/// One file of a repository.
#[derive(Debug, Clone)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
    /// the LFS SHA-256 (large files); None for small git files
    pub sha256: Option<String>,
}

pub fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").ok().filter(|e| !e.is_empty()).unwrap_or_else(|| "https://huggingface.co".into())
        .trim_end_matches('/').to_string()
}

/// The access token: the one given, else HF_TOKEN / HUGGING_FACE_HUB_TOKEN, else the token file of `hf auth login`
/// ($HF_HOME/token or ~/.cache/huggingface/token). None: public repositories only.
pub fn token(given: Option<&str>) -> Option<String> {
    if let Some(t) = given.filter(|t| !t.is_empty()) {
        return Some(t.to_string());
    }
    for v in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Some(t) = std::env::var(v).ok().filter(|t| !t.trim().is_empty()) {
            return Some(t.trim().to_string());
        }
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let hf_home = std::env::var("HF_HOME").unwrap_or_else(|_| format!("{home}/.cache/huggingface"));
    std::fs::read_to_string(format!("{hf_home}/token")).ok().map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
}

pub struct Hub {
    client: reqwest::Client,
    token: Option<String>,
    pub endpoint: String,
}

impl Hub {
    pub fn new(token: Option<String>) -> Result<Hub> {
        let client = reqwest::Client::builder().user_agent("am2strata/0.1").connect_timeout(Duration::from_secs(20)).build()?;
        Ok(Hub { client, token, endpoint: endpoint() })
    }

    fn get(&self, url: &str) -> reqwest::RequestBuilder {
        let r = self.client.get(url);
        match &self.token {
            Some(t) => r.bearer_auth(t),
            None => r,
        }
    }

    /// A refusal from the Hub, said so a person knows what to do.
    fn refusal(&self, repo: &str, status: reqwest::StatusCode) -> Error {
        let page = format!("{}/{repo}", self.endpoint);
        match status.as_u16() {
            401 if self.token.is_none() => msg(format!(
                "{repo} needs a Hugging Face token (a gated or private repository): accept its terms at {page}, then pass \
                 --token hf_... or set HF_TOKEN")),
            401 => msg(format!("the token was refused for {repo}: check it at {}/settings/tokens", self.endpoint)),
            403 => msg(format!("{repo} is gated: accept its terms at {page} with the account of the token, then try again")),
            404 => msg(format!("{repo} was not found on {} (check the name; a private repository also needs a token)", self.endpoint)),
            s => msg(format!("the Hub answered HTTP {s} for {repo}")),
        }
    }

    /// Every file of the repository at `revision` (the tree API, followed page by page).
    pub async fn files(&self, repo: &str, revision: &str) -> Result<Vec<RepoFile>> {
        let mut url = format!("{}/api/models/{repo}/tree/{revision}?recursive=true&expand=false", self.endpoint);
        let mut out = Vec::new();
        for _ in 0..200 {
            let r = self.get(&url).send().await?;
            if !r.status().is_success() {
                return Err(self.refusal(repo, r.status()));
            }
            let next = r.headers().get("link").and_then(|v| v.to_str().ok()).and_then(|l| {
                l.split(',').find(|p| p.contains("rel=\"next\"")).and_then(|p| p.split(['<', '>']).nth(1)).map(str::to_owned)
            });
            let items: Vec<Value> = r.json().await?;
            for it in items {
                if it["type"] != "file" {
                    continue;
                }
                let lfs = it.get("lfs").filter(|l| l.is_object());
                out.push(RepoFile {
                    path: it["path"].as_str().unwrap_or("").to_string(),
                    size: lfs.and_then(|l| l["size"].as_u64()).or_else(|| it["size"].as_u64()).unwrap_or(0),
                    sha256: lfs.and_then(|l| l["oid"].as_str()).map(str::to_owned),
                });
            }
            match next {
                Some(n) => url = n,
                None => return Ok(out),
            }
        }
        Ok(out)
    }

    /// Download one file to `dest`, resuming a `.part` left by an interrupted run and checking its SHA-256.
    /// `progress(done, total)` is called about once a second.
    pub async fn download(&self, repo: &str, revision: &str, f: &RepoFile, dest: &Path,
                          progress: &mut (dyn FnMut(u64, u64) + Send)) -> Result<()> {
        let io = |p: &Path, e| Error::Io { path: p.display().to_string(), source: e };
        if dest.exists() && std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0) == f.size {
            if verified(dest, f)? {
                return Ok(());
            }
            std::fs::remove_file(dest).map_err(|e| io(dest, e))?;
        }
        if let Some(d) = dest.parent() {
            std::fs::create_dir_all(d).map_err(|e| io(d, e))?;
        }
        let part = dest.with_extension(format!("{}.part", dest.extension().and_then(|e| e.to_str()).unwrap_or("")));
        let url = format!("{}/{repo}/resolve/{revision}/{}", self.endpoint, f.path);
        let mut tries = 0;
        loop {
            let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0).min(f.size);
            let mut req = self.get(&url);
            if have > 0 {
                req = req.header("Range", format!("bytes={have}-"));
            }
            let attempt: Result<()> = async {
                let mut r = req.send().await?;
                if r.status().as_u16() == 416 {
                    return Ok(()); // the part is whole already
                }
                if !r.status().is_success() {
                    return Err(self.refusal(repo, r.status()));
                }
                let append = have > 0 && r.status().as_u16() == 206;
                let mut out = std::fs::OpenOptions::new().create(true).write(true).append(append).truncate(!append)
                    .open(&part).map_err(|e| io(&part, e))?;
                let mut done = if append { have } else { 0 };
                let mut last = Instant::now();
                while let Some(chunk) = r.chunk().await? {
                    out.write_all(&chunk).map_err(|e| io(&part, e))?;
                    done += chunk.len() as u64;
                    if last.elapsed() >= Duration::from_secs(1) {
                        progress(done, f.size);
                        last = Instant::now();
                    }
                }
                out.flush().map_err(|e| io(&part, e))?;
                progress(done, f.size);
                Ok(())
            }.await;
            match attempt {
                Ok(()) => break,
                Err(Error::Msg(m)) => return Err(Error::Msg(m)), // a refusal: trying again will not help
                Err(e) if tries < 5 => {
                    tries += 1;
                    eprintln!("  ({e}; resuming, try {tries} of 5)");
                    tokio::time::sleep(Duration::from_secs(2 * tries)).await;
                }
                Err(e) => return Err(e),
            }
        }
        let got = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if got != f.size {
            return Err(msg(format!("{}: {got} bytes downloaded, the Hub says {}", f.path, f.size)));
        }
        if !verified(&part, f)? {
            let _ = std::fs::remove_file(&part);
            return Err(msg(format!("{}: the SHA-256 does not match the Hub's; the download was removed, run pull again", f.path)));
        }
        std::fs::rename(&part, dest).map_err(|e| io(dest, e))
    }
}

/// The file's SHA-256 is the Hub's (true when the Hub publishes none, a small git file).
fn verified(p: &Path, f: &RepoFile) -> Result<bool> {
    let Some(want) = &f.sha256 else { return Ok(true) };
    let mut h = Sha256::new();
    let mut file = std::fs::File::open(p).map_err(|e| Error::Io { path: p.display().to_string(), source: e })?;
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = file.read(&mut buf).map_err(|e| Error::Io { path: p.display().to_string(), source: e })?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    Ok(&got == want)
}

/// The quantizations a repository offers: its GGUF files grouped by quant tag (split shards counted once).
pub fn quants(files: &[RepoFile]) -> Vec<(String, u64, usize)> {
    let mut out: Vec<(String, u64, usize)> = Vec::new();
    for f in files.iter().filter(|f| is_model_gguf(&f.path)) {
        let q = quant_tag(&f.path).unwrap_or_else(|| file_name(&f.path).trim_end_matches(".gguf").to_string());
        match out.iter_mut().find(|(k, _, _)| *k == q) {
            Some(e) => {
                e.1 += f.size;
                e.2 += 1;
            }
            None => out.push((q, f.size, 1)),
        }
    }
    out.sort_by(|a, b| a.1.cmp(&b.1));
    out
}

fn file_name(p: &str) -> &str {
    p.rsplit('/').next().unwrap_or(p)
}

fn is_model_gguf(p: &str) -> bool {
    let n = file_name(p).to_lowercase();
    n.ends_with(".gguf") && !n.contains("mmproj") && !n.starts_with("imatrix")
}

const QUANTS: [&str; 40] = ["UD-IQ1_S", "UD-IQ1_M", "UD-IQ2_XXS", "UD-IQ2_M", "UD-IQ3_XXS", "UD-Q2_K_XL", "UD-Q3_K_XL",
    "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL", "Q2_K_L", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0",
    "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "BF16", "F16", "MXFP4"];

/// The quantization named in a file name (the longest known tag in it), e.g. "Q4_K_M" or "UD-Q6_K_XL".
pub fn quant_tag(path: &str) -> Option<String> {
    let up = file_name(path).to_uppercase();
    QUANTS.iter().filter(|q| up.contains(*q)).max_by_key(|q| q.len()).map(|q| q.to_string())
}

/// The files to fetch for `quant` (case-insensitive; matched against the quant tag, else a file-name substring):
/// the model GGUF(s) and, unless `mmproj` is false, the best mmproj (BF16, F16, F32, then any).
pub fn select(files: &[RepoFile], quant: &str, mmproj: bool) -> Result<Vec<RepoFile>> {
    let q = quant.to_uppercase();
    let by_tag: Vec<&RepoFile> = files.iter().filter(|f| is_model_gguf(&f.path) && quant_tag(&f.path).as_deref() == Some(q.as_str())).collect();
    let picked: Vec<&RepoFile> = if by_tag.is_empty() {
        files.iter().filter(|f| is_model_gguf(&f.path) && f.path.to_uppercase().contains(&q)).collect()
    } else {
        by_tag
    };
    if picked.is_empty() {
        let offered: Vec<String> = quants(files).into_iter().map(|(q, _, _)| q).collect();
        return Err(msg(format!("no GGUF for {quant:?}; this repository offers: {}", offered.join(", "))));
    }
    // several files: one split model (shards of one name) or several models matching - then refuse to guess
    let stems: std::collections::BTreeSet<String> = picked.iter().map(|f| split_base(&f.path)).collect();
    if stems.len() > 1 {
        return Err(msg(format!("{quant:?} matches several models: {}; name the file with --file",
                               stems.into_iter().collect::<Vec<_>>().join(", "))));
    }
    let mut out: Vec<RepoFile> = picked.into_iter().cloned().collect();
    out.sort_by(|a, b| a.path.cmp(&b.path));
    if mmproj {
        let projs: Vec<&RepoFile> = files.iter().filter(|f| file_name(&f.path).to_lowercase().contains("mmproj") && f.path.ends_with(".gguf")).collect();
        let rank = |f: &&RepoFile| {
            let n = f.path.to_uppercase();
            ["BF16", "F16", "F32"].iter().position(|t| n.contains(t)).unwrap_or(9)
        };
        if let Some(p) = projs.iter().min_by_key(|f| rank(*f)) {
            out.push((*p).clone());
        }
    }
    Ok(out)
}

/// A split file's name without its `-0000N-of-0000M` part.
fn split_base(p: &str) -> String {
    let n = file_name(p).trim_end_matches(".gguf");
    match n.rsplit_once("-of-") {
        Some((head, _)) => head.rsplit_once('-').map(|(b, _)| b.to_string()).unwrap_or_else(|| n.to_string()),
        None => n.to_string(),
    }
}

pub fn human(n: u64) -> String {
    let g = n as f64 / (1u64 << 30) as f64;
    if g >= 1.0 { format!("{g:.2} GB") } else { format!("{:.0} MB", n as f64 / (1u64 << 20) as f64) }
}

/// Free bytes on the filesystem of `dir` (df), None when it cannot be told.
pub fn free_space(dir: &Path) -> Option<u64> {
    let mut d = dir.to_path_buf();
    while !d.exists() {
        d = d.parent()?.to_path_buf();
    }
    let o = std::process::Command::new("df").args(["--output=avail", "-B1"]).arg(&d).output().ok()?;
    String::from_utf8_lossy(&o.stdout).lines().nth(1)?.trim().parse().ok()
}

pub fn local_path(dir: &Path, f: &RepoFile) -> PathBuf {
    dir.join(file_name(&f.path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(p: &str, s: u64) -> RepoFile {
        RepoFile { path: p.into(), size: s, sha256: None }
    }

    #[test]
    fn picks_a_quant_its_shards_and_the_best_mmproj() {
        let files = vec![f("M-Q4_K_M.gguf", 10), f("M-UD-Q4_K_XL-00001-of-00002.gguf", 5), f("M-UD-Q4_K_XL-00002-of-00002.gguf", 5),
                         f("mmproj-F32.gguf", 3), f("mmproj-BF16.gguf", 2), f("README.md", 1), f("M-Q8_0.gguf", 20)];
        let s = select(&files, "ud-q4_k_xl", true).unwrap();
        let names: Vec<&str> = s.iter().map(|x| x.path.as_str()).collect();
        assert_eq!(names, ["M-UD-Q4_K_XL-00001-of-00002.gguf", "M-UD-Q4_K_XL-00002-of-00002.gguf", "mmproj-BF16.gguf"]);
        assert_eq!(select(&files, "Q4_K_M", false).unwrap().len(), 1, "Q4_K_M is not UD-Q4_K_XL");
        let e = select(&files, "Q5_K_M", true).unwrap_err().to_string();
        assert!(e.contains("offers") && e.contains("Q8_0"), "{e}");
        let q = quants(&files);
        assert_eq!(q.iter().find(|x| x.0 == "UD-Q4_K_XL").unwrap().2, 2, "shards counted per quant");
    }

    #[test]
    fn quant_tags() {
        assert_eq!(quant_tag("Qwen3-30B-A3B-UD-Q6_K_XL.gguf").as_deref(), Some("UD-Q6_K_XL"));
        assert_eq!(quant_tag("x-iq3_xxs-00001-of-00002.gguf").as_deref(), Some("IQ3_XXS"));
        assert_eq!(split_base("dir/X-Q4-00002-of-00003.gguf"), "X-Q4");
    }
}
