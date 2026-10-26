//! The MTP block from the BF16 checkpoint (Qwen/Qwen3.8-Flash-Next, 131 safetensors shards), without the checkpoint:
//! the port of tools/mtp_fetch.py. Safetensors puts a JSON header at the start of each shard, so HTTP range requests
//! read the headers and then only the 31 `mtp.*` tensors.
//!
//! - `inventory`: the headers only -> mtp-inventory.json / .md
//! - `fetch`: the tensors (resumable) -> tensors/<name>.bin and mtp-manifest.json
//! - `verify`: the fetched tensors against the pinned revision's SHA-256 (exit code 3 when one is missing or wrong)
//!
//! A range read must come back as 206 with the Content-Range asked for (a mirror that ignores Range sends the whole
//! shard), and every tensor of the pinned revision is checked against its SHA-256.
//!
//! As Strata 0.2's mtp_fetch.py: a pinned revision gone from the Hub stops the fetch (STRATA_ALLOW_UNPINNED=1 reads
//! the current files, still checked against the pinned SHA-256); ranges are read STRATA_DL_CONNECTIONS (16) at a time,
//! in order, through the CDN address of the first redirect; a dropped connection ends only that try (fewer
//! connections after it); one fetch writes a folder at a time (a lock file); STRATA_EXIT_WITH_PARENT=1 stops it with
//! the process that started it; progress is one line every two seconds, in setup's form.

use crate::num::py_json;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

pub const PINNED_REVISION: &str = "de4b8e4d43b917e7706784d8bb445c9af86a3540";
pub const BAD: i32 = 3;

/// sha256 of every MTP tensor at PINNED_REVISION (mtp_fetch.py SHA256)
pub const SHA256: [(&str, &str); 31] = [
    ("mtp.fc_embedding.weight", "241d6e87ded317a61b1aeb3ca4e28c53f1caba09ad5be66092e91fe5e3e14751"),
    ("mtp.fc_hidden.weight", "97c8e427ad0177e2c713ed24a397d815b523eda15ef453f1ae6463f6c70a8f96"),
    ("mtp.hyper_connection_mixer.hc_norm.weight", "ce948a398e15aeaa403c424cda34e6eed296ab2570d591bb5ff3da642b255d64"),
    ("mtp.hyper_connection_mixer.input_mix_weight_down.weight", "c633d2841fda463e82045e0b93fb8cb8d511ec0c9eda616020f01aca18d71947"),
    ("mtp.hyper_connection_mixer.input_mix_weight_up.weight", "6565285f0ba2923703252763cfc11d8a9bda40ab42e7548e12464b04dbdbec13"),
    ("mtp.layers.0.attn_hyper_connection.block_inject_weight.weight", "c8afff64e7f54412d02120097c4255033c55245e1c1eed84cc630d428ccb864c"),
    ("mtp.layers.0.attn_hyper_connection.hc_norm.weight", "1fb619aeaaf637ccd70f46f1f8f8116e6a46448cb43026b246de8ee8286ed809"),
    ("mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight", "e80ac8bf3e270e9d5a949ec7c913329dc8de017d69851334820fae7b8c86a7f5"),
    ("mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight", "f80f1fd44e1183e7764ea496b11ff89ead9dd758f861d4e79a770bf54956b1a5"),
    ("mtp.layers.0.mlp.experts.down_proj", "539c46234f73bac70450f731f40e715f77ac32faf77def0f3f800e58aaabbbc0"),
    ("mtp.layers.0.mlp.experts.gate_up_proj", "839e98f9642ebc884b178abd2090b6dc8b0bcd81ed92d7f7a08314c6a68f96d5"),
    ("mtp.layers.0.mlp.gate.weight", "f0d35fa41b360a08bc0c5b1e5766aa42df960291716ee189208b5d2599462c20"),
    ("mtp.layers.0.mlp.shared_expert.down_proj.weight", "d2ced8c81367cc494599d1a8cdc279c3dd6579b628ded05acd7495e99cee01d8"),
    ("mtp.layers.0.mlp.shared_expert.gate_proj.weight", "f4a81109165853b3b64f36d809f98094b599763364070f5fa079200e1c386d4c"),
    ("mtp.layers.0.mlp.shared_expert.up_proj.weight", "172586861eea53efbff10afc43ded5b6e49c83879e9efc01e089c652bbb0a156"),
    ("mtp.layers.0.mlp.shared_expert_gate.weight", "acdd306d9996d356e4667a16ff63af6fef086fc7f570f1bc00bbe882a2a93f6e"),
    ("mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight", "c5782dd92b479a1f8ef43d683fe9c917fd9e8fba81f23c1aae8817bfe4b476a6"),
    ("mtp.layers.0.mlp_hyper_connection.hc_norm.weight", "21a504876b5e7b788301557c68dfb757999430eba36c66ec8ab96a63604dbfd1"),
    ("mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight", "b71240036dea954e034d5ae04e097bcd7ac51cb28a44e4c93d198af3447278a1"),
    ("mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight", "40756866bf2a5e82c28e407f717b6fa07614d593f78e79c7ceb2cd4ffe6dda94"),
    ("mtp.layers.0.self_attn.indexer.index_qk_proj.weight", "2befafae6e944c1fc651fd73d4d527eb6f138a6c1679065b003128ee3150e859"),
    ("mtp.layers.0.self_attn.indexer.k_layernorm.weight", "2db621f6a08b4085756fc69d60a4a1bd1d7fbf5d406456d2477075842efaae54"),
    ("mtp.layers.0.self_attn.indexer.q_layernorm.weight", "afaf59f68e251939970ffdeacad0667dffc657a522bea62cdc44a52816fc49d6"),
    ("mtp.layers.0.self_attn.k_norm.weight", "8f924b0357cf936e14fc34518205d000e1423aa60626827df81b0804fae8cdfd"),
    ("mtp.layers.0.self_attn.k_proj.weight", "1a295dc2d6e633e808aeb19e00c97856a1817bbfcc83f5b610e2bfcacfcb8c38"),
    ("mtp.layers.0.self_attn.o_proj.weight", "9d2fcf0ac6ac19de87f638f710cfd1af6ad14141e75b7d3d5d29fbd06f374219"),
    ("mtp.layers.0.self_attn.q_norm.weight", "57957aa1e7c5a3ac628bff0ec5522ad55a39b8c0b8bd2d324ae640e3771617ef"),
    ("mtp.layers.0.self_attn.q_proj.weight", "5220aef3d288f40759f7356af9ef453e5ac0256a38749a9bb803a4e9db58150d"),
    ("mtp.layers.0.self_attn.v_proj.weight", "39cb12f5424e965fa71fb81db0ee230c701b7b1e0b72c36641fde7e0e41bff41"),
    ("mtp.pre_fc_norm_embedding.weight", "04c4a570850e06f2d8913da8220d54d4c7f87db6eb6d45480b938e8ba41d6a86"),
    ("mtp.pre_fc_norm_hidden.weight", "b933b44559879449b1baf2af8385c5c237347307e3fd452394870cb53e6eab6d"),
];

fn want_sha(name: &str) -> Option<&'static str> {
    SHA256.iter().find(|(n, _)| *n == name).map(|(_, s)| *s)
}

/// Where the checkpoint is read from: the pinned revision on Hugging Face (or HF_ENDPOINT, or STRATA_MTP_REVISION),
/// or ModelScope's current files with STRATA_SOURCE=modelscope; `pinned` is the URL the hashes belong to.
pub struct Source {
    pub repo: String,
    pub pinned: String,
    revision: String,
    client: reqwest::blocking::Client,
    retries: u32,
    /// a /resolve/ URL -> the signed CDN address its redirect led to, reused for CDN_TTL
    cdn: Mutex<HashMap<String, (String, Instant)>>,
}

const CDN_TTL: Duration = Duration::from_secs(600);

impl Source {
    pub fn from_env() -> Source {
        let env = |k: &str| std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let revision = env("STRATA_MTP_REVISION").unwrap_or_else(|| PINNED_REVISION.into());
        let hf = env("HF_ENDPOINT").map(|v| v.trim_end_matches('/').to_string()).unwrap_or_else(|| "https://huggingface.co".into());
        let mut repo = format!("{hf}/Qwen/Qwen3.8-Flash-Next/resolve/{revision}/");
        let mut pinned = format!("{hf}/Qwen/Qwen3.8-Flash-Next/resolve/{PINNED_REVISION}/");
        if matches!(env("STRATA_SOURCE").map(|s| s.to_lowercase()).as_deref(), Some("ms") | Some("modelscope")) {
            let ms = env("MODELSCOPE_ENDPOINT").map(|v| v.trim_end_matches('/').to_string()).unwrap_or_else(|| "https://www.modelscope.cn".into());
            repo = format!("{ms}/models/Qwen/Qwen3.8-Flash-Next/resolve/master/");
            pinned = repo.clone();
        }
        let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(120)).user_agent("strata-mtp-fetch")
            .build().expect("http client");
        Source { repo, pinned, revision, client, retries: 6, cdn: Mutex::new(HashMap::new()) }
    }

    fn is_pinned(&self) -> bool {
        self.repo == self.pinned
    }

    /// GET a URL (a byte range when `range` is given), retried six times with backoff (at most 16 s), through the CDN
    /// address an earlier redirect gave (forgotten after any error, so the next try asks the Hub again).
    fn get(&self, url: &str, range: Option<(u64, u64)>) -> Result<Vec<u8>, String> {
        let mut last = String::new();
        for attempt in 0..self.retries {
            let used = self.cdn.lock().unwrap().get(url).filter(|(_, until)| *until > Instant::now()).map(|(u, _)| u.clone())
                .unwrap_or_else(|| url.to_string());
            let r = (|| -> Result<Vec<u8>, String> {
                let mut req = self.client.get(&used);
                if let Some((a, b)) = range {
                    req = req.header("Range", format!("bytes={a}-{b}"));
                }
                let resp = req.send().map_err(|e| e.to_string())?;
                if let Some((a, b)) = range {
                    if resp.status().as_u16() != 206 {
                        return Err(format!("range request not honoured: HTTP {} instead of 206 (a mirror or proxy that ignores \
                                            the Range header sends the whole file)", resp.status().as_u16()));
                    }
                    let cr = resp.headers().get("content-range").and_then(|v| v.to_str().ok()).unwrap_or("").trim().to_string();
                    if !cr.starts_with(&format!("bytes {a}-{b}/")) {
                        return Err(format!("range request answered with Content-Range {cr:?}, not bytes {a}-{b}/..."));
                    }
                } else if !resp.status().is_success() {
                    return Err(format!("HTTP {} for {url}", resp.status().as_u16()));
                }
                let landed = resp.url().to_string();
                let data = resp.bytes().map_err(|e| e.to_string())?.to_vec();
                if let Some((a, b)) = range {
                    if data.len() as u64 != b - a + 1 {
                        return Err(format!("short range read: {} of {}", data.len(), b - a + 1));
                    }
                }
                if used == url && landed != url {
                    self.cdn.lock().unwrap().insert(url.to_string(), (landed, Instant::now() + CDN_TTL));
                }
                Ok(data)
            })();
            match r {
                Ok(d) => return Ok(d),
                Err(e) => {
                    self.cdn.lock().unwrap().remove(url);
                    last = e;
                    if attempt + 1 < self.retries {
                        std::thread::sleep(Duration::from_secs((1u64 << attempt).min(16)));
                        eprintln!("retry {url}: {last}");
                    }
                }
            }
        }
        Err(last)
    }

    /// The pinned revision gone from the repository (a 404 on its index) stops the fetch, unless
    /// STRATA_ALLOW_UNPINNED=1: then the current files (main) are read, their tensors still checked against the pinned
    /// SHA-256 (Strata 0.2).
    pub fn resolve(&mut self) -> Result<(), String> {
        let url = format!("{}model.safetensors.index.json", self.repo);
        if let Ok(r) = self.client.head(&url).send() {
            if r.status().as_u16() == 404 && self.repo.contains(&format!("/resolve/{}/", self.revision)) {
                if std::env::var("STRATA_ALLOW_UNPINNED").map(|v| v.trim() == "1").unwrap_or(false) {
                    eprintln!("STRATA_ALLOW_UNPINNED=1: the checkpoint's pinned revision {} is gone: reading its current files (main); \
                               every tensor is still checked against the pinned SHA256", self.revision);
                    self.repo = self.repo.replace(&format!("/resolve/{}/", self.revision), "/resolve/main/");
                } else {
                    return Err(format!("the pinned version of the checkpoint ({}) is no longer on Hugging Face: update am2Strata \
                                        (or set STRATA_ALLOW_UNPINNED=1 to read its current files)", self.revision));
                }
            }
        }
        Ok(())
    }

    /// The bytes start..=last of `url`, in `chunk`-sized ranges, `workers` of them in flight at once, handed to
    /// `sink` in order (a slow range holds only its own slot; the others read ahead, at most ~512 MiB).
    fn ranged(&self, url: &str, start: u64, last: u64, chunk: u64, workers: usize, sink: &mut dyn FnMut(&[u8]) -> Result<(), String>)
              -> Result<(), String> {
        let mut spans = Vec::new();
        let mut pos = start;
        while pos <= last {
            let e = (pos + chunk - 1).min(last);
            spans.push((pos, e));
            pos = e + 1;
        }
        if workers <= 1 || spans.len() <= 1 {
            for (a, b) in spans {
                sink(&self.get(url, Some((a, b)))?)?;
            }
            return Ok(());
        }
        let depth = workers.max((3 * workers).min(((512u64 << 20) / chunk.max(1)) as usize));
        // results by span index; `next` is the span the sink takes next, a worker waits while it is `depth` ahead
        let state = Mutex::new((HashMap::<usize, Result<Vec<u8>, String>>::new(), 0usize, 0usize, false)); // (done, next, handed, stop)
        let cv = Condvar::new();
        let n = spans.len();
        std::thread::scope(|sc| -> Result<(), String> {
            for _ in 0..workers.min(n) {
                sc.spawn(|| loop {
                    let i = {
                        let mut g = state.lock().unwrap();
                        loop {
                            if g.3 || g.2 >= n {
                                return;
                            }
                            if g.2 < g.1 + depth {
                                break;
                            }
                            g = cv.wait(g).unwrap();
                        }
                        g.2 += 1;
                        g.2 - 1
                    };
                    let r = self.get(url, Some(spans[i]));
                    let mut g = state.lock().unwrap();
                    g.0.insert(i, r);
                    cv.notify_all();
                });
            }
            let mut out = Ok(());
            for i in 0..n {
                let r = {
                    let mut g = state.lock().unwrap();
                    loop {
                        if let Some(r) = g.0.remove(&i) {
                            g.1 = i + 1;
                            cv.notify_all();
                            break r;
                        }
                        g = cv.wait(g).unwrap();
                    }
                };
                if let Err(e) = r.and_then(|d| sink(&d)) {
                    out = Err(e);
                    break;
                }
            }
            let mut g = state.lock().unwrap();
            g.3 = true;
            cv.notify_all();
            drop(g);
            out
        })
    }

    fn shard_header(&self, shard: &str) -> Result<(u64, Map<String, Value>), String> {
        let url = format!("{}{shard}", self.repo);
        let n = u64::from_le_bytes(self.get(&url, Some((0, 7)))?.try_into().map_err(|_| "short header length")?);
        let header: Map<String, Value> = serde_json::from_slice(&self.get(&url, Some((8, 8 + n - 1)))?).map_err(|e| e.to_string())?;
        Ok((8 + n, header))
    }
}

pub fn sha256_of(path: &Path) -> Result<String, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 24];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(crate::iq_pack::hex(&h.finalize()))
}

pub fn inventory(src: &Source, out: &Path) -> Result<Vec<Value>, String> {
    let index: Value = serde_json::from_slice(&src.get(&format!("{}model.safetensors.index.json", src.repo), None)?).map_err(|e| e.to_string())?;
    let wm = index["weight_map"].as_object().ok_or("the index has no weight_map")?;
    let mtp: Map<String, Value> = wm.iter().filter(|(k, _)| k.starts_with("mtp.")).map(|(k, v)| (k.clone(), v.clone())).collect();
    let mut shards: Vec<&str> = mtp.values().filter_map(Value::as_str).collect();
    shards.sort();
    shards.dedup();
    let (mut rows, mut total) = (Vec::new(), 0u64);
    // one small request pair per shard (28 of them): side by side, not one after the other (Strata 0.2: 28 s -> 4 s)
    let k = connections().min(8).min(shards.len()).max(1);
    let headers: Vec<Result<(u64, Map<String, Value>), String>> = std::thread::scope(|sc| {
        let hs: Vec<_> = shards.chunks(shards.len().div_ceil(k)).map(|part| sc.spawn(move || part.iter().map(|sh| src.shard_header(sh)).collect::<Vec<_>>())).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    for (shard, h) in shards.iter().zip(headers) {
        let (base, header) = h?;
        for (name, meta) in &header {
            if mtp.get(name).and_then(Value::as_str) == Some(*shard) {
                let off = meta["data_offsets"].as_array().ok_or("no data_offsets")?;
                let (a, b) = (off[0].as_u64().unwrap(), off[1].as_u64().unwrap());
                let mut r = Map::new();
                r.insert("name".into(), json!(name));
                r.insert("shard".into(), json!(shard));
                r.insert("dtype".into(), meta["dtype"].clone());
                r.insert("shape".into(), meta["shape"].clone());
                r.insert("start".into(), json!(base + a));
                r.insert("end".into(), json!(base + b - 1));
                r.insert("bytes".into(), json!(b - a));
                rows.push(Value::Object(r));
                total += b - a;
            }
        }
    }
    let have: std::collections::BTreeSet<&str> = rows.iter().map(|r| r["name"].as_str().unwrap()).collect();
    let mut missing: Vec<&str> = mtp.keys().map(String::as_str).filter(|k| !have.contains(k)).collect();
    if !missing.is_empty() {
        missing.sort();
        return Err(format!("tensors named in the index but absent from their shard headers: {missing:?}"));
    }
    rows.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let inv = json!({"repo": src.repo, "total_bytes": total, "tensors": rows});
    std::fs::write(out.join("mtp-inventory.json"), py_json(&inv, Some(1))).map_err(|e| e.to_string())?;
    let n_shards = rows.iter().map(|r| r["shard"].as_str().unwrap()).collect::<std::collections::BTreeSet<_>>().len();
    let mut lines = vec!["# MTP block in the BF16 checkpoint".to_string(), String::new(),
                         format!("{} tensors, {:.3} GB, in {n_shards} shards.", rows.len(), total as f64 / 1e9), String::new(),
                         "| tensor | dtype | shape | MB |".into(), "|---|---|---|---:|".into()];
    for r in &rows {
        let shape: Vec<String> = r["shape"].as_array().unwrap().iter().map(|d| d.to_string()).collect();
        lines.push(format!("| `{}` | {} | {} | {:.1} |", r["name"].as_str().unwrap(), r["dtype"].as_str().unwrap(), shape.join("x"),
                           r["bytes"].as_u64().unwrap() as f64 / 1e6));
    }
    let text = lines.join("\n") + "\n";
    std::fs::write(out.join("mtp-inventory.md"), &text).map_err(|e| e.to_string())?;
    println!("{text}");
    Ok(rows)
}

pub fn fetch(src: &Source, out: &Path, only: Option<&str>) -> Result<(), String> {
    let inv_path = out.join("mtp-inventory.json");
    let tdir = out.join("tensors");
    let inv: Option<Value> = std::fs::read_to_string(&inv_path).ok().and_then(|t| serde_json::from_str(&t).ok());
    let same = inv.as_ref().is_some_and(|i| i["repo"].as_str() == Some(&src.repo));
    let rows: Vec<Value> = if same { inv.unwrap()["tensors"].as_array().cloned().unwrap_or_default() } else { inventory(src, out)? };
    std::fs::create_dir_all(&tdir).map_err(|e| e.to_string())?;
    let mut manifest = Vec::new();
    let mut workers = connections();
    let chunk: u64 = if workers > 1 { 16 << 20 } else { 64 << 20 };
    let todo: Vec<&Value> = rows.iter().filter(|r| only.is_none_or(|o| r["name"].as_str().unwrap().contains(o))).collect();
    let size = |p: &PathBuf| std::fs::metadata(p).map(|m| m.len()).ok();
    let have_all: u64 = todo.iter().map(|r| {
        let p = tdir.join(format!("{}.bin", r["name"].as_str().unwrap()));
        size(&p).unwrap_or(0).min(r["bytes"].as_u64().unwrap())
    }).sum();
    let mut progress = Progress::new(todo.iter().map(|r| r["bytes"].as_u64().unwrap()).sum(), have_all);
    progress.show(true);
    for r in todo {
        let name = r["name"].as_str().unwrap();
        let (start, end, bytes) = (r["start"].as_u64().unwrap(), r["end"].as_u64().unwrap(), r["bytes"].as_u64().unwrap());
        let shard = r["shard"].as_str().unwrap();
        let path = tdir.join(format!("{name}.bin"));
        let want = if src.is_pinned() { want_sha(name) } else { None };
        if path.exists() && !same && (want.is_none() || size(&path) != Some(bytes) || sha256_of(&path)? != want.unwrap()) {
            let _ = std::fs::remove_file(&path);
        }
        let mut digest = String::new();
        let (mut attempt, mut net_fails) = (0, 0);
        while attempt < 2 {
            let mut have = size(&path).unwrap_or(0);
            if have > bytes {
                let _ = std::fs::remove_file(&path);
                have = 0;
            }
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path).map_err(|e| e.to_string())?;
            let got = src.ranged(&format!("{}{shard}", src.repo), start + have, end, chunk, workers, &mut |d: &[u8]| {
                f.write_all(d).map_err(|e| e.to_string())?;
                progress.add(d.len() as u64);
                Ok(())
            });
            drop(f);
            if let Err(e) = got {
                // a timeout or a dropped connection mid-tensor ends only this try: what was written stays, the next try
                // goes on after it with fewer connections (a link that drops 16 at once often holds 4)
                net_fails += 1;
                if net_fails > 3 {
                    return Err(format!("{name}: the download keeps failing ({e}). Check the connection and try again - what is \
                                        downloaded is kept; STRATA_SOURCE=modelscope reads the same files elsewhere"));
                }
                workers = (workers / 2).max(4).min(workers.max(1));
                eprintln!("{name}: {e} - continuing with {workers} connections (try {})", net_fails + 1);
                std::thread::sleep(Duration::from_secs((5 * net_fails as u64).min(20)));
                continue;
            }
            if size(&path) != Some(bytes) {
                return Err(format!("{}: size {} != {bytes}", path.display(), size(&path).unwrap_or(0)));
            }
            digest = sha256_of(&path)?;
            if want.is_none() || Some(digest.as_str()) == want {
                break;
            }
            progress.done = progress.done.saturating_sub(bytes);
            let _ = std::fs::remove_file(&path);
            attempt += 1;
            if attempt > 1 {
                return Err(format!("{name}: sha256 {digest} is not the checkpoint's {} - the download source sends wrong data \
                                    (a mirror that ignores Range requests?)", want.unwrap()));
            }
            eprintln!("{name}: wrong bytes (sha256 {}...), fetching it again", &digest[..12]);
        }
        let mut m = r.as_object().unwrap().clone();
        m.insert("file".into(), json!(format!("tensors/{name}.bin")));
        m.insert("sha256".into(), json!(digest));
        manifest.push(Value::Object(m));
        eprintln!("{name} done");
    }
    progress.show(true);
    std::fs::write(out.join("mtp-manifest.json"), py_json(&Value::Array(manifest), Some(1))).map_err(|e| e.to_string())
}

/// Ranges read at once: STRATA_DL_CONNECTIONS (setup's download setting), else 16; 1 reads them one by one.
pub fn connections() -> usize {
    std::env::var("STRATA_DL_CONNECTIONS").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(16).clamp(1, 32)
}

/// One line every two seconds, `  MTP draft layer: 1.23 / 5.04 GB (24%)`, the form of setup's download line (the
/// app's progress parser reads it). `done` counts what an earlier run already has.
struct Progress {
    total: u64,
    done: u64,
    last: Option<Instant>,
}

impl Progress {
    fn new(total: u64, done: u64) -> Self {
        Progress { total, done, last: None }
    }
    fn add(&mut self, n: u64) {
        self.done += n;
        self.show(false);
    }
    fn show(&mut self, force: bool) {
        if !force && self.last.is_some_and(|t| t.elapsed() < Duration::from_secs(2)) {
            return;
        }
        self.last = Some(Instant::now());
        if self.total > 0 {
            let d = self.done.min(self.total);
            eprintln!("  MTP draft layer: {:6.2} / {:.2} GB ({}%)", d as f64 / 1e9, self.total as f64 / 1e9, 100 * d / self.total);
        }
    }
}

/// Only one fetch writes a folder: a second one (a retry while the first still runs) waits for the first, then finds
/// its tensors done. Keep the returned file open; Err when the first does not end in `wait`.
pub fn lock_folder(out: &Path, wait: Duration) -> Result<std::fs::File, String> {
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let f = std::fs::OpenOptions::new().create(true).append(true).read(true).open(out.join(".mtp_fetch.lock")).map_err(|e| e.to_string())?;
    let end = Instant::now() + wait;
    let mut told = false;
    loop {
        use std::os::fd::AsRawFd;
        if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            return Ok(f);
        }
        if Instant::now() > end {
            return Err(format!("another MTP download is still writing {}: wait for it or end it, then try again", out.display()));
        }
        if !told {
            eprintln!("another MTP download is writing {}: waiting for it to end", out.display());
            told = true;
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}

/// STRATA_EXIT_WITH_PARENT=1 (set by a setup that runs this): end when the process that started this one ends, so a
/// cancelled setup does not leave a download running on its own.
pub fn exit_with_parent() {
    if std::env::var("STRATA_EXIT_WITH_PARENT").map(|v| v.trim() != "1").unwrap_or(true) {
        return;
    }
    let parent = unsafe { libc::getppid() };
    if parent <= 1 {
        return;
    }
    std::thread::spawn(move || loop {
        if unsafe { libc::getppid() } != parent {
            eprintln!("the process that started this download has ended: stopping");
            std::process::exit(1);
        }
        std::thread::sleep(Duration::from_millis(1500));
    });
}

/// The fetched tensors against SHA256 -> the names that are missing or wrong ([] when there is nothing to check).
/// A verdict is kept in tensors/verified.json by size and mtime, so a later run hashes only what changed.
pub fn verify(src: &Source, out: &Path) -> Vec<String> {
    let tdir = out.join("tensors");
    if !src.is_pinned() || !tdir.is_dir() {
        return Vec::new();
    }
    let stamp_path = tdir.join("verified.json");
    let stamps: Map<String, Value> = std::fs::read_to_string(&stamp_path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    let (mut bad, mut good) = (Vec::new(), Map::new());
    let mut names: Vec<&(&str, &str)> = SHA256.iter().collect();
    names.sort();
    for (name, want) in names {
        let path = tdir.join(format!("{name}.bin"));
        let Ok(md) = std::fs::metadata(&path) else {
            bad.push(name.to_string());
            continue;
        };
        use std::os::unix::fs::MetadataExt;
        let mtime_ns = md.mtime() as i128 * 1_000_000_000 + md.mtime_nsec() as i128;
        let key = json!([md.len(), mtime_ns as i64, want]);
        if stamps.get(*name) != Some(&key) && sha256_of(&path).ok().as_deref() != Some(*want) {
            bad.push(name.to_string());
            continue;
        }
        good.insert(name.to_string(), key);
    }
    let _ = std::fs::write(&stamp_path, py_json(&Value::Object(good), Some(1)));
    bad
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    /// One HTTP answer per connection, from a local server.
    fn serve(answers: Vec<String>) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = format!("http://{}/", l.local_addr().unwrap());
        std::thread::spawn(move || {
            for a in answers {
                let (mut s, _) = l.accept().unwrap();
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut line = String::new();
                while r.read_line(&mut line).unwrap() > 2 {
                    line.clear();
                }
                s.write_all(a.as_bytes()).unwrap();
            }
        });
        addr
    }

    #[test]
    fn a_range_read_must_be_answered_as_asked() {
        let mut src = Source::from_env();
        src.retries = 1;
        let base = serve(vec![
            "HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\nABCDEFGH".into(),
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 4-11/99\r\nContent-Length: 8\r\n\r\nABCDEFGH".into(),
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-7/99\r\nContent-Length: 8\r\n\r\nABCDEFGH".into(),
        ]);
        let e = src.get(&base, Some((0, 7))).unwrap_err();
        assert!(e.contains("not honoured"), "a 200 (the whole file) is refused: {e}");
        let e = src.get(&base, Some((0, 7))).unwrap_err();
        assert!(e.contains("Content-Range"), "another range is refused: {e}");
        assert_eq!(src.get(&base, Some((0, 7))).unwrap(), b"ABCDEFGH");
    }
}
