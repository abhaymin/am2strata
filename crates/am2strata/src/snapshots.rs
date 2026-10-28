//! Prompt snapshots (docs/KNOWLEDGE-CACHE.md phase E): the conversation an engine holds, saved to the SSD with the
//! engine's SAVE and read back with RESTORE, so a long prompt is not read again after the model was stopped (a media
//! job, a switch, an unload, a restart) or after other conversations pushed it out of the engine's cache.
//!
//! - **Saved** before a model with session files is stopped, and after a request whose prompt is long
//!   (LONG_PROMPT tokens) when no snapshot already covers most of it (an agent's system prompt and tools).
//! - **Restored** before a request when a snapshot shares at least MIN_GAIN more tokens with its prompt than the
//!   engine holds now. The engine checks the file against the model (fingerprint, geometry); a refusal drops it.
//! - **Kept** under `<data>/cache/kv/<model>/`, one `.sess` file and its token ids (`.ids`, u32 little-endian) each,
//!   within a size budget ("session_cache_gb" in the run config, else a quarter of the free disk up to 40 GB),
//!   the least recently used going first.
//!
//! Only engines with session files take part: `strata --serve` (Orca's engine) with --prompt-cache > 0, no --batch
//! and no --layer-split; strata-poly and am2strata-hybrid have no SAVE. "session_cache": false turns it off.

use crate::manager::Loaded;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

/// A snapshot is restored when it gives at least this many prompt tokens more than the engine holds.
pub const MIN_GAIN: usize = 2048;
/// A conversation shorter than this is not worth a file.
pub const MIN_SAVE: usize = 2048;
/// A prompt this long is saved after its request (unless a snapshot covers 90% of it already).
pub const LONG_PROMPT: usize = 16384;
/// How long the engine may be silent during a session file before it is given up on.
const SILENCE: Duration = Duration::from_secs(120);

#[derive(Clone)]
struct Snap {
    model: String,
    file: PathBuf,
    ids: Vec<u32>,
    bytes: u64,
    used: u64,
}

pub struct Snapshots {
    dir: PathBuf,
    budget: Mutex<Option<u64>>,
    index: Mutex<Vec<Snap>>,
    /// saved / restored / refused counts and tokens, for Monitor
    stats: Mutex<Value>,
}

fn now() -> u64 {
    crate::now()
}

/// The common prefix of two token lists.
pub fn common(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn safe(id: &str) -> String {
    id.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' }).collect()
}

fn read_ids(p: &Path) -> Option<Vec<u32>> {
    let b = std::fs::read(p).ok()?;
    Some(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn free_disk(p: &Path) -> Option<u64> {
    let out = std::process::Command::new("df").arg("-B1").arg("--output=avail").arg(p).output().ok()?;
    String::from_utf8_lossy(&out.stdout).lines().nth(1)?.trim().parse().ok()
}

impl Snapshots {
    pub fn open(data_dir: &Path) -> Snapshots {
        let dir = data_dir.join("cache").join("kv");
        let mut index = Vec::new();
        for m in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let model = m.file_name().to_string_lossy().into_owned();
            for f in std::fs::read_dir(m.path()).into_iter().flatten().flatten() {
                let file = f.path();
                if file.extension().is_some_and(|x| x == "sess") {
                    if let Some(ids) = read_ids(&file.with_extension("ids")) {
                        let meta = f.metadata().ok();
                        let used = meta.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| d.as_secs()).unwrap_or(0);
                        index.push(Snap { model: model.clone(), file, ids, bytes: meta.map(|m| m.len()).unwrap_or(0), used });
                    }
                }
            }
        }
        Snapshots { dir, budget: Mutex::new(None), index: Mutex::new(index),
                    stats: Mutex::new(json!({"saved": 0, "restored": 0, "refused": 0, "tokens_restored": 0})) }
    }

    /// The engine of this model has session files (and the config has not turned them off).
    pub fn capable(l: &Loaded) -> bool {
        let cfg = &l.entry.cfg;
        if cfg.rest.get("session_cache") == Some(&Value::Bool(false)) {
            return false;
        }
        if cfg.rest.get("session_files") == Some(&Value::Bool(true)) {
            return true; // the config says its engine has SAVE / RESTORE
        }
        let exe = cfg.exe.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let args: Vec<String> = cfg.args.iter().map(|a| a.to_string()).collect();
        let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
        exe == "strata" && arg("--prompt-cache").is_none_or(|n| n != "0") && arg("--batch").is_none_or(|n| n == "0")
            && !args.iter().any(|a| a == "--layer-split")
    }

    fn budget_for(&self, rest: &std::collections::BTreeMap<String, Value>) -> u64 {
        if let Some(gb) = rest.get("session_cache_gb").and_then(Value::as_f64) {
            return (gb * 1e9) as u64;
        }
        let mut b = self.budget.lock().unwrap();
        *b.get_or_insert_with(|| {
            let _ = std::fs::create_dir_all(&self.dir);
            let used: u64 = self.index.lock().unwrap().iter().map(|s| s.bytes).sum();
            ((free_disk(&self.dir).unwrap_or(0) + used) / 4).min(40_000_000_000)
        })
    }

    pub fn stats(&self) -> Value {
        let idx = self.index.lock().unwrap();
        let mut v = self.stats.lock().unwrap().clone();
        v["files"] = json!(idx.len());
        v["bytes"] = json!(idx.iter().map(|s| s.bytes).sum::<u64>());
        v
    }

    fn bump(&self, k: &str, by: u64) {
        let mut s = self.stats.lock().unwrap();
        s[k] = json!(s[k].as_u64().unwrap_or(0) + by);
    }

    /// The snapshot sharing the most tokens with `ids` for this model -> (its file, its ids, the shared count).
    fn best(&self, model: &str, ids: &[u32]) -> Option<(PathBuf, Vec<u32>, usize)> {
        let idx = self.index.lock().unwrap();
        idx.iter().filter(|s| s.model == model).map(|s| (s, common(&s.ids, ids))).max_by_key(|(_, n)| *n)
            .map(|(s, n)| (s.file.clone(), s.ids.clone(), n))
    }

    fn forget(&self, file: &Path) {
        self.index.lock().unwrap().retain(|s| s.file != file);
        let _ = std::fs::remove_file(file);
        let _ = std::fs::remove_file(file.with_extension("ids"));
    }

    /// Before a request (the caller holds the turn): restore the snapshot that gives the most of `ids` when it gives
    /// MIN_GAIN more than the engine holds. -> the tokens restored.
    pub async fn before(&self, l: &Loaded, ids: &[u32], has_images: bool) -> Option<usize> {
        if has_images || ids.len() < MIN_GAIN || !Self::capable(l) {
            return None;
        }
        let model = safe(&l.entry.id);
        let held = common(&l.held.lock().unwrap(), ids);
        let (file, snap_ids, n) = self.best(&model, ids)?;
        if n < held + MIN_GAIN {
            return None;
        }
        match l.engine.session(false, &file, SILENCE).await {
            Ok((tokens, bytes, ms)) => {
                tracing::info!(model = %l.entry.id, tokens, gb = bytes as f64 / 1e9, ms, shared = n,
                               "prompt snapshot restored: {n} of the prompt's {} tokens need not be read", ids.len());
                *l.held.lock().unwrap() = snap_ids;
                if let Some(s) = self.index.lock().unwrap().iter_mut().find(|s| s.file == file) {
                    s.used = now();
                }
                let _ = std::fs::File::options().append(true).open(&file).and_then(|f| f.set_modified(std::time::SystemTime::now()));
                self.bump("restored", 1);
                self.bump("tokens_restored", n as u64);
                Some(n)
            }
            Err(e) => {
                // another model's, an older engine's, or a damaged file: it goes
                tracing::warn!(model = %l.entry.id, file = %file.display(), "prompt snapshot not restored ({e}); removed");
                self.bump("refused", 1);
                self.forget(&file);
                None
            }
        }
    }

    /// After a request with a long prompt (the caller still holds the turn): saved when no snapshot covers 90% of it.
    pub async fn after(&self, l: &Loaded, prompt_len: usize) {
        if prompt_len < LONG_PROMPT || !Self::capable(l) {
            return;
        }
        let held = l.held.lock().unwrap().clone();
        let model = safe(&l.entry.id);
        if self.best(&model, &held).is_some_and(|(_, _, n)| n * 10 >= prompt_len * 9) {
            return;
        }
        self.save(l, "a long prompt").await;
    }

    /// Save what the engine holds (before the model stops, or after a long prompt).
    pub async fn save(&self, l: &Loaded, why: &str) {
        let held = l.held.lock().unwrap().clone();
        if held.len() < MIN_SAVE || !Self::capable(l) {
            return;
        }
        let model = safe(&l.entry.id);
        // the conversation it holds may already be on disk (restored, or saved before): nothing new to write
        if self.best(&model, &held).is_some_and(|(_, ids, n)| n == held.len() && ids.len() == held.len()) {
            return;
        }
        let dir = self.dir.join(&model);
        if std::fs::create_dir_all(&dir).is_err() {
            return;
        }
        let file = dir.join(format!("{:x}-{}.sess", now(), held.len()));
        let t0 = std::time::Instant::now();
        match l.engine.session(true, &file, SILENCE).await {
            Ok((tokens, bytes, _)) => {
                let raw: Vec<u8> = held.iter().flat_map(|t| t.to_le_bytes()).collect();
                if std::fs::write(file.with_extension("ids"), raw).is_err() {
                    let _ = std::fs::remove_file(&file);
                    return;
                }
                tracing::info!(model = %l.entry.id, tokens, gb = bytes as f64 / 1e9, s = t0.elapsed().as_secs_f64(),
                               "prompt snapshot saved ({why})");
                // a snapshot this one extends is not needed any more
                let older: Vec<PathBuf> = self.index.lock().unwrap().iter()
                    .filter(|s| s.model == model && s.ids.len() <= held.len() && common(&s.ids, &held) == s.ids.len())
                    .map(|s| s.file.clone()).collect();
                for f in older {
                    self.forget(&f);
                }
                self.index.lock().unwrap().push(Snap { model, file, ids: held, bytes, used: now() });
                self.bump("saved", 1);
                self.trim(self.budget_for(&l.entry.cfg.rest));
            }
            Err(e) => {
                tracing::info!(model = %l.entry.id, "prompt snapshot not saved: {e}");
                let _ = std::fs::remove_file(&file);
            }
        }
    }

    /// The least recently used snapshots go until the rest fit the budget.
    fn trim(&self, budget: u64) {
        loop {
            let victim = {
                let idx = self.index.lock().unwrap();
                let total: u64 = idx.iter().map(|s| s.bytes).sum();
                if total <= budget || idx.len() <= 1 {
                    return;
                }
                idx.iter().min_by_key(|s| s.used).map(|s| s.file.clone())
            };
            match victim {
                Some(f) => self.forget(&f),
                None => return,
            }
        }
    }

    /// Settings › Caches: every snapshot of every model goes.
    pub fn clear(&self) -> u64 {
        let files: Vec<(PathBuf, u64)> = self.index.lock().unwrap().iter().map(|s| (s.file.clone(), s.bytes)).collect();
        let freed = files.iter().map(|(_, b)| b).sum();
        for (f, _) in files {
            self.forget(&f);
        }
        freed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shared_prefix_is_counted() {
        assert_eq!(common(&[1, 2, 3, 4], &[1, 2, 9]), 2);
        assert_eq!(common(&[], &[1]), 0);
        assert_eq!(common(&[5, 6], &[5, 6, 7]), 2);
        assert_eq!(safe("orca/../x y"), "orca_.._x_y");
    }
}
