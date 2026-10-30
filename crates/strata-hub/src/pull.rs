//! One model from Hugging Face, end to end: the files of a quantization (or named files), downloaded and checked,
//! then its run config when an engine here runs its architecture. Shared by `am2strata-run pull` and the web app's
//! Settings › Models (POST /api/hub/pull); progress comes back as events.

use crate::{config, free_space, human, local_path, quant_tag, Hub, RepoFile};
use std::path::{Path, PathBuf};

pub struct Spec {
    pub repo: String,
    pub revision: String,
    pub quant: Option<String>,
    pub files: Vec<String>,
    pub token: Option<String>,
    pub mmproj: bool,
    /// where the model files and packs go (Strata-data)
    pub data_dir: PathBuf,
    /// the PolyStrata folder: engines, and where the run config is written
    pub models_dir: PathBuf,
    pub context: Option<u64>,
    pub write_config: bool,
    /// the folder under models/ (default: the repository's name and the quantization)
    pub dir: Option<String>,
}

pub enum Event {
    Note(String),
    /// file name, bytes done, bytes in all
    Progress { file: String, done: u64, total: u64 },
    FileDone { file: String, checked: bool },
}

pub enum Outcome {
    /// the run config was written: the model's id
    Configured { id: String, config: PathBuf },
    /// downloaded; no config (an architecture no engine runs, or `write_config` false)
    Downloaded { dir: PathBuf, why: Option<String> },
}

/// `owner/name` from what a person pastes (a hub address works too).
/// A name that stays inside the folder it is joined to.
pub fn safe_name(n: &str) -> bool {
    !n.is_empty() && n != "." && n != ".." && !n.contains(['/', '\\', '\0']) && !n.starts_with('-')
}

pub fn repo_name(s: &str) -> String {
    s.trim().trim_start_matches("https://huggingface.co/").trim_start_matches("hf.co/").trim_end_matches('/').to_string()
}

pub async fn run(spec: &Spec, ev: &mut (dyn FnMut(Event) + Send)) -> Result<Outcome, String> {
    let repo = repo_name(&spec.repo);
    let hub = Hub::new(spec.token.clone()).map_err(|e| e.to_string())?;
    let all = hub.files(&repo, &spec.revision).await.map_err(|e| e.to_string())?;
    let chosen: Vec<RepoFile> = if !spec.files.is_empty() {
        spec.files.iter().map(|want| all.iter().find(|f| &f.path == want).cloned().ok_or(format!("{repo} has no file {want}")))
            .collect::<Result<_, _>>()?
    } else if let Some(q) = &spec.quant {
        crate::select(&all, q, spec.mmproj).map_err(|e| e.to_string())?
    } else {
        return Err("name a quantization or the files to fetch".into());
    };
    let total: u64 = chosen.iter().map(|f| f.size).sum();
    let tag = quant_tag(&chosen[0].path).or_else(|| spec.quant.as_ref().map(|q| q.to_uppercase())).unwrap_or_else(|| "model".into());
    let base = repo.rsplit('/').next().unwrap_or(&repo).trim_end_matches("-GGUF").trim_end_matches("-gguf").to_string();
    let name = spec.dir.clone().filter(|d| safe_name(d)).unwrap_or(format!("{base}-{tag}"));
    // the folder is named from the request (its quant) and the repository: one plain name, never a path
    if !safe_name(&name) {
        return Err(format!("{name:?} cannot be a folder name"));
    }
    if let Some(bad) = chosen.iter().map(|f| crate::file_name(&f.path)).find(|n| !safe_name(n)) {
        return Err(format!("the repository lists a file named {bad:?}, which cannot be saved"));
    }
    let dir = spec.data_dir.join("models").join(name);
    ev(Event::Note(format!("{} file(s), {} -> {}", chosen.len(), human(total), dir.display())));
    let have: u64 = chosen.iter().map(|f| std::fs::metadata(local_path(&dir, f)).map(|m| m.len()).unwrap_or(0)).sum();
    if let Some(free) = free_space(&dir) {
        if free + have < total + (1 << 30) {
            return Err(format!("not enough disk space: {} needed, {} free", human(total - have.min(total)), human(free)));
        }
    }
    for f in &chosen {
        let name = f.path.rsplit('/').next().unwrap_or(&f.path).to_string();
        let mut show = |done: u64, all: u64| ev(Event::Progress { file: name.clone(), done, total: all });
        hub.download(&repo, &spec.revision, f, &local_path(&dir, f), &mut show).await.map_err(|e| e.to_string())?;
        ev(Event::FileDone { file: name, checked: f.sha256.is_some() });
        record_source(&dir, &repo, &spec.revision, f)?;
    }
    if !spec.write_config {
        return Ok(Outcome::Downloaded { dir, why: None });
    }
    let models: Vec<&RepoFile> = chosen.iter().filter(|f| !f.path.to_lowercase().contains("mmproj")).collect();
    let first = local_path(&dir, models.first().ok_or("no model file among the chosen ones")?);
    let mmproj = chosen.iter().find(|f| f.path.to_lowercase().contains("mmproj")).map(|f| local_path(&dir, f));
    let g = config::inspect(&first)?;
    let arch = g.architecture().unwrap_or("?").to_string();
    ev(Event::Note(format!("architecture {arch}, context {}, {} experts",
                           g.arch_u64("context_length").map(|c| c.to_string()).unwrap_or("?".into()),
                           g.arch_u64("expert_count").map(|c| c.to_string()).unwrap_or("no".into()))));
    if let Err(why) = config::supported(&arch) {
        return Ok(Outcome::Downloaded { dir, why: Some(why) });
    }
    let file_name = first.file_name().and_then(|n| n.to_str()).unwrap_or("model");
    let id = config::id_of(file_name);
    let display = format!("{} {tag}", g.str("general.name").filter(|n| !n.is_empty()).unwrap_or(&base));
    let w = config::write(&spec.models_dir, &spec.data_dir.join("packs"), &first, &g, mmproj.as_deref(), spec.context, &id, &display)?;
    // a file without a chat template gets the model's own from the model's repository, pinned like the files (setup.py
    // "template"): maternion's Clover files have none, DiffuSpace's repository has it
    if arch == "clover_moe" {
        let tpl = w.tokenizer.join("chat_template.jinja");
        if !tpl.exists() {
            let f = RepoFile { path: "chat_template.jinja".into(), size: 7912,
                               sha256: Some("01094df7cd8905d1220ea8b31e4f1086dc9bedf0305d3c2fa6578b5fa33b6815".into()) };
            let mut quiet = |_: u64, _: u64| {};
            hub.download("Diffuspace/Clover-35B-A3B-Preview", "4b0402b65e523d0183a14c431955438714cb16ff", &f, &tpl, &mut quiet).await.map_err(|e| e.to_string())?;
            ev(Event::Note("chat template: from Diffuspace/Clover-35B-A3B-Preview".into()));
        }
    }
    // a Strata expert profile published beside the model (*.prof, STRP): in the form its engine reads
    if let Some(pf) = all.iter().find(|f| f.path.ends_with(".prof") && f.size < (16 << 20)) {
        let pack = spec.data_dir.join("packs").join(&id);
        let raw = pack.join("expert-profile.strp");
        let mut quiet = |_: u64, _: u64| {};
        match hub.download(&repo, &spec.revision, pf, &raw, &mut quiet).await {
            Ok(()) => match profile_for_config(&raw, &w.config) {
                Ok(note) => ev(Event::Note(format!("{}: {note}", pf.path))),
                Err(e) => ev(Event::Note(format!("{}: not used ({e})", pf.path))),
            },
            Err(e) => ev(Event::Note(format!("{}: not downloaded ({e})", pf.path))),
        }
    }
    Ok(Outcome::Configured { id, config: w.config })
}

/// The quantizations a repository offers: (quant, bytes, files).
pub async fn list(repo: &str, token: Option<String>) -> Result<Vec<(String, u64, usize)>, String> {
    let repo = repo_name(repo);
    let hub = Hub::new(token).map_err(|e| e.to_string())?;
    let files = hub.files(&repo, "main").await.map_err(|e| e.to_string())?;
    Ok(crate::quants(&files))
}

#[allow(dead_code)]
fn _p(_: &Path) {}

/// Where a downloaded model came from (<model dir>/am2strata-source.json): the repository, the revision and each file's
/// size and SHA-256, so that a newer upload of the same files can be found later (`check_update`).
pub const SOURCE_FILE: &str = "am2strata-source.json";

fn record_source(dir: &Path, repo: &str, revision: &str, f: &RepoFile) -> Result<(), String> {
    let path = dir.join(SOURCE_FILE);
    let mut v: serde_json::Value = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok())
        .filter(|v: &serde_json::Value| v["repo"] == repo)
        .unwrap_or_else(|| serde_json::json!({"repo": repo, "revision": revision, "files": {}}));
    v["files"][&f.path] = serde_json::json!({"size": f.size, "sha256": f.sha256});
    v["updated"] = serde_json::json!(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0));
    std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).map_err(|e| format!("{}: {e}", path.display()))
}

/// A model folder's files against the Hub now: the repository, and the files whose SHA-256 or size changed there
/// (or that were removed from it).
pub struct UpdateCheck {
    pub repo: String,
    pub revision: String,
    pub changed: Vec<String>,
    pub removed: Vec<String>,
    pub bytes: u64,
}

pub async fn check_update(dir: &Path, token: Option<String>) -> Result<UpdateCheck, String> {
    let path = dir.join(SOURCE_FILE);
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)
        .map_err(|_| "this model was not downloaded by am2Strata, so its source is not known".to_string())?)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let repo = v["repo"].as_str().ok_or("the source record names no repository")?.to_string();
    let revision = v["revision"].as_str().unwrap_or("main").to_string();
    let hub = Hub::new(token).map_err(|e| e.to_string())?;
    let now = hub.files(&repo, &revision).await.map_err(|e| e.to_string())?;
    let (mut changed, mut removed, mut bytes) = (Vec::new(), Vec::new(), 0u64);
    for (p, was) in v["files"].as_object().into_iter().flatten() {
        match now.iter().find(|f| &f.path == p) {
            None => removed.push(p.clone()),
            Some(f) if f.sha256.as_deref() != was["sha256"].as_str() || Some(f.size) != was["size"].as_u64() => {
                changed.push(p.clone());
                bytes += f.size;
            }
            _ => {}
        }
    }
    Ok(UpdateCheck { repo, revision, changed, removed, bytes })
}

/// A STRP expert profile (PolyStrata's ranked (layer, expert) list) as strata-poly's "layer expert count" lines: the
/// hottest expert gets the highest count. Layers past `n_layers` (an MTP layer) are left out.
pub fn strp_to_counts(strp: &[u8]) -> Result<String, String> {
    if strp.len() < 24 || &strp[..4] != b"STRP" {
        return Err("not a STRP expert profile".into());
    }
    let h = |i: usize| u32::from_le_bytes(strp[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
    let (version, n) = (h(0), h(4));
    if version != 1 || strp.len() < 24 + 4 * n {
        return Err(format!("a version {version} profile or a truncated one"));
    }
    let mut out = String::new();
    for r in 0..n {
        let at = 24 + 4 * r;
        let (l, e) = (u16::from_le_bytes([strp[at], strp[at + 1]]), u16::from_le_bytes([strp[at + 2], strp[at + 3]]));
        out.push_str(&format!("{l} {e} {}\n", n - r));
    }
    Ok(out)
}

/// Puts a downloaded STRP profile where the config's engine reads it: strata-poly's `--expert-profile` text file, or the
/// file itself for engines that read STRP (strata, am2strata-hybrid).
fn profile_for_config(raw: &Path, config: &Path) -> Result<String, String> {
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(config).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let args: Vec<&str> = cfg["args"].as_array().map(|a| a.iter().filter_map(|v| v.as_str()).collect()).unwrap_or_default();
    let target = args.windows(2).find(|w| w[0] == "--expert-profile").map(|w| PathBuf::from(w[1])).ok_or("the config takes no expert profile")?;
    let bytes = std::fs::read(raw).map_err(|e| e.to_string())?;
    if cfg["exe"].as_str().unwrap_or("").contains("strata-poly") {
        let text = strp_to_counts(&bytes)?;
        std::fs::write(&target, text).map_err(|e| format!("{}: {e}", target.display()))?;
        Ok(format!("written as strata-poly's expert counts ({})", target.display()))
    } else {
        std::fs::copy(raw, &target).map_err(|e| format!("{}: {e}", target.display()))?;
        Ok(format!("copied to {}", target.display()))
    }
}

#[cfg(test)]
mod safety {
    #[test]
    fn folder_and_file_names_stay_inside() {
        for bad in ["", ".", "..", "a/b", "../x", "a\\b", "-rf"] {
            assert!(!super::safe_name(bad), "{bad:?}");
        }
        assert!(super::safe_name("Qwen3-8B-Q4_K_M") && super::safe_name("model.gguf"));
    }
}
