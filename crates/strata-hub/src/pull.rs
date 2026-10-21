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
    let name = spec.dir.clone().filter(|d| !d.is_empty() && !d.contains('/') && d != "..").unwrap_or(format!("{base}-{tag}"));
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
