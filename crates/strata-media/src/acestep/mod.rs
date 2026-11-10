//! ACE-Step 1.5 in Rust (docs/ACESTEP-RUST-PLAN.md): songs with vocals from a caption and lyrics.
pub mod dit;
pub mod oobleck;
pub mod pipeline;
pub mod planner;
pub mod text;

use serde_json::{json, Value};
use std::path::PathBuf;

/// The song model in the media catalog: ACE-Step 1.5's turbo DiT, its caption encoder and its audio decoder.
pub const NAME: &str = "ace-step-1.5-turbo";
pub const REPO: &str = "ACE-Step/Ace-Step1.5";
pub const FILES: &[&str] = &[
    "Qwen3-Embedding-0.6B/config.json", "Qwen3-Embedding-0.6B/tokenizer.json", "Qwen3-Embedding-0.6B/model.safetensors",
    "acestep-v15-turbo/config.json", "acestep-v15-turbo/silence_latent.pt", "acestep-v15-turbo/model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
];
/// The song planner's files, downloaded with the rest; a song without them is made without the planner.
pub const PLANNER_FILES: &[&str] = &[
    "acestep-5Hz-lm-1.7B/config.json", "acestep-5Hz-lm-1.7B/tokenizer.json", "acestep-5Hz-lm-1.7B/model.safetensors",
];
pub const SIZE_GB: f64 = 9.8;
/// The vocal languages ACE-Step sings (its 50+; the most asked for first), as (code, name).
pub const LANGUAGES: &[(&str, &str)] = &[
    ("en", "English"), ("hi", "Hindi"), ("ml", "Malayalam"), ("ta", "Tamil"), ("te", "Telugu"), ("bn", "Bengali"),
    ("es", "Spanish"), ("fr", "French"), ("de", "German"), ("it", "Italian"), ("pt", "Portuguese"), ("nl", "Dutch"),
    ("pl", "Polish"), ("ru", "Russian"), ("tr", "Turkish"), ("ar", "Arabic"), ("zh", "Chinese"), ("ja", "Japanese"),
    ("ko", "Korean"), ("id", "Indonesian"), ("vi", "Vietnamese"), ("th", "Thai"), ("sv", "Swedish"), ("fi", "Finnish"),
];

fn cache() -> hf_hub::Cache {
    hf_hub::Cache::new(crate::audio::cache_dir())
}

/// The downloaded checkpoints folder (the snapshot holding the three subfolders), if every file is there.
pub fn local() -> Option<PathBuf> {
    let repo = cache().model(REPO.to_string());
    let files: Vec<PathBuf> = FILES.iter().map(|f| repo.get(f)).collect::<Option<_>>()?;
    files[0].parent()?.parent().map(PathBuf::from)
}

pub fn pull() -> anyhow::Result<()> {
    let api = hf_hub::api::sync::ApiBuilder::new().with_cache_dir(crate::audio::cache_dir()).with_progress(false).build()?;
    let repo = api.model(REPO.to_string());
    for f in FILES.iter().chain(PLANNER_FILES) {
        repo.get(f).map_err(|e| anyhow::anyhow!("{REPO}: {f}: {e}"))?;
    }
    Ok(())
}

/// Whether the song planner is downloaded too.
pub fn planner_local() -> bool {
    let repo = cache().model(REPO.to_string());
    PLANNER_FILES.iter().all(|f| repo.get(f).is_some())
}

/// Its row in the media catalog.
pub fn entry() -> Value {
    json!({"name": NAME, "display_name": "ACE-Step 1.5 turbo", "kind": "music", "family": "ace-step",
           "description": "songs with vocals from a style and lyrics, 10 s to 10 min, in 50+ languages (Rust, with ACE-Step's song planner)",
           "licence": "MIT", "size_gb": SIZE_GB, "gated": false, "downloaded": local().is_some(), "planner": planner_local(), "runtime": "candle",
           "languages": LANGUAGES.iter().map(|l| json!({"code": l.0, "name": l.1})).collect::<Vec<_>>(),
           "defaults": {"duration": 60, "steps": 8}})
}

/// The folder removing it deletes.
pub fn folder() -> PathBuf {
    crate::audio::cache_dir().join(format!("models--{}", REPO.replace('/', "--")))
}
