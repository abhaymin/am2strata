//! The run config for a downloaded model (strata-<id>.json, beside the PolyStrata ones so both servers list it), with
//! its tokenizer folder extracted from the GGUF, written as PolyStrata's setup writes a strata-poly model's.

use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use strata_gguf::{extract_tokenizer, Gguf};

/// GGUF architectures strata-poly runs (src/poly/engine.cpp Engine::open), with their display family.
pub const POLY_ARCHS: [(&str, &str); 6] = [("qwen35moe", "Qwen3.5/3.6 MoE"), ("glm5next", "GLM-5 Next"), ("glm5-next", "GLM-5 Next"),
                                           ("gemma4", "Gemma 4"), ("deepseek4", "DeepSeek V4"), ("lfm2moe", "LFM2 MoE")];

/// GGUF architectures am2strata-hybrid runs (crates/strata-hybrid, checked against llama.cpp), with their family.
pub const HYBRID_ARCHS: [(&str, &str); 7] = [("nemotron_h_moe", "Nemotron-H MoE"), ("nemotron_h", "Nemotron-H"), ("deepseek2", "GLM-4.7-Flash / DeepSeek"),
                                             ("llama", "Llama / Mistral / Mixtral"), ("mistral3", "Mistral 3"), ("qwen3", "Qwen3"), ("qwen3moe", "Qwen3 MoE")];

pub struct Plan {
    pub id: String,
    pub display: String,
    pub arch: String,
    pub native_context: Option<u64>,
}

/// What a model file is and whether an engine here runs it.
pub fn inspect(first_shard: &Path) -> Result<Gguf, String> {
    Gguf::open(first_shard).map_err(|e| e.to_string())
}

pub fn supported(arch: &str) -> Result<&'static str, String> {
    if let Some((_, fam)) = POLY_ARCHS.iter().find(|(a, _)| *a == arch) {
        return Ok(fam);
    }
    if let Some((_, fam)) = HYBRID_ARCHS.iter().find(|(a, _)| *a == arch) {
        return Ok(fam);
    }
    if arch == "qwen4exp" {
        return Err("a qwen4exp (Qwen3.8-Flash-Next family) model runs on the strata engine, which needs a prepared pack \
                    that only PolyStrata's setup builds today (not ported to am2Strata yet)".into());
    }
    let list: Vec<&str> = POLY_ARCHS.iter().chain(HYBRID_ARCHS.iter()).map(|(a, _)| *a).collect();
    Err(format!("its architecture is '{arch}', which no engine here runs yet (the engines run: {})", list.join(", ")))
}

/// A config id from a file name: lower case, the shard suffix and extension dropped.
pub fn id_of(file: &str) -> String {
    let stem = file.trim_end_matches(".gguf");
    let stem = match stem.rsplit_once("-of-") {
        Some((head, _)) => head.rsplit_once('-').map(|(b, _)| b).unwrap_or(stem),
        None => stem,
    };
    stem.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' { c } else { '-' }).collect()
}

/// The keys a new config copies from an existing one, so it serves the same way (port, address, key, ...).
const SHARED_KEYS: [&str; 7] = ["port", "host", "api_key", "lib_dirs", "trusted_origins", "cors_origins", "fit_max_tokens"];

pub struct Written {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
}

/// Write the tokenizer folder and strata-<id>.json. `models_dir`: the PolyStrata folder (configs, engines);
/// `packs`: where the model's tokenizer and expert profile go; `mmproj`: the picture encoder file, if fetched.
pub fn write(models_dir: &Path, packs: &Path, model: &Path, g: &Gguf, mmproj: Option<&Path>, context: Option<u64>,
             id: &str, display: &str) -> Result<Written, String> {
    let arch = g.architecture().unwrap_or("").to_string();
    supported(&arch)?;
    let pack = packs.join(id);
    extract_tokenizer(g, &pack).map_err(|e| e.to_string())?;
    let poly = models_dir.join("engine-poly");
    let native = g.arch_u64("context_length");
    let ctx = context.or(native).unwrap_or(32768).min(native.unwrap_or(u64::MAX));
    if HYBRID_ARCHS.iter().any(|(a, _)| *a == arch) {
        // the Rust engine: its pack is built on the first load (am2Strata runs iq-pack before starting the engine), the
        // expert profile is learned while it runs, and the hottest experts stay in RAM
        let ctx = context.unwrap_or(ctx.min(32768));
        let mut args = vec![json!("--model"), json!(model), json!("--max-context"), json!(ctx.to_string()), json!("--pack"), json!(pack)];
        if g.arch_u64("expert_count").is_some_and(|n| n > 0) {
            let prof = pack.join("expert-profile.bin");
            args.extend([json!("--expert-profile"), json!(prof), json!("--expert-profile-save"), json!(prof), json!("--resident-experts")]);
        }
        let cfg = json!({
            "exe": "${AM2STRATA_HOME}/crates/strata-media/target/release/am2strata-hybrid", "args": args, "cwd": "${AM2STRATA_HOME}",
            "tokenizer": pack.join("tokenizer"), "model_name": id, "display_name": display,
            "log": models_dir.join(format!("strata-{id}.log")), "pulled_by": "am2strata",
        });
        let path = models_dir.join(format!("strata-{id}.json"));
        std::fs::write(&path, pretty1(&cfg)).map_err(|e| format!("{}: {e}", path.display()))?;
        return Ok(Written { config: path, tokenizer: pack.join("tokenizer") });
    }
    let mut args = vec![json!("--model"), json!(model), json!("--max-context"), json!(ctx.to_string()),
                        json!("--expert-profile"), json!(pack.join("expert-profile.txt")),
                        json!("--conversation-cache-mib"), json!("8192")];
    if arch == "qwen35moe" {
        args.extend([json!("--kv"), json!("q8_0")]);
    }
    let mut cfg = json!({
        "exe": poly.join("strata-poly"), "args": args, "cwd": models_dir, "tokenizer": pack.join("tokenizer"),
        "model_name": id, "display_name": display, "log": models_dir.join(format!("strata-{id}.log")), "gpu": 0,
        "pulled_by": "am2strata",
    });
    // serve like the installed models: copy the shared keys from an existing config (the first one found)
    if let Some(existing) = first_config(models_dir) {
        if let Some(o) = existing.as_object() {
            for k in SHARED_KEYS {
                if let Some(v) = o.get(k) {
                    cfg[k] = v.clone();
                }
            }
        }
    }
    if cfg.get("lib_dirs").is_none() {
        cfg["lib_dirs"] = json!(["/usr/local/cuda/bin", "/usr/local/cuda/lib64"]);
    }
    if let Some(mm) = mmproj {
        cfg["vision"] = json!({"exe": poly.join("strata-vision"), "mmproj": mm, "model": model, "gpu": true, "max_tokens": 1024});
    }
    let path = models_dir.join(format!("strata-{id}.json"));
    std::fs::write(&path, pretty1(&cfg)).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Written { config: path, tokenizer: pack.join("tokenizer") })
}

/// JSON indented by one space, as PolyStrata's configs are (json.dumps(indent=1)).
fn pretty1(v: &Value) -> String {
    use serde::Serialize;
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, serde_json::ser::PrettyFormatter::with_indent(b" "));
    v.serialize(&mut ser).unwrap();
    String::from_utf8(buf).unwrap()
}

fn first_config(dir: &Path) -> Option<Value> {
    let mut names: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("strata-") && n.ends_with(".json")))
        .collect();
    names.sort();
    names.into_iter().find_map(|p| {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&p).ok()?).ok()?;
        (v.get("pulled_by").is_none() && v.get("exe").is_some()).then_some(v)
    })
}

#[allow(dead_code)]
fn _unused(_: Map<String, Value>) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_and_support() {
        assert_eq!(id_of("Qwen3.6-35B-A3B-UD-Q4_K_XL-00001-of-00002.gguf"), "qwen3.6-35b-a3b-ud-q4_k_xl");
        assert!(supported("qwen35moe").is_ok());
        assert!(supported("qwen4exp").unwrap_err().contains("pack"));
        assert!(supported("llama").is_ok());
        assert!(supported("gpt2").unwrap_err().contains("'gpt2'"));
    }
}
