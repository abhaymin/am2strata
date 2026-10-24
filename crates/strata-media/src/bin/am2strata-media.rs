//! am2strata-media: the media engine process (images now; video and voice next), driven by am2Strata's server over
//! JSON lines on stdin/stdout, as the text engines are driven over their line protocol. One request at a time.
//!
//!   -> {"op":"pull","model":"z-image-turbo:q8"}           <- {"event":"pulled","model":...}
//!   -> {"op":"load","model":"z-image-turbo:q8"}           <- {"event":"loaded","model":...,"vram_bytes":...}
//!   -> {"op":"generate","request":{prompt,width,...}}     <- {"event":"step","step":3,"total":8} ...
//!                                                         <- {"event":"image","index":0,"width":..,"height":..,"png":"<base64>"}
//!                                                         <- {"event":"video","format":"mp4","frames":..,"fps":..,"data":"<base64>"}
//!                                                         <- {"event":"done","ms":...}
//!   -> {"op":"unload"}                                    <- {"event":"unloaded"}
//!   -> {"op":"list"}                                      <- {"event":"models","models":[{name,kind,family,...}]}
//!   any failure                                           <- {"event":"error","message":...}
//!
//! Models and their files come from mold's catalog, stored under $MOLD_HOME (am2Strata sets <Strata-data>/media).

use anyhow::{anyhow, Context, Result};
use base64::Engine as _;
use mold_inference::engine::{InferenceEngine, LoadStrategy};
use mold_inference::progress::ProgressEvent;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

fn emit(v: Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{v}");
    let _ = out.flush();
}

struct State {
    config: mold_core::Config,
    engine: Option<Box<dyn InferenceEngine>>,
    model: Option<String>,
}

fn pull(rt: &tokio::runtime::Runtime, model: &str) -> Result<()> {
    let manifest = mold_core::manifest::find_manifest(model).ok_or_else(|| anyhow!("no model {model:?} in the catalog"))?;
    rt.block_on(mold_core::download::pull_model(manifest, &mold_core::download::PullOptions::default()))
        .map_err(|e| anyhow!("{e}"))?;
    Ok(())
}

/// The catalog: every generation model mold knows, with its kind (image or video) and whether it is downloaded.
fn list(st: &mut State) -> Value {
    st.config = mold_core::Config::load_or_default();
    let models: Vec<Value> = mold_core::manifest::visible_manifests()
        .filter(|m| m.is_generation_model())
        .map(|m| {
            let f = m.family.to_lowercase();
            let kind = if f.starts_with("hunyuan3d") {
                "3d"
            } else if f.starts_with("wan") || f.starts_with("ltx") || f.contains("video") || f.starts_with("minimax-h3") {
                "video"
            } else {
                "image"
            };
            json!({"name": m.name, "display_name": m.display_name(), "family": m.family, "kind": kind,
                   "description": m.description, "size_gb": (m.total_size_bytes() as f64 / 1e8).round() / 10.0, "gated": m.is_gated(),
                   "downloaded": mold_core::ModelPaths::resolve(&m.name, &st.config).is_some(),
                   "loaded": st.model.as_deref() == Some(m.name.as_str()),
                   "defaults": {"steps": m.defaults.steps, "guidance": m.defaults.guidance, "width": m.defaults.width,
                                "height": m.defaults.height, "frames": m.defaults.frames, "fps": m.defaults.fps}})
        })
        .collect();
    json!({"event": "models", "models": models})
}

fn load(st: &mut State, model: &str) -> Result<()> {
    if st.model.as_deref() == Some(model) && st.engine.as_ref().is_some_and(|e| e.is_loaded()) {
        return Ok(());
    }
    if let Some(mut e) = st.engine.take() {
        e.unload();
    }
    st.config = mold_core::Config::load_or_default();
    let paths = mold_core::ModelPaths::resolve(model, &st.config)
        .ok_or_else(|| anyhow!("{model} is not downloaded (pull it first)"))?;
    let mut engine = mold_inference::create_engine(model.to_string(), paths, &st.config, LoadStrategy::Eager, 0, false)
        .with_context(|| format!("creating the engine for {model}"))?;
    engine.load().with_context(|| format!("loading {model}"))?;
    st.engine = Some(engine);
    st.model = Some(model.to_string());
    Ok(())
}

fn generate(st: &mut State, req: &Value) -> Result<()> {
    let mut req = req.clone();
    if req.get("model").is_none() {
        req["model"] = json!(st.model.clone().ok_or_else(|| anyhow!("no model loaded"))?);
    }
    let model = req["model"].as_str().unwrap_or_default().to_string();
    load(st, &model)?;
    let request: mold_core::GenerateRequest = serde_json::from_value(req).context("the request")?;
    let engine = st.engine.as_mut().unwrap();
    let t0 = std::time::Instant::now();
    let last = Arc::new(Mutex::new(0usize));
    let seen = last.clone();
    engine.set_on_progress(Box::new(move |ev| {
        if let ProgressEvent::DenoiseStep { step, total, .. } = ev {
            *seen.lock().unwrap() = step;
            emit(json!({"event": "step", "step": step, "total": total}));
        }
    }));
    let resp = engine.generate(&request);
    engine.clear_on_progress();
    let resp = resp?;
    for img in &resp.images {
        emit(json!({"event": "image", "index": img.index, "width": img.width, "height": img.height,
                    "png": base64::engine::general_purpose::STANDARD.encode(&img.data)}));
    }
    if let Some(v) = &resp.video {
        let format = serde_json::to_value(v.format).ok().and_then(|f| f.as_str().map(str::to_string)).unwrap_or_default();
        emit(json!({"event": "video", "format": format, "width": v.width, "height": v.height, "frames": v.frames, "fps": v.fps,
                    "data": base64::engine::general_purpose::STANDARD.encode(&v.data)}));
    }
    emit(json!({"event": "done", "ms": t0.elapsed().as_millis() as u64, "seed": resp.seed_used}));
    Ok(())
}

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("tokio");
    let mut st = State { config: mold_core::Config::load_or_default(), engine: None, model: None };
    emit(json!({"event": "ready", "cuda": cfg!(feature = "cuda"), "version": env!("CARGO_PKG_VERSION")}));
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let r: Result<()> = (|| {
            let cmd: Value = serde_json::from_str(&line).context("not a JSON command")?;
            let model = cmd.get("model").and_then(Value::as_str).unwrap_or_default().to_string();
            match cmd.get("op").and_then(Value::as_str).unwrap_or_default() {
                "pull" => {
                    pull(&rt, &model)?;
                    emit(json!({"event": "pulled", "model": model}));
                }
                "load" => {
                    load(&mut st, &model)?;
                    let vram = st.engine.as_ref().and_then(|e| e.resident_vram_bytes());
                    emit(json!({"event": "loaded", "model": model, "vram_bytes": vram}));
                }
                "generate" => generate(&mut st, cmd.get("request").unwrap_or(&Value::Null))?,
                "unload" => {
                    if let Some(mut e) = st.engine.take() {
                        e.unload();
                    }
                    st.model = None;
                    emit(json!({"event": "unloaded"}));
                }
                "list" => emit(list(&mut st)),
                "quit" => std::process::exit(0),
                op => return Err(anyhow!("unknown op {op:?}")),
            }
            Ok(())
        })();
        if let Err(e) = r {
            emit(json!({"event": "error", "message": format!("{e:#}")}));
        }
    }
}
