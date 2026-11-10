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
//!   -> {"op":"accept_license","id":"qwen-research"}       <- {"event":"accepted","id":...}  (the user agreed, in the app)
//!   a generate request's edit_images (base64 PNG/JPEG): the first is edited, the rest are references
//!   -> {"op":"transcribe","model":"whisper-large-v3-turbo","audio":"<base64 file>","language"?,"translate"?,"device"?}
//!                                                         <- {"event":"segment","start":..,"end":..,"text":..} ...
//!                                                         <- {"event":"transcript","text":..,"language":..,"segments":[..],"duration":..}
//!   -> {"op":"speak","model":"parler-tts-mini-v1","text":..,"voice"?,"seed"?,"device"?}
//!      a chatterbox-* model (the onnx feature) also takes "voice_audio" (base64 clip of the voice), "language",
//!      "exaggeration", "temperature", "threads"; a named voice is a clip Parler made once (audio/voices/<name>.wav)
//!                                                         <- {"event":"part","done":1,"total":3} ...
//!                                                         <- {"event":"speech","format":"wav","sample_rate":..,"seconds":..,"data":"<base64>"}
//!   -> {"op":"unload_audio"}                              <- {"event":"unloaded"}  (Whisper and Parler only)
//!   -> {"op":"remove_plan","model":...}  <- {"event":"removal","model","files":[{path,bytes}],"kept":[{path,used_by}],"bytes"}
//!   -> {"op":"remove","model":...}       <- {"event":"removed","model","freed_bytes","removed":[...],"warnings":[...]}
//!   any failure                                           <- {"event":"error","message":...}
//!
//! Models and their files come from mold's catalog, stored under $MOLD_HOME (am2Strata sets <Strata-data>/media).

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine as _;
use mold_inference::engine::{InferenceEngine, LoadStrategy};
use mold_inference::progress::ProgressEvent;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

#[path = "../audio.rs"]
mod audio;
#[path = "../parler_hf.rs"]
mod parler_hf;
#[path = "../acestep/mod.rs"]
mod acestep;
#[cfg(feature = "onnx")]
#[path = "../chatterbox.rs"]
mod chatterbox;

fn emit(v: Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{v}");
    let _ = out.flush();
}

struct State {
    config: mold_core::Config,
    engine: Option<Box<dyn InferenceEngine>>,
    model: Option<String>,
    /// the device placement the loaded model was loaded with (a request's "placement"; None: mold's own choice)
    placement: Option<Value>,
    whisper: Option<audio::Whisper>,
    parler: Option<audio::Parler>,
    #[cfg(feature = "onnx")]
    chatterbox: Option<chatterbox::Chatterbox>,
    song: Option<acestep::pipeline::AceStep>,
}

impl State {
    /// Frees the speech models (Whisper, Parler, Chatterbox).
    fn drop_audio(&mut self) {
        self.whisper = None;
        self.parler = None;
        #[cfg(feature = "onnx")]
        {
            self.chatterbox = None;
        }
    }
}

/// What removing `model` deletes (its own files) and keeps (files another model also uses).
fn removal(st: &mut State, model: &str) -> Result<Value> {
    if model == acestep::NAME {
        let dir = acestep::folder();
        if !dir.exists() {
            return Err(anyhow!("{model} is not downloaded"));
        }
        let bytes = audio::disk_bytes(&dir);
        return Ok(json!({"event": "removal", "model": model, "files": [{"path": dir, "bytes": bytes}], "kept": [], "bytes": bytes}));
    }
    if let Some(e) = audio::entry(model) {
        let dir = audio::folder(e);
        if !dir.exists() {
            return Err(anyhow!("{model} is not downloaded"));
        }
        let bytes = audio::disk_bytes(&dir);
        return Ok(json!({"event": "removal", "model": model, "files": [{"path": dir, "bytes": bytes}], "kept": [], "bytes": bytes}));
    }
    st.config = mold_core::Config::load_or_default();
    let canonical = mold_core::manifest::resolve_model_name(model);
    let plan = mold_core::removal::plan_removal(&st.config, &canonical);
    if plan.unique_files.is_empty() && plan.shared_files.is_empty() {
        return Err(anyhow!("{model} is not downloaded"));
    }
    let bytes: u64 = plan.unique_files.iter().map(|(_, b)| b).sum();
    Ok(json!({"event": "removal", "model": canonical,
              "files": plan.unique_files.iter().map(|(p, b)| json!({"path": p, "bytes": b})).collect::<Vec<_>>(),
              "kept": plan.shared_files.iter().map(|(p, by)| json!({"path": p, "used_by": by})).collect::<Vec<_>>(),
              "bytes": bytes}))
}

fn transcribe(st: &mut State, cmd: &Value) -> Result<Value> {
    let name = cmd["model"].as_str().unwrap_or("whisper-large-v3-turbo");
    let e = audio::entry(name).filter(|e| e.task == "stt").ok_or_else(|| anyhow!("{name} is not a speech-to-text model"))?;
    if st.whisper.as_ref().is_none_or(|w| w.name != name) {
        st.whisper = None;
        st.whisper = Some(audio::Whisper::load(e, cmd)?);
    }
    let bytes = base64::engine::general_purpose::STANDARD.decode(cmd["audio"].as_str().unwrap_or("")).context("audio: base64")?;
    let (pcm, rate, warnings) = audio::decode(bytes)?;
    let pcm = audio::resample(&pcm, rate, 16000);
    let t0 = std::time::Instant::now();
    let mut r = st.whisper.as_mut().unwrap().run(&pcm, cmd["language"].as_str().filter(|l| !l.is_empty()),
        cmd["translate"].as_bool().unwrap_or(false), &mut |seg| {
            let mut ev = seg.clone();
            ev["event"] = json!("segment");
            emit(ev);
        })?;
    r["event"] = json!("transcript");
    r["warnings"] = json!(warnings);
    r["ms"] = json!(t0.elapsed().as_millis() as u64);
    r["model"] = json!(name);
    Ok(r)
}

fn speak(st: &mut State, cmd: &Value) -> Result<Value> {
    let name = cmd["model"].as_str().unwrap_or("parler-tts-mini-v1");
    let e = audio::entry(name).filter(|e| e.task == "tts").ok_or_else(|| anyhow!("{name} is not a text-to-speech model"))?;
    if e.family == "chatterbox" {
        return speak_chatterbox(st, e, cmd);
    }
    if st.parler.as_ref().is_none_or(|p| p.name != name) {
        st.parler = None;
        st.parler = Some(audio::Parler::load(e, cmd)?);
    }
    let text = cmd["text"].as_str().filter(|t| !t.trim().is_empty()).ok_or_else(|| anyhow!("no text to speak"))?;
    let voice = cmd["voice"].as_str().unwrap_or("laura");
    let description = audio::VOICES.iter().find(|v| v.0 == voice.to_lowercase()).map(|v| v.1.to_string())
        .unwrap_or_else(|| voice.to_string());
    let t0 = std::time::Instant::now();
    let (pcm, rate) = st.parler.as_mut().unwrap().speak(text, &description, cmd["seed"].as_u64().unwrap_or(0),
        cmd["temperature"].as_f64().unwrap_or(0.7), // a round trip through Whisper: 2% WER at 0.7, 25% at 1.0
        &mut |done, total| emit(json!({"event": "part", "done": done, "total": total})))?;
    Ok(json!({"event": "speech", "format": "wav", "sample_rate": rate, "seconds": (pcm.len() as f64 / rate as f64 * 100.0).round() / 100.0,
              "ms": t0.elapsed().as_millis() as u64, "model": name,
              "data": base64::engine::general_purpose::STANDARD.encode(audio::wav(&pcm, rate))}))
}

/// The clip a named voice is cloned from: made once with Parler from the voice's description (seed 3), kept in
/// audio/voices/<name>.wav, so "laura" sounds the same in every model that clones a voice.
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
fn voice_clip(st: &mut State, voice: &str, rate: u32, language: Option<&str>) -> Result<Vec<f32>> {
    let voice = voice.to_lowercase();
    // another language than English: the native voice of the named voice's gender, made once by a multilingual Parler
    // model speaking that language (a synthetic voice), so the language is not spoken with an English accent
    if let (Some(lang), Some((_, gender))) = (language.filter(|l| *l != "en"), audio::VOICE_GENDER.iter().find(|v| v.0 == voice)) {
        let path = audio::voices_dir().join(format!("{lang}-{gender}.wav"));
        if !path.exists() {
            if let (Some(maker), Some((_, line)), Some((_, desc))) = (audio::native_maker(lang), audio::NATIVE_LINES.iter().find(|l| l.0 == lang),
                                                                       audio::NATIVE_VOICES.iter().find(|v| v.0 == *gender)) {
                if st.parler.as_ref().is_none_or(|p| p.name != maker.name) {
                    st.parler = None;
                    st.parler = Some(audio::Parler::load(maker, &json!({"device": "cpu"}))?);
                }
                let (pcm, r) = st.parler.as_mut().unwrap().speak(line, desc, 3, 0.7, &mut |_, _| {})?;
                std::fs::create_dir_all(audio::voices_dir())?;
                std::fs::write(&path, audio::wav(&pcm, r))?;
            }
        }
        if path.exists() {
            let (pcm, r, _) = audio::decode(std::fs::read(&path)?)?;
            return Ok(audio::resample(&pcm, r, rate));
        }
        // no native voice for this language yet: the English clip of that gender
    }
    let voice = match voice.as_str() { "female" => "laura".to_string(), "male" => "jon".to_string(), _ => voice };
    let (_, description) = audio::VOICES.iter().find(|v| v.0 == voice)
        .ok_or_else(|| anyhow!("no voice {voice:?}: give one of {} or a clip of the voice (voice_audio)",
                               audio::VOICES.iter().map(|v| v.0).collect::<Vec<_>>().join(", ")))?;
    let path = audio::voices_dir().join(format!("{voice}.wav"));
    if !path.exists() {
        let parler = audio::entry("parler-tts-mini-v1").unwrap();
        if !audio::downloaded(parler) {
            bail!("the voice {voice:?} is made with Parler-TTS mini v1 the first time: download it, or give a clip of the voice (voice_audio)");
        }
        if st.parler.as_ref().is_none_or(|p| p.name != parler.name) {
            st.parler = None;
            st.parler = Some(audio::Parler::load(parler, &json!({"device": "cpu"}))?);
        }
        let line = "Hello, and welcome. This is a short sample of my natural speaking voice, recorded calmly and clearly for you today.";
        let (pcm, r) = st.parler.as_mut().unwrap().speak(line, description, 3, 0.7, &mut |_, _| {})?;
        std::fs::create_dir_all(audio::voices_dir())?;
        std::fs::write(&path, audio::wav(&pcm, r))?;
    }
    let (pcm, r, _) = audio::decode(std::fs::read(&path)?)?;
    Ok(audio::resample(&pcm, r, rate))
}

#[cfg(feature = "onnx")]
fn speak_chatterbox(st: &mut State, e: &'static audio::Entry, cmd: &Value) -> Result<Value> {
    let text = cmd["text"].as_str().filter(|t| !t.trim().is_empty()).ok_or_else(|| anyhow!("no text to speak"))?.to_string();
    let clip = match cmd["voice_audio"].as_str().filter(|b| !b.is_empty()) {
        Some(b64) => {
            let bytes = base64::engine::general_purpose::STANDARD.decode(b64).context("voice_audio: base64")?;
            let (pcm, r, _) = audio::decode(bytes)?;
            audio::resample(&pcm, r, chatterbox::RATE)
        }
        None => voice_clip(st, cmd["voice"].as_str().unwrap_or("laura"), chatterbox::RATE, cmd["language"].as_str())?,
    };
    if st.chatterbox.as_ref().is_none_or(|c| c.name != e.name) {
        st.chatterbox = None;
        st.chatterbox = Some(chatterbox::Chatterbox::load(e, cmd["threads"].as_u64().unwrap_or(0) as usize)?);
    }
    let mut s = chatterbox::Sampling { seed: cmd["seed"].as_u64().unwrap_or(0), ..Default::default() };
    if let Some(t) = cmd["temperature"].as_f64() {
        s.temperature = t as f32;
    }
    // the take check: Whisper (the model the request names, on the CPU) hears each part, and a part that does not
    // match its text is made again (see Chatterbox::speak)
    let language = cmd["language"].as_str().filter(|l| !l.is_empty());
    let mut whisper = None;
    if let Some(e) = cmd["check_model"].as_str().and_then(audio::entry).filter(|e| e.task == "stt" && audio::downloaded(e)) {
        whisper = match st.whisper.take() {
            Some(w) if w.name == e.name => Some(w),
            _ => Some(audio::Whisper::load(e, &json!({"device": "cpu"}))?),
        };
    }
    let mut takes = 0usize;
    let mut check = |said: &str, pcm: &[f32]| -> Option<f32> {
        let w = whisper.as_mut()?;
        takes += 1;
        let heard = w.run(&audio::resample(pcm, chatterbox::RATE, 16000), language, false, &mut |_| {}).ok()?;
        Some(audio::words_score(said, heard["text"].as_str().unwrap_or("")))
    };
    let t0 = std::time::Instant::now();
    let pcm = st.chatterbox.as_mut().unwrap().speak(&text, &clip, language,
        cmd["exaggeration"].as_f64().unwrap_or(0.5) as f32, &s,
        &mut |done, total| emit(json!({"event": "part", "done": done, "total": total})), &mut check)?;
    st.whisper = whisper.or(st.whisper.take());
    let rate = chatterbox::RATE;
    Ok(json!({"event": "speech", "format": "wav", "sample_rate": rate, "seconds": (pcm.len() as f64 / rate as f64 * 100.0).round() / 100.0,
              "ms": t0.elapsed().as_millis() as u64, "model": e.name, "checked_takes": takes,
              "data": base64::engine::general_purpose::STANDARD.encode(audio::wav(&pcm, rate))}))
}

#[cfg(not(feature = "onnx"))]
fn speak_chatterbox(_: &mut State, e: &'static audio::Entry, _: &Value) -> Result<Value> {
    bail!("{} runs on ONNX Runtime: this media engine was built without it (build with --features onnx)", e.name)
}

/// Models am2Strata adds to mold's catalog: another transformer on a family mold runs, with that family's shared
/// files (text encoder, VAE, tokenizer) from the mold manifest named, and the licence those weights carry.
struct Extra {
    name: &'static str,
    base: &'static str,
    repo: &'static str,
    file: &'static str,
    size: u64,
    sha256: &'static str,
    description: &'static str,
    steps: u32,
    guidance: f64,
    negative: &'static str,
    license: &'static mold_core::license_acceptance::ThirdPartyLicense,
}

/// ChatR Labs' Omni LoRA merged into Qwen-Image 2.1 at strength 1.0 (text-to-image and editing). Their model card's
/// settings: 35 steps, true CFG 5.0 with a single-space negative prompt (an empty one turns CFG off).
const EXTRAS: &[Extra] = &[
    Extra { name: "chatr-omni-qwen-image-2.1:q8", base: "qwen-image-2.1:q8", repo: "chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized",
            file: "qwen-image-2.1-step90000-Q8_0.gguf", size: 7_687_183_136,
            sha256: "58ab4d7baaa58ad7bdda18e6aafc22eebce1f494932d08c34667b88ab4fee5ca",
            description: "ChatR Labs Uncensored Omni on Qwen Image 2.1, Q8_0: text to image and editing (Qwen Research License, non-commercial)",
            steps: 35, guidance: 5.0, negative: " ", license: &mold_core::license_acceptance::QWEN_RESEARCH },
    Extra { name: "chatr-omni-qwen-image-2.1:bf16", base: "qwen-image-2.1:q8", repo: "chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized",
            file: "qwen-image-2.1-step90000-BF16.gguf", size: 14_230_297_376,
            sha256: "027501fd6abb571a2415e9728c128e96acdf3271222ea4fbf69f9efad4f136d3",
            description: "ChatR Labs Uncensored Omni on Qwen Image 2.1, BF16: text to image and editing (Qwen Research License, non-commercial)",
            steps: 35, guidance: 5.0, negative: " ", license: &mold_core::license_acceptance::QWEN_RESEARCH },
];

fn extra(model: &str) -> Option<&'static Extra> {
    EXTRAS.iter().find(|e| e.name == model)
}

/// The extra's manifest: its base's, with the transformer swapped.
fn extra_manifest(e: &Extra) -> Option<mold_core::manifest::ModelManifest> {
    use mold_core::manifest::{ModelComponent, ModelFile};
    let mut m = mold_core::manifest::find_manifest(e.base)?.clone();
    m.files.retain(|f| !matches!(f.component, ModelComponent::Transformer | ModelComponent::TransformerShard));
    m.files.insert(0, ModelFile { hf_repo: e.repo.to_string(), hf_filename: e.file.to_string(), component: ModelComponent::Transformer,
                                  size_bytes: e.size, gated: false, sha256: Some(e.sha256) });
    m.name = e.name.to_string();
    m.description = e.description.to_string();
    m.defaults.steps = e.steps;
    m.defaults.guidance = e.guidance;
    m.defaults.negative_prompt = Some(e.negative.to_string());
    Some(m)
}

fn mold_home() -> Option<std::path::PathBuf> {
    mold_core::Config::mold_dir()
}

fn pull(rt: &tokio::runtime::Runtime, model: &str) -> Result<()> {
    if let Some(e) = audio::entry(model) {
        return audio::pull(e);
    }
    if model == acestep::NAME {
        return acestep::pull();
    }
    if let Some(e) = extra(model) {
        // mold gates its own Qwen Image 2.1 tiers on this licence by name; an extra on the same weights asks the same
        if !mold_home().is_some_and(|h| mold_core::license_acceptance::is_accepted(&h, e.license)) {
            return Err(anyhow!("{model} needs the {} accepted first ({}): {}", e.license.name, e.license.canonical, e.license.summary));
        }
        let m = extra_manifest(e).ok_or_else(|| anyhow!("mold's catalog has no {}", e.base))?;
        let paths = rt.block_on(mold_core::download::pull_model(&m, &mold_core::download::PullOptions::default()))
            .map_err(|e| anyhow!("{e}"))?;
        // the paths go into mold's config, where ModelPaths::resolve and the engine factory (by family) find them
        let mut mc = m.to_model_config(&paths);
        mc.family = Some(m.family.clone());
        mc.default_steps = Some(e.steps);
        mc.default_guidance = Some(e.guidance);
        mc.negative_prompt = Some(e.negative.to_string());
        mc.description = Some(e.description.to_string());
        let mut config = mold_core::Config::load_or_default();
        config.upsert_model(m.name.clone(), mc);
        config.save().context("saving mold's config")?;
        return Ok(());
    }
    let manifest = mold_core::manifest::find_manifest(model).ok_or_else(|| anyhow!("no model {model:?} in the catalog"))?;
    rt.block_on(mold_core::download::pull_model(manifest, &mold_core::download::PullOptions::default()))
        .map_err(|e| anyhow!("{e}"))?;
    Ok(())
}

/// The catalog: every generation model mold knows, with its kind (image or video) and whether it is downloaded.
fn list(st: &mut State) -> Value {
    st.config = mold_core::Config::load_or_default();
    let extras: Vec<mold_core::manifest::ModelManifest> = EXTRAS.iter().filter_map(extra_manifest).collect();
    let home = mold_home();
    let mut all: Vec<&mold_core::manifest::ModelManifest> = mold_core::manifest::visible_manifests().filter(|m| m.is_generation_model()).collect();
    all.extend(extras.iter());
    let models: Vec<Value> = all.into_iter()
        .map(|m| {
            let f = m.family.to_lowercase();
            let kind = if f.starts_with("hunyuan3d") {
                "3d"
            } else if f.starts_with("wan") || f.starts_with("ltx") || f.contains("video") || f.starts_with("minimax-h3") {
                "video"
            } else {
                "image"
            };
            // the licence mold gates the download on (accepted here, not on Hugging Face)
            let license = extra(&m.name).map(|e| e.license)
                .or_else(|| mold_core::license_acceptance::licenses_for_manifest(m).into_iter().next());
            let mut v = json!({"name": m.name, "display_name": m.display_name(), "family": m.family, "kind": kind,
                   "description": m.description, "size_gb": (m.total_size_bytes() as f64 / 1e8).round() / 10.0, "gated": m.is_gated(),
                   "downloaded": mold_core::ModelPaths::resolve(&m.name, &st.config).is_some(),
                   "loaded": st.model.as_deref() == Some(m.name.as_str()),
                   // takes edit images: the first is edited, the rest are references
                   "edits": m.family == "qwen-image21" || m.family.contains("edit") || m.name.contains("edit"),
                   "defaults": {"steps": m.defaults.steps, "guidance": m.defaults.guidance, "width": m.defaults.width,
                                "height": m.defaults.height, "frames": m.defaults.frames, "fps": m.defaults.fps,
                                "negative_prompt": m.defaults.negative_prompt}});
            if let Some(l) = license {
                v["license"] = json!({"id": l.id, "name": l.name, "url": l.canonical, "summary": l.summary,
                                      "accepted": home.as_deref().is_some_and(|h| mold_core::license_acceptance::is_accepted(h, l))});
            }
            v
        })
        .collect();
    let mut models = models;
    models.extend(audio::list());
    models.push(acestep::entry());
    json!({"event": "models", "models": models})
}

/// A song (ACE-Step 1.5 in Rust): {"op":"song","caption","lyrics","language","duration","seed","bpm","key","time_signature",
/// "steps","device","planner" (default true),"planner_temperature","planner_cfg","planner_top_p"} -> {"event":"step",...}... then {"event":"song","format":"wav","data":<base64>,"seconds",...}.
fn song(st: &mut State, cmd: &Value) -> Result<Value> {
    let dir = acestep::local().ok_or_else(|| anyhow!("{} is not downloaded (pull it first)", acestep::NAME))?;
    if st.song.is_none() {
        // the image or video model leaves the card first
        if let Some(mut e) = st.engine.take() {
            e.unload();
        }
        st.model = None;
        let device = match cmd["device"].as_str() {
            Some("cpu") => candle_core::Device::Cpu,
            _ if cfg!(feature = "cuda") => candle_core::Device::new_cuda(cmd["gpu"].as_u64().unwrap_or(0) as usize)?,
            _ => candle_core::Device::Cpu,
        };
        st.song = Some(acestep::pipeline::AceStep::load(&dir, &device)?);
    }
    let text = |k: &str| cmd[k].as_str().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
    let duration = (cmd["duration"].as_u64().unwrap_or(60) as u32).clamp(acestep::pipeline::MIN_SECONDS, acestep::pipeline::MAX_SECONDS);
    let s = acestep::pipeline::Song {
        caption: text("caption").ok_or_else(|| anyhow!("caption: describe the song (genre, instruments, voice, mood)"))?,
        lyrics: text("lyrics").unwrap_or_else(|| "[Instrumental]".into()),
        language: text("language").unwrap_or_else(|| "en".into()),
        metas: acestep::text::Metas { bpm: cmd["bpm"].as_u64().map(|b| b as u32), time_signature: text("time_signature"),
                                      key_scale: text("key"), duration_s: duration },
        seed: cmd["seed"].as_u64().unwrap_or(0),
        steps: (cmd["steps"].as_u64().unwrap_or(8) as usize).clamp(1, 20),
        shift: cmd["shift"].as_f64().unwrap_or(1.0) as f32,
        // the song planner unless asked not to ("planner": false), when its files are downloaded
        plan: (cmd["planner"].as_bool() != Some(false)).then(|| acestep::planner::Sampling {
            temperature: cmd["planner_temperature"].as_f64().unwrap_or(0.85) as f32,
            cfg_scale: cmd["planner_cfg"].as_f64().unwrap_or(2.0) as f32,
            top_p: cmd["planner_top_p"].as_f64().unwrap_or(0.9) as f32,
            seed: cmd["seed"].as_u64().unwrap_or(0),
        }),
    };
    let planned = s.plan.is_some() && st.song.as_ref().unwrap().has_planner();
    let t0 = std::time::Instant::now();
    let audio = st.song.as_mut().unwrap().song(&s, &mut |step, total| emit(json!({"event": "step", "step": step, "total": total})))?;
    let wav = acestep::pipeline::wav(&audio)?;
    Ok(json!({"event": "song", "format": "wav", "sample_rate": acestep::oobleck::SAMPLE_RATE, "seconds": duration,
              "ms": t0.elapsed().as_millis() as u64, "model": acestep::NAME, "seed": s.seed, "planner": planned,
              // what the planner's reasoning wrote for a song asked without them
              "planned": st.song.as_ref().unwrap().last_thought().map(|t| json!({"bpm": t.bpm, "key": t.key_scale,
                                                                                 "time_signature": t.time_signature})),
              "data": base64::engine::general_purpose::STANDARD.encode(wav)}))
}

fn load(st: &mut State, model: &str) -> Result<()> {
    load_placed(st, model, None)
}

/// Loads `model`, with `request`'s placement when it has one (its components on the devices it names); a loaded model
/// placed otherwise is loaded again.
fn load_placed(st: &mut State, model: &str, request: Option<&mold_core::GenerateRequest>) -> Result<()> {
    // a song model on the card goes first
    st.song = None;
    let placement = request.and_then(|r| r.placement.as_ref()).map(|p| serde_json::to_value(p).unwrap_or(Value::Null));
    if st.model.as_deref() == Some(model) && st.placement == placement && st.engine.as_ref().is_some_and(|e| e.is_loaded()) {
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
    match request.filter(|r| r.placement.is_some()) {
        Some(r) => engine.load_for_request(r),
        None => engine.load(),
    }.with_context(|| format!("loading {model}"))?;
    st.engine = Some(engine);
    st.model = Some(model.to_string());
    st.placement = placement;
    Ok(())
}

fn generate(st: &mut State, req: &Value) -> Result<()> {
    let mut req = req.clone();
    if req.get("model").is_none() {
        req["model"] = json!(st.model.clone().ok_or_else(|| anyhow!("no model loaded"))?);
    }
    let model = req["model"].as_str().unwrap_or_default().to_string();
    // what the request leaves out comes from the model: its config entry (an extra), else its manifest
    let mc = st.config.lookup_model_config(&model);
    let man = mold_core::manifest::find_manifest(&model).cloned().or_else(|| extra(&model).and_then(extra_manifest));
    if req.get("steps").is_none_or(Value::is_null) {
        req["steps"] = json!(mc.as_ref().and_then(|c| c.default_steps).or(man.as_ref().map(|m| m.defaults.steps)).unwrap_or(20));
    }
    if req.get("guidance").is_none_or(Value::is_null) {
        if let Some(g) = mc.as_ref().and_then(|c| c.default_guidance).or(man.as_ref().map(|m| m.defaults.guidance)) {
            req["guidance"] = json!(g);
        }
    }
    if req.get("negative_prompt").is_none_or(Value::is_null) {
        if let Some(n) = mc.as_ref().and_then(|c| c.negative_prompt.clone()).or(man.and_then(|m| m.defaults.negative_prompt)) {
            req["negative_prompt"] = json!(n);
        }
    }
    let request: mold_core::GenerateRequest = serde_json::from_value(req).context("the request")?;
    load_placed(st, &model, Some(&request))?;
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

/// A long video as a chain of clips (mold's chain: each clip continues from the one before, with its motion tail as
/// the next clip's conditioning on a model that takes a source image, then stitched and encoded as one MP4):
/// {"op": "chain", "request": mold's ChainRequest (model, prompt + total_frames + clip_frames, or stages; width,
/// height, fps, steps, guidance, seed, motion_tail_frames)} -> {"event": "step", "step", "total", "stage", "stages"}...
/// then {"event": "video", ...} as a one-clip generate gives, and {"event": "done"}.
fn chain(st: &mut State, req: &Value) -> Result<()> {
    let model = req["model"].as_str().map(str::to_string).or_else(|| st.model.clone()).ok_or_else(|| anyhow!("no model named"))?;
    let mut r = req.clone();
    r["model"] = json!(model);
    for (k, v) in [("strength", json!(1.0)), ("output_format", json!("mp4")), ("steps", json!(20)), ("guidance", json!(5.0))] {
        if r.get(k).is_none_or(Value::is_null) {
            r[k] = v;
        }
    }
    let family = mold_core::manifest::find_manifest(&model).map(|m| m.family.clone()).unwrap_or_else(|| "wan".into());
    let request: mold_core::ChainRequest = serde_json::from_value(r).context("the chain request")?;
    let request = request.normalise_with_family(Some(&family)).map_err(|e| anyhow!("{e}"))?;
    load_placed(st, &model, None)?;
    let engine = st.engine.as_mut().unwrap();
    let t0 = std::time::Instant::now();
    let renderer = engine.as_chain_renderer().ok_or_else(|| anyhow!("{model} cannot chain clips"))?;
    let stages = request.stages.len() as u64;
    let mut progress = |ev: mold_core::ChainProgressEvent| {
        if let mold_core::ChainProgressEvent::DenoiseStep { stage_idx, step, total } = ev {
            emit(json!({"event": "step", "step": stage_idx as u64 * total as u64 + step as u64, "total": stages * total as u64,
                        "stage": stage_idx + 1, "stages": stages}));
        }
    };
    let out = mold_inference::chain::ChainOrchestrator::new(renderer).run(&request, Some(&mut progress))
        .map_err(|e| anyhow!("the chain: {e:?}"))?;
    let n = out.stage_frames.len();
    let boundaries: Vec<mold_core::TransitionMode> = request.stages.iter().skip(1).map(|s| s.transition).collect();
    let fade_lens: Vec<u32> = request.stages.iter().skip(1).map(|s| s.fade_frames.unwrap_or(8)).collect();
    let plan = mold_inference::chain::stitch::StitchPlan { clips: out.stage_frames, boundaries, fade_lens,
                                                           motion_tail_frames: request.motion_tail_frames };
    let frames = plan.assemble().map_err(|e| anyhow!("stitching the clips: {e:?}"))?;
    let (w, h) = frames.first().map(|f| (f.width(), f.height())).unwrap_or((request.width, request.height));
    let enc = mold_inference::chain::encode::encode_chain_frames(&frames, request.fps, request.output_format, None)?;
    let format = serde_json::to_value(enc.format).ok().and_then(|f| f.as_str().map(str::to_string)).unwrap_or_default();
    emit(json!({"event": "video", "format": format, "width": w, "height": h, "frames": frames.len(), "fps": request.fps,
                "clips": n, "data": base64::engine::general_purpose::STANDARD.encode(&enc.bytes)}));
    emit(json!({"event": "done", "ms": t0.elapsed().as_millis() as u64, "seed": request.seed}));
    Ok(())
}

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("tokio");
    let mut st = State { config: mold_core::Config::load_or_default(), engine: None, model: None, placement: None, whisper: None, parler: None,
                         #[cfg(feature = "onnx")]
                         chatterbox: None, song: None };
    // the video formats this build can encode (mold: mp4 with its mp4 feature, webp with its webp feature)
    emit(json!({"event": "ready", "cuda": cfg!(feature = "cuda"), "version": env!("CARGO_PKG_VERSION"),
                "video_formats": ["mp4", "apng", "gif"]}));
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
                "chain" => chain(&mut st, cmd.get("request").unwrap_or(&Value::Null))?,
                "unload" => {
                    if let Some(mut e) = st.engine.take() {
                        e.unload();
                    }
                    st.model = None;
                    st.drop_audio();
                    st.song = None;
                    emit(json!({"event": "unloaded"}));
                }
                "list" => emit(list(&mut st)),
                "accept_license" => {
                    let id = cmd["id"].as_str().unwrap_or_default();
                    let l = mold_core::license_acceptance::license_by_id(id).ok_or_else(|| anyhow!("no licence {id:?}"))?;
                    let home = mold_home().ok_or_else(|| anyhow!("no MOLD_HOME"))?;
                    mold_core::license_acceptance::record_acceptance(&home, l).context("recording the acceptance")?;
                    emit(json!({"event": "accepted", "id": l.id}));
                }
                "unload_audio" => {
                    st.drop_audio();
                    emit(json!({"event": "unloaded"}));
                }
                "transcribe" => {
                    let r = transcribe(&mut st, &cmd)?;
                    emit(r)
                }
                "speak" => {
                    let r = speak(&mut st, &cmd)?;
                    emit(r)
                }
                "song" => {
                    let r = song(&mut st, &cmd)?;
                    emit(r)
                }
                "remove_plan" => emit(removal(&mut st, &model)?),
                "remove" => {
                    // out of memory first: the engine (or the speech models) may hold it
                    if st.model.as_deref() == Some(model.as_str()) {
                        if let Some(mut e) = st.engine.take() {
                            e.unload();
                        }
                        st.model = None;
                    }
                    if model == acestep::NAME {
                        st.song = None;
                        let dir = acestep::folder();
                        let bytes = audio::disk_bytes(&dir);
                        std::fs::remove_dir_all(&dir).with_context(|| format!("removing {}", dir.display()))?;
                        emit(json!({"event": "removed", "model": model, "bytes": bytes}));
                        return Ok(());
                    }
                    if let Some(e) = audio::entry(&model) {
                        st.drop_audio();
                        let dir = audio::folder(e);
                        let freed = audio::disk_bytes(&dir);
                        std::fs::remove_dir_all(&dir).with_context(|| format!("removing {}", dir.display()))?;
                        emit(json!({"event": "removed", "model": model, "freed_bytes": freed, "removed": [dir], "warnings": []}));
                    } else {
                        st.config = mold_core::Config::load_or_default();
                        let canonical = mold_core::manifest::resolve_model_name(&model);
                        let plan = mold_core::removal::plan_removal(&st.config, &canonical);
                        if plan.unique_files.is_empty() && plan.shared_files.is_empty() {
                            return Err(anyhow!("{model} is not downloaded"));
                        }
                        let out = mold_core::removal::execute_removal(&st.config, &plan);
                        mold_core::download::remove_pulling_marker(&canonical);
                        if st.config.models.contains_key(&canonical) {
                            st.config.remove_model(&canonical);
                            let _ = st.config.save();
                        }
                        emit(json!({"event": "removed", "model": canonical, "freed_bytes": out.freed_bytes,
                                    "removed": out.removed, "warnings": out.warnings}));
                    }
                }
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
