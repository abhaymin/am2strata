//! Chatterbox text to speech (Resemble AI, MIT) through ONNX Runtime: the `onnx` feature.
//!
//! The ONNX exports (ResembleAI/chatterbox-turbo-ONNX, onnx-community/chatterbox-multilingual-ONNX) are four graphs:
//! - `speech_encoder`: a 24 kHz reference clip -> its conditioning embeddings, prompt speech tokens, speaker x-vector
//!   and mel features;
//! - `embed_tokens`: text or speech token ids -> embeddings;
//! - `language_model`: embeddings plus a KV cache -> next-token logits (speech tokens);
//! - `conditional_decoder`: speech tokens with the speaker's x-vector and features -> a 24 kHz waveform.
//!
//! This file is the rest: the text preparation and tokenizer, the sampling loop with its cache, and the clip in and
//! audio out, as the model cards' reference code and chatterbox's own `tts_turbo.py` / `mtl_tts.py` do it.
//!
//! The runtime is loaded when the first model loads (ort's `load-dynamic`): `ORT_DYLIB_PATH`, else the system's
//! `libonnxruntime`. Without one, the model fails to load and says how to get one.

use anyhow::{anyhow, bail, Context, Result};
use ort::session::Session;
use ort::value::{DynValue, Tensor};
use std::collections::HashMap;
use tokenizers::Tokenizer;

use crate::audio::{self, Entry};

pub const RATE: u32 = 24000;
const START_SPEECH: i64 = 6561;
const STOP_SPEECH: i64 = 6562;
/// Turbo's decoder ends on three of these, so the last word is not cut off
const SILENCE: i64 = 4299;
const KV_HEADS: usize = 16;
const HEAD_DIM: usize = 64;
/// the decoder's reference: at most 10 s of the clip (chatterbox's DEC_COND_LEN)
const REF_SECONDS: usize = 10;

/// How the next speech token is drawn. Greedy at temperature 0; else chatterbox's defaults.
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling { temperature: 0.8, top_p: 0.95, min_p: 0.0, top_k: 1000, repetition_penalty: 1.2, seed: 0 }
    }
}

/// One voice, as the speech encoder read its clip: kept so later requests in that voice skip the encoder.
struct Voice {
    cond: (Vec<i64>, Vec<f32>),
    prompt: Vec<i64>,
    x_vector: Vec<f32>,
    features: (Vec<i64>, Vec<f32>),
}

pub struct Chatterbox {
    pub name: String,
    multilingual: bool,
    layers: usize,
    encoder: Session,
    embed: Session,
    lm: Session,
    decoder: Session,
    tok: Tokenizer,
    voices: HashMap<u64, Voice>,
}

static RUNTIME: std::sync::OnceLock<std::result::Result<(), String>> = std::sync::OnceLock::new();

/// ONNX Runtime, loaded once: ORT_DYLIB_PATH, else the usual library names (the system's package, or one beside the
/// engine).
fn runtime() -> Result<()> {
    RUNTIME.get_or_init(|| {
        let candidates: Vec<std::path::PathBuf> = match std::env::var_os("ORT_DYLIB_PATH") {
            Some(p) => vec![p.into()],
            None => {
                let mut v = Vec::new();
                if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.to_path_buf())) {
                    v.push(dir.join(if cfg!(windows) { "onnxruntime.dll" } else if cfg!(target_os = "macos") { "libonnxruntime.dylib" } else { "libonnxruntime.so" }));
                }
                if cfg!(target_os = "linux") {
                    for d in ["/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/local/lib", "/usr/lib"] {
                        v.push(format!("{d}/libonnxruntime.so.1.23").into());
                        v.push(format!("{d}/libonnxruntime.so").into());
                    }
                }
                v
            }
        };
        let found = candidates.iter().find(|p| p.exists()).ok_or_else(|| {
            "ONNX Runtime 1.23 was not found: install it (Debian/Ubuntu: sudo apt install libonnxruntime1.23) or set ORT_DYLIB_PATH to libonnxruntime".to_string()
        })?;
        ort::init_from(found).map_err(|e| format!("{}: {e}", found.display()))?.with_name("am2strata-media").commit();
        Ok(())
    }).clone().map_err(|e| anyhow!(e))
}

fn session(path: &std::path::Path, threads: usize) -> Result<Session> {
    // basic optimizations only and no memory pattern: the full set folds constants into new copies of the weights while
    // loading, which about doubles the peak (6 GB and more for Turbo's 3.3 GB of fp32 weights)
    let mut b = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level1)?
        .with_memory_pattern(false)?;
    if threads > 0 {
        b = b.with_intra_threads(threads)?;
    }
    b.commit_from_file(path).with_context(|| format!("{}", path.display()))
}

fn tensor<T: ort::tensor::PrimitiveTensorElementType + Clone + std::fmt::Debug + 'static>(shape: Vec<i64>, data: Vec<T>) -> Result<DynValue> {
    Ok(Tensor::from_array((shape, data))?.into_dyn())
}

fn take<T: ort::tensor::PrimitiveTensorElementType + Clone + 'static>(v: &DynValue) -> Result<(Vec<i64>, Vec<T>)> {
    let (shape, data) = v.try_extract_tensor::<T>()?;
    Ok((shape.iter().copied().collect(), data.to_vec()))
}

/// chatterbox's `punc_norm`: a capital first letter, one space between words, uncommon punctuation replaced, and a
/// full stop when the text has no ending.
pub fn punc_norm(text: &str) -> String {
    let mut t: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.is_empty() {
        return "You need to add some text for me to talk.".into();
    }
    if let Some(c) = t.chars().next().filter(|c| c.is_lowercase()) {
        t = c.to_uppercase().collect::<String>() + &t[c.len_utf8()..];
    }
    for (a, b) in [("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "), ("—", "-"), ("–", "-"), (" ,", ","),
                   ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'")] {
        t = t.replace(a, b);
    }
    let t = t.trim_end().to_string();
    if [".", "!", "?", "-", ","].iter().any(|e| t.ends_with(e)) { t } else { t + "." }
}

/// Korean syllables decomposed into their letters (jamo), as the multilingual tokenizer was trained on.
fn korean(text: &str) -> String {
    text.chars().flat_map(|c| {
        let u = c as u32;
        if !(0xAC00..=0xD7AF).contains(&u) {
            return vec![c];
        }
        let b = u - 0xAC00;
        let mut v = vec![char::from_u32(0x1100 + b / (21 * 28)).unwrap(), char::from_u32(0x1161 + (b % (21 * 28)) / 28).unwrap()];
        if b % 28 > 0 {
            v.push(char::from_u32(0x11A7 + b % 28).unwrap());
        }
        v
    }).collect::<String>().trim().to_string()
}

/// Draws the next token from `logits` (one row), after the repetition penalty on `seen`.
fn next_token(logits: &mut [f32], seen: &[i64], s: &Sampling, rng: &mut u64) -> i64 {
    for &t in seen {
        if let Some(l) = logits.get_mut(t as usize) {
            *l = if *l < 0.0 { *l * s.repetition_penalty } else { *l / s.repetition_penalty };
        }
    }
    if s.temperature <= 0.0 {
        return logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as i64).unwrap_or(STOP_SPEECH);
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    idx.truncate(s.top_k.max(1));
    let top = logits[idx[0]];
    let mut p: Vec<f32> = idx.iter().map(|&i| ((logits[i] - top) / s.temperature).exp()).collect();
    let sum: f32 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= sum);
    // min-p: drop what is less likely than min_p times the likeliest; top-p: keep the smallest set covering top_p
    let floor = s.min_p * p[0];
    let mut keep = p.iter().take_while(|&&x| x >= floor).count().max(1);
    if s.top_p < 1.0 {
        let mut acc = 0.0;
        keep = keep.min(p.iter().position(|&x| { acc += x; acc >= s.top_p }).map(|i| i + 1).unwrap_or(p.len()));
    }
    let total: f32 = p[..keep].iter().sum();
    // splitmix64
    *rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *rng;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let r = ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32 * total;
    let mut acc = 0.0;
    for k in 0..keep {
        acc += p[k];
        if r < acc {
            return idx[k] as i64;
        }
    }
    idx[keep - 1] as i64
}

impl Chatterbox {
    /// `threads`: ONNX Runtime's threads per graph (0: its own choice, all cores).
    pub fn load(e: &Entry, threads: usize) -> Result<Chatterbox> {
        runtime()?;
        let files = audio::local_files(e).ok_or_else(|| anyhow!("{} is not downloaded (pull it first)", e.name))?;
        let file = |name: &str| files.iter().find(|p| p.ends_with(name)).cloned().ok_or_else(|| anyhow!("{name} missing"));
        let tok = Tokenizer::from_file(file("tokenizer.json")?).map_err(|e| anyhow!("{e}"))?;
        let lm = session(&file("onnx/language_model.onnx")?, threads)?;
        let layers = lm.inputs().iter().filter(|i| i.name().starts_with("past_key_values.") && i.name().ends_with(".key")).count();
        Ok(Chatterbox {
            name: e.name.to_string(),
            multilingual: e.name == "chatterbox-multilingual",
            layers,
            encoder: session(&file("onnx/speech_encoder.onnx")?, threads)?,
            embed: session(&file("onnx/embed_tokens.onnx")?, threads)?,
            lm,
            decoder: session(&file("onnx/conditional_decoder.onnx")?, threads)?,
            tok,
            voices: HashMap::new(),
        })
    }

    /// The speech encoder's reading of a clip (24 kHz mono), cached by the clip's contents.
    fn voice(&mut self, clip: &[f32]) -> Result<u64> {
        let clip = &clip[..clip.len().min(REF_SECONDS * RATE as usize)];
        let key = clip.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100_0000_01b3));
        if !self.voices.contains_key(&key) {
            if clip.len() < RATE as usize {
                bail!("the voice clip is too short: give 5 to 10 seconds of one person speaking");
            }
            let out = self.encoder.run(ort::inputs!["audio_values" => tensor(vec![1, clip.len() as i64], clip.to_vec())?])?;
            let v = Voice {
                cond: take::<f32>(&out["audio_features"])?,
                prompt: take::<i64>(&out["audio_tokens"])?.1,
                x_vector: take::<f32>(&out["speaker_embeddings"])?.1,
                features: take::<f32>(&out["speaker_features"])?,
            };
            if self.voices.len() >= 8 {
                self.voices.clear();
            }
            self.voices.insert(key, v);
        }
        Ok(key)
    }

    fn embed(&mut self, ids: &[i64], positions: &[i64], exaggeration: f32) -> Result<(Vec<i64>, Vec<f32>)> {
        let n = ids.len() as i64;
        let out = if self.multilingual {
            self.embed.run(ort::inputs![
                "input_ids" => tensor(vec![1, n], ids.to_vec())?,
                "position_ids" => tensor(vec![1, n], positions.to_vec())?,
                "exaggeration" => tensor(vec![1], vec![exaggeration])?,
            ])?
        } else {
            self.embed.run(ort::inputs!["input_ids" => tensor(vec![1, n], ids.to_vec())?])?
        };
        take::<f32>(&out["inputs_embeds"])
    }

    /// The speech tokens for one part of the text (no prompt, no start or stop token).
    fn tokens(&mut self, text: &str, key: u64, language: Option<&str>, exaggeration: f32, s: &Sampling, max_new: usize) -> Result<Vec<i64>> {
        let mut text = punc_norm(text);
        if self.multilingual {
            let lang = language.unwrap_or("en").to_lowercase();
            if !audio::CHATTERBOX_LANGUAGES.iter().any(|l| l.0 == lang) {
                bail!("language {lang:?} is not one this model speaks here; one of: {}",
                      audio::CHATTERBOX_LANGUAGES.iter().map(|l| l.0).collect::<Vec<_>>().join(", "));
            }
            if lang == "ko" {
                text = korean(&text);
            }
            text = format!("[{lang}]{text}");
        }
        let ids: Vec<i64> = self.tok.encode(text.as_str(), true).map_err(|e| anyhow!("{e}"))?.get_ids().iter().map(|&i| i as i64).collect();
        // the multilingual embedder takes the positions: text from -1 up, its speech tokens at 0 (the model card's rule)
        let positions: Vec<i64> = ids.iter().enumerate().map(|(i, &t)| if t >= START_SPEECH { 0 } else { i as i64 - 1 }).collect();
        let (eshape, text_emb) = self.embed(&ids, &positions, exaggeration)?;
        let v = &self.voices[&key];
        let (cshape, cond) = (v.cond.0.clone(), v.cond.1.clone());
        let width = *eshape.last().unwrap();
        let mut emb = cond;
        emb.extend(text_emb);
        let len = cshape[1] + eshape[1];
        let mut inputs = tensor(vec![1, len, width], emb)?;
        // an empty cache: from_array refuses a 0-sized dimension, the allocator's constructor does not
        let alloc = ort::memory::Allocator::default();
        let mut past: Vec<DynValue> = (0..self.layers * 2)
            .map(|_| Ok(Tensor::<f32>::new(&alloc, vec![1, KV_HEADS as i64, 0, HEAD_DIM as i64])?.into_dyn())).collect::<Result<_>>()?;
        let mut pos: Vec<i64> = (0..len).collect();
        let mut seen = vec![START_SPEECH];
        let mut rng = s.seed ^ 0x5EED;
        let mut total = len;
        for step in 0..max_new {
            let mut feed: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> = vec![
                ("inputs_embeds".into(), inputs.into()),
                ("attention_mask".into(), tensor(vec![1, total], vec![1i64; total as usize])?.into()),
            ];
            if !self.multilingual {
                feed.push(("position_ids".into(), tensor(vec![1, pos.len() as i64], pos.clone())?.into()));
            }
            for (l, kv) in past.drain(..).enumerate() {
                feed.push((format!("past_key_values.{}.{}", l / 2, if l % 2 == 0 { "key" } else { "value" }).into(), kv.into()));
            }
            let mut out = self.lm.run(feed)?;
            let (shape, logits) = out["logits"].try_extract_tensor::<f32>()?;
            let vocab = *shape.last().unwrap() as usize;
            let mut last = logits[logits.len() - vocab..].to_vec();
            for l in 0..self.layers {
                for kv in ["key", "value"] {
                    past.push(out.remove(format!("present.{l}.{kv}")).ok_or_else(|| anyhow!("no present.{l}.{kv}"))?);
                }
            }
            drop(out);
            let t = next_token(&mut last, &seen, s, &mut rng);
            seen.push(t);
            if t == STOP_SPEECH {
                break;
            }
            total += 1;
            pos = vec![*pos.last().unwrap() + 1];
            let (_, e) = self.embed(&[t], &[step as i64 + 1], exaggeration)?;
            inputs = tensor(vec![1, 1, width], e)?;
        }
        Ok(seen.into_iter().skip(1).filter(|&t| t != STOP_SPEECH && t < START_SPEECH).collect())
    }

    /// Speech for `text` in the voice of `clip` (24 kHz mono): part by part, joined with a short pause. `on_part` sees
    /// (done, total) parts.
    pub fn speak(&mut self, text: &str, clip: &[f32], language: Option<&str>, exaggeration: f32, s: &Sampling,
                 on_part: &mut dyn FnMut(usize, usize)) -> Result<Vec<f32>> {
        let parts = audio::sentences(text, 300);
        if parts.is_empty() {
            bail!("nothing to say");
        }
        let key = self.voice(clip)?;
        let pause = vec![0f32; RATE as usize / 4];
        let mut out = Vec::new();
        for (k, p) in parts.iter().enumerate() {
            // 25 speech tokens a second; a generous bound for the words given (it stops at its own end)
            let max_new = ((p.chars().count() as f64 / 12.0 + 3.0) * 25.0).min(1500.0) as usize;
            let mut sk = *s;
            sk.seed = s.seed.wrapping_add(k as u64);
            let speech = self.tokens(p, key, language, exaggeration, &sk, max_new)?;
            let v = &self.voices[&key];
            let mut toks = v.prompt.clone();
            toks.extend(&speech);
            if !self.multilingual {
                toks.extend([SILENCE; 3]);
            }
            let (x, f) = (v.x_vector.clone(), v.features.clone());
            let n = toks.len() as i64;
            let wav = self.decoder.run(ort::inputs![
                "speech_tokens" => tensor(vec![1, n], toks)?,
                "speaker_embeddings" => tensor(vec![1, x.len() as i64], x)?,
                "speaker_features" => tensor(f.0, f.1)?,
            ])?;
            let (_, pcm) = take::<f32>(&wav["waveform"])?;
            if k > 0 {
                out.extend_from_slice(&pause);
            }
            out.extend(pcm);
            on_part(k + 1, parts.len());
        }
        Ok(audio::loudness(&out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn punctuation_is_normalized_as_chatterbox_does() {
        assert_eq!(punc_norm("hello   world"), "Hello world.");
        assert_eq!(punc_norm("Wait... what: this; that — ok?"), "Wait,  what, this,  that - ok?");
        assert_eq!(punc_norm(""), "You need to add some text for me to talk.");
    }

    #[test]
    fn korean_syllables_become_letters() {
        assert_eq!(korean("한"), "\u{1112}\u{1161}\u{11AB}");
        assert_eq!(korean("abc"), "abc");
    }

    #[test]
    fn sampling_is_greedy_at_temperature_zero_and_seeded_above() {
        let s = Sampling { temperature: 0.0, ..Default::default() };
        let mut l = vec![0.0, 3.0, 1.0];
        assert_eq!(next_token(&mut l, &[], &s, &mut 1), 1);
        // the penalty turns the repeated favourite down
        let s = Sampling { temperature: 0.0, repetition_penalty: 4.0, ..Default::default() };
        let mut l = vec![0.0, 3.0, 1.0];
        assert_eq!(next_token(&mut l, &[1], &s, &mut 1), 2);
        let s = Sampling::default();
        let draw = |seed| { let mut r = seed; (0..20).map(|_| next_token(&mut vec![1.0, 1.0, 1.0, 1.0], &[], &s, &mut r)).collect::<Vec<_>>() };
        assert_eq!(draw(7), draw(7));
        assert_ne!(draw(7), draw(8));
    }
}
