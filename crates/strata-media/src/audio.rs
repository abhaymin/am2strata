//! Speech in and out for am2strata-media (v2.3), in Rust on candle:
//! - speech to text: Whisper (openai/whisper-*), the decoding loop of candle's whisper example (greedy at temperature
//!   0 with the temperature fallback, language detection, translation to English, segment timestamps);
//! - text to speech: Parler-TTS mini v1 (a voice described in words), its DAC codes decoded to 44.1 kHz audio.
//!
//! Input audio is any file symphonia reads (WAV, FLAC, MP3, OGG/Vorbis, M4A/AAC) - the web app records WAV - downmixed
//! to mono and resampled to Whisper's 16 kHz. Models come from Hugging Face into <MOLD_HOME>/audio (hf-hub's cache
//! layout), like the image and video models into MOLD_HOME. They run on the CPU unless a request asks for "cuda"
//! (an engine built with the cuda feature): the GPU stays with the text model.

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as wm, audio as wa, Config as WhisperConfig};
use serde_json::{json, Value};
use std::path::PathBuf;
use tokenizers::Tokenizer;

pub struct Entry {
    pub name: &'static str,
    pub display: &'static str,
    pub repo: &'static str,
    /// "stt" (speech to text) or "tts" (text to speech)
    pub task: &'static str,
    pub about: &'static str,
    pub licence: &'static str,
    pub size_gb: f64,
}

const FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];

pub const CATALOG: &[Entry] = &[
    Entry { name: "whisper-large-v3-turbo", display: "Whisper large-v3 turbo", repo: "openai/whisper-large-v3-turbo", task: "stt",
            about: "speech to text in 99 languages, and translation to English; the best of these", licence: "MIT", size_gb: 1.62 },
    Entry { name: "whisper-small", display: "Whisper small", repo: "openai/whisper-small", task: "stt",
            about: "speech to text, multilingual: faster on the CPU, a little less accurate", licence: "Apache-2.0", size_gb: 0.97 },
    Entry { name: "whisper-base", display: "Whisper base", repo: "openai/whisper-base", task: "stt",
            about: "speech to text, multilingual: the fastest, for dictation", licence: "Apache-2.0", size_gb: 0.29 },
    Entry { name: "parler-tts-mini-v1", display: "Parler-TTS mini v1", repo: "parler-tts/parler-tts-mini-v1", task: "tts",
            about: "text to speech in English; the voice is described in words (who speaks, how fast, how close)",
            licence: "Apache-2.0", size_gb: 3.51 },
];

/// The voices offered by name (OpenAI's `voice`), as Parler-TTS descriptions; any other text is used as the description.
pub const VOICES: &[(&str, &str)] = &[
    ("laura", "Laura's voice is calm and clear, with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."),
    ("jon", "Jon's voice is warm and steady, speaking at a moderate pace with a slightly low pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."),
    ("lea", "Lea speaks in a bright, expressive and animated way, slightly fast, with a higher pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."),
    ("gary", "Gary speaks slowly and deliberately with a deep voice, in a quiet room. The recording is of very high quality, with the speaker's voice sounding clear and very close up."),
];

pub fn entry(name: &str) -> Option<&'static Entry> {
    CATALOG.iter().find(|e| e.name == name)
}

fn cache_dir() -> PathBuf {
    let home = std::env::var_os("MOLD_HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join("audio")
}

fn local(e: &Entry) -> Option<Vec<PathBuf>> {
    let repo = hf_hub::Cache::new(cache_dir()).model(e.repo.to_string());
    FILES.iter().map(|f| repo.get(f)).collect()
}

pub fn downloaded(e: &Entry) -> bool {
    local(e).is_some()
}

/// Downloads a model's three files (resumable by hf-hub; no progress bar: stdout is the protocol).
pub fn pull(e: &Entry) -> Result<()> {
    let api = hf_hub::api::sync::ApiBuilder::new().with_cache_dir(cache_dir()).with_progress(false).build()?;
    let repo = api.model(e.repo.to_string());
    for f in FILES {
        repo.get(f).with_context(|| format!("{}: {f}", e.repo))?;
    }
    Ok(())
}

pub fn list() -> Vec<Value> {
    CATALOG.iter().map(|e| json!({
        "name": e.name, "display_name": e.display, "kind": "audio", "task": e.task, "family": if e.task == "stt" { "whisper" } else { "parler-tts" },
        "description": e.about, "licence": e.licence, "size_gb": e.size_gb, "gated": false, "downloaded": downloaded(e),
        "voices": if e.task == "tts" { json!(VOICES.iter().map(|v| v.0).collect::<Vec<_>>()) } else { Value::Null },
    })).collect()
}

fn device(req: &Value) -> Result<Device> {
    match req["device"].as_str() {
        Some("cuda") if cfg!(feature = "cuda") => Ok(Device::new_cuda(0)?),
        Some("cuda") => bail!("this media engine was built without CUDA: set where speech runs (audio_device) to cpu, or build it with --features cuda"),
        _ => Ok(Device::Cpu),
    }
}

// ------------------------------------------------------------------ audio files

/// Any file symphonia reads -> (mono samples, sample rate, warnings: packets that did not decode).
pub fn decode(bytes: Vec<u8>) -> Result<(Vec<f32>, u32, Vec<String>)> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::conv::FromSample;
    fn mix<T>(out: &mut Vec<f32>, b: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>)
    where T: symphonia::core::sample::Sample, f32: FromSample<T> {
        let ch = b.spec().channels.count().max(1);
        for i in 0..b.frames() {
            let s: f32 = (0..ch).map(|c| f32::from_sample(b.chan(c)[i])).sum();
            out.push(s / ch as f32);
        }
    }
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(std::io::Cursor::new(bytes)), Default::default());
    let probed = symphonia::default::get_probe()
        .format(&Default::default(), mss, &Default::default(), &Default::default())
        .map_err(|e| anyhow!("not an audio file this engine reads (WAV, FLAC, MP3, OGG/Vorbis, M4A): {e}"))?;
    let mut format = probed.format;
    let track = format.tracks().iter().find(|t| t.codec_params.codec != CODEC_TYPE_NULL).ok_or_else(|| anyhow!("no audio track"))?;
    let rate = track.codec_params.sample_rate.unwrap_or(0);
    let id = track.id;
    let mut dec = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| anyhow!("unsupported audio codec: {e}"))?;
    let (mut out, mut packets, mut bad, mut ended) = (Vec::new(), 0usize, 0usize, None);
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(e) => {
                ended = Some(e.to_string());
                break;
            }
        };
        if packet.track_id() != id {
            continue;
        }
        packets += 1;
        let buf = match dec.decode(&packet) {
            Ok(b) => b,
            Err(symphonia::core::errors::Error::DecodeError(_)) => {
                bad += 1; // a damaged packet: skipped, and said so
                continue;
            }
            Err(e) => return Err(anyhow!("{e}")),
        };
        match buf {
            AudioBufferRef::F32(b) => mix(&mut out, b),
            AudioBufferRef::F64(b) => mix(&mut out, b),
            AudioBufferRef::S32(b) => mix(&mut out, b),
            AudioBufferRef::S24(b) => mix(&mut out, b),
            AudioBufferRef::S16(b) => mix(&mut out, b),
            AudioBufferRef::S8(b) => mix(&mut out, b),
            AudioBufferRef::U32(b) => mix(&mut out, b),
            AudioBufferRef::U24(b) => mix(&mut out, b),
            AudioBufferRef::U16(b) => mix(&mut out, b),
            AudioBufferRef::U8(b) => mix(&mut out, b),
        }
    }
    if rate == 0 || out.is_empty() {
        bail!("the audio file holds no samples");
    }
    let mut warnings = Vec::new();
    if bad > 0 {
        warnings.push(format!("decoded {} of {packets} packets: {bad} were damaged and skipped", packets - bad));
    }
    if let Some(e) = ended {
        warnings.push(format!("the file ended early ({e}): the transcript covers what was read"));
    }
    Ok((out, rate, warnings))
}

/// Band-limited resampling (a Hann-windowed sinc, 16 zero crossings), for any rate pair.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return x.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let cutoff = ratio.min(1.0) * 0.95;
    let half = 16.0 / cutoff;
    let n = (x.len() as f64 * ratio).round() as usize;
    (0..n).map(|i| {
        let t = i as f64 / ratio;
        let (lo, hi) = (((t - half).floor().max(0.0)) as usize, ((t + half).ceil() as usize).min(x.len().saturating_sub(1)));
        let mut acc = 0.0;
        for j in lo..=hi {
            let d = j as f64 - t;
            let w = 0.5 * (1.0 + (std::f64::consts::PI * d / half).cos());
            let s = if d.abs() < 1e-9 { 1.0 } else { (std::f64::consts::PI * d * cutoff).sin() / (std::f64::consts::PI * d) };
            acc += x[j] as f64 * s * cutoff * w;
        }
        acc as f32
    }).collect()
}

/// 16-bit PCM WAV, mono.
pub fn wav(samples: &[f32], rate: u32) -> Vec<u8> {
    let mut w = Vec::with_capacity(44 + samples.len() * 2);
    let data = samples.len() as u32 * 2;
    w.extend(b"RIFF");
    w.extend((36 + data).to_le_bytes());
    w.extend(b"WAVEfmt ");
    w.extend(16u32.to_le_bytes());
    w.extend(1u16.to_le_bytes());
    w.extend(1u16.to_le_bytes());
    w.extend(rate.to_le_bytes());
    w.extend((rate * 2).to_le_bytes());
    w.extend(2u16.to_le_bytes());
    w.extend(16u16.to_le_bytes());
    w.extend(b"data");
    w.extend(data.to_le_bytes());
    for s in samples {
        w.extend(((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    w
}

// ------------------------------------------------------------------ Whisper

const MEL_80: &[u8] = include_bytes!("../assets/melfilters.bytes");
const MEL_128: &[u8] = include_bytes!("../assets/melfilters128.bytes");

const LANGUAGES: [&str; 99] = ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi",
    "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
    "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr",
    "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"];

pub struct Whisper {
    pub name: String,
    model: wm::model::Whisper,
    tok: Tokenizer,
    config: WhisperConfig,
    filters: Vec<f32>,
    device: Device,
}

fn tid(tok: &Tokenizer, t: &str) -> Result<u32> {
    tok.token_to_id(t).ok_or_else(|| anyhow!("no token {t}"))
}

impl Whisper {
    pub fn load(e: &Entry, req: &Value) -> Result<Whisper> {
        let files = local(e).ok_or_else(|| anyhow!("{} is not downloaded (pull it first)", e.name))?;
        let config: WhisperConfig = serde_json::from_str(&std::fs::read_to_string(&files[0])?)?;
        let tok = Tokenizer::from_file(&files[1]).map_err(|e| anyhow!("{e}"))?;
        let device = device(req)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&files[2]], wm::DTYPE, &device)? };
        let model = wm::model::Whisper::load(&vb, config.clone())?;
        let bytes = match config.num_mel_bins { 80 => MEL_80, 128 => MEL_128, n => bail!("unexpected num_mel_bins {n}") };
        let filters = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok(Whisper { name: e.name.to_string(), model, tok, config, filters, device })
    }

    /// The best language from a window's encoder features (detect_language of candle's example): (code, probability).
    fn detect(&mut self, feats: &Tensor) -> Result<(String, f32)> {
        let ids: Vec<u32> = LANGUAGES.iter().map(|l| tid(&self.tok, &format!("<|{l}|>"))).collect::<Result<_>>()?;
        let tokens = Tensor::new(&[[tid(&self.tok, wm::SOT_TOKEN)?]], &self.device)?;
        let ys = self.model.decoder.forward(&tokens, feats, true)?;
        let logits = self.model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
        let logits = logits.index_select(&Tensor::new(ids.as_slice(), &self.device)?, 0)?;
        let probs = candle_nn::ops::softmax(&logits, D::Minus1)?.to_vec1::<f32>()?;
        let (i, p) = probs.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        Ok((LANGUAGES[i].to_string(), *p))
    }

    /// One window's tokens at temperature `t` from its encoder features (computed once per window, shared by the
    /// language detection and every temperature) -> (sampled tokens after the prefix, average log-probability,
    /// no-speech probability). With timestamps, the sampling follows Whisper's timestamp rules.
    fn decode_window(&mut self, feats: &Tensor, prefix: &[u32], t: f64, suppress: &Tensor, ts: Option<&Timestamps>, rng: &mut u64)
                     -> Result<(Vec<u32>, f64, f64)> {
        let eot = tid(&self.tok, wm::EOT_TOKEN)?;
        let no_speech = wm::NO_SPEECH_TOKENS.iter().find_map(|t| self.tok.token_to_id(t)).ok_or_else(|| anyhow!("no no-speech token"))?;
        let mut tokens = prefix.to_vec();
        let (mut sum_lp, mut no_speech_p) = (0f64, f64::NAN);
        for i in 0..self.config.max_target_positions / 2 {
            let x = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let ys = self.model.decoder.forward(&x, feats, i == 0)?;
            if i == 0 {
                let l = self.model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_p = candle_nn::ops::softmax(&l, 0)?.i(no_speech as usize)?.to_scalar::<f32>()? as f64;
            }
            let (_, n, _) = ys.dims3()?;
            let mut logits = self.model.decoder.final_linear(&ys.i((..1, n - 1..))?)?.i(0)?.i(0)?.broadcast_add(suppress)?;
            if let Some(ts) = ts {
                logits = ts.apply(&logits, &tokens[prefix.len()..], tokens.len() == prefix.len(), eot)?;
            }
            let next = if t > 0.0 {
                let p = candle_nn::ops::softmax(&(&logits / t)?, 0)?.to_vec1::<f32>()?;
                *rng ^= *rng << 13;
                *rng ^= *rng >> 7;
                *rng ^= *rng << 17;
                let mut r = (*rng >> 11) as f64 / (1u64 << 53) as f64;
                let mut pick = p.len() - 1;
                for (k, v) in p.iter().enumerate() {
                    r -= *v as f64;
                    if r <= 0.0 {
                        pick = k;
                        break;
                    }
                }
                pick as u32
            } else {
                let v = logits.to_vec1::<f32>()?;
                v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(k, _)| k as u32).unwrap()
            };
            tokens.push(next);
            let p = candle_nn::ops::softmax(&logits, D::Minus1)?.i(next as usize)?.to_scalar::<f32>()? as f64;
            if next == eot || tokens.len() > self.config.max_target_positions {
                break;
            }
            sum_lp += p.ln();
        }
        let avg = sum_lp / tokens.len() as f64;
        let mut out = tokens.split_off(prefix.len());
        if out.last() == Some(&eot) {
            out.pop();
        }
        Ok((out, avg, no_speech_p))
    }

    /// Transcribes (or translates to English) mono 16 kHz samples. `on_segment` sees each segment as it is done.
    /// Windows of 30 s are decoded with Whisper's timestamp tokens and the next window starts after the last complete
    /// segment (OpenAI's transcribe), so a word at a window's end is read in the next one instead of being cut.
    pub fn run(&mut self, pcm: &[f32], language: Option<&str>, translate: bool, on_segment: &mut dyn FnMut(&Value)) -> Result<Value> {
        let mel = wa::pcm_to_mel(&self.config, pcm, &self.filters);
        let total_frames = mel.len() / self.config.num_mel_bins;
        let mel = Tensor::from_vec(mel, (1, self.config.num_mel_bins, total_frames), &self.device)?;
        let content = (pcm.len() / wm::HOP_LENGTH).min(total_frames);
        let duration = pcm.len() as f64 / wm::SAMPLE_RATE as f64;
        let multilingual = self.tok.token_to_id("<|de|>").is_some();
        let no_ts = tid(&self.tok, wm::NO_TIMESTAMPS_TOKEN)?;
        let tb = no_ts + 1; // the first timestamp token, <|0.00|>
        let ts = Timestamps { begin: tb, vocab: self.config.vocab_size as u32, device: self.device.clone() };
        let suppress: Vec<f32> = (0..self.config.vocab_size as u32)
            .map(|i| if self.config.suppress_tokens.contains(&i) || i == no_ts { f32::NEG_INFINITY } else { 0.0 }).collect();
        let suppress = Tensor::new(suppress.as_slice(), &self.device)?;
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let (mut seek, mut segments, mut text) = (0usize, Vec::<Value>::new(), String::new());
        let (mut lang, mut lang_p, mut prefix) = (None::<String>, None::<f32>, Vec::<u32>::new());
        while seek < content {
            let window = mel.narrow(2, seek, wm::N_FRAMES.min(total_frames - seek))?;
            let size = wm::N_FRAMES.min(content - seek);
            let offset = (seek * wm::HOP_LENGTH) as f64 / wm::SAMPLE_RATE as f64;
            // silence: Whisper writes a word into it ("you", "Thank you."), so a window without sound is no speech
            let (a, z) = (seek * wm::HOP_LENGTH, ((seek + size) * wm::HOP_LENGTH).min(pcm.len()));
            let rms = (pcm[a..z].iter().map(|v| v * v).sum::<f32>() / (z - a).max(1) as f32).sqrt();
            if rms < 1e-3 {
                seek += size;
                continue;
            }
            let feats = self.model.encoder.forward(&window, true)?;
            if prefix.is_empty() {
                lang = match language {
                    Some(l) if multilingual => {
                        tid(&self.tok, &format!("<|{l}|>")).map_err(|_| anyhow!("language {l:?} is not one Whisper knows"))?;
                        Some(l.to_string())
                    }
                    Some(_) => None,
                    None if multilingual => {
                        let (l, p) = self.detect(&feats)?;
                        lang_p = Some(p);
                        Some(l)
                    }
                    None => None,
                };
                prefix.push(tid(&self.tok, wm::SOT_TOKEN)?);
                if let Some(l) = &lang {
                    prefix.push(tid(&self.tok, &format!("<|{l}|>"))?);
                }
                prefix.push(tid(&self.tok, if translate { wm::TRANSLATE_TOKEN } else { wm::TRANSCRIBE_TOKEN })?);
            }
            // the temperature fallback: a window decoded too unsure is decoded again, a little warmer
            let mut best = None;
            for (k, &t) in wm::TEMPERATURES.iter().enumerate() {
                let r = self.decode_window(&feats, &prefix, t, &suppress, Some(&ts), &mut rng)?;
                let fallback = r.1 < wm::LOGPROB_THRESHOLD && r.2 <= wm::NO_SPEECH_THRESHOLD;
                best = Some(r);
                if !fallback || k == wm::TEMPERATURES.len() - 1 {
                    break;
                }
            }
            let (tokens, avg, ns) = best.unwrap();
            if ns > wm::NO_SPEECH_THRESHOLD && avg < wm::LOGPROB_THRESHOLD {
                seek += size; // no speech in this window
                continue;
            }
            let (spans, advance) = split_by_timestamps(&tokens, tb, size);
            for (start, end, body) in spans {
                let s = self.tok.decode(&body, true).map_err(|e| anyhow!("{e}"))?.trim().to_string();
                if s.is_empty() {
                    continue;
                }
                let (a, b) = ((offset + start).min(duration), (offset + end).min(duration));
                let seg = json!({"id": segments.len(), "start": (a * 100.0).round() / 100.0, "end": (b.max(a) * 100.0).round() / 100.0,
                                 "text": s, "avg_logprob": (avg * 1000.0).round() / 1000.0, "no_speech_prob": (ns * 1000.0).round() / 1000.0});
                on_segment(&seg);
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&s);
                segments.push(seg);
            }
            seek += advance.max(1);
        }
        Ok(json!({"text": text, "language": lang, "language_probability": lang_p, "segments": segments,
                  "duration": (duration * 100.0).round() / 100.0, "task": if translate { "translate" } else { "transcribe" }}))
    }
}

/// Whisper's timestamp rules (candle's whisper example apply_timestamp_rules, after OpenAI's ApplyTimestampRules):
/// timestamps come in pairs, never go back, and a window starts with one.
struct Timestamps {
    begin: u32,
    vocab: u32,
    device: Device,
}

impl Timestamps {
    fn apply(&self, logits: &Tensor, sampled: &[u32], first: bool, eot: u32) -> Result<Tensor> {
        let tb = self.begin;
        let mut mask = vec![0f32; self.vocab as usize];
        let mut any = false;
        let mut forbid = |r: std::ops::Range<u32>, mask: &mut Vec<f32>| {
            for i in r {
                if (i as usize) < mask.len() {
                    mask[i as usize] = f32::NEG_INFINITY;
                }
            }
            any = true;
        };
        if first {
            forbid(0..tb, &mut mask); // a window opens with a timestamp
        } else {
            let last_ts = sampled.last().is_some_and(|t| *t >= tb);
            let penult_ts = sampled.len() < 2 || sampled[sampled.len() - 2] >= tb;
            if last_ts {
                if penult_ts {
                    forbid(tb..self.vocab, &mut mask); // a closed pair: text next
                } else {
                    forbid(0..eot, &mut mask); // an open one: its closing timestamp (or the end) next
                }
            }
            // timestamps never go back
            if let Some(&last) = sampled.iter().rev().find(|t| **t >= tb) {
                let from = if last_ts && !penult_ts { last } else { last + 1 };
                forbid(tb..from, &mut mask);
            }
        }
        let mut logits = logits.clone();
        if any {
            logits = logits.broadcast_add(&Tensor::new(mask.as_slice(), &self.device)?)?;
        }
        // when the timestamps together are likelier than any one text token, a timestamp it is
        let lp = candle_nn::ops::log_softmax(&logits, 0)?.to_vec1::<f32>()?;
        let ts_lp = {
            let m = lp[tb as usize..].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if m.is_finite() { m + lp[tb as usize..].iter().map(|v| (v - m).exp()).sum::<f32>().ln() } else { m }
        };
        let text_max = lp[..tb as usize].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if ts_lp > text_max {
            let mut m2 = vec![0f32; self.vocab as usize];
            m2[..tb as usize].iter_mut().for_each(|v| *v = f32::NEG_INFINITY);
            logits = logits.broadcast_add(&Tensor::new(m2.as_slice(), &self.device)?)?;
        }
        Ok(logits)
    }
}

/// A window's tokens split at their timestamps (OpenAI's transcribe): (start s, end s, text tokens) per segment, and
/// how many mel frames the next window moves on - to the last complete segment's end, or the whole window when the
/// output ends on a single timestamp or holds no pair.
pub fn split_by_timestamps(tokens: &[u32], tb: u32, size: usize) -> (Vec<(f64, f64, Vec<u32>)>, usize) {
    let is_ts = |t: u32| t >= tb;
    let secs = |t: u32| (t - tb) as f64 * 0.02;
    let n = tokens.len();
    let single_end = n >= 2 && !is_ts(tokens[n - 2]) && is_ts(tokens[n - 1]);
    let mut cuts: Vec<usize> = (1..n).filter(|&i| is_ts(tokens[i]) && is_ts(tokens[i - 1])).collect();
    if cuts.is_empty() {
        let body: Vec<u32> = tokens.iter().copied().filter(|t| !is_ts(*t)).collect();
        let last = tokens.iter().rev().find(|t| is_ts(**t)).copied();
        let end = match last {
            Some(t) if t != tb => secs(t),
            _ => size as f64 * wm::HOP_LENGTH as f64 / wm::SAMPLE_RATE as f64,
        };
        let start = tokens.first().filter(|t| is_ts(**t)).map(|t| secs(*t)).unwrap_or(0.0);
        return (vec![(start, end, body)], size);
    }
    if single_end {
        cuts.push(n);
    }
    let (mut spans, mut last) = (Vec::new(), 0usize);
    for c in &cuts {
        let part = &tokens[last..*c];
        if let (Some(&a), Some(&b)) = (part.first(), part.last()) {
            let body: Vec<u32> = part.iter().copied().filter(|t| !is_ts(*t)).collect();
            spans.push((if is_ts(a) { secs(a) } else { 0.0 }, if is_ts(b) { secs(b) } else { secs(a.max(tb)) }, body));
        }
        last = *c;
    }
    let advance = if single_end {
        size
    } else {
        // the last complete segment's end, in mel frames (one timestamp step = 0.02 s = 2 frames)
        ((tokens[last - 1].saturating_sub(tb)) as usize * 2).min(size)
    };
    (spans, advance)
}

// ------------------------------------------------------------------ Parler-TTS

pub struct Parler {
    pub name: String,
    model: candle_transformers::models::parler_tts::Model,
    tok: Tokenizer,
    rate: u32,
    device: Device,
}

impl Parler {
    pub fn load(e: &Entry, req: &Value) -> Result<Parler> {
        let files = local(e).ok_or_else(|| anyhow!("{} is not downloaded (pull it first)", e.name))?;
        let config: candle_transformers::models::parler_tts::Config = serde_json::from_reader(std::fs::File::open(&files[0])?)?;
        let tok = Tokenizer::from_file(&files[1]).map_err(|e| anyhow!("{e}"))?;
        let device = device(req)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&files[2]], DType::F32, &device)? };
        let rate = config.audio_encoder.sampling_rate;
        let model = candle_transformers::models::parler_tts::Model::new(&config, vb)?;
        Ok(Parler { name: e.name.to_string(), model, tok, rate, device })
    }

    /// Speech for `text` in the voice `description`: sentence by sentence (each a short generation), joined with a
    /// short pause. `on_part` sees (done, total) parts.
    pub fn speak(&mut self, text: &str, description: &str, seed: u64, temperature: f64, on_part: &mut dyn FnMut(usize, usize)) -> Result<(Vec<f32>, u32)> {
        let parts = sentences(text, 220);
        if parts.is_empty() {
            bail!("nothing to say");
        }
        let desc = self.tok.encode(description, true).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
        let desc = Tensor::new(desc, &self.device)?.unsqueeze(0)?;
        let mut out = Vec::new();
        let pause = vec![0f32; (self.rate as f32 * 0.25) as usize];
        for (k, p) in parts.iter().enumerate() {
            let ids = self.tok.encode(p.as_str(), true).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
            let prompt = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
            // about 86 codec frames a second of speech; a generous bound for the words given (it stops at its own end)
            let steps = ((p.chars().count() as f64 / 10.0 + 3.0) * 86.0).min(2600.0) as usize;
            let lp = candle_transformers::generation::LogitsProcessor::new(seed + k as u64, Some(temperature), None);
            let codes = self.model.generate(&prompt, &desc, lp, steps)?;
            let codes = codes.to_dtype(DType::I64)?.unsqueeze(0)?;
            let pcm = self.model.audio_encoder.decode_codes(&codes.to_device(&self.device)?)?;
            let pcm = pcm.i((0, 0))?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
            if k > 0 {
                out.extend_from_slice(&pause);
            }
            out.extend(pcm);
            on_part(k + 1, parts.len());
        }
        Ok((loudness(&out), self.rate))
    }
}

/// Text cut at sentence ends into parts of at most `max` characters (a long sentence at its commas, then its spaces).
pub fn sentences(text: &str, max: usize) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    let mut cur = String::new();
    let push = |parts: &mut Vec<String>, s: &str| {
        let s = s.trim();
        if !s.is_empty() {
            parts.push(s.to_string());
        }
    };
    for ch in text.chars() {
        cur.push(ch);
        let end = matches!(ch, '.' | '!' | '?' | '\n' | ';' | '。' | '！' | '？');
        if (end && cur.trim().len() > 1) || cur.len() >= max {
            if cur.len() >= max && !end {
                // cut at the last comma or space instead of in a word
                if let Some(i) = cur.rfind([',', ' ']) {
                    let rest = cur.split_off(i + 1);
                    push(&mut parts, &cur);
                    cur = rest;
                    continue;
                }
            }
            push(&mut parts, &cur);
            cur.clear();
        }
    }
    push(&mut parts, &cur);
    parts
}

/// RMS to about -18 dBFS, a soft limit at full scale.
fn loudness(x: &[f32]) -> Vec<f32> {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt();
    if rms < 1e-4 {
        return x.to_vec();
    }
    let gain = 0.125 / rms;
    x.iter().map(|v| (v * gain).tanh()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentences_are_cut_at_their_ends_and_long_ones_at_commas() {
        assert_eq!(sentences("Hello there. How are you? Fine!", 200), ["Hello there.", "How are you?", "Fine!"]);
        let long = "one, two, three, four, five, six";
        let p = sentences(long, 12);
        assert!(p.iter().all(|s| s.len() <= 12), "{p:?}");
        assert_eq!(p.join(" ").replace("  ", " "), long);
    }

    #[test]
    fn segments_follow_their_timestamps_and_the_next_window_starts_after_the_last_complete_one() {
        let tb = 1000;
        // <|0.00|> a b <|1.00|><|1.00|> c <|2.50|>  then an open segment d (cut by the window)
        let toks = [tb, 1, 2, tb + 50, tb + 50, 3, tb + 125, 4];
        let (spans, adv) = split_by_timestamps(&toks, tb, 3000);
        assert_eq!(spans.len(), 1, "{spans:?}");
        assert_eq!((spans[0].0, spans[0].1, spans[0].2.clone()), (0.0, 1.0, vec![1, 2]));
        assert_eq!(adv, 100, "moves to 1.00 s (100 frames), so 'c' and 'd' are read again in the next window");
        // ending on a single timestamp: the whole window is done
        let toks = [tb, 1, tb + 50, tb + 50, 2, tb + 125];
        let (spans, adv) = split_by_timestamps(&toks, tb, 3000);
        assert_eq!((spans.len(), adv), (2, 3000));
        assert_eq!((spans[1].0, spans[1].1), (1.0, 2.5));
        // no timestamp pair: one segment, the whole window
        let (spans, adv) = split_by_timestamps(&[tb, 5, 6, tb + 200], tb, 3000);
        assert_eq!((spans.len(), spans[0].1, adv), (1, 4.0, 3000));
    }

    #[test]
    fn a_damaged_file_says_so() {
        let tone: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let mut w = wav(&tone, 16000);
        w.truncate(w.len() - 7001); // cut mid-sample
        let (pcm, _, _) = decode(w).unwrap();
        assert!(pcm.len() < 16000 && pcm.len() > 12000, "{}", pcm.len());
    }

    #[test]
    fn wav_round_trips_through_the_decoder_and_resampling_keeps_a_tone() {
        let tone: Vec<f32> = (0..44100).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5).collect();
        let (back, rate, warn) = decode(wav(&tone, 44100)).unwrap();
        assert!(warn.is_empty());
        assert_eq!((rate, back.len()), (44100, 44100));
        assert!((back[1000] - tone[1000]).abs() < 1e-3);
        let r = resample(&back, 44100, 16000);
        assert!((r.len() as i64 - 16000).abs() <= 1);
        // a 440 Hz tone at 16 kHz still crosses zero about 880 times a second
        let crossings = r.windows(2).filter(|w| w[0].signum() != w[1].signum()).count();
        assert!((860..=900).contains(&crossings), "{crossings}");
    }
}
