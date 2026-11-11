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
    /// "whisper", "parler-tts" or "chatterbox"
    pub family: &'static str,
    /// the repo's files this engine reads (the first is what `downloaded` checks first)
    pub files: &'static [&'static str],
    /// files from other repos it also reads: (repo, file)
    pub extra: &'static [(&'static str, &'static str)],
    /// the languages it speaks (ISO 639-1); empty: English
    pub languages: &'static [&'static str],
}

/// Flan-T5's tokenizer: what the newer Parler checkpoints read the voice description with (their own tokenizer is for
/// the words spoken)
const FLAN_T5: &[(&str, &str)] = &[("google/flan-t5-large", "tokenizer.json")];
const PARLER_ML_LANGUAGES: &[&str] = &["en", "fr", "es", "pt", "pl", "de", "it", "nl"];
const INDIC_LANGUAGES: &[&str] = &["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "or", "pa", "as", "ne", "en"];

const HF3: &[&str] = &["config.json", "tokenizer.json", "model.safetensors"];
/// a Chatterbox ONNX export: the four graphs, each with its weights beside it
const CHATTERBOX_ONNX: &[&str] = &["tokenizer.json",
    "onnx/speech_encoder.onnx", "onnx/speech_encoder.onnx_data", "onnx/embed_tokens.onnx", "onnx/embed_tokens.onnx_data",
    "onnx/language_model.onnx", "onnx/language_model.onnx_data", "onnx/conditional_decoder.onnx", "onnx/conditional_decoder.onnx_data"];

pub const CATALOG: &[Entry] = &[
    Entry { name: "whisper-large-v3-turbo", display: "Whisper large-v3 turbo", repo: "openai/whisper-large-v3-turbo", task: "stt",
            about: "speech to text in 99 languages, and translation to English; the best of these", licence: "MIT", size_gb: 1.62, family: "whisper", files: HF3, extra: &[], languages: &[] },
    Entry { name: "whisper-small", display: "Whisper small", repo: "openai/whisper-small", task: "stt",
            about: "speech to text, multilingual: faster on the CPU, a little less accurate", licence: "Apache-2.0", size_gb: 0.97,
            family: "whisper", files: HF3, extra: &[], languages: &[] },
    Entry { name: "whisper-base", display: "Whisper base", repo: "openai/whisper-base", task: "stt",
            about: "speech to text, multilingual: the fastest, for dictation", licence: "Apache-2.0", size_gb: 0.29,
            family: "whisper", files: HF3, extra: &[], languages: &[] },
    Entry { name: "parler-tts-mini-v1", display: "Parler-TTS mini v1", repo: "parler-tts/parler-tts-mini-v1", task: "tts",
            about: "text to speech in English; the voice is described in words (who speaks, how fast, how close)",
            licence: "Apache-2.0", size_gb: 3.51, family: "parler-tts", files: HF3, extra: &[], languages: &[] },
    Entry { name: "chatterbox-turbo", display: "Chatterbox Turbo (ONNX)", repo: "ResembleAI/chatterbox-turbo-ONNX", task: "tts",
            about: "text to speech in English in the voice of a short reference clip; tags like [laugh] and [sigh] (ONNX Runtime)",
            licence: "MIT", size_gb: 3.29, family: "chatterbox", files: CHATTERBOX_ONNX, extra: &[], languages: &[] },
    Entry { name: "chatterbox-multilingual", display: "Chatterbox Multilingual (ONNX)", repo: "onnx-community/chatterbox-multilingual-ONNX",
            task: "tts", about: "text to speech in 20 languages (Hindi, Spanish, French and more) in the voice of a short reference clip (ONNX Runtime)",
            licence: "MIT", size_gb: 3.27, family: "chatterbox", files: CHATTERBOX_ONNX, extra: &[], languages: &[] },
    Entry { name: "parler-tts-mini-multilingual", display: "Parler-TTS mini multilingual v1.1",
            repo: "parler-tts/parler-tts-mini-multilingual-v1.1", task: "tts",
            about: "text to speech in 8 European languages; the voice is described in words. Makes the native female and male voices other models clone",
            licence: "Apache-2.0", size_gb: 3.75, family: "parler-tts-hf", files: HF3, extra: FLAN_T5, languages: PARLER_ML_LANGUAGES },
    Entry { name: "indic-parler-tts", display: "Indic Parler-TTS", repo: "ai4bharat/indic-parler-tts", task: "tts",
            about: "text to speech in Hindi and 20 Indian languages; the voice is described in words. Gated: accept its terms on Hugging Face first",
            licence: "Apache-2.0", size_gb: 3.75, family: "parler-tts-hf", files: HF3, extra: FLAN_T5, languages: INDIC_LANGUAGES },
];

/// A language's English name ("Hindi" for "hi"); the code when it is not known here.
pub fn language_name(code: &str) -> String {
    const MORE: &[(&str, &str)] = &[("en", "English"), ("bn", "Bengali"), ("ta", "Tamil"), ("te", "Telugu"), ("mr", "Marathi"),
        ("gu", "Gujarati"), ("kn", "Kannada"), ("ml", "Malayalam"), ("or", "Odia"), ("pa", "Punjabi"), ("as", "Assamese"), ("ne", "Nepali")];
    CHATTERBOX_LANGUAGES.iter().chain(MORE).find(|l| l.0 == code).map(|l| l.1.to_string()).unwrap_or_else(|| code.to_string())
}

/// Each named voice's gender: in another language than English it is spoken by the native voice of that gender.
pub const VOICE_GENDER: &[(&str, &str)] = &[("laura", "female"), ("lea", "female"), ("female", "female"),
                                            ("jon", "male"), ("gary", "male"), ("male", "male"), ("gautham", "male")];

/// The model that makes a named voice's reference clip, when it is not Parler-TTS mini v1: Gautham is Indic Parler's
/// English male speaker (Thoma), a warm Indian-English voice.
pub const VOICE_MAKERS: &[(&str, &str)] = &[("gautham", "indic-parler-tts")];

/// The native voices' descriptions (for a multilingual Parler model; synthetic voices, no person's recording)
pub const NATIVE_VOICES: &[(&str, &str)] = &[
    ("female", "A female speaker with a clear, warm voice speaks at a moderate pace with a natural pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."),
    ("male", "A male speaker with a clear, warm voice speaks at a moderate pace with a slightly low pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."),
];

/// What a native voice says for its reference clip (about 8 to 10 s), per language.
pub const NATIVE_LINES: &[(&str, &str)] = &[
    ("es", "Hola, bienvenidos. Esta es una breve muestra de mi voz natural, grabada con calma y claridad para ustedes hoy."),
    ("fr", "Bonjour et bienvenue. Voici un court extrait de ma voix naturelle, enregistré calmement et clairement pour vous aujourd'hui."),
    ("pt", "Olá, sejam bem-vindos. Esta é uma breve amostra da minha voz natural, gravada com calma e clareza para vocês hoje."),
    ("pl", "Dzień dobry i witam. To jest krótka próbka mojego naturalnego głosu, nagrana spokojnie i wyraźnie specjalnie dla was."),
    ("de", "Hallo und herzlich willkommen. Dies ist eine kurze Probe meiner natürlichen Stimme, ruhig und deutlich für Sie aufgenommen."),
    ("it", "Ciao e benvenuti. Questo è un breve esempio della mia voce naturale, registrato con calma e chiarezza per voi oggi."),
    ("nl", "Hallo en welkom. Dit is een korte opname van mijn natuurlijke stem, rustig en duidelijk voor u opgenomen."),
    ("hi", "नमस्ते और स्वागत है। यह मेरी स्वाभाविक आवाज़ का एक छोटा सा नमूना है, जिसे आज आपके लिए शांति और स्पष्टता से रिकॉर्ड किया गया है।"),
];

/// The downloaded model that makes the native voices of `language`.
pub fn native_maker(language: &str) -> Option<&'static Entry> {
    CATALOG.iter().find(|e| e.family == "parler-tts-hf" && e.languages.contains(&language) && downloaded(e))
}

/// The voices offered by name (OpenAI's `voice`), as Parler-TTS descriptions; any other text is used as the description.
pub const VOICES: &[(&str, &str)] = &[
    ("female", "A female speaker with a clear, calm voice speaks at a moderate pace with a natural pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."),
    ("male", "A male speaker with a clear, calm voice speaks at a moderate pace with a slightly low pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."),
    ("laura", "Laura's voice is calm and clear, with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."),
    ("jon", "Jon's voice is warm and steady, speaking at a moderate pace with a slightly low pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."),
    ("lea", "Lea speaks in a bright, expressive and animated way, slightly fast, with a higher pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."),
    ("gary", "Gary speaks slowly and deliberately with a deep voice, in a quiet room. The recording is of very high quality, with the speaker's voice sounding clear and very close up."),
    ("gautham", "Thoma's voice is calm and warm, speaking at a moderate pace with a natural pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up, with almost no background noise."),
];

/// Indic Parler's recommended speakers by language, (female, male), from its model card (ai4bharat/indic-parler-tts).
/// A language with one recommended voice of one gender has None for the other.
pub const INDIC_SPEAKERS: &[(&str, Option<&str>, Option<&str>)] = &[
    ("as", Some("Sita"), Some("Amit")), ("bn", Some("Aditi"), Some("Arjun")), ("gu", Some("Neha"), Some("Yash")),
    ("hi", Some("Divya"), Some("Rohit")), ("kn", Some("Anu"), Some("Suresh")), ("ml", Some("Anjali"), Some("Harish")),
    ("mr", Some("Sunita"), Some("Sanjay")), ("ne", Some("Amrita"), None), ("or", Some("Debjani"), Some("Manas")),
    ("ta", Some("Jaya"), None), ("te", Some("Lalitha"), Some("Prakash")), ("en", Some("Mary"), Some("Thoma")),
];

/// For Indic Parler: a named voice (laura, jon, female, male, ...) becomes the language's recommended speaker of that
/// voice's gender, in the card's way of describing a speaker. None: keep the voice's own description.
pub fn indic_description(voice: &str, language: &str) -> Option<String> {
    let gender = VOICE_GENDER.iter().find(|g| g.0 == voice.to_lowercase())?.1;
    let row = INDIC_SPEAKERS.iter().find(|r| r.0 == language)?;
    let name = if gender == "female" { row.1 } else { row.2 }?;
    Some(format!("{name}'s voice is calm and clear, speaking at a moderate pace with a natural pitch. The recording is of very high \
                  quality, with the speaker's voice sounding clear and very close up, with almost no background noise."))
}

/// Chatterbox Multilingual's languages whose text preparation this engine has: its 23 less Chinese (Cangjie codes with
/// a word segmenter), Japanese (kana from kanji) and Hebrew (vowel points), which need models of their own first.
pub const CHATTERBOX_LANGUAGES: &[(&str, &str)] = &[
    ("ar", "Arabic"), ("da", "Danish"), ("de", "German"), ("el", "Greek"), ("en", "English"), ("es", "Spanish"), ("fi", "Finnish"),
    ("fr", "French"), ("hi", "Hindi"), ("it", "Italian"), ("ko", "Korean"), ("ms", "Malay"), ("nl", "Dutch"), ("no", "Norwegian"),
    ("pl", "Polish"), ("pt", "Portuguese"), ("ru", "Russian"), ("sv", "Swedish"), ("sw", "Swahili"), ("tr", "Turkish"),
];

pub fn entry(name: &str) -> Option<&'static Entry> {
    CATALOG.iter().find(|e| e.name == name)
}

pub fn cache_dir() -> PathBuf {
    let home = std::env::var_os("MOLD_HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join("audio")
}

fn local(e: &Entry) -> Option<Vec<PathBuf>> {
    let cache = hf_hub::Cache::new(cache_dir());
    let repo = cache.model(e.repo.to_string());
    let mut files: Vec<PathBuf> = e.files.iter().map(|f| repo.get(f)).collect::<Option<_>>()?;
    for (r, f) in e.extra {
        files.push(cache.model(r.to_string()).get(f)?);
    }
    Some(files)
}

/// Where the clips of the named voices are kept (a model that clones a voice reads them).
pub fn voices_dir() -> PathBuf {
    cache_dir().join("voices")
}

/// The model's whole Hugging Face cache folder (blobs, snapshots, refs): what removing it deletes.
pub fn folder(e: &Entry) -> PathBuf {
    cache_dir().join(format!("models--{}", e.repo.replace('/', "--")))
}

/// Bytes on disk under `p`, symlinks counted as links (the HF cache's snapshots point at its blobs).
pub fn disk_bytes(p: &std::path::Path) -> u64 {
    let Ok(m) = std::fs::symlink_metadata(p) else { return 0 };
    if m.is_dir() {
        std::fs::read_dir(p).map(|rd| rd.flatten().map(|e| disk_bytes(&e.path())).sum()).unwrap_or(0)
    } else if m.file_type().is_symlink() {
        0
    } else {
        m.len()
    }
}

/// The model's files in the cache, in `e.files` order (None: not all downloaded).
pub fn local_files(e: &Entry) -> Option<Vec<PathBuf>> {
    local(e)
}

pub fn downloaded(e: &Entry) -> bool {
    local(e).is_some()
}

/// Download progress: `(file, file_index, files, done_bytes, total_bytes)`, bytes of this download only (files already
/// in the cache are left out of both).
pub type OnBytes<'a> = &'a mut dyn FnMut(&str, usize, usize, u64, u64);

/// Fetches files from Hugging Face into the cache, skipping cached ones, and tells `on` the bytes as they arrive.
/// `estimate`: the whole model's size, the total until each file's real size is known.
pub struct Fetcher<'a> {
    on: OnBytes<'a>,
    files: usize,
    index: usize,
    done: u64,
    estimate: u64,
}

struct Tick<'a, 'b> {
    f: &'b mut Fetcher<'a>,
    name: String,
    size: u64,
    got: u64,
}

impl hf_hub::api::Progress for Tick<'_, '_> {
    fn init(&mut self, size: usize, filename: &str) {
        self.size = size as u64;
        self.name = filename.to_string();
        self.report();
    }
    fn update(&mut self, size: usize) {
        self.got += size as u64;
        self.report();
    }
    fn finish(&mut self) {
        self.got = self.size;
        self.report();
        // this file is in: the next one counts from here
        self.f.done += self.size;
        self.got = 0;
        self.size = 0;
    }
}

impl Tick<'_, '_> {
    fn report(&mut self) {
        let done = self.f.done + self.got;
        let total = self.f.estimate.max(self.f.done + self.size);
        (self.f.on)(&self.name, self.f.index, self.f.files, done, total);
    }
}

impl<'a> Fetcher<'a> {
    pub fn new(files: usize, estimate_gb: f64, on: OnBytes<'a>) -> Fetcher<'a> {
        Fetcher { on, files, index: 0, done: 0, estimate: (estimate_gb * 1e9) as u64 }
    }

    pub fn get(&mut self, repo: &hf_hub::api::sync::ApiRepo, cached: &hf_hub::CacheRepo, file: &str) -> Result<PathBuf> {
        self.index += 1;
        if let Some(p) = cached.get(file) {
            // already here: it is not part of this download
            let have = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            self.estimate = self.estimate.saturating_sub(have);
            return Ok(p);
        }
        let path = repo.download_with_progress(file, Tick { f: self, name: file.to_string(), size: 0, got: 0 })?;
        self.estimate = self.estimate.max(self.done);
        Ok(path)
    }
}

/// Downloads a model's files (resumable by hf-hub), telling `on` the bytes as they arrive (stdout is the protocol, so
/// no progress bar).
pub fn pull(e: &Entry, on: OnBytes) -> Result<()> {
    let api = hf_hub::api::sync::ApiBuilder::new().with_cache_dir(cache_dir()).with_progress(false).build()?;
    let cache = hf_hub::Cache::new(cache_dir());
    let repo = api.model(e.repo.to_string());
    let mut f = Fetcher::new(e.files.len() + e.extra.len(), e.size_gb, on);
    for file in e.files {
        f.get(&repo, &cache.model(e.repo.to_string()), file).with_context(|| format!("{}: {file}", e.repo))
            .with_context(|| if e.name == "indic-parler-tts" { "Indic Parler-TTS is gated: accept its terms at https://huggingface.co/ai4bharat/indic-parler-tts with the account whose token this PC uses".to_string() } else { String::new() })?;
    }
    for (r, file) in e.extra {
        f.get(&api.model(r.to_string()), &cache.model(r.to_string()), file).with_context(|| format!("{r}: {file}"))?;
    }
    Ok(())
}

pub fn list() -> Vec<Value> {
    CATALOG.iter().map(|e| json!({
        "name": e.name, "display_name": e.display, "kind": "audio", "task": e.task, "family": e.family,
        "description": e.about, "licence": e.licence, "size_gb": e.size_gb, "downloaded": downloaded(e),
        "voices": if e.task == "tts" { json!(VOICES.iter().map(|v| v.0).collect::<Vec<_>>()) } else { Value::Null },
        "languages": if e.name == "chatterbox-multilingual" { json!(CHATTERBOX_LANGUAGES.iter().map(|l| json!({"code": l.0, "name": l.1})).collect::<Vec<_>>()) }
                     else if !e.languages.is_empty() { json!(e.languages.iter().map(|c| json!({"code": c, "name": language_name(c)})).collect::<Vec<_>>()) }
                     else { Value::Null },
        "gated": e.name == "indic-parler-tts",
        "runtime": if e.family == "chatterbox" { json!("onnx") } else { json!("candle") },
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
    /// the description's tokenizer (the same as `tok` for Parler-TTS mini v1)
    desc_tok: Tokenizer,
    /// the words are read as English (numbers spelled out in English)
    english_only: bool,
    rate: u32,
    device: Device,
}

impl Parler {
    pub fn load(e: &Entry, req: &Value) -> Result<Parler> {
        let files = local(e).ok_or_else(|| anyhow!("{} is not downloaded (pull it first)", e.name))?;
        let mut raw: Value = serde_json::from_reader(std::fs::File::open(&files[0])?)?;
        let hf = e.family == "parler-tts-hf";
        if hf {
            // transformers' DacConfig, in candle's DAC fields
            let d = crate::parler_hf::dac_config(&raw["audio_encoder"]);
            raw["audio_encoder"] = json!({"num_codebooks": d.num_codebooks, "model_bitrate": d.model_bitrate, "codebook_size": d.codebook_size,
                                          "latent_dim": d.latent_dim, "frame_rate": d.frame_rate, "sampling_rate": d.sampling_rate});
        }
        let config: candle_transformers::models::parler_tts::Config = serde_json::from_value(raw.clone())?;
        let tok = Tokenizer::from_file(&files[1]).map_err(|e| anyhow!("{e}"))?;
        let desc_tok = Tokenizer::from_file(files.get(3).filter(|_| hf).unwrap_or(&files[1])).map_err(|e| anyhow!("{e}"))?;
        let device = device(req)?;
        let vb = if hf {
            let rows = raw["decoder"]["vocab_size"].as_u64().unwrap_or(1088) as usize;
            crate::parler_hf::var_builder(&files[2], rows, DType::F32, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[&files[2]], DType::F32, &device)? }
        };
        let rate = config.audio_encoder.sampling_rate;
        let model = candle_transformers::models::parler_tts::Model::new(&config, vb)?;
        Ok(Parler { name: e.name.to_string(), model, tok, desc_tok, english_only: !hf, rate, device })
    }

    /// Speech for `text` in the voice `description`: sentence by sentence (each a short generation), joined with a
    /// short pause. `on_part` sees (done, total) parts.
    pub fn speak(&mut self, text: &str, description: &str, seed: u64, temperature: f64, on_part: &mut dyn FnMut(usize, usize)) -> Result<(Vec<f32>, u32)> {
        // Parler reads words, not digits: numbers, times, prices and percentages spelled out first (in English; the
        // multilingual checkpoints get the text as it is)
        let parts = sentences(&if self.english_only { spoken(text) } else { text.to_string() }, 220);
        if parts.is_empty() {
            bail!("nothing to say");
        }
        let desc = self.desc_tok.encode(description, true).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
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
            // Parler sometimes holds its last sound (a vowel or a hum that goes on): trimmed to a short natural end
            let (pcm, held) = trim_held_tail(&pcm, self.rate);
            if held > 0.0 {
                eprintln!("[media] speech part {}: a held last sound of {held:.1} s trimmed", k + 1);
            }
            if k > 0 {
                out.extend_from_slice(&pause);
            }
            out.extend(pcm);
            on_part(k + 1, parts.len());
        }
        Ok((loudness(&out), self.rate))
    }
}

/// A sentence's held last sound trimmed (VOICES-PLAN: Parler sometimes holds its last vowel or a hum for a second or
/// more). In 20 ms frames, a run of steady frames (loudness and zero crossings barely changing, still clearly voiced)
/// that ends the speech and lasts over 0.5 s is cut to its first 0.15 s, with a 80 ms fade; speech itself never holds
/// that still for that long. -> the samples, and the seconds cut (0 when nothing was).
pub fn trim_held_tail(pcm: &[f32], rate: u32) -> (Vec<f32>, f32) {
    let f = (rate as usize / 50).max(1);
    let frames: Vec<(f32, f32)> = pcm.chunks(f).filter(|c| c.len() == f).map(|c| {
        let rms = (c.iter().map(|x| x * x).sum::<f32>() / f as f32).sqrt();
        let zc = c.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count() as f32 / f as f32;
        (rms, zc)
    }).collect();
    let peak = frames.iter().map(|x| x.0).fold(0.0f32, f32::max);
    if peak <= 0.0 {
        return (pcm.to_vec(), 0.0);
    }
    let Some(last) = frames.iter().rposition(|x| x.0 > peak * 0.05) else { return (pcm.to_vec(), 0.0) };
    let mut start = last;
    while start > 0 {
        let (a, b) = (frames[start - 1], frames[start]);
        let steady = (a.0 - b.0).abs() < peak * 0.06 && (a.1 - b.1).abs() < 0.02 && a.0 > peak * 0.05;
        if !steady {
            break;
        }
        start -= 1;
    }
    let run = last + 1 - start;
    if run * 20 <= 500 {
        return (pcm.to_vec(), 0.0);
    }
    let keep = (start + 8) * f; // the first 0.15 s of the held sound
    let fade = 4 * f;
    let mut out = pcm[..keep.min(pcm.len())].to_vec();
    let n = out.len();
    for i in 0..fade.min(n) {
        out[n - 1 - i] *= i as f32 / fade as f32;
    }
    let cut = (pcm.len() - out.len()) as f32 / rate as f32;
    (out, cut)
}

/// How well `heard` (a transcript) matches `said`: the words they share in order (longest common subsequence) over
/// the longer of the two, from 0 to 1. Added words (babble at the start or end of a take) and dropped words both lower
/// it. Case and punctuation do not count; letters of any script do.
pub fn words_score(said: &str, heard: &str) -> f32 {
    let words = |t: &str| -> Vec<String> {
        t.to_lowercase().chars().map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' }).collect::<String>()
            .split_whitespace().map(str::to_string).collect()
    };
    let (a, b) = (words(said), words(heard));
    if a.is_empty() || b.is_empty() {
        return if a.is_empty() && b.is_empty() { 1.0 } else { 0.0 };
    }
    let mut prev = vec![0usize; b.len() + 1];
    for x in &a {
        let mut cur = vec![0usize; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = if x == y { prev[j] + 1 } else { cur[j].max(prev[j + 1]) };
        }
        prev = cur;
    }
    prev[b.len()] as f32 / a.len().max(b.len()) as f32
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

const ONES: [&str; 20] = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                          "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

/// A whole number in English words ("one thousand two hundred thirty-four").
pub fn number_words(n: u64) -> String {
    fn below_1000(n: u64) -> String {
        let (h, r) = (n / 100, n % 100);
        let mut out = Vec::new();
        if h > 0 {
            out.push(format!("{} hundred", ONES[h as usize]));
        }
        if r > 0 || h == 0 {
            out.push(if r < 20 { ONES[r as usize].to_string() } else if r % 10 == 0 { TENS[(r / 10) as usize].to_string() }
                     else { format!("{}-{}", TENS[(r / 10) as usize], ONES[(r % 10) as usize]) });
        }
        out.join(" ")
    }
    if n < 1000 {
        return below_1000(n);
    }
    let mut out = Vec::new();
    for (scale, name) in [(1_000_000_000_000u64, "trillion"), (1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")] {
        if n >= scale && n / scale % 1000 > 0 || (scale == 1_000_000_000_000 && n >= scale) {
            out.push(format!("{} {name}", if scale == 1_000_000_000_000 { number_words(n / scale) } else { below_1000(n / scale % 1000) }));
        }
    }
    if n % 1000 > 0 {
        out.push(below_1000(n % 1000));
    }
    out.join(" ")
}

fn ordinal(words: &str) -> String {
    let (head, last) = match words.rfind([' ', '-']) {
        Some(i) => (&words[..=i], &words[i + 1..]),
        None => ("", words),
    };
    let last = match last {
        "one" => "first".to_string(), "two" => "second".to_string(), "three" => "third".to_string(),
        "five" => "fifth".to_string(), "eight" => "eighth".to_string(), "nine" => "ninth".to_string(), "twelve" => "twelfth".to_string(),
        l if l.ends_with('y') => format!("{}ieth", &l[..l.len() - 1]),
        l => format!("{l}th"),
    };
    format!("{head}{last}")
}

/// Text as it is read aloud: numbers in words (with thousands separators, decimals, ordinals), clock times, prices in
/// $ / EUR / GBP, percentages and "&". Everything else is left as it is.
pub fn spoken(text: &str) -> String {
    let c: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + 16);
    let mut i = 0;
    let digits = |i: usize| -> usize { let mut j = i; while j < c.len() && c[j].is_ascii_digit() { j += 1 } j };
    while i < c.len() {
        let ch = c[i];
        let currency = match ch { '$' => Some(("dollar", "cent")), '€' => Some(("euro", "cent")), '£' => Some(("pound", "penny")), _ => None };
        if let Some((unit, sub)) = currency.filter(|_| i + 1 < c.len() && c[i + 1].is_ascii_digit()) {
            // $1,200.50 -> one thousand two hundred dollars and fifty cents
            let mut j = i + 1;
            let mut whole = String::new();
            while j < c.len() && (c[j].is_ascii_digit() || (c[j] == ',' && j + 3 < c.len() + 0 && c.get(j + 1).is_some_and(|d| d.is_ascii_digit()))) {
                if c[j] != ',' { whole.push(c[j]) }
                j += 1;
            }
            let n: u64 = whole.parse().unwrap_or(0);
            let mut cents = None;
            if j + 2 < c.len() + 1 && c.get(j) == Some(&'.') && c.get(j + 1).is_some_and(|d| d.is_ascii_digit()) && c.get(j + 2).is_some_and(|d| d.is_ascii_digit())
                && !c.get(j + 3).is_some_and(|d| d.is_ascii_digit()) {
                cents = Some((c[j + 1] as u64 - 48) * 10 + (c[j + 2] as u64 - 48));
                j += 3;
            }
            let plural = |w: &str, k: u64| if k == 1 { w.to_string() } else if w == "penny" { "pence".into() } else { format!("{w}s") };
            out.push_str(&format!("{} {}", number_words(n), plural(unit, n)));
            if let Some(ct) = cents.filter(|ct| *ct > 0) {
                out.push_str(&format!(" and {} {}", number_words(ct), plural(sub, ct)));
            }
            i = j;
            continue;
        }
        if ch == '&' {
            // "R&D" -> "R and D"; "you & me" keeps its own spaces
            if !out.ends_with(' ') { out.push(' ') }
            out.push_str("and");
            if !c.get(i + 1).is_some_and(|x| x.is_whitespace()) { out.push(' ') }
            i += 1;
            continue;
        }
        if !ch.is_ascii_digit() || (i > 0 && (c[i - 1].is_alphanumeric())) {
            out.push(ch);
            i += 1;
            continue;
        }
        let j = digits(i);
        let int: String = c[i..j].iter().collect();
        // 7:45 -> seven forty-five, 7:00 -> seven o'clock, 7:05 -> seven oh five
        if c.get(j) == Some(&':') && digits(j + 1) == j + 3 && int.len() <= 2 {
            let (h, m): (u64, u64) = (int.parse().unwrap_or(0), c[j + 1..j + 3].iter().collect::<String>().parse().unwrap_or(0));
            if h < 24 && m < 60 {
                out.push_str(&number_words(h));
                out.push_str(&match m { 0 => " o'clock".to_string(), 1..=9 => format!(" oh {}", number_words(m)), _ => format!(" {}", number_words(m)) });
                i = j + 3;
                continue;
            }
        }
        // 1,234,567 (groups of three after the first)
        let (mut whole, mut k) = (int.clone(), j);
        while c.get(k) == Some(&',') && digits(k + 1) == k + 4 && int.len() <= 3 {
            whole.extend(&c[k + 1..k + 4]);
            k += 4;
        }
        let Ok(n) = whole.parse::<u64>() else {
            out.push_str(&whole); // longer than any number word: digit by digit is no better
            i = k;
            continue;
        };
        let mut w = number_words(n);
        // 3.14 -> three point one four
        if c.get(k) == Some(&'.') && c.get(k + 1).is_some_and(|d| d.is_ascii_digit()) {
            let e = digits(k + 1);
            w.push_str(" point");
            for d in &c[k + 1..e] {
                w.push(' ');
                w.push_str(ONES[*d as usize - 48]);
            }
            k = e;
        }
        // 1st 2nd 3rd 4th
        let suf: String = c[k..(k + 2).min(c.len())].iter().collect::<String>().to_lowercase();
        if matches!(suf.as_str(), "st" | "nd" | "rd" | "th") && !c.get(k + 2).is_some_and(|x| x.is_alphanumeric()) {
            w = ordinal(&w);
            k += 2;
        }
        out.push_str(&w);
        if c.get(k) == Some(&'%') {
            out.push_str(" percent");
            k += 1;
        }
        i = k;
    }
    out
}

/// RMS to about -18 dBFS, a soft limit at full scale.
pub fn loudness(x: &[f32]) -> Vec<f32> {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt();
    if rms < 1e-4 {
        return x.to_vec();
    }
    let gain = 0.125 / rms;
    x.iter().map(|v| (v * gain).tanh()).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_take_with_babble_or_missing_words_scores_low() {
        use super::words_score;
        let said = "Hola. Este modelo funciona en tu propio ordenador, y tus datos no salen de él.";
        assert!(words_score(said, "Hola, este modelo funciona en tu propio ordenador y tus datos no salen de él.") > 0.99);
        assert!(words_score(said, "¡Hola! Este modelo funciona en tu propio ordenador y tus datos no salen de él. Ya has testoiz esta app si para the space") < 0.75);
        assert!(words_score(said, "Hola, este modelo funciona.") < 0.75);
        assert!(words_score("नमस्ते। यह मॉडल आपके अपने कंप्यूटर पर चलता है।", "नमस्ते यह मॉडल आपके अपने कंप्यूटर पर चलता है") > 0.99);
    }

    use super::*;

    #[test]
    fn numbers_times_and_prices_are_spelled_out_for_speech() {
        assert_eq!(spoken("The train leaves at 7:45, platform 12; tickets cost $30."),
                   "The train leaves at seven forty-five, platform twelve; tickets cost thirty dollars.");
        assert_eq!(spoken("We grew 12.5% to 1,234,567 users & 3 teams on the 21st, at 9:05 and 10:00."),
                   "We grew twelve point five percent to one million two hundred thirty-four thousand five hundred sixty-seven users and three teams on the twenty-first, at nine oh five and ten o'clock.");
        assert_eq!(spoken("£1.01 and €2,000.50 for 1 user"), "one pound and one penny and two thousand euros and fifty cents for one user");
        assert_eq!(number_words(0), "zero");
        assert_eq!(number_words(100_000_000_105), "one hundred billion one hundred five");
        assert_eq!(spoken("GPT4 and H100 stay"), "GPT4 and H100 stay");
    }

    #[test]
    fn gautham_is_a_male_voice_made_by_indic_parler() {
        assert_eq!(VOICE_GENDER.iter().find(|v| v.0 == "gautham").map(|v| v.1), Some("male"));
        assert_eq!(VOICE_MAKERS.iter().find(|v| v.0 == "gautham").map(|v| v.1), Some("indic-parler-tts"));
        assert!(VOICES.iter().find(|v| v.0 == "gautham").unwrap().1.starts_with("Thoma's voice"), "Indic Parler's English male speaker");
        assert!(entry("indic-parler-tts").is_some());
        // in an Indian language Gautham speaks as that language's male speaker
        assert!(indic_description("gautham", "ml").unwrap().starts_with("Harish's"));
    }

    #[test]
    fn indic_parler_gets_the_languages_own_speaker() {
        assert!(indic_description("laura", "ml").unwrap().starts_with("Anjali's voice"));
        assert!(indic_description("jon", "ml").unwrap().starts_with("Harish's voice"));
        assert!(indic_description("male", "hi").unwrap().starts_with("Rohit's"));
        assert!(indic_description("gary", "ta").is_none(), "Tamil has no recommended male voice: the description stays");
        assert!(indic_description("A deep voice", "ml").is_none(), "a description of one's own is kept");
    }

    #[test]
    fn a_held_last_sound_is_trimmed_and_speech_is_left_alone() {
        let rate = 24_000u32;
        let t = |i: usize| i as f32 / rate as f32;
        // speech-like: a voice whose loudness and pitch move with the syllables (4 a second)
        let speech: Vec<f32> = (0..rate as usize).map(|i| {
            let x = t(i);
            let syl = (x * 4.0 * std::f32::consts::TAU).sin().abs();
            syl * 0.5 * (x * (180.0 + 60.0 * (x * 3.0).sin()) * std::f32::consts::TAU).sin()
        }).collect();
        let (same, cut) = trim_held_tail(&speech, rate);
        assert_eq!((same.len(), cut), (speech.len(), 0.0), "speech is not cut");
        // the same, then a steady hum held for 1.2 s
        let mut held = speech.clone();
        held.extend((0..(rate as f32 * 1.2) as usize).map(|i| 0.3 * (t(i) * 200.0 * std::f32::consts::TAU).sin()));
        let (out, cut) = trim_held_tail(&held, rate);
        assert!(cut > 0.9 && cut < 1.2, "cut {cut}");
        assert!(out.len() > speech.len(), "the speech is kept");
        assert!(out.last().unwrap().abs() < 0.01, "faded out");
        // a natural ending that dies away is not a held sound
        let mut decay = speech.clone();
        decay.extend((0..(rate as f32 * 0.8) as usize).map(|i| 0.3 * (1.0 - i as f32 / (rate as f32 * 0.8)).powi(3) * (t(i) * 200.0 * std::f32::consts::TAU).sin()));
        assert_eq!(trim_held_tail(&decay, rate).1, 0.0);
    }

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
