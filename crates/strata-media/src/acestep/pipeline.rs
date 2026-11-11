//! A song from a caption and lyrics, all in Rust: the caption encoder and lyric embeddings (text.rs), the condition
//! encoder and the turbo DiT (dit.rs), the Oobleck decoder (oobleck.rs), and, when its files are there, the song
//! planner (planner.rs): its 5 Hz codes, through the detokenizer, are the DiT's source, so the song follows the
//! lyrics and the structure the planner wrote. Without it (ACE-Step's low-VRAM mode) the DiT starts from silence and
//! its vocals follow the lyrics only sometimes.

use super::{dit, oobleck, planner, text};
use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::io::Read;
use std::path::Path;

/// Latent frames a second (48 kHz / 1920).
pub const FRAMES_PER_SECOND: usize = 25;
/// ACE-Step's own range: 10 s to 10 minutes.
pub const MIN_SECONDS: u32 = 10;
pub const MAX_SECONDS: u32 = 600;

#[derive(Clone, Debug)]
pub struct Song {
    /// the style: genre, instruments, voice, mood ("warm indie pop, female vocal, acoustic guitar")
    pub caption: String,
    /// the lyrics with structure tags ([verse], [chorus], [bridge]); "[Instrumental]" for none
    pub lyrics: String,
    /// the vocal language as ACE-Step names it ("en", "hi", "es", ...)
    pub language: String,
    pub metas: text::Metas,
    pub seed: u64,
    pub steps: usize,
    pub shift: f32,
    /// the song planner's sampling, or None for no planner
    pub plan: Option<planner::Sampling>,
}

/// ACE-Step's silence latent ([1, 15000, 64]: 10 minutes at 25 Hz), the source a new song starts from and the timbre
/// when no reference is given. Read from its PyTorch file (a zip whose `data/0` is the float32 storage of [1, 64,
/// 15000]) without Python.
pub fn silence_latent(path: &Path, device: &Device) -> Result<Tensor> {
    let mut z = zip::ZipArchive::new(std::fs::File::open(path)?).context("silence_latent.pt is not a PyTorch zip")?;
    let name = (0..z.len()).filter_map(|i| z.by_index(i).ok().map(|f| f.name().to_string()))
        .find(|n| n.ends_with("/data/0")).ok_or_else(|| anyhow!("silence_latent.pt: no data/0"))?;
    let mut bytes = Vec::new();
    z.by_name(&name)?.read_to_end(&mut bytes)?;
    const N: usize = 64 * 15000;
    if bytes.len() != N * 4 {
        bail!("silence_latent.pt: {} bytes of data, {} expected", bytes.len(), N * 4);
    }
    let v: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    Ok(Tensor::from_vec(v, (1, 64, 15000), device)?.transpose(1, 2)?.contiguous()?)
}

/// Standard normal noise [1, frames, 64] from `seed`, the same on every device (a SplitMix64 stream, Box-Muller).
pub fn noise(seed: u64, frames: usize, device: &Device) -> Result<Tensor> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let n = frames * 64;
    let mut v = Vec::with_capacity(n + 1);
    while v.len() < n {
        let (u1, u2) = (next().max(1e-300), next());
        let r = (-2.0 * u1.ln()).sqrt();
        v.push((r * (std::f64::consts::TAU * u2).cos()) as f32);
        v.push((r * (std::f64::consts::TAU * u2).sin()) as f32);
    }
    v.truncate(n);
    Ok(Tensor::from_vec(v, (1, frames, 64), device)?)
}

/// The frames over which a continuation's start is crossfaded with its source (ACE-Step's handler passes 12).
pub const CROSSFADE: usize = 12;

pub struct AceStep {
    text: text::TextEncoder,
    cond: dit::ConditionEncoder,
    dit: dit::Dit,
    vae: oobleck::Decoder,
    /// the audio encoder, loaded at the first continuation
    encoder: Option<oobleck::Encoder>,
    vae_path: std::path::PathBuf,
    silence: Tensor,
    detok: dit::Detokenizer,
    /// the planner, loaded at its first song (3.4 GB in bf16); its folder, if downloaded
    planner: Option<planner::Planner>,
    planner_dir: Option<std::path::PathBuf>,
    thought: Option<planner::Thought>,
    dtype: DType,
    device: Device,
}

/// The planner's folder in ACE-Step's checkpoints.
pub const PLANNER_DIR: &str = "acestep-5Hz-lm-1.7B";

impl AceStep {
    /// `dir`: ACE-Step's checkpoints folder (`Qwen3-Embedding-0.6B`, `acestep-v15-turbo`, `vae`). bf16 on a GPU, f32
    /// on the CPU.
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let dtype = if device.is_cpu() { DType::F32 } else { DType::BF16 };
        let text = text::TextEncoder::load(&dir.join("Qwen3-Embedding-0.6B"), dtype, device)?;
        let ck = dir.join("acestep-v15-turbo");
        let cfg: dit::Config = serde_json::from_reader(std::fs::File::open(ck.join("config.json"))?)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[ck.join("model.safetensors")], dtype, device)? };
        let cond = dit::ConditionEncoder::new(&cfg, vb.pp("encoder"))?;
        let dit = dit::Dit::new(&cfg, vb.pp("decoder"))?;
        let detok = dit::Detokenizer::new(&cfg, vb.clone())?;
        let planner_dir = Some(dir.join(PLANNER_DIR)).filter(|d| d.join("model.safetensors").exists());
        // the decoder in f32: its Snake activations and long convolutions lose detail in bf16
        let vbv = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("vae/diffusion_pytorch_model.safetensors")], DType::F32, device)? };
        let vae = oobleck::Decoder::new(vbv.pp("decoder"))?;
        let silence = silence_latent(&ck.join("silence_latent.pt"), device)?.to_dtype(dtype)?;
        Ok(Self { text, cond, dit, vae, encoder: None, vae_path: dir.join("vae/diffusion_pytorch_model.safetensors"), silence, detok,
                  planner: None, planner_dir, thought: None, dtype, device: device.clone() })
    }

    pub fn has_planner(&self) -> bool {
        self.planner_dir.is_some()
    }

    fn ensure_planner(&mut self) -> Result<()> {
        if self.planner.is_none() {
            let d = self.planner_dir.as_ref().ok_or_else(|| anyhow!("the song planner ({PLANNER_DIR}) is not downloaded"))?;
            self.planner = Some(planner::Planner::load(d, &self.device)?);
        }
        Ok(())
    }

    /// What the planner's reasoning filled in for the last song (None: nothing was missing, or no planner).
    pub fn last_thought(&self) -> Option<planner::Thought> {
        self.thought.clone()
    }

    /// The planner's codes for the song (`secs` * 5 of them); `on_code` sees each.
    pub fn plan(&mut self, s: &Song, sampling: &planner::Sampling, secs: usize, on_code: &mut dyn FnMut(usize, usize)) -> Result<Vec<u32>> {
        self.ensure_planner()?;
        // the vocal language in the planner's reasoning for every language but English: measured on this PC
        // (Whisper, lyric words heard in order), 3-minute Hindi songs went from 32% to 60% with it, 60 s English
        // songs from 91% to 76%; ACE-Step itself leaves it out when the other values are all given
        let lang = Some(s.language.as_str()).filter(|l| !l.eq_ignore_ascii_case("en"));
        let (c, u) = planner::prompts(&s.caption, &s.lyrics, &s.metas, lang);
        self.planner.as_mut().unwrap().codes(&c, &u, secs * planner::CODES_PER_SECOND, sampling, on_code)
    }

    /// The song's latent [1, frames, 64]: `on_step` sees each step (done, total): the planner's codes, if it is used,
    /// then the diffusion steps.
    pub fn latent(&mut self, s: &Song, on_step: &mut dyn FnMut(usize, usize)) -> Result<Tensor> {
        self.thought = None;
        let secs = s.metas.duration_s.clamp(MIN_SECONDS, MAX_SECONDS) as usize;
        let frames = secs * FRAMES_PER_SECOND;
        let plan = s.plan.clone().filter(|_| self.has_planner());
        // a song without its tempo, key or time signature: the planner's reasoning writes them first (ACE-Step's phase
        // 1), and both the planner and the DiT get them
        let mut s = s.clone();
        if let Some(sampling) = &plan {
            let m = &s.metas;
            if m.bpm.is_none() || m.key_scale.is_none() || m.time_signature.is_none() {
                self.ensure_planner()?;
                let (t, text) = self.planner.as_mut().unwrap().think(&s.caption, &s.lyrics, sampling)?;
                eprintln!("the planner's reasoning: {t:?} ({} characters)", text.len());
                if s.metas.bpm.is_none() { s.metas.bpm = t.bpm; }
                if s.metas.key_scale.is_none() { s.metas.key_scale = t.key_scale.clone(); }
                if s.metas.time_signature.is_none() { s.metas.time_signature = t.time_signature.clone(); }
                self.thought = Some(t);
            }
        }
        let s = &s;
        let n_codes = if plan.is_some() { secs * planner::CODES_PER_SECOND } else { 0 };
        let total = n_codes + s.steps;
        // the planner's codes as the source (ACE-Step's "cover" of the planned song), else silence
        let src = match &plan {
            Some(sampling) => {
                let codes = self.plan(s, sampling, secs, &mut |i, _| on_step(i, total))?;
                self.detok.hints(&codes, self.dtype, &self.device)?.narrow(1, 0, frames)?
            }
            None => self.silence.narrow(1, 0, frames)?,
        };
        let instruction = if plan.is_some() { text::PLANNED_INSTRUCTION } else { text::DIT_INSTRUCTION };
        let caption = self.text.caption(&text::caption_prompt_for(instruction, &s.caption, &s.metas))?.to_dtype(self.dtype)?;
        let lyrics = self.text.lyrics(&text::lyric_prompt(&s.lyrics, &s.language))?.to_dtype(self.dtype)?;
        let timbre = self.silence.narrow(1, 0, 750)?;
        let cond = self.cond.forward(&caption, &lyrics, &timbre)?;
        // every frame to be made (chunk mask 1)
        let context = Tensor::cat(&[&src, &src.ones_like()?], 2)?;
        let x = noise(s.seed, frames, &self.device)?.to_dtype(self.dtype)?;
        Ok(dit::sample(&self.dit, &x, &context, &cond, &dit::schedule(s.steps, s.shift), &mut |i, _| on_step(n_codes + i, total))?)
    }

    /// Stereo audio [2, samples] at 48 kHz from a latent [1, frames, 64], decoded in overlapping chunks (so a 10-minute
    /// song does not need the whole decoder in memory at once).
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        const CHUNK: usize = 128;
        const OVERLAP: usize = 16;
        let z = latent.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
        let frames = z.dim(2)?;
        let mut parts = Vec::new();
        let mut start = 0;
        while start < frames {
            let end = (start + CHUNK).min(frames);
            let (a, b) = (start.saturating_sub(OVERLAP), (end + OVERLAP).min(frames));
            let audio = self.vae.decode(&z.narrow(2, a, b - a)?)?;
            parts.push(audio.narrow(2, (start - a) * oobleck::HOP, (end - start) * oobleck::HOP)?);
            start = end;
        }
        Ok(Tensor::cat(&parts, 2)?.squeeze(0)?)
    }

    /// The latent [1, frames, 64] of `audio` [2, samples] at 48 kHz (the encoder's mean; samples past a whole frame
    /// are left out).
    pub fn encode(&mut self, audio: &Tensor) -> Result<Tensor> {
        if self.encoder.is_none() {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&self.vae_path], DType::F32, &self.device)? };
            self.encoder = Some(oobleck::Encoder::new(vb.pp("encoder"))?);
        }
        let n = audio.dim(1)? / oobleck::HOP * oobleck::HOP;
        let a = audio.narrow(1, 0, n)?.to_dtype(DType::F32)?.to_device(&self.device)?.unsqueeze(0)?;
        // in pieces of about 20 s with 2 s either side, so a long source does not need the whole encoder at once
        const CHUNK: usize = 500;
        const PAD: usize = 50;
        let frames = n / oobleck::HOP;
        let enc = self.encoder.as_ref().unwrap();
        let mut parts = Vec::new();
        let mut start = 0;
        while start < frames {
            let end = (start + CHUNK).min(frames);
            let (lo, hi) = (start.saturating_sub(PAD), (end + PAD).min(frames));
            let z = enc.encode(&a.narrow(2, lo * oobleck::HOP, (hi - lo) * oobleck::HOP)?)?;
            parts.push(z.narrow(2, start - lo, end - start)?);
            start = end;
        }
        Ok(Tensor::cat(&parts, 2)?.transpose(1, 2)?.contiguous()?.to_dtype(self.dtype)?)
    }

    /// A song continued from `source`, the last seconds of the one before ([2, samples] at 48 kHz): ACE-Step's repaint
    /// run past the end of its source (outpainting). The source is encoded and kept; the frames after it, `s`'s
    /// duration, are made from noise with the source as context (explicit chunk mask: 0 kept, 1 made), the kept frames
    /// pulled back to the noised source for the first half of the steps, and the boundary crossfaded over 12 frames.
    /// The source also gives the timbre when `timbre_from_source` (else silence, as ACE-Step without a reference).
    /// -> the whole latent (source and new part) and the source's length in frames.
    pub fn continue_latent(&mut self, s: &Song, source: &Tensor, timbre_from_source: bool,
                           on_step: &mut dyn FnMut(usize, usize)) -> Result<(Tensor, usize)> {
        let kept = self.encode(source)?;
        let k = kept.dim(1)?;
        let new = s.metas.duration_s.clamp(MIN_SECONDS, MAX_SECONDS) as usize * FRAMES_PER_SECOND;
        let total = k + new;
        if total > self.silence.dim(1)? {
            bail!("the source and the continuation together are past 10 minutes");
        }
        let mut metas = s.metas.clone();
        metas.duration_s = (total / FRAMES_PER_SECOND) as u32;
        let caption = self.text.caption(&text::caption_prompt_for(text::REPAINT_INSTRUCTION, &s.caption, &metas))?.to_dtype(self.dtype)?;
        let lyrics = self.text.lyrics(&text::lyric_prompt(&s.lyrics, &s.language))?.to_dtype(self.dtype)?;
        let timbre = if timbre_from_source { kept.clone() } else { self.silence.narrow(1, 0, 750)? };
        let cond = self.cond.forward(&caption, &lyrics, &timbre)?;
        // with the planner (not in ACE-Step's reference, which skips it for repaint and keeps codes out of a repaint
        // range): its codes for the new part's lyrics and length, detokenized, as the context where frames are made
        let new_secs = new / FRAMES_PER_SECOND;
        let hints = match s.plan.clone().filter(|_| self.has_planner()) {
            Some(sampling) => {
                let mut part = s.clone();
                part.metas.duration_s = new_secs as u32;
                let n_codes = new_secs * planner::CODES_PER_SECOND;
                let codes = self.plan(&part, &sampling, new_secs, &mut |i, _| on_step(i, n_codes + s.steps))?;
                Some(self.detok.hints(&codes, self.dtype, &self.device)?.narrow(1, 0, new)?)
            }
            None => None,
        };
        let (latent, _) = self.repaint_with(&kept, new, hints.as_ref(), &cond, s.seed, s.steps, s.shift, on_step)?;
        Ok((latent, k))
    }

    /// The repaint itself, from conditions already made: `kept` [1, k, 64] then `new` frames made.
    pub fn repaint(&self, kept: &Tensor, new: usize, cond: &Tensor, seed: u64, steps: usize, shift: f32,
                   on_step: &mut dyn FnMut(usize, usize)) -> Result<(Tensor, Tensor)> {
        self.repaint_with(kept, new, None, cond, seed, steps, shift, on_step)
    }

    /// `repaint` with `hints` [1, new, 64] (the planner's detokenized codes) as the context of the frames made
    /// instead of silence; the source frames and the injection are as before.
    #[allow(clippy::too_many_arguments)]
    pub fn repaint_with(&self, kept: &Tensor, new: usize, hints: Option<&Tensor>, cond: &Tensor, seed: u64, steps: usize,
                        shift: f32, on_step: &mut dyn FnMut(usize, usize)) -> Result<(Tensor, Tensor)> {
        let k = kept.dim(1)?;
        let total = k + new;
        // the source, then silence where frames are made (ACE-Step's src_latents and clean_src_latents agree here)
        let clean = Tensor::cat(&[kept, &self.silence.narrow(1, k, new)?], 1)?;
        let src = match hints {
            Some(h) => Tensor::cat(&[kept, &h.to_dtype(self.dtype)?], 1)?,
            None => clean.clone(),
        };
        let m: Vec<f32> = (0..total).map(|i| if i >= k { 1.0 } else { 0.0 }).collect();
        let mask = Tensor::from_vec(m, (1, total, 1), &self.device)?.to_dtype(self.dtype)?;
        let context = Tensor::cat(&[&src, &mask.broadcast_as((1, total, 64))?.contiguous()?], 2)?;
        let x = noise(seed, total, &self.device)?.to_dtype(self.dtype)?;
        let rp = dit::Repaint { mask: &mask, clean: &clean, injection_ratio: 0.5, crossfade: CROSSFADE };
        let latent = dit::sample_repaint(&self.dit, &x, &context, cond, &dit::schedule(steps, shift), Some(&rp), on_step)?;
        Ok((latent, context))
    }

    /// The continuation's audio: the latent decoded whole (so the decoder sees the source), then only the frames from
    /// CROSSFADE before the source's end, which replace the previous song's last CROSSFADE frames (0.48 s).
    pub fn continuation(&mut self, s: &Song, source: &Tensor, timbre_from_source: bool,
                        on_step: &mut dyn FnMut(usize, usize)) -> Result<Tensor> {
        let (latent, k) = self.continue_latent(s, source, timbre_from_source, on_step)?;
        let from = k.saturating_sub(CROSSFADE);
        let audio = self.decode(&latent)?;
        Ok(audio.narrow(1, from * oobleck::HOP, audio.dim(1)? - from * oobleck::HOP)?)
    }

    pub fn song(&mut self, s: &Song, on_step: &mut dyn FnMut(usize, usize)) -> Result<Tensor> {
        let latent = self.latent(s, on_step)?;
        self.decode(&latent)
    }
}

/// A 16-bit stereo WAV of `audio` [2, samples] at 48 kHz.
pub fn wav(audio: &Tensor) -> Result<Vec<u8>> {
    let (c, n) = audio.dims2()?;
    let a = audio.t()?.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(44 + a.len() * 2);
    let data = (a.len() * 2) as u32;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(c as u16).to_le_bytes());
    out.extend_from_slice(&oobleck::SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&(oobleck::SAMPLE_RATE * c as u32 * 2).to_le_bytes());
    out.extend_from_slice(&((c * 2) as u16).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    for x in a {
        out.extend_from_slice(&((x.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    let _ = n;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> std::path::PathBuf {
        std::env::var("ACESTEP_REF").map(Into::into)
            .unwrap_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("Development/ref"))
    }

    #[test]
    fn noise_is_standard_normal_and_the_same_for_a_seed() {
        let a = noise(7, 500, &Device::Cpu).unwrap();
        let v = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 0.02 && (var - 1.0).abs() < 0.03, "mean {mean}, variance {var}");
        assert_eq!(v, noise(7, 500, &Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap());
        assert_ne!(v, noise(8, 500, &Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap());
    }

    /// The golden song (acestep-golden.py) made by this pipeline: every input computed here (text, conditions,
    /// context, timbre), only the noise taken from the recording, so the final latent must match Python's.
    #[test]
    #[ignore = "needs ACE-Step's checkpoints and the golden files (ACESTEP_REF)"]
    fn the_golden_song_is_made_again_from_its_noise() {
        let r = reference();
        let dev = Device::Cpu;
        let mut a = AceStep::load(&r.join("ACE-Step-1.5/checkpoints"), &dev).unwrap();
        let g = |f: &str| candle_core::safetensors::load(r.join(format!("acestep-golden/{f}.safetensors")), &dev).unwrap();
        let (genr, steps, encg) = (g("generate"), g("dit_steps"), g("encoder"));
        let m = text::Metas { bpm: Some(100), time_signature: Some("4".into()), key_scale: Some("C major".into()), duration_s: 10 };
        let caption = a.text.caption(&text::caption_prompt("warm indie pop, female vocal, acoustic guitar, soft drums, hopeful", &m)).unwrap();
        let lyrics = a.text.lyrics(&text::lyric_prompt("[verse]\nLate at night the city hums\nA little light, a server runs", "en")).unwrap();
        let cond = a.cond.forward(&caption, &lyrics, &a.silence.narrow(1, 0, 750).unwrap()).unwrap();
        let cos = |x: &Tensor, y: &Tensor| {
            let (x, y) = (x.flatten_all().unwrap(), y.flatten_all().unwrap());
            let d = (&x * &y).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            d / (x.sqr().unwrap().sum_all().unwrap().sqrt().unwrap().to_scalar::<f32>().unwrap()
                 * y.sqr().unwrap().sum_all().unwrap().sqrt().unwrap().to_scalar::<f32>().unwrap())
        };
        eprintln!("conditions against Python's: cosine {:.5}", cos(&cond, &encg["0.out.0"]));
        let src = a.silence.narrow(1, 0, 250).unwrap();
        let context = Tensor::cat(&[&src, &src.ones_like().unwrap()], 2).unwrap();
        eprintln!("context against Python's: cosine {:.5}", cos(&context, &steps["0.kwargs.context_latents"]));
        let x0 = dit::sample(&a.dit, &steps["0.kwargs.hidden_states"], &context, &cond, &dit::schedule(8, 1.0), &mut |_, _| {}).unwrap();
        let c = cos(&x0, &genr["out.target_latents"]);
        eprintln!("final latent against Python's: cosine {c:.5}");
        std::fs::write(r.join("acestep-golden/rust-golden-song.wav"), wav(&a.decode(&x0).unwrap()).unwrap()).unwrap();
        assert!(c > 0.97, "final latent: cosine {c}");
    }

    /// The golden repaint (acestep-golden-repaint.py): the golden song's first 4 s kept, frames 100..250 made after
    /// it. The conditions are made here; the noise is the recorded one, so the final latent must match Python's.
    #[test]
    #[ignore = "needs ACE-Step's checkpoints and the golden files (ACESTEP_REF)"]
    fn the_golden_repaint_is_made_again_from_its_noise() {
        let r = reference();
        let dev = Device::Cpu;
        let mut a = AceStep::load(&r.join("ACE-Step-1.5/checkpoints"), &dev).unwrap();
        let g = |f: &str| candle_core::safetensors::load(r.join(format!("acestep-golden/{f}.safetensors")), &dev).unwrap();
        let (genr, rp) = (g("generate"), g("repaint"));
        let m = text::Metas { bpm: Some(100), time_signature: Some("4".into()), key_scale: Some("C major".into()), duration_s: 10 };
        let caption = a.text.caption(&text::caption_prompt("warm indie pop, female vocal, acoustic guitar, soft drums, hopeful", &m)).unwrap();
        let lyrics = a.text.lyrics(&text::lyric_prompt("[verse]\nLate at night the city hums\nA little light, a server runs", "en")).unwrap();
        let cond = a.cond.forward(&caption, &lyrics, &a.silence.narrow(1, 0, 750).unwrap()).unwrap();
        let cos = |x: &Tensor, y: &Tensor| {
            let (x, y) = (x.flatten_all().unwrap(), y.flatten_all().unwrap());
            let d = (&x * &y).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            d / (x.sqr().unwrap().sum_all().unwrap().sqrt().unwrap().to_scalar::<f32>().unwrap()
                 * y.sqr().unwrap().sum_all().unwrap().sqrt().unwrap().to_scalar::<f32>().unwrap())
        };
        let kept = genr["out.target_latents"].narrow(1, 0, 100).unwrap();
        // the inputs built here against Python's
        let clean = Tensor::cat(&[&kept, &a.silence.narrow(1, 100, 150).unwrap()], 1).unwrap();
        eprintln!("source against Python's: cosine {:.6}", cos(&clean, &rp["clean_src_latents"]));
        let total = 250;
        let mvec: Vec<f32> = (0..total).map(|i| if i >= 100 { 1.0 } else { 0.0 }).collect();
        let mask = Tensor::from_vec(mvec, (1, total, 1), &dev).unwrap();
        let context = Tensor::cat(&[&clean, &mask.broadcast_as((1, total, 64)).unwrap().contiguous().unwrap()], 2).unwrap();
        let want_ctx = Tensor::cat(&[&rp["src_latents"], &rp["chunk_masks"]], 2).unwrap();
        let cctx = cos(&context, &want_ctx);
        eprintln!("context against Python's: cosine {cctx:.6}");
        let r2 = dit::Repaint { mask: &mask, clean: &clean, injection_ratio: 0.5, crossfade: CROSSFADE };
        let x0 = dit::sample_repaint(&a.dit, &rp["noise"], &context, &cond, &dit::schedule(8, 1.0), Some(&r2), &mut |_, _| {}).unwrap();
        let want = &rp["out"];
        let (c, cnew) = (cos(&x0, want), cos(&x0.narrow(1, 100, 150).unwrap(), &want.narrow(1, 100, 150).unwrap()));
        let kept_err = (x0.narrow(1, 0, 88).unwrap() - kept.narrow(1, 0, 88).unwrap()).unwrap().abs().unwrap()
            .flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        eprintln!("final latent against Python's: cosine {c:.5} (the new part {cnew:.5}); the kept frames off the source by at most {kept_err:.2e}");
        std::fs::write(r.join("acestep-golden/rust-repaint.wav"), wav(&a.decode(&x0).unwrap()).unwrap()).unwrap();
        assert!(cctx > 0.9999 && c > 0.97 && cnew > 0.97 && kept_err < 1e-4, "cosine {c}, new part {cnew}, kept {kept_err}");
    }

    /// A whole song in Rust (set ACESTEP_DEVICE=cuda for the GPU); writes it for listening and Whisper.
    #[test]
    #[ignore = "needs ACE-Step's checkpoints (ACESTEP_REF); writes acestep-golden/rust-song.wav"]
    fn a_song_is_made_from_a_caption_and_lyrics() {
        let r = reference();
        let dev = if std::env::var("ACESTEP_DEVICE").as_deref() == Ok("cuda") { Device::new_cuda(0).unwrap() } else { Device::Cpu };
        let sil = silence_latent(&r.join("ACE-Step-1.5/checkpoints/acestep-v15-turbo/silence_latent.pt"), &Device::Cpu).unwrap();
        assert_eq!(sil.dims(), &[1, 15000, 64]);
        let t0 = std::time::Instant::now();
        let mut a = AceStep::load(&r.join("ACE-Step-1.5/checkpoints"), &dev).unwrap();
        eprintln!("loaded in {:.1} s", t0.elapsed().as_secs_f64());
        let secs: u32 = std::env::var("ACESTEP_SECONDS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
        // the defaults, or a song from ACESTEP_CAPTION, ACESTEP_LYRICS_FILE and ACESTEP_LANG
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let song = Song {
            caption: env("ACESTEP_CAPTION").unwrap_or_else(|| "warm indie pop, female vocal, acoustic guitar, soft drums, hopeful".into()),
            lyrics: env("ACESTEP_LYRICS_FILE").map(|f| std::fs::read_to_string(f).unwrap()).unwrap_or_else(|| {
                "[verse]\nLate at night the city hums\nA little light, a server runs\nNo cloud above, just you and me\nA quiet hum of melody\n\n\
                 [chorus]\nOne command and it begins\nA voice of mine, a song within".into()
            }),
            language: env("ACESTEP_LANG").unwrap_or_else(|| "en".into()),
            metas: if std::env::var("ACESTEP_NO_METAS").is_ok() { text::Metas { duration_s: secs, ..Default::default() } }
                   else { text::Metas { bpm: Some(100), time_signature: Some("4".into()), key_scale: Some("C major".into()), duration_s: secs } },
            seed: std::env::var("ACESTEP_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(42), steps: 8, shift: 1.0,
            plan: (std::env::var("ACESTEP_PLANNER").as_deref() != Ok("0"))
                .then(|| planner::Sampling { seed: 42, ..Default::default() }),
        };
        let t0 = std::time::Instant::now();
        let audio = a.song(&song, &mut |i, n| eprintln!("step {i} of {n} at {:.1} s", t0.elapsed().as_secs_f64())).unwrap();
        let (c, n) = audio.dims2().unwrap();
        eprintln!("{c} channels, {:.1} s of audio in {:.1} s", n as f64 / 48000.0, t0.elapsed().as_secs_f64());
        assert_eq!((c, n), (2, secs as usize * 25 * 1920));
        let peak = audio.abs().unwrap().max_keepdim(1).unwrap().max_keepdim(0).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!(peak > 0.05, "silent: peak {peak}");
        let out = env("ACESTEP_OUT").map(std::path::PathBuf::from).unwrap_or_else(|| r.join(format!("acestep-golden/rust-song-{}.wav", song.seed)));
        std::fs::write(out, wav(&audio).unwrap()).unwrap();
    }
}
