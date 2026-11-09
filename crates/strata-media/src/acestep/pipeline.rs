//! A song from a caption and lyrics, all in Rust: the caption encoder and lyric embeddings (text.rs), the condition
//! encoder and the turbo DiT (dit.rs), the Oobleck decoder (oobleck.rs). Without the song planner (ACE-Step's
//! low-VRAM mode): the DiT sings the given lyrics in the given style.

use super::{dit, oobleck, text};
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

pub struct AceStep {
    text: text::TextEncoder,
    cond: dit::ConditionEncoder,
    dit: dit::Dit,
    vae: oobleck::Decoder,
    silence: Tensor,
    dtype: DType,
    device: Device,
}

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
        // the decoder in f32: its Snake activations and long convolutions lose detail in bf16
        let vbv = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("vae/diffusion_pytorch_model.safetensors")], DType::F32, device)? };
        let vae = oobleck::Decoder::new(vbv.pp("decoder"))?;
        let silence = silence_latent(&ck.join("silence_latent.pt"), device)?.to_dtype(dtype)?;
        Ok(Self { text, cond, dit, vae, silence, dtype, device: device.clone() })
    }

    /// The song's latent [1, frames, 64]: `on_step` sees each diffusion step (done, total).
    pub fn latent(&mut self, s: &Song, on_step: &mut dyn FnMut(usize, usize)) -> Result<Tensor> {
        let secs = s.metas.duration_s.clamp(MIN_SECONDS, MAX_SECONDS) as usize;
        let frames = secs * FRAMES_PER_SECOND;
        let caption = self.text.caption(&text::caption_prompt(&s.caption, &s.metas))?.to_dtype(self.dtype)?;
        let lyrics = self.text.lyrics(&text::lyric_prompt(&s.lyrics, &s.language))?.to_dtype(self.dtype)?;
        let timbre = self.silence.narrow(1, 0, 750)?;
        let cond = self.cond.forward(&caption, &lyrics, &timbre)?;
        // text to song: the source is silence, every frame to be made (chunk mask 1)
        let src = self.silence.narrow(1, 0, frames)?;
        let context = Tensor::cat(&[&src, &src.ones_like()?], 2)?;
        let x = noise(s.seed, frames, &self.device)?.to_dtype(self.dtype)?;
        Ok(dit::sample(&self.dit, &x, &context, &cond, &dit::schedule(s.steps, s.shift), on_step)?)
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
        let song = Song {
            caption: "warm indie pop, female vocal, acoustic guitar, soft drums, hopeful".into(),
            lyrics: "[verse]\nLate at night the city hums\nA little light, a server runs\nNo cloud above, just you and me\nA quiet hum of melody\n\n\
                     [chorus]\nOne command and it begins\nA voice of mine, a song within".into(),
            language: "en".into(),
            metas: text::Metas { bpm: Some(100), time_signature: Some("4".into()), key_scale: Some("C major".into()), duration_s: secs },
            seed: std::env::var("ACESTEP_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(42), steps: 8, shift: 1.0,
        };
        let t0 = std::time::Instant::now();
        let audio = a.song(&song, &mut |i, n| eprintln!("step {i} of {n} at {:.1} s", t0.elapsed().as_secs_f64())).unwrap();
        let (c, n) = audio.dims2().unwrap();
        eprintln!("{c} channels, {:.1} s of audio in {:.1} s", n as f64 / 48000.0, t0.elapsed().as_secs_f64());
        assert_eq!((c, n), (2, secs as usize * 25 * 1920));
        let peak = audio.abs().unwrap().max_keepdim(1).unwrap().max_keepdim(0).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!(peak > 0.05, "silent: peak {peak}");
        std::fs::write(r.join(format!("acestep-golden/rust-song-{}.wav", song.seed)), wav(&audio).unwrap()).unwrap();
    }
}
