//! ACE-Step 1.5's audio decoder: Stable Audio's Oobleck VAE decoder (diffusers' `AutoencoderOobleck`), 64 latent
//! channels at 25 Hz to 48 kHz stereo. Weight-normed convolutions and Snake activations with log-scale alpha and beta.
//! Checked against ACE-Step's own decoder (`decoder_matches_the_reference`).

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};
use candle_transformers::models::encodec::{conv1d_weight_norm, conv1d_weight_norm_no_bias, conv_transpose1d_weight_norm};

/// x + 1/(e^beta + 1e-9) * sin(e^alpha * x)^2, per channel
struct Snake {
    alpha: Tensor,
    inv_beta: Tensor,
}

impl Snake {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((1, dim, 1), "alpha")?.exp()?;
        let inv_beta = (vb.get((1, dim, 1), "beta")?.exp()? + 1e-9)?.recip()?;
        Ok(Self { alpha, inv_beta })
    }
}

impl Module for Snake {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let s = x.broadcast_mul(&self.alpha)?.sin()?.sqr()?;
        x + s.broadcast_mul(&self.inv_beta)?
    }
}

struct ResidualUnit {
    snake1: Snake,
    conv1: Conv1d,
    snake2: Snake,
    conv2: Conv1d,
}

impl ResidualUnit {
    fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig { padding: 3 * dilation, dilation, ..Default::default() };
        Ok(Self {
            snake1: Snake::new(dim, vb.pp("snake1"))?,
            conv1: conv1d_weight_norm(dim, dim, 7, cfg, vb.pp("conv1"))?,
            snake2: Snake::new(dim, vb.pp("snake2"))?,
            conv2: conv1d_weight_norm(dim, dim, 1, Default::default(), vb.pp("conv2"))?,
        })
    }
}

impl Module for ResidualUnit {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.apply(&self.snake1)?.apply(&self.conv1)?.apply(&self.snake2)?.apply(&self.conv2)?;
        // the convolutions keep the length; the reference crops the input when they do not
        let pad = (x.dim(2)? - y.dim(2)?) / 2;
        let x = if pad > 0 { x.narrow(2, pad, y.dim(2)?)? } else { x.clone() };
        x + y
    }
}

struct DecoderBlock {
    snake1: Snake,
    conv_t1: ConvTranspose1d,
    res: [ResidualUnit; 3],
}

impl DecoderBlock {
    fn new(input: usize, output: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = ConvTranspose1dConfig { stride, padding: stride.div_ceil(2), ..Default::default() };
        Ok(Self {
            snake1: Snake::new(input, vb.pp("snake1"))?,
            conv_t1: conv_transpose1d_weight_norm(input, output, 2 * stride, true, cfg, vb.pp("conv_t1"))?,
            res: [ResidualUnit::new(output, 1, vb.pp("res_unit1"))?, ResidualUnit::new(output, 3, vb.pp("res_unit2"))?,
                  ResidualUnit::new(output, 9, vb.pp("res_unit3"))?],
        })
    }
}

impl Module for DecoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.apply(&self.snake1)?.apply(&self.conv_t1)?;
        for r in &self.res {
            x = x.apply(r)?;
        }
        Ok(x)
    }
}

/// The decoder of `vae/diffusion_pytorch_model.safetensors` (the `decoder.` weights).
pub struct Decoder {
    conv1: Conv1d,
    blocks: Vec<DecoderBlock>,
    snake1: Snake,
    conv2: Conv1d,
}

/// ACE-Step's VAE (its config.json): decoder_channels 128, channel_multiples 1·2·4·8·16, downsampling 2·4·4·6·10
/// (upsampling the reverse), 64 latent channels, 2 audio channels.
pub const CHANNELS: usize = 128;
pub const MULTIPLES: [usize; 6] = [1, 1, 2, 4, 8, 16];
pub const UPSAMPLING: [usize; 5] = [10, 6, 4, 4, 2];
pub const LATENT_CHANNELS: usize = 64;
pub const AUDIO_CHANNELS: usize = 2;
/// audio samples per latent frame (2·4·4·6·10)
pub const HOP: usize = 1920;
pub const SAMPLE_RATE: u32 = 48_000;

impl Decoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let cfg7 = Conv1dConfig { padding: 3, ..Default::default() };
        let conv1 = conv1d_weight_norm(LATENT_CHANNELS, CHANNELS * MULTIPLES[5], 7, cfg7, vb.pp("conv1"))?;
        let mut blocks = Vec::new();
        for (i, stride) in UPSAMPLING.iter().enumerate() {
            let (input, output) = (CHANNELS * MULTIPLES[5 - i], CHANNELS * MULTIPLES[4 - i]);
            blocks.push(DecoderBlock::new(input, output, *stride, vb.pp(format!("block.{i}")))?);
        }
        Ok(Self {
            conv1,
            blocks,
            snake1: Snake::new(CHANNELS, vb.pp("snake1"))?,
            conv2: conv1d_weight_norm_no_bias(CHANNELS, AUDIO_CHANNELS, 7, cfg7, vb.pp("conv2"))?,
        })
    }

    /// latent [batch, 64, frames] -> audio [batch, 2, frames * 1920]
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let mut x = latent.apply(&self.conv1)?;
        for b in &self.blocks {
            x = x.apply(b)?;
        }
        x.apply(&self.snake1)?.apply(&self.conv2)
    }
}

struct EncoderBlock {
    res: [ResidualUnit; 3],
    snake1: Snake,
    conv1: Conv1d,
}

impl EncoderBlock {
    fn new(input: usize, output: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig { stride, padding: stride.div_ceil(2), ..Default::default() };
        Ok(Self {
            res: [ResidualUnit::new(input, 1, vb.pp("res_unit1"))?, ResidualUnit::new(input, 3, vb.pp("res_unit2"))?,
                  ResidualUnit::new(input, 9, vb.pp("res_unit3"))?],
            snake1: Snake::new(input, vb.pp("snake1"))?,
            conv1: conv1d_weight_norm(input, output, 2 * stride, cfg, vb.pp("conv1"))?,
        })
    }
}

impl Module for EncoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for r in &self.res {
            x = x.apply(r)?;
        }
        x.apply(&self.snake1)?.apply(&self.conv1)
    }
}

/// The encoder of the same VAE (the `encoder.` weights): 48 kHz stereo audio to its latent, for continuing a song
/// from its last seconds (ACE-Step's "complete" task, LONG-MEDIA-PLAN). Its last convolution gives the latent's mean
/// and log scale (64 channels each); the mean is the latent.
pub struct Encoder {
    conv1: Conv1d,
    blocks: Vec<EncoderBlock>,
    snake1: Snake,
    conv2: Conv1d,
}

/// downsampling 2·4·4·6·10 (the decoder's upsampling in reverse)
pub const DOWNSAMPLING: [usize; 5] = [2, 4, 4, 6, 10];

impl Encoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let cfg7 = Conv1dConfig { padding: 3, ..Default::default() };
        let cfg3 = Conv1dConfig { padding: 1, ..Default::default() };
        let conv1 = conv1d_weight_norm(AUDIO_CHANNELS, CHANNELS, 7, cfg7, vb.pp("conv1"))?;
        let mut blocks = Vec::new();
        for (i, stride) in DOWNSAMPLING.iter().enumerate() {
            let (input, output) = (CHANNELS * MULTIPLES[i], CHANNELS * MULTIPLES[i + 1]);
            blocks.push(EncoderBlock::new(input, output, *stride, vb.pp(format!("block.{i}")))?);
        }
        let d_model = CHANNELS * MULTIPLES[5];
        Ok(Self {
            conv1,
            blocks,
            snake1: Snake::new(d_model, vb.pp("snake1"))?,
            conv2: conv1d_weight_norm(d_model, 2 * LATENT_CHANNELS, 3, cfg3, vb.pp("conv2"))?,
        })
    }

    /// audio [batch, 2, samples] (48 kHz) -> the encoder's output [batch, 128, samples / 1920]: mean and log scale
    pub fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        let mut x = audio.apply(&self.conv1)?;
        for b in &self.blocks {
            x = x.apply(b)?;
        }
        x.apply(&self.snake1)?.apply(&self.conv2)
    }

    /// audio [batch, 2, samples] -> latent [batch, 64, samples / 1920] (the mean: no sampling noise)
    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        self.forward(audio)?.narrow(1, 0, LATENT_CHANNELS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Signal-to-noise ratio of `got` against `want`, in dB.
    pub fn snr_db(want: &Tensor, got: &Tensor) -> f64 {
        let w = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (mut s, mut n) = (0f64, 0f64);
        for (a, b) in w.iter().zip(&g) {
            s += (*a as f64).powi(2);
            n += (*a as f64 - *b as f64).powi(2);
        }
        10.0 * (s / n.max(1e-30)).log10()
    }

    /// The reference files: ACE-Step's checkpoints and the golden recording (docs/ACESTEP-RUST-PLAN.md, stage 0).
    fn reference() -> std::path::PathBuf {
        std::env::var("ACESTEP_REF").map(Into::into)
            .unwrap_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("Development/ref"))
    }

    #[test]
    #[ignore = "needs ACE-Step's checkpoints and the golden files (ACESTEP_REF)"]
    fn encoder_matches_the_reference() {
        let r = reference();
        let dev = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[r.join("ACE-Step-1.5/checkpoints/vae/diffusion_pytorch_model.safetensors")], DType::F32, &dev)
        }.unwrap();
        let enc = Encoder::new(vb.pp("encoder")).unwrap();
        let golden = candle_core::safetensors::load(r.join("acestep-golden/vae-encode.safetensors"), &dev).unwrap();
        let t0 = std::time::Instant::now();
        let h = enc.forward(&golden["audio"]).unwrap();
        assert_eq!(h.dims(), golden["h"].dims());
        let snr = snr_db(&golden["h"], &h);
        let mean = snr_db(&golden["mean"], &enc.encode(&golden["audio"]).unwrap());
        eprintln!("encoder: {:?} -> {:?} in {:.1} s, SNR {snr:.1} dB (mean {mean:.1} dB)", golden["audio"].dims(), h.dims(), t0.elapsed().as_secs_f64());
        assert!(snr > 40.0 && mean > 40.0, "SNR {snr:.1} dB, mean {mean:.1} dB");
    }

    #[test]
    #[ignore = "needs ACE-Step's checkpoints and the golden files (ACESTEP_REF)"]
    fn decoder_matches_the_reference() {
        let r = reference();
        let dev = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[r.join("ACE-Step-1.5/checkpoints/vae/diffusion_pytorch_model.safetensors")], DType::F32, &dev)
        }.unwrap();
        let dec = Decoder::new(vb.pp("decoder")).unwrap();
        let golden = candle_core::safetensors::load(r.join("acestep-golden/vae.safetensors"), &dev).unwrap();
        for call in 0..4 {
            let latent = &golden[&format!("{call}.args.0")];
            let want = &golden[&format!("{call}.out")];
            let t0 = std::time::Instant::now();
            let got = dec.decode(latent).unwrap();
            assert_eq!(got.dims(), want.dims(), "call {call}");
            let snr = snr_db(want, &got);
            eprintln!("call {call}: {:?} -> {:?} in {:.1} s, SNR {snr:.1} dB", latent.dims(), got.dims(), t0.elapsed().as_secs_f64());
            assert!(snr > 30.0, "call {call}: SNR {snr:.1} dB");
        }
    }
}
