//! What ACE-Step 1.5's DiT is conditioned on: the caption prompt through the caption encoder (Qwen3-Embedding-0.6B,
//! all 28 layers), and the lyric prompt through that model's token table only. The prompt formats are ACE-Step's
//! (`SFT_GEN_PROMPT`, `_format_lyrics`, `_dict_to_meta_string`); the tokenizer adds the closing <|endoftext|> itself.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, VarBuilder};
use candle_transformers::models::qwen3;
use std::path::Path;
use tokenizers::Tokenizer;

pub const DIT_INSTRUCTION: &str = "Fill the audio semantic mask based on the given conditions:";
/// caption and lyric prompts are cut to these many tokens, as ACE-Step does
pub const MAX_CAPTION_TOKENS: usize = 256;
pub const MAX_LYRIC_TOKENS: usize = 2048;

/// The song's musical facts; None is written "N/A", as ACE-Step does.
#[derive(Clone, Debug, Default)]
pub struct Metas {
    pub bpm: Option<u32>,
    /// beats per bar: "2", "3", "4" or "6"
    pub time_signature: Option<String>,
    /// e.g. "C major", "A minor"
    pub key_scale: Option<String>,
    pub duration_s: u32,
}

impl Metas {
    pub fn text(&self) -> String {
        let na = |v: Option<String>| v.filter(|s| !s.trim().is_empty()).unwrap_or_else(|| "N/A".into());
        format!("- bpm: {}\n- timesignature: {}\n- keyscale: {}\n- duration: {} seconds\n",
                na(self.bpm.map(|b| b.to_string())), na(self.time_signature.clone()), na(self.key_scale.clone()), self.duration_s)
    }
}

/// The caption encoder's prompt.
pub fn caption_prompt(caption: &str, metas: &Metas) -> String {
    format!("# Instruction\n{DIT_INSTRUCTION}\n\n# Caption\n{caption}\n\n# Metas\n{}<|endoftext|>\n", metas.text())
}

/// The lyric prompt: the language code (as ACE-Step names it, e.g. "en", "hi") and the lyrics with their structure tags.
pub fn lyric_prompt(lyrics: &str, language: &str) -> String {
    format!("# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>")
}

pub struct TextEncoder {
    /// the encoder is built for each caption (once a song): candle's Qwen3 keeps a KV cache it does not let us clear
    cfg: qwen3::Config,
    vb: VarBuilder<'static>,
    embed: Embedding,
    tok: Tokenizer,
    device: Device,
}

impl TextEncoder {
    /// `dir`: ACE-Step's `Qwen3-Embedding-0.6B` folder (config.json, model.safetensors, tokenizer.json).
    pub fn load(dir: &Path, dtype: DType, device: &Device) -> Result<Self> {
        let cfg: qwen3::Config = serde_json::from_reader(std::fs::File::open(dir.join("config.json"))?)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], dtype, device)? };
        let embed = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        // candle's Qwen3 reads "model.layers...", the embedding checkpoint has no "model." prefix
        let vb = vb.rename_f(|n: &str| n.strip_prefix("model.").unwrap_or(n).to_string());
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
        Ok(Self { cfg, vb, embed, tok, device: device.clone() })
    }

    pub fn ids(&self, text: &str, max: usize) -> Result<Vec<u32>> {
        let mut ids = self.tok.encode(text, true).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
        ids.truncate(max);
        Ok(ids)
    }

    /// The caption prompt's hidden states [1, tokens, 1024] (the encoder's last layer, after its norm).
    pub fn caption(&mut self, prompt: &str) -> Result<Tensor> {
        let ids = self.ids(prompt, MAX_CAPTION_TOKENS)?;
        let mut model = qwen3::Model::new(&self.cfg, self.vb.clone())?;
        Ok(model.forward(&Tensor::new(ids, &self.device)?.unsqueeze(0)?, 0)?)
    }

    /// The lyric prompt's token embeddings [1, tokens, 1024].
    pub fn lyrics(&self, prompt: &str) -> Result<Tensor> {
        let ids = self.ids(prompt, MAX_LYRIC_TOKENS)?;
        Ok(self.embed.forward(&Tensor::new(ids, &self.device)?.unsqueeze(0)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> std::path::PathBuf {
        std::env::var("ACESTEP_REF").map(Into::into)
            .unwrap_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("Development/ref"))
    }

    #[test]
    fn prompts_are_written_as_acestep_writes_them() {
        let m = Metas { bpm: Some(100), time_signature: Some("4".into()), key_scale: Some("C major".into()), duration_s: 10 };
        assert_eq!(caption_prompt("warm pop", &m),
            "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n# Caption\nwarm pop\n\n# Metas\n\
             - bpm: 100\n- timesignature: 4\n- keyscale: C major\n- duration: 10 seconds\n<|endoftext|>\n");
        assert_eq!(Metas { duration_s: 30, ..Default::default() }.text(),
                   "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 30 seconds\n");
        assert_eq!(lyric_prompt("[verse]\nla", "hi"), "# Languages\nhi\n\n# Lyric\n[verse]\nla<|endoftext|>");
    }

    #[test]
    #[ignore = "needs ACE-Step's checkpoints and the golden files (ACESTEP_REF)"]
    fn caption_and_lyrics_match_the_reference() {
        let r = reference();
        let dev = Device::Cpu;
        let mut enc = TextEncoder::load(&r.join("ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B"), DType::F32, &dev).unwrap();
        let g = candle_core::safetensors::load(r.join("acestep-golden/text_encoder.safetensors"), &dev).unwrap();
        let gen = candle_core::safetensors::load(r.join("acestep-golden/generate.safetensors"), &dev).unwrap();
        // the recorded song (acestep-golden.py)
        let m = Metas { bpm: Some(100), time_signature: Some("4".into()), key_scale: Some("C major".into()), duration_s: 10 };
        let prompt = caption_prompt("warm indie pop, female vocal, acoustic guitar, soft drums, hopeful", &m);
        let want_ids: Vec<i64> = g["0.kwargs.input_ids"].flatten_all().unwrap().to_vec1().unwrap();
        let ids: Vec<i64> = enc.ids(&prompt, MAX_CAPTION_TOKENS).unwrap().into_iter().map(i64::from).collect();
        assert_eq!(ids, want_ids, "caption token ids");
        let max_diff = |a: &Tensor, b: &Tensor| (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        let cap = enc.caption(&prompt).unwrap();
        let want = &g["0.out.last_hidden_state"];
        let d = max_diff(&cap, want);
        let scale = want.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        eprintln!("caption: {:?}, max |diff| {d:.4} (largest value {scale:.2})", cap.dims());
        // per token: cosine similarity of the two hidden states
        let (a, b) = (cap.squeeze(0).unwrap(), want.squeeze(0).unwrap());
        let dot = (&a * &b).unwrap().sum(1).unwrap();
        let na = a.sqr().unwrap().sum(1).unwrap().sqrt().unwrap();
        let nb = b.sqr().unwrap().sum(1).unwrap().sqrt().unwrap();
        let cos: Vec<f32> = (dot / (na * nb).unwrap()).unwrap().to_vec1().unwrap();
        let worst = cos.iter().cloned().fold(1f32, f32::min);
        eprintln!("caption cosine per token: worst {worst:.5}, first {:?}", &cos[..6.min(cos.len())]);
        assert!(worst > 0.999, "caption hidden states differ: worst cosine {worst}");
        let lyr = enc.lyrics(&lyric_prompt("[verse]\nLate at night the city hums\nA little light, a server runs", "en")).unwrap();
        let want = &gen["in.lyric_hidden_states"];
        assert_eq!(lyr.dims(), want.dims());
        let d = max_diff(&lyr, want);
        eprintln!("lyrics: {:?}, max |diff| {d:.5}", lyr.dims());
        assert!(d < 1e-2, "lyric embeddings differ: {d}");
    }
}
