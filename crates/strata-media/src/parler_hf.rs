//! Parler-TTS checkpoints saved by newer `parler_tts` / transformers releases (Parler-TTS mini multilingual v1.1,
//! Indic Parler-TTS), loaded into candle's Parler model unchanged.
//!
//! They differ from the checkpoints candle's model was written for (Parler-TTS mini v1) in their names only:
//! - the nine codebook heads are one fused matrix, `decoder.lm_heads.weight` [9 x 1088, 1024]; candle wants
//!   `decoder.lm_heads.{i}.weight`, so row block i is handed out as head i;
//! - the audio codec is transformers' `DacModel`: plain conv weights under module names (`decoder.block.0.conv_t1`,
//!   `res_unit1.snake2`...), where candle's DAC reads descript-audio-codec's weight-normed numbered modules
//!   (`decoder.model.1.block.1.weight_g` / `weight_v`). A plain weight W is handed out as weight_v = W and
//!   weight_g = |W| (the norm over all but the first axis, as candle computes it), which gives W back exactly.
//!
//! Everything else (the T5 description encoder, the decoder layers, the prompt embedding) has the same names.

use candle_core::{safetensors::MmapedSafetensors, DType, Device, Result, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, Init, VarBuilder};

/// What a candle name is made from.
#[derive(Debug, PartialEq)]
enum Source {
    /// the tensor of that name, as it is
    Same(String),
    /// head `i` of the fused heads: rows i*rows .. (i+1)*rows
    Head { name: String, i: usize },
    /// weight_v: the plain weight
    V(String),
    /// weight_g: the norm of the plain weight
    G(String),
}

/// The checkpoint name behind the candle name `name`.
fn source(name: &str, heads_rows: usize) -> Source {
    if let Some(rest) = name.strip_prefix("decoder.lm_heads.") {
        if let Some((i, "weight")) = rest.split_once('.') {
            if let Ok(i) = i.parse::<usize>() {
                let _ = heads_rows;
                return Source::Head { name: "decoder.lm_heads.weight".into(), i };
            }
        }
    }
    let Some(rest) = name.strip_prefix("audio_encoder.model.") else {
        return Source::Same(name.to_string());
    };
    let p: Vec<&str> = rest.split('.').collect();
    // the last part is the parameter; weight_g / weight_v are the weight-normed conv weight
    let (path, leaf) = p.split_at(p.len() - 1);
    let mapped = dac_path(path);
    let base = format!("audio_encoder.{}", mapped.join("."));
    match leaf[0] {
        "weight_v" => Source::V(format!("{base}.weight")),
        "weight_g" => Source::G(format!("{base}.weight")),
        other => Source::Same(format!("{base}.{other}")),
    }
}

/// A residual unit's numbered parts: snake1, conv1, snake2, conv2.
fn res_part(i: &str) -> &'static str {
    match i { "0" => "snake1", "1" => "conv1", "2" => "snake2", _ => "conv2" }
}

/// descript-audio-codec's module path (as candle names it, below `audio_encoder.model.`) in transformers' names.
fn dac_path(p: &[&str]) -> Vec<String> {
    let s = |x: &str| x.to_string();
    match p {
        // encoder: block.0 conv1, block.1-4 encoder blocks, block.5 snake1, block.6 conv2
        ["encoder", "block", "0"] => vec![s("encoder"), s("conv1")],
        ["encoder", "block", "5"] => vec![s("encoder"), s("snake1")],
        ["encoder", "block", "6"] => vec![s("encoder"), s("conv2")],
        ["encoder", "block", b, "block", u @ ("0" | "1" | "2"), "block", part] => {
            let b = b.parse::<usize>().unwrap_or(1) - 1;
            vec![s("encoder"), s("block"), b.to_string(), format!("res_unit{}", u.parse::<usize>().unwrap_or(0) + 1), s(res_part(part))]
        }
        ["encoder", "block", b, "block", "3"] => vec![s("encoder"), s("block"), (b.parse::<usize>().unwrap_or(1) - 1).to_string(), s("snake1")],
        ["encoder", "block", b, "block", "4"] => vec![s("encoder"), s("block"), (b.parse::<usize>().unwrap_or(1) - 1).to_string(), s("conv1")],
        // decoder: model.0 conv1, model.1-4 decoder blocks, model.5 snake1, model.6 conv2
        ["decoder", "model", "0"] => vec![s("decoder"), s("conv1")],
        ["decoder", "model", "5"] => vec![s("decoder"), s("snake1")],
        ["decoder", "model", "6"] => vec![s("decoder"), s("conv2")],
        ["decoder", "model", b, "block", "0"] => vec![s("decoder"), s("block"), (b.parse::<usize>().unwrap_or(1) - 1).to_string(), s("snake1")],
        ["decoder", "model", b, "block", "1"] => vec![s("decoder"), s("block"), (b.parse::<usize>().unwrap_or(1) - 1).to_string(), s("conv_t1")],
        ["decoder", "model", b, "block", u @ ("2" | "3" | "4"), "block", part] => {
            let b = b.parse::<usize>().unwrap_or(1) - 1;
            vec![s("decoder"), s("block"), b.to_string(), format!("res_unit{}", u.parse::<usize>().unwrap_or(2) - 1), s(res_part(part))]
        }
        // the quantizer has the same names
        other => other.iter().map(|x| s(x)).collect(),
    }
}

struct Backend {
    st: MmapedSafetensors,
    /// rows of each codebook head in the fused matrix
    heads_rows: usize,
}

impl SimpleBackend for Backend {
    fn get(&self, s: Shape, name: &str, _: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        let t = self.get_unchecked(name, dtype, dev)?;
        if t.shape() != &s {
            candle_core::bail!("{name}: shape {:?} in the checkpoint, {s:?} expected", t.shape());
        }
        Ok(t)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        let load = |n: &str| self.st.load(n, dev)?.to_dtype(dtype);
        match source(name, self.heads_rows) {
            Source::Same(n) => load(&n),
            Source::Head { name, i } => load(&name)?.narrow(0, i * self.heads_rows, self.heads_rows)?.contiguous(),
            Source::V(n) => load(&n),
            Source::G(n) => {
                let w = load(&n)?;
                let dims: Vec<usize> = (1..w.rank()).collect();
                w.sqr()?.sum_keepdim(dims)?.sqrt()
            }
        }
    }

    fn contains_tensor(&self, name: &str) -> bool {
        match source(name, self.heads_rows) {
            Source::Same(n) | Source::V(n) | Source::G(n) | Source::Head { name: n, .. } => self.st.get(&n).is_ok(),
        }
    }
}

/// A VarBuilder over a newer Parler checkpoint, with candle's names. `heads_rows` is the decoder's vocab_size
/// (codebook size plus its special tokens: 1088).
pub fn var_builder(path: &std::path::Path, heads_rows: usize, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
    let st = unsafe { MmapedSafetensors::new(path)? };
    Ok(VarBuilder::from_backend(Box::new(Backend { st, heads_rows }), dtype, device.clone()))
}

/// candle's DAC config from transformers' `DacConfig` (44.1 kHz, 9 codebooks of 1024, 86 frames a second).
pub fn dac_config(c: &serde_json::Value) -> candle_transformers::models::dac::Config {
    let rate = c["sampling_rate"].as_u64().unwrap_or(44100) as u32;
    let hop = c["hop_length"].as_u64().unwrap_or(512) as u32;
    candle_transformers::models::dac::Config {
        num_codebooks: c["n_codebooks"].as_u64().unwrap_or(9) as usize,
        model_bitrate: 8,
        codebook_size: c["codebook_size"].as_u64().unwrap_or(1024) as usize,
        latent_dim: c["hidden_size"].as_u64().unwrap_or(1024) as usize,
        frame_rate: rate.div_ceil(hop),
        sampling_rate: rate,
    }
}

#[cfg(test)]
mod tests {
    use super::{source, Source};

    fn same(s: &str) -> Source { Source::Same(s.into()) }

    #[test]
    fn candle_names_map_to_the_transformers_checkpoint() {
        assert_eq!(source("decoder.lm_heads.3.weight", 1088), Source::Head { name: "decoder.lm_heads.weight".into(), i: 3 });
        assert_eq!(source("decoder.model.decoder.layers.0.fc1.weight", 1088), same("decoder.model.decoder.layers.0.fc1.weight"));
        assert_eq!(source("embed_prompts.weight", 1088), same("embed_prompts.weight"));
        // encoder
        assert_eq!(source("audio_encoder.model.encoder.block.0.weight_v", 1088), Source::V("audio_encoder.encoder.conv1.weight".into()));
        assert_eq!(source("audio_encoder.model.encoder.block.0.bias", 1088), same("audio_encoder.encoder.conv1.bias"));
        assert_eq!(source("audio_encoder.model.encoder.block.2.block.1.block.3.weight_g", 1088),
                   Source::G("audio_encoder.encoder.block.1.res_unit2.conv2.weight".into()));
        assert_eq!(source("audio_encoder.model.encoder.block.4.block.3.alpha", 1088), same("audio_encoder.encoder.block.3.snake1.alpha"));
        assert_eq!(source("audio_encoder.model.encoder.block.4.block.4.bias", 1088), same("audio_encoder.encoder.block.3.conv1.bias"));
        assert_eq!(source("audio_encoder.model.encoder.block.5.alpha", 1088), same("audio_encoder.encoder.snake1.alpha"));
        assert_eq!(source("audio_encoder.model.encoder.block.6.weight_v", 1088), Source::V("audio_encoder.encoder.conv2.weight".into()));
        // decoder
        assert_eq!(source("audio_encoder.model.decoder.model.0.weight_v", 1088), Source::V("audio_encoder.decoder.conv1.weight".into()));
        assert_eq!(source("audio_encoder.model.decoder.model.1.block.0.alpha", 1088), same("audio_encoder.decoder.block.0.snake1.alpha"));
        assert_eq!(source("audio_encoder.model.decoder.model.1.block.1.weight_g", 1088), Source::G("audio_encoder.decoder.block.0.conv_t1.weight".into()));
        assert_eq!(source("audio_encoder.model.decoder.model.4.block.4.block.2.alpha", 1088),
                   same("audio_encoder.decoder.block.3.res_unit3.snake2.alpha"));
        assert_eq!(source("audio_encoder.model.decoder.model.5.alpha", 1088), same("audio_encoder.decoder.snake1.alpha"));
        assert_eq!(source("audio_encoder.model.decoder.model.6.bias", 1088), same("audio_encoder.decoder.conv2.bias"));
        // quantizer
        assert_eq!(source("audio_encoder.model.quantizer.quantizers.2.in_proj.weight_v", 1088),
                   Source::V("audio_encoder.quantizer.quantizers.2.in_proj.weight".into()));
        assert_eq!(source("audio_encoder.model.quantizer.quantizers.2.codebook.weight", 1088),
                   same("audio_encoder.quantizer.quantizers.2.codebook.weight"));
    }
}
