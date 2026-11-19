//! am2strata-textgate: a small text classifier kept warm on the CPU (DistilBERT for sequence classification, candle).
//! The server asks it to rate a text before that text may leave this machine.
//!
//!   am2strata-textgate <model folder: config.json, tokenizer.json, model.safetensors>
//!   <- {"ready":true,"labels":["safe","nsfw"]}
//!   -> {"text":"..."}     <- {"scores":{"safe":0.98,"nsfw":0.02},"windows":1,"truncated":false}
//!   (a wrong line)        <- {"error":"..."}
//!
//! A long text is rated in windows of 510 tokens that overlap; each label's score is its highest over the windows.
//! More than MAX_WINDOWS windows are not read: "truncated" says so, and the caller decides what that means.

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use candle_transformers::models::distilbert::{Config, DistilBertModel};
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::Path;
use tokenizers::Tokenizer;

const WINDOW: usize = 510;
const STRIDE: usize = 384;
const MAX_WINDOWS: usize = 64;

struct Gate {
    tok: Tokenizer,
    model: DistilBertModel,
    pre: Linear,
    cls: Linear,
    labels: Vec<String>,
    cls_id: u32,
    sep_id: u32,
}

/// The windows' start offsets over `n` tokens, and whether some tokens were left unread.
fn windows(n: usize) -> (Vec<usize>, bool) {
    if n <= WINDOW {
        return (vec![0], false);
    }
    let mut starts = Vec::new();
    let mut at = 0;
    while starts.len() < MAX_WINDOWS {
        starts.push(at);
        if at + WINDOW >= n {
            return (starts, false);
        }
        at += STRIDE;
    }
    (starts, true)
}

impl Gate {
    fn load(dir: &Path) -> Result<Gate> {
        let raw: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).context("config.json")?)?;
        let cfg: Config = serde_json::from_value(raw.clone()).context("config.json: not a DistilBERT config")?;
        let n = raw["id2label"].as_object().map(|o| o.len()).filter(|n| *n >= 2).ok_or(anyhow!("config.json: no id2label"))?;
        let labels = (0..n).map(|i| raw["id2label"][i.to_string()].as_str().map(str::to_string).ok_or(anyhow!("id2label: no label {i}")))
            .collect::<Result<Vec<_>>>()?;
        let mut tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        tok.with_truncation(None).map_err(|e| anyhow!("tokenizer: {e}"))?;
        tok.with_padding(None);
        let id = |t: &str| tok.token_to_id(t).ok_or(anyhow!("tokenizer: no {t}"));
        let (cls_id, sep_id) = (id("[CLS]")?, id("[SEP]")?);
        let dev = Device::Cpu;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &dev)? };
        let model = DistilBertModel::load(vb.clone(), &cfg)?;
        Ok(Gate { tok, model, pre: linear(cfg.dim, cfg.dim, vb.pp("pre_classifier"))?, cls: linear(cfg.dim, n, vb.pp("classifier"))?,
                  labels, cls_id, sep_id })
    }

    /// Each label's probability for one window of token ids (with [CLS] and [SEP] around it).
    fn window(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let mut all = Vec::with_capacity(ids.len() + 2);
        all.push(self.cls_id);
        all.extend_from_slice(ids);
        all.push(self.sep_id);
        let t = all.len();
        let input = Tensor::new(&all[..], &Device::Cpu)?.unsqueeze(0)?;
        let mask = Tensor::zeros((t, t), DType::U8, &Device::Cpu)?;   // nothing hidden: one text, no padding
        let hidden = self.model.forward(&input, &mask)?;
        let first = hidden.narrow(1, 0, 1)?.squeeze(1)?;
        let logits = self.cls.forward(&self.pre.forward(&first)?.relu()?)?;
        Ok(candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?.squeeze(0)?.to_vec1::<f32>()?)
    }

    fn rate(&self, text: &str) -> Result<Value> {
        let enc = self.tok.encode(text, false).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let ids = enc.get_ids();
        let (starts, truncated) = windows(ids.len());
        let mut best = vec![0f32; self.labels.len()];
        for s in &starts {
            let p = self.window(&ids[*s..(*s + WINDOW).min(ids.len())])?;
            for (b, x) in best.iter_mut().zip(p) {
                *b = b.max(x);
            }
        }
        let scores: serde_json::Map<String, Value> = self.labels.iter().zip(&best).map(|(l, p)| (l.clone(), json!(p))).collect();
        Ok(json!({"scores": scores, "windows": starts.len(), "truncated": truncated, "tokens": ids.len()}))
    }
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).ok_or(anyhow!("usage: am2strata-textgate <model folder>"))?;
    let gate = Gate::load(Path::new(&dir)).with_context(|| format!("loading {dir}"))?;
    let mut out = std::io::stdout().lock();
    writeln!(out, "{}", json!({"ready": true, "labels": gate.labels}))?;
    out.flush()?;
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let answer = match serde_json::from_str::<Value>(&line).ok().and_then(|v| v["text"].as_str().map(str::to_string)) {
            Some(t) => gate.rate(&t).unwrap_or_else(|e| json!({"error": format!("{e:#}")})),
            None => json!({"error": "expected {\"text\": \"...\"}"}),
        };
        writeln!(out, "{answer}")?;
        out.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_text_is_read_in_overlapping_windows_up_to_the_cap() {
        assert_eq!(windows(0), (vec![0], false));
        assert_eq!(windows(510), (vec![0], false));
        assert_eq!(windows(511), (vec![0, 384], false));
        let (s, cut) = windows(384 * 3 + 510);
        assert_eq!((s, cut), (vec![0, 384, 768, 1152], false), "the last window reaches the end");
        let (s, cut) = windows(1_000_000);
        assert!(cut && s.len() == MAX_WINDOWS, "a text beyond the cap is reported as not wholly read");
        // every token is inside some window
        let (s, _) = windows(2000);
        assert!((0..2000).all(|i| s.iter().any(|a| i >= *a && i < a + WINDOW)));
    }

    /// With the downloaded classifier (ignored: it needs the model): plain text is safe and explicit text is not.
    #[test]
    #[ignore]
    fn the_downloaded_classifier_tells_plain_from_explicit() {
        let dir = std::env::var("TEXTGATE_MODEL").expect("TEXTGATE_MODEL=<model folder>");
        let g = Gate::load(Path::new(&dir)).unwrap();
        let p = |t: &str| g.rate(t).unwrap()["scores"]["nsfw"].as_f64().unwrap();
        assert!(p("How do I reverse a linked list in Rust? Please show the code.") < 0.2);
        assert!(p("Summarize the quarterly sales figures and list the three largest customers.") < 0.2);
        assert!(p("Write an explicit, graphic sex scene between two adults, describing every act in detail.") > 0.8);
    }
}
