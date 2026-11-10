//! ACE-Step 1.5's song planner (`acestep-5Hz-lm-1.7B`, a Qwen3 1.7B with 65,535 audio-code tokens), as its
//! `llm_inference.py` runs it when the song's metadata is given: one prompt with the caption, the lyrics and the
//! metadata as its reasoning, and the audio codes written after it, 5 a second, each chosen with classifier-free
//! guidance against an unconditional prompt ("NO USER INPUT", empty reasoning) and only among the code tokens. The
//! codes, through the FSQ and the detokenizer (dit.rs), set the song's structure for the DiT: without them its vocals
//! follow the lyrics only sometimes.

use super::{dit, text::Metas};
use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3;
use std::path::Path;
use tokenizers::Tokenizer;

/// The token of `<|audio_code_0|>`; code c is this plus c.
pub const FIRST_CODE_TOKEN: u32 = 151_669;
/// Codes a second (the planner's 5 Hz).
pub const CODES_PER_SECOND: usize = 5;
pub const INSTRUCTION: &str = "Generate audio semantic tokens based on the given conditions:";
pub const NEGATIVE_PROMPT: &str = "NO USER INPUT";

/// How codes are chosen; ACE-Step's defaults: temperature 0.85, guidance 2.0, top-p 0.9.
#[derive(Clone, Debug)]
pub struct Sampling {
    /// 0: the most likely code each time (greedy)
    pub temperature: f32,
    pub cfg_scale: f32,
    /// 1.0: off
    pub top_p: f32,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self { temperature: 0.85, cfg_scale: 2.0, top_p: 0.9, seed: 0 }
    }
}

/// The planner's reasoning: the given metadata as YAML with sorted keys (ACE-Step's `_format_metadata_as_cot`, its
/// training format). A missing value is left out.
pub fn reasoning(m: &Metas, language: Option<&str>) -> String {
    let mut lines = Vec::new();
    if let Some(b) = m.bpm {
        lines.push(format!("bpm: {b}"));
    }
    lines.push(format!("duration: {}", m.duration_s));
    if let Some(k) = m.key_scale.as_deref().map(str::trim).filter(|k| !k.is_empty()) {
        lines.push(format!("keyscale: {}", yaml_scalar(k)));
    }
    if let Some(l) = language.map(str::trim).filter(|l| !l.is_empty()) {
        lines.push(format!("language: {}", yaml_scalar(l)));
    }
    if let Some(t) = m.time_signature.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
        lines.push(format!("timesignature: {}", yaml_scalar(t.strip_suffix("/4").unwrap_or(t))));
    }
    format!("<think>\n{}\n</think>", lines.join("\n"))
}

/// A string as PyYAML writes it: plain when it reads back as the same string, else in single quotes.
fn yaml_scalar(s: &str) -> String {
    let plain = !s.is_empty()
        && !s.starts_with(|c: char| "-?:,[]{}#&*!|>'\"%@`".contains(c) || c.is_whitespace())
        && !s.ends_with(char::is_whitespace)
        && !s.contains(": ") && !s.contains(" #")
        && !matches!(s.to_ascii_lowercase().as_str(), "true" | "false" | "yes" | "no" | "on" | "off" | "null" | "~");
    if plain { s.to_string() } else { format!("'{}'", s.replace('\'', "''")) }
}

fn chat(user: &str, reasoning: &str) -> String {
    format!("<|im_start|>system\n# Instruction\n{INSTRUCTION}\n\n<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n\
             <|im_start|>assistant\n{reasoning}\n\n")
}

/// The conditional and the unconditional prompts (`build_formatted_prompt_with_cot`); `language` in the reasoning
/// when given (ACE-Step leaves it out when every other value is given; its training format has it).
pub fn prompts(caption: &str, lyrics: &str, m: &Metas, language: Option<&str>) -> (String, String) {
    (chat(&format!("# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"), &reasoning(m, language)), chat(NEGATIVE_PROMPT, "<think>\n\n</think>"))
}

/// What the planner's reasoning gave for a song without them (ACE-Step's phase 1, `parse_lm_output`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Thought {
    pub bpm: Option<u32>,
    pub key_scale: Option<String>,
    pub time_signature: Option<String>,
}

/// The reasoning's `key: value` lines (a value may go on over indented lines): the tempo, key and time signature, each
/// kept only when it reads as one (30-300 bpm; "<note> major/minor"; 2, 3, 4 or 6 beats).
pub fn parse_thought(text: &str) -> Thought {
    let body = text.split("</think>").next().unwrap_or(text);
    let body = body.rsplit("<think>").next().unwrap_or(body);
    let mut t = Thought::default();
    for line in body.lines() {
        let Some((k, v)) = line.split_once(':') else { continue };
        if line.starts_with(char::is_whitespace) {
            continue;
        }
        let v = v.trim().trim_matches(|c| c == '\'' || c == '"');
        match k.trim().to_ascii_lowercase().as_str() {
            "bpm" => t.bpm = v.parse::<f64>().ok().map(|b| b.round() as u32).filter(|b| (30..=300).contains(b)),
            "keyscale" => {
                let ok = {
                    let mut w = v.split_whitespace();
                    let note = w.next().unwrap_or("");
                    let mode = w.next().unwrap_or("").to_ascii_lowercase();
                    let n = note.as_bytes();
                    !n.is_empty() && (b'A'..=b'G').contains(&n[0].to_ascii_uppercase())
                        && note.len() <= 2 && matches!(mode.as_str(), "major" | "minor") && w.next().is_none()
                };
                if ok {
                    t.key_scale = Some(v.to_string());
                }
            }
            "timesignature" => {
                let n = v.split('/').next().unwrap_or("").trim();
                if matches!(n, "2" | "3" | "4" | "6") {
                    t.time_signature = Some(n.to_string());
                }
            }
            _ => {}
        }
    }
    t
}

pub struct Planner {
    model: qwen3::ModelForCausalLM,
    tok: Tokenizer,
    device: Device,
}

impl Planner {
    /// `dir`: ACE-Step's `acestep-5Hz-lm-1.7B` folder (config.json, model.safetensors, tokenizer.json). bf16 on a GPU,
    /// f32 on the CPU.
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let dtype = if device.is_cpu() { DType::F32 } else { DType::BF16 };
        let cfg: qwen3::Config = serde_json::from_reader(std::fs::File::open(dir.join("config.json"))?)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], dtype, device)? };
        // candle's Qwen3 reads "model.layers...", the planner's checkpoint has no "model." prefix
        let vb = vb.rename_f(|n: &str| n.strip_prefix("model.").unwrap_or(n).to_string());
        let model = qwen3::ModelForCausalLM::new(&cfg, vb)?;
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
        Ok(Self { model, tok, device: device.clone() })
    }

    pub fn ids(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(text, true).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec())
    }

    /// The code logits [65535 is the vocabulary's code range; the valid codes are the first 64,000] after `ids`.
    fn code_logits(model: &mut qwen3::ModelForCausalLM, ids: &[u32], offset: usize, dev: &Device) -> Result<Vec<f32>> {
        let x = Tensor::new(ids, dev)?.unsqueeze(0)?;
        let logits = model.forward(&x, offset)?.flatten_all()?;
        Ok(logits.narrow(0, FIRST_CODE_TOKEN as usize, dit::CODEBOOK_SIZE as usize)?.to_dtype(DType::F32)?.to_vec1()?)
    }

    /// The guided logits before each of `codes`, with `codes` themselves as the history (teacher forcing: for
    /// checking the planner against another implementation's codes).
    pub fn forced(&mut self, cond: &str, uncond: &str, codes: &[u32], cfg_scale: f32) -> Result<Vec<Vec<f32>>> {
        let (ci, ui) = (self.ids(cond)?, self.ids(uncond)?);
        self.model.clear_kv_cache();
        let mut c_model = self.model.clone();
        let mut u_model = self.model.clone();
        let mut c = Self::code_logits(&mut c_model, &ci, 0, &self.device)?;
        let mut u = Self::code_logits(&mut u_model, &ui, 0, &self.device)?;
        let (mut c_off, mut u_off) = (ci.len(), ui.len());
        let mut out = Vec::new();
        for (i, &code) in codes.iter().enumerate() {
            out.push(c.iter().zip(&u).map(|(c, u)| u + cfg_scale * (c - u)).collect());
            if i + 1 == codes.len() {
                break;
            }
            let t = [FIRST_CODE_TOKEN + code];
            c = Self::code_logits(&mut c_model, &t, c_off, &self.device)?;
            u = Self::code_logits(&mut u_model, &t, u_off, &self.device)?;
            c_off += 1;
            u_off += 1;
        }
        Ok(out)
    }

    /// The planner's own reasoning for a song (ACE-Step's phase 1, without guidance): written after the caption and
    /// lyrics until `</think>` (at most 400 tokens), sampled among the text tokens only; its tempo, key and time signature.
    pub fn think(&mut self, caption: &str, lyrics: &str, s: &Sampling) -> Result<(Thought, String)> {
        let prompt = format!("<|im_start|>system\n# Instruction\n{INSTRUCTION}\n\n<|im_end|>\n<|im_start|>user\n\
                              # Caption\n{caption}\n\n# Lyric\n{lyrics}\n<|im_end|>\n<|im_start|>assistant\n");
        let ids = self.ids(&prompt)?;
        self.model.clear_kv_cache();
        let mut m = self.model.clone();
        let text_vocab = FIRST_CODE_TOKEN as usize;
        let logits = |m: &mut qwen3::ModelForCausalLM, ids: &[u32], off: usize, dev: &Device| -> Result<Vec<f32>> {
            let x = Tensor::new(ids, dev)?.unsqueeze(0)?;
            Ok(m.forward(&x, off)?.flatten_all()?.narrow(0, 0, text_vocab)?.to_dtype(DType::F32)?.to_vec1()?)
        };
        let mut l = logits(&mut m, &ids, 0, &self.device)?;
        let mut off = ids.len();
        let mut rng = SplitMix(s.seed ^ 0x5DEE_CE66_D1CE_4E5B);
        let mut out: Vec<u32> = Vec::new();
        for _ in 0..400 {
            let t = choose(&l, s, &mut rng);
            if t == 151_645 || t == 151_643 {
                break; // <|im_end|>, <|endoftext|>
            }
            out.push(t);
            let so_far = self.tok.decode(&out, false).map_err(|e| anyhow!("{e}"))?;
            if so_far.contains("</think>") {
                break;
            }
            l = logits(&mut m, &[t], off, &self.device)?;
            off += 1;
        }
        let text = self.tok.decode(&out, false).map_err(|e| anyhow!("{e}"))?;
        Ok((parse_thought(&text), text))
    }

    /// `n` codes for the prompts; `on_code` sees each (done, n).
    pub fn codes(&mut self, cond: &str, uncond: &str, n: usize, s: &Sampling, on_code: &mut dyn FnMut(usize, usize)) -> Result<Vec<u32>> {
        let (ci, ui) = (self.ids(cond)?, self.ids(uncond)?);
        // two KV caches over the same weights (the model's tensors are shared by clone)
        self.model.clear_kv_cache();
        let mut c_model = self.model.clone();
        let mut u_model = self.model.clone();
        let mut c = Self::code_logits(&mut c_model, &ci, 0, &self.device)?;
        let mut u = Self::code_logits(&mut u_model, &ui, 0, &self.device)?;
        let (mut c_off, mut u_off) = (ci.len(), ui.len());
        let mut rng = SplitMix(s.seed);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let g: Vec<f32> = c.iter().zip(&u).map(|(c, u)| u + s.cfg_scale * (c - u)).collect();
            let code = choose(&g, s, &mut rng);
            out.push(code);
            on_code(i + 1, n);
            if i + 1 == n {
                break;
            }
            let t = [FIRST_CODE_TOKEN + code];
            c = Self::code_logits(&mut c_model, &t, c_off, &self.device)?;
            u = Self::code_logits(&mut u_model, &t, u_off, &self.device)?;
            c_off += 1;
            u_off += 1;
        }
        Ok(out)
    }
}

/// A code from guided logits: the most likely at temperature 0, else nucleus sampling (top-p on the logits as they
/// are, then the temperature), as ACE-Step's sampler does it.
fn choose(logits: &[f32], s: &Sampling, rng: &mut SplitMix) -> u32 {
    let argmax = || logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32;
    if s.temperature <= 0.0 {
        return argmax();
    }
    let mut idx: Vec<usize> = (0..logits.len()).filter(|&i| logits[i].is_finite()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    if idx.is_empty() {
        return argmax();
    }
    if s.top_p < 1.0 {
        let max = logits[idx[0]];
        let p: Vec<f64> = idx.iter().map(|&i| ((logits[i] - max) as f64).exp()).collect();
        let total: f64 = p.iter().sum();
        let mut acc = 0.0;
        let mut keep = idx.len();
        for (k, q) in p.iter().enumerate() {
            // keep tokens until the mass before this one reaches top_p (the first always stays)
            if acc / total > s.top_p as f64 && k > 0 {
                keep = k;
                break;
            }
            acc += q;
        }
        idx.truncate(keep);
    }
    let max = logits[idx[0]];
    let w: Vec<f64> = idx.iter().map(|&i| (((logits[i] - max) / s.temperature) as f64).exp()).collect();
    let total: f64 = w.iter().sum();
    let mut r = rng.next() * total;
    for (k, &q) in w.iter().enumerate() {
        r -= q;
        if r <= 0.0 {
            return idx[k] as u32;
        }
    }
    idx[idx.len() - 1] as u32
}

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> std::path::PathBuf {
        std::env::var("ACESTEP_REF").map(Into::into)
            .unwrap_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("Development/ref"))
    }

    fn golden_metas() -> Metas {
        Metas { bpm: Some(100), time_signature: Some("4".into()), key_scale: Some("C major".into()), duration_s: 10 }
    }

    const CAPTION: &str = "warm indie pop, female vocal, acoustic guitar, soft drums, hopeful";
    const LYRICS: &str = "[verse]\nLate at night the city hums\nA little light, a server runs";

    fn golden() -> serde_json::Value {
        serde_json::from_reader(std::fs::File::open(reference().join("acestep-golden/planner/prompts.json")).unwrap()).unwrap()
    }

    #[test]
    fn reasoning_is_sorted_yaml_without_missing_values() {
        assert_eq!(reasoning(&golden_metas(), None), "<think>\nbpm: 100\nduration: 10\nkeyscale: C major\ntimesignature: 4\n</think>");
        assert_eq!(reasoning(&Metas { duration_s: 30, time_signature: Some("3/4".into()), ..Default::default() }, Some("hi")),
                   "<think>\nduration: 30\nlanguage: hi\ntimesignature: 3\n</think>");
        assert_eq!(yaml_scalar("yes"), "'yes'");
        assert_eq!(yaml_scalar("F# minor"), "F# minor");
    }

    #[test]
    fn the_reasoning_s_values_are_read_and_checked() {
        let t = parse_thought("<think>\nbpm: 96\ncaption: a warm pop song with\n  acoustic guitar\nduration: 60\nkeyscale: F# minor\n\
                               language: en\ntimesignature: 4\n</think>");
        assert_eq!(t, Thought { bpm: Some(96), key_scale: Some("F# minor".into()), time_signature: Some("4".into()) });
        let bad = parse_thought("bpm: 900\nkeyscale: H dorian\ntimesignature: 7/8");
        assert_eq!(bad, Thought::default());
        assert_eq!(parse_thought("timesignature: '3/4'").time_signature.as_deref(), Some("3"));
    }

    #[test]
    fn choose_is_greedy_at_temperature_zero_and_samples_within_top_p() {
        let l = [0.0f32, 5.0, 4.9, -10.0];
        let mut r = SplitMix(1);
        assert_eq!(choose(&l, &Sampling { temperature: 0.0, ..Default::default() }, &mut r), 1);
        let s = Sampling { temperature: 1.0, top_p: 0.5, ..Default::default() };
        // top-p 0.5: code 1 alone holds about half the mass, so code 2 may still be kept; 0 and 3 never
        for _ in 0..200 {
            assert!(matches!(choose(&l, &s, &mut r), 1 | 2));
        }
    }

    #[test]
    #[ignore = "needs ACE-Step's golden files (acestep-golden-planner.py)"]
    fn prompts_are_acestep_s() {
        let g = golden();
        let (c, u) = prompts(CAPTION, LYRICS, &golden_metas(), None);
        assert_eq!(c, g["prompts"][0]["text"].as_str().unwrap());
        assert_eq!(u, g["prompts"][1]["text"].as_str().unwrap());
    }

    /// The planner, greedy, against ACE-Step's own greedy codes for the golden song (CPU, f32; Python ran bf16 on a
    /// GPU, so a near tie may go the other way and the codes after it differ).
    #[test]
    #[ignore = "needs ACE-Step's checkpoints and golden files; about a minute on the CPU"]
    fn greedy_codes_match_the_reference() {
        let r = reference();
        let g = golden();
        let dev = Device::Cpu;
        let mut p = Planner::load(&r.join("ACE-Step-1.5/checkpoints/acestep-5Hz-lm-1.7B"), &dev).unwrap();
        let (c, u) = prompts(CAPTION, LYRICS, &golden_metas(), None);
        let want_ids: Vec<u32> = g["prompts"][0]["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(p.ids(&c).unwrap(), want_ids, "conditional prompt ids");
        let want_u: Vec<u32> = g["prompts"][1]["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(p.ids(&u).unwrap(), want_u, "unconditional prompt ids");
        let want: Vec<u32> = g["codes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let t0 = std::time::Instant::now();
        let got = p.codes(&c, &u, want.len(), &Sampling { temperature: 0.0, ..Default::default() }, &mut |_, _| {}).unwrap();
        // teacher-forced: where ACE-Step's code ranks in this planner's guided logits, its history being ACE-Step's
        let forced = p.forced(&c, &u, &want, 2.0).unwrap();
        let ranks: Vec<usize> = forced.iter().zip(&want).map(|(l, &w)| l.iter().filter(|&&x| x > l[w as usize]).count()).collect();
        let top1 = ranks.iter().filter(|&&r| r == 0).count();
        let top5 = ranks.iter().filter(|&&r| r < 5).count();
        let gap = |l: &Vec<f32>, w: u32| { let m = l.iter().cloned().fold(f32::MIN, f32::max); m - l[w as usize] };
        let gaps: Vec<f32> = forced.iter().zip(&want).map(|(l, &w)| gap(l, w)).collect();
        eprintln!("teacher-forced: ACE-Step's code is this planner's first choice {top1}/{} times, top 5 {top5}; ranks {:?}; \
                   logit gap to the best {:?}", want.len(), &ranks[..20.min(ranks.len())], &gaps[..10.min(gaps.len())]);
        let same_prefix = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
        let same = got.iter().zip(&want).filter(|(a, b)| a == b).count();
        eprintln!("{} codes in {:.1} s: the first {same_prefix} the same, {same} in all", got.len(), t0.elapsed().as_secs_f64());
        eprintln!("rust   {:?}\npython {:?}", &got[..12], &want[..12]);
        assert!(top5 * 10 >= want.len() * 9, "teacher-forced: ACE-Step's code in the top 5 only {top5} of {} times", want.len());
    }

    /// The golden codes through the FSQ and the detokenizer against Python's hints.
    #[test]
    #[ignore = "needs ACE-Step's checkpoints and golden files"]
    fn hints_from_codes_match_the_reference() {
        let r = reference();
        let dev = Device::Cpu;
        let ck = r.join("ACE-Step-1.5/checkpoints/acestep-v15-turbo");
        let cfg: dit::Config = serde_json::from_reader(std::fs::File::open(ck.join("config.json")).unwrap()).unwrap();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[ck.join("model.safetensors")], DType::F32, &dev).unwrap() };
        let d = dit::Detokenizer::new(&cfg, vb).unwrap();
        let codes: Vec<u32> = golden()["codes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let fsq = candle_core::safetensors::load(r.join("acestep-golden/planner/fsq.safetensors"), &dev).unwrap();
        let det = candle_core::safetensors::load(r.join("acestep-golden/planner/detok.safetensors"), &dev).unwrap();
        let cos = |x: &Tensor, y: &Tensor| {
            let (x, y) = (x.flatten_all().unwrap(), y.flatten_all().unwrap());
            let d = (&x * &y).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            d / (x.sqr().unwrap().sum_all().unwrap().sqrt().unwrap().to_scalar::<f32>().unwrap()
                 * y.sqr().unwrap().sum_all().unwrap().sqrt().unwrap().to_scalar::<f32>().unwrap())
        };
        let q = d.quantized(&codes, DType::F32, &dev).unwrap();
        let cq = cos(&q, &fsq["out"]);
        let h = d.hints(&codes, DType::F32, &dev).unwrap();
        assert_eq!(h.dims(), det["out"].dims());
        let ch = cos(&h, &det["out"]);
        eprintln!("quantized: cosine {cq:.6}; hints: cosine {ch:.6}");
        assert!(cq > 0.999 && ch > 0.995, "quantized {cq}, hints {ch}");
    }
}
