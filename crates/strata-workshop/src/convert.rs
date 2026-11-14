//! HF `LlamaForCausalLM` safetensors -> GGUF, for the llama architecture, ported from llama.cpp's
//! conversion/convert_hf_to_gguf.py. It does exactly what that script's `LlamaModel` does here: the same tensor-name
//! mapping (tensor_mapping.py), the same q/k rotary permute (`LlamaModel.permute`; HF stores the two halves of each
//! head's dim interleaved differently, so they are swapped), and the same gguf metadata keys + tokenizer extraction
//! (constants.py Keys.*{arch}). Tensor data is streamed out with `gguf_write`, as F32 or F16 per the `--outtype`.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device};
use serde_json::Value as Json;

use strata_pack::gguf_write::{Kv, TensorSpec, Writer};
use crate::safetensors_io;

/// Output element precision (the `--outtype` setting), mapping to llama.cpp's file-type enum: F32 = ALL_F32, F16 =
/// MOSTLY_F16.
#[derive(Debug, Clone, Copy)]
pub enum OutType {
    F32,
    F16,
}

impl OutType {
    pub fn from_s(s: &str) -> Result<Self> {
        match s {
            "f32" => Ok(OutType::F32),
            "f16" => Ok(OutType::F16),
            other => bail!("--outtype supports f32 or f16, got {other:?}"),
        }
    }
    /// general.file_type value (llama.cpp's LlamaFileType int for this precision).
    pub fn file_type(self) -> u32 {
        match self {
            OutType::F32 => 0, // ALL_F32
            OutType::F16 => 1, // MOSTLY_F16
        }
    }
}

/// A real `config.json` (LlamaForCausalLM), read the fields llama.cpp needs to build its metadata.
#[derive(Debug, Clone)]
pub struct HfConfig {
    pub arch: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: u32,
}

impl HfConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let d = dir.display();
        let c: Json = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("reading config for {d}"))?)?;
        let s = |k: &str| -> Result<f64> {
            c[k].as_f64().ok_or_else(|| anyhow!("{d}: config.{k} missing or not a number"))
        };
        let arch = c["architectures"].as_array()
            .and_then(|a| a.first()).and_then(Json::as_str)
            .ok_or_else(|| anyhow!("{d}: config.architectures empty"))?.to_string();
        Ok(HfConfig {
            arch,
            hidden_size: s("hidden_size")? as usize,
            num_hidden_layers: s("num_hidden_layers")? as usize,
            num_attention_heads: s("num_attention_heads")? as usize,
            num_key_value_heads: match c["num_key_value_heads"].as_f64() {
                Some(n) => n as usize,
                None => s("num_attention_heads")? as usize,
            },
            intermediate_size: s("intermediate_size")? as usize,
            rms_norm_eps: s("rms_norm_eps")? as f32,
            vocab_size: s("vocab_size")? as usize,
            max_position_embeddings: s("max_position_embeddings")? as usize,
            rope_theta: s("rope_theta")? as u32,
        })
    }

    /// Whether the config describes a LlamaForCausalLM (this converter only handles that architecture).
    pub fn is_llama(&self) -> bool {
        self.arch.contains("Llama")
    }
}

/// One HF tensor read from disk as its f32 values (row-major, numpy-order shape).
type TensorF32 = (Vec<usize>, Vec<f32>);

/// All tensors of the checkpoint, keyed by their HF name (`model.embed_tokens.weight`, ...).
pub fn read_tensors(dir: &Path) -> Result<BTreeMap<String, TensorF32>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        let d = dir.display();
        bail!("{d}: no *.safetensors shards found");
    }
    let mut out = BTreeMap::new();
    for f in files {
        let shard = safetensors_io::Shard::open(&f)?;
        for t in shard.tensors() {
            if t.dtype() != safetensors_io::Dtype::F32 {
                bail!("{}: tensor {} is {}, only F32 sources are supported by the converter", f.display(), t.name(), t.dtype().as_str());
            }
            let n = t.numel() as usize;
            let mut vals = vec![0f32; n];
            let mut data: &[u8] = shard.bytes(t);
            std::io::Read::read_exact(&mut data, bytemuck_like(&mut vals))
                .map_err(|e| anyhow!("{}: reading tensor {}: {e}", f.display(), t.name()))?;
            out.insert(t.name().to_string(), (t.shape().to_vec(), vals));
        }
    }
    Ok(out)
}

/// f32 slice as a `&mut [u8]` view, for the mmap byte copy.
fn bytemuck_like(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Shared,
    Block,
    PermuteQ,
    PermuteK,
}

/// The GGUF name a HF llama tensor maps to (tensor_mapping.py `llama-hf` patterns).
fn hf_to_gguf_name(name: &str) -> Option<(String, Kind)> {
    let mut it = name.split('.');
    match it.next()? {
        "model" => match it.next()? {
            "embed_tokens" => return Some(("token_embd".into(), Kind::Shared)),
            "norm" => return Some(("output_norm".into(), Kind::Shared)),
            _ => {}
        },
        "lm_head" => return Some(("output".into(), Kind::Shared)),
        _ => {}
    }
    // model.layers.{bid}.{...}
    let mut it = name.split('.');
    if it.next()? != "model" || it.next()? != "layers" {
        return None;
    }
    let bid = it.next()?;
    let module = it.next()?; // input_layernorm / post_attention_layernorm / self_attn / mlp
    let (seg, kind) = match module {
        "input_layernorm" => ("attn_norm", Kind::Block),
        "post_attention_layernorm" => ("ffn_norm", Kind::Block),
        _ => match it.next()? { // q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj
            "q_proj" => ("attn_q", Kind::PermuteQ),
            "k_proj" => ("attn_k", Kind::PermuteK),
            "v_proj" => ("attn_v", Kind::Block),
            "o_proj" => ("attn_output", Kind::Block),
            "gate_proj" => ("ffn_gate", Kind::Block),
            "up_proj" => ("ffn_up", Kind::Block),
            "down_proj" => ("ffn_down", Kind::Block),
            _ => return None,
        },
    };
    Some((format!("blk.{bid}.{seg}"), kind))
}

/// HF -> gguf rotary permute of q/k, the exact transform `LlamaModel.permute` applies (swap the two halves of each
/// head's dim so llama.cpp's rotary reads the interleaving HF stores differently). w is row-major [out, in].
fn permute(w: &[f32], heads: usize, in_: usize) -> Vec<f32> {
    let out = w.len() / in_;
    let head = out / heads; // head_dim (must be even)
    let half = head / 2;
    let mut dst = Vec::with_capacity(out * in_);
    for a in 0..heads {
        for b in 0..half {
            for c in 0..2 {
                for d in 0..in_ {
                    // output[(a*half+b)*2+c][d] = w[(a*2+c)*half+b][d] (LlamaModel.permute)
                    let i = (((a * 2 + c) * half + b) * in_) + d;
                    dst.push(w[i]);
                }
            }
        }
    }
    dst
}

/// One tensor ready to stream: its gguf name, numpy-order shape and f32 values (q/k already permuted).
struct Mapped {
    name: String,
    shape: Vec<u64>,
    vals: Vec<f32>,
}

fn mapped_tensors(tensors: &BTreeMap<String, TensorF32>) -> Result<Vec<Mapped>> {
    let mut out: Vec<Mapped> = Vec::new();
    for (hf, gguf) in [
        ("model.embed_tokens.weight", "token_embd"),
        ("model.norm.weight", "output_norm"),
        ("lm_head.weight", "output"),
    ] {
        if let Some((shape, vals)) = tensors.get(hf) {
            out.push(Mapped { name: gguf.to_string(), shape: shape.iter().map(|&d| d as u64).collect(), vals: vals.clone() });
        }
    }

    let mut bids: Vec<usize> = tensors.keys()
        .filter_map(|n| {
            let p: Vec<&str> = n.split('.').collect();
            if p.len() == 5 && p[0] == "model" && p[1] == "layers" {
                p[2].parse::<usize>().ok()
            } else { None }
        }).collect();
    bids.sort_unstable();
    bids.dedup(); // one entry per weight shared the same layer id; process each layer once

    for bid in bids {
        let p = |key: &str| format!("model.layers.{bid}.{key}");
        // attn_q (permute with attention heads), attn_k (permute with kv heads).
        for (name, gguf, heads) in [
            (p("self_attn.q_proj.weight").as_str(), format!("blk.{bid}.attn_q"), 2),
            (p("self_attn.k_proj.weight").as_str(), format!("blk.{bid}.attn_k"), 2),
        ] {
            if let Some((shape, vals)) = tensors.get(name) {
                let vals = permute(vals, heads, shape[1] as usize);
                out.push(Mapped { name: gguf, shape: shape.iter().map(|&d| d as u64).collect(), vals });
            }
        }
        for (hf, gguf) in [
            (p("input_layernorm.weight").as_str(), format!("blk.{bid}.attn_norm")),
            (p("self_attn.v_proj.weight").as_str(), format!("blk.{bid}.attn_v")),
            (p("self_attn.o_proj.weight").as_str(), format!("blk.{bid}.attn_output")),
            (p("post_attention_layernorm.weight").as_str(), format!("blk.{bid}.ffn_norm")),
            (p("mlp.gate_proj.weight").as_str(), format!("blk.{bid}.ffn_gate")),
            (p("mlp.up_proj.weight").as_str(), format!("blk.{bid}.ffn_up")),
            (p("mlp.down_proj.weight").as_str(), format!("blk.{bid}.ffn_down")),
        ] {
            if let Some((shape, vals)) = tensors.get(hf) {
                out.push(Mapped { name: gguf, shape: shape.iter().map(|&d| d as u64).collect(), vals: vals.clone() });
            }
        }
    }
    Ok(out)
}

/// Serialize f32 values to the target precision's raw bytes (F16 via candle).
fn serialize(vals: &[f32], outtype: OutType) -> Vec<u8> {
    match outtype {
        OutType::F32 => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
        OutType::F16 => candle_core::Tensor::from_vec(vals.to_vec(), /*shape=*/ &[], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<half::f16>()
            .unwrap()
            .into_iter().flat_map(|h| h.to_bits().to_le_bytes()).collect(),
    }
}

/// The five tokenizer arrays (pre, model, tokens, scores, token_type) + the bos/eos ids, from tokenizer.json /
/// tokenizer_config.json. Mirrors llama.cpp's `_set_vocab_llama_hf`: model "llama", pre "default", tokens sorted by
/// id, special-token scores 100.0 and base-token scores -1000.0 (SpecialVocab's write_internal_scores), token_type
/// CONTROL for the special ids and NORMAL (1) otherwise.
fn tokenizer_arrays(dir: &Path, vocab_size: usize) -> Result<(Vec<Kv>, u32, u32)> {
    let tj: Json = serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer.json"))
        .with_context(|| format!("reading tokenizer.json in {}", dir.display()))?)?;
    let tc: Json = serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json"))
        .with_context(|| format!("reading tokenizer_config.json in {}", dir.display()))?)?;

    let mut vocab: Vec<(String, usize)> = tj["model"]["vocab"]
        .as_object()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0) as usize)).collect())
        .ok_or_else(|| anyhow!("tokenizer.json model.vocab is not an object"))?;
    vocab.sort_by_key(|(_, id)| *id); // HF orders tokens by id

    let mut special_ids: HashSet<usize> = HashSet::new();
    for key in ["bos_token", "eos_token"] {
        if let Some(txt) = special_text(&tc, key) {
            if let Some(id) = vocab.iter().find(|(t, _)| *t == txt).map(|(_, id)| *id) {
                special_ids.insert(id);
            }
        }
    }

    // Token ids of the special tokens, as SpecialVocab writes them for tokenizer.ggml.bos_token_id / .eos_token_id;
    // 0 when absent (mirrors llama.cpp's default when config.json omits an id).
    let token_id_for = |key: &str| -> usize {
        special_text(&tc, key)
            .and_then(|txt| vocab.iter().find(|(t, _)| *t == txt).map(|(_, id)| *id))
            .unwrap_or(0)
    };

    let tokens: Vec<String> = vocab.iter().map(|(t, _)| t.clone()).collect();
    // SpecialVocab's internal scores: special tokens (bos/eos) bias to 100.0, base tokens to -1000.0 — the value this
    // version's HF tokenizer extraction writes for every non-special token.
    let scores: Vec<f32> = (0..vocab_size).map(|i| if special_ids.contains(&i) { 100.0 } else { -1000.0 }).collect();
    let token_type: Vec<i32> = (0..vocab_size)
        .map(|i| if special_ids.contains(&i) { 3 } else { 1 }) // CONTROL / NORMAL (gguf TokenType enum)
        .collect();

    Ok((
        vec![
            Kv::Str("default".into()),
            Kv::Str("llama".into()),
            Kv::Strings(tokens),
            Kv::Floats(scores),
            Kv::Ints(token_type),
        ],
        token_id_for("bos_token") as u32,
        token_id_for("eos_token") as u32,
    ))
}

/// The token text a tokenizer_config special-token entry refers to (string form, or the dict "content" field).
fn special_text(tc: &Json, key: &str) -> Option<String> {
    match tc.get(key)? {
        Json::String(s) => Some(s.clone()),
        Json::Object(m) => m.get("content").and_then(Json::as_str).map(str::to_string),
        _ => None,
    }
}

/// The chat template string (may be empty).
fn chat_template(tc: &Json) -> String {
    tc["chat_template"].as_str().unwrap_or("").to_string()
}

/// Write the GGUF file. `dir` holds the HF checkpoint (config/tokenizer/shards); `out` is the destination path.
pub fn convert(dir: &Path, out: &Path, outtype: OutType) -> Result<()> {
    let hparams = HfConfig::load(dir)?;
    if !hparams.is_llama() {
        bail!("converter only handles LlamaForCausalLM (config.architectures={:?})", hparams.arch);
    }
    let tensors = read_tensors(dir)?;

    let (tok_arrays, bos_id, eos_id) = tokenizer_arrays(dir, hparams.vocab_size)?;
    let tc: Json = serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json"))
        .with_context(|| format!("reading tokenizer_config.json in {}", dir.display()))?)?;
    let chat = chat_template(&tc);

    // metadata key/value pairs (general first, then the llama.* hyper-parameters, then tokenizer).
    let mut kv: Vec<(&'static str, Kv)> = vec![
        ("general.architecture", Kv::Str("llama".into())),
        ("general.file_type", Kv::U32(outtype.file_type())),
        ("llama.context_length", Kv::U32(hparams.max_position_embeddings as u32)),
        ("llama.embedding_length", Kv::U32(hparams.hidden_size as u32)),
        ("llama.block_count", Kv::U32(hparams.num_hidden_layers as u32)),
        ("llama.feed_forward_length", Kv::U32(hparams.intermediate_size as u32)),
        ("llama.attention.head_count", Kv::U32(hparams.num_attention_heads as u32)),
        ("llama.attention.head_count_kv", Kv::U32(hparams.num_key_value_heads as u32)),
        ("llama.attention.layer_norm_rms_epsilon", Kv::F32(hparams.rms_norm_eps)),
        ("llama.rope.freq_base", Kv::U32(hparams.rope_theta)),
        ("llama.vocab_size", Kv::U32(hparams.vocab_size as u32)),
    ];
    kv.push(("tokenizer.ggml.pre", tok_arrays[0].clone()));
    kv.push(("tokenizer.ggml.model", tok_arrays[1].clone()));
    kv.push(("tokenizer.ggml.tokens", tok_arrays[2].clone()));
    kv.push(("tokenizer.ggml.scores", tok_arrays[3].clone()));
    kv.push(("tokenizer.ggml.token_type", tok_arrays[4].clone()));
    kv.push(("tokenizer.ggml.bos_token_id", Kv::U32(bos_id)));
    kv.push(("tokenizer.ggml.eos_token_id", Kv::U32(eos_id)));
    // add_bos/add_eos flags mirror what llama.cpp's SpecialVocab writes from tokenizer_config.json.
    kv.push(("tokenizer.ggml.add_bos_token", Kv::Bool(get_bool(&tc, "add_bos_token"))));
    kv.push(("tokenizer.ggml.add_eos_token", Kv::Bool(get_bool(&tc, "add_eos_token"))));
    if !chat.is_empty() {
        kv.push(("tokenizer.chat_template", Kv::Str(chat)));
    }

    let mapped = mapped_tensors(&tensors)?;
    let specs: Vec<TensorSpec> = mapped.iter().map(|m| TensorSpec {
        name: m.name.clone(),
        shape: m.shape.clone(),
        ggml_type: outtype.file_type(),
        nbytes: serialize(&m.vals, outtype).len() as u64,
    }).collect();

    let mut w = Writer::create_typed(out, &kv, specs)?;
    for m in &mapped {
        let bytes = serialize(&m.vals, outtype);
        w.tensor_data(&bytes)?;
        w.end_tensor()?;
    }
    w.finish()?;
    Ok(())
}

fn get_bool(tc: &Json, key: &str) -> bool {
    match tc.get(key) {
        Some(Json::Bool(b)) => *b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_mapping_matches_llama_hf_patterns() {
        let (n, _) = hf_to_gguf_name("model.embed_tokens.weight").unwrap();
        assert_eq!(n, "token_embd");
        let (n, _) = hf_to_gguf_name("lm_head.weight").unwrap();
        assert_eq!(n, "output");
        let (n, _) = hf_to_gguf_name("model.norm.weight").unwrap();
        assert_eq!(n, "output_norm");
        for i in 0..2 {
            assert_eq!(hf_to_gguf_name(&format!("model.layers.{i}.input_layernorm.weight")).unwrap().0, format!("blk.{i}.attn_norm"));
            assert_eq!(hf_to_gguf_name(&format!("model.layers.{i}.self_attn.q_proj.weight")).unwrap().0, format!("blk.{i}.attn_q"));
            assert_eq!(hf_to_gguf_name(&format!("model.layers.{i}.self_attn.k_proj.weight")).unwrap().0, format!("blk.{i}.attn_k"));
            assert_eq!(hf_to_gguf_name(&format!("model.layers.{i}.self_attn.v_proj.weight")).unwrap().0, format!("blk.{i}.attn_v"));
            assert_eq!(hf_to_gguf_name(&format!("model.layers.{i}.self_attn.o_proj.weight")).unwrap().0, format!("blk.{i}.attn_output"));
            assert_eq!(hf_to_gguf_name(&format!("model.layers.{i}.post_attention_layernorm.weight")).unwrap().0, format!("blk.{i}.ffn_norm"));
            assert_eq!(hf_to_gguf_name(&format!("model.layers.{i}.mlp.gate_proj.weight")).unwrap().0, format!("blk.{i}.ffn_gate"));
            assert_eq!(hf_to_gguf_name(&format!("model.layers.{i}.mlp.up_proj.weight")).unwrap().0, format!("blk.{i}.ffn_up"));
            assert_eq!(hf_to_gguf_name(&format!("model.layers.{i}.mlp.down_proj.weight")).unwrap().0, format!("blk.{i}.ffn_down"));
        }
    }

    #[test]
    fn permute_swaps_the_two_halves_of_each_head() {
        // heads=2, head_dim=4 -> out=8, in=1. Each head's two halves interleave: for h the dims become [h0,h2,h1,h3].
        let w: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let got = permute(&w, 2, 1);
        let expected: Vec<f32> = vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0];
        assert_eq!(got, expected, "q/k permute bytes must match LlamaModel.permute");
    }

    #[test]
    fn tokenizer_orders_by_id_and_flags_specials() {
        let dir = std::env::temp_dir().join("convert-tok-selfcheck");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("tokenizer.json"),
            r#"{"version":"1.1","model":{"type":"BPE","byte_fallback":true,"vocab":{"tok_0":0,"tok_1":1,"tok_2":2,"tok_3":3},"merges":[]}}"#).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"),
            r#"{"add_bos_token":true,"add_eos_token":false,"bos_token":"tok_0","eos_token":"tok_1"}"#).unwrap();
        let (arr, bos, eos) = tokenizer_arrays(&dir, 4).unwrap();
        assert_eq!(bos, 0);
        assert_eq!(eos, 1);
        if let Kv::Strings(toks) = &arr[2] { assert_eq!(toks, &vec!["tok_0", "tok_1", "tok_2", "tok_3"]); } else { panic!("tokens array"); }
        if let Kv::Ints(tt) = &arr[4] { assert_eq!(*tt, vec![3, 3, 1, 1]); } else { panic!("token_type array"); }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn outtype_maps_to_file_type_and_precision() {
        assert_eq!(OutType::F32.file_type(), 0);
        assert_eq!(OutType::F16.file_type(), 1);
        assert!(OutType::from_s("f32").is_ok());
        assert!(OutType::from_s("bf16").is_err());
    }
}
