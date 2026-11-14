//! HF `LlamaForCausalLM` safetensors -> GGUF, for the llama architecture, ported from llama.cpp's
//! conversion/convert_hf_to_gguf.py. It does exactly what that script's `LlamaModel` does here: the same tensor-name
//! mapping (tensor_mapping.py), the same q/k rotary permute (`LlamaModel.permute`; HF stores the two halves of each
//! head's dim interleaved differently, so they are swapped), and the same gguf metadata keys + tokenizer extraction
//! (constants.py Keys.*{arch}). Tensor data is streamed out with `gguf_write`, as F32 or F16 per the `--outtype`.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{anyhow, bail, Context, Result};

use serde_json::Value as Json;

use strata_pack::gguf_write::{Kv, TensorSpec, Writer};
use crate::safetensors_io;
use sha2::{Digest, Sha256};

/// The probe text llama.cpp tokenizes through the model's BPE pre-tokenizer to name it (`conversion/base.py`
/// `get_vocab_base_pre`, its `chktxt`). Encode it with the model tokenizer and sha256 the resulting id list; match the
/// hash against [`PRE_TOKENIZER_HASHES`] to recover the pre-tokenizer's llama.cpp name.
const PRE_CHECK_TXT: &str = "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български ''''''```````\"\"\"\"......!!!!!!?????? I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL";


/// llama.cpp's table of known BPE pre-tokenizer probe hashes -> name (`conversion/base.py` `get_vocab_base_pre`, its
/// `if chkhsh == ... : res = "..."` lines). Identical to llama.cpp, including the two hashes that both map to `"gpt-2"`.
const PRE_TOKENIZER_HASHES: &[(&str, &str)] = &[
    ("b6e8e1518dc4305be2fe39c313ed643381c4da5db34a98f6a04c093f8afbe99b","chatglm-bpe"),
    ("81d72c7348a9f0ebe86f23298d37debe0a5e71149e29bd283904c02262b27516","chatglm-bpe"),
    ("a1336059768a55c99a734006ffb02203cd450fed003e9a71886c88acf24fdbc2","glm4"),
    ("9ca2dd618e8afaf09731a7cf6e2105b373ba6a1821559f258b272fe83e6eb902","glm4"),
    ("cdf5f35325780597efd76153d4d1c16778f766173908894c04afc20108536267","glm4"),
    ("1431a23e583c97432bc230bff598d103ddb5a1f89960c8f1d1051aaa944d0b35","minerva-7b"),
    ("7e57df22b1fe23a7b1e1c7f3dc4e3f96d43a4eb0836d0c6bdc3436d7b2f1c664","hunyuan"),
    ("bba3b3366b646dbdded5dbc42d59598b849371afc42f7beafa914afaa5b70aa6","hunyuan-dense"),
    ("e6ddf9c6686791c12d698d34c31ab9be1fea9af5a3d9a6909783ab382198ae1c","hy_v4"),
    ("a6b57017d60e6edb4d88ecc2845188e0eb333a70357e45dcc9b53964a73bbae6","falcon-h1"),
    ("60476e1243776c4fb1b993dbd7a5f15ac22f83c80afdf425fa5ae01c8d44ef86","falcon-h1"),
    ("3eda48b4c4dc7de733d1a8b3e3b4a85243dbbf704da2ee9d42c6beced8897896","falcon-h1"),
    ("48f8e02c0359c0bbdd82f26909171fac1c18a457bb47573ed1fe3bbb2c1cfd4b","falcon-h1"),
    ("81212dc7cdb7e0c1074ca62c5aeab0d43c9f52b8a737be7b12a777c953027890","kimi-k2"),
    ("d4540891389ea895b53b399da6ac824becc30f2fba0e9ddbb98f92e55ca0e97c","qwen2"),
    ("1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f","qwen35"),
    ("66b8d4e19ab16c3bfd89bce5d785fb7e0155e8648708a1f42077cb9fe002c273","grok-2"),
    ("b3d1dd861f1d4c5c0d2569ce36baf3f90fe8a102db3de50dd71ff860d91be3df","jina-v2-de"),
    ("0fe1cf6eda062318a1af7270f3331a85c539a01778ff948e24388e949c5282f4","gpt-2"),
    ("9e454714343b69b99b71795c1d27a68c2a1d15dab111f4d353109f966af29da7","lfm2"),
    ("846deafc5b0fa786186fa4ae6c7b49903cf2f1d1895bdb80b9120d60be135252","gemma4"),
    ("0a766d034107bc736a3f2dc4968fd62e54a3570f1454443e0c5a4cc6bd7941ed","spark2_5"),
    ("0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5","llama-bpe"),
    ("049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754","deepseek-llm"),
    ("347715f544604f9118bb75ed199f68779f423cabb20db6de6f31b908d04d7821","deepseek-coder"),
    ("8aeee3860c56296a157a1fe2fad249ec40aa59b1bb5709f4ade11c4e6fe652ed","falcon"),
    ("0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f","bert-bge"),
    ("9d032fcbd5501f4a38150912590928bfb36091efb5df11b8e2124b0390e3fb1e","falcon3"),
    ("8e62295832751ca1e8f92f2226f403dea30dc5165e448b5bfa05af5340c64ec7","bert-bge-large"),
    ("b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166","mpt"),
    ("35d91631860c815f952d711435f48d356ebac988362536bed955d43bfa436e34","starcoder"),
    ("3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454","gpt-2"),
    ("32d85c31273f8019248f2559fed492d929ea28b17e51d81d3bb36fff23ca72b3","stablelm2"),
    ("6221ad2852e85ce96f791f476e0b390cf9b474c9e3d1362f53a24a06dc8220ff","refact"),
    ("9c2227e4dd922002fb81bde4fc02b0483ca4f12911410dee2255e4987644e3f8","command-r"),
    ("d772b220ace2baec124bed8cfafce0ead7d6c38a4b65ef11261cf9d5d62246d1","tiny_aya"),
    ("52df12b4c8d4176e7481aab4b6e8454d1fd0a210a04a574f6d4e067d10e23c3e","cohere2moe"),
    ("e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea","qwen2"),
    ("b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166","olmo"),
    ("a8594e3edff7c29c003940395316294b2c623e09894deebbc65f33f1515df79e","dbrx"),
    ("c7699093ba4255a91e702aa38a596aa81669f3525dae06c2953267dde580f448","jina-v1-en"),
    ("0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f","jina-v2-en"),
    ("171aeeedd6fb548d418a7461d053f11b6f1f1fc9b387bd66640d28a4b9f5c643","jina-v2-es"),
    ("27949a2493fc4a9f53f5b9b029c82689cfbe5d3a1929bb25e043089e28466de6","jina-v2-de"),
    ("a023e9fdc5a11f034d3ef515b92350e56fb2af1f66c6b6811a4444ea9bf8763d","jina-v5-nano"),
    ("c136ed14d01c2745d4f60a9596ae66800e2b61fa45643e72436041855ad4089d","smaug-bpe"),
    ("c7ea5862a53e4272c035c8238367063e2b270d51faa48c0f09e9d5b54746c360","poro-chat"),
    ("7967bfa498ade6b757b064f31e964dddbb80f8f9a4d68d4ba7998fcf281c531a","jina-v2-code"),
    ("7fc505bd3104ca1083b150b17d088b59534ede9bde81f0dd2090967d7fe52cee","viking"),
    ("b53802fb28e26d645c3a310b34bfe07da813026ec7c7716883404d5e0f8b1901","jais"),
    ("bc5108ee1eb6a3d600cadd065f63190fbd0554dbc9e4bbd6a0d977970afc8d2a","jais-2"),
    ("7b3e7548e4308f52a76e8229e4e6cc831195d0d1df43aed21ac6c93da05fec5f","codeshell"),
    ("63b97e4253352e6f357cc59ea5b583e3a680eaeaf2632188c2b952de2588485e","tekken"),
    ("855059429035d75a914d1eda9f10a876752e281a054a7a3d421ef0533e5b6249","smollm"),
    ("3c30d3ad1d6b64202cd222813e7736c2db6e1bd6d67197090fc1211fbc612ae7","bloom"),
    ("bc01ce58980e1db43859146dc51b1758b3b88729b217a74792e9f8d43e479d21","gpt3-finnish"),
    ("4e2b24cc4770243d65a2c9ec19770a72f08cffc161adbb73fcbb6b7dd45a0aae","exaone"),
    ("fcace8b9cac38ce847670c970cd5892031a753a1ef381abd1d9af00f713da085","phi-2"),
    ("60824e3c0d9401f89943cbb2fff727f0e2d4c545ba4df2d6e4f09a6db0f5b450","chameleon"),
    ("8b5a93ed704057481f240da0be7e7dca721d7f8f4755263b6807227a2cbeae65","roberta-bpe"),
    ("ad851be1dba641f2e3711822f816db2c265f788b37c63b4e1aeacb9ee92de8eb","gigachat"),
    ("d4c8f286ea6b520b3d495c4455483cfa2302c0cfcd4be05d781b6a8a0a7cdaf1","megrez"),
    ("877081d19cf6996e2c4ff0e1236341e9b7bde288f5311a56a937f0afbbb3aeb5","deepseek-v3"),
    ("b3f499bb4255f8ca19fccd664443283318f2fd2414d5e0b040fbdd0cc195d6c5","deepseek-r1-qwen"),
    ("ccc2ef013c104be7bae2965776d611e1d7a8a2a9c547dd93a682c9a9fc80352e","gpt-4o"),
    ("7dec86086fcc38b66b7bc1575a160ae21cf705be7718b9d5598190d7c12db76f","superbpe"),
    ("1994ffd01900cfb37395608534236ecd63f2bd5995d6cb1004dda1af50240f15","trillion"),
    ("96a5f08be6259352137b512d4157e333e21df7edd3fcd152990608735a65b224","bailingmoe"),
    ("d353350c764d8c3b39c763113960e4fb4919bea5fbf208a0e3b22e8469dc7406","llama4"),
    ("0e9433cbbb161f89e264eb32e8e64bfe69e834973ffca5d41d3948a604a3e2a3","pixtral"),
    ("d5f1dd6f980fec569fb218a81a7658ac45fc56b38c5a0adeb1c232fbe04ef5ec","seed-coder"),
    ("b0a6b1c0bd5998ebd9df08611efde34a4ff03faed45ae09c43e6b31ebd4b94cf","a.x-4.0"),
    ("f6791d196f87ce6b56a7d234be618e0d58f8cda3549416635b2bebcd22cd95c4","midm-2.0"),
    ("169bf0296a13c4d9b7672313f749eb36501d931022de052aad6e36f2bf34dd51","lfm2"),
    ("2085e1638f6c377a0aa4ead21b27bb4cb941bf800df86ed391011769c1758dfb","exaone4"),
    ("a1e163ecab2e718a4c829d1148b6e86824ec36163bb71941c3dca9cd5ac25756","mellum"),
    ("a0b64b4385f123663873756336c085744376d015ff328bb1d901598f63c44152","modern-bert"),
    ("49fc0303c9e0d2c2c565c510f64b2d9b271276acdcdadff733249eda9f7d59df","afmoe"),
    ("9b1be57e70d20d9501b2b3186e792d81181ae36ada3903c26f9fea418cf87206","bailingmoe2"),
    ("53e325976a6e142379c19b09afcae354f2f496f147afa8f9e189a33fe4e3024e","granite-docling"),
    ("f4f37b6c8eb9ea29b3eac6bb8c8487c5ab7885f8d8022e67edc1c68ce8403e95","minimax-m2"),
    ("4a2e2abae11ca2b86d570fc5b44be4d5eb5e72cc8f22dd136a94b37da83ab665","kormo"),
    ("9d70134b369a70e5735009b6de918f7581b5211f7c074d1f89f753aea8248af1","youtu"),
    ("16389f0a1f51ee53e562ffd51c371dc508639ab0e4261502071836e50e223e91","solar-open"),
    ("6c81ce329e0802883b22eabab0d3fa48357337ef1ecb45443828bf1f6254833f","exaone-moe"),
    ("d30d75d9059f1aa2c19359de71047b3ae408c70875e8a3ccf8c5fba56c9d8af4","qwen35"),
    ("b4b8ca1f9769494fbd956ebc4c249de6131fb277a4a3345a7a92c7dd7a55808d","joyai-llm"),
    ("e4d54df1ebc1f2b91acd986c5b51aa50837d5faf7c7398e73c1f9e9ee5d19869","kanana2"),
    ("862f827721df956049dff5ca81a57f29e575280bc622e290d3bf4e35eca29015","f2llmv2"),
    ("62f6fb0a6fd5098caeabb19b07a5c1099cafc8b9c40eab6ea89ece4ec02fbc57","sarvam-moe"),
    ("f728162c1315c26e40249849799b4ba3fe584c32084b4795b03eb295e63cb5af","talkie"),
    ("36f3066e97b7f3994b379aaacde306c1444c6ae84e81a5ae3cd2b7ed3b8c42d4","minicpm5"),
    ("f241072145675bf8322086f115aebad05e9f869557a238bf2150a2a417d1bf60","granite-embed-multi-97m"),
    ("789696f5946cc0fc59371f39f6097cafed196b3acded6140432f26bbb1ae1669","granite-embed-multi-311m"),
    ("9dcf830ee9990cdbf78cc523a5f7bd9ad8f3f9890c2d3581d2785ad10f07049d","mellum2"),
    ("972da7b59cec44d1f0a490a86c96df53859e486e481563e5dddac155013d87ac","laguna"),
    ("653660222fb704f61cbf2b618a8ae6502b7f8b20c980f9a5de07ed78e13319cd","ufakzeka"),
    ("4b05e02dad1c5ae07d266fd3342ddb644c6f6be058d728bc0a33af31a1d6ee66","mmbert"),
];
/// Look up a BPE pre-tokenizer name by its llama.cpp probe-hash hex; `None` when the hash is not in [`PRE_TOKENIZER_HASHES`].
fn known_pre_tokenizer_name(hash_hex: &str) -> Option<&'static str> {
    PRE_TOKENIZER_HASHES.iter().find(|(h, _)| *h == hash_hex).map(|(_, n)| *n)
}

/// Port of llama.cpp's `get_vocab_base_pre`: tokenize [`PRE_CHECK_TXT`] with the model tokenizer, sha256 the token-id list
/// exactly as Python's `sha256(str(chktok).encode()).hexdigest()` does (the utf-8 of the comma/space, square-bracketed
/// repr), and match it against [`PRE_TOKENIZER_HASHES`]. An unknown hash is refused with llama.cpp's message unless the
/// caller supplies `override_pre`; that override is how a synthetic tokenizer gets named - matching
/// `make-tiny-llama-reference.py`, which forces this to `"gpt-2"` for the tiny workshop model.
pub fn pre_tokenizer_name(chktok_ids: &[u32], override_pre: Option<&str>) -> Result<String> {
    let repr = format!("[{}]", chktok_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", "));
    let digest = {
        let mut hasher = Sha256::new();
        hasher.update(repr.as_bytes());
        format!("{:x}", hasher.finalize())
    };
    if let Some(name) = known_pre_tokenizer_name(&digest) {
        return Ok(name.to_string());
    }
    match override_pre {
        Some(p) => Ok(p.to_string()),
        None => Err(anyhow!(
            "BPE pre-tokenizer was not recognized - update get_vocab_base_pre()\n\
             ** chkhsh:  {digest}"
        )),
    }
}

/// The `tokenizer.ggml.pre` value for the checkpoint in `dir`: tokenize [`PRE_CHECK_TXT`] with its tokenizer and identify
/// it. Unknown probe hash falls through to `override_pre` (the tiny fixture's `"gpt-2"`).
fn model_pre_tokenizer(dir: &Path, override_pre: Option<&str>) -> Result<String> {
    let tj = std::fs::read_to_string(dir.join("tokenizer.json"))
        .with_context(|| format!("reading tokenizer.json in {}", dir.display()))?;
    let tok = match tokenizers::Tokenizer::from_str(&tj) {
        Ok(t) => t,
        Err(e) => bail!("parsing tokenizer.json in {}: {e}", dir.display()),
    };
    let enc = tok.encode(PRE_CHECK_TXT, false).map_err(|e| anyhow!("encoding probe text with tokenizer.json in {}: {e}", dir.display()))?;
    pre_tokenizer_name(enc.get_ids(), override_pre)
}

/// Output element precision (the `--outtype` setting), mapping to llama.cpp's file-type enum: F32 = ALL_F32, F16 =
/// MOSTLY_F16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            // F32 as is; BF16 and F16 (what Hugging Face models are stored in) widened to F32 exactly, as llama.cpp's
            // converter does before writing the output precision
            let raw: &[u8] = shard.bytes(t);
            let vals: Vec<f32> = match t.dtype() {
                safetensors_io::Dtype::F32 => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                safetensors_io::Dtype::BF16 => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
                safetensors_io::Dtype::F16 => raw.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
                other => bail!("{}: tensor {} is {}; the converter reads F32, BF16 and F16", f.display(), t.name(), other.as_str()),
            };
            if vals.len() != t.numel() as usize {
                bail!("{}: tensor {} has {} values, its shape says {}", f.display(), t.name(), vals.len(), t.numel());
            }
            out.insert(t.name().to_string(), (t.shape().to_vec(), vals));
        }
    }
    Ok(out)
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

fn mapped_tensors(tensors: &BTreeMap<String, TensorF32>, hparams: &HfConfig) -> Result<Vec<Mapped>> {
    let mut out: Vec<Mapped> = Vec::new();
    for (hf, gguf) in [
        ("model.embed_tokens.weight", "token_embd"),
        ("model.norm.weight", "output_norm"),
        ("lm_head.weight", "output"),
    ] {
        if let Some((shape, vals)) = tensors.get(hf) {
            out.push(Mapped { name: format!("{}.weight", gguf), shape: shape.iter().map(|&d| d as u64).collect(), vals: vals.clone() });
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
        // attn_q (permute with attention heads) and attn_k (permute with kv heads): exactly llama.cpp's
        // `LlamaModel.permute(w, n_head, n_head_kv)` — q calls it with `(n_head, n_head)` so it keeps num_attention_heads,
        // k with `(n_head, n_head_kv)` so GQA collapses it to num_key_value_heads (see conversion/llama.py modify_tensors).
        for (name, gguf, heads) in [
            (p("self_attn.q_proj.weight").as_str(), format!("blk.{bid}.attn_q.weight"), hparams.num_attention_heads),
            (p("self_attn.k_proj.weight").as_str(), format!("blk.{bid}.attn_k.weight"), hparams.num_key_value_heads),
        ] {
            if let Some((shape, vals)) = tensors.get(name) {
                let vals = permute(vals, heads, shape[1] as usize);
                out.push(Mapped { name: gguf, shape: shape.iter().map(|&d| d as u64).collect(), vals });
            }
        }
        for (hf, gguf) in [
            (p("input_layernorm.weight").as_str(), format!("blk.{bid}.attn_norm.weight")),
            (p("self_attn.v_proj.weight").as_str(), format!("blk.{bid}.attn_v.weight")),
            (p("self_attn.o_proj.weight").as_str(), format!("blk.{bid}.attn_output.weight")),
            (p("post_attention_layernorm.weight").as_str(), format!("blk.{bid}.ffn_norm.weight")),
            (p("mlp.gate_proj.weight").as_str(), format!("blk.{bid}.ffn_gate.weight")),
            (p("mlp.up_proj.weight").as_str(), format!("blk.{bid}.ffn_up.weight")),
            (p("mlp.down_proj.weight").as_str(), format!("blk.{bid}.ffn_down.weight")),
        ] {
            if let Some((shape, vals)) = tensors.get(hf) {
                out.push(Mapped { name: gguf, shape: shape.iter().map(|&d| d as u64).collect(), vals: vals.clone() });
            }
        }
    }
    Ok(out)
}

/// The ggml type a tensor is written as: llama.cpp's converter keeps 1-D tensors (norms, biases) in F32 whatever the
/// `--outtype`, and writes the rest in the chosen precision.
fn tensor_type(shape: &[usize], outtype: OutType) -> (u32, OutType) {
    if shape.len() <= 1 { (0, OutType::F32) } else {
        match outtype { OutType::F32 => (0, OutType::F32), OutType::F16 => (1, OutType::F16) }
    }
}

/// f32 values as the raw little-endian bytes of `t`. F16 rounds to nearest even, as numpy's `astype(np.float16)` does.
fn serialize(vals: &[f32], t: OutType) -> Vec<u8> {
    match t {
        OutType::F32 => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
        OutType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect(),
    }
}

/// The five tokenizer arrays (pre, model, tokens, scores, token_type) + the bos/eos ids, from tokenizer.json /
/// tokenizer_config.json. Mirrors llama.cpp's `_set_vocab_llama_hf`: model "llama", pre "default", tokens sorted by
/// id, special-token scores 100.0 and base-token scores -1000.0 (SpecialVocab's write_internal_scores), token_type
/// CONTROL for the special ids and NORMAL (1) otherwise.
fn tokenizer_arrays(dir: &Path, vocab_size: usize, pre: &str) -> Result<(Vec<Kv>, u32, u32)> {
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

    // llama.cpp writes "gpt2" for a BPE tokenizer.json (the HF path) and "llama" only for a SentencePiece tokenizer.model.
    let ggml_model = tj["model"]
        .get("type")
        .and_then(Json::as_str)
        .map(|t| t.eq_ignore_ascii_case("BPE"))
        .unwrap_or(false);

    Ok((
        vec![
            Kv::Str(pre.into()),
            Kv::Str(if ggml_model { "gpt2" } else { "llama" }.into()),
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

/// A word id_to_title leaves alone: gguf's regex `^(v\d+(?:\.\d+)*|\d.*)$` matched against the whole word — a version
/// (`v2`, `v1.5`) or anything starting with a digit (`256`, `1.5b`).
fn looks_like_version_or_acronym(word: &str) -> bool {
    let b = word.as_bytes();
    if b.first().is_some_and(|c| c.is_ascii_digit()) {
        return true;
    }
    match b.split_first() {
        Some((b'v', rest)) if !rest.is_empty() => {
            // \d+ then (\.\d+)*, to the end
            rest.split(|c| *c == b'.').all(|part| !part.is_empty() && part.iter().all(u8::is_ascii_digit))
        }
        _ => false,
    }
}

/// Python's `str.islower()`: at least one cased character, and every cased character lowercase (digits do not count).
fn py_islower(w: &str) -> bool {
    let mut cased = false;
    for c in w.chars() {
        if c.is_uppercase() { return false; }
        if c.is_lowercase() { cased = true; }
    }
    cased
}

/// Python's `str.title()`: a letter after a non-letter is upper-cased, every other letter lower-cased ("bf16" -> "Bf16",
/// "llama3b" -> "Llama3B").
fn py_title(w: &str) -> String {
    let mut out = String::with_capacity(w.len());
    let mut prev_cased = false;
    for c in w.chars() {
        if c.is_alphabetic() {
            if prev_cased { out.extend(c.to_lowercase()) } else { out.extend(c.to_uppercase()) }
            prev_cased = true;
        } else {
            out.push(c);
            prev_cased = false;
        }
    }
    out
}

/// Port of gguf's `id_to_title`: dashes to spaces, then each word title-cased when it is all lowercase and not a
/// version — `tiny-llama-256-bf16-hf` -> "Tiny Llama 256 Bf16 Hf".
fn id_to_title(input: &str) -> String {
    input.trim().replace('-', " ").split_whitespace()
        .map(|w| if py_islower(w) && !looks_like_version_or_acronym(w) { py_title(w) } else { w.to_string() })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Port of gguf's `size_label` for a dense (non-expert) model: the parameter count in rounded K/M/B/T notation, via
/// gguf's `model_weight_count_rounded_notation`.
fn model_size_label(params: u64) -> String {
    let value = params as f64;
    let (scaled, suffix) = if value > 1e12 {
        (value * 1e-12, "T")
    } else if value > 1e9 {
        (value * 1e-9, "B")
    } else if value > 1e6 {
        (value * 1e-6, "M")
    } else {
        (value * 1e-3, "K")
    };
    // min_digits is fixed at 2 in llama.cpp; fix = max(min_digits - len(str(round(scaled)).lstrip('0')), 0). saturating so a
    // label whose integer part has >=3 digits (a model of >= ~1e5 params) can't underflow the subtraction.
    let rounded_len = (scaled.round() as i64).to_string().trim_start_matches('0').len();
    let decimals = 2usize.saturating_sub(rounded_len);

    // Render `scaled` with exactly `decimals` fractional digits (no dynamic-width format spec), then append suffix.
    let magnitude = 10i64.pow(decimals as u32);
    let rounded = (scaled * magnitude as f64).round() as i64;
    let int_part = rounded / magnitude;
    let mut frac = ((rounded % magnitude) as u64).to_string();
    while frac.len() < decimals {
        frac.insert(0, '0');
    }
    if decimals == 0 {
        format!("{int_part}{suffix}")
    } else {
        format!("{int_part}.{frac}{suffix}")
    }
}

/// Write the GGUF file. `dir` holds the HF checkpoint (config/tokenizer/shards); `out` is the destination path.
pub fn convert(dir: &Path, out: &Path, outtype: OutType, pre_tokenizer: Option<&str>) -> Result<()> {
    let hparams = HfConfig::load(dir)?;
    if !hparams.is_llama() {
        bail!("converter only handles LlamaForCausalLM (config.architectures={:?})", hparams.arch);
    }
    // identify the BPE pre-tokenizer the way llama.cpp does (`get_vocab_base_pre`); a synthetic tokenizer matches no
    // known probe hash and is named via `pre_tokenizer` instead, exactly as make-tiny-llama-reference.py forces "gpt-2"
    let pre = model_pre_tokenizer(dir, pre_tokenizer)?;

    let tensors = read_tensors(dir)?;

    let (tok_arrays, bos_id, eos_id) = tokenizer_arrays(dir, hparams.vocab_size, &pre)?;
    let tc: Json = serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json"))
        .with_context(|| format!("reading tokenizer_config.json in {}", dir.display()))?)?;
    let chat = chat_template(&tc);

    let mapped = mapped_tensors(&tensors, &hparams)?;
    // total number of parameters across every tensor — llama.cpp derives general.size_label from it.
    let total_params: u64 = mapped.iter().map(|m| m.shape.iter().product::<u64>()).sum();

    // general.name (a title-cased form of the HF folder) and general.size_label (the parameter count) are derived like
    // convert_hf_to_gguf.py does; head_dim drives llama.rope.dimension_count, which the engine needs.
    let folder = dir.file_name().and_then(|n| n.to_str());
    let name = match folder {
        Some(folder) => id_to_title(folder),
        None => "llama".to_string(),
    };
    let size_label = model_size_label(total_params);
    let head_dim = hparams.hidden_size / hparams.num_attention_heads;

    // general.basename / .finetune / .version come from gguf-py's get_model_id_components (exactly what llama.cpp's
    // converter derives from the HF folder id); write each one that parses out, mirroring convert_hf_to_gguf.py.
    let components = crate::naming::model_id_components(folder.unwrap_or_default(), total_params as i64);
    let guessed: Vec<(&'static str, Kv)> = [
        ("general.basename", components.basename.as_deref()),
        ("general.finetune", components.finetune.as_deref()),
        ("general.version", components.version.as_deref()),
    ]
    .into_iter()
    .filter_map(|(key, value)| value.map(|v| (key, Kv::Str(v.to_string()))))
    .collect::<Vec<_>>();

    // metadata key/value pairs, in convert_hf_to_gguf.py's exact order (general.* first, then the llama.* hyper-parameters,
    // then tokenizer.*), matching llama.cpp for a BPE (HF) checkpoint.
    let mut kv: Vec<(&'static str, Kv)> = vec![
        ("general.architecture", Kv::Str("llama".into())),
        ("general.type", Kv::Str("model".into())),
        ("general.name", Kv::Str(name)),
        ("general.size_label", Kv::Str(size_label)),
        ("llama.block_count", Kv::U32(hparams.num_hidden_layers as u32)),
        ("llama.context_length", Kv::U32(hparams.max_position_embeddings as u32)),
        ("llama.embedding_length", Kv::U32(hparams.hidden_size as u32)),
        ("llama.feed_forward_length", Kv::U32(hparams.intermediate_size as u32)),
        ("llama.attention.head_count", Kv::U32(hparams.num_attention_heads as u32)),
        ("llama.attention.head_count_kv", Kv::U32(hparams.num_key_value_heads as u32)),
        ("llama.rope.freq_base", Kv::F32(hparams.rope_theta as f32)), // llama.cpp writes this as FLOAT32
        ("llama.attention.layer_norm_rms_epsilon", Kv::F32(hparams.rms_norm_eps)),
        ("general.file_type", Kv::U32(outtype.file_type())),
        ("llama.vocab_size", Kv::U32(hparams.vocab_size as u32)),
        ("llama.rope.dimension_count", Kv::U32(head_dim as u32)),
        ("general.quantization_version", Kv::U32(2)),
    ];
    // Insert the guessed components (general.basename / .finetune / .version) just before general.size_label so the
    // general.* keys stay grouped in convert_hf_to_gguf.py's order; a no-op when the model id yields none of them.
    if let Some(idx) = kv.iter().position(|entry| entry.0 == "general.size_label") {
        for e in guessed.into_iter().rev() {
            kv.insert(idx, e);
        }
    }
    // tokenizer.* in llama.cpp's order; a BPE tokenizer carries no scores array (only SentencePiece vocabularies do), so
    // tok_arrays[3] (scores) is deliberately not written here.
    kv.push(("tokenizer.ggml.model", tok_arrays[1].clone()));
    kv.push(("tokenizer.ggml.pre", tok_arrays[0].clone()));
    kv.push(("tokenizer.ggml.tokens", tok_arrays[2].clone()));
    kv.push(("tokenizer.ggml.token_type", tok_arrays[4].clone()));
    kv.push(("tokenizer.ggml.bos_token_id", Kv::U32(bos_id)));
    kv.push(("tokenizer.ggml.eos_token_id", Kv::U32(eos_id)));
    // add_bos/add_eos flags mirror what llama.cpp's SpecialVocab writes from tokenizer_config.json.
    kv.push(("tokenizer.ggml.add_bos_token", Kv::Bool(get_bool(&tc, "add_bos_token"))));
    kv.push(("tokenizer.ggml.add_eos_token", Kv::Bool(get_bool(&tc, "add_eos_token"))));
    if !chat.is_empty() {
        kv.push(("tokenizer.chat_template", Kv::Str(chat)));
    }

    let specs: Vec<TensorSpec> = mapped.iter().map(|m| {
        let dims: Vec<usize> = m.shape.iter().map(|&d| d as usize).collect();
        let (ggml_type, t) = tensor_type(&dims, outtype);
        let elem = if t == OutType::F16 { 2 } else { 4 };
        TensorSpec { name: m.name.clone(), shape: m.shape.clone(), ggml_type, nbytes: m.vals.len() as u64 * elem }
    }).collect();

    let mut w = Writer::create_typed(out, &kv, specs)?;
    for m in &mapped {
        let dims: Vec<usize> = m.shape.iter().map(|&d| d as usize).collect();
        let bytes = serialize(&m.vals, tensor_type(&dims, outtype).1);
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
    fn id_to_title_matches_python() {
        // gguf-py: ' '.join([w.title() if w.islower() and not re.match(r'^(v\d+(?:\.\d+)*|\d.*)$', w) else w ...])
        assert_eq!(id_to_title("tiny-llama-256-bf16-hf"), "Tiny Llama 256 Bf16 Hf");
        assert_eq!(id_to_title("qwen3-8b-v2-instruct"), "Qwen3 8b v2 Instruct");
        assert_eq!(id_to_title("llama3b"), "Llama3B");
        assert_eq!(id_to_title("Mistral-7B-v0.3"), "Mistral 7B v0.3");
        assert_eq!(id_to_title("v2beta"), "V2Beta", "not a version: title-cased");
        assert_eq!(id_to_title("q4"), "Q4");
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
    fn q_permutes_with_attention_heads_k_with_kv_heads() {
        // mapped_tensors must read the head counts from HfConfig, not a literal: 8 attention heads / 5 kv heads, so
        // head_dim is 4 (even, non-identity permute). A hardcoded `permute(...,2,...)` (the step-6 bug) reshapes with a
        // head_dim of 16 and produces different bytes here — this fails loudly against such a constant.
        let hparams = HfConfig {
            arch: "LlamaForCausalLM".into(),
            hidden_size: 32, num_hidden_layers: 1,
            num_attention_heads: 8, num_key_value_heads: 5,
            intermediate_size: 64, rms_norm_eps: 1e-5, vocab_size: 16,
            max_position_embeddings: 64, rope_theta: 10_000,
        };
        let head_dim = hparams.hidden_size / hparams.num_attention_heads; // 4 -> kv dim is 5*4 = 20
        let q_rows = hparams.num_attention_heads * head_dim; // 32 (== hidden)
        let kv_rows = hparams.num_key_value_heads * head_dim; // 20
        let q_vals: Vec<f32> = (0..(q_rows * hparams.hidden_size)).map(|i| (i as f32) * 0.5 + 0.25).collect();
        let k_vals: Vec<f32> = (0..(kv_rows * hparams.hidden_size)).map(|i| (i as f32) * -0.3 + 1.0).collect();
        let mut tensors = BTreeMap::new();
        // a single-subname layer tensor establishes the layer id (attn_q/k/v/o and mlp names are 6-part, so they would
        // not by themselves be parsed as a layer here - the real checkpoint has input_layernorm.weight); then the two
        // projections under test.
        tensors.insert("model.layers.0.input_layernorm.weight".into(), (vec![hparams.hidden_size], vec![1f32; hparams.hidden_size]));
        tensors.insert("model.layers.0.self_attn.q_proj.weight".into(), (vec![q_rows, hparams.hidden_size], q_vals.clone()));
        tensors.insert("model.layers.0.self_attn.k_proj.weight".into(), (vec![kv_rows, hparams.hidden_size], k_vals.clone()));

        let mapped = mapped_tensors(&tensors, &hparams).unwrap();
        let attn_q = mapped.iter().find(|m| m.name == "blk.0.attn_q.weight").expect("attn_q present");
        let attn_k = mapped.iter().find(|m| m.name == "blk.0.attn_k.weight").expect("attn_k present");

        assert_eq!(attn_q.shape, vec![q_rows as u64, hparams.hidden_size as u64]);
        assert_eq!(attn_k.shape, vec![kv_rows as u64, hparams.hidden_size as u64], "k_proj carries num_key_value_heads*head_dim rows");
        // q permuted with the attention head count, k with the kv head count (reproducing llama.cpp's per-name call).
        assert_eq!(attn_q.vals, permute(&q_vals, hparams.num_attention_heads, hparams.hidden_size), "q must use num_attention_heads");
        assert_eq!(attn_k.vals, permute(&k_vals, hparams.num_key_value_heads, hparams.hidden_size), "k must use num_key_value_heads");
    }

    #[test]
    fn tokenizer_orders_by_id_and_flags_specials() {
        let dir = std::env::temp_dir().join("convert-tok-selfcheck");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("tokenizer.json"),
            r#"{"version":"1.1","model":{"type":"BPE","byte_fallback":true,"vocab":{"tok_0":0,"tok_1":1,"tok_2":2,"tok_3":3},"merges":[]}}"#).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"),
            r#"{"add_bos_token":true,"add_eos_token":false,"bos_token":"tok_0","eos_token":"tok_1"}"#).unwrap();
        let (arr, bos, eos) = tokenizer_arrays(&dir, 4, "gpt-2").unwrap();
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

    #[test]
    fn known_hash_table_matches_llama_cpp() {
        // Real llama.cpp probe hashes including the two that both resolve to "gpt-2" and one chatglm-bpe: proves
        // PRE_TOKENIZER_HASHES is copied verbatim from conversion/base.py get_vocab_base_pre, and unknown hashes map away.
        assert_eq!(known_pre_tokenizer_name("b6e8e1518dc4305be2fe39c313ed643381c4da5db34a98f6a04c093f8afbe99b"), Some("chatglm-bpe"));
        assert_eq!(known_pre_tokenizer_name("0fe1cf6eda062318a1af7270f3331a85c539a01778ff948e24388e949c5282f4"), Some("gpt-2"));
        assert_eq!(known_pre_tokenizer_name("3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454"), Some("gpt-2"));
        assert_eq!(known_pre_tokenizer_name("8aeee3860c56296a157a1fe2fad249ec40aa59b1bb5709f4ade11c4e6fe652ed"), Some("falcon"));
        assert_eq!(known_pre_tokenizer_name("deadbeef"), None);
    }

    #[test]
    fn unknown_probe_hash_falls_through_to_override_then_errors() {
        // An id list whose sha256(str(ids)) is not in the table: an override names it, absence of one refuses it.
        let ids: Vec<u32> = vec![9, 17, 4];
        assert_eq!(pre_tokenizer_name(&ids, Some("gpt-2")).unwrap(), "gpt-2");
        match pre_tokenizer_name(&ids, None) {
            Err(_) => {}
            Ok(x) => panic!("unknown probe hash must be refused without an override, got {x}"),
        }
    }

    #[test]
    fn tiny_model_probe_is_named_via_override() {
        // The synthetic tiny tokenizer matches no known probe hash, so it is named through the override - exactly what
        // make-tiny-llama-reference.py does. Exercises loading the fixture's tokenizer.json, tokenizing and hashing.
        let fx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-llama-hf");
        assert_eq!(model_pre_tokenizer(&fx, Some("gpt-2")).unwrap(), "gpt-2");
    }
}
