//! Step 5 — fold a LoRA adapter into its base safetensors, shard by shard.
//!
//! The base is opened with [`crate::safetensors_io`] (each shard mmap'd, never the whole model loaded at once). Every
//! base weight gets each trained adapter's delta `scaling * B @ A` added to it (PEFT folds the adapter into the base on
//! export); everything else — norms, MLP, embeddings, and untargeted projections — is copied back byte for byte. The
//! shards come out under the base's own names with the same split they had going in (plus an index file when the base
//! was sharded), so a merged checkpoint drops straight into the engine.
//!
//! * **Scaling** (`LoraConfig::scaling`): `alpha / r`, or `alpha / sqrt(r)` with rsLoRA, read from `adapter_config.json`.
//! * **Name mapping** ([`peft_to_base_name`]): drop `base_model.model.` and the `.lora_A.weight`/`.lora_B.weight` suffix,
//!   then add `.weight`, so `...q_proj.lora_B.weight` -> `...q_proj.weight`.
//! * **2-D targets** (the q/k/v/o projections): `delta[o][i] = scaling * sum_r B[o][r] * A[r][i]`.
//! * **Fused MoE experts** ([`add_expert_delta`]): a 3-D `[E, d1, d2]` target takes the expert einsum `einsum("o r e, e r i -> e*")`, per-expert
//!   `scaling * B_e @ A_e`. PEFT has two layouts (`-> e i o` by default, `-> e o i` after `_did_swap_in_out_features`) and the fold picks the one
//!   whose output shape matches the parameter's own `[E, d1, d2]`, refusing a square target its layouts are indistinguishable on.
//! * **Report** ([`MergeReport`]): per tensor the relative-norm delta `|ΔW|/|W|` — targeted tensors are > 0, untargeted
//!   ones are exactly 0 (their bytes never left disk).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use serde_json::{json, Value};

use crate::checkpoint;
use crate::safetensors_io::{self, Dtype, Shard};

/// One base weight and what it became after the merge: same shape, plus how big its relative change was.
#[derive(Debug, Clone)]
pub struct TensorDelta {
    pub name: String,
    pub shape: Vec<usize>,
    /// |ΔW| / |W| over the original base weight; 0.0 for a tensor the adapter never touched.
    pub delta_rel_norm: f64,
}

/// The outcome of [`merge`]: where the merged checkpoint was written and a per-tensor relative-norm summary.
pub struct MergeReport {
    pub out_dir: PathBuf,
    pub per_tensor: Vec<TensorDelta>,
}

/// A LoRA delta pair for one base weight, resolved from the adapter's A/B matrices. The 2-D vs fused-MoE path is chosen by
/// the base weight's rank at merge time ([`run`]).
#[derive(Clone)]
struct Adapter {
    a: Vec<f32>,
    b: Vec<f32>,
}

/// A base weight read from a shard: its name, dtype, shape, and the raw bytes straight off the mmap.
struct BaseTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

/// Merge `adapter_dir`'s adapter into every base shard under `base_dir`, writing the result to `out_dir`. Scaling is read
/// from `adapter_config.json`; set a [`Merger`] with [`Merger::set_scaling`] to fold in a deliberately wrong factor.
pub struct Merger<'a> {
    base_dir: &'a Path,
    adapter_dir: &'a Path,
    out_dir: &'a Path,
    scaling_override: Option<f32>,
}

impl<'a> Merger<'a> {
    pub fn new(base_dir: &'a Path, adapter_dir: &'a Path, out_dir: &'a Path) -> Self {
        Self { base_dir, adapter_dir, out_dir, scaling_override: None }
    }

    /// Force a scaling factor instead of the one in `adapter_config.json` (used by the rsLoRA gate to prove the default
    /// scaling is what makes the merged logits match). The wrong factor must fail the logit-parity gate.
    pub fn set_scaling(&mut self, s: f32) -> &mut Self {
        self.scaling_override = Some(s);
        self
    }

    pub fn run(self) -> Result<MergeReport> {
        let scaling = match self.scaling_override {
            Some(s) => s,
            None => load_scaling(self.adapter_dir)?,
        };
        // The whole model is never resident at once: enumerate the base shards in order and read every weight off its mmap,
        // tagged with the shard it came from so the output reproduces the same split. Only the tiny adapter (already on the
        // CPU) is fully materialised into Rust.
        let fname = base_first_name(self.base_dir)?;
        let first = self.base_dir.join(fname);
        let shard_files = safetensors_io::shards(&first);
        let n_shards = shard_files.len();

        let adapters = load_adapters(self.adapter_dir)?;
        let mut shards: Vec<Vec<BaseTensor>> = (0..n_shards).map(|_| Vec::new()).collect();
        for (i, sf) in shard_files.iter().enumerate() {
            let shard = Shard::open(sf)?;
            for t in shard.tensors() {
                shards[i].push(BaseTensor {
                    name: t.name().to_string(),
                    dtype: t.dtype(),
                    shape: t.shape().to_vec(),
                    bytes: shard.bytes(t).to_vec(),
                });
            }
        }

        std::fs::create_dir_all(self.out_dir)?;
        let mut report: Vec<TensorDelta> = Vec::new();
        // weight-map entries for a multi-shard index (name -> file, dtype, shape, in-shard offset span).
        let mut weight_map: serde_json::Map<String, Value> = serde_json::Map::new();

        for (i, in_shard) in shards.into_iter().enumerate() {
            let fname = format!("model-{:05}-of-{:05}.safetensors", i + 1, n_shards);

            // pack: for each weight either copy its mmap bytes unchanged (untargeted) or rewrite with the folded delta.
            let mut payload: Vec<u8> = Vec::new();
            let mut header: serde_json::Map<String, Value> = serde_json::Map::new();
            for wt in in_shard {
                let (data, rel_norm): (Vec<u8>, f64) = match adapters.get(&wt.name) {
                    None => (wt.bytes.clone(), 0.0), // untouched: the mmap bytes go back out verbatim
                    Some(ad) => match wt.shape.len() {
                        2 => {
                            let o = wt.shape[0];
                            let in_ = wt.shape[1];
                            let r = ad.a.len().div_ceil(in_);
                            let base = le_f32s(&wt.bytes)?;
                            let (w, delta) = add_delta_2d(&base, &ad.a, &ad.b, o, in_, r, scaling);
                            (f32_le_bytes(&w), rel_norm(&base, &delta))
                        }
                        3 => {
                            let e = wt.shape[0];
                            let d1 = wt.shape[1];
                            let d2 = wt.shape[2];
                            let base = le_f32s(&wt.bytes)?;
                            let (w, delta) = add_expert_delta(&wt.name, &base, &ad.a, &ad.b, e, d1, d2, scaling)?;
                            (f32_le_bytes(&w), rel_norm(&base, &delta))
                        }
                        other => return Err(anyhow!("{}: base weight of rank {other} is not a LoRA target", wt.name)),
                    },
                };
                let n = data.len();
                let start = payload.len();
                payload.extend_from_slice(&data);
                header.insert(wt.name.clone(), json!({
                    "dtype": wt.dtype.as_str(),
                    "shape": wt.shape,
                    "data_offsets": [start, start + n],
                }));
                report.push(TensorDelta { name: wt.name.clone(), shape: wt.shape.clone(), delta_rel_norm: rel_norm });
                if n_shards > 1 {
                    weight_map.insert(wt.name.clone(), json!({
                        "dtype": wt.dtype.as_str(),
                        "shape": wt.shape,
                        "data_offsets": [start, start + n],
                        "file": fname.clone(),
                    }));
                }
            }

            let out = write_shard(&payload, &header)?;
            std::fs::write(self.out_dir.join(fname.clone()), out)?;
        }

        if n_shards > 1 {
            let index = json!({ "format": "PT", "total": weight_map.len(), "weight_map": weight_map });
            std::fs::write(self.out_dir.join("model.safetensors.index.json"), serde_json::to_string_pretty(&index)?)?;
        }

        Ok(MergeReport { out_dir: self.out_dir.to_path_buf(), per_tensor: report })
    }
}

/// Shorthand for [`Merger::run`]: fold `adapter_dir`'s adapter into the base under `base_dir`, writing to `out_dir`.
pub fn merge(base_dir: &Path, adapter_dir: &Path, out_dir: &Path) -> Result<MergeReport> {
    Merger::new(base_dir, adapter_dir, out_dir).run()
}

/// Map a PEFT adapter tensor name to the base weight it folds into. Drop `base_model.model.` and the `.lora_{A,B}.weight`
/// suffix, then re-add `.weight`: `...q_proj.lora_B.weight` -> `...q_proj.weight`. Returns `None` for anything not a LoRA
/// matrix (a plain base weight never carries this shape anyway).
pub fn peft_to_base_name(peft: &str) -> Option<String> {
    let s = peft.strip_prefix("base_model.model.")?;
    if s.ends_with(".lora_A.weight") || s.ends_with(".lora_B.weight") {
        Some(format!("{}.weight", s.strip_suffix(".lora_A.weight").or_else(|| s.strip_suffix(".lora_B.weight"))?))
    } else {
        None
    }
}

/// Build the adapter map (PEFT name -> [`Adapter`]) from the loaded A/B matrices. Only fused MoE targets are 3-D; the merge
/// routes by the base weight's rank at fold time, so here both shapes arrive as flat f32 and dispatch happens later.
fn build_adapters(adps: &std::collections::HashMap<String, Tensor>) -> Result<HashMap<String, Adapter>> {
    let mut a_of: HashMap<String, Vec<f32>> = HashMap::new();
    let mut b_of: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, t) in adps {
        let flat = t.flatten_all()?.to_vec1::<f32>()?;
        if name.ends_with(".lora_A.weight") {
            a_of.insert(peft_to_base_name(name).ok_or_else(|| anyhow!("{name}: not a LoRA matrix"))?, flat);
        } else {
            b_of.insert(peft_to_base_name(name).ok_or_else(|| anyhow!("{name}: not a LoRA matrix"))?, flat);
        }
    }
    let mut out = HashMap::new();
    for (base, a) in &a_of {
        let b = b_of.get(base).ok_or_else(|| anyhow!("adapter has lora_A for {base} but no lora_B"))?;
        out.insert(base.clone(), Adapter { a: a.clone(), b: b.clone() });
    }
    Ok(out)
}

/// Fold `scaling * B @ A` into a 2-D weight. Row-major: `A:[r,in]`, `B:[out,r]`, `W:[out,in]`. Returns the new weight and
/// the delta (same length) so the caller can report the relative norm.
fn add_delta_2d(base: &[f32], a: &[f32], b: &[f32], out: usize, in_: usize, r: usize, scaling: f32) -> (Vec<f32>, Vec<f32>) {
    let mut w = base.to_vec();
    let mut delta = vec![0f32; out * in_];
    for o in 0..out {
        for i in 0..in_ {
            let mut acc = 0f32;
            for rr in 0..r {
                acc += b[o * r + rr] * a[rr * in_ + i];
            }
            delta[o * in_ + i] = scaling * acc;
        }
    }
    for (wi, di) in w.iter_mut().zip(delta.iter()) {
        *wi += *di;
    }
    (w, delta)
}

/// Fold `scaling * B @ A` into a fused-MoE expert weight of the parameter's own shape `[E, d1, d2]` — which is also the delta's.
/// PEFT computes each expert `scaling * B_e @ A_e` with one of two layouts: default `einsum("o r e, e r i -> e i o")`, a delta of
/// shape `[E, in, out]`, and after `_did_swap_in_out_features` `einsum("o r e, e r i -> e o i")`, a delta of `[E, out, in]`. The
/// flattened A is `[E*r, a_in]` and B is `[b_out, r*E]` either way, so `(a_in, b_out)` is the parameter's inner dims in some order:
/// pick the layout matching it against `(d1, d2)`, refuse when `d1 == d2` (both layouts look identical from the files), and refuse
/// anything that matches neither. `name` names the tensor in those error messages.
fn add_expert_delta(name: &str, base: &[f32], a: &[f32], b: &[f32], e: usize, d1: usize, d2: usize, scaling: f32) -> Result<(Vec<f32>, Vec<f32>)> {
    let len_a = a.len();
    let len_b = b.len();

    // Default layout `-> e i o`: (a_in, b_out) == (d1, d2). The result is placed at delta[e][ii][oo], so base dim d1 is `in`, d2 is `out`.
    if let Some(r) = layout_rank(len_a, e, d1, len_b, d2) {
        if d1 == d2 {
            return Err(anyhow!("{name}: fused-expert parameter is square [{e}, {d1}, {d2}]; PEFT's two expert layouts are indistinguishable from the shapes alone, so refuse rather than guess"));
        }
        return Ok(expert_delta(base, a, b, e, d2, d1, false, r, scaling));
    }

    // Swapped layout `-> e o i`: (a_in, b_out) == (d2, d1). The result is placed at delta[e][oo][ii], so base dim d1 is `out`, d2 is `in`.
    if let Some(r) = layout_rank(len_a, e, d2, len_b, d1) {
        return Ok(expert_delta(base, a, b, e, d1, d2, true, r, scaling));
    }

    if d1 == d2 {
        return Err(anyhow!("{name}: fused-expert parameter is square [{e}, {d1}, {d2}]; PEFT's two expert layouts are indistinguishable from the shapes alone, so refuse rather than guess"));
    }
    Err(anyhow!("{name}: fused-expert adapter A[E*r, {len_a}/{e}] and B[{len_b}/{e}, r*E] don't fit parameter shape [{e}, {d1}, {d2}]"))
}

/// Consistent rank when A is `[E*r, a_in]` and B is `[b_out, r*E]`: returns `r` only if both flat lengths divide evenly on it and the
/// two independent estimates agree — this is what disambiguates the two PEFT layouts from shapes alone.
fn layout_rank(len_a: usize, e: usize, a_in: usize, len_b: usize, out_b: usize) -> Option<usize> {
    if e == 0 || a_in == 0 || out_b == 0 {
        return None;
    }
    if len_a % (e * a_in) != 0 || len_b % (e * out_b) != 0 {
        return None;
    }
    let r_a = len_a / (e * a_in);
    let r_b = len_b / (e * out_b);
    if r_a == r_b { Some(r_a) } else { None }
}

/// Per-expert `scaling * B_e @ A_e` folded into `base`, writing the result as `[E, out_dim, in_dim]` when `o_first` is true (delta[e][oo][ii],
/// base dim d1 is the outer `out`) or `[E, in_dim, out_dim]` when false (delta[e][ii][oo]). A is `[E*r, a_in=in_dim]`, B is `[b_out=out_dim, r*E]`;
/// both axes are contracted over rank `r`.
fn expert_delta(base: &[f32], a: &[f32], b: &[f32], e: usize, out_dim: usize, in_dim: usize, o_first: bool, r: usize, scaling: f32) -> (Vec<f32>, Vec<f32>) {
    let mut w = base.to_vec();
    let mut delta = vec![0f32; e * out_dim * in_dim];
    let re = b.len() / out_dim; // r*E: stride pairing B's contracted rank with its expert axis
    for ee in 0..e {
        for oo in 0..out_dim {
            for ii in 0..in_dim {
                let mut acc = 0f32;
                // b reshaped [out,r,E]: element[oo][rr][ee] lives at flat oo*(r*E) + rr*E + ee
                for rr in 0..r {
                    let be = b[oo * re + rr * e + ee];
                    // a reshaped [E,r,in]: element[ee][rr][ii] lives at flat (ee*r + rr)*in + ii
                    let ae = a[(ee * r + rr) * in_dim + ii];
                    acc += be * ae;
                }
                if o_first {
                    delta[(ee * out_dim + oo) * in_dim + ii] = scaling * acc; // delta[e][oo][ii], base dim d1 is `out`
                } else {
                    delta[(ee * in_dim + ii) * out_dim + oo] = scaling * acc; // delta[e][ii][oo], base dim d1 is `in`
                }
            }
        }
    }
    for (wi, di) in w.iter_mut().zip(delta.iter()) {
        *wi += *di;
    }
    (w, delta)
}

/// Relative L2 norm of `delta` over the original weight `w` (unchanged by the caller).
fn rel_norm(w: &[f32], delta: &[f32]) -> f64 {
    let wn = w.iter().fold(0.0_f64, |a, &x| a + x as f64 * x as f64);
    let dn = delta.iter().fold(0.0_f64, |a, &x| a + x as f64 * x as f64);
    (dn.sqrt() / wn.sqrt()).clamp(0.0, f64::INFINITY)
}

/// Scaling from `adapter_config.json`: `alpha / r`, or `alpha / sqrt(r)` when rsLoRA is on.
fn load_scaling(adapter_dir: &Path) -> Result<f32> {
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(adapter_dir.join(checkpoint::CONFIG_FILE))?)?;
    let r = cfg["r"].as_u64().ok_or_else(|| anyhow!("adapter_config.json has no r"))?;
    let alpha = cfg["lora_alpha"].as_f64().ok_or_else(|| anyhow!("adapter_config.json has no lora_alpha"))? as f32;
    Ok(apply_scaling(alpha, r, cfg["use_rslora"].as_bool().unwrap_or(false)))
}

/// The scaling factor for rank `r`, alpha and rsLoRA flag — duplicated from [`crate::lora::LoraConfig`] so the merge works
/// off the on-disk config alone.
fn apply_scaling(alpha: f32, r: u64, use_rslora: bool) -> f32 {
    let r = r as f32;
    if use_rslora {
        alpha / r.sqrt()
    } else {
        alpha / r
    }
}

/// Load the adapter's A/B matrices (flat f32) under their PEFT names.
fn load_adapters(adapter_dir: &Path) -> Result<HashMap<String, Adapter>> {
    let adps = candle_core::safetensors::load(adapter_dir.join(checkpoint::ADAPTER_FILE), &Device::Cpu)?;
    build_adapters(&adps)
}

/// Write one safetensors shard and return its raw bytes. `header` is the JSON map of name -> {dtype, shape, data_offsets},
/// already carrying each tensor's offset in the contiguous payload; a u64 length prefix precedes the header. Matches the
/// reader in [`crate::safetensors_io`] byte for byte, so an untouched tensor round-trips unchanged.
fn write_shard(payload: &[u8], header: &serde_json::Map<String, Value>) -> Result<Vec<u8>> {
    let header_json = json!(header).to_string();
    let mut out = Vec::with_capacity(8 + header_json.len() + payload.len());
    // safetensors layout: u64 (LE) length of the JSON header that follows, then the header, then the tensor bytes.
    out.extend_from_slice((header_json.len() as u64).to_le_bytes().as_slice());
    out.extend_from_slice(header_json.as_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

/// f32 -> little-endian bytes.
fn f32_le_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// little-endian bytes -> f32 (the base fixture is all F32).
fn le_f32s(b: &[u8]) -> Result<Vec<f32>> {
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// The first base shard's file name (`model-00001-of-0000N.safetensors`).
fn base_first_name(base_dir: &Path) -> Result<String> {
    let files = safetensors_io::shards(&base_dir.join("model-00001-of-00001.safetensors"));
    let first = files
        .first()
        .ok_or_else(|| anyhow!("no base shards in {}", base_dir.display()))?;
    Ok(first.file_name().unwrap().to_string_lossy().to_string())
}

#[cfg(test)]
#[path = "merge_unit.rs"]
mod unit_tests;
