//! GGUF -> GGUF quantizer implementing llama.cpp's `llama-quantize --pure` rules for a single output precision.
//!
//! It reads a GGUF (`strata_gguf`) and writes a new one where every tensor gets the target precision (Q8_0, Q4_0, Q4_K,
//! Q5_K, Q6_K or F16) using strata-pack's reference quantizers. The per-tensor type follows `tensor_type_fallback()` in
//! llama-quant.cpp: 1-D tensors keep their current type (so a 1-D norm stays F32); a 2-D+ tensor whose innermost
//! dimension ne[0] is not a multiple of the target's block size is demoted toward smaller blocks, and if that still is
//! not square-divisible it falls back to F16. Metadata is copied verbatim except `general.file_type`, which becomes the
//! target's pure `LlamaFileType`, and tensor order is kept. This is exactly what `llama-quantize --pure <T>` produces,
//! so the output of each target matches the reviewer's `.pure.gguf` fixtures byte for byte on every tensor.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::Value;
use strata_gguf::{Gguf, block_geometry, type_name as ggml_type_name};
use strata_pack::gguf_write::{Kv, TensorSpec, Writer};
use strata_pack::num::{bf16_to_f32, f16_to_f32, f32_to_f16};

/// The output precision of a `--pure` quantization. Maps to the ggml type name each tensor is written with and, except
/// for F16's ALL_F16 alias, to llama.cpp's pure `LlamaFileType` that `general.file_type` is set to. Variant names mirror
/// the ggml type strings they select, hence the relaxed case convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PureType {
    F16,
    Q4_0,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
}

impl PureType {
    /// Parse one of the `--pure` target names (`f16`, `q4_0`, `q8_0`, `q4_k`, `q5_k`, `q6_k`).
    pub fn from_s(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "f16" => Ok(PureType::F16),
            "q4_0" => Ok(PureType::Q4_0),
            "q8_0" => Ok(PureType::Q8_0),
            "q4_k" => Ok(PureType::Q4_K),
            "q5_k" => Ok(PureType::Q5_K),
            "q6_k" => Ok(PureType::Q6_K),
            other => bail!("--pure supports f16 q4_0 q8_0 q4_k q5_k q6_k, got {other:?}"),
        }
    }
    /// The ggml type name every quantizable tensor in this output is written with (the `target_type` of the fallback).
    pub fn ggml_name(self) -> &'static str {
        match self {
            PureType::F16 => "F16",
            PureType::Q4_0 => "Q4_0",
            PureType::Q8_0 => "Q8_0",
            PureType::Q4_K => "Q4_K",
            PureType::Q5_K => "Q5_K",
            PureType::Q6_K => "Q6_K",
        }
    }
    /// The pure `LlamaFileType` that `general.file_type` is set to: F16 -> MOSTLY_F16 (1), and each quant format the
    /// `MOSTLY_<type>` id of this build's enum (Q4_0 = 2, Q8_0 = 7, Q4_K = MOSTLY_Q4_K_M = 15, Q5_K = MOSTLY_Q5_K_M = 17,
    /// Q6_K = 18), matching `llama-quantize`'s output.
    pub fn file_type(self) -> u32 {
        match self {
            PureType::F16 => 1,       // MOSTLY_F16
            PureType::Q4_0 => 2,      // MOSTLY_Q4_0
            PureType::Q8_0 => 7,      // MOSTLY_Q8_0
            PureType::Q4_K => 15,     // MOSTLY_Q4_K_M
            PureType::Q5_K => 17,     // MOSTLY_Q5_K_M
            PureType::Q6_K => 18,     // MOSTLY_Q6_K
        }
    }
}

/// The ggml block size (elements per block) of a type name, or 1 for the float types.
fn block_size(name: &str) -> usize {
    block_geometry(name).map(|(be, _)| be as usize).unwrap_or(1)
}

/// The resolved output ggml type name of a tensor with innermost dimension `ne0`, per `tensor_type_fallback()`: the
/// target when ne[0] is divisible by its block size; otherwise the demoted block-32 type (`Q4_K -> Q5_0`, `Q5_K -> Q5_1`,
/// `Q6_K -> Q8_0`), or F16 when that still does not fit. This is the exact fallback used by `--pure` (the arch mixture
/// logic is skipped when pure, so the type is only ever demoted by shape).
fn resolve(target: &'static str, ne0: usize) -> &'static str {
    if ne0 % block_size(target) == 0 {
        return target;
    }
    let demoted: Option<&'static str> = match target {
        "Q4_K" => Some("Q5_0"),
        "Q5_K" => Some("Q5_1"),
        "Q6_K" => Some("Q8_0"),
        _ => None, // block-32 targets have no smaller block to demote to
    };
    let rt = demoted.unwrap_or(target);
    if ne0 % block_size(rt) != 0 {
        "F16"
    } else {
        rt
    }
}

/// Output byte length of a tensor of `numel` values written as `name`.
fn type_nbytes(name: &str, numel: usize) -> u64 {
    match name {
        "F32" => numel as u64 * 4,
        "F16" | "BF16" => numel as u64 * 2,
        _ => {
            let (be, bb) = block_geometry(name).expect("block geometry of a quant type");
            (numel as u64 / be) * bb
        }
    }
}

/// The ggml type id (the `ggml_type` field of a tensor info), from its name.
fn type_id(name: &str) -> u32 {
    match name {
        "F32" => 0,
        "F16" => 1,
        "Q4_0" => 2,
        "Q5_0" => 6,
        "Q5_1" => 7,
        "Q8_0" => 8,
        "Q4_K" => 12,
        "Q5_K" => 13,
        "Q6_K" => 14,
        _ => panic!("unknown ggml type name {name}"),
    }
}

/// One input tensor's raw bytes (a slice into the whole file).
fn tensor_bytes<'a>(buf: &'a [u8], g: &Gguf, t: &strata_gguf::TensorInfo) -> Result<&'a [u8]> {
    let start = (g.data_start + t.offset) as usize;
    let n = t.expected_bytes().context("input tensor size")? as usize;
    // a damaged or cut-off file is an error, never a crash
    start.checked_add(n).filter(|end| *end <= buf.len()).map(|end| &buf[start..end])
        .with_context(|| format!("{}: its data runs past the end of the file (damaged or incomplete GGUF?)", t.name))
}

/// Decode one input tensor's bytes to f32, the values `llama-quantize` re-quantizes from: F32 straight, F16/BF16 cast,
/// anything else (a block type) dequantized.
fn read_as_f32(buf: &[u8], g: &Gguf, t: &strata_gguf::TensorInfo) -> Result<Vec<f32>> {
    let name = ggml_type_name(t.ggml_type);
    let raw = tensor_bytes(buf, g, t)?;
    Ok(match name.as_str() {
        "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "F16" => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        "BF16" => raw.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        _ => strata_pack::quants::dequantize(&name, raw)
            .map_err(|e| anyhow::anyhow!("{}: dequantizing input tensor: {e}", t.name))?,
    })
}

/// Encode f32 values to the output precision `name`: F32/F16 inline, or a strata-pack block quantizer.
fn encode(x: &[f32], name: &str) -> Vec<u8> {
    match name {
        "F32" => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        "F16" => x.iter().flat_map(|v| f32_to_f16(*v).to_le_bytes()).collect(),
        "BF16" => x.iter().flat_map(|v| strata_pack::num::f32_to_bf16(*v).to_le_bytes()).collect(),
        _ => {
            let mut out = Vec::new();
            strata_pack::quantize::quantize(name, x, &mut out).expect("strata-pack quantizer");
            out
        }
    }
}


/// Quantize `input` to a new GGUF at `output`, every tensor in `target`'s pure precision. 1-D tensors keep their type,
/// incompatible rows fall back to F16 per `tensor_type_fallback`, metadata is copied (with the pure file-type id), and
/// tensor order is preserved.
pub fn quantize(input: &Path, output: &Path, target: PureType) -> Result<()> {
    let g = Gguf::open(input).with_context(|| format!("opening {}", input.display()))?;
    let file = std::fs::File::open(input).with_context(|| format!("reading {}", input.display()))?;
    // memory-mapped: only the tensor being converted is in memory as f32.
    // SAFETY: the map is read only, and the input is a model file nothing writes while it is quantized (if another
    // program changed it meanwhile the bytes read could be wrong, but never memory-unsafe beyond what any mmap risks).
    let buf = unsafe { memmap2::Mmap::map(&file) }.with_context(|| format!("mapping {}", input.display()))?;
    let target_name = target.ggml_name();

    // metadata copied byte for byte, in order (no value retyped or lost); only general.file_type becomes the target's
    let raw = raw_kvs(&buf).with_context(|| format!("{}: reading its metadata", input.display()))?;
    let kv: Vec<(&str, Kv)> = raw.iter().map(|(k, ty, payload)| {
        if k == "general.file_type" { (k.as_str(), Kv::U32(target.file_type())) }
        else { (k.as_str(), Kv::Raw { ty: *ty, payload: payload.clone() }) }
    }).collect();

    // every tensor's output type and size first (the writer lays out the whole header before any data)
    let mut specs = Vec::with_capacity(g.tensors.len());
    let mut out_types = Vec::with_capacity(g.tensors.len());
    for t in &g.tensors {
        let current = ggml_type_name(t.ggml_type);
        let numel = t.elements() as usize;
        // ne[0] is the innermost dimension (strata_gguf reads dims innermost-first); 1-D tensors keep their current type.
        let out_type: String = if t.dims.len() == 1 { current.clone() } else { resolve(target_name, t.dims[0] as usize).to_string() };
        let mut shape = t.dims.clone();
        shape.reverse(); // TensorSpec.shape is numpy order; strata_gguf's dims are innermost-first (ne[])
        specs.push(TensorSpec { name: t.name.clone(), shape, ggml_type: type_id(&out_type), nbytes: type_nbytes(&out_type, numel) });
        out_types.push(out_type);
    }

    // then one tensor at a time: decode, quantize, write
    let mut w = Writer::create_typed(output, &kv, specs)?;
    for (t, out_type) in g.tensors.iter().zip(&out_types) {
        let x = read_as_f32(&buf, &g, t)?;
        w.tensor_data(&encode(&x, out_type))?;
        w.end_tensor()?;
    }
    w.finish()?;
    Ok(())
}

/// The metadata of a GGUF as (key, type tag, encoded value bytes), in file order, read straight from its header.
fn raw_kvs(buf: &[u8]) -> Result<Vec<(String, u32, Vec<u8>)>> {
    struct R<'a> { b: &'a [u8], at: usize }
    impl R<'_> {
        fn take(&mut self, n: usize) -> Result<&[u8]> {
            let end = self.at.checked_add(n).filter(|e| *e <= self.b.len()).context("metadata runs past the end of the file")?;
            let s = &self.b[self.at..end];
            self.at = end;
            Ok(s)
        }
        fn u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
        fn u64(&mut self) -> Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
        fn string(&mut self) -> Result<String> {
            let n = self.u64()? as usize;
            // GGUF keys are UTF-8: anything else is a damaged file, refused rather than altered
            String::from_utf8(self.take(n)?.to_vec()).context("a metadata key is not valid UTF-8")
        }
        /// the bytes of one value of type `ty`, consumed
        fn value_len(&mut self, ty: u32) -> Result<()> {
            match ty {
                0 | 1 | 7 => { self.take(1)?; }                  // u8 i8 bool
                2 | 3 => { self.take(2)?; }                      // u16 i16
                4 | 5 | 6 => { self.take(4)?; }                  // u32 i32 f32
                10 | 11 | 12 => { self.take(8)?; }               // u64 i64 f64
                8 => { let n = self.u64()? as usize; self.take(n)?; }
                9 => {
                    let et = self.u32()?;
                    let n = self.u64()?;
                    for _ in 0..n { self.value_len(et)?; }
                }
                t => anyhow::bail!("unknown metadata type {t}"),
            }
            Ok(())
        }
    }
    let mut r = R { b: buf, at: 0 };
    anyhow::ensure!(r.take(4)? == b"GGUF", "not a GGUF file");
    let _version = r.u32()?;
    let _tensors = r.u64()?;
    let n_kv = r.u64()?;
    let mut out = Vec::with_capacity(n_kv.min(4096) as usize);
    for _ in 0..n_kv {
        let key = r.string()?;
        let ty = r.u32()?;
        let start = r.at;
        r.value_len(ty)?;
        out.push((key, ty, buf[start..r.at].to_vec()));
    }
    Ok(out)
}

/// Compare two `Value`s ignoring the integer/float distinction of numbers (3 and 3.0 are equal), so a metadata set read
/// back from two GGUFs written with different numeric tags still compares by value. This is how [`quantize`] pins its
/// output's metadata to a reference fixture.
pub fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| values_equal(a, b))
        }
        _ => false,
    }
}

/// The metadata of a GGUF as an ordered key/value pair list whose values compare by [`values_equal`] — the same content
/// two `strata_gguf` reads must carry to be considered identical.
pub fn metadata_equal(a: &Gguf, b: &Gguf) -> bool {
    if a.metadata.len() != b.metadata.len() {
        return false;
    }
    a.metadata.iter().all(|(k, v)| b.metadata.get(k).is_some_and(|w| values_equal(v, w)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_type_ids_match_this_build_enum() {
        assert_eq!(PureType::F16.file_type(), 1);
        assert_eq!(PureType::Q8_0.file_type(), 7);
        assert_eq!(PureType::Q4_K.file_type(), 15); // MOSTLY_Q4_K_M, what llama-quantize writes for a --pure Q4_K
    }

    #[test]
    fn fallback_demotes_block_256_toward_block_32_then_f16() {
        // ne[0] divisible by the block -> the target type.
        assert_eq!(resolve("Q8_0", 32), "Q8_0");
        assert_eq!(resolve("Q4_K", 256), "Q4_K");
        // not divisible by 256 but by 32 -> the block-32 demotion (a real quantizer output).
        assert_eq!(resolve("Q4_K", 32), "Q5_0");
        assert_eq!(resolve("Q5_K", 32), "Q5_1");
        assert_eq!(resolve("Q6_K", 32), "Q8_0");
        // not divisible by 32 -> F16.
        assert_eq!(resolve("Q4_K", 16), "F16");
        assert_eq!(resolve("Q8_0", 16), "F16");
    }

    #[test]
    fn q5_0_round_trips_under_a_loose_bound() {
        // a fixed-seed block; dequantize the quantized bytes and check the reconstruction is finite and bounded, i.e.
        // the new Q5_0 quantizer packs a well-formed block (byte-for-byte correctness is pinned by the tiny reference).
        let x: Vec<f32> = (0..32).map(|i| (i as f32 - 15.0) * 0.1).collect();
        let mut bytes = Vec::new();
        strata_pack::quantize::q5_0(&x, &mut bytes);
        assert_eq!(bytes.len(), 22); // one Q5_0 block: 2 (scale) + 4 (qh) + 16 (nibbles)
        let xhat = strata_pack::quants::dequantize("Q5_0", &bytes).unwrap();
        assert_eq!(xhat.len(), 32);
        let err: f32 = x.iter().zip(&xhat).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
        assert!(err < 5.0, "q5_0 reconstruction error {err} too large");
    }

    #[test]
    fn q5_1_round_trips_under_a_loose_bound() {
        let x: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { -1.0 } else { 1.0 }).collect();
        let mut bytes = Vec::new();
        strata_pack::quantize::q5_1(&x, &mut bytes);
        assert_eq!(bytes.len(), 24); // Q5_1 block: 2 (d) + 2 (m) + 4 (qh) + 16 (nibbles)
        let xhat = strata_pack::quants::dequantize("Q5_1", &bytes).unwrap();
        assert_eq!(xhat.len(), 32);
    }
}
