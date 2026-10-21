//! The MTP draft layer's runtime files, from the packed MTP GGUF: the port of tools/mtp_rt.py.
//!
//! - `experts.bin`: the 512 routed experts in the engine's blob layout: gate/up rows interleaved (2r = gate r,
//!   2r+1 = up r), then down rows; the Q2_0 codes in one plane and the fp16 scales in another (a lossless relayout).
//! - `dense.bin`: every other tensor: the large projections quantized to Q8_0 (ggml's reference rounding), the
//!   hyper-connection and router weights kept BF16, the RMSNorm weights as F32 with the Gemma "+1" applied.
//! - `dense.txt`: one line per tensor: name kind rows cols offset bytes.

use crate::num::f32_to_f16;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use strata_gguf::Gguf;

const H: usize = 2560;
const FF: usize = 640;
const NE: usize = 512;
const BLOB: usize = 3 * (H * FF * 18 / 64);
const Q8: [&str; 10] = ["fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.indexer.index_qk_proj.weight",
                        "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"];

pub struct Options {
    pub gguf: PathBuf,
    pub out: PathBuf,
}

/// ggml quantize_row_q8_0_ref: 32-value blocks, fp16 d = amax / 127, q = round half away from zero.
pub fn q8_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / 32 * 34);
    for b in x.chunks_exact(32) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0f32;
        let inv = if d > 0.0 { 1.0f32 / d } else { 0.0 };
        out.extend(f32_to_f16(d).to_le_bytes());
        for v in b {
            let v = v * inv;
            let q = v.signum() * (v.abs() + 0.5f32).floor();
            let q = if v == 0.0 { 0.0 } else { q };
            out.push((q as i8) as u8);
        }
    }
    out
}

/// gu: one expert's 1280 Q2_0 rows of 720 bytes (gate 0..639, up 640..1279); dn: 2560 rows of 180 bytes.
fn blob_of(gu: &[u8], dn: &[u8]) -> Vec<u8> {
    let (gr, dr) = (H / 64 * 18, FF / 64 * 18);
    let rows: Vec<&[u8]> = (0..FF).flat_map(|r| [&gu[r * gr..(r + 1) * gr], &gu[(FF + r) * gr..(FF + r + 1) * gr]]).collect();
    let mut out = Vec::with_capacity(BLOB);
    for r in &rows {
        for b in r.chunks_exact(18) {
            out.extend(&b[2..]);
        }
    }
    for r in dn.chunks_exact(dr) {
        for b in r.chunks_exact(18) {
            out.extend(&b[2..]);
        }
    }
    for r in &rows {
        for b in r.chunks_exact(18) {
            out.extend(&b[..2]);
        }
    }
    for r in dn.chunks_exact(dr) {
        for b in r.chunks_exact(18) {
            out.extend(&b[..2]);
        }
    }
    assert_eq!(out.len(), BLOB);
    out
}

pub fn run(o: &Options) -> Result<(), String> {
    std::fs::create_dir_all(&o.out).map_err(|e| e.to_string())?;
    let g = Gguf::open(&o.gguf).map_err(|e| e.to_string())?;
    let f = File::open(&o.gguf).map_err(|e| e.to_string())?;
    let read = |t: &strata_gguf::TensorInfo| -> Result<Vec<u8>, String> {
        let n = t.expected_bytes().ok_or_else(|| format!("{}: unsupported type", t.name))?;
        let mut b = vec![0u8; n as usize];
        f.read_exact_at(&mut b, g.data_start + t.offset).map_err(|e| format!("{}: {e}", t.name))?;
        Ok(b)
    };
    let find = |n: &str| g.tensors.iter().find(|t| t.name == n).ok_or_else(|| format!("{n} is not in {}", o.gguf.display()));
    let gu_t = find("mtp.layers.0.mlp.experts.gate_up_proj")?;
    let dn_t = find("mtp.layers.0.mlp.experts.down_proj")?;
    if gu_t.dims != [H as u64, 2 * FF as u64, NE as u64] || dn_t.dims != [FF as u64, H as u64, NE as u64] || gu_t.ggml_type != 42 || dn_t.ggml_type != 42 {
        return Err(format!("unexpected expert tensors: {:?} {:?} (types {} {})", gu_t.dims, dn_t.dims, gu_t.ggml_type, dn_t.ggml_type));
    }
    let (gu_e, dn_e) = (2 * FF * (H / 64 * 18), H * (FF / 64 * 18));
    let mut eo = BufWriter::with_capacity(32 << 20, File::create(o.out.join("experts.bin")).map_err(|e| e.to_string())?);
    let (mut gu, mut dn) = (vec![0u8; gu_e], vec![0u8; dn_e]);
    for e in 0..NE {
        f.read_exact_at(&mut gu, g.data_start + gu_t.offset + (e * gu_e) as u64).map_err(|e| e.to_string())?;
        f.read_exact_at(&mut dn, g.data_start + dn_t.offset + (e * dn_e) as u64).map_err(|e| e.to_string())?;
        eo.write_all(&blob_of(&gu, &dn)).map_err(|e| e.to_string())?;
    }
    eo.flush().map_err(|e| e.to_string())?;
    let mut lines = Vec::new();
    let mut off = 0usize;
    let mut fo = BufWriter::new(File::create(o.out.join("dense.bin")).map_err(|e| e.to_string())?);
    for t in &g.tensors {
        if t.name.contains("experts.") {
            continue;
        }
        let short = t.name.strip_prefix("mtp.").unwrap_or(&t.name);
        let short = short.strip_prefix("layers.0.").unwrap_or(short);
        let data = read(t)?;
        let (raw, kind, rows, cols) = if t.ggml_type == 0 {
            let v: Vec<u8> = data.chunks_exact(4).flat_map(|c| (f32::from_le_bytes(c.try_into().unwrap()) + 1.0f32).to_le_bytes()).collect();
            let n = v.len() / 4;
            (v, "f32", 1, n)
        } else {
            let rows = *t.dims.last().unwrap() as usize;
            let cols = t.elements() as usize / rows;
            if Q8.contains(&short) {
                let x: Vec<f32> = data.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
                (q8_0(&x), "q8_0", rows, cols)
            } else {
                (data, "bf16", rows, cols)
            }
        };
        let pad = (256 - off % 256) % 256;
        fo.write_all(&vec![0u8; pad]).map_err(|e| e.to_string())?;
        off += pad;
        fo.write_all(&raw).map_err(|e| e.to_string())?;
        lines.push(format!("{short} {kind} {rows} {cols} {off} {}", raw.len()));
        off += raw.len();
    }
    fo.flush().map_err(|e| e.to_string())?;
    std::fs::write(o.out.join("dense.txt"), lines.join("\n") + "\n").map_err(|e| e.to_string())?;
    println!("experts.bin {} B, dense.bin {off} B, {} tensors -> {}", NE * BLOB, lines.len(), o.out.display());
    for l in &lines {
        println!("  {l}");
    }
    Ok(())
}
