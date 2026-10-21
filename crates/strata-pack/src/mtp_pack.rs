//! The MTP draft block as a PolyStrata GGUF, from the BF16 tensors mtp_fetch wrote: the port of tools/mtp_pack.py.
//! Dense tensors stay BF16 (F32 for 1-D norms); the routed experts keep the checkpoint's fused layout, quantized along
//! the last axis to Q2_0 (a per-block MSE scale search over the grid {-1, 0, 1, 2}·d), Q4_0 or Q8_0 (ggml reference).
//! The GGUF is the one mtp_pack.py writes, byte for byte; the report's numbers too (its "seconds" is a timing).

use crate::gguf_write::{TensorSpec, Writer};
use crate::num::{f16_to_f32, f32_to_f16, jf, pairwise_f32, pairwise_f64, py_json, py_round, rint_f32};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

const EXPERT_TENSORS: [&str; 2] = ["mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj"];

pub struct Options {
    pub src: PathBuf,
    pub experts: String,
    pub out: PathBuf,
    pub check_experts: usize,
}

/// (ggml type, block elements, block bytes)
fn format_of(kind: &str) -> Option<(u32, usize, usize)> {
    match kind {
        "q2_0" => Some((42, 64, 18)),
        "q4_0" => Some((2, 32, 18)),
        "q8_0" => Some((8, 32, 34)),
        _ => None,
    }
}

/// Q2_0: fp16 d, 16 bytes of 2-bit codes (code = q + 1, four per byte); d chosen per block to minimize the squared error.
pub fn q2_0(x: &[f32], out: &mut Vec<u8>) {
    let mut err = [0f32; 64];
    for b in x.chunks_exact(64) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let (mut best_err, mut best_d) = (f32::INFINITY, 0f32);
        for k in 0..17 {
            let f = 0.5f32 + k as f32 / 32.0;   // np.linspace(0.5, 1.0, 17): exact multiples of 1/32
            let d = amax * f;
            let inv = if d > 0.0 { 1.0f32 / d } else { 0.0 };
            for (e, v) in err.iter_mut().zip(b) {
                let q = rint_f32(v * inv).clamp(-1.0, 2.0);
                let r = q * d - v;
                *e = r * r;
            }
            let s = pairwise_f32(&err);
            if s < best_err {
                best_err = s;
                best_d = d;
            }
        }
        let d16 = f32_to_f16(best_d);
        let d = f16_to_f32(d16);
        let inv = if d > 0.0 { 1.0f32 / d } else { 0.0 };
        out.extend(d16.to_le_bytes());
        for c in b.chunks_exact(4) {
            let code = |v: f32| (rint_f32(v * inv).clamp(-1.0, 2.0) + 1.0) as u8;
            out.push(code(c[0]) | code(c[1]) << 2 | code(c[2]) << 4 | code(c[3]) << 6);
        }
    }
}

/// ggml quantize_row_q4_0_ref: d = (signed max) / -8, code = min(15, trunc(x/d + 8.5)).
pub fn q4_0(x: &[f32], out: &mut Vec<u8>) {
    for b in x.chunks_exact(32) {
        let mut idx = 0;
        for (i, v) in b.iter().enumerate() {
            if v.abs() > b[idx].abs() {
                idx = i;
            }
        }
        let d = b[idx] / -8.0f32;
        let inv = if d != 0.0 { 1.0f32 / d } else { 0.0 };
        let q: Vec<u8> = b.iter().map(|v| (v * inv + 8.5f32).trunc().min(15.0) as u8).collect();
        out.extend(f32_to_f16(d).to_le_bytes());
        for j in 0..16 {
            out.push(q[j] | (q[16 + j] << 4));
        }
    }
}

/// ggml reference Q8_0 as mtp_pack.py writes it: d = amax / 127, q = rint(x / d).
pub fn q8_0(x: &[f32], out: &mut Vec<u8>) {
    for b in x.chunks_exact(32) {
        let d = b.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0f32;
        let inv = if d > 0.0 { 1.0f32 / d } else { 0.0 };
        out.extend(f32_to_f16(d).to_le_bytes());
        out.extend(b.iter().map(|v| (rint_f32(v * inv) as i8) as u8));
    }
}

fn dequant(kind: &str, blob: &[u8], n: usize) -> Vec<f32> {
    let mut y = Vec::with_capacity(n + 64);
    match kind {
        "q2_0" => {
            for b in blob.chunks_exact(18) {
                let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                for q in &b[2..] {
                    for s in [0, 2, 4, 6] {
                        y.push((((q >> s) & 3) as f32 - 1.0) * d);
                    }
                }
            }
        }
        "q4_0" => {
            for b in blob.chunks_exact(18) {
                let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                for q in &b[2..] {
                    y.push(((q & 15) as f32 - 8.0) * d);
                }
                for q in &b[2..] {
                    y.push(((q >> 4) as f32 - 8.0) * d);
                }
            }
        }
        _ => {
            for b in blob.chunks_exact(34) {
                let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                for q in &b[2..] {
                    y.push((*q as i8) as f32 * d);
                }
            }
        }
    }
    y.truncate(n);
    y
}

fn bf16_file(path: &Path, n: usize) -> Result<Vec<u16>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() != 2 * n {
        return Err(format!("{}: {} values, expected {n}", path.display(), b.len() / 2));
    }
    Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

/// float32 mean as numpy computes it: the pairwise sum divided by the count.
fn mean32(a: &[f32]) -> f32 {
    pairwise_f32(a) / a.len() as f32
}

pub fn run(o: &Options) -> Result<(), String> {
    let (qtype, block, block_bytes) = format_of(&o.experts).ok_or_else(|| format!("--experts must be q2_0, q4_0 or q8_0, not {}", o.experts))?;
    let manifest: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(o.src.join("mtp-manifest.json")).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let mut sums = Map::new();
    let mut by_name: Vec<&Value> = manifest.iter().collect();
    by_name.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    {
        let mut names: Vec<(&str, &str)> = manifest.iter().map(|t| (t["name"].as_str().unwrap(), t["sha256"].as_str().unwrap())).collect();
        names.sort();
        for (n, s) in names {
            sums.insert(n.into(), json!(s));
        }
    }
    let source_sha = crate::iq_pack::hex(&Sha256::digest(py_json(&Value::Object(sums), None).as_bytes()));
    let shape_of = |t: &Value| -> Vec<u64> { t["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect() };
    let mut specs = Vec::new();
    for t in &by_name {
        let name = t["name"].as_str().unwrap();
        let shape = shape_of(t);
        let n: u64 = shape.iter().product();
        let (ty, nbytes) = if EXPERT_TENSORS.contains(&name) {
            let inner = *shape.last().unwrap();
            if inner % block as u64 != 0 {
                return Err(format!("{name}: inner dim {inner} not a multiple of {block}"));
            }
            (qtype, n / block as u64 * block_bytes as u64)
        } else if shape.len() == 1 {
            (0, n * 4)
        } else {
            (30, n * 2)
        };
        specs.push(TensorSpec { name: name.into(), shape, ggml_type: ty, nbytes });
    }
    let kv = [("general.architecture", "qwen4exp-mtp".to_string()),
              ("strata.mtp.source", "Qwen/Qwen3.8-Flash-Next BF16 checkpoint, mtp.* tensors".to_string()),
              ("strata.mtp.source_sha256", source_sha),
              ("strata.mtp.expert_format", o.experts.clone()),
              ("strata.mtp.expert_quantizer", if o.experts == "q2_0" { "per-block MSE scale search" } else { "ggml reference" }.to_string())];
    let mut w = Writer::create(&o.out, &kv, specs).map_err(|e| format!("{}: {e}", o.out.display()))?;
    let mut report = Vec::new();
    for t in &by_name {
        let name = t["name"].as_str().unwrap();
        let shape = shape_of(t);
        let file = o.src.join(t["file"].as_str().unwrap());
        let t0 = Instant::now();
        let mut entry = Map::new();
        entry.insert("tensor".into(), json!(name));
        if EXPERT_TENSORS.contains(&name) {
            let per: usize = shape[1..].iter().product::<u64>() as usize;
            let f = File::open(&file).map_err(|e| format!("{}: {e}", file.display()))?;
            let n_exp = shape[0] as usize;
            let mut errs = Vec::new();
            let mut total = 0usize;
            // one expert per thread, a batch at a time (each is independent; the parts are written in order)
            let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 32);
            let one = |e: usize| -> Result<(Vec<u8>, Option<f64>), String> {
                let mut raw = vec![0u8; per * 2];
                f.read_exact_at(&mut raw, (e * per * 2) as u64).map_err(|err| format!("{}: {err}", file.display()))?;
                let x: Vec<f32> = raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
                let mut part = Vec::with_capacity(per / block * block_bytes);
                match o.experts.as_str() {
                    "q2_0" => q2_0(&x, &mut part),
                    "q4_0" => q4_0(&x, &mut part),
                    _ => q8_0(&x, &mut part),
                }
                let err = (e < o.check_experts).then(|| {
                    let got = dequant(&o.experts, &part, per);
                    let d2: Vec<f32> = got.iter().zip(&x).map(|(g, r)| (g - r) * (g - r)).collect();
                    let r2: Vec<f32> = x.iter().map(|r| r * r).collect();
                    let num = mean32(&d2).sqrt();
                    let den = mean32(&r2).sqrt() + 1e-12f64 as f32;
                    (num / den) as f64
                });
                Ok((part, err))
            };
            for start in (0..n_exp).step_by(threads) {
                let batch: Vec<Result<(Vec<u8>, Option<f64>), String>> = std::thread::scope(|s| {
                    let hs: Vec<_> = (start..(start + threads).min(n_exp)).map(|e| s.spawn(move || one(e))).collect();
                    hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("a quantizer thread panicked".into()))).collect()
                });
                for r in batch {
                    let (part, err) = r?;
                    errs.extend(err);
                    total += part.len();
                    w.tensor_data(&part).map_err(|e| e.to_string())?;
                }
            }
            entry.insert("format".into(), json!(o.experts));
            entry.insert("bytes".into(), json!(total));
            entry.insert("relative_rms_error_first_experts".into(), jf(py_round(pairwise_f64(&errs) / errs.len() as f64, 5)));
            entry.insert("seconds".into(), jf(py_round(t0.elapsed().as_secs_f64(), 1)));
        } else if shape.len() == 1 {
            let n = shape[0] as usize;
            let v = bf16_file(&file, n)?;
            let bytes: Vec<u8> = v.iter().flat_map(|h| f32::from_bits((*h as u32) << 16).to_le_bytes()).collect();
            w.tensor_data(&bytes).map_err(|e| e.to_string())?;
            entry.insert("format".into(), json!("F32"));
            entry.insert("bytes".into(), json!(n * 4));
        } else {
            let n: usize = shape.iter().product::<u64>() as usize;
            let v = bf16_file(&file, n)?;
            let bytes: Vec<u8> = v.iter().flat_map(|h| h.to_le_bytes()).collect();
            w.tensor_data(&bytes).map_err(|e| e.to_string())?;
            entry.insert("format".into(), json!("BF16"));
            entry.insert("bytes".into(), json!(n * 2));
        }
        w.end_tensor().map_err(|e| e.to_string())?;
        println!("{name}: {} {:.1} MB", entry["format"].as_str().unwrap(), entry["bytes"].as_u64().unwrap() as f64 / 1e6);
        report.push(Value::Object(entry));
    }
    w.finish().map_err(|e| e.to_string())?;
    let total: u64 = report.iter().map(|r| r["bytes"].as_u64().unwrap()).sum();
    let rep = json!({"total_bytes": total, "tensors": report});
    let rpath = PathBuf::from(format!("{}.report.json", o.out.display()));
    std::fs::write(&rpath, py_json(&rep, Some(1))).map_err(|e| e.to_string())?;
    println!("wrote {}: {:.3} GB of tensor data", o.out.display(), total as f64 / 1e9);
    Ok(())
}
