//! A native pack for the `strata` engine from any of the model files (Q2_0, IQ2_XS, IQ3_XXS, ...): the port of
//! PolyStrata's tools/iq_pack.py. The pack keeps every quantized tensor in its GGUF form:
//!
//! - `index.txt`: the table the engine loads; quantized dense tensors, token_embd and output are served natively
//!   from the GGUF (`--native`): their rows carry the shape only.
//! - `dense.bin`: the BF16/F16/F32 tensors in the form the engine reads them, converted where the GGUF stores them
//!   otherwise (FORM), each conversion recorded in `conversions.json`.
//! - `native_experts.txt`: one line per layer (types, blob offset and size, the GGUF offsets of gate/up/down, the
//!   shard); written last: a pack without it is unfinished.
//! - `experts.bin` (optional, `--experts-bin`): the experts cut out of the GGUF, with `experts.bin.src.json`.
//! - `tokenizer/`: exported from the GGUF.
//!
//! Every file is the one iq_pack.py writes, byte for byte (tests/iq_pack_golden.rs).

use crate::num::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, jf, py_json};
use crate::quants;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use strata_gguf::{Gguf, TensorInfo};

const FLOAT: [&str; 3] = ["BF16", "F32", "F16"];
const NOT_IN_PACK: &str = "per_layer_token_embd.weight";
const NATIVE_PLE_KEY: [&str; 4] = ["Q2_0", "Q8_0", "IQ3_XXS", "IQ4_XS"];
const ROLES: [&str; 3] = ["gate", "up", "down"];
const ALIGN: u64 = 64;
const NVFP4_TAIL: u64 = 16;

/// The float form the engine reads each pack tensor in (name without `blk.N.`; iq_pack.py FORM).
fn form_of(name: &str) -> Option<&'static str> {
    let short = match name.strip_prefix("blk.") {
        Some(rest) => match rest.split_once('.') {
            Some((n, tail)) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => tail,
            _ => name,
        },
        None => name,
    };
    Some(match short {
        "ffn_gate_inp.weight" | "ffn_gate_inp_shexp.weight" | "hc_attn_down.weight" | "hc_attn_up.weight"
        | "hc_attn_inject.weight" | "hc_ffn_down.weight" | "hc_ffn_up.weight" | "hc_ffn_inject.weight"
        | "output_hc_down.weight" | "output_hc_up.weight" | "indexer.k_proj.weight" | "indexer.q_proj.weight"
        | "ssm_alpha.weight" | "ssm_beta.weight" | "ple_key.weight" | "ple_value.weight" => "BF16",
        "ple_conv1d.weight" => "F16",
        "attn_q_norm.weight" | "attn_k_norm.weight" | "hc_attn_norm.weight" | "hc_ffn_norm.weight" | "output_hc_norm.weight"
        | "indexer.q_norm.weight" | "indexer.k_norm.weight" | "ple_norm_conv.weight" | "ple_norm_key.weight"
        | "ple_norm_query.weight" | "ssm_a" | "ssm_conv1d.weight" | "ssm_dt.bias" | "ssm_norm.weight" => "F32",
        _ => return None,
    })
}

fn kind_of(t: &str) -> &'static str {
    match t {
        "BF16" => "4",
        "F16" => "5",
        _ => "2",
    }
}

fn is_float(t: &str) -> bool {
    FLOAT.contains(&t)
}

fn is_expert(name: &str) -> bool {
    name.starts_with("blk.") && name.ends_with("_exps.weight")
}

pub struct Options {
    pub gguf: PathBuf,
    pub out: PathBuf,
    pub base: Option<PathBuf>,
    pub skip_experts: bool,
    pub compat_bf16: bool,
    pub experts_bin: bool,
}

struct Shard {
    path: PathBuf,
    file: File,
    g: Gguf,
    size: u64,
}

/// All shards of one model: name -> (shard index, tensor), in file order.
struct Model {
    shards: Vec<Shard>,
    order: Vec<String>,
    where_: HashMap<String, (usize, TensorInfo)>,
}

impl Model {
    fn open(first: &Path) -> Result<Model, String> {
        let name = first.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let mut paths = vec![first.to_path_buf()];
        if let Some((base, total)) = split_name(name) {
            paths = (1..=total).map(|i| first.with_file_name(format!("{base}-{i:05}-of-{total:05}.gguf"))).collect();
        }
        let missing: Vec<String> = paths.iter().filter(|p| !p.is_file()).map(|p| p.display().to_string()).collect();
        if !missing.is_empty() {
            return Err(format!("missing model shards (wait for the download): {}", missing.join(", ")));
        }
        let mut shards = Vec::new();
        for p in &paths {
            let g = Gguf::open(p).map_err(|e| e.to_string())?;
            if g.version != 3 {
                return Err(format!("{}: GGUF v{}, this reader handles v3", fname(p), g.version));
            }
            let file = File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
            let size = file.metadata().map_err(|e| e.to_string())?.len();
            shards.push(Shard { path: p.clone(), file, g, size });
        }
        check_split(&shards)?;
        let mut where_ = HashMap::new();
        let mut order = Vec::new();
        for (i, s) in shards.iter().enumerate() {
            for t in &s.g.tensors {
                let size = t.expected_bytes();
                if size.is_none() || s.g.data_start + t.offset + size.unwrap() > s.size {
                    return Err(format!("{}: unsupported or truncated tensor {}", fname(&s.path), t.name));
                }
                if where_.contains_key(&t.name) {
                    return Err(format!("{}: duplicate tensor {}", fname(&s.path), t.name));
                }
                order.push(t.name.clone());
                where_.insert(t.name.clone(), (i, t.clone()));
            }
        }
        Ok(Model { shards, order, where_ })
    }

    fn tensor(&self, name: &str) -> &TensorInfo {
        &self.where_[name].1
    }

    fn abs_offset(&self, name: &str) -> u64 {
        let (i, t) = &self.where_[name];
        self.shards[*i].g.data_start + t.offset
    }

    fn read_at(&self, name: &str, rel: u64, n: u64) -> Result<Vec<u8>, String> {
        let (i, _) = &self.where_[name];
        let mut buf = vec![0u8; n as usize];
        self.shards[*i].file.read_exact_at(&mut buf, self.abs_offset(name) + rel).map_err(|e| format!("{name}: {e}"))?;
        Ok(buf)
    }

    fn bytes(&self, name: &str) -> Result<Vec<u8>, String> {
        let n = self.tensor(name).expected_bytes().unwrap();
        self.read_at(name, 0, n)
    }
}

fn fname(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// `<base>-00001-of-00003.gguf` -> (base, 3): exactly five digits each, as iq_pack.py's pattern.
fn split_name(name: &str) -> Option<(String, usize)> {
    let stem = name.strip_suffix(".gguf")?;
    if stem.len() < 15 {
        return None;
    }
    let (head, tail) = stem.split_at(stem.len() - 15);
    let b = tail.as_bytes();
    let digits = |s: &[u8]| s.iter().all(u8::is_ascii_digit);
    if b[0] == b'-' && digits(&b[1..6]) && &tail[6..10] == "-of-" && digits(&b[10..15]) {
        return Some((head.to_string(), tail[10..15].parse().ok()?));
    }
    None
}

fn check_split(shards: &[Shard]) -> Result<(), String> {
    let n = shards.len() as u64;
    let meta0 = &shards[0].g.metadata;
    let num = |m: &Map<String, Value>, k: &str| m.get(k).and_then(Value::as_u64);
    if n == 1 {
        if let Some(c) = num(meta0, "split.count").filter(|c| *c > 1) {
            return Err(format!("{} is shard 1 of {c}, but its name has no -00001-of-0000N.gguf to find the others by",
                               fname(&shards[0].path)));
        }
        return Ok(());
    }
    if !meta0.contains_key("general.architecture") {
        return Err(format!("{} has no general.architecture; the first shard of a split model carries the metadata",
                           fname(&shards[0].path)));
    }
    let total = num(meta0, "split.tensors.count");
    for (i, s) in shards.iter().enumerate() {
        let md = &s.g.metadata;
        if num(md, "split.count") != Some(n) || num(md, "split.no") != Some(i as u64)
            || (total.is_some() && num(md, "split.tensors.count") != total) {
            return Err(format!("{} does not declare itself shard {} of {n} of this model \
                                (split.count / split.no / split.tensors.count)", fname(&s.path), i + 1));
        }
    }
    let sum: usize = shards.iter().map(|s| s.g.tensors.len()).sum();
    if let Some(t) = total {
        if sum as u64 != t {
            return Err(format!("the {n} shards hold {sum} tensors, but split.tensors.count is {t}"));
        }
    }
    Ok(())
}

/// A tensor's values as f32: float types exactly, quantized ones through the dequantizer.
fn values_of(raw: &[u8], t: &str) -> Result<Vec<f32>, String> {
    Ok(match t {
        "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        "F16" => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        "BF16" => raw.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        _ => quants::dequantize(t, raw)?,
    })
}

fn max_abs_err(got: impl Iterator<Item = f32>, want: &[f32]) -> f64 {
    let mut m = 0.0f64;
    let mut any = false;
    for (g, w) in got.zip(want) {
        let d = (g as f64 - *w as f64).abs();
        if !any || d > m {
            m = d;
            any = true;
        }
    }
    m
}

struct Converted {
    kind: &'static str,
    data: Vec<u8>,
    dst_bytes: u64,
    rec: Option<Map<String, Value>>,
}

/// The pack form of one tensor the engine reads from dense.bin, or the reason it is refused (iq_pack.py convert).
fn convert(name: &str, t: &str, raw: &[u8], compat: bool) -> Result<Converted, String> {
    let form = form_of(name);
    if form.is_none() || form == Some(t) {
        return Ok(Converted { kind: kind_of(t), data: raw.to_vec(), dst_bytes: raw.len() as u64, rec: None });
    }
    let form = form.unwrap();
    let values = values_of(raw, t)?;
    let mut rec = Map::new();
    rec.insert("src_type".into(), json!(t));
    rec.insert("dst_type".into(), json!(form));
    if form == "BF16" {
        if t == "F32" && values.iter().all(|v| v.to_bits() & 0xffff == 0) {
            if !values.iter().all(|v| v.is_finite()) {
                return Err(format!("{name} is F32 with non-finite values; refused rather than converted"));
            }
            let data: Vec<u8> = values.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
            rec.insert("method".into(), json!("f32->bf16: the values are BF16's (low 16 bits zero)"));
            rec.insert("exact".into(), json!(true));
            rec.insert("max_abs_err".into(), jf(0.0));
            let n = data.len() as u64;
            return Ok(Converted { kind: "4", data, dst_bytes: n, rec: Some(rec) });
        }
        if !compat {
            let what = if t == "F32" { "F32 with values that are not BF16".to_string() } else { t.to_string() };
            return Err(format!("{name} is {what}, but the engine requires BF16; use --compat-bf16 (rounds to nearest-even)"));
        }
        if !values.iter().all(|v| v.is_finite()) {
            return Err("cannot convert non-finite weights to BF16".into());
        }
        let bf: Vec<u16> = values.iter().map(|v| f32_to_bf16(*v)).collect();
        let data: Vec<u8> = bf.iter().flat_map(|b| b.to_le_bytes()).collect();
        let method = if !is_float(t) {
            "dequantize->bf16 round-to-nearest-even (--compat-bf16)".to_string()
        } else {
            format!("{}->bf16 round-to-nearest-even (--compat-bf16)", t.to_lowercase())
        };
        rec.insert("method".into(), json!(method));
        let exact = bf.iter().zip(&values).all(|(b, v)| bf16_to_f32(*b) == *v);
        rec.insert("exact".into(), json!(exact));
        rec.insert("max_abs_err".into(), jf(max_abs_err(bf.iter().map(|b| bf16_to_f32(*b)), &values)));
        let n = data.len() as u64;
        return Ok(Converted { kind: "4", data, dst_bytes: n, rec: Some(rec) });
    }
    if form == "F16" {
        if t != "F32" && !compat {
            return Err(format!("{name} is {t}, but the engine reads F16; use --compat-bf16 (rounds to nearest-even)"));
        }
        if !values.iter().all(|v| v.is_finite()) {
            return Err(format!("{name}: cannot convert non-finite weights to F16"));
        }
        let hv: Vec<u16> = values.iter().map(|v| f32_to_f16(*v)).collect();
        if hv.iter().any(|x| x & 0x7c00 == 0x7c00) {
            return Err(format!("{name}: a value is beyond F16's range, which the engine reads it as"));
        }
        rec.insert("exact".into(), json!(hv.iter().zip(&values).all(|(x, v)| f16_to_f32(*x) == *v)));
        rec.insert("max_abs_err".into(), jf(max_abs_err(hv.iter().map(|x| f16_to_f32(*x)), &values)));
        if t == "F32" {
            rec.insert("method".into(), json!("f32->f16 by the loader (index kind 3, round-to-nearest-even)"));
            return Ok(Converted { kind: "3", data: raw.to_vec(), dst_bytes: raw.len() as u64 / 2, rec: Some(rec) });
        }
        rec.insert("method".into(), json!(format!("{}->f16 round-to-nearest-even (--compat-bf16)", t.to_lowercase())));
        let data: Vec<u8> = hv.iter().flat_map(|x| x.to_le_bytes()).collect();
        let n = data.len() as u64;
        return Ok(Converted { kind: "5", data, dst_bytes: n, rec: Some(rec) });
    }
    // form == "F32"
    if t != "F16" && t != "BF16" && !compat {
        return Err(format!("{name} is {t}, but the engine reads F32; use --compat-bf16"));
    }
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    rec.insert("method".into(), json!(format!("{}->f32 widening", t.to_lowercase())));
    rec.insert("exact".into(), json!(true));
    rec.insert("max_abs_err".into(), jf(0.0));
    let n = data.len() as u64;
    Ok(Converted { kind: "2", data, dst_bytes: n, rec: Some(rec) })
}

fn row(fields: &[String]) -> Vec<String> {
    fields.to_vec()
}

fn shape01(t: &TensorInfo) -> (String, String) {
    let rows: u64 = t.dims[1..].iter().product();
    (t.dims[0].to_string(), if t.dims.len() > 1 { rows.to_string() } else { "0".into() })
}

fn rename(a: &Path, b: &Path) -> Result<(), String> {
    std::fs::rename(a, b).map_err(|e| format!("{} -> {}: {e}", a.display(), b.display()))
}

fn write_index(out: &Path, rows: &mut [Vec<String>], src: &Path, served: usize, n_extra: usize, publish: bool) -> Result<(), String> {
    let mut at: u64 = 0;
    for r in rows.iter_mut() {
        r[5] = at.to_string();
        at += r[6].parse::<u64>().unwrap().div_ceil(ALIGN) * ALIGN;
    }
    let mut text = format!("# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from {}\n", fname(src));
    text += &format!("# align {ALIGN} pool {at} tensors {}\n", rows.len());
    for r in rows.iter() {
        text += &r.join(" ");
        text.push('\n');
    }
    std::fs::write(out.join("index.txt.tmp"), text).map_err(|e| e.to_string())?;
    if publish {
        rename(&out.join("index.txt.tmp"), &out.join("index.txt"))?;
    }
    println!("index.txt: {} tensors, {served} served natively, {n_extra} in extra.bin, arena {:.2} GiB",
             rows.len(), at as f64 / (1u64 << 30) as f64);
    Ok(())
}

fn index_standalone(src: &Path, out: &Path, model: &Model, compat: bool) -> Result<(), String> {
    let mut todo = Vec::new();
    let mut problems = Vec::new();
    for name in &model.order {
        let t = model.tensor(name);
        if is_expert(&t.name) || t.name == NOT_IN_PACK {
            continue;
        }
        // a 3-D tensor outside the experts (MLA's per-head attn_k_b / attn_v_b) is a stack of matrices: the index holds
        // it as one matrix of all their rows (shape01); iq_pack.py refuses these models
        if t.dims.len() > 3 {
            return Err(format!("tensor {} has {} dimensions; the index holds two", t.name, t.dims.len()));
        }
        let tn = t.type_name();
        let form = form_of(name);
        let native = !is_float(&tn) && (form.is_none() || (name == "blk.1.ple_key.weight" && NATIVE_PLE_KEY.contains(&tn.as_str())));
        if !is_float(&tn) && !native && !compat {
            problems.push(format!("{name} is {tn}, but the engine requires {}; use --compat-bf16", form.unwrap_or("None")));
        }
        todo.push((name.clone(), native));
    }
    if !problems.is_empty() {
        let mut msg: Vec<String> = problems.iter().take(8).cloned().collect();
        if problems.len() > 8 {
            msg.push(format!("... and {} more", problems.len() - 8));
        }
        return Err(msg.join("\n"));
    }
    let mut rows: Vec<Vec<String>> = Vec::new();
    let (mut at, mut served) = (0u64, 0usize);
    let (mut converted, mut records): (Vec<Value>, Vec<Value>) = (Vec::new(), Vec::new());
    let tmp = out.join("dense.bin.tmp");
    let mut fo = BufWriter::with_capacity(16 << 20, File::create(&tmp).map_err(|e| e.to_string())?);
    for (name, native) in &todo {
        let (si, t) = &model.where_[name];
        let (ne0, ne1) = shape01(t);
        if *native {
            served += 1;
            let mut r: Vec<String> = [t.name.as_str(), "0", "0", "0", "0", "0", "0", &ne0, &ne1, "8", "0", "32"].iter().map(|s| s.to_string()).collect();
            r.extend(std::iter::repeat_n("0".to_string(), 7));
            rows.push(row(&r));
            continue;
        }
        let raw = model.bytes(name)?;
        let got = match convert(name, &t.type_name(), &raw, compat) {
            Ok(g) => g,
            Err(e) => {
                drop(fo);
                let _ = std::fs::remove_file(&tmp);
                return Err(e);
            }
        };
        if let Some(mut rec) = got.rec {
            let mut full = Map::new();
            full.insert("name".into(), json!(name));
            full.insert("shard".into(), json!(fname(&model.shards[*si].path)));
            full.append(&mut rec);
            full.insert("source_sha256".into(), json!(hex(&Sha256::digest(&raw))));
            if full["method"].as_str().unwrap().contains("--compat-bf16") {
                converted.push(json!({"name": name, "source_type": t.type_name(), "bytes": got.data.len()}));
            }
            records.push(Value::Object(full));
        }
        let mut r: Vec<String> = vec![t.name.clone(), "0".into(), got.kind.into(), at.to_string(), got.data.len().to_string(), "0".into(),
                                      got.dst_bytes.to_string(), ne0, ne1, "0".into(), "0".into(), "1".into()];
        r.extend(std::iter::repeat_n("0".to_string(), 7));
        rows.push(r);
        fo.write_all(&got.data).map_err(|e| e.to_string())?;
        let pad = (ALIGN - got.data.len() as u64 % ALIGN) % ALIGN;
        fo.write_all(&vec![0u8; pad as usize]).map_err(|e| e.to_string())?;
        at += got.data.len() as u64 + pad;
    }
    fo.flush().map_err(|e| e.to_string())?;
    drop(fo);
    write_index(out, &mut rows, src, served, 0, false)?;
    let conv = json!({
        "schema": 1, "tool": "tools/iq_pack.py", "compat_bf16": compat,
        "source_shards": model.shards.iter().map(|s| json!({"name": fname(&s.path), "size": s.size})).collect::<Vec<_>>(),
        "tensors": records,
    });
    std::fs::write(out.join("conversions.json.tmp"), py_json(&conv, Some(1)) + "\n").map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(out.join("native_experts.txt"));
    rename(&tmp, &out.join("dense.bin"))?;
    rename(&out.join("index.txt.tmp"), &out.join("index.txt"))?;
    rename(&out.join("conversions.json.tmp"), &out.join("conversions.json"))?;
    if !records.is_empty() {
        let exact = records.iter().filter(|r| r["exact"] == true).count();
        let maxe = records.iter().map(|r| r["max_abs_err"].as_f64().unwrap()).fold(f64::MIN, f64::max);
        println!("conversions.json: {} tensors converted to the engine's form ({exact} exact, {} rounded; max |err| {})",
                 records.len(), records.len() - exact, g3(maxe));
        for r in &records {
            if r["dst_type"] == "F16" {
                println!("narrowed to F16 by the loader: {} ({}, index kind 3)", r["name"].as_str().unwrap(), r["method"].as_str().unwrap());
            }
        }
    }
    if compat {
        let gib: f64 = converted.iter().map(|t| t["bytes"].as_u64().unwrap() as f64).sum::<f64>() / (1u64 << 30) as f64;
        let v = json!({"source": src.display().to_string(), "rounding": "nearest-even", "tensors": converted});
        std::fs::write(out.join("compat-bf16.json"), py_json(&v, Some(2)) + "\n").map_err(|e| e.to_string())?;
        println!("compat-bf16: {} tensors, {gib:.2} GiB; expert and PLE table bytes unchanged", v["tensors"].as_array().unwrap().len());
    }
    Ok(())
}

/// %.3g
fn g3(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let e = x.abs().log10().floor() as i32;
    if !(-5..3).contains(&e) {
        let s = format!("{:.2e}", x);
        let (m, ex) = s.split_once('e').unwrap();
        let m = m.trim_end_matches('0').trim_end_matches('.');
        let ex: i32 = ex.parse().unwrap();
        return format!("{m}e{}{:02}", if ex < 0 { '-' } else { '+' }, ex.abs());
    }
    let s = format!("{:.*}", (2 - e).max(0) as usize, x);
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn read_index(path: &Path) -> Result<(Vec<String>, Vec<(String, Vec<String>)>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (mut header, mut rows) = (Vec::new(), Vec::new());
    for line in text.lines() {
        if line.starts_with('#') {
            header.push(line.to_string());
            continue;
        }
        let f: Vec<String> = line.split_whitespace().map(String::from).collect();
        if let Some(n) = f.first() {
            let n = n.clone();
            // a dict: a repeated name keeps its first position, with the last row
            match rows.iter_mut().find(|(k, _): &&mut (String, Vec<String>)| *k == n) {
                Some(r) => r.1 = f,
                None => rows.push((n, f)),
            }
        }
    }
    Ok((header, rows))
}

fn index_from_base(src: &Path, base: &Path, out: &Path, model: &Model) -> Result<(), String> {
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(base.join("manifest.json")).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let base_src = PathBuf::from(manifest["source"]["shard1"].as_str().unwrap_or(""));
    if !base_src.exists() {
        return Err("cannot find the base pack's shard 1 from its manifest.json".into());
    }
    let bg = Gguf::open(&base_src).map_err(|e| e.to_string())?;
    let bfile = File::open(&base_src).map_err(|e| e.to_string())?;
    let bt: HashMap<&str, &TensorInfo> = bg.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let (_, rows) = read_index(&base.join("index.txt"))?;
    let (mut new_rows, mut extra): (Vec<Vec<String>>, Vec<Vec<u8>>) = (Vec::new(), Vec::new());
    let mut served = 0;
    let shard0: HashMap<&str, &TensorInfo> = model.shards[0].g.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    for (name, f) in rows {
        let (t, b) = match (shard0.get(name.as_str()), bt.get(name.as_str())) {
            (Some(t), Some(b)) => (*t, *b),
            _ => return Err(format!("tensor {name} missing from one of the models")),
        };
        let (tn, btn) = (t.type_name(), b.type_name());
        if is_float(&tn) && is_float(&btn) {
            let same = tn == btn && t.dims == b.dims && {
                let n = t.expected_bytes().unwrap();
                let mut x = vec![0u8; n as usize];
                let mut y = vec![0u8; n as usize];
                model.shards[0].file.read_exact_at(&mut x, model.shards[0].g.data_start + t.offset).is_ok()
                    && bfile.read_exact_at(&mut y, bg.data_start + b.offset).is_ok() && x == y
            };
            if !same {
                return Err(format!("float tensor {name} differs from the base model; this pack cannot reuse its dense.bin"));
            }
            new_rows.push(f.clone());
        } else if is_float(&tn) {
            if tn != "BF16" {
                return Err(format!("unexpected float type {tn} for {name}"));
            }
            let nbytes = t.expected_bytes().unwrap();
            let off: u64 = extra.iter().map(|b| b.len() as u64 + (ALIGN - b.len() as u64 % ALIGN) % ALIGN).sum();
            let mut x = vec![0u8; nbytes as usize];
            model.shards[0].file.read_exact_at(&mut x, model.shards[0].g.data_start + t.offset).map_err(|e| e.to_string())?;
            extra.push(x);
            let mut r = vec![name.clone(), "3".into(), "4".into(), off.to_string(), nbytes.to_string(), "0".into(), nbytes.to_string(),
                             f[7].clone(), f[8].clone()];
            r.extend(std::iter::repeat_n("0".to_string(), 10));
            new_rows.push(r);
        } else {
            served += 1;
            let mut r = vec![name.clone(), f[1].clone(), "0".into(), "0".into(), "0".into(), "0".into(), "0".into(), f[7].clone(),
                             f[8].clone(), "8".into(), "0".into(), "32".into()];
            r.extend(std::iter::repeat_n("0".to_string(), 7));
            new_rows.push(r);
        }
    }
    let n_extra = extra.len();
    write_index(out, &mut new_rows, src, served, n_extra, true)?;
    let mut fo = BufWriter::new(File::create(out.join("extra.bin")).map_err(|e| e.to_string())?);
    for b in &extra {
        fo.write_all(b).map_err(|e| e.to_string())?;
        fo.write_all(&vec![0u8; ((ALIGN - b.len() as u64 % ALIGN) % ALIGN) as usize]).map_err(|e| e.to_string())?;
    }
    fo.flush().map_err(|e| e.to_string())?;
    let dense = out.join("dense.bin");
    if !dense.exists() && std::fs::hard_link(base.join("dense.bin"), &dense).is_err() {
        std::fs::copy(base.join("dense.bin"), &dense).map_err(|e| e.to_string())?;
    }
    if base.join("tokenizer").exists() && !out.join("tokenizer").exists() {
        copy_dir(&base.join("tokenizer"), &out.join("tokenizer"))?;
    }
    Ok(())
}

fn copy_dir(a: &Path, b: &Path) -> Result<(), String> {
    std::fs::create_dir_all(b).map_err(|e| e.to_string())?;
    for e in std::fs::read_dir(a).map_err(|e| e.to_string())?.flatten() {
        let p = e.path();
        if p.is_dir() {
            copy_dir(&p, &b.join(e.file_name()))?;
        } else {
            std::fs::copy(&p, b.join(e.file_name())).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

struct Layer {
    l: usize,
    gu_type: u32,
    d_type: u32,
    offset: u64,
    blob: u64,
    names: [String; 3],
    per: [u64; 3],
    nvfp4: bool,
    /// this layer's experts (the same in every layer, except in a per-layer pruned model)
    n: u64,
}

/// The pack's expert table and native_experts.txt (iq_pack.py expert_layout).
fn expert_layout(model: &Model, src: &Path) -> Result<(Vec<Layer>, String, u64, u64), String> {
    let exps: Vec<&String> = model.order.iter().filter(|n| n.starts_with("blk.") && n.ends_with("_exps.weight")).collect();
    if exps.is_empty() {
        return Err("the model has no expert tensors (blk.N.ffn_{gate,up,down}_exps.weight)".into());
    }
    let n_layers = 1 + exps.iter().filter_map(|n| n.split('.').nth(1)?.parse::<usize>().ok()).max().unwrap_or(0);
    let router = |l: usize| model.where_.get(&format!("blk.{l}.ffn_gate_inp.weight")).map(|(_, t)| t.dims.get(1).copied().unwrap_or(0));
    // layers without a router hold no experts (a leading dense layer, Mamba or attention layers of a hybrid model)
    let moe_layers: Vec<usize> = (0..n_layers).filter(|&l| router(l).is_some()).collect();
    let n_expert = moe_layers.iter().filter_map(|&l| router(l)).max().ok_or("no blk.N.ffn_gate_inp.weight")?;
    // every layer the same expert count: v3 / v4 as iq_pack.py writes them; a per-layer pruned model (e.g. NVIDIA's
    // Nemotron 3.5 Bolt) gets am2strata's own "p1", which adds the layer's count - read by am2strata-hybrid, not by
    // strata's C++ engine. (Not "v5": Strata 0.2's v5 is its Hadamard-folded experts.)
    let pruned = moe_layers.iter().any(|&l| router(l) != Some(n_expert));
    let (mut layout, mut lines, mut offset, mut n_split) = (Vec::new(), Vec::<String>::new(), 0u64, 0usize);
    for &l in &moe_layers {
        let n_expert = router(l).unwrap();
        let names: [String; 3] = ROLES.map(|r| format!("blk.{l}.ffn_{r}_exps.weight"));
        // experts without a gate (relu^2 up/down, as Nemotron-H's): the gate's slot stays empty
        let gateless = !model.where_.contains_key(&names[0]) && model.where_.contains_key(&names[1]);
        let missing: Vec<&String> = names.iter().enumerate().filter(|(i, n)| !(gateless && *i == 0) && !model.where_.contains_key(*n)).map(|(_, n)| n).collect();
        if !missing.is_empty() {
            return Err(format!("layer {l}: missing {}", missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
        }
        let ts: Vec<&TensorInfo> = names.iter().enumerate().map(|(i, n)| model.tensor(if gateless && i == 0 { &names[1] } else { n })).collect();
        if ts.iter().any(|t| t.expected_bytes().is_none() || t.dims.len() != 3 || t.dims[2] != n_expert) {
            return Err(format!("layer {l}: an expert tensor is not [*, *, {n_expert}] of whole blocks"));
        }
        let mut per = [0, 1, 2].map(|i| ts[i].expected_bytes().unwrap() / n_expert);
        if gateless {
            per[0] = 0;
        }
        if !gateless && (per[0] != per[1] || ts[0].ggml_type != ts[1].ggml_type) {
            return Err(format!("layer {l}: gate and up differ in type"));
        }
        let mut blob = per[0] + per[1] + per[2];
        let nvfp4 = ts[0].type_name() == "NVFP4" || ts[2].type_name() == "NVFP4";
        if nvfp4 {
            if ts.iter().zip(ROLES).any(|(t, r)| t.type_name() == "NVFP4" && !model.where_.contains_key(&format!("blk.{l}.ffn_{r}_exps.scale"))) {
                return Err(format!("layer {l}: NVFP4 experts without their .scale tensors"));
            }
            blob += NVFP4_TAIL;
        }
        if gateless && nvfp4 {
            return Err(format!("layer {l}: NVFP4 experts without a gate are not supported"));
        }
        let files: Vec<String> = names.iter().enumerate().filter(|(i, _)| !(gateless && *i == 0)).map(|(_, n)| {
            let p = &model.shards[model.where_[n].0].path;
            if p == src { String::new() } else { fname(p) }
        }).collect();
        let uniq: BTreeSet<&String> = files.iter().collect();
        let column = if uniq.len() == 1 { files[0].clone() } else { files.join(",") };
        n_split += (uniq.len() != 1) as usize;
        layout.push(Layer { l, gu_type: ts[0].ggml_type, d_type: ts[2].ggml_type, offset, blob, names: names.clone(), per, nvfp4, n: n_expert });
        if nvfp4 {
            lines.push(format!("{l} {} {} {offset} {blob}", ts[0].ggml_type, ts[2].ggml_type));
            offset += blob * n_expert;
            continue;
        }
        let gate_off = if gateless { 0 } else { model.abs_offset(&names[0]) };
        let mut line = format!("{l} {} {} {offset} {blob} {gate_off} {} {}", ts[0].ggml_type, ts[2].ggml_type,
                               model.abs_offset(&names[1]), model.abs_offset(&names[2]));
        if pruned {
            line += &format!(" {n_expert}");
        }
        if !column.is_empty() {
            line += &format!(" {column}");
        }
        lines.push(line);
        offset += blob * n_expert;
    }
    let head = if pruned {
        format!("# am2strata native experts p1: layer gu_type d_type offset blob_bytes gate_off up_off down_off n_expert [shard | gate,up,down] \
                 (n_expert up to {n_expert}, per layer; gate_off 0 for experts without a gate; total {offset}; absolute offsets in {}, \
                 or in the named shard beside it)\n", fname(src))
    } else if n_split > 0 {
        let which: Vec<String> = layout.iter().zip(&lines).filter(|(_, line)| line.contains(',')).map(|(x, _)| x.l.to_string()).collect();
        println!("{n_split} layer(s) have their gate/up/down in different shards: native_experts.txt v4, per-role shard column for {}",
                 which.join(", "));
        format!("# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard | gate,up,down] \
                 (n_expert {n_expert}, total {offset}; absolute offsets in {}, or in the named shard beside it - per role where the \
                 column is gate,up,down)\n", fname(src))
    } else {
        format!("# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] (n_expert {n_expert}, \
                 total {offset}; absolute offsets in {}, or in the named shard beside it)\n", fname(src))
    };
    let text = head + &lines.iter().map(|l| format!("{l}\n")).collect::<String>();
    Ok((layout, text, n_expert, offset))
}

fn experts_source(model: &Model, text: &str, total: u64) -> Value {
    json!({"schema": 1,
           "shards": model.shards.iter().map(|s| json!({"name": fname(&s.path), "size": s.size})).collect::<Vec<_>>(),
           "native_experts_sha256": hex(&Sha256::digest(text.as_bytes())),
           "bytes": total})
}

/// The NVFP4 scale tail of one expert: {s_gate, s_up, s_down, 0} (1.0 for a projection in another format).
fn nvfp4_tail(model: &Model, x: &Layer, e: u64) -> Result<[u8; 16], String> {
    let mut s = [1.0f32; 3];
    for (i, r) in ROLES.iter().enumerate() {
        if model.tensor(&x.names[i]).type_name() == "NVFP4" {
            let b = model.read_at(&format!("blk.{}.ffn_{r}_exps.scale", x.l), e * 4, 4)?;
            s[i] = f32::from_le_bytes(b.try_into().unwrap());
        }
    }
    if s.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(format!("layer {}: a non-finite or non-positive NVFP4 scale", x.l));
    }
    let mut out = [0u8; 16];
    for i in 0..3 {
        out[i * 4..i * 4 + 4].copy_from_slice(&s[i].to_le_bytes());
    }
    Ok(out)
}

fn expert_blob(model: &Model, x: &Layer, e: u64) -> Result<Vec<u8>, String> {
    let mut b = Vec::with_capacity(x.blob as usize);
    for i in 0..3 {
        if x.per[i] == 0 {
            continue; // no gate
        }
        b.extend(model.read_at(&x.names[i], e * x.per[i], x.per[i])?);
    }
    if x.nvfp4 {
        b.extend(nvfp4_tail(model, x, e)?);
    }
    Ok(b)
}

pub fn run(o: &Options) -> Result<(), String> {
    if o.compat_bf16 && o.base.is_some() {
        return Err("--compat-bf16 cannot reuse --base dense weights".into());
    }
    let src = std::path::absolute(&o.gguf).map_err(|e| e.to_string())?;
    let out = o.out.clone();
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let model = Model::open(&src)?;
    if model.shards[0].g.metadata.contains_key("prism.hadamard.version") {
        return Err(format!("{} holds Hadamard-folded experts (PRISM): their pack is Strata 0.2's (native_experts.txt v5 with \
                            hadamard.txt) and runs on Strata 0.2's engine; this tool and these engines are Strata 0.1.45's",
                           fname(&src)));
    }
    if model.shards.len() > 1 {
        println!("model shards: {}", model.shards.iter().map(|s| fname(&s.path)).collect::<Vec<_>>().join(", "));
    }
    // a dense model (no experts): the pack is its index, its float tensors and its tokenizer (am2strata-hybrid reads
    // dense models from packs too); iq_pack.py packs only MoE models
    if !model.order.iter().any(|n| n.starts_with("blk.") && (n.ends_with("_exps.weight") || n.contains(".ffn_gate.0."))) {
        index_standalone(&src, &out, &model, o.compat_bf16)?;
        if !out.join("tokenizer/vocab.json").exists() || !out.join("tokenizer/chat_template.jinja").exists() {
            strata_gguf::extract_tokenizer(&model.shards[0].g, &out).map_err(|e| e.to_string())?;
        }
        let _ = std::fs::remove_file(out.join("native_experts.txt"));
        println!("a dense model: no experts to place");
        return Ok(());
    }
    let (layout, text, _n_expert, offset) = expert_layout(&model, &src)?;
    let mut experts_bin = o.experts_bin;
    if layout.iter().any(|x| x.names.iter().any(|n| model.where_.contains_key(n) && model.tensor(n).type_name() == "NVFP4")) {
        experts_bin = true;
    }
    let path = out.join("experts.bin");
    let sidecar = out.join("experts.bin.src.json");
    let want = experts_source(&model, &text, offset);
    let read_json = |p: &Path| std::fs::read_to_string(p).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok());
    let mut reuse = path.exists() && std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) == offset && read_json(&sidecar).as_ref() == Some(&want);
    if reuse {
        let mut f = File::open(&path).map_err(|e| e.to_string())?;
        'check: for x in [&layout[0], &layout[layout.len() / 2], &layout[layout.len() - 1]] {
            for e in [0, x.n - 1] {
                f.seek(SeekFrom::Start(x.offset + e * x.blob)).map_err(|e| e.to_string())?;
                let mut have = vec![0u8; x.blob as usize];
                let ok = f.read_exact(&mut have).is_ok();
                let want_blob = match expert_blob(&model, x, e) {
                    Ok(b) => b,
                    Err(_) => {
                        reuse = false;
                        break 'check;
                    }
                };
                if !ok || have != want_blob {
                    reuse = false;
                }
            }
        }
        if !reuse {
            println!("{} matches {} but not this GGUF's expert blobs", path.display(), fname(&sidecar));
        }
    }
    if path.exists() && !reuse && !experts_bin {
        if sidecar.exists() {
            return Err(format!("{} was cut from another source than this model ({}): the engine would read it instead of the GGUF.  \
                                Delete it, or rerun with --experts-bin to rewrite it.", path.display(), fname(&sidecar)));
        }
        println!("warning: {} has no {}, so nothing says it belongs to this model; the engine reads it instead of the GGUF \
                  (rerun with --experts-bin to rewrite it)", path.display(), fname(&sidecar));
    }
    if let Some(base) = &o.base {
        if model.order.iter().any(|n| n != NOT_IN_PACK && model.shards[model.where_[n].0].path != src) {
            return Err("--base needs a model whose tensors are all in shard 1".into());
        }
        let _ = std::fs::remove_file(out.join("native_experts.txt"));
        let base = std::fs::canonicalize(base).map_err(|e| format!("{}: {e}", base.display()))?;
        index_from_base(&src, &base, &out, &model)?;
    } else {
        index_standalone(&src, &out, &model, o.compat_bf16)?;
    }
    if !out.join("tokenizer/vocab.json").exists() || !out.join("tokenizer/chat_template.jinja").exists() {
        strata_gguf::extract_tokenizer(&model.shards[0].g, &out).map_err(|e| e.to_string())?;
    }
    let tmp = out.join("native_experts.txt.tmp");
    std::fs::write(&tmp, &text).map_err(|e| e.to_string())?;
    rename(&tmp, &out.join("native_experts.txt"))?;
    if o.skip_experts || !experts_bin {
        if path.exists() && !experts_bin {
            println!("note: {}/experts.bin exists; the engine reads it instead of the GGUF", out.display());
        }
        return Ok(());
    }
    if reuse {
        println!("experts.bin was cut from this model's shards ({}); not rewritten", fname(&sidecar));
        return Ok(());
    }
    let _ = std::fs::remove_file(&sidecar);
    let part = out.join("experts.bin.tmp");
    let mut fo = BufWriter::with_capacity(32 << 20, File::create(&part).map_err(|e| e.to_string())?);
    for x in &layout {
        for e in 0..x.n {
            fo.write_all(&expert_blob(&model, x, e)?).map_err(|e| e.to_string())?;
        }
        if x.l % 8 == 0 {
            println!("  layer {:2}  {:<8}/{:<7} blob {:8}  at {:.2} GiB", x.l, strata_gguf::type_name(x.gu_type),
                     strata_gguf::type_name(x.d_type), x.blob, x.offset as f64 / (1u64 << 30) as f64);
        }
    }
    fo.flush().map_err(|e| e.to_string())?;
    drop(fo);
    rename(&part, &path)?;
    let side_tmp = out.join("experts.bin.src.json.tmp");
    std::fs::write(&side_tmp, py_json(&want, Some(2)) + "\n").map_err(|e| e.to_string())?;
    rename(&side_tmp, &sidecar)?;
    println!("experts.bin: {} layers, {:.2} GiB", layout.len(), offset as f64 / (1u64 << 30) as f64);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forms_and_names() {
        assert_eq!(form_of("blk.12.ssm_a"), Some("F32"));
        assert_eq!(form_of("output_hc_down.weight"), Some("BF16"));
        assert_eq!(form_of("blk.x.ssm_a"), None);
        assert_eq!(split_name("M-IQ3_XXS-00001-of-00002.gguf"), Some(("M-IQ3_XXS".into(), 2)));
        assert_eq!(split_name("M-1-of-2.gguf"), None);
        assert_eq!(g3(0.001953125), "0.00195");
        assert_eq!(g3(0.25), "0.25");
        assert_eq!(g3(1.5e-7), "1.5e-07");
    }
}
