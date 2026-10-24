//! A GGUF with one tensor per expert (`blk.N.ffn_{gate,up,down}.E.weight`, converted before April 2024: most Mixtral
//! 8x7B files) rewritten with the experts stacked (`blk.N.ffn_{gate,up,down}_exps.weight`), the layout llama.cpp has
//! read since and the only one recent llama.cpp loads. The weights are copied byte for byte (expert E's rows follow
//! expert E-1's, as llama.cpp stacks them); the key/values are copied verbatim; only the tensor infos change.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

pub struct Options {
    pub input: PathBuf,
    pub output: PathBuf,
}

struct Tensor {
    name: String,
    dims: Vec<u64>,
    ty: u32,
    offset: u64,
    bytes: u64,
}

fn rd<const N: usize>(f: &mut impl Read) -> std::io::Result<[u8; N]> {
    let mut b = [0u8; N];
    f.read_exact(&mut b)?;
    Ok(b)
}
fn u32_(f: &mut impl Read) -> std::io::Result<u32> {
    Ok(u32::from_le_bytes(rd::<4>(f)?))
}
fn u64_(f: &mut impl Read) -> std::io::Result<u64> {
    Ok(u64::from_le_bytes(rd::<8>(f)?))
}
fn string(f: &mut impl Read) -> std::io::Result<String> {
    let n = u64_(f)? as usize;
    let mut b = vec![0u8; n];
    f.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

/// Skips one value of GGUF type `t`.
fn skip(f: &mut (impl Read + Seek), t: u32) -> std::io::Result<()> {
    match t {
        0 | 1 | 7 => f.seek(SeekFrom::Current(1)).map(|_| ()),
        2 | 3 => f.seek(SeekFrom::Current(2)).map(|_| ()),
        4..=6 => f.seek(SeekFrom::Current(4)).map(|_| ()),
        10..=12 => f.seek(SeekFrom::Current(8)).map(|_| ()),
        8 => {
            let n = u64_(f)?;
            f.seek(SeekFrom::Current(n as i64)).map(|_| ())
        }
        9 => {
            let et = u32_(f)?;
            let n = u64_(f)?;
            for _ in 0..n {
                skip(f, et)?;
            }
            Ok(())
        }
        _ => Err(std::io::Error::other(format!("unknown GGUF value type {t}"))),
    }
}

fn parts(name: &str) -> Option<(usize, &str, usize)> {
    // blk.N.ffn_ROLE.E.weight
    let rest = name.strip_prefix("blk.")?;
    let (l, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("ffn_")?;
    let (role, rest) = rest.split_once('.')?;
    let (e, tail) = rest.split_once('.')?;
    if tail != "weight" || !matches!(role, "gate" | "up" | "down") {
        return None;
    }
    Some((l.parse().ok()?, role, e.parse().ok()?))
}

pub fn run(o: &Options) -> Result<(), String> {
    let err = |e: std::io::Error| e.to_string();
    let mut f = File::open(&o.input).map_err(|e| format!("{}: {e}", o.input.display()))?;
    if &rd::<4>(&mut f).map_err(err)? != b"GGUF" || u32_(&mut f).map_err(err)? != 3 {
        return Err("not a GGUF v3 file".into());
    }
    let n_tensors = u64_(&mut f).map_err(err)?;
    let n_kv = u64_(&mut f).map_err(err)?;
    let kv_start = f.stream_position().map_err(err)?;
    let mut alignment = 32u64;
    for _ in 0..n_kv {
        let k = string(&mut f).map_err(err)?;
        let t = u32_(&mut f).map_err(err)?;
        if k == "general.alignment" && t == 4 {
            alignment = u32_(&mut f).map_err(err)? as u64;
        } else {
            skip(&mut f, t).map_err(err)?;
        }
    }
    let kv_end = f.stream_position().map_err(err)?;
    let g = strata_gguf::Gguf::open(&o.input).map_err(|e| e.to_string())?;
    let mut tensors: Vec<Tensor> = Vec::new();
    for t in &g.tensors {
        let bytes = t.expected_bytes().ok_or_else(|| format!("{}: unsupported type", t.name))?;
        tensors.push(Tensor { name: t.name.clone(), dims: t.dims.clone(), ty: t.ggml_type, offset: g.data_start + t.offset, bytes });
    }
    if tensors.len() as u64 != n_tensors {
        return Err("the tensor count disagrees with the header".into());
    }
    // group the per-expert tensors: (layer, role) -> [expert index -> tensor index]
    let mut groups: HashMap<(usize, String), Vec<(usize, usize)>> = HashMap::new();
    for (i, t) in tensors.iter().enumerate() {
        if let Some((l, role, e)) = parts(&t.name) {
            groups.entry((l, role.to_string())).or_default().push((e, i));
        }
    }
    if groups.is_empty() {
        return Err("no per-expert tensors (blk.N.ffn_ROLE.E.weight): nothing to stack".into());
    }
    // the output's tensors, in the input's order, a stacked tensor where its expert 0 was
    struct Out {
        name: String,
        dims: Vec<u64>,
        ty: u32,
        src: Vec<usize>,
        bytes: u64,
    }
    let mut out: Vec<Out> = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        match parts(&t.name) {
            Some((l, role, 0)) => {
                let mut g = groups[&(l, role.to_string())].clone();
                g.sort();
                if g.iter().enumerate().any(|(k, (e, _))| *e != k) {
                    return Err(format!("layer {l} {role}: the experts are not numbered 0..{}", g.len()));
                }
                let first = &tensors[g[0].1];
                if g.iter().any(|(_, j)| tensors[*j].ty != first.ty || tensors[*j].dims != first.dims) {
                    return Err(format!("layer {l} {role}: the experts differ in type or shape, so they cannot be stacked"));
                }
                let mut dims = first.dims.clone();
                dims.push(g.len() as u64);
                out.push(Out { name: format!("blk.{l}.ffn_{role}_exps.weight"), dims, ty: first.ty,
                               src: g.iter().map(|(_, j)| *j).collect(), bytes: first.bytes * g.len() as u64 });
            }
            Some(_) => {}
            None => out.push(Out { name: t.name.clone(), dims: t.dims.clone(), ty: t.ty, src: vec![i], bytes: t.bytes }),
        }
    }
    let pad = |n: u64| n.div_ceil(alignment) * alignment - n;
    let mut head = Vec::new();
    head.extend(b"GGUF");
    head.extend(3u32.to_le_bytes());
    head.extend((out.len() as u64).to_le_bytes());
    head.extend(n_kv.to_le_bytes());
    f.seek(SeekFrom::Start(kv_start)).map_err(err)?;
    let mut kv = vec![0u8; (kv_end - kv_start) as usize];
    f.read_exact(&mut kv).map_err(err)?;
    head.extend(kv);
    let mut off = 0u64;
    for t in &out {
        head.extend((t.name.len() as u64).to_le_bytes());
        head.extend(t.name.as_bytes());
        head.extend((t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            head.extend(d.to_le_bytes());
        }
        head.extend(t.ty.to_le_bytes());
        head.extend(off.to_le_bytes());
        off += t.bytes + pad(t.bytes);
    }
    head.extend(vec![0u8; pad(head.len() as u64) as usize]);
    let tmp = o.output.with_extension("gguf.tmp");
    let mut w = BufWriter::with_capacity(32 << 20, File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?);
    w.write_all(&head).map_err(err)?;
    let mut buf = vec![0u8; 16 << 20];
    for (k, t) in out.iter().enumerate() {
        for &j in &t.src {
            let s = &tensors[j];
            f.seek(SeekFrom::Start(s.offset)).map_err(err)?;
            let mut left = s.bytes;
            while left > 0 {
                let n = left.min(buf.len() as u64) as usize;
                f.read_exact(&mut buf[..n]).map_err(err)?;
                w.write_all(&buf[..n]).map_err(err)?;
                left -= n as u64;
            }
        }
        w.write_all(&vec![0u8; pad(t.bytes) as usize]).map_err(err)?;
        if k % 50 == 0 {
            println!("  {k}/{} tensors", out.len());
        }
    }
    w.flush().map_err(err)?;
    drop(w);
    std::fs::rename(&tmp, &o.output).map_err(err)?;
    println!("{}: {} tensors ({} stacked from {} per-expert tensors)", o.output.display(), out.len(), groups.len(),
             groups.values().map(|g| g.len()).sum::<usize>());
    Ok(())
}
