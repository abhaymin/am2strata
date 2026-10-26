//! Strata's placement inputs for this engine: a pack (`--pack`, written by `am2strata-run iq-pack` / PolyStrata's
//! tools/iq_pack.py), the expert loadings (`--expert-profile`, PolyStrata's STRP v1 ranked list of (layer, expert)),
//! and the residency they drive (`--resident-experts`).
//!
//! - The pack's dense.bin holds the float tensors (norms, routers, SSM parameters) in the forms strata's engines read;
//!   they replace the GGUF's copies, so this engine computes with exactly what the CUDA engine uploads. Its
//!   native_experts.txt names, per MoE layer, the GGUF offsets of the gate/up/down expert tensors: a pack built from a
//!   different file is refused instead of read.
//! - The profile ranks experts hottest first. The engine counts how often each expert is routed to and writes a
//!   learned profile (`--expert-profile-save`) in the same format, ranked as PolyStrata's rank_learned_profile does:
//!   by heat, then by the prior profile's rank.
//! - Residency: the hottest experts within `--expert-ram-gib` are paged in at load and kept (mlock when the memlock limit
//!   allows it, else MADV_WILLNEED), the others are marked MADV_COLD so that under memory pressure the kernel evicts them
//!   first: a model larger than the free RAM then reads only its cold experts from the SSD.

use crate::model::{Model, Moe, QMat};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::Ordering;

/// A ranked expert profile: (layer, expert) pairs, hottest first.
#[derive(Debug)]
pub struct Profile {
    pub n_layers: usize,
    pub n_expert: usize,
    pub ranked: Vec<(u16, u16)>,
}

impl Profile {
    /// Reads a version 1 STRP file (expert_cache.cpp read_expert_profile), checked against the model's shape.
    pub fn read(path: &Path, n_layers: usize, n_expert: usize) -> Result<Profile, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if b.len() < 24 || &b[..4] != b"STRP" {
            return Err(format!("{} is not an expert profile (no STRP header)", path.display()));
        }
        let h = |i: usize| u32::from_le_bytes(b[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
        let (version, nl, ne, slots, n) = (h(0), h(1), h(2), h(3), h(4));
        if version != 1 {
            return Err(format!("{} is a version {version} profile; this engine reads version 1", path.display()));
        }
        if nl != n_layers || ne != n_expert {
            return Err(format!("{} is {nl}x{ne} but this model is {n_layers}x{n_expert}: a profile for a different model", path.display()));
        }
        if n > slots || b.len() < 24 + 4 * n {
            return Err(format!("{}: the ranked list is truncated or longer than its slots", path.display()));
        }
        let mut ranked = Vec::with_capacity(n);
        for i in 0..n {
            let at = 24 + 4 * i;
            let (l, e) = (u16::from_le_bytes([b[at], b[at + 1]]), u16::from_le_bytes([b[at + 2], b[at + 3]]));
            if l as usize >= nl || e as usize >= ne {
                return Err(format!("{}: pair {i} is (layer {l}, expert {e}), out of range", path.display()));
            }
            ranked.push((l, e));
        }
        Ok(Profile { n_layers, n_expert, ranked })
    }

    /// Writes it as write_expert_profile does: the header, the pairs, then the rank of every (layer, expert) (-1 when
    /// unranked); through a temporary file, so a reader never sees half of one.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        if self.n_layers > 65535 || self.n_expert > 65535 {
            return Err("the model's layout does not fit the profile format".into());
        }
        let mut table = vec![-1i32; self.n_layers * self.n_expert];
        let mut out = Vec::with_capacity(24 + self.ranked.len() * 4 + table.len() * 4);
        out.extend_from_slice(b"STRP");
        for v in [1, self.n_layers, self.n_expert, self.ranked.len(), self.ranked.len()] {
            out.extend_from_slice(&(v as u32).to_le_bytes());
        }
        for (r, &(l, e)) in self.ranked.iter().enumerate() {
            table[l as usize * self.n_expert + e as usize] = r as i32;
            out.extend_from_slice(&l.to_le_bytes());
            out.extend_from_slice(&e.to_le_bytes());
        }
        for v in table {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let tmp = path.with_extension("tmp");
        let mut f = std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        f.write_all(&out).and_then(|_| f.sync_all()).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The learned ranking from routing counts (rank_learned_profile without the resident tier): by heat, then the
    /// prior profile's rank, then (layer, expert). Only experts that exist in their layer are listed.
    pub fn learned(model: &Model, prior: Option<&Profile>) -> Profile {
        let ne = model.profile_experts;
        let nl = model.layers.len();
        let mut prior_rank = vec![usize::MAX; nl * ne];
        if let Some(p) = prior {
            for (r, &(l, e)) in p.ranked.iter().enumerate() {
                let i = l as usize * ne + e as usize;
                if prior_rank[i] == usize::MAX {
                    prior_rank[i] = r;
                }
            }
        }
        let mut all: Vec<usize> = model.moes().flat_map(|m| (0..m.n_expert).map(move |e| m.layer * ne + e)).collect();
        let heat = |i: usize| model.heat[i].load(Ordering::Relaxed);
        all.sort_by(|&a, &b| heat(b).cmp(&heat(a)).then(prior_rank[a].cmp(&prior_rank[b])).then(a.cmp(&b)));
        Profile { n_layers: nl, n_expert: ne, ranked: all.into_iter().map(|i| ((i / ne) as u16, (i % ne) as u16)).collect() }
    }
}

/// The pack's index (index.txt v3) and its native experts (native_experts.txt v3 / v4, or am2strata's p1).
pub struct Pack {
    pub dir: std::path::PathBuf,
    /// float tensors stored in dense.bin: name -> (kind, offset, bytes)
    floats: HashMap<String, (u32, u64, u64)>,
    /// per MoE layer: (gate, up, down) absolute offsets in the GGUF named by the header (0 when the role is absent)
    experts: HashMap<usize, [u64; 3]>,
    n_native: usize,
}

impl Pack {
    pub fn open(dir: &Path) -> Result<Pack, String> {
        let read = |n: &str| std::fs::read_to_string(dir.join(n)).map_err(|e| format!("pack {}: {n}: {e}", dir.display()));
        let index = read("index.txt")?;
        if !index.starts_with("# strata pack index v3") {
            return Err(format!("pack {}: index.txt is not a v3 strata pack index", dir.display()));
        }
        let mut floats = HashMap::new();
        let mut n_native = 0;
        for line in index.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 12 {
                return Err(format!("pack {}: a short index row: {line}", dir.display()));
            }
            let kind: u32 = f[2].parse().map_err(|_| format!("pack: bad kind in {line}"))?;
            if kind == 0 {
                n_native += 1;
                continue;
            }
            let num = |i: usize| f[i].parse::<u64>().map_err(|_| format!("pack: bad number in {line}"));
            floats.insert(f[0].to_string(), (kind, num(3)?, num(4)?));
        }
        let mut experts = HashMap::new();
        if let Ok(text) = read("native_experts.txt") {
            // by its header: strata's v3 / v4 and am2strata's p1 (per-layer pruned, an n_expert column); Strata 0.2's v5
            // is Hadamard-folded experts, which only Strata 0.2's engine computes - refused, never misread
            let head = text.lines().next().unwrap_or("");
            if head.starts_with("# strata native experts v5:") {
                return Err(format!("pack {}: native_experts.txt v5 holds Hadamard-folded experts (Strata 0.2), which this \
                                    engine does not compute", dir.display()));
            }
            if !(head.starts_with("# strata native experts v3:") || head.starts_with("# strata native experts v4:")
                 || head.starts_with("# am2strata native experts p1:")) {
                return Err(format!("pack {}: native_experts.txt has a header this engine does not know: {:?}", dir.display(),
                                   head.chars().take(48).collect::<String>()));
            }
            for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
                let f: Vec<&str> = line.split_whitespace().collect();
                if f.len() >= 8 {
                    let n = |i: usize| f[i].parse::<u64>().unwrap_or(0);
                    experts.insert(n(0) as usize, [n(5), n(6), n(7)]);
                }
            }
        }
        Ok(Pack { dir: dir.to_path_buf(), floats, experts, n_native })
    }

    /// Checks the pack against the GGUF's tensors (its expert offsets must be this file's), then swaps in the float
    /// tensors from dense.bin in the pack's forms. Returns how many tensors now come from the pack.
    pub fn apply(&self, tensors: &mut HashMap<String, QMat>, offsets: &HashMap<String, u64>) -> Result<usize, String> {
        for (&l, want) in &self.experts {
            for (i, role) in ["gate", "up", "down"].iter().enumerate() {
                let name = format!("blk.{l}.ffn_{role}_exps.weight");
                match offsets.get(&name) {
                    Some(&off) if want[i] != 0 && want[i] != off => {
                        return Err(format!("pack {}: {name} is at {} in the pack but at {off} in this GGUF: the pack was built \
                                            from a different file", self.dir.display(), want[i]));
                    }
                    None if want[i] != 0 && i > 0 => {
                        return Err(format!("pack {}: {name} is in the pack but not in this GGUF: a pack for a different model",
                                           self.dir.display()));
                    }
                    _ => {}
                }
            }
        }
        let f = std::fs::File::open(self.dir.join("dense.bin")).map_err(|e| format!("pack: dense.bin: {e}"))?;
        let mm: &'static memmap2::Mmap = Box::leak(Box::new(unsafe { memmap2::Mmap::map(&f) }.map_err(|e| e.to_string())?));
        let mut n = 0;
        for (name, &(kind, off, bytes)) in &self.floats {
            let Some(q) = tensors.get_mut(name) else {
                return Err(format!("pack {}: {name} is not in this GGUF: a pack for a different model", self.dir.display()));
            };
            let end = (off + bytes) as usize;
            if end > mm.len() {
                return Err(format!("pack: {name} lies beyond dense.bin"));
            }
            let raw: &'static [u8] = &mm[off as usize..end];
            // kinds: 2 F32, 3 F32 that the loader rounds to F16 (done here, as the CUDA engine's loader does), 4 BF16, 5 F16
            *q = match kind {
                2 => q.with_data(raw, "F32")?,
                3 => {
                    let h: Vec<u8> = raw.chunks_exact(4)
                        .flat_map(|c| strata_pack::num::f32_to_f16(f32::from_le_bytes(c.try_into().unwrap())).to_le_bytes()).collect();
                    q.with_data(Box::leak(h.into_boxed_slice()), "F16")?
                }
                4 => q.with_data(raw, "BF16")?,
                5 => q.with_data(raw, "F16")?,
                k => return Err(format!("pack: {name} has an unknown kind {k}")),
            };
            n += 1;
        }
        Ok(n)
    }

    pub fn n_native(&self) -> usize {
        self.n_native
    }
}

/// What residency did: experts kept in RAM (and how), and the rest marked cold.
pub struct Residency {
    pub resident: Vec<bool>,
    pub n_resident: usize,
    pub bytes: u64,
    pub locked: bool,
}

fn expert_ranges(m: &Moe, e: usize) -> Vec<&'static [u8]> {
    let mut v = vec![];
    if let Some(g) = &m.gate_exps {
        v.push(g.0[e].data);
    }
    v.push(m.up_exps.0[e].data);
    v.push(m.down_exps.0[e].data);
    v
}

fn advise(b: &[u8], advice: libc::c_int) -> bool {
    // madvise wants page-aligned starts: round the start down, the length up
    let page = 4096usize;
    let start = b.as_ptr() as usize & !(page - 1);
    let len = (b.as_ptr() as usize + b.len() - start).div_ceil(page) * page;
    unsafe { libc::madvise(start as *mut libc::c_void, len, advice) == 0 }
}

/// Keeps the hottest experts (profile order, else layer order) within `budget` bytes in RAM; marks the rest cold.
pub fn place(model: &Model, profile: Option<&Profile>, budget: u64) -> Residency {
    let ne = model.profile_experts;
    let moes: HashMap<usize, &Moe> = model.moes().map(|m| (m.layer, m)).collect();
    let mut order: Vec<(usize, usize)> = match profile {
        Some(p) => p.ranked.iter().map(|&(l, e)| (l as usize, e as usize)).collect(),
        None => vec![],
    };
    // experts the profile does not rank follow, layer by layer
    let mut seen = vec![false; model.layers.len() * ne];
    order.iter().for_each(|&(l, e)| seen[l * ne + e] = true);
    for m in model.moes() {
        for e in 0..m.n_expert {
            if !seen[m.layer * ne + e] {
                order.push((m.layer, e));
            }
        }
    }
    let mut resident = vec![false; model.layers.len() * ne];
    let (mut bytes, mut n_resident, mut locked) = (0u64, 0usize, true);
    for &(l, e) in &order {
        let Some(m) = moes.get(&l) else { continue };
        if e >= m.n_expert {
            continue;
        }
        let ranges = expert_ranges(m, e);
        let size: u64 = ranges.iter().map(|r| r.len() as u64).sum();
        if bytes + size > budget {
            break;
        }
        for r in &ranges {
            if locked && unsafe { libc::mlock(r.as_ptr() as *const libc::c_void, r.len()) } != 0 {
                locked = false;
            }
            if !locked {
                advise(r, libc::MADV_WILLNEED);
            }
        }
        resident[l * ne + e] = true;
        bytes += size;
        n_resident += 1;
    }
    // the cold experts: evicted first under memory pressure, and read without readahead
    for m in model.moes() {
        for e in 0..m.n_expert {
            if !resident[m.layer * ne + e] {
                for r in expert_ranges(m, e) {
                    advise(r, libc::MADV_COLD);
                    advise(r, libc::MADV_RANDOM);
                }
            }
        }
    }
    Residency { resident, n_resident, bytes, locked: locked && n_resident > 0 }
}

/// Routing since the last call that went to resident experts: (resident hits, all routed).
pub fn hit_rate(model: &Model, res: &Residency, last: &mut Vec<u32>) -> (u64, u64) {
    if last.len() != model.heat.len() {
        *last = vec![0; model.heat.len()];
    }
    let (mut hit, mut all) = (0u64, 0u64);
    for (i, h) in model.heat.iter().enumerate() {
        let now = h.load(Ordering::Relaxed);
        let d = (now - last[i]) as u64;
        last[i] = now;
        all += d;
        if res.resident.get(i).copied().unwrap_or(false) {
            hit += d;
        }
    }
    (hit, all)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_round_trip() {
        let p = Profile { n_layers: 3, n_expert: 4, ranked: vec![(1, 2), (0, 3), (2, 0)] };
        let path = std::env::temp_dir().join(format!("am2-prof-{}.bin", std::process::id()));
        p.write(&path).unwrap();
        let b = std::fs::read(&path).unwrap();
        // header, 3 pairs, then the rank table of 3 x 4 (write_expert_profile's layout)
        assert_eq!(b.len(), 24 + 3 * 4 + 12 * 4);
        assert_eq!(&b[..4], b"STRP");
        let rank_of = |l: usize, e: usize| i32::from_le_bytes(b[36 + 4 * (l * 4 + e)..40 + 4 * (l * 4 + e)].try_into().unwrap());
        assert_eq!((rank_of(1, 2), rank_of(0, 3), rank_of(2, 0), rank_of(0, 0)), (0, 1, 2, -1));
        let q = Profile::read(&path, 3, 4).unwrap();
        assert_eq!(q.ranked, p.ranked);
        assert!(Profile::read(&path, 3, 5).unwrap_err().contains("different model"));
        let _ = std::fs::remove_file(&path);
    }
}
