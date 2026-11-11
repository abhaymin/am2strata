//! The machine's facts (docs/OPTIMIZER-PLAN.md, step 1): what the optimizer tunes for. The cheap facts (CPU, RAM,
//! GPUs and their links, the models' disk) are read at once; the disk's read speed is measured only when asked
//! (`POST /api/machine/measure`), since it reads a gigabyte. Kept in `<data>/machine.json` with a fingerprint of the
//! hardware, so a changed machine (a card added, RAM changed) is noticed and its old measurements are not used.

use serde_json::{json, Value};
use std::path::Path;

/// The facts, read now (no measurement).
pub fn facts(models_dir: &Path) -> Value {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let field = |k: &str| cpuinfo.lines().find(|l| l.starts_with(k)).and_then(|l| l.split(':').nth(1)).map(|v| v.trim().to_string());
    let cores: std::collections::HashSet<(String, String)> = cpuinfo.split("\n\n").filter_map(|b| {
        let get = |k: &str| b.lines().find(|l| l.starts_with(k)).and_then(|l| l.split(':').nth(1)).map(|v| v.trim().to_string());
        Some((get("physical id")?, get("core id")?))
    }).collect();
    let flags = field("flags").unwrap_or_default();
    let has = |f: &str| flags.split_whitespace().any(|x| x == f);
    let simd: Vec<&str> = ["avx2", "fma", "avx512f", "avx512_bf16", "avx512_vnni", "amx_tile", "amx_bf16"].into_iter().filter(|f| has(f)).collect();
    json!({
        "cpu": {"name": field("model name"), "cores": (!cores.is_empty()).then_some(cores.len()),
                "threads": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1), "simd": simd},
        "ram_gb": (crate::mem::total_gb() * 10.0).round() / 10.0,
        "gpus": gpus(),
        "disk": disk(models_dir),
    })
}

/// Each NVIDIA GPU: name, VRAM, and its link (PCIe generation and width, now and at most; a card on USB4 or
/// OCuLink shows a narrow link, which bounds expert streaming).
fn gpus() -> Value {
    let q = "index,name,memory.total,pci.bus_id,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.gpucurrent,pcie.link.width.current";
    let Ok(o) = std::process::Command::new("nvidia-smi").args([&format!("--query-gpu={q}"), "--format=csv,noheader,nounits"]).output() else {
        return json!([]);
    };
    parse_gpus(&String::from_utf8_lossy(&o.stdout))
}

fn parse_gpus(text: &str) -> Value {
    let num = |s: &str| s.trim().parse::<f64>().ok();
    Value::Array(text.lines().filter(|l| !l.trim().is_empty()).map(|l| {
        let f: Vec<&str> = l.split(',').map(str::trim).collect();
        let g = |i: usize| f.get(i).copied().unwrap_or("");
        let (gen, width) = (num(g(4)), num(g(5)));
        json!({"index": num(g(0)), "name": g(1), "vram_gb": num(g(2)).map(|m| (m / 1024.0 * 10.0).round() / 10.0),
               "pci": g(3), "pcie_gen_max": gen, "pcie_width_max": width, "pcie_gen": num(g(6)), "pcie_width": num(g(7)),
               // GB/s one way at the link's best: about 0.985 GB/s a lane at gen 3, doubling each generation
               "link_gb_s": gen.zip(width).map(|(g, w)| ((0.985 * 2f64.powf(g - 3.0) * w) * 10.0).round() / 10.0)})
    }).collect())
}

/// The disk the models are on: its device, whether it spins, its model.
fn disk(models_dir: &Path) -> Value {
    let dev = std::fs::read_to_string("/proc/mounts").ok().and_then(|m| {
        let dir = std::fs::canonicalize(models_dir).ok()?;
        m.lines().filter_map(|l| { let mut f = l.split_whitespace(); Some((f.next()?.to_string(), f.next()?.to_string())) })
            .filter(|(d, mnt)| d.starts_with("/dev/") && dir.starts_with(mnt)).max_by_key(|(_, mnt)| mnt.len()).map(|(d, _)| d)
    });
    let block = dev.as_deref().map(|d| {
        let name = d.trim_start_matches("/dev/");
        // nvme0n1p2 -> nvme0n1, sda3 -> sda
        let base = if name.starts_with("nvme") { name.split('p').next().unwrap_or(name).to_string() }
                   else { name.trim_end_matches(|c: char| c.is_ascii_digit()).to_string() };
        base
    });
    let read = |f: &str| block.as_ref().and_then(|b| std::fs::read_to_string(format!("/sys/block/{b}/{f}")).ok()).map(|s| s.trim().to_string());
    json!({"device": dev, "model": read("device/model"), "rotational": read("queue/rotational").map(|r| r == "1")})
}

/// A fingerprint of the hardware the measurements belong to.
pub fn fingerprint(f: &Value) -> String {
    use sha2::Digest;
    let gpus: Vec<String> = f["gpus"].as_array().into_iter().flatten()
        .map(|g| format!("{}/{}/{}", g["name"], g["vram_gb"], g["pcie_width_max"])).collect();
    let key = format!("{}|{}|{}|{}", f["cpu"]["name"], f["ram_gb"].as_f64().unwrap_or(0.0).round(), gpus.join(","), f["disk"]["model"]);
    let h = sha2::Sha256::digest(key.as_bytes());
    h.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// The disk's sequential read speed in MB/s, from the largest model file (up to 1 GiB, past the page cache).
pub fn measure_read(models_dir: &Path) -> Option<(f64, String)> {
    let mut best: Option<(u64, std::path::PathBuf)> = None;
    let mut stack = vec![models_dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).ok()?.flatten() {
            let Ok(m) = e.metadata() else { continue };
            if m.is_dir() && stack.len() < 64 {
                stack.push(e.path());
            } else if m.is_file() && best.as_ref().is_none_or(|(n, _)| m.len() > *n) {
                best = Some((m.len(), e.path()));
            }
        }
    }
    let (size, path) = best?;
    read_speed(&path, size.min(1 << 30)).map(|s| (s, path.display().to_string()))
}

fn read_speed(path: &Path, bytes: u64) -> Option<f64> {
    use std::io::Read;
    use std::os::unix::io::AsRawFd;
    let mut f = std::fs::File::open(path).ok()?;
    // drop the file's cached pages, so the disk is read and not memory
    unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, bytes as i64, libc::POSIX_FADV_DONTNEED) };
    let mut buf = vec![0u8; 8 << 20];
    let (t0, mut got) = (std::time::Instant::now(), 0u64);
    while got < bytes {
        let n = f.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        got += n as u64;
    }
    let s = t0.elapsed().as_secs_f64();
    unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, bytes as i64, libc::POSIX_FADV_DONTNEED) };
    (got > 0 && s > 0.0).then(|| (got as f64 / 1e6 / s).round())
}

/// The saved record: the facts now, and the measurements kept from before only when the fingerprint is the same.
pub fn view(data_dir: &Path, models_dir: &Path) -> Value {
    let f = facts(models_dir);
    let fp = fingerprint(&f);
    let saved: Value = std::fs::read_to_string(data_dir.join("machine.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
    let changed = saved["fingerprint"].as_str().is_some_and(|s| s != fp);
    let measured = if changed { Value::Null } else { saved["measured"].clone() };
    let v = json!({"fingerprint": fp, "facts": f, "measured": measured, "hardware_changed": changed});
    if saved.is_null() || changed {
        let _ = std::fs::write(data_dir.join("machine.json"), serde_json::to_string_pretty(&v).unwrap_or_default());
    }
    v
}

pub fn measure(data_dir: &Path, models_dir: &Path) -> Value {
    let mut v = view(data_dir, models_dir);
    let m = measure_read(models_dir);
    v["measured"] = json!({"at": crate::now(), "disk_read_mb_s": m.as_ref().map(|x| x.0), "read_from": m.map(|x| x.1)});
    v["hardware_changed"] = json!(false);
    let _ = std::fs::write(data_dir.join("machine.json"), serde_json::to_string_pretty(&v).unwrap_or_default());
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpus_and_their_links_are_read() {
        let g = parse_gpus("0, NVIDIA GeForce RTX 5060 Ti, 16311, 00000000:01:00.0, 5, 8, 1, 4\n1, NVIDIA GeForce RTX 5060 Ti, 16311, 00000000:08:00.0, 4, 4, 1, 4\n");
        assert_eq!(g.as_array().unwrap().len(), 2);
        assert_eq!(g[0]["vram_gb"], 15.9);
        assert_eq!(g[0]["link_gb_s"], 31.5, "gen 5 x8");
        assert_eq!(g[1]["link_gb_s"], 7.9, "gen 4 x4 (the USB4 dock)");
        assert_eq!(g[1]["pci"], "00000000:08:00.0");
    }

    #[test]
    fn the_fingerprint_follows_the_hardware() {
        let a = json!({"cpu": {"name": "x"}, "ram_gb": 62.4, "gpus": [{"name": "g", "vram_gb": 16.0, "pcie_width_max": 8}], "disk": {"model": "d"}});
        let mut b = a.clone();
        b["ram_gb"] = json!(62.1);
        assert_eq!(fingerprint(&a), fingerprint(&b), "RAM read a little differently is the same machine");
        b["gpus"].as_array_mut().unwrap().push(json!({"name": "g2"}));
        assert_ne!(fingerprint(&a), fingerprint(&b), "a card added is a new machine");
    }

    #[test]
    fn the_read_speed_is_measured_on_a_file() {
        let d = std::env::temp_dir().join(format!("machine-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("m.gguf"), vec![1u8; 4 << 20]).unwrap();
        let (mb_s, from) = measure_read(&d).unwrap();
        assert!(mb_s > 0.0 && from.ends_with("m.gguf"));
        std::fs::remove_dir_all(&d).unwrap();
    }
}
