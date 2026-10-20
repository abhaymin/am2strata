//! The machine the engine runs on, and the launch settings that follow from it.
//!
//! The engine already sizes its expert cache to the VRAM it finds, which is most of the per-card tuning. What the
//! server decides is what goes beside it: where the image encoder runs (on the GPU only with 15.5 GB of VRAM or more,
//! PolyStrata setup's rule, so a 12 GB card keeps its VRAM for experts), and whether the engine binary was compiled
//! for this card at all (an RTX 3060 is sm_86, an RTX 5060 Ti sm_120).

use std::path::Path;
use std::process::Command;

/// VRAM below which the image encoder runs on the CPU (setup.py POLY_VISION_GPU_GB).
pub const VISION_GPU_MIN_GB: f64 = 15.5;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Gpu {
    pub index: u32,
    pub name: String,
    pub vram_mib: u64,
    /// compute capability as an architecture number: 8.6 -> 86, 12.0 -> 120
    pub arch: u32,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Hardware {
    pub gpus: Vec<Gpu>,
    pub ram_mib: u64,
    pub cpus: usize,
}

impl Hardware {
    /// nvidia-smi for the cards, /proc/meminfo for the RAM. Missing tools leave the fields empty.
    pub fn detect() -> Self {
        let mut gpus = Vec::new();
        if let Ok(out) = Command::new("nvidia-smi")
            .args(["--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader,nounits"])
            .output()
        {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                let f: Vec<&str> = line.split(',').map(str::trim).collect();
                if f.len() < 4 {
                    continue;
                }
                let arch = f[3]
                    .split_once('.')
                    .and_then(|(a, b)| Some(a.parse::<u32>().ok()? * 10 + b.parse::<u32>().ok()?))
                    .unwrap_or(0);
                gpus.push(Gpu {
                    index: f[0].parse().unwrap_or(0),
                    name: f[1].to_string(),
                    vram_mib: f[2].parse().unwrap_or(0),
                    arch,
                });
            }
        }
        let ram_mib = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|m| {
                m.lines().find(|l| l.starts_with("MemTotal:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
            })
            .map(|kb| kb / 1024)
            .unwrap_or(0);
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        Self { gpus, ram_mib, cpus }
    }

    pub fn gpu(&self, index: Option<u32>) -> Option<&Gpu> {
        match index {
            Some(i) => self.gpus.iter().find(|g| g.index == i),
            None => self.gpus.first(),
        }
    }
}

/// The architectures an engine directory's BUILD.json says it was compiled for.
pub fn engine_archs(exe: &Path) -> Vec<u32> {
    let Some(dir) = exe.parent() else { return Vec::new() };
    std::fs::read_to_string(dir.join("BUILD.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("archs").and_then(|a| a.as_array()).cloned())
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect())
        .unwrap_or_default()
}

/// Whether a binary built for `archs` runs on a card of `arch`: CUDA runs SASS on the same major architecture at an
/// equal or newer minor revision (sm_86 code runs on sm_89, not on sm_120 or sm_75).
pub fn arch_supported(archs: &[u32], arch: u32) -> bool {
    archs.is_empty() || archs.iter().any(|&a| a / 10 == arch / 10 && a <= arch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_rules() {
        assert!(arch_supported(&[120], 120));
        assert!(!arch_supported(&[120], 86)); // a 5060-only build does not run on a 3060
        assert!(arch_supported(&[86, 120], 86));
        assert!(arch_supported(&[86], 89));
        assert!(arch_supported(&[], 86)); // unknown build: let the engine decide
    }
}
