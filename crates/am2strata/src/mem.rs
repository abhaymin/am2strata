//! This PC's free memory, and what each model took from it when it loaded (PRODUCTION-PLAN.md, step 3): a load that
//! would not fit waits a little for memory (the model stopped before it is still giving its memory back), then is
//! refused with a plain message, instead of an out-of-memory kill in the middle of it.

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// MemAvailable in GB (0 when unknown). AM2_TEST_MEM_GB stands in for it in the tests.
pub fn available_gb() -> f64 {
    if let Some(v) = std::env::var("AM2_TEST_MEM_GB").ok().and_then(|v| v.parse().ok()) {
        return v;
    }
    available_mib() as f64 / 1024.0
}

pub fn total_gb() -> f64 {
    if let Some(v) = std::env::var("AM2_TEST_MEM_GB").ok().and_then(|v| v.parse().ok()) {
        return v;
    }
    total_mib() as f64 / 1024.0
}

/// Total RAM this PC has (MiB; 0 when it cannot tell), from strata_pack::sysmem (Linux, macOS, Windows).
pub fn total_mib() -> u64 { strata_pack::sysmem::total_mib() }

/// RAM the OS says is free right now (MiB; 0 when it cannot tell).
pub fn available_mib() -> u64 { strata_pack::sysmem::available_mib() }

#[cfg(test)]
use strata_pack::sysmem::{kb_to_mib, meminfo_kb};

/// Memory kept free beside a model (the OS, the web app, a browser)
pub const MARGIN_GB: f64 = 2.0;
/// How long a load waits for memory the model before it is still giving back
pub const LOAD_WAIT: Duration = Duration::from_secs(60);

/// Is this PC short of memory? (under 3 GB free, or under 5% of the total)
pub fn low(avail: f64, total: f64) -> bool {
    avail > 0.0 && (avail < 3.0 || (total > 0.0 && avail < total * 0.05))
}

/// What each model took from free memory when it last loaded, kept in `<data>/model-memory.json`.
pub struct Needs {
    path: PathBuf,
    map: std::sync::Mutex<BTreeMap<String, f64>>,
}

impl Needs {
    pub fn open(data_dir: &Path) -> Needs {
        Self::open_file(data_dir.join("model-memory.json"))
    }

    /// Another per-model record (model-load-s.json: how long each model takes to load).
    pub fn open_file(path: PathBuf) -> Needs {
        let map = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice::<BTreeMap<String, f64>>(&b).ok()).unwrap_or_default();
        Needs { path, map: std::sync::Mutex::new(map) }
    }

    pub fn get(&self, id: &str) -> Option<f64> {
        self.map.lock().unwrap().get(id).copied()
    }

    /// A load of `id` took `gb` of free memory (the largest seen is kept: a model's need grows with its cache).
    pub fn record(&self, id: &str, gb: f64) {
        let mut m = self.map.lock().unwrap();
        let v = m.entry(id.to_string()).or_insert(0.0);
        *v = v.max(gb.max(0.0));
        let _ = std::fs::write(&self.path, serde_json::to_vec_pretty(&*m).unwrap_or_default());
    }

    pub fn view(&self) -> Value {
        json!(*self.map.lock().unwrap())
    }

    /// A moving average instead of the largest (a load time).
    pub fn record_mean(&self, id: &str, v: f64) {
        let mut m = self.map.lock().unwrap();
        let x = m.entry(id.to_string()).or_insert(v);
        *x = *x * 0.5 + v * 0.5;
        let _ = std::fs::write(&self.path, serde_json::to_vec_pretty(&*m).unwrap_or_default());
    }

    pub fn all(&self) -> Vec<(String, f64)> {
        self.map.lock().unwrap().iter().map(|(k, v)| (k.clone(), *v)).collect()
    }
}

/// A model's files (a split GGUF's parts too) and their size in GB.
pub fn model_files(first: &Path) -> (Vec<PathBuf>, f64) {
    let name = first.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let mut files = vec![first.to_path_buf()];
    if let (Some(dir), Some(i)) = (first.parent(), name.find("-00001-of-")) {
        let stem = &name[..i];
        if let Ok(rd) = std::fs::read_dir(dir) {
            files = rd.flatten().map(|e| e.path())
                .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(stem) && n.contains("-of-") && n.ends_with(".gguf")))
                .collect();
            files.sort();
        }
    }
    let gb = files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len() as f64).sum::<f64>() / 1073741824.0;
    (files, gb)
}

/// Read files once so they are in the file cache (the next load reads RAM, not the SSD): a plain sequential read,
/// in 8 MiB steps, at the lowest I/O and CPU priority the process can ask for.
pub fn warm(files: &[PathBuf]) -> std::io::Result<u64> {
    use std::io::Read;
    #[cfg(unix)]
    unsafe {
        // nice 19 for this thread (the read competes with nothing that matters)
        let _ = libc_nice();
    }
    let mut total = 0u64;
    let mut buf = vec![0u8; 8 << 20];
    for f in files {
        let mut file = std::fs::File::open(f)?;
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            total += n as u64;
        }
    }
    Ok(total)
}

#[cfg(unix)]
unsafe fn libc_nice() -> i32 {
    extern "C" {
        fn nice(inc: i32) -> i32;
    }
    nice(19)
}

/// Wait up to `wait` for `need` GB (plus the margin) to be free; Err says how much is free.
pub async fn wait_for(need: f64, wait: Duration) -> Result<(), f64> {
    let t0 = std::time::Instant::now();
    loop {
        let avail = available_gb();
        if avail <= 0.0 || avail >= need + MARGIN_GB {
            return Ok(());
        }
        if t0.elapsed() >= wait || std::env::var("AM2_TEST_MEM_GB").is_ok() {
            return Err(avail);
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_memory_and_needs() {
        assert!(low(2.0, 64.0) && low(3.5, 128.0) && !low(10.0, 64.0) && !low(0.0, 64.0));
        // a folder of its own for this run (a CI runner may keep an earlier run's temporary files)
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("am2-mem-{}-{nanos}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let n = Needs::open(&dir);
        n.record("a", 12.0);
        n.record("a", 9.0);
        assert_eq!(Needs::open(&dir).get("a"), Some(12.0), "the largest seen, kept on disk");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // CI runs on Linux, macOS and Windows; on whichever OS the test is on, all RAM is reported and none is over it.
    #[test]
    fn total_is_positive_and_available_le_total() {
        let total = total_mib();
        let avail = available_mib();
        assert!(total > 0, "total_mib() was 0: the memory source is missing on this OS");
        assert!(avail <= total, "available {avail} MiB is more than total {total} MiB");
    }

    #[test]
    fn units_are_mib_and_gib() {
        // 16 GiB of RAM as /proc/meminfo writes it: 16777216 kB
        assert_eq!(kb_to_mib(16_777_216), 16_384);
        // and the GiB view the callers use is MiB / 1024
        assert_eq!(16_384f64 / 1024.0, 16.0);
    }

    // on the machine the test runs on, total RAM is a plausible size in MiB (between 512 MiB and 16 TiB): catches
    // bytes, kB or GiB passed off as MiB on any OS
    #[test]
    fn total_is_a_plausible_number_of_mib() {
        let t = total_mib();
        assert!((512..16 * 1024 * 1024).contains(&t), "total_mib() = {t}: not MiB?");
        let gib = total_gb();
        assert!(gib > 0.5 && gib < 16384.0, "the GiB view is {gib}");
    }

    #[test]
    fn meminfo_parses_a_fixed_string() {
        let text = "MemTotal:       16384 kB\nMemAvailable:     8192 kB\nSwapTotal:        4096 kB\n";
        assert_eq!(meminfo_kb(text, "MemTotal:"), Some(16384));
        assert_eq!(meminfo_kb(text, "MemAvailable:"), Some(8192));
        assert_eq!(meminfo_kb(text, "SwapTotal:"), Some(4096));
        assert_eq!(meminfo_kb(text, "Missing:"), None);
        // only the line's own number counts (a later kB in the row is ignored)
        assert_eq!(meminfo_kb(text, "MemTotal: 99"), None, "the key must start the line");
    }
}
