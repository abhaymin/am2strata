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
    meminfo("MemAvailable:").unwrap_or(0.0)
}

pub fn total_gb() -> f64 {
    if let Some(v) = std::env::var("AM2_TEST_MEM_GB").ok().and_then(|v| v.parse().ok()) {
        return v;
    }
    meminfo("MemTotal:").unwrap_or(0.0)
}

fn meminfo(key: &str) -> Option<f64> {
    std::fs::read_to_string("/proc/meminfo").ok()?
        .lines().find(|l| l.starts_with(key))?.split_whitespace().nth(1)?.parse::<f64>().ok().map(|kb| kb / 1048576.0)
}

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
}
