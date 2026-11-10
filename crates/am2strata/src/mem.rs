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
        let path = data_dir.join("model-memory.json");
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
        let dir = std::env::temp_dir().join(format!("am2-mem-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = Needs::open(&dir);
        n.record("a", 12.0);
        n.record("a", 9.0);
        assert_eq!(Needs::open(&dir).get("a"), Some(12.0), "the largest seen, kept on disk");
    }
}
