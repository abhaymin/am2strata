//! am2strata-hybrid: an engine in Rust for the architectures in model.rs (Nemotron-H, Mistral / Mixtral, GLM-4.7-Flash
//! and the DeepSeek family), speaking the strata
//! engine protocol (docs/PROTOCOL.md) so am2Strata's server drives it like `strata` and `strata-poly`:
//!   am2strata-hybrid --serve --model <shard 1>.gguf [--max-context N] [--threads N]
//!       [--pack DIR] [--expert-profile FILE] [--expert-profile-save FILE] [--resident-experts [--expert-ram-gib N]]
//! The computation runs on the CPU (correctness first, compared with llama.cpp by tools/reference-llamacpp); the
//! placement is strata's (pack.rs): a pack's float tensors, the expert loadings (profile), and the hottest experts kept
//! in RAM with the rest left to the SSD. The GPU tier (dense weights and hot experts in VRAM) comes next.

mod cpu_dot;
mod model;
#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
mod graph;
mod pack;

use std::io::{BufRead, Write};
use std::sync::mpsc;
use std::time::Instant;

fn out(line: &str) {
    let mut o = std::io::stdout().lock();
    let _ = writeln!(o, "{line}");
    let _ = o.flush();
}

/// xorshift64*: a small seeded generator for sampling
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

struct Sampling {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    rng: Rng,
}

fn sample(logits: &[f32], s: &mut Sampling) -> u32 {
    let argmax = || logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0);
    if s.temperature <= 0.0 {
        return argmax();
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let k = if s.top_k == 0 { 64 } else { s.top_k.min(64) };
    idx.select_nth_unstable_by(k, |a, b| logits[*b].total_cmp(&logits[*a]));
    idx.truncate(k);
    idx.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
    let m = logits[idx[0]];
    let mut p: Vec<f32> = idx.iter().map(|&i| ((logits[i] - m) / s.temperature).exp()).collect();
    let sum: f32 = p.iter().sum();
    p.iter_mut().for_each(|v| *v /= sum);
    if s.min_p > 0.0 {
        let cut = p[0] * s.min_p;
        let n = p.iter().take_while(|v| **v >= cut).count().max(1);
        p.truncate(n);
    }
    if s.top_p < 1.0 {
        let mut acc = 0.0;
        let mut n = 0;
        for v in &p {
            acc += v;
            n += 1;
            if acc >= s.top_p {
                break;
            }
        }
        p.truncate(n.max(1));
    }
    let total: f32 = p.iter().sum();
    let mut r = s.rng.next_f32() * total;
    for (j, v) in p.iter().enumerate() {
        r -= v;
        if r <= 0.0 {
            return idx[j] as u32;
        }
    }
    idx[p.len() - 1] as u32
}

/// MemAvailable from /proc/meminfo, in bytes (0 when unknown).
fn mem_available() -> u64 {
    std::fs::read_to_string("/proc/meminfo").ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()))
        .map(|kb| kb * 1024).unwrap_or(0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    // the GPUs this engine can use here, as JSON (am2strata's Settings and --devices take these names)
    if args.iter().any(|a| a == "--list-devices") {
        #[cfg(feature = "gpu")]
        let list: Vec<serde_json::Value> = strata_gpu::list().into_iter()
            .map(|d| serde_json::json!({"name": d.name, "backend": d.backend, "detail": d.detail})).collect();
        #[cfg(not(feature = "gpu"))]
        let list: Vec<serde_json::Value> = vec![serde_json::json!({"name": "cpu", "backend": "cpu", "detail": "built without the gpu feature"})];
        println!("{}", serde_json::Value::Array(list));
        return;
    }
    let Some(model_path) = arg("--model") else {
        eprintln!("usage: am2strata-hybrid --serve --model <shard 1>.gguf [--max-context N] [--threads N]");
        std::process::exit(2);
    };
    let max_ctx: usize = arg("--max-context").and_then(|v| v.parse().ok()).unwrap_or(32768);
    if let Some(t) = arg("--threads").and_then(|v| v.parse::<usize>().ok()) {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(t).build_global();
    }
    let t0 = Instant::now();
    let fail = |e: String| -> ! {
        out(&format!("ERR {e}"));
        eprintln!("{e}");
        std::process::exit(1);
    };
    let pk = arg("--pack").map(|d| pack::Pack::open(std::path::Path::new(&d)).unwrap_or_else(|e| fail(e)));
    #[allow(unused_mut)]
    let (mut m, from_pack) = model::Model::load(std::path::Path::new(&model_path), pk.as_ref()).unwrap_or_else(|e| fail(e));
    let n_layers = m.layers.len();
    // a profile not written yet (the first run of a config that also saves to it) is learned from scratch
    let profile = arg("--expert-profile").filter(|p| std::path::Path::new(p).exists())
        .map(|p| pack::Profile::read(std::path::Path::new(&p), n_layers, m.profile_experts).unwrap_or_else(|e| fail(e)));
    let save = arg("--expert-profile-save").map(std::path::PathBuf::from);
    let mut placement = "cpu".to_string();
    let res = if args.iter().any(|a| a == "--resident-experts") {
        // the budget: --expert-ram-gib, else what the expert tensors need, capped at 3/4 of the RAM available now
        let all: u64 = m.moes().map(|x| x.up_exps.bytes() + x.down_exps.bytes() + x.gate_exps.as_ref().map(|g| g.bytes()).unwrap_or(0)).sum();
        let budget = arg("--expert-ram-gib").and_then(|v| v.parse::<f64>().ok()).map(|g| (g * (1u64 << 30) as f64) as u64)
            .unwrap_or_else(|| all.min(mem_available() / 4 * 3));
        let r = pack::place(&m, profile.as_ref(), budget);
        placement = format!("cpu resident_experts={} resident_gib={:.2} of_gib={:.2} pinned={}", r.n_resident,
                            r.bytes as f64 / (1u64 << 30) as f64, all as f64 / (1u64 << 30) as f64,
                            if r.locked { "mlock" } else { "advice" });
        Some(r)
    } else {
        None
    };
    // weights on GPUs: --devices cuda:0=12,hip:0=6 (GiB each), in that order of preference
    #[cfg(feature = "gpu")]
    if let Some(spec) = arg("--devices") {
        let devs = gpu::parse(&spec).unwrap_or_else(|e| fail(e));
        let summary = gpu::place(&mut m, &devs, profile.as_ref()).unwrap_or_else(|e| fail(e));
        placement = format!("{placement} {summary}");
    }
    #[cfg(not(feature = "gpu"))]
    if arg("--devices").is_some() {
        fail("--devices: this am2strata-hybrid was built without the gpu feature".into());
    }
    // --graph: a whole token on the device (dense attention models whose matrices are all on one device)
    #[cfg(feature = "gpu")]
    let graph = if args.iter().any(|a| a == "--graph") {
        let g = graph::Graph::new(&m, max_ctx).unwrap_or_else(|e| fail(e));
        placement = format!("{placement} graph=1");
        Some(g)
    } else {
        None
    };
    let forward = |st: &mut model::State, t: u32| -> Vec<f32> {
        #[cfg(feature = "gpu")]
        if let Some(g) = &graph {
            return g.forward(&m, st, t);
        }
        m.forward(st, t)
    };
    let packed = match &pk {
        Some(p) => format!(" pack={} pack_floats={from_pack} pack_native={}", p.dir.display(), p.n_native()),
        None => String::new(),
    };
    let prof = match &profile {
        Some(p) => format!(" expert_profile={}", p.ranked.len()),
        None => String::new(),
    };
    out(&format!("INFO engine=am2strata-hybrid arch={} context={max_ctx} {} placement={placement}{packed}{prof} load_ms={}", m.arch,
                 m.summary(), t0.elapsed().as_millis()));
    let mut last_heat: Vec<u32> = Vec::new();
    out("READY");

    // commands come on their own thread, so a STOP is seen while a request runs
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut st = m.new_state();
    let mut history: Vec<u32> = Vec::new();
    while let Ok(line) = rx.recv() {
        let line = line.trim().to_string();
        if line == "QUIT" {
            break;
        }
        if line == "STOP" || line.is_empty() {
            continue;
        }
        let Some(rest) = line.strip_prefix("GEN ") else {
            out(&format!("ERR unsupported command: {}", line.split_whitespace().next().unwrap_or("")));
            continue;
        };
        let parts: Vec<&str> = rest.split_whitespace().collect();
        if parts.len() < 2 {
            out("ERR GEN needs: GEN <max_new> [key=value ...] <id,id,...>");
            continue;
        }
        let max_new: usize = parts.first().and_then(|v| v.parse().ok()).unwrap_or(256);
        let ids: Vec<u32> = parts.last().map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect()).unwrap_or_default();
        let mut s = Sampling { temperature: 0.0, top_k: 64, top_p: 1.0, min_p: 0.0, rng: Rng(0x9E3779B97F4A7C15) };
        for kv in &parts[1..parts.len().saturating_sub(1)] {
            if let Some((k, v)) = kv.split_once('=') {
                match k {
                    "temperature" => s.temperature = v.parse().unwrap_or(0.0),
                    "top_k" => s.top_k = v.parse().unwrap_or(64),
                    "top_p" => s.top_p = v.parse().unwrap_or(1.0),
                    "min_p" => s.min_p = v.parse().unwrap_or(0.0),
                    "seed" => s.rng = Rng(v.parse::<u64>().unwrap_or(1).max(1)),
                    _ => {}
                }
            }
        }
        if ids.is_empty() || ids.len() + max_new + 8 > max_ctx {
            out(&format!("ERR the prompt ({} tokens) and {max_new} new tokens do not fit the context of {max_ctx}", ids.len()));
            continue;
        }
        // a conversation that goes on: the recurrent state cannot be rewound, only extended
        let reused = if ids.len() > history.len() && ids[..history.len()] == history[..] { history.len() } else { 0 };
        if reused == 0 {
            st = m.new_state();
            history.clear();
        } else {
            out(&format!("RESUME {reused}"));
        }
        let tp = Instant::now();
        let mut logits = Vec::new();
        let todo = &ids[reused..];
        for (i, &t) in todo.iter().enumerate() {
            logits = forward(&mut st, t);
            history.push(t);
            if (i + 1) % 16 == 0 || i + 1 == todo.len() {
                out(&format!("PP {} {} {}", reused + i + 1, ids.len(), tp.elapsed().as_millis()));
            }
        }
        let prompt_ms = tp.elapsed().as_millis();
        let force: Option<Vec<u32>> = std::env::var("AM2_DEBUG_FORCE").ok().map(|v| v.split(',').filter_map(|t| t.parse().ok()).collect());
        let td = Instant::now();
        let (mut n, mut finish) = (0usize, "length");
        while n < max_new {
            if let Ok(cmd) = rx.try_recv() {
                if cmd.trim() == "STOP" {
                    finish = "stop";
                    break;
                }
            }
            if logits.iter().any(|v| !v.is_finite()) {
                finish = "nonfinite";
                break;
            }
            if std::env::var_os("AM2_DEBUG_TOP").is_some() {
                // the five best logits, as tools/reference-llamacpp/oracle prints them (for parity checks)
                let mut ix: Vec<usize> = (0..logits.len()).collect();
                ix.select_nth_unstable_by(5, |a, b| logits[*b].total_cmp(&logits[*a]));
                ix.truncate(5);
                ix.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
                let top: Vec<String> = ix.iter().map(|&i| format!("[{i},{:.4}]", logits[i])).collect();
                eprintln!("{{\"step\":{n},\"top\":[{}]}}", top.join(","));
            }
            let mut t = sample(&logits, &mut s);
            // parity tests (teacher forcing): continue with the reference's tokens, so that a near-tie both engines
            // resolve differently does not change everything after it
            if let Some(forced) = force.as_ref().and_then(|f| f.get(n)) {
                t = *forced;
            }
            out(&format!("T {t}"));
            n += 1;
            history.push(t);
            if m.stops.contains(&t) {
                finish = "eos";
                break;
            }
            logits = forward(&mut st, t);
        }
        // the expert loadings: how much of this request's routing the resident experts served, and the learned profile
        if let Some(r) = &res {
            let (hit, all) = pack::hit_rate(&m, r, &mut last_heat);
            if all > 0 {
                out(&format!("INFO experts_routed={all} resident_hit={:.3}", hit as f64 / all as f64));
            }
        }
        if let Some(p) = &save {
            if let Err(e) = pack::Profile::learned(&m, profile.as_ref()).write(p) {
                out(&format!("INFO expert_profile_save_failed={}", e.replace(' ', "_")));
            }
        }
        out(&format!("DONE {n} {} {prompt_ms} {} {finish} 0 0 0 {reused}", ids.len(), td.elapsed().as_millis()));
    }
}
