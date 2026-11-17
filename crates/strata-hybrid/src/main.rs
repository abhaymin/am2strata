//! am2strata-hybrid: an engine in Rust for the architectures in model.rs (Nemotron-H, Mistral / Mixtral, GLM-4.7-Flash,
//! the DeepSeek family and Qwen3.5/3.6 MoE), speaking the strata
//! engine protocol (docs/PROTOCOL.md) so am2Strata's server drives it like `strata` and `strata-poly`:
//!   am2strata-hybrid --serve --model <shard 1>.gguf [--max-context N] [--threads N]
//!       [--pack DIR] [--expert-profile FILE] [--expert-profile-save FILE] [--resident-experts [--expert-ram-gib N]]
//!       [--remote-experts HOST:PORT --remote-pin PIN --identity node-tls.json]     (another machine's memory, remote.rs)
//!   am2strata-hybrid --serve-experts ADDR --model M --identity node-tls.json --allow-pin PIN[,PIN] [--expert-ram-gib N]
//!   am2strata-hybrid --make-identity node-tls.json [--name NAME]      (prints the new identity's pin)
//! The computation runs on the CPU (correctness first, compared with llama.cpp by tools/reference-llamacpp); the
//! placement is strata's (pack.rs): a pack's float tensors, the expert loadings (profile), and the hottest experts kept
//! in RAM with the rest left to the SSD. The GPU tier (dense weights and hot experts in VRAM) comes next.

mod cpu_dot;
mod cpu_q8;
mod model;
#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
mod graph;
mod pack;
mod remote;

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

/// Prompt lookup: the tokens that followed the most recent earlier occurrence of the history's last 4, 3 or 2 tokens
/// (longest first), at most `k` of them. Code edits and quoted text repeat the context, so the guesses are often right;
/// the model checks them all in one pass and keeps the ones it would have chosen itself.
fn lookup_draft(history: &[u32], k: usize) -> Vec<u32> {
    let n = history.len();
    for ng in [4usize, 3, 2] {
        if k == 0 || n <= ng {
            continue;
        }
        let key = &history[n - ng..];
        // the latest earlier match whose continuation lies before the key
        for j in (0..n - ng).rev() {
            if &history[j..j + ng] == key {
                let from = j + ng;
                let to = (from + k).min(n);
                if from < to {
                    return history[from..to].to_vec();
                }
            }
        }
    }
    Vec::new()
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

/// Free RAM in bytes (0 when unknown), on Linux, macOS and Windows (strata_pack::sysmem).
fn mem_available() -> u64 {
    let sys = strata_pack::sysmem::available_mib() * 1024 * 1024;
    cgroup_room().map_or(sys, |c| c.min(sys))
}

/// Room left under this process's cgroup v2 memory limit (a container, or systemd's MemoryMax), when it has one.
/// Linux cgroup v2 only: None on cgroup v1, macOS and Windows, where the system's free memory is used instead.
fn cgroup_room() -> Option<u64> {
    let path = std::fs::read_to_string("/proc/self/cgroup").ok()?.lines().find_map(|l| l.strip_prefix("0::").map(str::to_string))?;
    let dir = std::path::Path::new("/sys/fs/cgroup").join(path.trim_start_matches('/'));
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok().and_then(|v| v.trim().parse::<u64>().ok());
    let (max, cur) = (read("memory.max")?, read("memory.current").unwrap_or(0));
    Some(max.saturating_sub(cur))
}

fn main() {
    // a device thread that panics leaves wrong numbers behind it: the engine stops, and the server says so
    let report = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        report(info);
        std::process::exit(101);
    }));
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
    // a node identity in the cluster's format (node-tls.json), for machines without am2Strata's server and for tests
    if let Some(p) = arg("--make-identity") {
        match remote::Identity::make(std::path::Path::new(&p), &arg("--name").unwrap_or_else(|| "node".into())) {
            Ok(id) => { println!("{}", id.pin); return; }
            Err(e) => { eprintln!("--make-identity: {e}"); std::process::exit(1); }
        }
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
    // the expert host: this machine lends its memory to a paired master (remote.rs); it serves until stopped
    if let Some(addr) = arg("--serve-experts") {
        let id = remote::Identity::load(std::path::Path::new(&arg("--identity").unwrap_or_else(|| fail("--serve-experts needs --identity node-tls.json".into()))))
            .unwrap_or_else(|e| fail(e));
        let allow: Vec<String> = arg("--allow-pin").unwrap_or_default().split(',').map(|p| p.trim().to_lowercase()).filter(|p| p.len() == 64).collect();
        if allow.is_empty() {
            fail("--serve-experts needs --allow-pin: the pins of the masters it serves (64 hex digits each)".into());
        }
        let budget = arg("--expert-ram-gib").and_then(|v| v.parse::<f64>().ok()).map(|g| (g * (1u64 << 30) as f64) as u64)
            .unwrap_or_else(|| mem_available() / 4 * 3);
        let model_id = remote::fingerprint(std::path::Path::new(&model_path)).unwrap_or_else(|e| fail(e));
        let mm: &'static model::Model = Box::leak(Box::new(m));
        remote::serve(mm, model_id, &addr, id, allow, budget).unwrap_or_else(|e| fail(e));
        return;
    }
    // a profile not written yet (the first run of a config that also saves to it) is learned from scratch
    let profile = arg("--expert-profile").filter(|p| std::path::Path::new(p).exists())
        .map(|p| pack::Profile::read(std::path::Path::new(&p), n_layers, m.profile_experts).unwrap_or_else(|e| fail(e)));
    let save = arg("--expert-profile-save").map(std::path::PathBuf::from);
    let mut placement = "cpu".to_string();
    let res = if args.iter().any(|a| a == "--resident-experts") {
        // the budget: --expert-ram-gib, else what the expert tensors need, capped at 3/4 of the RAM available now
        let all: u64 = m.moes().map(|x| x.up_exps.bytes() + x.down_exps.bytes() + x.gate_exps.as_ref().map(|g| g.bytes()).unwrap_or(0)).sum();
        // the matrices every token uses stay in RAM too: the experts get what is left of 3/4 of the free RAM
        let always: u64 = m.matrices_mut().0.iter().map(|q| q.data.len() as u64).sum();
        let budget = arg("--expert-ram-gib").and_then(|v| v.parse::<f64>().ok()).map(|g| (g * (1u64 << 30) as f64) as u64)
            .unwrap_or_else(|| all.min((mem_available() / 4 * 3).saturating_sub(always)));
        let r = pack::place(&m, profile.as_ref(), budget);
        // without mlock, advice does not keep them: the hot experts are copied into memory of our own
        // (AM2_RESIDENT_COPY=0 keeps the old advice-only behaviour, to compare)
        let how = if r.locked {
            match pack::keep_always(&mut m) {
                "mlock" => "mlock",
                _ => "mlock always=copy",
            }
        } else if std::env::var("AM2_RESIDENT_COPY").ok().as_deref() != Some("0") {
            pack::copy_resident(&mut m, &r);
            "copy"
        } else {
            "advice"
        };
        placement = format!("cpu resident_experts={} resident_gib={:.2} of_gib={:.2} always_gib={:.2} pinned={how}", r.n_resident,
                            r.bytes as f64 / (1u64 << 30) as f64, all as f64 / (1u64 << 30) as f64, always as f64 / (1u64 << 30) as f64);
        Some(r)
    } else {
        None
    };
    // the experts each step chose, asked for at once: on with residency (the cold ones are on disk), AM2_PREFETCH=0/1
    let prefetch = match std::env::var("AM2_PREFETCH").ok().as_deref() { Some("0") => false, Some(_) => true, None => res.is_some() };
    m.prefetch.store(prefetch, std::sync::atomic::Ordering::Relaxed);
    if prefetch {
        placement = format!("{placement} prefetch=on");
    }
    // weights on GPUs: --devices cuda:0=12,hip:0=6 (GiB each), in that order of preference
    #[cfg(feature = "gpu")]
    if let Some(spec) = arg("--devices") {
        let devs = gpu::parse(&spec).unwrap_or_else(|e| fail(e));
        let (summary, dense_on_device, experts_on_device) = gpu::place(&mut m, &devs, profile.as_ref()).unwrap_or_else(|e| fail(e));
        placement = format!("{placement} {summary}");
        // findings item 2: --devices asked the GPU to take weights but nothing landed on it (the device opened and its
        // self-check passed, yet no matrix fit or a kernel was absent). A chat then only keep-alives for minutes. So
        // when nothing went on any requested device, fall back to the CPU and say so in the log line the server reads.
        if gpu::nothing_placed(dense_on_device, experts_on_device) {
            eprintln!("no matrices were placed on any requested device ({summary}); running the model on the CPU");
            placement = format!("{placement} ran_on_cpu");
        } else if dense_on_device == 0 {
            eprintln!("no dense or attention matrix went on a device ({summary}); those compute on the CPU, experts where placed stayed on device");
        }
    }
    #[cfg(not(feature = "gpu"))]
    if arg("--devices").is_some() {
        fail("--devices: this am2strata-hybrid was built without the gpu feature".into());
    }
    // another machine's memory, the third tier: after this machine's GPUs (placed just above) and its RAM, the next-hottest
    // experts that would otherwise come from the disk; an expert on a GPU or in RAM here is never sent away
    if let Some(addr) = arg("--remote-experts") {
        let Some(r) = &res else { fail("--remote-experts needs --resident-experts: this machine's own RAM tier is placed first".into()) };
        let pin = arg("--remote-pin").map(|p| p.trim().to_lowercase()).filter(|p| p.len() == 64)
            .unwrap_or_else(|| fail("--remote-experts needs --remote-pin: the expert host's pin (64 hex digits)".into()));
        let id = remote::Identity::load(std::path::Path::new(&arg("--identity").unwrap_or_else(|| fail("--remote-experts needs --identity node-tls.json".into()))))
            .unwrap_or_else(|e| fail(e));
        let model_id = remote::fingerprint(std::path::Path::new(&model_path)).unwrap_or_else(|e| fail(e));
        let ne = m.profile_experts;
        match remote::Remote::connect(&addr, &pin, &id, &model_id, n_layers, ne) {
            Ok((mut rm, free)) => {
                let sizes: std::collections::HashMap<(usize, usize), u64> = m.moes().flat_map(|x| (0..x.n_expert).map(move |e|
                    ((x.layer, e), (x.up_exps.0[e].data.len() + x.down_exps.0[e].data.len() + x.gate_exps.as_ref().map_or(0, |g| g.0[e].data.len())) as u64))).collect();
                // the profile's order, then layer by layer; only experts not already in this machine's RAM
                let mut order: Vec<(usize, usize)> = profile.as_ref().map(|p| p.ranked.iter().map(|&(l, e)| (l as usize, e as usize)).collect()).unwrap_or_default();
                let mut seen = std::collections::HashSet::new();
                order.retain(|k| sizes.contains_key(k) && seen.insert(*k));
                let mut rest: Vec<(usize, usize)> = sizes.keys().copied().filter(|k| !seen.contains(k)).collect();
                rest.sort();
                order.extend(rest);
                let (mut take, mut bytes) = (Vec::new(), 0u64);
                for k in order {
                    if r.resident.get(k.0 * ne + k.1).copied().unwrap_or(false) || m.expert_on_device(k.0, k.1) {
                        continue;
                    }
                    let b = sizes[&k];
                    if bytes + b > free {
                        break;
                    }
                    bytes += b;
                    take.push(k);
                }
                rm.assign(&m, &take).unwrap_or_else(|e| fail(format!("--remote-experts {addr}: {e}")));
                placement = format!("{placement} remote={addr} remote_experts={} remote_gib={:.2}", rm.n_held, rm.bytes as f64 / (1u64 << 30) as f64);
                m.remote = Some(rm);
            }
            // no worker: the model still runs, its cold experts from this machine's disk
            Err(e) => {
                eprintln!("remote experts: {e}; running without them");
                placement = format!("{placement} remote_unavailable");
            }
        }
    }
    // a whole token on the device (dense attention models whose matrices are all on one device): asked with --graph, and
    // taken by itself whenever --devices placed such a model, because the matrix-by-matrix path syncs with the device
    // after every product (about 6x slower on a Mac M4's Metal); --no-graph keeps that path
    #[cfg(feature = "gpu")]
    let graph = if args.iter().any(|a| a == "--graph") {
        let g = graph::Graph::new(&m, max_ctx).unwrap_or_else(|e| fail(e));
        placement = format!("{placement} graph=1");
        Some(g)
    } else if args.iter().any(|a| a == "--devices") && !args.iter().any(|a| a == "--no-graph") && std::env::var_os("AM2_DEBUG_LAYERS").is_some() {
        // the per-layer checksums live in Model::forward; the graph would print none of them (Kilo on #76)
        eprintln!("AM2_DEBUG_LAYERS: matrix by matrix on the device, not the graph, so every layer's checksum is printed");
        None
    } else if args.iter().any(|a| a == "--devices") && !args.iter().any(|a| a == "--no-graph") {
        match graph::Graph::new(&m, max_ctx) {
            Ok(g) => { placement = format!("{placement} graph=auto"); Some(g) }
            Err(why) => { eprintln!("matrix by matrix on the device ({why})"); None }
        }
    } else {
        None
    };
    #[cfg(feature = "gpu")]
    if let (Some(g), Some(_)) = (&graph, std::env::var_os("AM2_GPU_BENCH")) {
        g.bench(&m);
    }
    let forward = |st: &mut model::State, t: u32| -> Vec<f32> {
        #[cfg(feature = "gpu")]
        if let Some(g) = &graph {
            return g.forward(&m, st, t);
        }
        m.forward(st, t)
    };
    // prompt tokens per chunk: 32 for models whose layers forward_batch handles, on the CPU path, and on a device graph
    // whose device has the chunk kernels (native Metal: Graph::batches; 128 tokens there); AM2_PROMPT_BATCH=1 reads the
    // prompt one token at a time (for comparisons)
    #[cfg(feature = "gpu")]
    let (on_graph, graph_batches) = (graph.is_some(), graph.as_ref().is_some_and(|g| g.batches()));
    #[cfg(not(feature = "gpu"))]
    let (on_graph, graph_batches) = (false, false);
    let batch: usize = if (m.batchable() && !on_graph) || graph_batches {
        std::env::var("AM2_PROMPT_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(if graph_batches { 128 } else { 32 }).max(1)
    } else {
        1
    };
    let forward_batch = |st: &mut model::State, ts: &[u32]| -> Vec<f32> {
        #[cfg(feature = "gpu")]
        if let Some(g) = graph.as_ref().filter(|g| g.batches()) {
            return g.forward_batch(&m, st, ts);
        }
        m.forward_batch(st, ts)
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
        // drafted tokens per step from prompt lookup (0: off); the CPU path only, where a chunk's logits come back
        let mut lookup: usize = std::env::var("AM2_LOOKUP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        for kv in &parts[1..parts.len().saturating_sub(1)] {
            if let Some((k, v)) = kv.split_once('=') {
                match k {
                    "temperature" => s.temperature = v.parse().unwrap_or(0.0),
                    "top_k" => s.top_k = v.parse().unwrap_or(64),
                    "top_p" => s.top_p = v.parse().unwrap_or(1.0),
                    "min_p" => s.min_p = v.parse().unwrap_or(0.0),
                    "seed" => s.rng = Rng(v.parse::<u64>().unwrap_or(1).max(1)),
                    "lookup" => lookup = v.parse().unwrap_or(0),
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
        if batch > 1 {
            // the prompt in chunks: each weight row read once per chunk, the same values as token by token
            let mut i = 0;
            while i < todo.len() {
                let end = (i + batch).min(todo.len());
                logits = forward_batch(&mut st, &todo[i..end]);
                history.extend_from_slice(&todo[i..end]);
                i = end;
                out(&format!("PP {} {} {}", reused + i, ids.len(), tp.elapsed().as_millis()));
            }
        } else {
            for (i, &t) in todo.iter().enumerate() {
                logits = forward(&mut st, t);
                history.push(t);
                if (i + 1) % 16 == 0 || i + 1 == todo.len() {
                    out(&format!("PP {} {} {}", reused + i + 1, ids.len(), tp.elapsed().as_millis()));
                }
            }
        }
        let prompt_ms = tp.elapsed().as_millis();
        let force: Option<Vec<u32>> = std::env::var("AM2_DEBUG_FORCE").ok().map(|v| v.split(',').filter_map(|t| t.parse().ok()).collect());
        let td = Instant::now();
        let (mut n, mut finish) = (0usize, "length");
        if on_graph || batch <= 1 || force.is_some() {
            lookup = 0;
        }
        // a token already sampled: a rejected draft's replacement, the model's own choice at that position
        let mut next: Option<u32> = None;
        let (mut drafted, mut accepted, mut checks) = (0usize, 0usize, 0usize);
        // how many guesses to check now: halved when most of a check's are wrong (a wrong guess costs its share of
        // the pass and a re-run), doubled back when all are right; none right and drafting rests for 32 tokens
        let (mut want, mut rest) = (lookup, 0usize);
        while n < max_new {
            if let Ok(cmd) = rx.try_recv() {
                if cmd.trim() == "STOP" {
                    finish = "stop";
                    break;
                }
            }
            if next.is_none() && logits.iter().any(|v| !v.is_finite()) {
                finish = "nonfinite";
                break;
            }
            if next.is_none() && std::env::var_os("AM2_DEBUG_TOP").is_some() {
                // the five best logits, as tools/reference-llamacpp/oracle prints them (for parity checks)
                let mut ix: Vec<usize> = (0..logits.len()).collect();
                ix.select_nth_unstable_by(5, |a, b| logits[*b].total_cmp(&logits[*a]));
                ix.truncate(5);
                ix.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
                let top: Vec<String> = ix.iter().map(|&i| format!("[{i},{:.4}]", logits[i])).collect();
                eprintln!("{{\"step\":{n},\"top\":[{}]}}", top.join(","));
            }
            let mut t = match next.take() {
                Some(t) => t,
                None => sample(&logits, &mut s),
            };
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
            rest = rest.saturating_sub(1);
            if want == 0 && rest == 0 {
                want = 1;
            }
            let draft = if lookup > 0 && want > 0 && n < max_new { lookup_draft(&history, want.min(max_new - n)) } else { Vec::new() };
            if draft.is_empty() {
                logits = forward(&mut st, t);
                continue;
            }
            // t and the guesses in one pass; position i's logits are the model's choice after draft[..i]. Each is
            // sampled once, in order, as without drafting, so greedy and seeded sampling give the same tokens.
            let ck = st.checkpoint();
            let mut seq = vec![t];
            seq.extend_from_slice(&draft);
            let all = m.forward_batch_all(&mut st, &seq);
            checks += 1;
            drafted += draft.len();
            let mut i = 0;
            let mut stopped = false;
            while i < draft.len() && n < max_new {
                let c = sample(&all[i], &mut s);
                if c != draft[i] {
                    next = Some(c);
                    break;
                }
                out(&format!("T {c}"));
                n += 1;
                history.push(c);
                accepted += 1;
                i += 1;
                if m.stops.contains(&c) {
                    finish = "eos";
                    stopped = true;
                    break;
                }
            }
            if i == draft.len() {
                want = (want * 2).min(lookup);
            } else if i == 0 {
                // not one guess right: the pass and the re-run were wasted, so rest
                want = 0;
                rest = 32;
            } else if i * 2 < draft.len() {
                want /= 2;
            }
            if i < draft.len() {
                // the state saw guesses that were not kept: back to before t, then t and the kept ones again
                st.rollback(ck);
                m.forward_batch(&mut st, &seq[..=i]);
            } else {
                logits = all[i].clone();
            }
            if stopped {
                break;
            }
        }
        if lookup > 0 {
            out(&format!("INFO lookup_checks={checks} lookup_drafted={drafted} lookup_accepted={accepted}"));
        }
        // the expert loadings: how much of this request's routing the resident experts served, and the learned profile
        if let Some(r) = &res {
            let (hit, all) = pack::hit_rate(&m, r, &mut last_heat);
            if all > 0 {
                out(&format!("INFO experts_routed={all} resident_hit={:.3}", hit as f64 / all as f64));
            }
        }
        if let Some(rm) = &m.remote {
            out(&format!("INFO {}", rm.take_stats()));
        }
        if let Some(p) = &save {
            if let Err(e) = pack::Profile::learned(&m, profile.as_ref()).write(p) {
                out(&format!("INFO expert_profile_save_failed={}", e.replace(' ', "_")));
            }
        }
        if model::times::on() {
            out(&format!("INFO times_ms={}", model::times::take()));
        }
        // fields 6 and 7: drafted tokens kept and offered (strata-protocol's drafts_accepted / drafts_offered)
        out(&format!("DONE {n} {} {prompt_ms} {} {finish} {accepted} {drafted} 0 {reused}", ids.len(), td.elapsed().as_millis()));
    }
}
