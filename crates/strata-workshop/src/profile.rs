//! `profile` and `autopilot`: what a training run needs before it starts, and settings that fit this machine.
//!
//! The idea is Soup's (`soup profile`, `soup autopilot`: estimate memory and time before training, and pick the
//! hyper-parameters from the model, the data and the hardware); the arithmetic here is this trainer's own, counted
//! from what [`crate::model`] and [`crate::finetune`] really keep in memory.
//!
//! **Memory** (bytes, at the sequence-length cap — the worst conversation):
//! - *weights*: the frozen base as held — matrices in the base dtype, norms and biases in f32; with the activation
//!   cache only the trained blocks and the output head stay;
//! - *adapters*: A and B, f32; *optimiser*: AdamW's two moments per adapter value; *gradients*: the step's running
//!   sum and the pass in flight (two more copies);
//! - *activations*: every tensor the trained blocks keep for the backward pass, counted operation by operation from
//!   the forward pass ([`graph_floats_per_token`]), doubled for the gradients that flow back through them; the output
//!   head's logits and their log-softmax on every position (the cap: only scored positions are computed); and one
//!   block's worth for the frozen part, which keeps nothing;
//! - *widening*: with a 16-bit base, the largest matrix as f32 for the one product that uses it;
//! - *cache fill*: before training, the activation cache is filled holding the embeddings, then one frozen block at a
//!   time; the peak is the larger of that and the training itself.
//!
//! **Time** is rougher: floating-point operations (two per weight per token forward, twice that again backward
//! through the trained part, the frozen part forward only, and only once per conversation with the cache) over a
//! throughput in GFLOP/s — measured on this CPU by a short matrix product ([`measure_cpu_gflops`]), assumed for a GPU.

use std::time::Instant;

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use serde_json::{json, Value};

use crate::model::{BaseDtype, Family, LoraSpec, ModelConfig, Target};

/// Share of a raw matrix-product benchmark that training really reaches on the CPU (the rest goes to memory traffic
/// and small operations). Measured with an f32 or bf16 base on a 12-block, 27 M-parameter model (docs/DEPLOYMENT.md:
/// 12.8 s for an estimate of 15 s at this value); an f16 base ran about twice as slow there (candle widens f16 slowly).
pub const CPU_TRAINING_EFFICIENCY: f64 = 0.6;
/// A 16 GB-class GPU's training throughput in GFLOP/s. **Assumed, not measured**: no GPU run is part of this trainer yet.
pub const ASSUMED_GPU_GFLOPS: f64 = 1500.0;
/// Room left for the process itself, the tokenizer and the allocator on top of the estimate.
pub const HEADROOM_BYTES: u64 = 512 << 20;

/// What a run is, as far as its cost goes.
#[derive(Debug, Clone)]
pub struct ProfileInput {
    pub cfg: ModelConfig,
    pub lora: LoraSpec,
    pub base_dtype: BaseDtype,
    pub max_len: usize,
    pub accumulation: usize,
    /// the activation cache is on (it needs `lora.train_layers` below the block count)
    pub cache: bool,
    pub examples: usize,
    pub epochs: f64,
    /// mean tokens per conversation, for the time estimate (the memory estimate uses `max_len`)
    pub avg_tokens: usize,
}

/// The estimate, part by part.
#[derive(Debug, Clone, PartialEq)]
pub struct Estimate {
    pub total_params: u64,
    pub trainable_params: u64,
    pub trained_blocks: usize,
    pub weights_bytes: u64,
    pub adapter_bytes: u64,
    pub optimizer_bytes: u64,
    pub gradient_bytes: u64,
    pub activation_bytes: u64,
    pub widening_bytes: u64,
    /// the peak while the activation cache is filled, one frozen block at a time (0 without the cache)
    pub fill_bytes: u64,
    /// the peak: the sum of the parts above, or the cache fill when that is larger
    pub total_bytes: u64,
    /// the activation cache on disk (0 without it)
    pub cache_disk_bytes: u64,
    pub steps: usize,
    /// floating-point operations for the whole run
    pub flops: f64,
}

/// f32 values per token that one trained block keeps for the backward pass, counted from `model.rs`'s forward:
/// two norms (3·H each), the q/k/v products with their head layout and rotation (11.5·Q + 8.5·KV where Q = heads ×
/// head_dim and KV = kv_heads × head_dim), the attention scores and their softmax as one node (2·heads·seq), the output product
/// and both residuals (3·H), the MLP (4·I + H), and per adapter its rank-r product and three output-sized tensors.
pub fn graph_floats_per_token(cfg: &ModelConfig, lora: &LoraSpec, seq: usize) -> f64 {
    let (h, i) = (cfg.hidden_size as f64, cfg.intermediate_size as f64);
    let (q, kv) = ((cfg.num_attention_heads * cfg.head_dim) as f64, (cfg.num_key_value_heads * cfg.head_dim) as f64);
    let adapters: f64 = lora.targets.iter().map(|t| (lora.r + 3 * cfg.target_shape(*t).0) as f64).sum();
    10.0 * h + 11.5 * q + 8.5 * kv + 4.0 * i + 2.0 * (cfg.num_attention_heads * seq) as f64 + adapters
}

/// Bytes of the base's 1-D tensors (always f32) and matrices (base dtype), for `blocks` blocks, with or without the
/// embeddings.
fn base_bytes(cfg: &ModelConfig, dtype: BaseDtype, blocks: usize, with_embeddings: bool) -> u64 {
    let emb = (cfg.vocab_size * cfg.hidden_size) as u64;
    let block_matrices = cfg.block_params() - cfg.block_small_params();
    // the head: its own tensor, or the embeddings when tied (which then always stay)
    let mut matrices = blocks as u64 * block_matrices + emb;
    if with_embeddings && !cfg.tie_word_embeddings {
        matrices += emb;
    }
    let vectors = blocks as u64 * cfg.block_small_params() + cfg.hidden_size as u64;
    matrices * dtype.bytes() + vectors * 4
}

/// The estimate for `p`. Pure arithmetic: no file, no machine.
pub fn estimate(p: &ProfileInput) -> Estimate {
    let cfg = &p.cfg;
    let layers = cfg.num_hidden_layers;
    let first = p.lora.first_trained(layers);
    let trained = layers - first;
    let cached = p.cache && first > 0;
    let seq = p.max_len.max(2);

    let trainable = p.lora.trainable_params(cfg);
    let weights = if cached { base_bytes(cfg, p.base_dtype, trained, false) } else { base_bytes(cfg, p.base_dtype, layers, true) };
    let adapter = trainable * 4;

    let per_token = graph_floats_per_token(cfg, &p.lora, seq);
    let graph = 2.0 * 4.0 * seq as f64 * per_token * trained as f64;
    let head = 6.0 * 4.0 * (seq * cfg.vocab_size) as f64;
    let frozen = if cached || first == 0 { 0.0 } else { 4.0 * seq as f64 * graph_floats_per_token(cfg, &LoraSpec { targets: vec![], ..p.lora.clone() }, seq) };
    let activations = (graph + head + frozen) as u64;

    let largest = Target::ALL.iter().map(|t| { let (o, i) = cfg.target_shape(*t); o * i }).max().unwrap_or(0).max(cfg.vocab_size * cfg.hidden_size) as u64;
    let widening = if p.base_dtype == BaseDtype::F32 { 0 } else { largest * 4 };

    // the cache is filled with the embeddings, then one frozen block, in memory — and one conversation's activations
    let frozen_tokens = 4.0 * seq as f64 * graph_floats_per_token(cfg, &LoraSpec { targets: vec![], ..p.lora.clone() }, seq);
    let fill = if cached {
        let block = (cfg.block_params() - cfg.block_small_params()) * p.base_dtype.bytes() + cfg.block_small_params() * 4;
        block.max((cfg.vocab_size * cfg.hidden_size) as u64 * p.base_dtype.bytes()) + frozen_tokens as u64 + widening
    } else {
        0
    };

    let steps = crate::finetune::total_steps(p.examples, p.epochs, p.accumulation, None);
    // per token: 2 operations per weight used, plus the attention products (4·Q·seq)
    let tokens = p.avg_tokens.clamp(2, seq) as f64;
    let q = (cfg.num_attention_heads * cfg.head_dim) as f64;
    let block_fwd = 2.0 * (cfg.block_params() - cfg.block_small_params()) as f64 + 4.0 * q * tokens;
    let head_fwd = 2.0 * (cfg.vocab_size * cfg.hidden_size) as f64;
    let passes = p.examples as f64 * p.epochs;
    let lower_passes = if cached { p.examples as f64 } else { passes };
    let flops = tokens * (lower_passes * first as f64 * block_fwd + passes * 3.0 * (trained as f64 * block_fwd + head_fwd));

    Estimate {
        total_params: cfg.total_params(), trainable_params: trainable, trained_blocks: trained, weights_bytes: weights, adapter_bytes: adapter,
        optimizer_bytes: 2 * adapter, gradient_bytes: 2 * adapter, activation_bytes: activations, widening_bytes: widening, fill_bytes: fill,
        total_bytes: (weights + 5 * adapter + activations + widening).max(fill),
        cache_disk_bytes: if cached { (p.examples * p.avg_tokens.clamp(2, seq) * cfg.hidden_size * 4) as u64 } else { 0 },
        steps, flops,
    }
}

/// One NVIDIA GPU's memory.
#[derive(Debug, Clone, PartialEq)]
pub struct Gpu {
    pub index: u32,
    pub name: String,
    pub total_bytes: u64,
    pub free_bytes: u64,
}

/// The machine a run would use.
#[derive(Debug, Clone, PartialEq)]
pub struct Machine {
    /// 0 when the OS does not say (not Linux)
    pub ram_total: u64,
    pub ram_available: u64,
    pub cpus: usize,
    pub gpus: Vec<Gpu>,
    /// raw matrix-product throughput of this CPU (GFLOP/s)
    pub cpu_gflops: f64,
}

/// `MemTotal` and `MemAvailable` of /proc/meminfo, in bytes.
pub fn parse_meminfo(text: &str) -> (u64, u64) {
    let kb = |key: &str| text.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()).unwrap_or(0);
    (kb("MemTotal:") * 1024, kb("MemAvailable:") * 1024)
}

/// The lines of `nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits` (MiB).
pub fn parse_nvidia_smi(text: &str) -> Vec<Gpu> {
    text.lines().filter_map(|l| {
        let f: Vec<&str> = l.split(',').map(str::trim).collect();
        Some(Gpu { index: f.first()?.parse().ok()?, name: f.get(1)?.to_string(), total_bytes: f.get(2)?.parse::<u64>().ok()? << 20, free_bytes: f.get(3)?.parse::<u64>().ok()? << 20 })
    }).collect()
}

/// A short f32 matrix product on this CPU, in GFLOP/s (about 0.2 s). The first product warms the thread pool.
pub fn measure_cpu_gflops() -> f64 {
    const N: usize = 384;
    let run = || -> candle_core::Result<f64> {
        let a = Tensor::ones((N, N), candle_core::DType::F32, &Device::Cpu)?;
        let b = Tensor::ones((N, N), candle_core::DType::F32, &Device::Cpu)?;
        a.matmul(&b)?;
        let t = Instant::now();
        let mut reps = 0u32;
        while t.elapsed().as_secs_f64() < 0.15 {
            a.matmul(&b)?;
            reps += 1;
        }
        Ok(reps as f64 * 2.0 * (N * N * N) as f64 / t.elapsed().as_secs_f64() / 1e9)
    };
    run().unwrap_or(0.0).max(0.1)
}

impl Machine {
    /// This machine: /proc/meminfo for the RAM and `nvidia-smi` for the cards (the server's own two sources), and a
    /// short benchmark for the CPU. Missing sources leave their fields empty.
    pub fn detect() -> Self {
        let (ram_total, ram_available) = std::fs::read_to_string("/proc/meminfo").map(|t| parse_meminfo(&t)).unwrap_or((0, 0));
        let gpus = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"]).output()
            .map(|o| parse_nvidia_smi(&String::from_utf8_lossy(&o.stdout))).unwrap_or_default();
        Self { ram_total, ram_available, cpus: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1), gpus, cpu_gflops: measure_cpu_gflops() }
    }
}

/// Whether an estimate fits, and how long it would take.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    /// the estimate plus 10 % and [`HEADROOM_BYTES`]
    pub need_bytes: u64,
    /// `None` when the free memory is unknown
    pub fits_ram: Option<bool>,
    /// per GPU: would the same run fit in its free memory (for when GPU runs land)
    pub fits_gpu: Vec<(u32, bool)>,
    pub cpu_seconds: f64,
    pub gpu_seconds: f64,
}

pub fn need_bytes(e: &Estimate) -> u64 {
    e.total_bytes + e.total_bytes / 10 + HEADROOM_BYTES
}

pub fn cpu_seconds(e: &Estimate, cpu_gflops: f64) -> f64 {
    e.flops / (cpu_gflops.max(0.01) * CPU_TRAINING_EFFICIENCY * 1e9)
}

pub fn verdict(e: &Estimate, m: &Machine) -> Verdict {
    let need = need_bytes(e);
    Verdict {
        need_bytes: need,
        fits_ram: (m.ram_available > 0).then_some(need <= m.ram_available),
        fits_gpu: m.gpus.iter().map(|g| (g.index, need <= g.free_bytes)).collect(),
        cpu_seconds: cpu_seconds(e, m.cpu_gflops),
        gpu_seconds: e.flops / (ASSUMED_GPU_GFLOPS * 1e9),
    }
}

pub fn human_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 { format!("{:.2} GB", b / 1e9) } else if b >= 1e6 { format!("{:.1} MB", b / 1e6) } else { format!("{:.0} KB", b / 1e3) }
}

pub fn human_seconds(s: f64) -> String {
    if s >= 2.0 * 86400.0 { format!("{:.1} days", s / 86400.0) } else if s >= 5400.0 { format!("{:.1} h", s / 3600.0) } else if s >= 90.0 { format!("{:.0} min", s / 60.0) } else { format!("{s:.0} s") }
}

/// The estimate as readable lines: what is trained, each part of the memory, the time, and what fits.
pub fn report_lines(p: &ProfileInput, e: &Estimate, m: &Machine) -> Vec<String> {
    let v = verdict(e, m);
    let mut out = vec![
        format!("model: {} · {} blocks · {:.2} M parameters · base held as {}", p.cfg.family.as_str(), p.cfg.num_hidden_layers, e.total_params as f64 / 1e6, p.base_dtype.as_str()),
        format!("trained: {} adapter parameters ({:.3} % of the model) at rank {} on {} in the top {} of {} blocks",
                e.trainable_params, 100.0 * e.trainable_params as f64 / e.total_params.max(1) as f64, p.lora.r,
                p.lora.targets.iter().map(|t| t.as_str()).collect::<Vec<_>>().join(","), e.trained_blocks, p.cfg.num_hidden_layers),
        format!("memory: weights {} + adapters {} + optimiser {} + gradients {} + activations at {} tokens {}{} = {}",
                human_bytes(e.weights_bytes), human_bytes(e.adapter_bytes), human_bytes(e.optimizer_bytes), human_bytes(e.gradient_bytes), p.max_len,
                human_bytes(e.activation_bytes), if e.widening_bytes > 0 { format!(" + widening one 16-bit matrix {}", human_bytes(e.widening_bytes)) } else { String::new() },
                human_bytes(e.total_bytes)),
    ];
    if e.cache_disk_bytes > 0 {
        out.push(format!("activation cache: about {} on disk; the {} frozen blocks run once per conversation, one block in memory at a time (peak {} while filling)",
                         human_bytes(e.cache_disk_bytes), p.cfg.num_hidden_layers - e.trained_blocks, human_bytes(e.fill_bytes)));
    }
    out.push(format!("time: {} steps; about {} on this CPU ({:.0} GFLOP/s measured, {:.0} % of it reached in training); about {} on a 16 GB-class GPU (assumed {:.0} GFLOP/s, not measured — GPU runs are not part of this trainer yet)",
                     e.steps, human_seconds(v.cpu_seconds), m.cpu_gflops, 100.0 * CPU_TRAINING_EFFICIENCY, human_seconds(v.gpu_seconds), ASSUMED_GPU_GFLOPS));
    out.push(match v.fits_ram {
        Some(true) => format!("fits: needs {} with headroom, {} of RAM is free", human_bytes(v.need_bytes), human_bytes(m.ram_available)),
        Some(false) => format!("DOES NOT FIT: needs {} with headroom, {} of RAM is free", human_bytes(v.need_bytes), human_bytes(m.ram_available)),
        None => format!("needs {} with headroom; this system does not say how much RAM is free", human_bytes(v.need_bytes)),
    });
    for (g, (_, fits)) in m.gpus.iter().zip(&v.fits_gpu) {
        out.push(format!("GPU {} ({}): {} free of {} — the same run {} there", g.index, g.name, human_bytes(g.free_bytes), human_bytes(g.total_bytes),
                         if *fits { "would fit" } else { "would not fit" }));
    }
    out
}

pub fn estimate_json(p: &ProfileInput, e: &Estimate, m: &Machine) -> Value {
    let v = verdict(e, m);
    json!({
        "model": {"family": p.cfg.family.as_str(), "blocks": p.cfg.num_hidden_layers, "parameters": e.total_params},
        "trainable_parameters": e.trainable_params, "trained_blocks": e.trained_blocks,
        "memory": {"weights": e.weights_bytes, "adapters": e.adapter_bytes, "optimizer": e.optimizer_bytes, "gradients": e.gradient_bytes,
                   "activations": e.activation_bytes, "widening": e.widening_bytes, "cache_fill_peak": e.fill_bytes, "total": e.total_bytes, "needed_with_headroom": v.need_bytes},
        "cache_disk_bytes": e.cache_disk_bytes, "steps": e.steps, "flops": e.flops,
        "seconds": {"cpu": v.cpu_seconds, "gpu_assumed": v.gpu_seconds},
        "machine": {"ram_total": m.ram_total, "ram_available": m.ram_available, "cpus": m.cpus, "cpu_gflops": m.cpu_gflops,
                    "gpus": m.gpus.iter().map(|g| json!({"index": g.index, "name": g.name, "total": g.total_bytes, "free": g.free_bytes})).collect::<Vec<_>>()},
        "fits_ram": v.fits_ram, "fits_gpu": v.fits_gpu.iter().map(|(i, f)| json!({"index": i, "fits": f})).collect::<Vec<_>>(),
    })
}

// ---------------------------------------------------------------------------------------------------------------------
// autopilot
// ---------------------------------------------------------------------------------------------------------------------

/// What the autopilot is told.
#[derive(Debug, Clone)]
pub struct AutopilotInput {
    pub cfg: ModelConfig,
    /// the dtype of the model's files (a 16-bit base is held as it is stored)
    pub source_dtype: BaseDtype,
    pub examples: usize,
    /// the 95th-percentile and mean token counts of the conversations
    pub p95_tokens: usize,
    pub avg_tokens: usize,
    /// memory the run may use
    pub budget_bytes: u64,
    pub time_budget_seconds: f64,
    pub cpu_gflops: f64,
}

/// One setting the autopilot chose, and why, in one line.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub setting: &'static str,
    pub value: String,
    pub why: String,
}

/// The autopilot's plan.
#[derive(Debug, Clone)]
pub struct Plan {
    pub lora: LoraSpec,
    pub base_dtype: BaseDtype,
    pub max_len: usize,
    pub accumulation: usize,
    pub epochs: f64,
    pub lr: f64,
    pub cache: bool,
    pub choices: Vec<Choice>,
    pub estimate: Estimate,
    pub seconds: f64,
    pub fits_memory: bool,
    pub fits_time: bool,
}

/// Pick rank, trained blocks, sequence length, base dtype, accumulation, epochs and rate for a dataset, a memory
/// budget and a time budget, explaining each. The rank, epoch and rate tables follow Soup's autopilot (small data:
/// small rank, more epochs); the rest follows this trainer's own costs: hold the base in 16 bits when f32 does not
/// fit, then train fewer blocks (with the activation cache) until the run fits the memory and the time.
pub fn autopilot(a: &AutopilotInput) -> Result<Plan> {
    if a.examples == 0 {
        bail!("no conversations to train on");
    }
    let layers = a.cfg.num_hidden_layers;
    let mut choices = Vec::new();
    let mut choose = |setting: &'static str, value: String, why: String| choices.push(Choice { setting, value, why });

    let (r, lr) = if a.examples < 1000 { (8, 3e-4) } else if a.examples < 10_000 { (16, 2e-4) } else { (32, 1e-4) };
    choose("rank", r.to_string(), format!("{} conversations: {}", a.examples,
           if r == 8 { "a small rank is enough and overfits less" } else if r == 16 { "a middle rank for a middle-sized dataset" } else { "enough data to fill a larger rank" }));
    choose("alpha", (2 * r).to_string(), "twice the rank, the usual scale".into());
    choose("targets", "q_proj,v_proj".into(), "PEFT's default pair for this family: the fewest adapter parameters; add the others if the loss stops falling".into());
    choose("learning rate", format!("{lr:e}"), format!("the usual rate for rank {r}; `find-lr` measures a better one on your data"));

    let mut epochs: f64 = if a.examples < 500 { 5.0 } else if a.examples < 5000 { 3.0 } else if a.examples < 50_000 { 2.0 } else { 1.0 };
    let accumulation = (a.examples / 8).clamp(1, 16);
    choose("accumulation", accumulation.to_string(), format!("the batch is always 1 (least memory); {accumulation} conversations per step keeps at least 8 steps an epoch"));

    let wanted_len = (a.p95_tokens + a.p95_tokens.div_ceil(10)).clamp(64, a.cfg.max_position_embeddings.min(2048).max(64));
    let mut max_len = wanted_len;

    let plan_for = |dtype: BaseDtype, train_layers: usize, max_len: usize, epochs: f64| {
        let lora = LoraSpec { r, alpha: 2.0 * r as f64, dropout: 0.05, use_rslora: false, targets: vec![Target::Q, Target::V],
                              train_layers: (train_layers < layers).then_some(train_layers), seed: 42 };
        // whenever blocks are frozen: besides the time over several epochs, it is what keeps them out of memory
        let cache = train_layers < layers;
        let e = estimate(&ProfileInput { cfg: a.cfg.clone(), lora: lora.clone(), base_dtype: dtype, max_len, accumulation, cache, examples: a.examples, epochs,
                                         avg_tokens: a.avg_tokens.min(max_len) });
        (lora, cache, e)
    };
    let fits_mem = |e: &Estimate| need_bytes(e) <= a.budget_bytes;
    let fits_time = |e: &Estimate| cpu_seconds(e, a.cpu_gflops) <= a.time_budget_seconds;

    // the base dtype: f32 when the whole run fits with every block training at the wanted length, else 16 bits
    let half = if a.source_dtype == BaseDtype::F32 { BaseDtype::F16 } else { a.source_dtype };
    let dtype = if a.source_dtype == BaseDtype::F32 && fits_mem(&plan_for(BaseDtype::F32, layers, max_len, epochs).2) { BaseDtype::F32 } else { half };
    let f32_need = need_bytes(&plan_for(BaseDtype::F32, layers, max_len, epochs).2);
    choose("base dtype", dtype.as_str().into(), if dtype == BaseDtype::F32 {
        format!("the base fits in f32 ({} needed of {})", human_bytes(f32_need), human_bytes(a.budget_bytes))
    } else if a.source_dtype != BaseDtype::F32 {
        format!("the model's files are {}: holding them as stored halves the memory and loses nothing", a.source_dtype.as_str())
    } else {
        format!("f32 would need {} of the {} available: 16 bits halves the base", human_bytes(f32_need), human_bytes(a.budget_bytes))
    });

    // fewer trained blocks until memory and time fit: all, half, a quarter, ... one
    let mut candidates = vec![layers];
    while *candidates.last().unwrap_or(&1) > 1 {
        candidates.push(candidates[candidates.len() - 1] / 2);
    }
    let mut picked = None;
    for &n in &candidates {
        let (_, _, e) = plan_for(dtype, n, max_len, epochs);
        if fits_mem(&e) && fits_time(&e) {
            picked = Some(n);
            break;
        }
    }
    let mut train_layers = picked.unwrap_or(1);
    // still not: a shorter sequence (memory), then fewer epochs (time)
    if picked.is_none() {
        while max_len > 128 && !fits_mem(&plan_for(dtype, train_layers, max_len, epochs).2) {
            max_len = (max_len / 2).max(128);
        }
        while epochs > 1.0 && !fits_time(&plan_for(dtype, train_layers, max_len, epochs).2) {
            epochs -= 1.0;
        }
        // with the pressure off one of them, more blocks may fit again
        for &n in &candidates {
            let (_, _, e) = plan_for(dtype, n, max_len, epochs);
            if fits_mem(&e) && fits_time(&e) {
                train_layers = n;
                break;
            }
        }
    }
    let (lora, cache, est) = plan_for(dtype, train_layers, max_len, epochs);
    let seconds = cpu_seconds(&est, a.cpu_gflops);
    let (ok_mem, ok_time) = (fits_mem(&est), fits_time(&est));

    choose("sequence length", max_len.to_string(), if max_len == wanted_len {
        format!("the 95th-percentile conversation is {} tokens; 10 % more, within the model's {} positions", a.p95_tokens, a.cfg.max_position_embeddings)
    } else {
        format!("cut from {wanted_len}: longer sequences do not fit the memory; conversations beyond {max_len} tokens are cut there")
    });
    choose("trained blocks", if train_layers == layers { format!("all {layers}") } else { format!("top {train_layers} of {layers}") }, if train_layers == layers {
        format!("every block fits the memory and the time ({} of {})", human_seconds(seconds), human_seconds(a.time_budget_seconds))
    } else {
        let (_, _, all) = plan_for(dtype, layers, max_len, epochs);
        format!("all {layers} would need {} and {}; the top {train_layers} need {} and {}, and the blocks below run without a backward pass",
                human_bytes(need_bytes(&all)), human_seconds(cpu_seconds(&all, a.cpu_gflops)), human_bytes(need_bytes(&est)), human_seconds(seconds))
    });
    choose("activation cache", if cache { "on" } else { "off" }.into(), if cache {
        format!("the {} frozen blocks run once per conversation, not once per epoch ({} on disk), one block in memory at a time", layers - train_layers, human_bytes(est.cache_disk_bytes))
    } else {
        "every block trains: there is no frozen part to cache".into()
    });
    choose("epochs", format!("{epochs}"), format!("{} conversations{}", a.examples, if ok_time { "" } else { ": even so the run is longer than the time budget" }));

    Ok(Plan { lora, base_dtype: dtype, max_len, accumulation, epochs, lr, cache, choices, estimate: est, seconds, fits_memory: ok_mem, fits_time: ok_time })
}

/// The dtype of a model folder's weights, from the first matrix of its first file.
pub fn source_dtype(dir: &std::path::Path) -> Result<BaseDtype> {
    let files = crate::model::weight_files(dir)?;
    let shard = crate::safetensors_io::Shard::open(&files[0])?;
    let t = shard.tensors().find(|t| t.shape().len() == 2).ok_or_else(|| anyhow::anyhow!("{}: no weight matrix", files[0].display()))?;
    BaseDtype::from_s(t.dtype().as_str())
}

/// A config for a known public model, for estimates without the files (`profile --preset`).
pub fn preset(name: &str) -> Option<ModelConfig> {
    let c = |family, vocab, hidden, inter, layers, heads, kv, positions, theta, tied| ModelConfig {
        family, vocab_size: vocab, hidden_size: hidden, intermediate_size: inter, num_hidden_layers: layers, num_attention_heads: heads,
        num_key_value_heads: kv, head_dim: hidden / heads, rms_norm_eps: 1e-5, rope_theta: theta, max_position_embeddings: positions, tie_word_embeddings: tied,
    };
    Some(match name.to_ascii_lowercase().as_str() {
        "llama-2-7b" => c(Family::Llama, 32000, 4096, 11008, 32, 32, 32, 4096, 10000.0, false),
        "mistral-7b" => c(Family::Mistral, 32000, 4096, 14336, 32, 32, 8, 32768, 10000.0, false),
        "llama-3.1-8b" => c(Family::Llama, 128256, 4096, 14336, 32, 32, 8, 131072, 500000.0, false),
        "qwen2.5-7b" => c(Family::Qwen2, 152064, 3584, 18944, 28, 28, 4, 32768, 1000000.0, false),
        "qwen2.5-0.5b" => c(Family::Qwen2, 151936, 896, 4864, 24, 14, 2, 32768, 1000000.0, true),
        _ => return None,
    })
}

pub const PRESETS: [&str; 5] = ["llama-2-7b", "mistral-7b", "llama-3.1-8b", "qwen2.5-7b", "qwen2.5-0.5b"];
