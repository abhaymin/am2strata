//! The model optimizer (docs/OPTIMIZER-PLAN.md, steps 2 and 3): a bounded search over the run settings that decide a
//! model's speed on this machine, measured on a fixed prompt set, kept within a quality floor, saved as a tuned
//! profile beside the run config (`<config>.tuned.json`), never over it; the profile is used from the next load.
//!
//! - **The settings tuned** are only those the run config already passes, so an engine is never given a flag it does
//!   not take: threads (`-t`), the micro-batch (`-ub`) and batch (`-b`), and the resident experts
//!   (`--resident-experts`). The KV cache's types are never changed (the quality floor).
//! - **The search:** coordinate descent from the run config as it is: each setting in turn, each candidate value
//!   tried once, the best kept when it gains 3% or more; rounds until one gains nothing (at most 3), within the time
//!   budget (default 20 minutes). A candidate that fails to start, fails a prompt or answers with other tokens than
//!   the starting config at temperature 0 is dropped.
//! - **The objective:** decode speed by default, or prefill speed (`"objective": "prefill"`, for agents with long
//!   prompts).

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The prompt set: a short chat, a long prompt (prefill), a long generation. Answered greedily.
pub fn prompts() -> Vec<(String, u32)> {
    let para = "The lighthouse keeper wrote in the log each night: the wind, the sea, the ships that passed, the hours \
                the lamp burned, and what needed mending before the next storm. ";
    vec![
        ("Say in two sentences why the sky is blue.".to_string(), 64),
        // about 8,000 tokens of text (about 40 tokens a repeat)
        (format!("{}\nIn one sentence, what did the keeper write about?", para.repeat(200)), 32),
        ("Write a short story about a lighthouse keeper and a storm.".to_string(), 256),
    ]
}

#[derive(Clone, Debug, PartialEq)]
pub struct Knob {
    pub flag: String,
    pub values: Vec<String>,
    /// the engine's own value when the run config does not pass the flag (Strata's engines' flags, added only then)
    pub default: Option<String>,
}

impl Knob {
    /// The value it has in `args`, or the engine's default.
    pub fn current(&self, args: &[String]) -> Option<String> {
        arg(args, &self.flag).or_else(|| self.default.clone())
    }
}

/// The value `flag` has in `args`.
pub fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

/// `args` with `flag` set to `value` (added at the end when it is not there).
pub fn with_arg(args: &[String], flag: &str, value: &str) -> Vec<String> {
    let mut a = args.to_vec();
    match a.iter().position(|x| x == flag) {
        Some(i) if i + 1 < a.len() => a[i + 1] = value.to_string(),
        _ => a.extend([flag.to_string(), value.to_string()]),
    }
    a
}

/// The settings to tune, from the flags the run config passes and the machine's facts. For Strata's own engines
/// (`strata`, `strata-poly`), their speed flags that leave the answers as they are, with the engine's defaults, even
/// when the run config does not pass them: `--threads`, `--window` (the tokens that go through at once, guesses
/// counted only when right), `--prompt-batch` and `--expert-swaps`. Never the ones that move the answers
/// (`--tail-skip`, `--skip-weight`, `--kv`).
pub fn knobs(exe: &Path, args: &[String], facts: &Value) -> Vec<Knob> {
    let mut k: Vec<Knob> = Vec::new();
    let cores = facts["cpu"]["cores"].as_u64().unwrap_or(0);
    let threads = facts["cpu"]["threads"].as_u64().unwrap_or(0);
    let list = |vals: Vec<u64>, cur: u64| -> Vec<String> {
        let mut v: Vec<u64> = vals.into_iter().chain([cur]).filter(|x| *x > 0).collect();
        v.sort_unstable();
        v.dedup();
        v.into_iter().map(|x| x.to_string()).collect()
    };
    let mut add = |flags: &[&str], vals: Vec<u64>| {
        for f in flags {
            if let Some(cur) = arg(args, f).and_then(|v| v.parse::<u64>().ok()) {
                k.push(Knob { flag: f.to_string(), values: list(vals, cur), default: None });
                return;
            }
        }
    };
    if cores > 0 {
        add(&["-t", "--threads"], vec![cores / 2, cores * 3 / 4, cores, threads]);
    }
    add(&["-ub", "--ubatch-size"], vec![128, 256, 512, 1024]);
    add(&["-b", "--batch-size"], vec![512, 1024, 2048]);
    if let Some(n) = arg(args, "--resident-experts").and_then(|v| v.parse::<u64>().ok()) {
        add(&["--resident-experts"], vec![n / 2, n * 3 / 4, n * 5 / 4, n * 3 / 2]);
    }
    let strata = exe.file_name().and_then(|n| n.to_str()).is_some_and(|n| n == "strata" || n.starts_with("strata-poly"));
    if strata {
        let own: [(&str, Vec<u64>, u64); 4] = [
            ("--threads", vec![cores * 3 / 4, cores, threads * 3 / 4], threads),
            ("--window", vec![1, 4, 8, 12], 8),
            ("--prompt-batch", vec![512, 1024, 2048], 1024),
            ("--expert-swaps", vec![0, 32, 64, 128], 64),
        ];
        for (flag, vals, default) in own {
            if k.iter().any(|x| x.flag == flag) || (flag == "--threads" && threads == 0) {
                continue;
            }
            match arg(args, flag).and_then(|v| v.parse::<u64>().ok()) {
                Some(cur) => k.push(Knob { flag: flag.into(), values: list(vals, cur), default: None }),
                None => k.push(Knob { flag: flag.into(), values: list(vals, default), default: Some(default.to_string()) }),
            }
        }
    }
    k
}

#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    pub decode_tok_s: f64,
    pub prefill_tok_s: f64,
    /// the time to the first token of the short chat, in ms
    pub first_token_ms: f64,
    /// each answer's tokens, for the quality floor
    pub tokens: Vec<Vec<u32>>,
}

impl Score {
    pub fn from_trial(runs: &[(Vec<u32>, strata_protocol::Done)]) -> Score {
        let (mut gen, mut dms, mut pt, mut pms) = (0u64, 0f64, 0u64, 0f64);
        for (_, d) in runs {
            gen += d.generated;
            dms += d.decode_ms;
            pt += d.prompt_tokens;
            pms += d.prompt_ms;
        }
        let rate = |n: u64, ms: f64| if ms > 0.0 { (n as f64 / ms * 1000.0 * 10.0).round() / 10.0 } else { 0.0 };
        Score { decode_tok_s: rate(gen, dms), prefill_tok_s: rate(pt, pms),
                first_token_ms: runs.first().map(|(_, d)| d.prompt_ms).unwrap_or(0.0),
                tokens: runs.iter().map(|(t, _)| t.clone()).collect() }
    }

    fn value(&self, objective: &str) -> f64 {
        if objective == "prefill" { self.prefill_tok_s } else { self.decode_tok_s }
    }

    pub fn view(&self) -> Value {
        json!({"decode_tok_s": self.decode_tok_s, "prefill_tok_s": self.prefill_tok_s, "first_token_ms": self.first_token_ms})
    }
}

#[derive(Debug)]
pub struct Outcome {
    pub args: Vec<String>,
    pub baseline: Score,
    pub best: Score,
    /// each trial: the setting, the value, its score or why it was dropped
    pub trials: Vec<Value>,
    pub stopped_by_budget: bool,
}

const GAIN: f64 = 1.03;

fn prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The quality floor: per prompt, a candidate's answer must agree with the run config's (the same tokens from the
/// start) as far as the run config agrees with itself: all of it when the engine is exact from run to run, else at
/// least half as far as its own second run did.
struct Floor {
    base: Vec<Vec<u32>>,
    self_agree: Vec<usize>,
}

impl Floor {
    fn new(a: &Score, b: &Score) -> Floor {
        Floor { base: a.tokens.clone(), self_agree: a.tokens.iter().zip(&b.tokens).map(|(x, y)| prefix(x, y)).collect() }
    }

    fn agree(&self, s: &Score) -> Vec<usize> {
        self.base.iter().zip(&s.tokens).map(|(x, y)| prefix(x, y)).collect()
    }

    fn passes(&self, s: &Score) -> bool {
        s.tokens.len() == self.base.len() && self.base.iter().zip(&s.tokens).zip(&self.self_agree).all(|((b, c), &me)| {
            if me >= b.len() { c == b } else { prefix(b, c) * 2 >= me }
        })
    }
}

/// The search. `trial(args)` starts the model with those settings and runs the prompt set.
pub async fn search<F, Fut>(start: Vec<String>, knobs: &[Knob], objective: &str, budget: Duration, mut trial: F) -> Result<Outcome>
where
    F: FnMut(Vec<String>) -> Fut,
    Fut: Future<Output = Result<Score>>,
{
    let t0 = Instant::now();
    let baseline = trial(start.clone()).await.map_err(|e| anyhow!("the run config as it is did not run: {e:#}"))?;
    // the run config twice: how far its greedy answers agree with themselves (an engine whose sums are not
    // bit-exact from run to run, as an expert cache that moves, parts at some token), the yardstick for the others
    let again = trial(start.clone()).await.map_err(|e| anyhow!("the run config as it is did not run again: {e:#}"))?;
    let floor = Floor::new(&baseline, &again);
    let (mut args, mut best) = (start, baseline.clone());
    let mut trials = vec![json!({"setting": "the run config", "score": baseline.view()}),
                          json!({"setting": "the run config, again", "score": again.view(), "agrees_to": floor.self_agree})];
    let mut stopped = false;
    'rounds: for _ in 0..3 {
        let mut gained = false;
        for k in knobs {
            let cur = k.current(&args);
            let mut pick: Option<(Vec<String>, Score)> = None;
            for v in &k.values {
                if Some(v) == cur.as_ref() {
                    continue;
                }
                if t0.elapsed() >= budget {
                    stopped = true;
                    break 'rounds;
                }
                let cand = with_arg(&args, &k.flag, v);
                if trials.iter().any(|t| t["args"] == json!(cand)) {
                    continue;
                }
                let row = match trial(cand.clone()).await {
                    Err(e) => json!({"dropped": format!("{e:#}")}),
                    Ok(s) if !floor.passes(&s) => json!({"dropped": "its answers part from the run config's sooner than the run config's \
                                                                       own do at temperature 0", "score": s.view(), "agrees_to": floor.agree(&s)}),
                    Ok(s) => {
                        let row = json!({"score": s.view()});
                        let to_beat = pick.as_ref().map(|p| p.1.value(objective)).unwrap_or(best.value(objective) * GAIN);
                        if s.value(objective) >= to_beat {
                            pick = Some((cand.clone(), s));
                        }
                        row
                    }
                };
                let mut row = row;
                row["setting"] = json!(k.flag);
                row["value"] = json!(v);
                row["args"] = json!(cand);
                trials.push(row);
            }
            if let Some((a, s)) = pick {
                args = a;
                best = s;
                gained = true;
            }
        }
        if !gained {
            break;
        }
    }
    Ok(Outcome { args, baseline, best, trials, stopped_by_budget: stopped })
}

/// The tuned profile's file, beside the run config.
pub fn profile_path(config: &Path) -> PathBuf {
    let name = config.file_name().and_then(|n| n.to_str()).unwrap_or("model.json");
    config.with_file_name(format!("{}.tuned.json", name.trim_end_matches(".json")))
}

/// The args a load uses: the tuned profile's when it is on and was tuned on this hardware, else None.
pub fn tuned_args(config: &Path, fingerprint: &str) -> Option<Vec<String>> {
    let p: Value = serde_json::from_str(&std::fs::read_to_string(profile_path(config)).ok()?).ok()?;
    if p["enabled"] == false || p["fingerprint"].as_str() != Some(fingerprint) {
        return None;
    }
    serde_json::from_value(p["args"].clone()).ok()
}


/// The optimizer's job, the one running or last run (GET /api/optimise).
pub static JOB: std::sync::Mutex<Value> = std::sync::Mutex::new(Value::Null);

fn set(f: impl FnOnce(&mut Value)) {
    let mut j = JOB.lock().unwrap();
    f(&mut j);
}

/// Tune `id`: hold the turn (chats wait, told in the queue), search, write the profile, load the model again (with
/// the profile when it gained). Runs in the background; its state is in JOB.
pub async fn run(mgr: std::sync::Arc<crate::manager::Manager>, id: String, objective: String, budget: Duration) {
    let t0 = Instant::now();
    set(|j| *j = json!({"state": "waiting", "model": id, "objective": objective, "budget_s": budget.as_secs(), "started": crate::now(),
                        "trials": 0}));
    let result = async {
        let _turn = mgr.queue.enter(crate::queue::Need::Text(id.clone()), "optimise", &id, |_| {}).await.map_err(|t| anyhow!("{t}"))?;
        let entry = mgr.entry(&id).ok_or_else(|| anyhow!("no installed model is named {id:?}"))?;
        let models_dir = entry.path.parent().map(Path::to_path_buf).unwrap_or_default();
        let facts = crate::machine::facts(&models_dir);
        let k = knobs(&entry.cfg.exe, &entry.cfg.args, &facts);
        if k.is_empty() {
            return Err(anyhow!("its run config passes none of the settings the optimizer tunes (-t, -ub, -b, --resident-experts; \
                                 for Strata's engines also --threads, --window, --prompt-batch, --expert-swaps)"));
        }
        set(|j| { j["state"] = json!("running"); j["settings"] = json!(k.iter().map(|k| &k.flag).collect::<Vec<_>>()); });
        let prompts = prompts();
        let out = search(entry.cfg.args.clone(), &k, &objective, budget, |args| {
            let (mgr, id, prompts) = (mgr.clone(), id.clone(), prompts.clone());
            set(|j| { j["trials"] = json!(j["trials"].as_u64().unwrap_or(0) + 1); j["trying"] = json!(args.join(" ")); });
            async move {
                // each trial is capped: a start or prompt that hangs is a failed trial
                let runs = tokio::time::timeout(Duration::from_secs(600), mgr.trial(&id, args, &prompts)).await
                    .map_err(|_| anyhow!("the trial took more than 10 minutes"))??;
                let s = Score::from_trial(&runs);
                set(|j| j["last"] = s.view());
                Ok(s)
            }
        }).await;
        // the model back in memory, whatever the search gave
        let out = out?;
        let gain = out.best.value(&objective) / out.baseline.value(&objective).max(1e-9);
        let profile = json!({"model": id, "args": out.args, "objective": objective, "enabled": gain >= GAIN,
                             "baseline": out.baseline.view(), "tuned": out.best.view(), "gain": (gain * 1000.0).round() / 1000.0,
                             "fingerprint": crate::machine::fingerprint(&facts), "machine": facts, "at": crate::now(),
                             "seconds": t0.elapsed().as_secs(), "stopped_by_budget": out.stopped_by_budget, "trials": out.trials,
                             "run_config_args": entry.cfg.args});
        std::fs::write(profile_path(&entry.path), serde_json::to_string_pretty(&profile)?)?;
        Ok::<Value, anyhow::Error>(profile)
    }.await;
    let reload = mgr.load(&id).await.err().map(|e| format!("{e:#}"));
    set(|j| {
        j["state"] = json!(if result.is_ok() { "done" } else { "error" });
        j["seconds"] = json!(t0.elapsed().as_secs());
        j["trying"] = Value::Null;
        match &result {
            Ok(p) => {
                for k in ["baseline", "tuned", "gain", "enabled", "args", "stopped_by_budget"] {
                    j[k] = p[k].clone();
                }
            }
            Err(e) => j["error"] = json!(format!("{e:#}")),
        }
        if let Some(r) = &reload {
            j["reload_error"] = json!(r);
        }
    });
}

/// What GET /api/models shows of a model's tuning.
pub fn view(config: &Path) -> Value {
    let Some(p) = std::fs::read_to_string(profile_path(config)).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) else {
        return Value::Null;
    };
    json!({"enabled": p["enabled"], "objective": p["objective"], "tuned": p["tuned"], "baseline": p["baseline"], "gain": p["gain"],
           "at": p["at"], "args": p["args"]})
}

/// Turn a profile on or off (it is kept either way).
pub fn set_enabled(config: &Path, on: bool) -> Result<()> {
    let path = profile_path(config);
    let mut p: Value = serde_json::from_str(&std::fs::read_to_string(&path).map_err(|_| anyhow!("this model has not been tuned"))?)?;
    p["enabled"] = json!(on);
    std::fs::write(path, serde_json::to_string_pretty(&p)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn only_the_flags_the_config_passes_are_tuned() {
        let facts = json!({"cpu": {"cores": 16, "threads": 32}});
        let k = knobs(Path::new("/x/llama-server"), &a("-m x.gguf -t 12 -ub 256 -ctk q8_0"), &facts);
        assert_eq!(k.iter().map(|k| k.flag.as_str()).collect::<Vec<_>>(), vec!["-t", "-ub"]);
        assert_eq!(k[0].values, vec!["8", "12", "16", "32"]);
        assert!(k.iter().all(|k| !k.flag.starts_with("-ct")), "the KV types are never changed");
        assert_eq!(with_arg(&a("-t 12 -ub 256"), "-ub", "512"), a("-t 12 -ub 512"));
        assert_eq!(with_arg(&a("-t 12"), "--window", "4"), a("-t 12 --window 4"), "a flag the config lacks is added");
    }

    #[test]
    fn strata_engines_get_their_own_flags_with_their_defaults() {
        let facts = json!({"cpu": {"cores": 8, "threads": 16}});
        let k = knobs(Path::new("/p/engine-poly/strata-poly"), &a("--model m.gguf --kv q8_0 --window 4"), &facts);
        let flags: Vec<&str> = k.iter().map(|k| k.flag.as_str()).collect();
        assert_eq!(flags, vec!["--threads", "--window", "--prompt-batch", "--expert-swaps"]);
        assert_eq!(k[0].default.as_deref(), Some("16"), "all usable threads, as strata-poly does");
        assert_eq!(k[0].values, vec!["6", "8", "12", "16"]);
        assert_eq!((k[1].default.as_deref(), k[1].current(&a("--window 4")).as_deref()), (None, Some("4")), "passed: its value");
        assert!(!flags.iter().any(|f| f.contains("skip") || *f == "--kv"), "never the flags that move the answers");
    }

    /// The mock of the plan's tests: speed and fit per setting from a table.
    fn table(args: Vec<String>) -> Result<Score> {
        let t: u64 = arg(&args, "-t").unwrap().parse().unwrap();
        let ub: u64 = arg(&args, "-ub").unwrap().parse().unwrap();
        if ub == 1024 {
            return Err(anyhow!("out of memory"));
        }
        // best at 12 threads (16 is the trap) and ub 512; ub 128 changes the answer (a broken kernel)
        let speed = 20.0 - (t as f64 - 12.0).abs() * 0.5 + if ub == 512 { 3.0 } else { 0.0 };
        let tokens = if ub == 128 { vec![vec![9]] } else { vec![vec![1, 2, 3]] };
        Ok(Score { decode_tok_s: speed, prefill_tok_s: 100.0, first_token_ms: 50.0, tokens })
    }

    #[tokio::test]
    async fn the_search_finds_the_tables_best_and_drops_what_fails() {
        let facts = json!({"cpu": {"cores": 16, "threads": 32}});
        let start = a("-m x.gguf -t 16 -ub 256");
        let k = knobs(Path::new("llama-server"), &start, &facts);
        let mut calls = 0;
        let out = search(start, &k, "decode", Duration::from_secs(60), |args| { calls += 1; async move { table(args) } }).await.unwrap();
        assert_eq!(arg(&out.args, "-t").as_deref(), Some("12"));
        assert_eq!(arg(&out.args, "-ub").as_deref(), Some("512"));
        assert_eq!(out.baseline.decode_tok_s, 18.0);
        assert_eq!(out.best.decode_tok_s, 23.0);
        let dropped: Vec<&Value> = out.trials.iter().filter(|t| t.get("dropped").is_some()).collect();
        assert!(dropped.iter().any(|t| t["value"] == "1024"), "the one that does not fit");
        assert!(dropped.iter().any(|t| t["value"] == "128"), "the one under the quality floor");
        assert!(calls < 20, "bounded: {calls} trials");
    }

    #[test]
    fn an_engine_that_is_not_exact_is_judged_by_its_own_agreement() {
        let sc = |t: Vec<u32>| Score { decode_tok_s: 1.0, prefill_tok_s: 1.0, first_token_ms: 1.0, tokens: vec![t] };
        // the run config twice: they part at token 6
        let f = Floor::new(&sc(vec![1, 2, 3, 4, 5, 6, 7, 8]), &sc(vec![1, 2, 3, 4, 5, 6, 9, 9]));
        assert!(f.passes(&sc(vec![1, 2, 3, 4, 0, 0, 0, 0])), "parts at 4: within half of 6");
        assert!(!f.passes(&sc(vec![1, 2, 0, 0, 0, 0, 0, 0])), "parts at 2: too soon");
        // an exact engine: every token
        let exact = Floor::new(&sc(vec![1, 2, 3]), &sc(vec![1, 2, 3]));
        assert!(!exact.passes(&sc(vec![1, 2, 4])));
    }

    #[tokio::test]
    async fn the_budget_stops_the_search_with_the_best_so_far() {
        let out = search(a("-t 16 -ub 256"), &knobs(Path::new("llama-server"), &a("-t 16 -ub 256"), &json!({"cpu": {"cores": 16, "threads": 32}})),
                         "decode", Duration::ZERO, |args| async move { table(args) }).await.unwrap();
        assert!(out.stopped_by_budget);
        assert_eq!(out.args, a("-t 16 -ub 256"));
    }

    #[test]
    fn a_profile_is_used_only_on_its_hardware_and_when_on() {
        let d = std::env::temp_dir().join(format!("opt-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let cfg = d.join("strata-tiel.json");
        assert_eq!(profile_path(&cfg), d.join("strata-tiel.tuned.json"));
        std::fs::write(profile_path(&cfg), json!({"args": ["-t", "12"], "fingerprint": "abc", "enabled": true}).to_string()).unwrap();
        assert_eq!(tuned_args(&cfg, "abc"), Some(a("-t 12")));
        assert_eq!(tuned_args(&cfg, "other"), None, "another machine's tuning is not used");
        std::fs::write(profile_path(&cfg), json!({"args": ["-t", "12"], "fingerprint": "abc", "enabled": false}).to_string()).unwrap();
        assert_eq!(tuned_args(&cfg, "abc"), None);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
