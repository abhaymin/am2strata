//! The trainer as the `am2strata-workshop` command: the `--json` events the server reads, the up-front refusal,
//! SIGTERM = checkpoint and stop, `--resume auto`, `--threads`, `find-lr`, `profile`, `autopilot`, `--soup`.

mod common;

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::Value;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_am2strata-workshop"))
}

/// Run to the end; the events and whether it succeeded.
fn run(args: &[&str]) -> (Vec<Value>, bool) {
    let out = bin().arg("--json").args(args).output().unwrap();
    let events = String::from_utf8_lossy(&out.stdout).lines().map(|l| serde_json::from_str(l).unwrap_or_else(|_| panic!("not a JSON event: {l}"))).collect();
    (events, out.status.success())
}

fn of<'a>(events: &'a [Value], kind: &str) -> Vec<&'a Value> {
    events.iter().filter(|e| e["event"] == kind).collect()
}

fn p(path: &Path) -> &str {
    path.to_str().unwrap()
}

const SMALL: [&str; 14] = ["--rank", "4", "--targets", "q_proj,v_proj", "--lr", "0.02", "--max-len", "64", "--accumulation", "2", "--logging-steps", "1", "--seed", "3"];

#[test]
fn train_speaks_the_event_protocol_and_writes_a_peft_adapter() {
    let (root, model, data) = common::model_and_data("cli-train");
    let out = root.join("adapter");
    let mut args = vec!["train", "--model", p(&model), "--data", p(&data), "--out", p(&out), "--epochs", "1", "--threads", "2"];
    args.extend(SMALL);
    let (events, ok) = run(&args);
    assert!(ok, "{events:?}");
    assert_eq!(events[0]["event"], "start");
    assert_eq!((events[0]["threads"].as_u64(), events[0]["device"].as_str()), (Some(2), Some("cpu")));
    // 24 conversations, 2 held out (10 %), 22 / 2 per step = 11 steps, each a progress event with the numbers
    let steps: Vec<&Value> = of(&events, "progress").into_iter().filter(|e| e["step"].is_u64()).collect();
    assert_eq!(steps.len(), 11);
    for (i, e) in steps.iter().enumerate() {
        assert_eq!((e["step"].as_u64(), e["done"].as_u64(), e["total"].as_u64()), (Some(i as u64 + 1), Some(i as u64 + 1), Some(11)));
        assert!(e["loss"].as_f64().unwrap() > 0.0 && e["lr"].is_f64() && e["eta"].is_f64() && e["message"].is_string());
    }
    let logs: Vec<&str> = of(&events, "log").iter().filter_map(|e| e["message"].as_str()).collect();
    assert!(logs.iter().any(|m| m.contains("a 4-bit base (QLoRA) is not done")), "the command says what it does not do: {logs:?}");
    assert!(logs.iter().any(|m| m.starts_with("fits: needs")), "{logs:?}");
    assert!(logs.iter().any(|m| m.starts_with("held-out loss")), "{logs:?}");
    let done = events.last().unwrap();
    assert_eq!((done["event"].as_str(), done["kind"].as_str(), done["steps"].as_u64(), done["stopped"].as_bool()), (Some("done"), Some("lora-adapter"), Some(11), Some(false)));
    assert_eq!(done["output"].as_str(), Some(p(&out)));
    assert!(done["eval_loss"].is_f64() && done["final_loss"].is_f64() && done["trainable_params"].as_u64() == Some(3 * 4 * ((32 + 32) + (32 + 16))));
    assert!(out.join("adapter_model.safetensors").exists() && out.join("adapter_config.json").exists() && out.join("training_report.json").exists());
    // --threads 2: the process ran with a couple of worker threads, not one per core
    if let Some(t) = done["os_threads"].as_u64() {
        assert!(t <= 6, "{t} threads with --threads 2");
    }
    // and the Workshop's own merge takes the adapter
    let merged = root.join("merged");
    let (ev, ok) = run(&["merge", p(&model), p(&out), "--out", p(&merged)]);
    assert!(ok, "{ev:?}");
    assert!(merged.join("config.json").exists() && merged.join("tokenizer.json").exists());
}

#[cfg(unix)]
#[test]
fn sigterm_saves_a_checkpoint_and_resume_finishes_the_run() {
    let (root, model, data) = common::model_and_data("cli-term");
    let out = root.join("adapter");
    let mut args = vec!["--json", "train", "--model", p(&model), "--data", p(&data), "--out", p(&out), "--epochs", "40"];
    args.extend(SMALL);
    let mut child = bin().args(&args).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    // once it is training (the third step is out), cancel it the way the server does
    let mut seen = Vec::new();
    for l in lines.by_ref() {
        let e: Value = serde_json::from_str(&l.unwrap()).unwrap();
        let third = e["step"].as_u64() == Some(3);
        seen.push(e);
        if third {
            break;
        }
    }
    unsafe { libc::kill(child.id() as i32, libc::SIGTERM); }
    seen.extend(lines.map(|l| serde_json::from_str::<Value>(&l.unwrap()).unwrap()));
    assert!(child.wait().unwrap().success(), "a cancelled run ends cleanly");
    let done = seen.last().unwrap();
    assert_eq!((done["event"].as_str(), done["kind"].as_str(), done["stopped"].as_bool()), (Some("done"), Some("checkpoint"), Some(true)));
    let at = done["steps"].as_u64().unwrap();
    assert!((3..440).contains(&at), "stopped at step {at} of 440");
    let ckpt = out.join(format!("checkpoint-{at}"));
    assert_eq!(done["output"].as_str(), Some(p(&ckpt)));
    for f in ["adapter_model.safetensors", "adapter_config.json", "optimizer.safetensors", "trainer_state.json"] {
        assert!(ckpt.join(f).exists(), "{f} is in the checkpoint");
    }
    assert!(!out.join("adapter_model.safetensors").exists());

    // the same command with --resume auto goes on from that step (two more steps here, to keep the test short)
    let resumed = bin().args(&args).args(["--resume", "auto", "--save-steps", "2"]).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
    let pid = resumed.id();
    let mut first_step = None;
    let mut resumed_from = false;
    let mut child = resumed;
    for l in BufReader::new(child.stdout.take().unwrap()).lines() {
        let e: Value = serde_json::from_str(&l.unwrap()).unwrap();
        resumed_from |= e["message"].as_str().is_some_and(|m| m.starts_with("resumed from") && m.ends_with(&format!("at step {at}")));
        if let Some(s) = e["step"].as_u64().filter(|_| e["event"] == "progress") {
            first_step.get_or_insert(s);
            if s >= at + 2 {
                unsafe { libc::kill(pid as i32, libc::SIGTERM); }
            }
        }
    }
    child.wait().unwrap();
    assert!(resumed_from, "the run said where it resumed");
    assert_eq!(first_step, Some(at + 1), "the first step after the resume is the next one");
}

#[test]
fn a_run_that_will_not_fit_is_refused_before_it_starts() {
    // the tiny model's config with 7B's sizes, and no weights: the refusal comes from the estimate alone
    let (root, model, data) = common::model_and_data("cli-fit");
    let big = root.join("big");
    std::fs::create_dir_all(&big).unwrap();
    let mut cfg: Value = serde_json::from_str(&std::fs::read_to_string(model.join("config.json")).unwrap()).unwrap();
    for (k, v) in [("hidden_size", 16384), ("intermediate_size", 65536), ("num_hidden_layers", 96), ("num_attention_heads", 64), ("num_key_value_heads", 64)] {
        cfg[k] = v.into();
    }
    std::fs::write(big.join("config.json"), cfg.to_string()).unwrap();
    for f in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::copy(model.join(f), big.join(f)).unwrap();
    }
    if std::fs::read_to_string("/proc/meminfo").is_err() {
        return; // only Linux says how much memory is free; elsewhere nothing is refused
    }
    let (events, ok) = run(&["train", "--model", p(&big), "--data", p(&data), "--out", p(&root.join("o"))]);
    assert!(!ok);
    let err = events.last().unwrap();
    assert_eq!(err["event"], "error");
    let m = err["message"].as_str().unwrap();
    assert!(m.contains("this run needs about") && m.contains("is free") && m.contains("autopilot") && m.contains("--force"), "{m}");
    assert!(!root.join("o").exists(), "nothing was written");
    // other refusals, each plain: a 4-bit base, a GPU
    let msg = |args: &[&str]| run(args).0.last().unwrap()["message"].as_str().unwrap().to_string();
    assert!(msg(&["train", "--model", p(&model), "--data", p(&data), "--out", p(&root.join("o")), "--base-dtype", "4bit"]).contains("QLoRA"));
    assert!(msg(&["train", "--model", p(&model), "--data", p(&data), "--out", p(&root.join("o")), "--device", "cuda"]).contains("CPU only"));
}

#[test]
fn find_lr_reports_a_rate_and_writes_nothing() {
    let (root, model, data) = common::model_and_data("cli-lr");
    let (report, never) = (root.join("lr.json"), root.join("never"));
    let mut args = vec!["find-lr", "--model", p(&model), "--data", p(&data), "--out", p(&never), "--steps", "16", "--start-lr", "1e-4", "--end-lr", "10", "--report", p(&report)];
    args.extend(SMALL);
    let (events, ok) = run(&args);
    assert!(ok, "{events:?}");
    let r = of(&events, "find-lr")[0];
    let (lrs, losses) = (r["lrs"].as_array().unwrap(), r["losses"].as_array().unwrap());
    assert_eq!(lrs.len(), losses.len());
    assert!(lrs.len() >= 5 && lrs.len() <= 16);
    assert!((lrs[0].as_f64().unwrap() - 1e-4).abs() < 1e-12);
    assert!(lrs.windows(2).all(|w| w[1].as_f64() > w[0].as_f64()), "the rate rises");
    let suggested = r["suggested_lr"].as_f64().expect("a suggestion");
    assert!(suggested >= 1e-4 && suggested < r["min_loss_lr"].as_f64().unwrap().max(1e-4) * 1.0001, "the suggestion is before the minimum");
    assert!(r["lines"][0].as_str().unwrap().starts_with("suggested learning rate"));
    assert!(of(&events, "done").is_empty() && !root.join("never").exists(), "a range test leaves no adapter");
    let saved: Value = serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(saved["suggested_lr"], r["suggested_lr"]);
}

#[test]
fn profile_and_autopilot_answer_before_any_training() {
    let (root, model, data) = common::model_and_data("cli-profile");
    // from the model and the data: 22 conversations to train on, rank 4 on q and v of 3 blocks
    let (events, ok) = run(&["profile", "--model", p(&model), "--data", p(&data), "--rank", "4", "--max-len", "64", "--cpu-gflops", "50"]);
    assert!(ok, "{events:?}");
    let pr = of(&events, "profile")[0];
    assert_eq!(pr["trainable_parameters"].as_u64(), Some(3 * 4 * ((32 + 32) + (32 + 16))));
    // a block: q and o 32×32, k and v 16×32, three MLP matrices of 32×64, two norms = 9280; embeddings and head 259×32 each
    assert_eq!(pr["model"]["parameters"].as_u64(), Some(2 * 259 * 32 + 3 * 9280 + 32));
    assert_eq!(pr["steps"].as_u64(), Some(17), "22 conversations × 3 epochs / 4 per step");
    assert!(pr["memory"]["total"].as_u64().unwrap() > pr["memory"]["weights"].as_u64().unwrap());
    assert_eq!(pr["machine"]["cpu_gflops"].as_f64(), Some(50.0));
    let lines: Vec<&str> = pr["lines"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    assert!(lines.iter().any(|l| l.starts_with("memory: weights")) && lines.iter().any(|l| l.contains("QLoRA")), "{lines:?}");
    // from a preset: no files at all
    let (events, ok) = run(&["profile", "--preset", "llama-2-7b", "--examples", "1000", "--avg-tokens", "300", "--max-len", "1024", "--rank", "16", "--alpha", "32",
                             "--targets", "q_proj,v_proj", "--train-layers", "2", "--base-dtype", "f16", "--cache-activations", p(&root.join("c"))]);
    assert!(ok, "{events:?}");
    let pr = of(&events, "profile")[0];
    assert_eq!((pr["model"]["parameters"].as_u64(), pr["trainable_parameters"].as_u64(), pr["memory"]["total"].as_u64()), (Some(6_738_415_616), Some(524_288), Some(6_604_537_856)));

    // autopilot: the plan, its reasons, and a command that the trainer accepts as it is
    let (events, ok) = run(&["autopilot", "--model", p(&model), "--data", p(&data), "--out", p(&root.join("auto")), "--time-budget", "1h", "--ram-budget", "4GB", "--cpu-gflops", "50"]);
    assert!(ok, "{events:?}");
    let plan = of(&events, "plan")[0];
    assert_eq!((plan["fits_memory"].as_bool(), plan["fits_time"].as_bool()), (Some(true), Some(true)));
    assert_eq!((plan["settings"]["rank"].as_u64(), plan["settings"]["base_dtype"].as_str(), plan["settings"]["train_layers"].is_null()), (Some(8), Some("f32"), true));
    assert!(plan["choices"].as_array().unwrap().len() >= 10);
    let command: Vec<&str> = plan["command"].as_str().unwrap().split_whitespace().skip(1).collect();
    let mut args = command.clone();
    args.extend(["--max-steps", "2"]);
    let (events, ok) = run(&args);
    assert!(ok, "the autopilot's own command failed: {events:?}");
    assert!(root.join("auto/adapter_model.safetensors").exists());
}

#[test]
fn a_soup_config_drives_the_trainer() {
    let (root, model, data) = common::model_and_data("cli-soup");
    let yaml = root.join("soup.yaml");
    std::fs::write(&yaml, format!("base: {}\ntask: sft\ndata:\n  train: {}\n  format: chatml\n  max_length: 64\ntraining:\n  epochs: 1\n  lr: 2e-2\n  batch_size: 2\n  \
        gradient_accumulation_steps: 2\n  lora:\n    r: 4\n    alpha: 8\n    target_modules: auto\noutput: {}\n", p(&model), p(&data), p(&root.join("soup-out")))).unwrap();
    // as written it asks for Soup's default 4-bit base: refused by that name
    let (events, ok) = run(&["train", "--soup", p(&yaml)]);
    assert!(!ok);
    assert!(events.last().unwrap()["message"].as_str().unwrap().contains("training.quantization: 4bit, Soup's default when unset"), "{events:?}");
    // with the stand-in allowed it trains: bf16 base, 2 × 2 conversations per step
    let (events, ok) = run(&["train", "--soup", p(&yaml), "--soup-relax", "--logging-steps", "1"]);
    assert!(ok, "{events:?}");
    assert_eq!(events[0]["base_dtype"], "bf16");
    let logs: Vec<&str> = of(&events, "log").iter().filter_map(|e| e["message"].as_str()).collect();
    assert!(logs.iter().any(|m| m.starts_with("soup: training.quantization") && m.contains("bf16")), "{logs:?}");
    // 22 conversations / 4 per step
    assert_eq!(events.last().unwrap()["steps"].as_u64(), Some(6));
    assert!(root.join("soup-out/adapter_config.json").exists());
}
