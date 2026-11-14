//! `soup.yaml` import: Soup's own generated configs (`tests/fixtures/soup/`, see the fixtures README), read into this
//! trainer's settings, with everything unsupported refused by name.

use std::path::{Path, PathBuf};

use strata_workshop::chatdata::DataFormat;
use strata_workshop::model::{BaseDtype, Target};
use strata_workshop::soup::{self, SoupError};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/soup").join(name)
}

fn unsupported(e: SoupError) -> Vec<String> {
    match e {
        SoupError::Unsupported(keys) => keys,
        other => panic!("expected a list of unsupported settings, got {other}"),
    }
}

#[test]
fn soup_init_chat_template() {
    // as Soup wrote it, it asks for a 4-bit base: refused, by that one name
    let keys = unsupported(soup::import_file(&fixture("init-chat.yaml"), false).unwrap_err());
    assert_eq!(keys.len(), 1, "{keys:?}");
    assert!(keys[0].starts_with("training.quantization: 4bit") && keys[0].contains("QLoRA is not done"), "{}", keys[0]);

    // with the stand-in allowed, everything else of the file is read
    let imp = soup::import_file(&fixture("init-chat.yaml"), true).unwrap();
    let s = &imp.settings;
    assert_eq!(imp.base, "meta-llama/Llama-3.1-8B-Instruct");
    assert_eq!(s.data, fixture("data/train.jsonl").parent().unwrap().parent().unwrap().join("./data/train.jsonl"));
    assert_eq!(s.out, fixture("output").parent().unwrap().join("./output"));
    assert_eq!(s.data_format, DataFormat::Alpaca);
    assert_eq!((s.val_split, s.max_len, s.epochs, s.lr), (0.1, 2048, 3.0, 2e-5));
    assert_eq!((s.lora.r, s.lora.alpha, s.lora.dropout, s.lora.use_rslora), (64, 16.0, 0.05, false));
    assert_eq!(s.lora.targets, vec![Target::Q, Target::V], "target_modules: auto is PEFT's default pair");
    // batch_size: auto → 1, times Soup's default accumulation of 4; Soup's other defaults for what the file leaves out
    assert_eq!((s.accumulation, s.seed, s.warmup_ratio, s.weight_decay, s.max_grad_norm, s.save_steps, s.logging_steps), (4, 42, 0.03, 0.01, 1.0, 100, 10));
    assert_eq!(s.base_dtype, BaseDtype::BF16);
    assert!(!s.score_everything);
    let notes = imp.notes.join("\n");
    for needle in ["is not a folder here", "batch_size: auto", "target_modules: auto", "QLoRA (a quantised base) is not done: the base is held in bf16"] {
        assert!(notes.contains(needle), "no note about {needle:?} in:\n{notes}");
    }
}

#[test]
fn soup_fetch_llama_example() {
    // Soup's bundled example: 4-bit, an 8-bit bitsandbytes optimizer, and `lora:` at the top level
    let keys = unsupported(soup::import_file(&fixture("fetch-llama-3.1-8b-lora.yaml"), false).unwrap_err());
    assert_eq!(keys.len(), 2, "{keys:?}");
    assert!(keys.iter().any(|k| k.starts_with("training.optimizer: adamw_bnb_8bit")), "{keys:?}");
    assert!(keys.iter().any(|k| k.starts_with("training.quantization: 4bit")), "{keys:?}");

    let imp = soup::import_file(&fixture("fetch-llama-3.1-8b-lora.yaml"), true).unwrap();
    let s = &imp.settings;
    assert_eq!((s.epochs, s.lr, s.max_len, s.data_format), (3.0, 2e-4, 4096, DataFormat::Auto));
    // the top-level lora block is what the file says, and is said to be read that way
    assert_eq!((s.lora.r, s.lora.alpha, s.lora.dropout), (16, 32.0, 0.05));
    assert_eq!(s.out, fixture("out").parent().unwrap().join("./out"));
    let notes = imp.notes.join("\n");
    assert!(notes.contains("lora: at the top level") && notes.contains("adamw_bnb_8bit — replaced by AdamW"), "{notes}");
}

#[test]
fn soup_init_moe_template_is_refused_by_name() {
    // Soup's MoE template: whatever it sets beyond plain LoRA is named, relaxed or not
    for relax in [false, true] {
        let keys = unsupported(soup::import_file(&fixture("init-moe.yaml"), relax).unwrap_err());
        assert!(keys.iter().any(|k| k.starts_with("training.moe_lora")), "{keys:?}");
    }
}

#[test]
fn every_unsupported_setting_is_listed() {
    let yaml = r#"
base: ./model
task: dpo
modality: vision
data:
  train: train.jsonl
  format: llava
  image_dir: ./images
  streaming: false
training:
  lr: 1e-4
  scheduler: linear
  quantization: none
  use_dora: false
  neftune_alpha: 5
  gradient_checkpointing: true
  lora:
    r: 8
    use_dora: true
    target_modules: [q_proj, embed_tokens]
eval:
  suite: bench.yaml
"#;
    let keys = unsupported(soup::import(yaml, Path::new("/cfg"), true).unwrap_err());
    let named: Vec<&str> = keys.iter().map(|k| k.split(':').next().unwrap()).collect();
    assert_eq!(named, ["eval", "task", "modality", "data.image_dir", "data.format", "training.neftune_alpha", "training.gradient_checkpointing", "training.scheduler",
                       "training.lora.use_dora", "training.lora.target_modules"], "{keys:?}");
    // what is switched off is not an obstacle
    assert!(!keys.iter().any(|k| k.starts_with("data.streaming") || k.starts_with("training.use_dora")));
}

#[test]
fn a_plain_config_imports_without_stand_ins() {
    let yaml = r#"
base: /models/tiny
data:
  train: /data/chat.jsonl
  format: chatml
  val_split: 0.2
  max_length: 512
  train_on_responses_only: false
training:
  epochs: 2
  lr: 2e-4          # PyYAML reads this as text; Soup takes it as a number
  batch_size: 8
  gradient_accumulation_steps: 2
  seed: 7
  quantization: none
  optimizer: adamw_torch
  scheduler: cosine
  lora:
    r: 16
    alpha: 32
    dropout: 0.1
    use_rslora: true
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
output: /runs/a
"#;
    let imp = soup::import(yaml, Path::new("/cfg"), false).unwrap();
    let s = &imp.settings;
    assert_eq!((s.data.as_path(), s.out.as_path()), (Path::new("/data/chat.jsonl"), Path::new("/runs/a")));
    assert_eq!((s.data_format, s.val_split, s.max_len, s.score_everything), (DataFormat::Messages, 0.2, 512, true));
    assert_eq!((s.epochs, s.lr, s.accumulation, s.seed, s.base_dtype), (2.0, 2e-4, 16, 7, BaseDtype::F32));
    assert_eq!((s.lora.r, s.lora.alpha, s.lora.dropout, s.lora.use_rslora, s.lora.seed), (16, 32.0, 0.1, true, 7));
    assert_eq!(s.lora.targets, Target::ALL.to_vec());
    assert!(imp.notes.iter().any(|n| n.contains("8 × gradient_accumulation_steps 2 = 16 conversations per step")));

    // not YAML, not a mapping, no base, a wrong kind of value: said plainly
    assert!(matches!(soup::import("- a\n- b\n", Path::new("."), false), Err(SoupError::Invalid(m)) if m.contains("mapping")));
    assert!(matches!(soup::import("data: {train: a.jsonl}\n", Path::new("."), false), Err(SoupError::Invalid(m)) if m.contains("base")));
    assert!(matches!(soup::import("base: m\ndata: {train: a.jsonl}\ntraining: {lr: fast, quantization: none}\n", Path::new("."), false), Err(SoupError::Invalid(m)) if m.contains("training.lr")));
}
