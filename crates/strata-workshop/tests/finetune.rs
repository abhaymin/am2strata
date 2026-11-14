//! Training on a real (tiny) model: chat data through the model's tokenizer and template, the loss on assistant
//! tokens only, accumulation, the schedule, checkpoints with exact resume, a PEFT adapter the Workshop's merge accepts.

mod common;

use std::cell::Cell;

use candle_core::Device;
use strata_workshop::chatdata::{self, ChatTokenizer, DataFormat};
use strata_workshop::dataset::Record;
use strata_workshop::finetune::{self, Event, Resume, TrainSettings};
use strata_workshop::model::{BaseDtype, Ctx, LlamaModel, LoraSpec, Target};
use strata_workshop::train::hf_cosine_lr;

fn settings(model: &std::path::Path, data: &std::path::Path, out: &std::path::Path) -> TrainSettings {
    let mut s = TrainSettings::new(model, data, out);
    s.lora = LoraSpec { r: 8, alpha: 16.0, dropout: 0.0, use_rslora: false, targets: Target::ALL.to_vec(), train_layers: None, seed: 7 };
    s.lr = 2e-2;
    s.epochs = 4.0;
    s.accumulation = 2;
    s.max_len = 64;
    s.val_split = 0.125;
    s.warmup_ratio = 0.1;
    s.save_steps = 0;
    s.logging_steps = 1;
    s.seed = 11;
    s
}

fn max_abs(a: &candle_core::Tensor, b: &candle_core::Tensor) -> f32 {
    let (a, b): (Vec<f32>, Vec<f32>) = (a.flatten_all().unwrap().to_vec1().unwrap(), b.flatten_all().unwrap().to_vec1().unwrap());
    assert_eq!(a.len(), b.len());
    a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn only_the_assistants_tokens_are_scored() {
    let (_root, model, _data) = common::model_and_data("mask");
    let tok = ChatTokenizer::load(&model, None).unwrap();
    let rec = Record { messages: vec![("system".into(), "Be brief.".into()), ("user".into(), "What colour is snow?".into()), ("assistant".into(), "It is white.".into()),
                                      ("user".into(), "And coal?".into()), ("assistant".into(), "Black.".into())] };
    let (text, spans) = tok.render_with_spans(&rec).unwrap();
    assert_eq!(text, "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nWhat colour is snow?<|im_end|>\n<|im_start|>assistant\nIt is white.<|im_end|>\n\
                      <|im_start|>user\nAnd coal?<|im_end|>\n<|im_start|>assistant\nBlack.<|im_end|>\n");
    let parts: Vec<&str> = spans.iter().map(|(s, e)| &text[*s..*e]).collect();
    assert_eq!(parts, ["It is white.<|im_end|>", "Black.<|im_end|>"], "each answer with its end-of-turn marker, nothing else");

    let ex = tok.encode(&rec, 512, false).unwrap();
    assert_eq!(ex.tokens.len(), ex.supervised.len());
    // the scored tokens, decoded, are exactly the two answers and their end markers (ids 0..2 are the special tokens)
    let scored: Vec<u32> = ex.tokens.iter().zip(&ex.supervised).filter(|(_, s)| **s).map(|(t, _)| *t).collect();
    assert_eq!(tok.decode(&scored).unwrap(), "It is white.<|im_end|>Black.<|im_end|>");
    assert_eq!(scored.iter().filter(|t| **t == 2).count(), 2, "both <|im_end|> tokens are scored");
    // byte-level: 12 + 1 and 6 + 1 tokens
    assert_eq!(ex.supervised_count(), 12 + 1 + 6 + 1);
    // every token is scored when asked (Soup's train_on_responses_only: false), except the first, which has no past
    let all = tok.encode(&rec, 512, true).unwrap();
    assert_eq!(all.supervised_count(), all.tokens.len() - 1);
}

#[test]
fn long_conversations_are_cut_and_hopeless_ones_left_out() {
    let (_root, model, _data) = common::model_and_data("cut");
    let tok = ChatTokenizer::load(&model, None).unwrap();
    let rec = |q: &str, a: &str| Record { messages: vec![("user".into(), q.into()), ("assistant".into(), a.into())] };
    let long_answer = rec("Hi", &"word ".repeat(100));
    let ex = tok.encode(&long_answer, 48, false).unwrap();
    assert_eq!(ex.tokens.len(), 48, "cut at the cap");
    assert!(ex.supervised_count() > 0);
    // a prompt longer than the cap leaves nothing to learn from: refused, and counted by `tokenize`
    let long_prompt = rec(&"word ".repeat(100), "Yes.");
    assert!(tok.encode(&long_prompt, 48, false).unwrap_err().to_string().contains("no assistant token is left"));
    let t = chatdata::tokenize(&tok, &[long_answer, long_prompt, rec("Hi", "Hello.")], 48, false);
    assert_eq!((t.examples.len(), t.skipped_count, t.truncated), (2, 1, 1));
}

#[test]
fn soups_row_shapes_become_messages() {
    let row = |v: serde_json::Value, f: DataFormat| chatdata::record_of(&v, f).map(|r| r.messages);
    let pair = |a: &str, b: &str| (a.to_string(), b.to_string());
    assert_eq!(row(serde_json::json!({"instruction": "Add.", "input": "1 and 2", "output": "3", "system": "Maths."}), DataFormat::Auto).unwrap(),
               vec![pair("system", "Maths."), pair("user", "Add.\n1 and 2"), pair("assistant", "3")]);
    assert_eq!(row(serde_json::json!({"instruction": "Say hi.", "output": "Hi."}), DataFormat::Alpaca).unwrap(), vec![pair("user", "Say hi."), pair("assistant", "Hi.")]);
    assert_eq!(row(serde_json::json!({"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]}), DataFormat::Auto).unwrap(),
               vec![pair("user", "Hi"), pair("assistant", "Hello")]);
    assert_eq!(row(serde_json::json!({"prompt": "Hi", "completion": "Hello"}), DataFormat::Auto).unwrap(), vec![pair("user", "Hi"), pair("assistant", "Hello")]);
    assert!(row(serde_json::json!({"messages": [{"role": "user", "content": "Hi"}]}), DataFormat::Auto).unwrap_err().to_string().contains("no assistant message"));
    assert!(row(serde_json::json!({"text": "plain"}), DataFormat::Auto).is_err());
    assert!(DataFormat::from_s("dpo").unwrap_err().to_string().contains("dpo"));
}

#[test]
fn the_schedule_is_hugging_faces_cosine_with_warmup() {
    // 2 warm-up steps of 10: 0, peak/2, then half a cosine from the peak to 0 over the 8 left
    let lr = |s| hf_cosine_lr(1.0, s, 2, 10);
    assert_eq!(lr(0), 0.0);
    assert_eq!(lr(1), 0.5);
    assert_eq!(lr(2), 1.0);
    assert!((lr(6) - 0.5).abs() < 1e-12, "half-way down");
    assert!((lr(9) - 0.5 * (1.0 + (std::f64::consts::PI * 7.0 / 8.0).cos())).abs() < 1e-12);
    // without warm-up the first step runs at the peak
    assert_eq!(hf_cosine_lr(3e-4, 0, 0, 10), 3e-4);
    // steps: ceil(ceil(n · epochs) / accumulation), capped
    assert_eq!(finetune::total_steps(21, 4.0, 2, None), 42);
    assert_eq!(finetune::total_steps(21, 1.0, 4, None), 6);
    assert_eq!(finetune::total_steps(10, 2.5, 4, None), 7);
    assert_eq!(finetune::total_steps(21, 4.0, 2, Some(5)), 5);
}

#[test]
fn training_lowers_the_loss_and_the_adapter_merges_to_the_same_logits() {
    let (root, model_dir, data) = common::model_and_data("train");
    let out = root.join("adapter");
    let s = settings(&model_dir, &data, &out);
    let mut events = Vec::new();
    let report = finetune::train(&s, &mut |e| events.push(e), &|| false).unwrap();

    // 24 conversations, an eighth held out; 21 · 4 epochs / 2 per step
    assert_eq!((report.train_examples, report.eval_examples, report.total_steps, report.steps_done), (21, 3, 42, 42));
    assert!(!report.stopped);
    let (first, last) = (report.losses[..4].iter().sum::<f64>() / 4.0, report.losses[38..].iter().sum::<f64>() / 4.0);
    println!("train loss {first:.3} → {last:.3}; held-out {:?}; {:.2} s", report.eval_loss, report.seconds);
    assert!(last < 0.75 * first, "the loss should fall well below its start: {first} → {last}");
    assert!(report.eval_loss.unwrap() < first, "the held-out loss is below the starting loss too");
    // every step was reported with its rate, and the rate followed the schedule
    let steps: Vec<(usize, f64)> = events.iter().filter_map(|e| match e { Event::Step { step, lr, .. } => Some((*step, *lr)), _ => None }).collect();
    assert_eq!(steps.len(), 42);
    assert_eq!(steps[0], (1, 0.0));
    assert!((steps[5].1 - s.lr).abs() < 1e-12, "5 warm-up steps (10 % of 42, rounded up), then the peak");

    // PEFT's two files, with PEFT's names
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out.join("adapter_config.json")).unwrap()).unwrap();
    assert_eq!((cfg["peft_type"].as_str(), cfg["r"].as_u64(), cfg["lora_alpha"].as_f64(), cfg["use_rslora"].as_bool()), (Some("LORA"), Some(8), Some(16.0), Some(false)));
    assert_eq!(cfg["target_modules"].as_array().unwrap().len(), 7);
    let tensors = candle_core::safetensors::load(out.join("adapter_model.safetensors"), &Device::Cpu).unwrap();
    assert_eq!(tensors.len(), 3 * 7 * 2);
    assert!(tensors.contains_key("base_model.model.model.layers.2.mlp.down_proj.lora_B.weight"));

    // the round trip: base + adapter (our forward) == the merged model (the Workshop's own merge), token for token
    let tokens: Vec<u32> = (0..20).map(|i| 3 + (i * 37 % 256) as u32).collect();
    let mut with_adapter = LlamaModel::load(&model_dir, BaseDtype::F32, &Device::Cpu).unwrap();
    let base_logits = with_adapter.forward(&tokens, &Ctx::eval()).unwrap();
    with_adapter.load_adapter(&out).unwrap();
    let adapter_logits = with_adapter.forward(&tokens, &Ctx::eval()).unwrap();
    assert!(max_abs(&base_logits, &adapter_logits) > 0.05, "the trained adapter changes the model");

    let merged_dir = root.join("merged");
    let merge_report = strata_workshop::merge::merge(&model_dir, &out, &merged_dir).unwrap();
    assert_eq!(merge_report.per_tensor.iter().filter(|t| t.delta_rel_norm > 0.0).count(), 3 * 7);
    let merged = LlamaModel::load(&merged_dir, BaseDtype::F32, &Device::Cpu).unwrap();
    let diff = max_abs(&merged.forward(&tokens, &Ctx::eval()).unwrap(), &adapter_logits);
    println!("merged vs base + LoRA: max abs logit difference {diff:.3e}");
    assert!(diff < 1e-4, "merged logits differ from base + LoRA by {diff}");
}

#[test]
fn a_stopped_run_resumes_to_exactly_the_same_adapter() {
    let (root, model_dir, data) = common::model_and_data("resume");
    let mut s = settings(&model_dir, &data, &root.join("straight"));
    s.max_steps = Some(8);
    s.lora.dropout = 0.1; // the dropout masks must repeat too
    let straight = finetune::train(&s, &mut |_| {}, &|| false).unwrap();

    // the same run, stopped after step 3 (as SIGTERM does) and continued from its checkpoint
    let mut s2 = s.clone();
    s2.out = root.join("stopped");
    let calls = Cell::new(0);
    let first = finetune::train(&s2, &mut |_| {}, &|| { calls.set(calls.get() + 1); calls.get() == 3 }).unwrap();
    assert!(first.stopped);
    assert_eq!(first.steps_done, 3);
    assert!(s2.out.join("checkpoint-3/optimizer.safetensors").exists() && s2.out.join("checkpoint-3/adapter_model.safetensors").exists());
    assert!(!s2.out.join("adapter_model.safetensors").exists(), "a stopped run leaves no finished adapter");
    s2.resume = Some(Resume::Auto);
    let second = finetune::train(&s2, &mut |_| {}, &|| false).unwrap();
    assert_eq!((second.steps_done, second.stopped), (8, false));
    assert_eq!(second.losses, straight.losses, "every step's loss, before and after the stop, is the uninterrupted run's");

    let load = |d: &std::path::Path| candle_core::safetensors::load(d.join("adapter_model.safetensors"), &Device::Cpu).unwrap();
    let (a, b) = (load(&s.out), load(&s2.out));
    assert_eq!(a.len(), b.len());
    for (name, t) in &a {
        assert_eq!(max_abs(t, &b[name]), 0.0, "{name} differs after the resume");
    }
    // a checkpoint is not continued with other settings
    s2.lr = 1e-3;
    s2.resume = Some(Resume::Dir(s2.out.join("checkpoint-3")));
    let e = finetune::train(&s2, &mut |_| {}, &|| false).unwrap_err().to_string();
    assert!(e.contains("other settings") && e.contains("lr"), "{e}");
}

#[test]
fn a_step_is_the_mean_over_its_conversations() {
    // at a vanishing rate the adapters stay put, so four steps of one conversation give the four starting losses, and
    // one step of four conversations (accumulation 4) reports their mean: the same conversations, in the same order
    let (root, model_dir, data) = common::model_and_data("accum");
    let mut one = settings(&model_dir, &data, &root.join("one"));
    one.val_split = 0.0;
    one.lr = 1e-12;
    one.warmup_ratio = 0.0;
    one.accumulation = 1;
    one.max_steps = Some(4);
    let singles = finetune::train(&one, &mut |_| {}, &|| false).unwrap();
    let mut four = one.clone();
    four.out = root.join("four");
    four.accumulation = 4;
    four.max_steps = Some(1);
    let together = finetune::train(&four, &mut |_| {}, &|| false).unwrap();
    assert_eq!((singles.losses.len(), together.losses.len()), (4, 1));
    assert_eq!(singles.tokens_seen, together.tokens_seen, "both runs read the same four conversations");
    let mean = singles.losses.iter().sum::<f64>() / 4.0;
    assert!((together.losses[0] - mean).abs() < 1e-5, "{} vs the mean {mean}", together.losses[0]);
    assert!(singles.losses.iter().any(|l| (l - mean).abs() > 1e-3), "the four losses are not all alike, so the mean means something");
}
