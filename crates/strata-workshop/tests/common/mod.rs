//! Shared by the trainer's tests: a tiny random chat model and a small learnable dataset, written to a temp folder.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use strata_workshop::testmodel::{write_chat_model, TinyLlama, CHAT_VOCAB};

static N: AtomicUsize = AtomicUsize::new(0);

/// A fresh folder under the system temp dir (never in the repository).
pub fn temp_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("am2-trainer-{tag}-{}-{nanos}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A 3-block chat model: hidden 32, 4 heads over 2 key/value heads, byte-level tokenizer, ChatML template.
pub fn chat_model(dir: &Path) {
    chat_model_sized(dir, 3, 32, 64);
}

pub fn chat_model_sized(dir: &Path, layers: usize, hidden: usize, intermediate: usize) {
    let m = TinyLlama { seed: 0xC4A7, num_layers: layers, vocab_size: CHAT_VOCAB, hidden, intermediate, num_attention_heads: 4, num_key_value_heads: 2, shard_count: 1 };
    write_chat_model(dir, &m).unwrap();
}

pub const THINGS: [(&str, &str); 8] = [("sky", "blue"), ("grass", "green"), ("snow", "white"), ("coal", "black"), ("sun", "yellow"), ("rose", "red"),
                                       ("plum", "purple"), ("rust", "orange")];

/// `n` conversations with something to learn: the colour of a thing, asked in a few ways.
pub fn chat_rows(n: usize) -> Vec<serde_json::Value> {
    let asks = ["What colour is {}?", "Tell me the colour of {}.", "{}: which colour?"];
    (0..n).map(|i| {
        let (thing, colour) = THINGS[i % THINGS.len()];
        serde_json::json!({"messages": [
            {"role": "user", "content": asks[(i / THINGS.len()) % asks.len()].replace("{}", thing)},
            {"role": "assistant", "content": format!("It is {colour}.")},
        ]})
    }).collect()
}

pub fn write_jsonl(path: &Path, rows: &[serde_json::Value]) {
    let text: String = rows.iter().map(|r| format!("{r}\n")).collect();
    std::fs::write(path, text).unwrap();
}

/// A model folder and a 24-conversation dataset in one temp folder: `(root, model, data)`.
pub fn model_and_data(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
    let root = temp_dir(tag);
    let (model, data) = (root.join("model"), root.join("train.jsonl"));
    chat_model(&model);
    write_jsonl(&data, &chat_rows(24));
    (root, model, data)
}
