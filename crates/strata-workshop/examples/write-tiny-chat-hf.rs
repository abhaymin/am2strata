//! Writes a random chat model (byte-level tokenizer, ChatML template) and a small dataset, for trying the trainer and
//! for the measurements in docs/DEPLOYMENT.md — nothing is downloaded:
//!
//!     cargo run --release -p strata-workshop --example write-tiny-chat-hf -- <out-dir> [layers] [hidden] [intermediate] [conversations]
//!
//! `<out-dir>/model` is the Hugging Face folder, `<out-dir>/train.jsonl` the conversations.

use std::path::PathBuf;

use strata_workshop::testmodel::{write_chat_model, TinyLlama, CHAT_VOCAB};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(args.first().ok_or_else(|| anyhow::anyhow!("usage: write-tiny-chat-hf <out-dir> [layers] [hidden] [intermediate] [conversations]"))?);
    let n = |i: usize, default: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(default);
    let (layers, hidden, intermediate, rows) = (n(1, 3), n(2, 32), n(3, 64), n(4, 24));
    let model = TinyLlama { seed: 0xC4A7, num_layers: layers, vocab_size: CHAT_VOCAB, hidden, intermediate, num_attention_heads: 4, num_key_value_heads: 2, shard_count: 1 };
    write_chat_model(&out.join("model"), &model)?;
    let things = [("sky", "blue"), ("grass", "green"), ("snow", "white"), ("coal", "black"), ("sun", "yellow"), ("rose", "red"), ("plum", "purple"), ("rust", "orange")];
    let asks = ["What colour is {}?", "Tell me the colour of {}.", "{}: which colour?"];
    let text: String = (0..rows).map(|i| {
        let (thing, colour) = things[i % things.len()];
        let ask = asks[(i / things.len()) % asks.len()].replace("{}", thing);
        format!("{}\n", serde_json::json!({"messages": [{"role": "user", "content": ask}, {"role": "assistant", "content": format!("It is {colour}.")}]}))
    }).collect();
    std::fs::write(out.join("train.jsonl"), text)?;
    println!("wrote {} ({layers} blocks, hidden {hidden}) and {rows} conversations", out.join("model").display());
    Ok(())
}
