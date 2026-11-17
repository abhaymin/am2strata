//! Tokenizes each line of a JSON-lines file (one JSON string per line) with the tokenizer a GGUF carries, one line of ids
//! per input line, for comparison with llama.cpp's `llama-tokenize` (tools/tokenizer-parity.sh).
//!   cargo run --release --example tok_ids -- <model.gguf> <cases.jsonl> [--special]
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (model, lines) = (&a[1], &a[2]);
    let special = a.iter().any(|x| x == "--special");
    let g = strata_gguf::Gguf::open(std::path::Path::new(model)).expect("gguf");
    let dir = std::env::temp_dir().join(format!("tok-ids-{}", std::process::id()));
    strata_gguf::extract_tokenizer(&g, &dir).expect("extract");
    let tok = strata_tokenizer::Tokenizer::from_dir(dir.join("tokenizer")).expect("tokenizer");
    let text = std::fs::read_to_string(lines).expect("lines");
    let mut total = std::time::Duration::ZERO;
    let mut bytes = 0usize;
    for line in text.lines() {
        let s: String = serde_json::from_str(line).expect("a JSON string per line");
        let t0 = Instant::now();
        let ids = tok.encode(&s, special).expect("encode");
        total += t0.elapsed();
        bytes += s.len();
        println!("{}", ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","));
    }
    eprintln!("encoded {bytes} bytes in {:.3} ms ({:.1} MB/s)", total.as_secs_f64() * 1e3, bytes as f64 / total.as_secs_f64() / 1e6);
    let _ = std::fs::remove_dir_all(&dir);
}
