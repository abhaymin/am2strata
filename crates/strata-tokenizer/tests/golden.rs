//! The Rust tokenizer against ids recorded from PolyStrata's Python tokenizer (tools/reference-polystrata/make_golden.py).

use serde_json::Value;
use strata_tokenizer::{Detokenizer, Tokenizer};

fn golden(model: &str) -> Value {
    let path = format!("{}/../../golden/{model}.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).expect("golden file")).expect("golden json")
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn check(model: &str) {
    let g = golden(model);
    let Some(dir) = tokenizer_dir(g["tokenizer_dir"].as_str().unwrap()) else { return };
    let tok = Tokenizer::from_dir(&dir).expect("load tokenizer");
    let mut failures = Vec::new();
    for c in g["tokenizer_cases"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let ps = c["parse_special"].as_bool().unwrap();
        let got = tok.encode(text, ps).unwrap();
        if got != ids(&c["ids"]) {
            failures.push(format!("tokenizer case {text:?} parse_special={ps}"));
        }
    }
    for c in g["cases"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let want = ids(&c["ids"]);
        let got = tok.encode(text, true).unwrap();
        if got != want {
            let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
            failures.push(format!("template case {}: first difference at id {at} (got {} ids, want {})",
                                  c["name"], got.len(), want.len()));
        }
        assert_eq!(tok.decode(&want).unwrap(), text, "{model}/{}: decode", c["name"]);
        // incremental decode of the same ids gives the same text
        let mut d = Detokenizer::new();
        let streamed: String = want.iter().map(|t| d.push(&tok.token_bytes(*t).unwrap())).collect();
        assert_eq!(streamed, text, "{model}/{}: streamed decode", c["name"]);
    }
    assert!(failures.is_empty(), "{model}: {} mismatches:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn tiel_matches_python() {
    check("tiel");
}

#[test]
fn orca_matches_python() {
    check("orca");
}

/// The recorded tokenizer, under the Strata data folder ($STRATA_DATA, default ~/Development/Strata-data); None (the
/// test is skipped) where it is not installed.
fn tokenizer_dir(rel: &str) -> Option<String> {
    let base = std::env::var("STRATA_DATA").unwrap_or_else(|_| format!("{}/Development/Strata-data", std::env::var("HOME").unwrap_or_default()));
    let dir = format!("{base}/{rel}");
    if std::path::Path::new(&dir).is_dir() { Some(dir) } else { eprintln!("skipped: no tokenizer at {dir}"); None }
}
