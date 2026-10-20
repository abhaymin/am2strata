//! The gemma4 (UTF-8 pieces, byte fallback) and deepseek-v3 (pattern sequence) pre-tokenizers against ids recorded
//! from PolyStrata's Python tokenizer on small synthetic vocabularies (tools/reference-polystrata/make_golden_tokenizers.py).

use serde_json::Value;
use strata_tokenizer::Tokenizer;

fn check(name: &str) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata").join(name);
    let tok = Tokenizer::from_dir(&dir).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(dir.join("golden.json")).unwrap()).unwrap();
    for c in &cases {
        let text = c["text"].as_str().unwrap();
        let ps = c["parse_special"].as_bool().unwrap();
        let want: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(tok.encode(text, ps).unwrap(), want, "{name}: {text:?} parse_special={ps}");
        if ps {
            assert_eq!(tok.decode(&want).unwrap(), text, "{name}: decodes back");
        }
    }
}

#[test]
fn gemma4_matches_python() {
    check("tokenizer-gemma4-mini");
}

#[test]
fn deepseek_sequence_matches_python() {
    check("tokenizer-deepseek-mini");
}
