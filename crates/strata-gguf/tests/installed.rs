//! The Rust extractor against the tokenizer folders PolyStrata's Python wrote for the installed models (ignored by
//! default: it needs the model files). `cargo test -p strata-gguf --release -- --ignored`

use serde_json::Value;
use std::path::Path;
use strata_gguf::{extract_tokenizer, Gguf};

fn data() -> String {
    format!("{}/Development/Strata-data", std::env::var("HOME").unwrap())
}

fn check(gguf: &str, pack: &str) {
    let g = Gguf::open(Path::new(&format!("{}/models/{gguf}", data()))).unwrap();
    let out = std::env::temp_dir().join(format!("am2strata-extract-{}", pack));
    let _ = std::fs::remove_dir_all(&out);
    extract_tokenizer(&g, &out).unwrap();
    let want = Path::new(&data()).join("packs").join(pack).join("tokenizer");
    let got = out.join("tokenizer");
    for f in ["vocab.json", "merges.txt", "token_type.json", "chat_template.jinja"] {
        let (a, b) = (std::fs::read(want.join(f)).unwrap(), std::fs::read(got.join(f)).unwrap());
        assert!(a == b, "{pack}/{f} differs ({} vs {} bytes)", a.len(), b.len());
    }
    let w: Value = serde_json::from_str(&std::fs::read_to_string(want.join("tokenizer.json")).unwrap()).unwrap();
    let r: Value = serde_json::from_str(&std::fs::read_to_string(got.join("tokenizer.json")).unwrap()).unwrap();
    for k in ["model", "pre", "vocab_size", "n_merges", "special_ids", "add_bos_token", "ignore_merges"] {
        assert_eq!(w[k], r[k], "{pack}: tokenizer.json {k}");
    }
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
#[ignore]
fn tiel_tokenizer_matches_python() {
    check("tiel-UD-Q6_K_XL/Tiel-Coder-35B-A3B-MTP-UD-Q6_K_XL.gguf", "tiel-ud-q6_k_xl");
}

#[test]
#[ignore]
fn minicpm_tokenizer_matches_python() {
    check("minicpmv-uncensored-AD-Q5_K-Q4_K/MiniCPM-V-4.7-35B-A3B-Uncensored-AD-Q5_K-Q4_K-00001-of-00012.gguf",
          "minicpmv-uncensored-ad-q5_k-q4_k");
}
