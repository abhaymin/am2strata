//! Against llama.cpp (tools/reference-llamacpp: the oracle built against a patched copy of PolyStrata's llama.cpp),
//! on the models installed on this machine; ignored by default (they need the models and the oracle):
//!   STRATA_MODELS="a.gguf:b.gguf" cargo test --release --manifest-path crates/strata-media/Cargo.toml \
//!       -p strata-hybrid -- --ignored --nocapture
//! - tokenizer: the tokenizer am2Strata extracts from each GGUF gives llama.cpp's ids for a set of texts;
//! - forward pass: tools/reference-llamacpp/parity.sh (greedy decoding with teacher forcing, near-ties allowed).

use std::path::{Path, PathBuf};
use std::process::Command;

fn oracle() -> PathBuf {
    std::env::var_os("ORACLE").map(PathBuf::from).unwrap_or_else(|| {
        let data = std::env::var_os("STRATA_DATA").map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join("Development/Strata-data"));
        data.join("oracle/oracle2")
    })
}

fn models() -> Vec<PathBuf> {
    std::env::var("STRATA_MODELS").map(|v| v.split(':').filter(|s| !s.is_empty()).map(PathBuf::from).collect()).unwrap_or_default()
}

fn llamacpp_ids(model: &Path, text: &str) -> Vec<u32> {
    let out = Command::new(oracle()).arg(model).arg("-1").arg(format!("text:{text}")).output().expect("run the oracle");
    let err = String::from_utf8_lossy(&out.stderr);
    let line = err.lines().find_map(|l| l.find("IDS ").map(|i| &l[i + 4..])).unwrap_or_else(|| panic!("oracle failed: {err}"));
    line.trim().split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect()
}

const TEXTS: &[&str] = &[
    "Hello world",
    "The quick brown fox jumps over the lazy dog. It's 12345 o'clock!",
    "  leading spaces,\ttabs\n\nand newlines\r\n",
    "fn main() { println!(\"{}\", 0x1F_u32 >> 2); } // café naïve résumé",
    "数学は難しいですか？ Привет, мир! مرحبا 😀👍🏽",
    "UPPERCase lowerCASE MiXeD CamelCaseWords iPhone McDonald's",
    "1234567890 3.14159 1,000,000 $42.00",
];

#[test]
#[ignore]
fn tokenizer_matches_llamacpp() {
    let ms = models();
    assert!(!ms.is_empty(), "set STRATA_MODELS");
    for m in &ms {
        let g = strata_gguf::Gguf::open(m).unwrap();
        let dir = std::env::temp_dir().join(format!("am2-tok-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        strata_gguf::extract_tokenizer(&g, &dir).unwrap();
        let tok = strata_tokenizer::Tokenizer::from_dir(dir.join("tokenizer")).unwrap();
        let bos = g.u64("tokenizer.ggml.bos_token_id").map(|v| v as u32);
        let add_bos = g.metadata.get("tokenizer.ggml.add_bos_token").and_then(|v| v.as_bool()).unwrap_or(false);
        let mut bad = 0;
        for t in TEXTS {
            let mut want = llamacpp_ids(m, t);
            if add_bos && want.first().copied() == bos {
                want.remove(0);
            }
            let got = tok.encode(t, true).unwrap();
            if got != want {
                bad += 1;
                eprintln!("{}: {t:?}\n  llama.cpp {want:?}\n  am2strata {got:?}", m.display());
            }
        }
        eprintln!("{} ({}): {}/{} texts identical", m.file_name().unwrap().to_string_lossy(), tok.pre(), TEXTS.len() - bad, TEXTS.len());
        assert_eq!(bad, 0, "{}", m.display());
    }
}

#[test]
#[ignore]
fn forward_matches_llamacpp() {
    let ms = models();
    assert!(!ms.is_empty(), "set STRATA_MODELS");
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/reference-llamacpp/parity.sh");
    for m in &ms {
        let st = Command::new("bash").arg(&script).arg(m).arg("16").arg("Question: Why is the sky blue? Answer:").status().unwrap();
        assert!(st.success(), "{}", m.display());
    }
}
