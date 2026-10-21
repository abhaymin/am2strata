//! strata-pack against the files PolyStrata's Python tools wrote for the installed Orca model (ignored by default:
//! they need the model, ~85 GB, and the MTP data). `cargo test -p strata-pack --release --test installed -- --ignored`

use std::path::{Path, PathBuf};
use strata_pack::{iq_pack, mtp_pack, mtp_rt};

fn data() -> PathBuf {
    std::env::var("STRATA_DATA").map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(format!("{}/Development/Strata-data", std::env::var("HOME").unwrap())))
}

fn scratch(tag: &str) -> PathBuf {
    let d = data().join(format!("port-test-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn same(a: &Path, b: &Path) {
    let (x, y) = (std::fs::read(a).unwrap(), std::fs::read(b).unwrap());
    assert!(x == y, "{} differs from {}", a.display(), b.display());
}

#[test]
#[ignore]
fn orca_pack_is_the_python_pack() {
    let model = data().join("models/orca-IQ3_XXS/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf");
    let want = data().join("packs/orca-iq3_xxs");
    let out = scratch("orca");
    iq_pack::run(&iq_pack::Options { gguf: model, out: out.clone(), base: None, skip_experts: false, compat_bf16: true, experts_bin: false })
        .unwrap();
    for f in ["index.txt", "native_experts.txt", "conversions.json", "compat-bf16.json", "dense.bin"] {
        same(&out.join(f), &want.join(f));
    }
    for f in ["vocab.json", "merges.txt", "token_type.json", "chat_template.jinja"] {
        same(&out.join("tokenizer").join(f), &want.join("tokenizer").join(f));
    }
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
#[ignore]
fn mtp_files_are_the_python_files() {
    let mtp = data().join("mtp");
    let out = scratch("mtp");
    let gguf = out.join("mtp-q2_0.gguf");
    mtp_pack::run(&mtp_pack::Options { src: mtp.clone(), experts: "q2_0".into(), out: gguf.clone(), check_experts: 8 }).unwrap();
    same(&gguf, &mtp.join("mtp-q2_0.gguf"));
    mtp_rt::run(&mtp_rt::Options { gguf, out: out.join("rt") }).unwrap();
    for f in ["experts.bin", "dense.bin", "dense.txt"] {
        same(&out.join("rt").join(f), &mtp.join("rt").join(f));
    }
    std::fs::remove_dir_all(&out).unwrap();
}
