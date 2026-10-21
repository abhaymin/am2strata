//! strata-pack against what PolyStrata's Python code produces (golden/pack/, recorded with
//! tools/reference-polystrata/make_golden_pack.py): every value and every file byte for byte.

use serde_json::Value;
use std::path::{Path, PathBuf};
use strata_pack::{iq_pack, mtp_pack, mtp_rt, num, quants};

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/pack")
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("strata-pack-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn every_dequantizer_matches_gguf_py() {
    for t in quants::SUPPORTED {
        let raw = std::fs::read(golden().join(format!("quants/{t}.in"))).unwrap();
        let want = f32s(&std::fs::read(golden().join(format!("quants/{t}.out"))).unwrap());
        let got = quants::dequantize(t, &raw).unwrap();
        assert_eq!(got.len(), want.len(), "{t}");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()), "{t} value {i}: {g} != {w}");
        }
    }
}

#[test]
fn numbers_match_numpy_and_python() {
    let n: Value = serde_json::from_str(&std::fs::read_to_string(golden().join("numbers.json")).unwrap()).unwrap();
    let bits = |k: &str| n[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect::<Vec<_>>();
    let (f, h, bf) = (bits("f32_bits"), bits("f16_bits"), bits("bf16_bits"));
    for i in 0..f.len() {
        let x = f32::from_bits(f[i] as u32);
        assert_eq!(num::f32_to_f16(x) as u64, h[i], "f16 of {x:e}");
        assert_eq!(num::f32_to_bf16(x) as u64, bf[i], "bf16 of {x:e}");
    }
    let hexf = |s: &str| -> f64 {
        // float.hex(): [-]0x1.<hex>p<exp> or 0x0.0p+0
        let neg = s.starts_with('-');
        let s = s.trim_start_matches('-').trim_start_matches("0x");
        let (m, e) = s.split_once('p').unwrap();
        let (ip, fp) = m.split_once('.').unwrap_or((m, ""));
        // the mantissa as one exact integer (at most 53 bits), then scaled by a power of two: no rounding
        let mant = u64::from_str_radix(&format!("{ip}{fp}"), 16).unwrap() as f64;
        let v = mant * 2f64.powi(e.parse::<i32>().unwrap() - 4 * fp.len() as i32);
        if neg { -v } else { v }
    };
    let doubles = n["doubles"].as_array().unwrap();
    for (i, d) in doubles.iter().enumerate() {
        let x = hexf(d.as_str().unwrap());
        assert_eq!(num::py_repr(x), n["repr"][i].as_str().unwrap(), "repr of {x:e}");
        assert_eq!(num::py_repr(num::py_round(x, 5)), n["round5"][i].as_str().unwrap(), "round(x, 5) of {x:e}");
    }
}

#[test]
fn quantizers_match_mtp_pack_and_mtp_rt() {
    let d = golden().join("quantizers");
    let x = f32s(&std::fs::read(d.join("x.f32")).unwrap());
    for (kind, f) in [("q2_0", mtp_pack::q2_0 as fn(&[f32], &mut Vec<u8>)), ("q4_0", mtp_pack::q4_0), ("q8_0", mtp_pack::q8_0)] {
        let mut got = Vec::new();
        f(&x, &mut got);
        assert!(got == std::fs::read(d.join(format!("{kind}.bin"))).unwrap(), "mtp_pack {kind}");
    }
    assert!(mtp_rt::q8_0(&x) == std::fs::read(d.join("rt_q8_0.bin")).unwrap(), "mtp_rt q8_0");
}

fn same_file(a: &Path, b: &Path) {
    let (x, y) = (std::fs::read(a).unwrap(), std::fs::read(b).unwrap());
    assert!(x == y, "{} differs from {} ({} vs {} bytes)", a.display(), b.display(), x.len(), y.len());
}

#[test]
fn iq_pack_writes_the_python_pack() {
    let src = golden().join("tiny/tiny-00001-of-00002.gguf").canonicalize().unwrap();
    let out = tmp("iq");
    std::fs::create_dir_all(out.join("tokenizer")).unwrap();
    for f in ["vocab.json", "chat_template.jinja"] {
        std::fs::write(out.join("tokenizer").join(f), "{}\n").unwrap();
    }
    iq_pack::run(&iq_pack::Options { gguf: src.clone(), out: out.clone(), base: None, skip_experts: false, compat_bf16: true,
                                     experts_bin: true }).unwrap();
    let want = golden().join("tiny/pack");
    for f in ["index.txt", "native_experts.txt", "conversions.json", "dense.bin", "experts.bin", "experts.bin.src.json"] {
        same_file(&out.join(f), &want.join(f));
    }
    let got = std::fs::read_to_string(out.join("compat-bf16.json")).unwrap().replace(&src.display().to_string(), "@SRC@");
    assert_eq!(got, std::fs::read_to_string(want.join("compat-bf16.json")).unwrap());
    // without --compat-bf16 the quantized projections are refused, and nothing of the pack changes
    let out2 = tmp("iq-refuse");
    let e = iq_pack::run(&iq_pack::Options { gguf: src, out: out2.clone(), base: None, skip_experts: false, compat_bf16: false,
                                             experts_bin: false }).unwrap_err();
    assert!(e.contains("use --compat-bf16"), "{e}");
    assert!(!out2.join("index.txt").exists() && !out2.join("native_experts.txt").exists());
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&out2);
}

#[test]
fn mtp_pack_writes_the_python_gguf() {
    let d = golden().join("tiny-mtp");
    let out = tmp("mtp");
    for kind in ["q2_0", "q4_0", "q8_0"] {
        let gguf = out.join(format!("mtp-{kind}.gguf"));
        mtp_pack::run(&mtp_pack::Options { src: d.join("src"), experts: kind.into(), out: gguf.clone(), check_experts: 3 }).unwrap();
        same_file(&gguf, &d.join(format!("mtp-{kind}.gguf")));
        // the report, but its "seconds" (a timing)
        let strip = |p: PathBuf| {
            let mut v: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
            for t in v["tensors"].as_array_mut().unwrap() {
                t.as_object_mut().unwrap().remove("seconds");
            }
            v
        };
        assert_eq!(strip(out.join(format!("mtp-{kind}.gguf.report.json"))), strip(d.join(format!("mtp-{kind}.gguf.report.json"))), "{kind}");
    }
    let _ = std::fs::remove_dir_all(&out);
}
