//! `--outtype f16` against llama.cpp's own F16 output for the wide tiny model (`tiny-llama-256.f16.gguf`, made by the
//! reviewer with convert_hf_to_gguf.py, pre forced to gpt-2 as for the F32 references): 2-D weights in F16 rounded as
//! numpy does, 1-D tensors (norms) kept in F32, every tensor's bytes and the metadata the same.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures") }

fn bytes_of(path: &Path, g: &strata_gguf::Gguf, t: &strata_gguf::TensorInfo) -> Vec<u8> {
    let mut f = std::fs::File::open(path).unwrap();
    f.seek(SeekFrom::Start(g.data_start + t.offset)).unwrap();
    let mut b = vec![0u8; t.expected_bytes().expect("a known type") as usize];
    f.read_exact(&mut b).unwrap();
    b
}

#[test]
fn f16_conversion_matches_llama_cpp() {
    let reference = fixtures().join("tiny-llama-256.f16.gguf");
    let out = std::env::temp_dir().join(format!("am2-f16-{}.gguf", std::process::id()));
    strata_workshop::convert::convert(&fixtures().join("tiny-llama-256-hf"), &out, strata_workshop::convert::OutType::F16, Some("gpt-2")).unwrap();
    let (ours, theirs) = (strata_gguf::Gguf::open(&out).unwrap(), strata_gguf::Gguf::open(&reference).unwrap());
    assert_eq!(ours.tensors.len(), theirs.tensors.len());
    for r in &theirs.tensors {
        let o = ours.tensors.iter().find(|t| t.name == r.name).unwrap_or_else(|| panic!("missing {}", r.name));
        assert_eq!((o.ggml_type, &o.dims), (r.ggml_type, &r.dims), "{}: type/shape", r.name);
        assert!(bytes_of(&out, &ours, o) == bytes_of(&reference, &theirs, r), "{}: bytes differ from llama.cpp", r.name);
    }
    for (k, v) in &theirs.metadata {
        assert_eq!(ours.metadata.get(k), Some(v), "metadata {k}");
    }
    assert_eq!(std::fs::metadata(&out).unwrap().len(), std::fs::metadata(&reference).unwrap().len(), "file size");
    let _ = std::fs::remove_file(&out);
}
