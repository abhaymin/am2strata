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
    same_as_llama_cpp("tiny-llama-256-hf", "tiny-llama-256.f16.gguf");
}

/// A BF16 checkpoint (how Hugging Face models are stored) widened exactly and written as F16, as llama.cpp does.
#[test]
fn a_bf16_checkpoint_converts_as_llama_cpp_does() {
    same_as_llama_cpp("tiny-llama-256-bf16-hf", "tiny-llama-256-bf16.f16.gguf");
}

fn same_as_llama_cpp(hf: &str, reference: &str) {
    let reference = fixtures().join(reference);
    let out = std::env::temp_dir().join(format!("am2-f16-{hf}-{}.gguf", std::process::id()));
    strata_workshop::convert::convert(&fixtures().join(hf), &out, strata_workshop::convert::OutType::F16, Some("gpt-2")).unwrap();
    let (ours, theirs) = (strata_gguf::Gguf::open(&out).unwrap(), strata_gguf::Gguf::open(&reference).unwrap());
    assert_eq!(ours.tensors.len(), theirs.tensors.len());
    for r in &theirs.tensors {
        let o = ours.tensors.iter().find(|t| t.name == r.name).unwrap_or_else(|| panic!("missing {}", r.name));
        assert_eq!((o.ggml_type, &o.dims), (r.ggml_type, &r.dims), "{}: type/shape", r.name);
        assert!(bytes_of(&out, &ours, o) == bytes_of(&reference, &theirs, r), "{}: bytes differ from llama.cpp", r.name);
    }
    // Every metadata key llama.cpp wrote must match ours exactly now that general.basename / .finetune / .version are
    // ported from gguf-py's get_model_id_components (the previous "guessed"-exception no longer applies).
    for (k, v) in &theirs.metadata {
        assert_eq!(ours.metadata.get(k), Some(v), "metadata {k}");
    }
    let _ = std::fs::remove_file(&out);
}
