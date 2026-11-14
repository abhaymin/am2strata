//! Step 6, part C — reference check against llama.cpp's `convert_hf_to_gguf.py`.
//!
//! - [`save_tiny_checkpoint`]: writes the tiny HF folder (config.json + tokenizer*.json + F32 safetensors shards) that
//!   BOTH our converter and the reviewer's `convert_hf_to_gguf.py` read. It is deterministic (`TinyLlama::default`, seed
//!   0x5EED), so it can be regenerated any time; the files are committed.
//! - [`converted_matches_llama_convert_hf_to_gguf`]: runs our converter on that same folder and compares tensor names/
//!   shapes/types, every tensor's bytes (the q/k rotary permute is included — a mismatched permute changes those bytes),
//!   and the full metadata against llama.cpp's committed `tiny-llama.f32.gguf`. It is **skipped with a clear message**
//!   only when that reference file is absent, so nothing in `cargo test` needs Python or llama.cpp.

use std::collections::BTreeMap;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use strata_gguf::Gguf;
use strata_workshop::convert::{convert, OutType};
use strata_workshop::testmodel::TinyLlama as Fixture;

const HF_DIR: &str = "tiny-llama-hf";
const REF_GGUF: &str = "tiny-llama.f32.gguf";

fn fixtures() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures") }

/// Write the shared tiny HF checkpoint folder deterministically, so our converter and llama.cpp read byte-identical
/// inputs. Returns its path (the committed `tiny-llama.f32.gguf` was produced from this same folder by
/// `convert_hf_to_gguf.py`).
fn write_tiny_checkpoint() -> PathBuf {
    let fx = fixtures();
    std::fs::create_dir_all(&fx).ok();
    let hf = fx.join(HF_DIR);
    // deterministic: rewrite from scratch each run so the committed folder stays in sync with this crate.
    std::fs::remove_dir_all(&hf).ok();
    Fixture::default().write(&hf).expect("tiny fixture writes a HF checkpoint");

    hf
}

/// One tensor's raw data bytes out of a GGUF file, at `data_start + offset` for exactly its declared length.
fn tensor_bytes(path: &Path, data_start: u64, t: &strata_gguf::TensorInfo) -> Option<Vec<u8>> {
    let nbytes = t.expected_bytes().unwrap_or_else(|| t.elements() * 4) as usize;
    let mut r = BufReader::new(std::fs::File::open(path).expect("open gguf"));
    r.seek(SeekFrom::Start(data_start + t.offset)).ok()?;
    let mut b = vec![0u8; nbytes];
    r.read_exact(&mut b).ok()?;
    Some(b)
}

#[test]
fn save_tiny_checkpoint() {
    write_tiny_checkpoint();
}

// Byte-for-byte (per tensor) plus names/shapes/types and metadata, against `convert_hf_to_gguf.py`'s committed output.
// Skips the whole comparison when the reference GGUF is absent (see write_tiny_checkpoint for how to make it).
#[test]
fn converted_matches_llama_convert_hf_to_gguf() {
    let hf = write_tiny_checkpoint();
    let ref_path = fixtures().join(REF_GGUF);
    if !ref_path.exists() {
        eprintln!("[skip] {REF_GGUF} is not present under tests/fixtures/; generate it locally with \
                  `convert_hf_to_gguf.py` on the tiny HF folder and commit it to run the llama.cpp comparison.");
        return;
    }

    // our converter, on the SAME committed HF folder the reviewer's tool read.
    let out = std::env::temp_dir().join(format!("our-tiny-llama-f32-{}.gguf", std::process::id()));
    convert(&hf, &out, OutType::F32, Some("gpt-2")).expect("converter writes the tiny GGUF");

    let our = Gguf::open(&out).expect("our GGUF opens");
    let theirs = Gguf::open(&ref_path).expect("reference GGUF opens");

    // tensor count, and per-tensor names/shapes/types + bytes (compared by name — converters may emit them in different orders)
    assert_eq!(our.tensors.len(), theirs.tensors.len(), "tensor count differs");
    let our_by: BTreeMap<&str, _> = our.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

    for r in &theirs.tensors {
        let o = our_by.get(r.name.as_str())
            .unwrap_or_else(|| panic!("name {:?} present in the reference but missing from ours", r.name));

        assert_eq!(o.dims, r.dims, "shape/dims of {:?}", r.name);
        assert_eq!(o.ggml_type, r.ggml_type, "ggml type of {:?}", r.name);
        let (our_b, ref_b) = (tensor_bytes(&out, our.data_start, o), tensor_bytes(&ref_path, theirs.data_start, r));
        match (our_b, ref_b) {
            (Some(a), Some(b)) => assert_eq!(a, b, "tensor bytes differ from llama.cpp for {:?}", r.name),
            _ => panic!("could not read one or both copies of tensor {:?}", r.name),
        }
    }

    // every metadata key and value identical (gguf-py gives arrays as lists; map equality ignores insertion order)
    assert_eq!(our.metadata, theirs.metadata, "metadata keys/values differ from llama.cpp");

    std::fs::remove_file(&out).ok();
}
