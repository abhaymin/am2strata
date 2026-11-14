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

/// The shared tiny HF checkpoint, written by this crate into a fresh temporary folder (never into the committed
/// fixtures: tests run in parallel, and another one reads that folder). Its files must equal the committed
/// `tests/fixtures/tiny-llama-hf/`, from which llama.cpp made `tiny-llama.f32.gguf` (checked by `save_tiny_checkpoint`).
fn write_tiny_checkpoint(tag: &str) -> PathBuf {
    let hf = std::env::temp_dir().join(format!("am2-tiny-llama-hf-{tag}-{}", std::process::id()));
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

/// The generator still writes exactly the committed tiny folder (if it drifted, the llama.cpp reference no longer
/// matches what our converter reads). To update the fixture deliberately: copy this test's folder over it.
#[test]
fn save_tiny_checkpoint() {
    let hf = write_tiny_checkpoint("check");
    let committed = fixtures().join(HF_DIR);
    let mut names: Vec<String> = std::fs::read_dir(&committed).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    for n in &names {
        let (a, b) = (std::fs::read(hf.join(n)).unwrap_or_default(), std::fs::read(committed.join(n)).unwrap());
        if n.ends_with(".json") {
            let (a, b): (serde_json::Value, serde_json::Value) = (serde_json::from_slice(&a).unwrap(), serde_json::from_slice(&b).unwrap());
            assert_eq!(a, b, "{n}: the generator and the committed fixture differ");
        } else {
            assert!(a == b, "{n}: the generator and the committed fixture differ");
        }
    }
    let _ = std::fs::remove_dir_all(&hf);
}

// Byte-for-byte (per tensor) plus names/shapes/types and metadata, against `convert_hf_to_gguf.py`'s committed output.
// Skips the whole comparison when the reference GGUF is absent (see write_tiny_checkpoint for how to make it).
#[test]
fn converted_matches_llama_convert_hf_to_gguf() {
    // the committed folder, exactly what llama.cpp read (only read here; save_tiny_checkpoint keeps it in sync)
    let hf = fixtures().join(HF_DIR);
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
