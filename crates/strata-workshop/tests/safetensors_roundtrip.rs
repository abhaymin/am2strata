//! Step 0 gate: tiny random safetensors writes and round-trips read (reading back via the safetensors crate here; the
//! workshop's own memmap2 reader comes in at step 1).

use std::collections::HashMap;
use std::path::PathBuf;

use safetensors::tensor::{Dtype, SafeTensors};
use strata_workshop::testmodel::TinyLlama;

/// Expected tensor keyed by name: dtype string, shape, and the exact f32 values that were written.
struct Expected {
    dtype: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn build_expected(model: &TinyLlama) -> HashMap<String, Expected> {
    let mut out = HashMap::new();
    for t in model.generate() {
        out.insert(
            t.name.clone(),
            Expected {
                dtype: Dtype::F32.to_string(),
                shape: t.shape.clone(),
                values: t.values,
            },
        );
    }
    out
}

fn temp_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("strata_workshop_{tag}_{nanos}"))
}

#[test]
fn tiny_safetensors_writes_and_round_trips() -> Result<(), Box<dyn std::error::Error>> {
    let model = TinyLlama::default();
    let dir = temp_dir("roundtrip");
    std::fs::remove_dir_all(&dir).ok();
    let written = model.write(&dir)?;

    // Read the single shard back with the safetensors crate (this is step 0's round-trip read); step 1 re-reads these
    // files with the workshop's own memmap2 reader.
    let bytes = std::fs::read(written.shard_files.first().expect("at least one shard"))?;
    let loaded = SafeTensors::deserialize(&bytes)?;

    let expected = build_expected(&model);
    assert_eq!(loaded.len() as usize, expected.len(), "tensor count mismatch");

    let mut checked = 0usize;
    for (name, view) in loaded.tensors() {
        let exp = expected.get(name.as_str()).unwrap_or_else(|| panic!("unexpected tensor {name}"));
        assert_eq!(view.dtype().to_string(), exp.dtype, "{name}: dtype mismatch");
        assert_eq!(view.shape(), exp.shape.as_slice(), "{name}: shape mismatch");

        // f32 is 4 little-endian bytes: decoded (a byte slice need not be aligned for f32)
        let got: Vec<f32> = view.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(got, exp.values, "{name}: values mismatch");
        checked += 1;
    }
    assert_eq!(checked, expected.len(), "not every tensor was read back");

    println!("tiny safetensors round-trip: {checked} tensors wrote and read back exactly");
    Ok(())
}
