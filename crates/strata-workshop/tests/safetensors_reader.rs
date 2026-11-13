//! Step 1: the memmap2 safetensors reader — shard enumeration in order, a full round-trip read across shards, and that a
//! truncated shard is rejected (offsets checked against the real file size). Values come from the seed-fixed `TinyLlama`
//! fixture written at step 0, so reading reproduces exactly what was written.

use std::collections::HashMap;

use strata_workshop::safetensors_io::{self, Shard};
use strata_workshop::testmodel::TinyLlama;

struct Expected {
    dtype: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

/// Every tensor the fixture writes, keyed by name — the reader must return exactly these.
fn expected_all(model: &TinyLlama) -> HashMap<String, Expected> {
    model.generate().into_iter().map(|t| {
        (
            t.name.clone(),
            Expected { dtype: "F32".to_string(), shape: t.shape.clone(), values: t.values },
        )
    })
    .collect()
}

/// Read every tensor from every shard, keyed by name, into owned f32 values (tiny fixture; a real merge streams bytes).
fn read_all(shard_files: &[std::path::PathBuf]) -> Result<HashMap<String, Expected>, Box<dyn std::error::Error>> {
    let mut out = HashMap::new();
    for path in shard_files {
        let shard = Shard::open(path)?;
        assert!(shard.len() >= 1, "{path:?}: empty shard");
        // `desc` borrows the shard immutably; mutating `out` is a separate borrow, so one pass suffices.
        for desc in shard.tensors() {
            let bytes = shard.bytes(&desc);
            assert_eq!(bytes.len() % 4, 0, "{}: f32 tensor is not a whole number of floats", desc.name());
            let got = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }.to_vec();
            // the first occurrence wins; if it was already present the name duplicated across shards
            let e = out.entry(desc.name().to_string()).or_insert_with(|| Expected { dtype: "F32".into(), shape: Vec::new(), values: vec![] });
            assert!(e.shape.is_empty(), "{} appears twice", desc.name());
            e.dtype = desc.dtype().as_str().to_string();
            e.shape = desc.shape().to_vec();
            e.values = got;
        }
    }
    Ok(out)
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("strata_workshop_{tag}_{nanos}"))
}

#[test]
fn shards_are_listed_in_order_and_read_back_across_them() -> Result<(), Box<dyn std::error::Error>> {
    let model = TinyLlama { shard_count: 2, ..Default::default() };
    let dir = temp_dir("reader");
    std::fs::remove_dir_all(&dir).ok();
    let written = model.write(&dir)?;

    // Enumeration from the first shard reproduces every shard in order.
    let enumerated = safetensors_io::shards(&written.shard_files[0]);
    assert_eq!(enumerated.len(), 2);
    assert_eq!(enumerated, written.shard_files, "shard enumeration did not match the written shards");

    // Round-trip: every tensor read across both shards equals what the seed generated.
    let got = read_all(&written.shard_files)?;
    let want = expected_all(&model);
    assert_eq!(got.len(), want.len(), "tensor count mismatch: {} vs {}", got.len(), want.len());

    for (name, e) in &want {
        let g = got.get(name).unwrap_or_else(|| panic!("{name}: missing from the read-back shards"));
        assert_eq!(g.dtype, e.dtype, "{name}: dtype {}/{}", g.dtype, e.dtype);
        assert_eq!(g.shape, e.shape, "{name}: shape {:?}/{:?}", g.shape, e.shape);
        assert_eq!(g.values, e.values, "{name}: values differ from what was written");
    }

    println!("safetensors reader: {} tensors read across {} shards exactly", got.len(), written.shard_files.len());
    Ok(())
}

#[test]
fn truncated_shard_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
    let model = TinyLlama { shard_count: 2, ..Default::default() };
    let dir = temp_dir("truncate");
    std::fs::remove_dir_all(&dir).ok();
    let written = model.write(&dir)?;

    // Drop the last few bytes of a shard: the final tensor's declared span then runs past the file.
    let mut bytes = std::fs::read(&written.shard_files[0])?;
    let keep = bytes.len().saturating_sub(16);
    bytes.truncate(keep);
    let corrupt = dir.join("model-00001-of-00002.safetensors");
    std::fs::write(&corrupt, &bytes)?;

    let err = Shard::open(&corrupt).expect_err("a shard with data past EOF should fail to open");
    assert!(err.to_string().contains("file is"), "expected a file-size error, got: {err}");
    Ok(())
}

#[test]
fn single_shard_file_reads_without_enumeration() -> Result<(), Box<dyn std::error::Error>> {
    let model = TinyLlama { shard_count: 1, ..Default::default() };
    let dir = temp_dir("single");
    std::fs::remove_dir_all(&dir).ok();
    let written = model.write(&dir)?;

    // One shard: enumeration yields just that file.
    assert_eq!(safetensors_io::shards(&written.shard_files[0]).len(), 1);
    let got = read_all(&written.shard_files)?;
    assert!(!got.is_empty());
    println!("single-shard safetensors read: {} tensors", got.len());
    Ok(())
}
