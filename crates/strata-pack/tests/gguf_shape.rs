//! The writer <-> reader shape convention, pinned once so the GGUF converter (step 6) has a fixed contract: `gguf_write`
//! takes `TensorSpec.shape` in numpy order (outermost/rows first) and rewrites it into the file's ggml dimension order
//! (innermost last). A [rows=2, cols=256] tensor therefore round-trips so that `Gguf::open` reports ne = [256, 2].

use strata_pack::{gguf_write::{TensorSpec, Writer}};

#[test]
fn numpy_shape_read_back_as_ggml_ne() {

    // rows=2, cols=256 in numpy order; F32, so this many bytes of data.
    let (rows, cols) = (2usize, 256usize);
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let specs = vec![TensorSpec { name: "weight".into(), shape: vec![rows as u64, cols as u64], ggml_type: 0 /* F32 */, nbytes: raw.len() as u64 }];
    let kv = [("general.architecture", "shape-test".to_string())];
    let path = std::env::temp_dir().join(format!("shape-convention-{}.gguf", std::process::id()));
    let mut w = Writer::create(&path, &kv, specs).unwrap();
    w.tensor_data(&raw).unwrap();
    w.end_tensor().unwrap();
    w.finish().unwrap();

    // ne (ggml dimension order) is the reverse of numpy shape: [256, 2]. This is what the converter must preserve.
    let g = strata_gguf::Gguf::open(&path).unwrap();
    let t = &g.tensors[0];
    assert_eq!(t.dims, vec![256, 2], "ggml ne of a [rows=2, cols=256] tensor must be [256, 2]");
    println!("numpy [2, 256] read back as ggml ne = {:?}", t.dims);
}
