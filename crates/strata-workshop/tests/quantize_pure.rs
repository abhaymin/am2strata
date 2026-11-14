//! Step 7: `strata_gguf` GGUF -> a new GGUF where every tensor is a chosen `--pure` precision, following llama.cpp's
//! `llama-quantize --pure` fallback rules (see [crate::quantize_gguf]). Two gates pin it:
//!
//! 1. [`tiny_matches_pure_reference`]: the tiny model (`tests/fixtures/tiny-llama.f32.gguf`, llama.cpp's own F32 output)
//!    quantized to F16/Q8_0/Q4_K must reproduce `tests/fixtures/tiny-llama.{F16,Q8_0,Q4_K}-pure.gguf` — same tensor names,
//!    types, shapes and bytes, and the same metadata. These fixtures exercise the shape FALLBACK (16-wide rows: only the
//!    single 32-square weight is quantized, everything else F16), plus the block-32/256 demotion Q4_K -> Q5_0.
//!
//! 2. [`wide_matches_pure_reference`]: a wide tiny model (hidden 256, so every row holds whole blocks and no fallback)
//!    quantized to Q8_0/Q4_0/Q4_K/Q5_K/Q6_K must reproduce the reviewer's `llama-quantize --pure` output. Those files are
//!    made once by the reviewer (`tests/fixtures/NEEDED-quant.md`) and each comparison is skipped until it exists, so the
//!    block quantizers themselves are byte-checked against llama.cpp rather than only the tiny fallback path.

use std::collections::BTreeMap;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use strata_gguf::Gguf;
use strata_workshop::convert::{convert, OutType};
use strata_workshop::quantize_gguf::{quantize, values_equal, PureType};
use strata_workshop::testmodel::TinyLlama as Fixture;

fn fixtures() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures") }

/// A fresh temp dir, unique per call so the three targets never collide.
fn out(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("am2-workshop-quantize-{tag}-{nanos}.gguf"))
}

/// The raw bytes and ggml type of one tensor from a GGUF, read at its own offset — mirrors how a reference file stores
/// each tensor. `nbytes` comes from the tensor's own declared type, so it works for both the F16 source and a quantized
/// output without assuming a size.
fn tensor_bytes(path: &Path, g: &Gguf, i: usize) -> (u32, Vec<u8>) {
    let t = &g.tensors[i];
    let nbytes = t.expected_bytes().expect("tensor byte length") as usize;
    let mut r = BufReader::new(std::fs::File::open(path).unwrap());
    r.seek(SeekFrom::Start(g.data_start + t.offset)).expect("seek tensor data");
    let mut b = vec![0u8; nbytes];
    r.read_exact(&mut b).unwrap();
    (t.ggml_type, b)
}

/// Assert two GGUFs carry identical tensors: the same name -> (innermost-first dims, ggml type, exact bytes), keyed by
/// name rather than file order, since llama.cpp's `.pure` files list tensors in its canonical order while this crate
/// preserves the input order. [`values_equal`] allows metadata to be compared by value below.
fn same_tensors(a: &Path, b: &Path) {
    let ga = Gguf::open(a).unwrap();
    let gb = Gguf::open(b).unwrap();
    assert_eq!(ga.tensors.len(), gb.tensors.len(), "tensor count differs");

    let mut da: std::collections::BTreeMap<String, (Vec<u64>, u32, Vec<u8>)> = BTreeMap::new();
    let mut db: std::collections::BTreeMap<String, (Vec<u64>, u32, Vec<u8>)> = BTreeMap::new();
    for t in &ga.tensors {
        let i = ga.tensors.iter().position(|x| x.name == t.name).unwrap();
        let (ty, bytes) = tensor_bytes(a, &ga, i);
        da.insert(t.name.clone(), (t.dims.clone(), ty, bytes));
    }
    for t in &gb.tensors {
        let i = gb.tensors.iter().position(|x| x.name == t.name).unwrap();
        let (ty, bytes) = tensor_bytes(b, &gb, i);
        db.insert(t.name.clone(), (t.dims.clone(), ty, bytes));
    }
    assert_eq!(da, db, "tensor names/types/shapes/bytes differ between the two GGUFs");
}

/// Metadata equal by value (numbers compared ignoring int/float), ignoring the two folder-derived fields that differ
/// between a converter's naming and llama.cpp's, so the comparison pins the quantization rather than general.name.
fn metadata_almost_equal(a: &Gguf, b: &Gguf) -> bool {
    if a.metadata.len() != b.metadata.len() { return false; }
    const SKIP: [&str; 2] = ["general.name", "general.size_label"];
    let filtered = |g: &Gguf| g.metadata.iter().filter(|(k, _)| !SKIP.contains(&k.as_str())).count();
    if filtered(a) != filtered(b) { return false; }
    a.metadata.iter().all(|(k, v)| {
        if SKIP.contains(&k.as_str()) { return true; }
        b.metadata.get(k).is_some_and(|w| values_equal(v, w))
    })
}

/// The name -> (dims, ggml type, bytes) map for one GGUF file, keyed by tensor name so converter/reference order drift
/// does not matter. Reads each tensor's raw data at its own offset from the given file path.
fn tensor_map(path: &Path, g: &Gguf) -> std::collections::BTreeMap<String, (Vec<u64>, u32, Vec<u8>)> {
    let mut m = std::collections::BTreeMap::new();
    for t in &g.tensors {
        let i = g.tensors.iter().position(|x| x.name == t.name).unwrap();
        let (ty, bytes) = tensor_bytes(path, g, i);
        m.insert(t.name.clone(), (t.dims.clone(), ty, bytes));
    }
    m
}

/// Exactly which tensors differ between our GGUF at `our_path` and the reference file — names only, so a converter
/// divergence such as an old hardcoded head count reports `blk.0.attn_q.weight` / `attn_k.weight` instead of both byte
/// maps. Also records count mismatches and tensors present on only one side.
fn differing_tensor_names(our_path: &Path, ref_path: &Path) -> Vec<String> {
    let ga = Gguf::open(our_path).expect("our GGUF opens");
    let gb = Gguf::open(ref_path).expect("reference GGUF opens");
    let mut diffs = Vec::new();
    if ga.tensors.len() != gb.tensors.len() {
        diffs.push(format!("<tensor count differs: {} ours vs {} reference>", ga.tensors.len(), gb.tensors.len()));
    }
    let our = tensor_map(our_path, &ga);
    for t in &gb.tensors {
        match our.get(&t.name) {
            Some(o) => {
                let j = gb.tensors.iter().position(|x| x.name == t.name).unwrap();
                let (_, bytes2) = tensor_bytes(ref_path, &gb, j);
                if o.0 != t.dims || o.1 != t.ggml_type || o.2 != bytes2 {
                    diffs.push(t.name.clone());
                }
            }
            None => diffs.push(format!("present in reference, missing from ours: {}", t.name)),
        }
    }
    for name in our.keys() {
        if !gb.tensors.iter().any(|x| x.name == *name) {
            diffs.push(format!("present in ours, missing from reference: {}", name));
        }
    }
    diffs
}

/// Gate 1: quantize the tiny F32 GGUF to each of its pure targets and confirm it reproduces llama.cpp's `.pure.gguf`
/// fixtures byte for byte on every tensor, and metadata by value. Skips a target when that reference is absent.
#[test]
fn tiny_matches_pure_reference() {
    let dir = std::env::temp_dir();
    let _ = std::fs::create_dir_all(&dir);
    let input = fixtures().join("tiny-llama.f32.gguf");
    if !input.exists() { eprintln!("[skip] no tiny-llama.f32.gguf to quantize"); return; }

    // (target, reference fixture suffix) — F16's ggml name is "F16", matching the committed `tiny-llama.F16-pure.gguf`.
    let cases: [(PureType, &str); 3] = [
        (PureType::F16, "F16"),
        (PureType::Q8_0, "Q8_0"),
        (PureType::Q4_K, "Q4_K"),
    ];
    for (target, suffix) in cases {
        let outp = out(&format!("tiny-{}", suffix));
        quantize(&input, &outp, target).expect("quantize tiny");

        let refp = fixtures().join(format!("tiny-llama.{suffix}-pure.gguf"));
        if !refp.exists() {
            eprintln!("[skip] no {refp:?}: generate it with `llama-quantize tiny-llama.f32.gguf {refp:?} {suffix}`");
            std::fs::remove_file(&outp).ok();
            continue;
        }

        same_tensors(&outp, &refp);
        let go = Gguf::open(&outp).unwrap();
        let gr = Gguf::open(&refp).unwrap();
        assert!(metadata_almost_equal(&go, &gr), "metadata differs from the {suffix} reference");

        println!("{target:?}: tiny reproduced all {} tensors byte-for-byte", go.tensors.len());
        std::fs::remove_file(&outp).ok();
    }
}

/// Gate 2: a wide tiny model whose every 2-D row is a whole block (hidden 256) really gets quantized. Convert its HF
/// checkpoint to F32 with this crate's converter, assert that F32 matches llama.cpp's byte-for-byte (naming exactly the
/// diverging tensor on failure - the q/k permute's head count), then quantize each target and compare byte-for-byte to
/// the reviewer's `llama-quantize --pure` reference — skipping until those files exist.
#[test]
fn wide_matches_pure_reference() {
    let root = std::env::temp_dir().join(format!("am2-workshop-256-hf-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    std::fs::create_dir_all(&root).unwrap();

    // hidden 256, intermediate 512, one layer, vocab 32: head_dim 16 (even, for the q/k permute), every row 256 wide.
    let wide = Fixture { seed: 0x5EED, num_layers: 1, vocab_size: 32, hidden: 256, intermediate: 512, num_attention_heads: 16, num_key_value_heads: 16, shard_count: 1 };
    let ckpt = wide.write(&root).expect("write the 256-hf checkpoint");

    // The F32 GGUF `llama-quantize` quantizes from; this crate's converter matches llama.cpp on tensor bytes (step 6).
    let base = root.join("model.f32.gguf");
    convert(&ckpt.dir, &base, OutType::F32, Some("gpt-2")).expect("convert 256-hf to F32 GGUF");

    // The loud gate: our F32 must equal llama.cpp's BEFORE any quantizing. Under the old hardcoded head count this
    // failed on blk.*.attn_q / attn_k bytes; failing here (not later) names exactly which tensor diverged. Skip until
    // the reviewer places tiny-llama-256.f32.gguf in tests/fixtures/ (see NEEDED-ref.md); a wrong permute makes the
    // .pure references below fail too, so this is where the bug shows up first and clearest.
    let ref_f32 = fixtures().join("tiny-llama-256.f32.gguf");
    if !ref_f32.exists() {
        eprintln!("[skip] no tiny-llama-256.f32.gguf to compare our F32 against; make it and commit it in tests/fixtures/ \
                  (see NEEDED-ref.md): `python3 .agent/ref/llama.cpp/convert_hf_to_gguf.py tiny-llama-256-hf --outtype f32 --outfile tiny-llama-256.f32.gguf`");
    } else {
        let diffs = differing_tensor_names(&base, &ref_f32);
        assert!(diffs.is_empty(), "our F32 conversion of the wide model differs from llama.cpp in: {}", diffs.join(", "));
        println!("F32: our wide converter matches llama.cpp byte-for-byte ({} tensors)", Gguf::open(&base).unwrap().tensors.len());
    }

    for target in [PureType::Q8_0, PureType::Q4_0, PureType::Q4_K, PureType::Q5_K, PureType::Q6_K] {
        let outp = out(&format!("wide-{}", target.ggml_name()));
        quantize(&base, &outp, target).expect("quantize wide");

        let refp = fixtures().join(format!("tiny-llama-256-{}-pure.gguf", target.ggml_name()));
        if !refp.exists() {
            eprintln!("[skip] no {refp:?}: generate it with `llama-quantize model.f32.gguf {refp:?} {target:?}` (see NEEDED-quant.md)");
            std::fs::remove_file(&outp).ok();
            continue;
        }

        same_tensors(&outp, &refp);
        let go = Gguf::open(&outp).unwrap();
        let gr = Gguf::open(&refp).unwrap();
        assert!(metadata_almost_equal(&go, &gr), "metadata differs from the {target:?} wide reference");

        println!("{target:?}: wide reproduced all {} tensors byte-for-byte", go.tensors.len());
        std::fs::remove_file(&outp).ok();
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// Gate 3 (GQA): a tiny model with grouped attention - num_attention_heads != num_key_value_heads. Its k/v projections
/// carry kv_heads*head_dim rows, so attn_k's rotary permute must use num_key_value_heads (llama.cpp collapses the k
/// permute to n_head_kv). Convert its HF checkpoint to F32 and assert it matches llama.cpp's; skip until the reviewer
/// places tiny-llama-gqa.f32.gguf in tests/fixtures/ (see NEEDED-gqa.md). A hardcoded head count in the converter fails
/// exactly here, where no fixture used that value before this test existed.
#[test]
fn gqa_matches_llama_f32_reference() {
    let root = std::env::temp_dir().join(format!("am2-workshop-gqa-hf-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    std::fs::create_dir_all(&root).unwrap();

    // 16 attention heads, 4 kv heads, hidden 256 (head_dim 16), kv dim 64 = 4*16; intermediate 512, one layer, vocab 32.
    let gqa = Fixture { seed: 0x5EED, num_layers: 1, vocab_size: 32, hidden: 256, intermediate: 512, num_attention_heads: 16, num_key_value_heads: 4, shard_count: 1 };
    let ckpt = gqa.write(&root).expect("write the GQA checkpoint");

    let base = root.join("model.f32.gguf");
    convert(&ckpt.dir, &base, OutType::F32, Some("gpt-2")).expect("convert GQA to F32 GGUF");

    // v_proj / k_proj must have kv dim 64 (4 heads * head_dim 16) and q_proj the full 256 - proves the fixture is really
    // GQA, so this test actually exercises a mismatched head count.
    let g = Gguf::open(&base).expect("GQA F32 GGUF opens");
    let q_rows = g.tensors.iter().find(|t| t.name == "blk.0.attn_q.weight").expect("attn_q present").dims[1];
    let k_rows = g.tensors.iter().find(|t| t.name == "blk.0.attn_k.weight").expect("attn_k present").dims[1];
    assert_eq!(q_rows, 256, "q_proj carries num_attention_heads*head_dim rows");
    assert_eq!(k_rows, 64, "k_proj carries num_key_value_heads*head_dim rows");

    let ref_f32 = fixtures().join("tiny-llama-gqa.f32.gguf");
    if !ref_f32.exists() {
        eprintln!("[skip] no tiny-llama-gqa.f32.gguf to compare our F32 against; make it and commit it in tests/fixtures/ \
                  (see NEEDED-gqa.md): `python3 .agent/ref/llama.cpp/convert_hf_to_gguf.py <GQA HF folder> --outtype f32 --outfile tiny-llama-gqa.f32.gguf`");
    } else {
        let diffs = differing_tensor_names(&base, &ref_f32);
        assert!(diffs.is_empty(), "our GQA F32 conversion differs from llama.cpp in: {}", diffs.join(", "));
        println!("F32: our GQA converter matches llama.cpp byte-for-byte ({} tensors)", g.tensors.len());
    }
    let _ = std::fs::remove_dir_all(&root);
}

