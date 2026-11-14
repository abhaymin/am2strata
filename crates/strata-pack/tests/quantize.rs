//! Quantizers against the reference reconstruction bound and against llama-quantize's bytes.
//!
//! - [`quantizer_error_under_bound`]: fixed-seed random f32 blocks quantized then `quants::dequantize`d; per-block
//!   relative RMS error must stay under each encoding's known ggml bound (a proxy for "round-trips faithfully").
//! - [`save_quant_input_fixture`] writes the shared fixed-seed F32 GGUF that `llama-quantize` is quantized from.
//! - [`quantized_bytes_match_llama_quantize`] compares our bytes with llama-quantize's, byte for byte on the committed
//!   `fixtures/quant-<TYPE>.gguf` (see fixtures/README.md), skipping any type whose reference file is absent.

use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use strata_gguf::Gguf;
use strata_pack::{gguf_write::{TensorSpec, Writer}, quantize, quants};

const TYPES: [&str; 5] = ["Q8_0", "Q4_0", "Q4_K", "Q5_K", "Q6_K"];

// a fixed-seed PRNG (splitmix64) so the blocks are reproducible without a `rand` dependency
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    // uniform f32 in (-2, 2): a bounded, well-spread range that exercises every encoding's scale without overflowing fp16
    fn bounded(&mut self) -> f32 {
        let m = (self.next_u64() >> 40) as u32; // top 24 bits, exact mantissa
        (m as f32 / (1u32 << 24) as f32 - 0.5) * 4.0
    }
}

fn fixtures() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures") }

fn ensure_dir() { let d = fixtures(); let _ = std::fs::create_dir_all(&d); }

/// Relative L2 reconstruction error over the whole buffer, `||x - xhat|| / ||x||` (gguf-py's dequantizer recomputes xhat).
fn rel_error(x: &[f32], xhat: &[f32]) -> f32 {
    let (mut num, mut den) = (0.0_f32, 0.0_f32);
    for (a, b) in x.iter().zip(xhat) {
        let d = a - b;
        num += d * d;
        den += a * a;
    }
    num.sqrt() / den.sqrt()
}

fn quantize_input(count: usize) -> Vec<f32> {
    let mut rng = Rng::new(0x2545_F491_4F6C_DD1D);
    (0..count).map(|_| rng.bounded()).collect()
}

#[test]
fn quantizer_error_under_bound() {
    // per-encoding relative RMS bound for fixed-seed random blocks: ~3x the measured reconstruction error, leaving room
    // for platform float differences; tight enough to catch a mis-quantizer (round-trip via quants::dequantize), loose
    // enough to hold. Q8_0's ~1/127 step is finest, the K formats' MSE scale search is better than uniform, Q4_0 coarsest.
    let bounds: [(&str, f32); 5] = [("Q8_0", 0.011), ("Q4_0", 0.19), ("Q4_K", 0.16), ("Q5_K", 0.083), ("Q6_K", 0.04)];
    for (t, bound) in bounds {
        let elems = 256 * 64;                 // a whole number of the encoding's 256- or 32-element blocks
        let x = quantize_input(elems);
        let mut bytes = Vec::new();
        quantize::quantize(t, &x, &mut bytes).unwrap();
        let xhat = quants::dequantize(t, &bytes).unwrap();
        let err = rel_error(&x, &xhat);
        println!("{t}: rel error {err:.5} (bound {bound})");
        assert!(err < bound, "{t}: rel error {err:.5} exceeded bound {bound}");
    }
}

#[test]
fn save_quant_input_fixture() {
    // a fixed-seed F32 GGUF with one tensor whose numpy shape is [64, 256], i.e. ggml ne = [256, 64]: the shared input
    // `llama-quantize` is run against. TensorSpec.shape is in numpy order (outermost first); the writer reverses it into
    // the file's ggml dimension order, so [64, 256] -> ne = [256, 64]. A K-quant needs the last dim a multiple of 8;
    // ne[1] = 64 is (ne[0] = 256 would not be).
    // written to a temporary file and compared with the committed one (never rewritten here: another test reads it
    // in parallel, and llama-quantize's references were made from exactly those bytes)
    let input = quantize_input(64 * 256);
    let raw: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();

    let specs = vec![TensorSpec { name: "weight".into(), shape: vec![64, 256], ggml_type: 0 /* F32 */, nbytes: raw.len() as u64 }];
    let kv = [("general.architecture", "quantize-fixture".to_string())];
    let path = std::env::temp_dir().join(format!("quant-input-f32-{}.gguf", std::process::id()));
    let mut w = Writer::create(&path, &kv, specs).unwrap();
    w.tensor_data(&raw).unwrap();
    w.end_tensor().unwrap();
    w.finish().unwrap();
    let (ours, committed) = (std::fs::read(&path).unwrap(), std::fs::read(fixtures().join("quant-input-f32.gguf")).unwrap());
    let _ = std::fs::remove_file(&path);
    assert!(ours == committed, "the generator no longer writes the committed quant-input-f32.gguf (the references were made from it)");
}

fn input_f32() -> Vec<f32> {
    let p = fixtures().join("quant-input-f32.gguf");
    let g = Gguf::open(&p).expect("quant-input-f32.gguf (run save_quant_input_fixture first)");
    let t = &g.tensors[0];
    let n = t.elements() as usize;
    let mut r = BufReader::new(std::fs::File::open(&p).unwrap());
    r.seek(SeekFrom::Start(g.data_start + t.offset)).expect("seek input tensor");
    let mut raw = vec![0u8; n * 4];
    r.read_exact(&mut raw).unwrap();
    raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn reference_tensor_bytes(path: &Path) -> (u32, Vec<u8>) {
    let g = Gguf::open(path).unwrap();
    let t = &g.tensors[0];
    let nbytes = t.expected_bytes().expect("reference tensor size") as usize;
    let mut r = BufReader::new(std::fs::File::open(path).unwrap());
    r.seek(SeekFrom::Start(g.data_start + t.offset)).expect("seek reference tensor");
    let mut b = vec![0u8; nbytes];
    r.read_exact(&mut b).unwrap();
    (t.ggml_type, b)
}

// Byte-for-byte against llama-quantize's output on the committed fixtures (`tests/fixtures/README.md` says how they were made).
// Runs by default now that the type ids are right; each type is skipped only when its reference file is absent.
#[test]
fn quantized_bytes_match_llama_quantize() {
    ensure_dir();
    let input = input_f32();
    for t in TYPES {
        let ref_path = fixtures().join(format!("quant-{t}.gguf"));
        if !ref_path.exists() {
            eprintln!("[skip] no {ref_path:?}: generate it first with `llama-quantize quant-input-f32.gguf quant-{t}.gguf {t}`");
            continue;
        }
        let mut our = Vec::new();
        quantize::quantize(t, &input, &mut our).unwrap();
        let (ref_ty, theirs) = reference_tensor_bytes(&ref_path);
        assert_eq!(ref_ty, block_type_id(t), "{t}: reference ggml type");
        assert_eq!(our.len(), theirs.len(), "{t}: byte length differs ({:?} vs {})", our.len(), theirs.len());
        assert_eq!(&our[..], &theirs[..], "{t}: quantized bytes differ from llama-quantize");
        println!("{t}: matched {}/{} bytes", our.len(), theirs.len());
    }
}

// ggml type ids (GGML_TYPE_*), which the reference GGUFs carry for each tensor. The K formats are the dense-encoding
// ids (12/13/14), not the 36/37/38 GGML_TYPE_Q{4,5,6}_K_2 "extra block" ids used only by some llama.cpp internals.
fn block_type_id(t: &str) -> u32 {
    match t { "Q8_0" => 8, "Q4_0" => 2, "Q4_K" => 12, "Q5_K" => 13, "Q6_K" => 14, _ => unreachable!() }
}
