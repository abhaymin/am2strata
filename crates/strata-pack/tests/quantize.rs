//! Quantizers against the reference reconstruction bound and against llama-quantize's bytes.
//!
//! - [`quantizer_error_under_bound`]: fixed-seed random f32 blocks quantized then `quants::dequantize`d; per-block
//!   relative RMS error must stay under each encoding's known ggml bound (a proxy for "round-trips faithfully").
//! - [`save_quant_input_fixture`] writes the shared fixed-seed F32 GGUF that `llama-quantize` is quantized from.
//! - [`quantized_bytes_match_llama_quantize`] compares our bytes with llama-quantize's, byte for byte; it is ignored
//!   until the reviewer generates `fixtures/quant-<TYPE>.gguf` (see fixtures/NEEDED.md) and commits them.

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
    // a fixed-seed F32 GGUF with one [256, 64] tensor: the shared input `llama-quantize` is run against.
    ensure_dir();
    let input = quantize_input(256 * 64);
    let raw: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();

    let specs = vec![TensorSpec { name: "weight".into(), shape: vec![256, 64], ggml_type: 0 /* F32 */, nbytes: raw.len() as u64 }];
    let kv = [("general.architecture", "quantize-fixture".to_string())];
    let path = fixtures().join("quant-input-f32.gguf");
    let mut w = Writer::create(&path, &kv, specs).unwrap();
    w.tensor_data(&raw).unwrap();
    w.end_tensor().unwrap();
    w.finish().unwrap();

    // tell the reviewer how to produce the references; committed alongside so they stay in sync.
    std::fs::write(fixtures().join("NEEDED.md"), concat!(
        "# quant-<TYPE>.gguf fixtures\n",
        "Each is produced by `llama-quantize` from `quant-input-f32.gguf`; tests compare our bytes to these byte for byte.\n",
        "Run (in the crate dir) and commit every output:\n\n",
        "    llama-quantize quant-input-f32.gguf quant-Q8_0.gguf Q8_0\n",
        "    llama-quantize quant-input-f32.gguf quant-Q4_0.gguf Q4_0\n",
        "    llama-quantize quant-input-f32.gguf quant-Q4_K.gguf Q4_K\n",
        "    llama-quantize quant-input-f32.gguf quant-Q5_K.gguf Q5_K\n",
        "    llama-quantize quant-input-f32.gguf quant-Q6_K.gguf Q6_K\n"
    )).unwrap();
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

#[test]
#[ignore = "run `llama-quantize quant-input-f32.gguf quant-<TYPE>.gguf <TYPE>` for Q8_0 Q4_0 Q4_K Q5_K Q6_K in crates/strata-pack/tests/fixtures, commit the outputs, then rerun to compare our bytes to theirs byte for byte"]
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

// ggml type ids (GGML_TYPE_*), which the reference GGUFs carry for each tensor
fn block_type_id(t: &str) -> u32 {
    match t { "Q8_0" => 8, "Q4_0" => 2, "Q4_K" => 36, "Q5_K" => 37, "Q6_K" => 38, _ => unreachable!() }
}
