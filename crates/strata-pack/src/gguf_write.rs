//! A GGUF writer laid out as gguf-py's GGUFWriter writes a single file: the header, string key/values in insertion
//! order (general.architecture first), the tensor infos (ggml dimension order, offsets aligned to 32), padding to 32,
//! then each tensor's bytes padded to 32. Tensor data is streamed, so a large tensor is never held in memory.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub const ALIGN: u64 = 32;

pub struct TensorSpec {
    pub name: String,
    /// numpy order (outermost first), as gguf-py's add_tensor takes it
    pub shape: Vec<u64>,
    pub ggml_type: u32,
    pub nbytes: u64,
}

pub struct Writer {
    out: BufWriter<File>,
    tensors: Vec<TensorSpec>,
    next: usize,
    written: u64,
}

fn pad(n: u64) -> u64 {
    n.div_ceil(ALIGN) * ALIGN - n
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}

// GGUF value types (gguf-py's GGUFValueType)
const UINT32: u32 = 4;
const INT32: u32 = 5;
const FLOAT32: u32 = 6;
const BOOL: u32 = 7;
const STRING: u32 = 8;
const ARRAY: u32 = 9;

/// A metadata value with its GGUF type, as llama.cpp's converter writes them: numbers as numbers, token lists as typed
/// arrays. On disk a scalar is `type u32, payload`; an array is `ARRAY u32, element type u32, count u64, elements`.
#[derive(Debug, Clone, PartialEq)]
pub enum Kv {
    Str(String),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    /// `tokenizer.ggml.tokens`, `.merges`
    Strings(Vec<String>),
    /// `tokenizer.ggml.token_type`
    Ints(Vec<i32>),
    /// `tokenizer.ggml.scores`
    Floats(Vec<f32>),
}

fn put_kv(b: &mut Vec<u8>, key: &str, v: &Kv) {
    put_str(b, key);
    match v {
        Kv::Str(s) => { b.extend(STRING.to_le_bytes()); put_str(b, s); }
        Kv::U32(n) => { b.extend(UINT32.to_le_bytes()); b.extend(n.to_le_bytes()); }
        Kv::I32(n) => { b.extend(INT32.to_le_bytes()); b.extend(n.to_le_bytes()); }
        Kv::F32(f) => { b.extend(FLOAT32.to_le_bytes()); b.extend(f.to_le_bytes()); }
        Kv::Bool(x) => { b.extend(BOOL.to_le_bytes()); b.push(*x as u8); }
        Kv::Strings(v) => {
            b.extend(ARRAY.to_le_bytes()); b.extend(STRING.to_le_bytes()); b.extend((v.len() as u64).to_le_bytes());
            for s in v { put_str(b, s); }
        }
        Kv::Ints(v) => {
            b.extend(ARRAY.to_le_bytes()); b.extend(INT32.to_le_bytes()); b.extend((v.len() as u64).to_le_bytes());
            for n in v { b.extend(n.to_le_bytes()); }
        }
        Kv::Floats(v) => {
            b.extend(ARRAY.to_le_bytes()); b.extend(FLOAT32.to_le_bytes()); b.extend((v.len() as u64).to_le_bytes());
            for f in v { b.extend(f.to_le_bytes()); }
        }
    }
}

impl Writer {
    /// Writes the header, the key/values (all strings) and the tensor infos; the data follows with `tensor_data`.
    /// The same bytes as `create_typed` with every value a `Kv::Str` (the MTP draft writer uses this).
    pub fn create(path: &Path, kv: &[(&str, String)], tensors: Vec<TensorSpec>) -> std::io::Result<Writer> {
        let typed: Vec<(&str, Kv)> = kv.iter().map(|(k, v)| (*k, Kv::Str(v.clone()))).collect();
        Self::create_typed(path, &typed, tensors)
    }

    /// Writes the header, typed key/values in the given order (general.architecture first, as gguf-py does) and the
    /// tensor infos; the data follows with `tensor_data`.
    pub fn create_typed(path: &Path, kv: &[(&str, Kv)], tensors: Vec<TensorSpec>) -> std::io::Result<Writer> {
        let mut b = Vec::new();
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        b.extend((tensors.len() as u64).to_le_bytes());
        b.extend((kv.len() as u64).to_le_bytes());
        for (k, v) in kv {
            put_kv(&mut b, k, v);
        }
        let mut off = 0u64;
        for t in &tensors {
            put_str(&mut b, &t.name);
            b.extend((t.shape.len() as u32).to_le_bytes());
            for d in t.shape.iter().rev() {
                b.extend(d.to_le_bytes());
            }
            b.extend(t.ggml_type.to_le_bytes());
            b.extend(off.to_le_bytes());
            off += t.nbytes + pad(t.nbytes);
        }
        b.extend(vec![0u8; pad(b.len() as u64) as usize]);
        let mut out = BufWriter::with_capacity(32 << 20, File::create(path)?);
        out.write_all(&b)?;
        Ok(Writer { out, tensors, next: 0, written: 0 })
    }

    /// The next tensor's bytes, in one or more pieces; `end_tensor` closes it.
    pub fn tensor_data(&mut self, piece: &[u8]) -> std::io::Result<()> {
        self.written += piece.len() as u64;
        self.out.write_all(piece)
    }

    pub fn end_tensor(&mut self) -> std::io::Result<()> {
        let t = &self.tensors[self.next];
        if self.written != t.nbytes {
            return Err(std::io::Error::other(format!("tensor {:?} wrote {} bytes, expected {}", t.name, self.written, t.nbytes)));
        }
        self.out.write_all(&vec![0u8; pad(t.nbytes) as usize])?;
        self.next += 1;
        self.written = 0;
        Ok(())
    }

    pub fn finish(mut self) -> std::io::Result<()> {
        if self.next != self.tensors.len() {
            return Err(std::io::Error::other("not every tensor was written"));
        }
        self.out.flush()
    }
}
