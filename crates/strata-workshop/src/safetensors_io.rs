//! On-disk safetensors via mmap(2), for the shard-by-shard merge/quantize steps. A shard is memory-mapped (never copied
//! into Rust) and its header is validated before any tensor is touched: the declared dtype, shape and data-offsets are
//! checked against the mapped file's actual size, so a truncated or corrupt shard is rejected rather than misread.
//! Shards of a split checkpoint are enumerated in order (see [`shards`]).

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use memmap2::Mmap;
use serde_json::Value;

/// The safetensors dtypes this reader knows. Unknown spellings are rejected on read — better to fail than misread a
/// block of the wrong width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    BF16,
    F32,
    F64,
}

impl Dtype {
    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::I8 | Dtype::U8 => 1,
            Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::BF16 => 2,
            Dtype::I32 | Dtype::U32 | Dtype::F32 => 4,
            Dtype::I64 | Dtype::U64 | Dtype::F64 => 8,
        }
    }

    /// The safetensors spelling, so a descriptor round-trips to exactly what was written.
    pub fn as_str(self) -> &'static str {
        match self {
            Dtype::I8 => "I8",
            Dtype::I16 => "I16",
            Dtype::I32 => "I32",
            Dtype::I64 => "I64",
            Dtype::U8 => "U8",
            Dtype::U16 => "U16",
            Dtype::U32 => "U32",
            Dtype::U64 => "U64",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::F32 => "F32",
            Dtype::F64 => "F64",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "I8" => Dtype::I8,
            "I16" => Dtype::I16,
            "I32" => Dtype::I32,
            "I64" => Dtype::I64,
            "U8" => Dtype::U8,
            "U16" => Dtype::U16,
            "U32" => Dtype::U32,
            "U64" => Dtype::U64,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "F32" => Dtype::F32,
            "F64" => Dtype::F64,
            _ => return None,
        })
    }
}

/// One tensor of a shard: its metadata and where its raw bytes live in the mmap.
#[derive(Debug, Clone)]
pub struct TensorDesc {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    /// absolute byte offset within the shard's mmap (already has the data section added)
    offset: usize,
    nbytes: usize,
}

impl TensorDesc {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Number of elements; a 0-d scalar counts as one.
    pub fn numel(&self) -> u64 {
        let n: u64 = self.shape.iter().map(|&d| d as u64).product();
        if n == 0 {
            1
        } else {
            n
        }
    }
}

/// A memory-mapped safetensors shard. The [`Mmap`] is kept alive here, which is what keeps the byte slices valid.
#[derive(Debug)]
pub struct Shard {
    path: PathBuf,
    map: Mmap,
    tensors: Vec<TensorDesc>,
}

impl Shard {
    /// Open one shard file: mmap it, parse and validate its header, and resolve every tensor's absolute byte range.
    pub fn open(path: &Path) -> Result<Shard> {
        let path_str = path.display().to_string();
        let file = std::fs::File::open(path).map_err(|e| anyhow!("{path_str}: {e}"))?;
        let map = unsafe { Mmap::map(&file) }.map_err(|e| anyhow!("{path_str}: mmap: {e}"))?;

        // safetensors layout: u64 (LE) header length, then the JSON header, then the tensor bytes.
        let map_len = map.len();
        if map_len < 8 {
            bail!("{path_str}: {map_len} bytes is too small for even the length prefix");
        }
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&map[0..8]);
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        let data_start = 8 + header_len;
        if data_start > map_len {
            bail!("{path_str}: header claims {header_len} bytes but the file is only {map_len}");
        }

        let header: Value = serde_json::from_slice(&map[8..data_start]).map_err(|e| anyhow!("{path_str}: header JSON: {e}"))?;
        let obj = match header {
            Value::Object(m) => m,
            _ => bail!("{path_str}: header is not a JSON object"),
        };

        let mut tensors = Vec::new();
        for (name, val) in obj.iter() {
            if name == "__metadata__" {
                continue; // global metadata block carried by the crate's writer; not a tensor
            }
            let fields = match val {
                Value::Object(m) => m,
                _ => bail!("{path_str}: entry {name:?} is not an object"),
            };
            let dtype_str = fields
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("{path_str}: tensor {name} has no dtype"))?;
            let dtype = Dtype::from_str(dtype_str).ok_or_else(|| anyhow!("{path_str}: unsupported dtype {dtype_str:?} for {name}"))?;

            let shape: Vec<usize> = fields.get("shape").and_then(Value::as_array).ok_or_else(|| anyhow!("{path_str}: {name} has no shape"))?.iter()
                .filter_map(|n: &Value| n.as_u64())
                .map(|x: u64| x as usize)
                .collect();

            let offs = fields
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("{path_str}: {name} has no data_offsets"))?;
            if offs.len() != 2 {
                bail!("{path_str}: {name} data_offsets has {} entries", offs.len());
            }
            let rel_start = offs[0].as_u64().ok_or_else(|| anyhow!("{path_str}: {name} bad start offset"))? as usize;
            let rel_end = offs[1].as_u64().ok_or_else(|| anyhow!("{path_str}: {name} bad end offset"))? as usize;

            // dtype + shape must account for exactly the declared byte span, and that span (added to the data section
            // start) must fall inside the file.
            let numel: u64 = shape.iter().map(|&d| d as u64).product();
            let expected = dtype.size_bytes() as u64 * numel;
            if rel_end < rel_start {
                bail!("{path_str}: {name} data_offsets are backwards: {rel_start} > {rel_end}");
            }
            let nbytes = rel_end - rel_start;
            if nbytes as u64 != expected {
                bail!("{path_str}: {name} bytes ({nbytes}) != numel*{dtype:?} size ({expected}); dtype/shape mismatch");
            }
            let offset = data_start + rel_start;
            let abs_end = offset + nbytes;
            if abs_end > map_len {
                bail!("{path_str}: {name} spans {offset}..{abs_end} but the file is {map_len}");
            }

            tensors.push(TensorDesc { name: name.clone(), dtype, shape, offset, nbytes });
        }

        Ok(Shard { path: path.to_path_buf(), map, tensors })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|t| t.name.as_str())
    }

    pub fn tensors(&self) -> impl Iterator<Item = &TensorDesc> {
        self.tensors.iter()
    }

    /// The bytes of one tensor, borrowed straight from the mmap (no copy).
    pub fn bytes(&self, t: &TensorDesc) -> &[u8] {
        let end = t.offset + t.nbytes;
        if end > self.map.len() {
            panic!("{}: tensor {} bytes run past the mapping", self.path.display(), t.name);
        }
        &self.map[t.offset..end]
    }

    pub fn get(&self, name: &str) -> Option<&TensorDesc> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

/// The shards of a split safetensors checkpoint (`model-00001-of-00003.safetensors`): all of them, in order, when the
/// first is given and they follow that naming; just the file otherwise. Reuses the same idea as `strata-gguf::shards`.
pub fn shards(first: &Path) -> Vec<PathBuf> {
    let name = first.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if let Some(stem) = name.strip_suffix(".safetensors") {
        if let Some((head, tail)) = stem.rsplit_once("-of-") {
            if let (Some((base, _)), Ok(total)) = (head.rsplit_once('-'), tail.parse::<usize>()) {
                if total > 0 {
                    return (1..=total).map(|i| first.with_file_name(format!("{base}-{i:05}-of-{tail}.safetensors"))).collect();
                }
            }
        }
    }
    vec![first.to_path_buf()]
}
