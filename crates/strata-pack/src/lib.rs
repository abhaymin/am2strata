//! PolyStrata's packing tools in Rust: the files they write are the Python tools' files, byte for byte.
//!
//! - [`iq_pack`]: a native pack for the `strata` engine from a model's GGUF (tools/iq_pack.py)
//! - [`mtp_fetch`]: the MTP tensors from the BF16 checkpoint by HTTP range reads (tools/mtp_fetch.py)
//! - [`mtp_pack`]: the MTP draft block as a GGUF, experts in Q2_0 / Q4_0 / Q8_0 (tools/mtp_pack.py)
//! - [`mtp_rt`]: the MTP draft layer's runtime files (tools/mtp_rt.py)

pub mod grids;
pub mod gguf_write;
pub mod iq_pack;
pub mod mtp_fetch;
pub mod mtp_pack;
pub mod mtp_rt;
pub mod num;
pub mod quants;
