//! # strata-workshop
//!
//! The model Workshop: train a LoRA adapter, merge it, convert to GGUF, quantize, and turn the result into something
//! the Strata engine runs — **all in Rust**, with no Python and no shelling out (the owner's rule). Only the llama.cpp
//! *parity oracle* is used as a reference, never to produce a model. This crate is the library; the server routes and
//! jobs go in `am2strata`.
//!
//! Build order (one commit per step): 0 scaffolding + candle training proof, 1 safetensors/GGUF reader, then the
//! quantizer, LoRA train, merge, convert, quantize, ONNX, GGUF→safetensors, strata-ready, dataset, API. See
//! `docs/MODEL-WORKSHOP-PLAN.md`.

pub mod safetensors_io;
pub mod testmodel;
