# Model Workshop — train, convert, and make Strata-ready models in Rust

Owner's decision (2026-11-14): am2Strata should *build* models like Unsloth Studio does — fine-tune an HF model into
a LoRA adapter, merge it to a full GGUF, quantize it, and turn it into something the Strata engine runs — all in Rust,
with a "Workshop" tab in the web app. This plan records **what already exists** so we build on it, the **design**, the
**build order** (each step one commit, each with its tests), and the **risks**. Nothing here is coded yet; this is the
plan only.

Owner's rule, restated: **all in Rust — no Python, no venv, no shelling out to `convert_hf_to_gguf.py`, `llama-quantize`,
`optimum`, or `peft`.** Use what the workspace already has and add crates only where needed. The one exception allowed:
the existing patched llama.cpp **oracle** (`tools/reference-llamacpp/oracle.cpp`) may be used by the *parity check* as a
reference, never as part of producing a model.

The work goes in a new crate, `crates/strata-workshop` (library); the server's routes and jobs go in `am2strata`.

## 0. What already exists (read before coding)

Reading what is here matters: several pieces are close to, or exactly, the building blocks this feature needs. A missing
piece has been flagged (`── MISSING ──`).

### `crates/strata-pack` — the native GGUF pack (no candle/safetensors dependency yet)
- **`quants.rs`** — ggml *dequantization*, value-for-value as gguf-py's numpy dequantizers compute it, of 17 types
  (Q4_0…Q8_0, the K families, IQ2/IQ3/IQ4). `dequantize` / `dequantize_into`. `tests/quants_golden.rs` compares every
  type with gguf-py on recorded blocks. **── MISSING ── only *dequantize* exists; there is no matching quantize.** This
  is the first thing a quantizer step reuses or replaces.
- **`gguf_write.rs`** — a streaming single-file GGUF writer: header, string key/values in insertion order (so
  `general.architecture` goes first), tensor infos in ggml dimension order aligned to 32, padding, then each tensor's
  bytes streamed (`tensor_data`/`end_tensor`). A large tensor is never held in memory. Needs a GGUF *reader* for the
  convert step (see below).
- **`iq_pack.rs`** — builds the native Strata pack from a model's GGUF (the port of `tools/iq_pack.py`); this is what
  "the native pack when the architecture needs it" produces at runtime. `grids.rs`, `num.rs` (`f16_to_f32`) support it.
- **`stack_experts.rs`** — rewrites a GGUF whose experts are per-expert (`blk.N.ffn_{gate,up,down}.E.weight`, converted
  before Apr/2024) into the stacked layout llama.cpp reads now (`..._exps.weight`): byte-for-byte expert copy + new
  tensor infos. Reusable to detect and handle fused MoE experts in a LoRA merge (requirement 5).

### `crates/strata-gguf` — GGUF header/metadata + tokenizer extraction (no safetensors dependency)
- **`Gguf::open(path)`** reads the header without the weights: metadata as a JSON map, tensor names/shapes/ggml-type/
  offsets (`TensorInfo`, `expected_bytes()`), plus `shards()` for split files and `type_name`/`block_geometry`. Needed
  to inspect a GGUF before conversion (requirement 4's architecture-refusal) and as the reader for convert/quantize.
- **`extract_tokenizer(gguf, dir)`** writes `<out_dir>/tokenizer/` from GGUF metadata — vocab.json, merges.txt/json,
  token_type.json, tokenizer.json (with the right pre-tokenizer pattern, per architecture), chat_template.jinja — exactly
  as `tools/strata_tokenizer.py extract` writes it. This is a strong start for the convert step's tokenizer; a port must
  also read HF's separate `tokenizer.json` / `tokenizer_config.json` (see requirement 3).

### `crates/strata-engine` — how "run it through the Strata engine" works
- The engine is a **subprocess wrapper around `llama-server`**, not an in-Rust model. `RunConfig` (`strata-<model>.json`)
  names the executable, args, tokenizer folder, and GPU placement; `Engine::start` spawns it with `child_env`;
  `generate`/`session` drive it. **── this is the "run parity" target of requirement 4 ──** a made-strata GGUF is fed to
  the engine via its RunConfig. `gguf_architecture(path)` reads `general.architecture` from a GGUF header; an unsupported
  architecture is refused **before** any conversion (requirement 3's "refuse clearly", requirement 4).

### `crates/strata-hybrid` + `tools/reference-llamacpp` — the parity reference (the allowed exception)
- `strata-hybrid` is the native Rust engine (CubeCL kernels), its own workspace. Its tests in `tests/llamacpp.rs` run
  teacher-forced **against the oracle** (`tools/reference-llamacpp/oracle2`, built from `oracle.cpp`: a CPU-only patched
  copy of PolyStrata's llama.cpp that prints greedy tokens and top-5 logits). **This is the reference the parity check
  compares against.** Per owner's rule it is *never part of producing a model* — only used to verify our GGUF/quantizer.

### How strata-media uses candle (the mold fork)
- `crates/strata-media/Cargo.toml` depends on **`candle-core-nold/nold-transformers-mold`** (the utensils/candle fork at a
  pinned rev, CUDA feature), plus `mold-ai-inference`. It loads HF safetensors checkpoints into candle, runs Whisper and
  Parler-TTS and the ACE-Step DiT. The mold fork's `candle-transformers` ships an **`llama.rs`** (confirmed in the cargo
  checkout); `VarMap`/`nn` loading of those definitions is exactly what LoRA training would reuse for the frozen base.
  **Verify before coding that this same fork exposes autograd + `candle_nn::AdamW`** (the task says it does; media never
  trained).

### The media job queue (`am2strata/src/media_jobs.rs`, `manager.rs`) — the pattern to mirror
- `media_jobs.rs::Jobs` is a persistent in-process scheduler: `submit(kind, args)` queues, `wait_done`, `resume()` from
  JSON files on disk (survives restarts), capability checks, and SSE-ish progress. **Training merges here as another job
  kind** (the task says so) — its `Job` record would add a `model`/`state` machine like the existing one rather than a new
  queue. `manager.rs::gpu_exclusive*` stops the loaded text model for a GPU-only job and reloads it after; training is the
  same ("stops the text model first and reloads it after, like the media jobs"). `model.load/unload_locked` is what we call.

### Unsloth compatibility (`am2strata/src/unsloth.rs`) — already present
- `/api/inference/*` (`health`, `loaded-models`, `status`, `load`) mimic Unsloth's UI Backend so its CLI points an agent at
  am2Strata. The task's remote-training path (`UNSLOTH_STUDIO_URL` + key) goes here too: a job that does not fit locally is
  forwarded to an external Unsloth server, then merged/converted/strata-ready *in Rust* locally (see "How it fits").

### Feature-flag / settings pattern (`am2strata/src/settings.rs`, `main.rs`)
- Optional features are wired as a route group in `main.rs::app()` behind an AppState field (e.g. media jobs:
  `state.media_jobs`). A setting like `"workshop": {"enabled": bool}` (task: off by default) gates the `/v1/workshop/*`
  routes; a missing key reads as disabled. Match that shape rather than inventing one.

## 1. Design

```
crates/strata-workshop/            (new library crate)
  lora_train.rs     LoRA: candle Var adapter over frozen HF base, masked loss, accum, AdamW, cosine+warmup, PEFT output
  qlora.rs          (later) dequantize-on-the-fly 4-bit base; until then the UI says "QLoRA not ready"
  merge.rs          shard-by-shard safetensors merge, alpha/r and alpha/sqrt(r); fused-einsum experts; norm-delta verify
  convert_hf.rs     HF (safetensors) -> GGUF: name map + arch metadata + tokenizer (from tokenizer.json/config)
  quantize.rs       Rust ports of ggml reference quantizers; round-trip vs quants::dequantize and llama-quantize bytes
  onnx.rs           Rust ONNX graph writer (prost/onnx.proto) for dense decoders only, external-data weights
  dequant_gguf.rs   GGUF -> safetensors via strata-pack quants + reverse name map (f16/bf16)
  vram.rs           VRAM estimate; fused-expert detection forbidding a LoRA GGUF
  dataset.rs        chunker, dedupe/minhash, quality filters (unit-tested pure Rust)
  ...and tests/ with each module.
```

**Shared pieces first.** Every step needs to *read* and *write* safetensors and to inspect a GGUF header. `strata-pack`
already dequantizes and writes GGUF but reads none and has no quantizer; add a small **safetensors reader (`memmap2`) +
GGUF reader** into `strata-pack` or `strata-workshop`, and the matching **quantize** alongside the existing
`dequantize`. Then:

- **Train** loads the base as candle Vars (bf16/f16) from HF safetensors using the mold fork's llama/qwen/mistral model
  defs; a LoRA adapter is a pair of trainable `Var`s per layer on top. Forward on token windows with the *prompt* tokens'
  loss masked, backprop, AdamW, cosine schedule with warm-up, grad checkpointing if candle permits. Checkpoint every N
  steps as PEFT `adapter_model.safetensors` + `adapter_config.json` (rank, alpha, rslora on/off) — resumable from the last
  one.
- **Merge** reads a base shard (memmap2), adds each LoRA delta (`alpha/r`, or `alpha/sqrt(r)` when `use_rslora`) scaled and
  applied, writes it; never loads the whole 60 GB model at once. Fused MoE experts use
  `einsum("ore,eri->eio", B.reshape(o,-1,E), A.reshape(E,-1,i)) * scaling`. **Every tensor is verified by relative-norm delta:
  targeted ≈ 1–3 %, untargeted exactly 0**, and that number is reported.
- **Convert** maps HF tensor names to GGUF names per architecture, writes the arch-specific metadata keys, pulls the tokenizer
  from `tokenizer.json`/`tokenizer_config.json` (vocab, merges, token types, special tokens, chat template), and writes with
  our `gguf_write`. Verified against a llama.cpp-converted GGUF of the same tiny model (same names/shapes/metadata).
- **Quantize** ports ggml's reference quantizers (MIT) for Q8_0/Q4_0/Q4_K/Q5_K/Q6_K (+ later IQ types `strata-pack` reads);
  each is round-tripped through `quants::dequantize` under the ggml bound and against llama-quantize bytes on identical input.
  A LoRA adapter becomes a LoRA GGUF **only when it has no fused-expert tensors** (else "merge first").
- **ONNX** writes a graph via `prost`/onnx.proto for dense decoders only, weights as external data; MoE big models are refused
  with one clear sentence.
- **GGUF → safetensors** dequantizes with `strata-pack::quants` and writes f16/bf16 safetensors, HF names restored by the
  reverse of the convert name-map. Refused where no converter exists.

The single end-to-end job (requirement 5, "recipe") chains train → merge → convert → quantize → strata-ready as steps, each
resumable from its checkpoint/last artifact.

**Server API / UI.** `POST /v1/workshop/jobs` (`dataset | train | merge | convert | quantize | strata | recipe`),
`GET /v1/workshop/jobs/{id}` (+ SSE progress), `DELETE /v1/workshop/jobs/{id}` (cancel). Jobs live in the existing queue;
training stops then reloads the text model like media jobs. Off by default (`"workshop": {"enabled": bool}`).

## 2. Build order — each step one commit, each gated by its own tests (no GPU / no downloads)

Tests share one setup: **build a tiny random 2-layer Llama/Qwen config in Rust** (a few MB, fixed seed, written as
safetensors + a tiny `tokenizer.json`) — never download anything. Then train 5 steps on CPU over 20 lines and see the loss
fall; merge → convert GGUF → quantize → load with `strata-gguf` → run through the Strata engine (`llama-server`).

0. **Scaffolding + dependencies (commit + `cargo test` passes).** New `crates/strata-workshop`, added to the workspace, plus
   the new crates from the plan only: **`safetensors`** (read via `memmap2`; write our own or the crate) and **`prost`** (ONNX
   proto). Confirm the mold candle fork exposes autograd + `candle_nn::AdamW` + `VarMap` and its llama model loads a frozen base.
   Gate: tiny random safetensors writes and round-trips read.
1. **Safetensors + GGUF reader.** Memmap2 reader (shards, ordered) for the merge/quantize steps; reuse `Gguf::open`; add a small
   writer if needed. Unit test: round-trip tiny tensors, shard enumeration.
2. **The quantizer alongside dequantize.** Port ggml Q8_0/Q4_0 first into `strata-pack` (matching the existing `dequantize`),
   round-trip it and check under the bound; then Q4_K/Q5_K/Q6_K. Gate: round-trip error below the ggml reference bound on random blocks.
3. **LoRA train.** Frozen base from safetensors, adapter `Var`s, prompt-masked loss, grad accumulation, AdamW, cosine+warmup, PEFT
   checkpoint. Gate: 5 CPU steps on 20 lines → loss falls; checkpoint reloads and resumes identical step.
4. **QLoRA note (commit but not a training path).** A dequantize-on-the-fly base with the backward = ∂L/∂x·∂(dequant)/∂w; until candle
   can backprop through its quantized matmul, the UI says plainly "QLoRA (4-bit base) not ready". Gate: no regression to the 16-bit path.
5. **Merge.** Shard-by-shard add with `alpha/r` (`/sqrt(r)` if rslora), fused-einsum experts, per-tensor relative-norm-delta verify
   (targeted ≈1–3 %, untargeted 0). Gate: tiny model merges and the deltas read as expected; 60 GB streaming loop reads one shard at a time.
6. **Convert to GGUF.** HF→GGUF name map + arch metadata + tokenizer from `tokenizer.json`/`tokenizer_config.json`. Gate: our GGUF of the
   tiny model matches llama.cpp's converter (names, shapes, metadata keys); LoRA-GGUF allowed only without fused experts (`vram.rs`).
7. **Quantize complete + ONNX.** Finish Q-types + later IQ; add the prost ONNX writer for dense decoders with external data, refuse MoE.
   Gate: quantizer round-trip vs `dequantize`; tiny Llama ONNX graph parses and is refused cleanly for MoE.
8. **GGUF → safetensors.** Dequantize via `quants`, write f16/bf16, reverse name map. Gate: a GGUF of the tiny model round-trips to matching safetensors.
9. **Strata-ready + parity.** Produce `strata-<model>.json`, tokenizer folder (reuse `extract_tokenizer`), native pack where needed, register in
   the model list, then teacher-forced top-1 agreement Strata vs oracle. Gate: small but nonzero agreement; unsupported architecture refused by name.
10. **Dataset generation + quality filters** (`dataset.rs`). Chunker, dedupe (MinHash/n-gram), too-short/long/refusal/off-language filters, judging pass,
    `dataset.jsonl` + `dataset.card.md`. Gate: pure-Rust unit tests; generation tested with the mock engine.
11. **API/UI.** `/v1/workshop/*` routes in `am2strata`, jobs in the existing queue (GPU stop/reload like media), recipe job, VRAM up-front refusal for our two
    16 GB GPUs, remote-Ursloth forwarding, HF download with size+approval, "Workshop" tab. Gate: job state-transition tests; refused/accept flows.

Each step commits on `agent/model-workshop` with the tests that gate it before the next begins.

## 3. Risks

- **candle autograd.** LoRA training assumes the mold fork exposes autograd + AdamW; media only *inference*. If this fork's
  autograd is limited, training either waits for a capability bump or moves to a different autograd path — check at step 0.
- **QLoRA correctness.** Dequantize-on-the-fly backward must exactly invert `quants::dequantize` (the same element order) or the
  quantized base trains wrong; keep it isolated and document "not ready" until tested against a reference LoRA.
- **GGUF metadata fidelity.** A converted GGUF that is structurally right but off by one metadata key breaks llama-server. The tiny-
  model diff vs llama.cpp's converter is the gate, not eyeballing keys.
- **Quantizer round-trip bound.** Our ports must stay under ggml's reference error and match `llama-quantize` bytes exactly; a layout
  mismatch (block packing order) surfaces only in byte comparison — do both checks from step 2.
- **Streaming merge at scale.** Shard-by-shard over 60 GB must not hold the base in memory; the merge test model is tiny, so also add a
  size-path check that reads one shard at a time.
- **Fused-expert detection.** A LoRA with fused MoE tensors must refuse a LoRA GGUF ("merge first"). The detection (`stack_experts`-style
  `blk.N.ffn_*_exps.weight`) has to be authoritative or the merge is silently wrong.
- **Tests staying offline/gpu-free.** The tiny-model setup must never download and must run on CPU under `cargo test`; a stray HF pull in a
  test breaks CI and the owner's rule.
- **Scope.** ONNX (smallest), data-generation, remote Unsloth, HF downloads — each is real but secondary; keep them as clearly-limited steps so
  nothing half-done looks like a finished feature.
