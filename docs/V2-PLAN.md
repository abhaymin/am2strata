# am2Strata v2: images, video and voice in Rust, and every model type

Status: started (v2.0). UI mockups: `design/V2*.dc.html` (Create › Image / Video / Voice, Settings › Models, Monitor).

## The idea

Strata's technique is placement: the hot part of a model on the GPU, the rest in pinned RAM (computed there, or
streamed to the GPU ahead of use), tables on the SSD, with transfers overlapped with compute; plus sharing the GPU with
other work. v2 applies it beyond text MoE models:

| Strata idea | Images / video | Voice |
|---|---|---|
| hot weights in VRAM, the rest in pinned RAM | diffusion blocks streamed with the next block prefetched (models larger than the card: Wan 14B, LTX 22B) | the TTS talker on the GPU when free, else the CPU |
| expert placement | Wan 2.2 A14B is a 2-expert MoE (high-noise / low-noise): the active expert in VRAM, the other in pinned RAM, prefetched before the switch step | - |
| sharing the GPU with the text model | keep / shrink / unload the text model during a job, then restore it (PolyStrata's gpu_policy) | speech-to-text on the CPU beside the text model |
| pinned RAM instead of disk | weights stay in RAM between jobs: no reload of ~13 GB per image | the same |
| caching | step caching, text-encoder output reuse | streaming audio as it is made |

## Engines: Rust

The media engine is Rust on candle (Hugging Face's ML framework; CUDA through cudarc, cuBLAS and cuDNN), built on
**mold** (MIT; its LTX-2 parts Apache-2.0), which already implements Z-Image, Qwen-Image, FLUX, SD3, Wan 2.1/2.2 and
LTX-2 in Rust with GGUF quantization. mold is pinned to one commit (`crates/strata-media/Cargo.toml`); the strata
features go on top. candle's GPU kernels are CUDA C compiled at build time and NVIDIA's libraries are linked, as for any
GPU engine; everything we write is Rust.

`am2strata-media` is its own process (a crash or a CUDA error cannot take the server down; its GPU memory is freed by
exiting), driven by the server over JSON lines like the text engines. `crates/strata-media` is a separate cargo
workspace with mold's lock file, so the server builds as before and the media engine only when asked:

    cargo build --release --manifest-path crates/strata-media/Cargo.toml --features cuda

## Models (16 GB card, 60 GB RAM)

| Type | Model | Licence |
|---|---|---|
| Image | Z-Image Turbo 6B (Q8_0 GGUF + Qwen3-4B encoder + VAE), 8 steps | Apache-2.0 |
| Image | Qwen-Image 2.1 Turbo (opt-in) | Qwen Research License |
| Video | Wan 2.2 TI2V-5B; Wan 2.2 T2V-A14B (2 x 14B experts, the expert swap) | Apache-2.0 |
| Video + audio | LTX-2 / 2.3 (opt-in) | revenue-capped licence |
| Speech in | Whisper large-v3-turbo (candle); Parakeet TDT 0.6B v3 | MIT / CC-BY |
| Speech out | Qwen3-TTS 0.6B / 1.7B (talker = a Qwen3 LM; codec to port); Kokoro-82M | Apache-2.0 |

## Phases

1. **v2.0 images (now):** `am2strata-media` with Z-Image via mold; `POST /v1/images/generations` (OpenAI's shape) in
   the server, the text model unloaded during a job and loaded again on the next chat request; Create › Image in the
   web app; mock-engine tests; a live check.
2. **v2.1 strata for images:** a warm media process with weights in pinned RAM, the text model shrinking its expert
   cache instead of unloading (engines started with `--vram-elastic`), the text encoder freed after use, VAE tiling,
   step caching. Each change measured against v2.0.
3. **v2.2 video:** `/v1/videos` (create, poll, download); Wan 2.2 TI2V-5B, then A14B with the expert swap (async copy on
   a second CUDA stream), then LTX block streaming; progress in Monitor.
4. **v2.3 voice:** `/v1/audio/transcriptions` and `/translations` (Whisper on candle, CPU first), `/v1/audio/speech`
   (Qwen3-TTS, Kokoro); a mic in the chat, read answers aloud, streaming.
5. **v2.4 dense text models** (Qwen3, Llama 3.x, Gemma 3, Mistral): `strata-dense` in Rust on candle (its quantized
   GGUF models), placing layers instead of experts, with KV streaming and the conversation cache. Gate: the same token
   ids as llama.cpp at temperature 0.
6. **v2.5 llama.cpp's features:** constrained sampling (GBNF / JSON schema) in the sampler, embeddings and rerank, LoRA
   hot-swap, a draft model, parallel slots, min-p / DRY / XTC / logprobs, FIM infill, tokenize endpoints.

## Every model in the UI (all phases)

One model library for every kind - text MoE, text dense, image, video, speech in, speech out - in Settings › Models
(mockup: `design/V2Models-*.dc.html`) and `/api/models`:

- **See** what is downloaded and what is loaded: per model its kind, engine, size on disk, licence, and where it sits
  (VRAM / pinned RAM / SSD); the text engine and the media engine each show their loaded model; for an MoE model its
  expert loadings (the profile's hottest experts, the resident share, the hit rate).
- **Download** any of them from Hugging Face (text GGUFs as today; image, video and voice models through the media
  engine's catalog), with progress, resumable and SHA-256 checked.
- **Load, switch, unload** from the list or the model picker; a media model loads when a job needs it.
- **Change** its run config (Model config) and defaults.
- **Remove** a model: its files, run config and pack, after a confirmation, never while it is loaded or in use, with
  the disk space it frees shown.
- **Downloaded but not runnable** models are listed too, with the reason (e.g. an architecture no engine runs), so
  nothing on disk is invisible.

API: `GET /api/models?kind=`, `POST /api/models/load` / `unload` (any kind), `POST /api/models/pull` (any kind),
`DELETE /api/models/{id}`; the same checks as today (key, this server's own page).

## Generate from the chat (all media phases)

Ask in the chat ("draw…", "make a 5-second clip of…", "read this aloud", or attach audio to transcribe) and the answer
comes back in the conversation:

- The chat model gets built-in tools, as MCP tools are offered today: `generate_image`, `generate_video`,
  `speak` (text to speech) and `transcribe`, with their parameters (prompt, size, length, voice, ...). The server runs
  the job with the media engine and puts the picture, video or audio in the answer (the canvas shows it; API clients
  get it as a file or URL); the model sees a short result ("image made: 1024x1024, seed 42") and can go on.
- **A capability check first, before anything is promised or started:**
  - the model for the job is installed (else: which one to download, its size, and a button);
  - the request fits the machine: VRAM and RAM for the model at that size or length (with the text model unloaded or
    shrunk), disk for the output, limits per model (sides, frames, seconds), and the licence notes;
  - an estimate of the time from measured speeds ("about 3 minutes for 5 s of video on this card"), so a long job is
    confirmed by the user first;
  - what is refused or adjusted is said plainly (e.g. "2048x2048 does not fit next to Orca; 1024x1024, or unload Orca
    for this one?").
  The same check answers `GET /api/media/capabilities` (what this PC can make, with which models, how big, how fast),
  which the web app reads to show or hide the Create tabs and the chat tools.
- The tools are offered only when the check passes and the user allows them (a switch per kind, like the MCP switch),
  never from another web page, and API clients opt in per request.

## New architectures

`am2strata-hybrid` (crates/strata-hybrid) is a Rust engine for the architectures strata's C++ engines do not run,
checked against llama.cpp (tools/reference-llamacpp: greedy tokens identical except near-ties, tokenizers identical):

| Architecture | Models | Status |
|---|---|---|
| `nemotron_h_moe` | NVIDIA Nemotron 3.5 Bolt 74B-A9B (Mamba-2, attention without positions, latent MoE, per-layer pruned experts) | done, parity |
| `deepseek2` | GLM-4.7-Flash 30B-A3B (MLA with the absorbed K/V, sigmoid routing with a bias, shared expert); DeepSeek V2/V3 family | done, parity on GLM-4.7-Flash |
| `llama` | Mistral 7B, Mistral Small / Devstral; Mixtral 8x7B / 8x22B (the same tensors plus `ffn_*_exps`, softmax top-2) | dense done, parity on Devstral; Mixtral's MoE path verified through Qwen3-MoE, its own parity needs a Mixtral GGUF |
| `mistral3` | Ministral 3, Mistral Small / Medium 3.x (YaRN, attention temperature) | done, parity on Ministral 3 3B |
| `qwen3`, `qwen3moe` | Qwen3 dense and MoE (per-head q/k norms, NEOX RoPE) | done, parity |
| `glm4moe` | GLM-4.5 / 4.6 / 4.7 (Air): GQA with partial RoPE and q/k biases, sigmoid routing, NextN layer | next: needs a GGUF to check against (the smallest, GLM-4.5-Air, is ~40-60 GB) |

Strata's placement in this engine: a pack (`--pack`, from `am2strata-run iq-pack`, which now also packs hybrid
layer schedules, gate-less experts and per-layer pruned experts - native_experts.txt v5), the expert loadings
(`--expert-profile` / `--expert-profile-save`: PolyStrata's STRP format, learned from routing counts), and residency
(`--resident-experts [--expert-ram-gib N]`: the hottest experts kept in RAM, the rest marked cold so a model larger than
the RAM reads only cold experts from the SSD; the hit rate is reported per request). Measured on GLM-4.7-Flash with
4 GiB of its 14.8 GiB of experts resident: 26.6% of the routing served from RAM by layer order, 70.9% with the learned
profile. The compute is on the CPU for now (~0.7 s/token for GLM-4.7-Flash, ~1.1 s for Nemotron Bolt on 12 threads);
next: the GPU tier (dense weights and the hot experts in VRAM via candle), then integer dot products on the CPU
(the activations are already quantized as llama.cpp does).

Run configs: `configs/strata-{nemotron-3.5-bolt,glm-4.7-flash,devstral-small}-*.json` (exe: am2strata-hybrid) (Nemotron Bolt, GLM-4.7-Flash, Devstral); `${AM2STRATA_HOME}` is this
checkout (default ~/Development/am2strata).

## Upstream

`am2strata-run upstream-check` compares the ports with PolyStrata and Strata (`tools/upstream.json`). As of
2026-10-26 they are current with PolyStrata main (b561a3b1) and with Strata 0.2.6's mtp_fetch and pack header;
Strata 0.2's Hadamard-folded packs (native_experts v5) are refused by name until PolyStrata carries Strata 0.2's
engine, and am2Strata's per-layer pruned packs are `am2strata native experts p1`.

## Alongside: the knowledge and memory cache

[KNOWLEDGE-CACHE.md](KNOWLEDGE-CACHE.md) is the other v2 track, for text models, server side and independent of the media
engine:

| Phase | What |
|---|---|
| A | answer cache: exact repeats (temperature 0 or a seed) answered from disk |
| B | knowledge collections: documents split into passages, keyword search, added to a request when asked |
| C | embeddings (in Rust) and automatic injection, with sources |
| D | shared memory: facts written by you or the models (a remember / recall tool), scoped per client |
| E | prompt snapshots: the KV state of long system prompts and documents saved to the SSD with the engine's SAVE / RESTORE, loaded instead of re-reading (~110 s -> ~1-3 s for 97K tokens) |
| F | a similar-question cache (opt-in) |

The media engine's own memory cache - weights kept in pinned RAM between jobs - is v2.1 above.

## Gates

Media cannot be bit-identical across frameworks: images are compared by similarity with reference renders from the
same initial noise, and speed against stable-diffusion.cpp on the same card. Everything else as before: Rust tests on a
mock engine, then a live GPU check (run only when the server is idle).
