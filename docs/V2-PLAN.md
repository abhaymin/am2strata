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

## Gates

Media cannot be bit-identical across frameworks: images are compared by similarity with reference renders from the
same initial noise, and speed against stable-diffusion.cpp on the same card. Everything else as before: Rust tests on a
mock engine, then a live GPU check (run only when the server is idle).
