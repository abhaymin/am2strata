# Chatterbox in am2Strata's media engine

Goal: Resemble AI's Chatterbox text-to-speech (MIT), native in `crates/strata-media` on candle like Whisper and
Parler. It gives:
- one voice for every line, from a reference clip;
- 23 languages with the multilingual checkpoint (Hindi, Spanish and French among them);
- an exaggeration control.

It goes through the existing `/v1/audio/speech` endpoint and the Audio page. Reference: `resemble-ai/chatterbox`
at ab55923, weights `ResembleAI/chatterbox` on Hugging Face (MIT, not gated).

## What the reference runs

Generating speech (`tts.py`):
1. **Conditioning, once per voice:**
   - the reference clip at 24 kHz and 16 kHz;
   - the voice encoder (an LSTM, `ve.safetensors`, 5.7 MB) gives the speaker embedding;
   - the S3 speech tokenizer (`speech_tokenizer_v2_25hz`, Whisper-style encoder with FSQ) gives the prompt speech
     tokens;
   - CAMPPlus (`xvector.py`) gives the x-vector, and a mel of the clip gives the decoder's prompt.
2. **T3** (`t3_cfg.safetensors`, 0.5B, Llama backbone):
   - text tokens (`tokenizer.json`), plus a conditioning prefix: speaker embedding, perceiver-resampled prompt
     tokens and the exaggeration;
   - speech tokens sampled with classifier-free guidance (two rows: conditioned and not), temperature, min-p,
     top-p and a repetition penalty.
3. **S3Gen** (`s3gen.safetensors`, 1.06 GB):
   - an upsampling conformer encoder;
   - the conditional flow-matching decoder (CFM, a U-Net of transformer blocks, ODE steps) gives the mel;
   - the HiFT vocoder (F0 predictor, source module, iSTFT) gives a 24 kHz waveform.
4. **The Perth watermark** (`resemble-perth`) on the output. It is a separate network; decided below.

Multilingual (`mtl_tts.py`) uses the same networks with `t3_mtl23ls_v3.safetensors` and `s3gen_v3`, plus a
language tag and per-language text preparation:
- Hindi, Spanish and French need only the grapheme table;
- Chinese (Cangjie) and Japanese (pykakasi) are later work.

Turbo (`tts_turbo.py`, 350M, English) comes after.

## Phases

0. **Reference and golden files:**
   - an isolated Python venv with the pinned packages (torch 2.6, chatterbox at ab55923);
   - for fixed seeds, inputs and a reference clip, dump each network's inputs and outputs (`.safetensors`):
     voice embedding, speech prompt tokens, x-vector, T3 logits for the first steps, speech tokens, encoder
     output, CFM mel, waveform.
1. **Front end:**
   - mel spectrograms (both settings), resampling;
   - text tokenizer (`tokenizers`), voice encoder, CAMPPlus, S3 tokenizer.
   - Gate: equal to the golden files within 1e-4 (f32 CPU).
2. **T3:**
   - candle Llama with input embeddings, conditioning encoder, perceiver;
   - the CFG sampling loop with a KV cache.
   - Gate: logits within 1e-3; greedy speech tokens identical.
3. **S3Gen:**
   - conformer encoder, CFM decoder (Euler steps as the reference), HiFT with iSTFT.
   - Gate: mel within 1e-3; waveform SNR above 40 dB against the reference for the same tokens.
4. **Engine and server:**
   - a `chatterbox` model family in the media catalog: download, a size per checkpoint, placement on CPU or GPU;
   - `/v1/audio/speech` gets `voice` as a saved reference voice or a clip, plus `language` and `exaggeration`;
   - Settings › Models › Audio lists it;
   - the Audio page gets a voice upload (a clip of 5 to 15 s) and a language picker.
   - Tests:
     - mock-engine server tests for the request fields;
     - an ignored real-model test: text → WAV → Whisper turbo, word error at most 15%, speaker embedding
       similarity to the reference above 0.75.
5. **Multilingual:**
   - `t3_mtl23ls_v3` with Hindi, Spanish and French text preparation;
   - the same round-trip test in each language with Whisper.
6. **Measured and documented:**
   - real-time factor and peak memory on the CPU and the RTX 5060 Ti;
   - NOTICE (MIT, Resemble AI), README, docs.

## Two paths (owner's decision, 2026-11-09: both)

- **ONNX first, behind an `onnx` cargo feature:**
  - `ort` 2.0.0-rc.11 (ONNX Runtime API 23) with `load-dynamic`, so nothing is downloaded at build time;
  - the engine loads `libonnxruntime` at startup: the system's 1.23 here (`libonnxruntime1.23`, CPU and oneDNN),
    or `ORT_DYLIB_PATH`;
  - no runtime library means the ONNX models are listed as unavailable, with the reason;
  - models: `ResembleAI/chatterbox-turbo-ONNX` (English, 350M, paralinguistic tags) and
    `onnx-community/chatterbox-multilingual-ONNX` (23 languages), fp32;
  - the Rust side does the tokenizer, the sampling loop with its KV cache, the repetition penalty, resampling to
    24 kHz and the WAV. The four sessions do the networks.
- **The native candle port (the phases below)**, checked against the Python reference and against the ONNX path.

## Devices in sequence or in parallel (owner's requirement, 2026-11-09)

The engine must hold several CPUs, GPUs, NPUs or TPUs and use them in two ways:
- **in parallel:** independent jobs on different devices at once (the lanes below);
- **in sequence:** one job's stages on different devices, one after another. Example: Qwen-Image's text encoder and VAE on
  GPU 1 while its transformer runs on GPU 0. Measured 2026-11-09: that job used only GPU 0 and peaked at 15.2 of
  16.3 GB in the VAE decode while GPU 1 sat empty. The same job failed with CUDA_ERROR_OUT_OF_MEMORY at 01:01.

So each device is described once and shared by both modes:
- `{id: "cuda:0" | "cuda:1" | "cpu" | "npu:0" | "tpu:0", kind, memory, free}`, found at start and refreshed when a
  card comes or goes (the USB4 card);
- a job's **plan** names a device per stage (`text_encoder`, `transformer`, `vae`, `llm`, `decoder`, ...). The
  scheduler gives the job the lanes of every device in its plan, in order (sequence), or splits jobs across free lanes
  (parallel);
- runtimes:
  - candle (CPU, CUDA; Metal on macOS);
  - ONNX Runtime execution providers: CPU, CUDA, Vitis AI for AMD NPUs, OpenVINO for Intel NPUs, CoreML on Apple;
  - TPUs where a runtime exists (Edge TPU through its own delegate, Cloud TPU through PJRT). None on this PC, so it is
    a slot in the design, not a claim.

## Lanes: CPU, GPU and NPU at once (owner's request, 2026-11-09)

Today one turn covers everything:
- the queue runs one job at a time;
- a GPU media job stops the text model;
- the media engine is a single process, one request at a time.

So a chat waits behind a voice job even when that job could run on the CPU (seen on 2026-11-08: a chat waited 77 s
behind Parler GPU jobs).

New rules:
- **A lane per device: `gpu`, `cpu`, `npu`.** A job names its device, chosen by `placement()` and the request's
  `device`. It waits only for jobs in its own lane, so a CPU speech job and a GPU chat run at the same time.
- **One lane per GPU** (`gpu:0`, `gpu:1`, ... as `nvidia-smi` numbers them). This PC has two RTX 5060 Ti 16 GB.
  - A job needs a set of cards. A text model split across both cards (Orca today) takes both lanes. A model that
    fits on one card takes one, which leaves the other card for a media job or a second model.
  - `placement()` picks the cards from free VRAM: a job goes to a card that is idle, else to a card whose resident
    model it can share, else it waits for the card that frees first.
  - Each lane keeps today's rules: a resident model per card, and the same-model batching bound against reloads.
  - The media engine runs one process per card it uses, started on that card (`CUDA_VISIBLE_DEVICES`).
  - A card that drops off the bus (the USB4 card's boot race, see the deployment notes) takes its lane away. Jobs
    placed on it go back to the queue with the reason.
- **The CPU lane has N slots**, each one a media engine process:
  - N comes from RAM: each slot holds its own copy of the model, and what the text model leaves free must cover it
    (Chatterbox fp32 is about 3.5 GB a slot);
  - N is at most `cpu_slots` in the run config, and at most the cores divided by the threads per slot;
  - each slot gets `intra_threads` = cores / N, so the slots don't fight for cores.
- **The NPU lane has one slot.** It exists only where an NPU and a runtime for it are found:
  - this PC has an AMD XDNA NPU (Ryzen 7 7840HS; `amdxdna`, `/dev/accel/accel0`);
  - ONNX Runtime reaches it through its Vitis AI execution provider, which needs AMD's Ryzen AI software;
  - Linux support for this first-generation XDNA is to be confirmed. Until a model is measured on it, the lane is
    off and says why.
- **Visible:**
  - `/api/queue` lists each lane's running jobs and its waiting ones;
  - Monitor shows one row per lane;
  - a chat's waiting banner names only jobs in its own lane.
- **Tests (mock engine):**
  - a CPU speech job and a chat run at once;
  - two CPU slots run two speech jobs at once, and a third waits;
  - a GPU media job still waits for the text model's turn on its card;
  - with two cards and a one-card text model, a GPU media job runs on the other card at once; with a text model
    split across both, it waits;
  - a slot that would not fit in RAM is not started.

## Decisions

- Native candle port, not ONNX Runtime: the engine stays one Rust program, and the CUDA path is the one Whisper
  and Parler already use. ONNX exports exist (`ResembleAI/chatterbox-turbo-ONNX`,
  `onnx-community/chatterbox-multilingual-ONNX`) as a fallback.
- Voice cloning copies a real person's voice. The server only takes clips that the person running it uploads, and
  the Audio page asks them to confirm that they have the speaker's consent.
- Watermark: port Perth (owner's decision, 2026-11-09). Every Chatterbox clip is marked as AI-made, as the
  reference does. A test checks that Perth's own detector finds it.

## Downloads (asked before)

| What | From | Size |
| --- | --- | --- |
| English weights: `t3_cfg`, `s3gen`, `ve`, `tokenizer.json` (safetensors) | huggingface.co/ResembleAI/chatterbox | 3.2 GB |
| S3 speech tokenizer v2 25 Hz (ONNX, read for its weights) | the `s3tokenizer` package's model URL | ~0.5 GB |
| Multilingual `t3_mtl23ls_v3`, `s3gen_v3`, the text tables (phase 5) | huggingface.co/ResembleAI/chatterbox | 3.2 GB |
| Python reference: torch 2.6 CPU, torchaudio, transformers, librosa and the rest, in its own venv | PyPI | ~1.5 GB |
