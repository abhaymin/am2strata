# ACE-Step 1.5 in Rust (songs in am2Strata)

Owner's decision (2026-11-09): songs are made inside am2Strata, in Rust (candle), not by calling ACE-Step's Python, and
they get their own section in the web app, with songs in many languages. The Python install in
`~/Development/ref/ACE-Step-1.5` is only the reference the port is checked against.

## What ACE-Step 1.5 is made of (the checkpoints downloaded here)

| Part | Checkpoint | Size | Rust |
| --- | --- | --- | --- |
| Caption encoder | `Qwen3-Embedding-0.6B` (Qwen3, 28 layers, hidden 1024) | 1.2 GB | candle's `qwen3` |
| Song planner (optional) | `acestep-5Hz-lm-1.7B` (Qwen3, 28 layers, hidden 2048; vocab 217,204 with the audio-code tokens) | 3.5 GB | candle's `qwen3` and its sampling loop |
| DiT (turbo, 8 steps) | `acestep-v15-turbo`: lyric encoder (8 layers), timbre encoder (4), attention pooler, FSQ audio tokenizer and detokenizer, 24-layer diffusion transformer (hidden 2048, 16 heads, 8 KV heads, sliding window 128, patch 2), timestep embedding | 4.5 GB | new: `crates/strata-media/src/acestep/` |
| Audio decoder | `vae`: AutoencoderOobleck (Stable Audio), 64 latent channels at 25 Hz, 48 kHz stereo, strides 2·4·4·6·10 | 0.3 GB | new, close to candle's DAC (Snake, weight-normed convs) |

Plus the sampler: flow-matching ODE (Euler) on the turbo schedule (shift 3: 1.0, 0.955, 0.9, 0.833, 0.75, 0.643, 0.5,
0.3), and the "DCW" wavelet correction it applies by default (Haar). Turbo has classifier-free guidance distilled
in: one forward pass a step.

The planner is optional: without it (ACE-Step's own low-VRAM mode) the DiT sings the given lyrics from the caption.
So the first milestone leaves it out.

## Stages, each with a gate against the Python reference

0. **Golden files** (`~/Development/ref/acestep-golden.py`): one 10 s song, fixed seed, planner off. Saved per stage
   (safetensors): caption and lyric token ids, caption encoder output, condition encoder output, initial noise,
   the latent after each of the 8 steps, the final latent, the decoded audio.
1. **Audio decoder (Oobleck):** final latent → audio. Gate: SNR above 40 dB against the reference audio (f32, CPU).
2. **Caption encoder (Qwen3-Embedding) and the text preparation** (ACE-Step's prompt format for caption and lyrics,
   per language). Gate: hidden states within 1e-3.
3. **DiT:** condition encoder (lyric, timbre, pooler), the 24 layers with sliding-window attention, timestep
   embedding, the Euler steps and DCW. Gate: each step's latent within 1e-2 (bf16 weights), the final audio SNR
   above 25 dB.
4. **Engine and server:** a `music` kind in the media catalog (download, size, GPU placement; the text model stops
   for the job like an image), `POST /v1/audio/music` (`lyrics`, `style`, `language`, `duration`, `seed`), a
   job with progress, the chat tool `make_song`, a Monitor row with a player. A lyric check: Whisper hears the vocals,
   and a take that misses too many lyric words is made again with another seed (the Hindi test needed it).
5. **Web app: a Songs tab under Media Generation** (beside Image, Video and Audio; owner's choice):
   - lyrics with structure tags (verse, chorus, bridge), a style description with presets, the vocal language
     (ACE-Step sings 50+; the picker lists them by name), duration, seed;
   - the player, Save, and "make another take";
   - a note that songs are AI-made.
5b. **Longer than 10 minutes:** a song continues from the last seconds of the one before (ACE-Step's "complete" task:
   the earlier audio's latents as the source with the chunk mask open after it), joined in one file. Part of the
   productions in `docs/LONG-MEDIA-PLAN.md`.
6. **The planner (Qwen3 1.7B):** audio codes at 5 Hz for song structure, and "write the lyrics for me" from a
   description. Gate: the same codes as Python for greedy sampling.
7. **Measured and documented:** time and VRAM on the RTX 5060 Ti against the Python numbers (about 12 s and 9.85 GB
   for 60 s), README, NOTICE (MIT, ACE-Step).

Reference numbers (Python, phase 0): see `docs/MUSIC-PLAN.md`.

## Progress (2026-11-10)

| Stage | State | Checked |
| --- | --- | --- |
| 0. Golden files | done | `~/Development/ref/acestep-golden.py` |
| 1. Audio decoder (Oobleck) | done | SNR 39.5 to 40.7 dB on the 4 recorded chunks (reference bf16) |
| 2. Caption encoder, lyric embeddings, prompts | done | same token ids; cosine 0.9992+ per token; lyrics exact |
| 3. Condition encoder, DiT, sampler, DCW | done | DiT against ACE-Step in f32: cosine 1.000000; the 8 bf16 steps 0.9926 to 0.9999; final latent 0.996 |
| Pipeline (`acestep/pipeline.rs`) | done | the golden song made again in Rust from its noise: final latent 0.995, and Whisper hears its lyrics as in Python's |
| 6. The planner (codes) | done (2026-11-10) | `planner.rs`, `dit::Detokenizer`; prompts and token ids the same as ACE-Step's; FSQ output cosine 0.99997, hints (detokenizer) 0.99999; teacher-forced on ACE-Step's greedy codes, its code is in Rust's top 5 at 47 of 50 steps, never more than 0.3 logits from the best (Python bf16 on a GPU, Rust f32 on the CPU: near ties in a flat 64,000-way choice) |
| 6b. "Write the lyrics for me" | next | the planner's own reasoning phase (caption, language and missing metadata written by it) |

A 20 s song entirely in Rust: 96 s on the CPU (f32). Its vocals, without the planner, are mostly "ooh"; ACE-Step's own
Python without the planner does the same (seeds 1 to 3: one half-heard line, "Thank you.", "Ah, ah"). With the
planner (the 60 s songs of phase 0) the lyrics are sung. So the planner (stage 6) comes before the server stage: it is
what makes the vocals follow the lyrics.

### The planner, measured (2026-11-10, one RTX 5060 Ti, ACE-Step's sampling: temperature 0.85, guidance 2.0, top-p 0.9)

Lyric words heard in order by Whisper (whisper.cpp large-v3-turbo), each seed made with and without the planner:

| Song | Without the planner | With it |
| --- | --- | --- |
| English, 60 s, 4 seeds | 47%, 56%, 22%, 61% (mean 47%) | 97%, 94%, 83%, 89% (mean 91%) |
| Hindi, 3 minutes, 5 seeds | 64%, 43%, 51%, 34%, 50% (mean 48%) | language left out of the reasoning (as ACE-Step does when the other values are given): mean 32%; with `language: hi` in it: 56%, 64%, 60%, 41%, 79% (mean 60%) |
| Malayalam, 60 s, 3 seeds | not measurable: Whisper writes Malayalam badly (in other scripts); by consonants heard after transliteration, 41%, 30%, 29% | 43%, 11%, 20% |

So the reasoning has the vocal language for every language but English (English with it: mean 76%). Time and memory:
a 60 s song in 26.7 s with the planner (its 300 codes about 19 s, the first load included), 7.5 s without; a
3-minute song in 62 s with it, 26 s without; 10.9 GB of VRAM at 60 s with it, 7.4 GB without. Malayalam and the
other languages Whisper hears badly need a better judge than Whisper (a native listener, or a better recogniser).
