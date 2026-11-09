# Songs in am2Strata (music with vocals)

Owner's request (2026-11-09): make songs, not only speech. Chatterbox is a speech model: it cannot follow a melody, so
singing needs a music model.

## Phase 0: findings (2026-11-09, from the projects' own pages)

**ACE-Step 1.5** is the choice for "lyrics and a style in, a finished song out":
- licence: **MIT** for the code (GitHub) and the weights (Hugging Face card). Generated music may be used
  commercially. Training data: licensed, royalty-free and synthetic, per its card;
- parts: a language model (0.6B, 1.7B or 4B, Qwen3-based) that plans the song and can expand tags and lyrics, and a
  diffusion transformer (DiT) that renders it: 2B ("turbo", about 4.7 GB of weights) or XL 4B (about 9 GB bf16);
- VRAM, as its README's table says: 6 GB or less with the LM off, INT8 and CPU offload; 24 GB or more for every part
  without offload. XL: 12 GB with offload and quantization, 20 GB without. Its headline says "under 4 GB"; the table is
  the more careful figure;
- takes user-written lyrics with structure tags, in 50+ languages; 10 s to 10 minutes;
- runtimes: PyTorch, vLLM, MLX, and GGML (its VST3 plugin). **No ONNX export** is published.

Others, and why not first:
- **LeVo 2 / SongGeneration (Tencent):** strong vocals, but a non-commercial Tencent licence;
- **YuE, DiffRhythm, SongBloom:** to be checked one by one (licence, VRAM) before any use. The first search did not
  give primary sources for them;
- **MusicGen** (non-commercial weights, no vocals), **Stable Audio Open** (instrumental, its own community licence):
  not for songs.

Exact melodies (lyrics plus notes) are a separate kind of model: **DiffSinger** (OpenVPI), which publishes ONNX
exports. That is phase 4 below.

## Phase 0: the reference run (2026-11-09)

ACE-Step 1.5's own pipeline (PyTorch 2.10, CUDA 12.8) on one RTX 5060 Ti 16 GB, the 2B turbo DiT (8 steps) with the
1.7B planner, script `~/Development/ref/acestep-phase0.py`:

| Song | Made in | Peak VRAM | Whisper turbo heard |
| --- | --- | --- | --- |
| English, 60 s, seed 42 | 11.9 s | 9.85 GB | every line of the verse, and the chorus |
| Spanish, 60 s, seed 7 | 11.6 s | 9.85 GB | every line ("un comando" heard as "un tomando") |
| Hindi, 60 s, seed 11 | 11.7 s | 9.86 GB | a long hummed opening, then the chorus only |

Loading: 131 s for the DiT (its first run included), 20 s for the planner; 9.35 GB of VRAM once loaded; 12 GB peak
RAM. Output: 48 kHz stereo. So a song takes about a fifth of its length on one card. Hindi needs another look (other
seeds; whether Whisper hears sung Hindi well).

Hindi again, seeds 1 to 4 (10.5 to 12.2 s each): seed 1 sings every line of the verse and the chorus; seed 4 nearly
all; seeds 2 and 3 mostly hum. So the lyrics come out, but per seed: the song job gets the same check as speech
(Whisper hears the vocals; a take that misses too many lyric words is made again with another seed, the best kept).

## Phases

0. **Reference run (needs the owner's yes for downloads):**
   - an isolated Python venv with ACE-Step 1.5's pinned packages (PyTorch with CUDA, about 3 GB), and the 2B turbo
     DiT with the 1.7B LM (about 8 GB of weights);
   - on one RTX 5060 Ti, with the text model unloaded (an idle window): three 60 s songs from fixed seeds. Measure time,
     peak VRAM and RAM, and listen;
   - record golden files per stage (LM tokens, DiT latents, decoded audio) for checking a port.
1. **Engine:** a `music` kind in the media catalog (download, size, device), run on the GPU like images: the text model
   stops for the job. Runtime, in order of preference:
   - its GGML path, if it can be driven from Rust (it already runs outside Python);
   - an ONNX export done here, if the DiT and decoder export cleanly;
   - a candle port of the LM (Qwen3, already close to what candle has), the DiT and the audio decoder: weeks of work,
     checked against the golden files.
   Long songs: stages split across both cards, as Qwen-Image's are.
2. **API** (with the lyric check above): `POST /v1/audio/music` (OpenAI has no such endpoint): `lyrics`, `style`, `duration`, `seed`, `format`. A job
   with progress (`/api/jobs/{id}`); the chat tool `make_song(lyrics, style, duration)`, queued like `generate_image`,
   so chats and remote agents can ask for one; a Monitor row with a player.
3. **Web app:** Audio › Song: lyrics, style with presets, duration, seed; player and Save; a note that songs are
   AI-made.
4. **Exact melodies:** `POST /v1/audio/sing` with lyrics and MIDI, DiffSinger through the existing ONNX path.
5. **Optional:** a song re-sung in a voice clip the owner has consent for (voice conversion), behind the same consent
   rule as Chatterbox clips.

Tests at each phase: mock-engine server tests for the fields and jobs; ignored real-model tests for length, not silent,
and Whisper finding most of the lyric words.

Sources: [ACE-Step 1.5 on GitHub](https://github.com/ace-step/ACE-Step-1.5),
[ACE-Step 1.5 on Hugging Face](https://huggingface.co/ACE-Step/Ace-Step1.5),
[ACE-Step 1.5 paper](https://arxiv.org/abs/2602.00744), [music model roundup](https://usefulai.com/models/music-generation).
