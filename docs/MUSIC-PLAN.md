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
2. **API:** `POST /v1/audio/music` (OpenAI has no such endpoint): `lyrics`, `style`, `duration`, `seed`, `format`. A job
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
