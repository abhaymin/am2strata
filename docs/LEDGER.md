# The ledger: every plan, where it stands, what comes next

The one page to start from. Each plan has its own document; this page says where each stands today and the next step,
and is updated in the same commit as any work on a plan (a plan's state never lives only in someone's memory or a
chat). Newest decisions first in each row. Dates are when the row last changed.

**Release:** v0.1.3 (2026-11-10). Live on the developer's PC: `am2strata-boot`, Tiel Coder Q8 by default, media engine
`--features cuda,onnx`, song model and multilingual voices downloaded.

## In progress, in the order they are worked on

| # | Plan | State | Next step | Updated |
| --- | --- | --- | --- | --- |
| 0 | v0.1.3 videos (owner, 2026-11-10) | Song "Peace and harmony" (3:24) and its music video made by am2Strata (10 Wan clips, 16 images); music video (youtu.be/7mq7Jgj9y7o) and its 58 s Short (youtube.com/shorts/UMKXHTjIuWw) uploaded as Private drafts in the am2Strata playlist, AI use disclosed. Demo narration done; README visuals updated. | Demo recordings (Chrome fails to start under the service unit: run in a scope), render, upload as a draft; owner makes drafts public. | 2026-11-10 |
| 1 | [Songs in Rust (ACE-Step 1.5)](ACESTEP-RUST-PLAN.md) | Stages 0-3 and the pipeline done and checked against Python (DiT cosine 1.000000 at f32); in the engine, `/v1/audio/music`, the Song tab, `make_song` / `compose_music`, 32 genres, takes. Measured: 60 s in 12 s on one RTX 5060 Ti. | **Stage 6, the song planner** (Qwen3 1.7B writing 5 Hz audio codes; FSQ, detokenizer): without it sung words follow the lyrics only sometimes (82-83% on good 60 s takes, 7% on a 3-minute Hindi song). Golden files for it from `acestep-golden.py` with the planner on. | 2026-11-10 |
| 2 | [Productions: long pieces built by am2Strata](LONG-MEDIA-PLAN.md) | Planned (owner's requirement 2026-11-10): one request, many parts (speech, songs continued past 10 minutes, music, video, images), joined into one file with chapters, resumable. | `POST /v1/media/productions` and the chat tool `produce`: plan, parts as jobs, the audio join (48 kHz, crossfades, loudness) in the engine; then songs continued with ACE-Step's "complete" task. | 2026-11-10 |
| 3 | [Long video by segments](LONG-MEDIA-PLAN.md) | Planned. Video is 257 frames a request today. | Measure one Wan 2.2 TI2V 5B segment on the RTX 5060 Ti; chain segments from their last frame; join into one MP4 (a production part). | 2026-11-10 |
| 4 | [Native voices](VOICES-PLAN.md) | Built: 8 languages female and male (Parler multilingual, Indic Parler), Chatterbox clones them; take check by Whisper. | Indic Parler as a speaker for Malayalam and the Indian languages Chatterbox lacks; a held-vowel check (Parler sometimes holds its last sound); synthetic native voices for Arabic, Russian, Korean and the rest. | 2026-11-10 |
| 5 | [MCP servers from the app](MCP-UI-PLAN.md) | Planned (owner 2026-11-09). | Settings › Tools: add, edit, test and remove MCP servers, kept in `<data>/mcp-servers.json`. | 2026-11-09 |
| 6 | [Server tools for every client](SERVER-TOOLS-PLAN.md) | Planned (owner 2026-11-09): memory, web, media and MCP run by the server inside OpenAI / Anthropic / Responses calls, for remote agents (`agentrun`, `unsloth start`). | Per-key client profiles naming the tools; the tool loop (now 100 rounds, with context shortening) on the API routes. | 2026-11-10 |
| 8 | Children's album (owner, 2026-11-10) | Planned, after the v0.1.3 videos: 6-8 original songs in English and Hindi, catchy, 1-2 minutes each (short songs sing their lyrics well before the planner), with gentle visuals; uploaded "Made for kids", labelled AI-made. Notes on rights and YouTube's rules given to the owner (AI-made music may not be copyrightable; YouTube demonetises mass-produced content; Made for kids turns off personalised ads). | Write the lyrics (counting, colours, animals, kindness, bedtime, a Hindi lullaby...), make each with takes, check every song by Whisper and by ear. | 2026-11-10 |
| 7 | [Chatterbox in Rust, devices and lanes](CHATTERBOX-PLAN.md) | ONNX path in use; native candle port, the Perth watermark and device lanes (CPU, GPU per card, NPU) planned. | Device lanes: a CPU speech job and a GPU chat at once. | 2026-11-09 |

## Done (kept for reference)

| Plan | What it gave | Updated |
| --- | --- | --- |
| [Music phase 0](MUSIC-PLAN.md) | ACE-Step chosen (MIT), measured in Python; the reference the Rust port is checked against. | 2026-11-10 |
| Speech in the language of the text | `language: auto`, the detector (`langdetect.rs`), model switching, Language picker; 20,000 characters. | 2026-11-10 |
| Chat depth | 100 tool rounds an answer (500 on request), older results shortened to stay in context. | 2026-11-10 |
| Monitor | History in pages (25-200), the table scrolls in its card. | 2026-11-10 |
| API calls | Kept across restarts (`<data>/api-calls/calls.jsonl`, newest 2,000), Clear all, dates on older calls (owner, 2026-11-10). | 2026-11-10 |
| [Knowledge cache, part D](KNOWLEDGE-CACHE.md) | Shared memory and chats past the context. Parts A-C, E still planned there. | 2026-10-24 |

## Longer-running engine work (their own plans)

| Plan | State |
| --- | --- |
| [The am2Strata engine](ENGINE-PLAN.md) | Rust CPU engine per family, gated token for token against llama.cpp; PolyStrata's CUDA engines stay the GPU path until a family passes. |
| [Every GPU in Rust](GPU-PLAN.md) | CUDA, ROCm and Vulkan through wgpu; status in the plan. |
| [v2: images, video, voice](V2-PLAN.md) | Images, video and speech built; the rest is in the rows above. |
| [Strata 0.2 survey](STRATA-0.2-SURVEY.md) | What upstream Strata/PolyStrata changes to port (`am2strata-run upstream-check`). |

## How this page is kept

- Work on a plan changes its row here in the same commit: the state, the next step, the date.
- A new request from the owner becomes a row (or a line in a plan) before work starts, with the date.
- Measurements are written with their conditions; nothing is "done" until it is checked on this PC.
