# polystrata-rs

A staged rewrite of [PolyStrata](https://github.com/VecSzn/PolyStrata) in Rust. The plan, with phases, gates and
risks, is the "PolyStrata in Rust" document. This repository is at **Phase 0 done, Phase 1 started**.

## What works

`strata-serve` runs every model of a PolyStrata folder (each `strata-*.json`) behind one address and switches
between them, driving PolyStrata's own C++/CUDA engines (`strata` for Orca / Qwen3.8, `strata-poly` for Tiel Coder,
MiniCPM-V and the others) over their line protocol.

| Area | State | Checked against |
| --- | --- | --- |
| Tokenizer, chat template, request normalization (OpenAI and Anthropic), quoted special tokens, prompt ids | done | Golden files recorded from PolyStrata's Python: 12 template cases, 16 tokenizer cases and 30 whole requests per model, all identical |
| Output parsing: reasoning, tool calls, streamed tool arguments, quoted `</think>`, calls in code or reasoning | done | 92 recorded parser runs, all identical |
| OpenAI `/v1/chat/completions` (stream and not, tools, images), `/v1/models` | done | live: Orca, Tiel Q6, MiniCPM-V |
| Anthropic `/v1/messages` (stream and not, thinking, tool_use), `/v1/messages/count_tokens` | done | live |
| Images (image encoder process, cache, `GENI`), complex documents | done | live: invoice and purchase order fields exact |
| Video | as frames: video parts are cut into frames with ffmpeg when installed; frames sent as images always work | live (frames); ffmpeg path needs `sudo apt install ffmpeg` |
| Model switching (`/api/models/load`, `/unload`, web picker) | done | live: Orca, MiniCPM-V, Tiel Q6 |
| Web app (PolyStrata's own files) | done: chat, model picker, settings, monitor polling | live in a browser |
| Stop strings, model-card default sampling, `fit_max_tokens` | done | live |
| Hardware adaptation | GPU detection; refuses an engine not compiled for the card (with the fix); image encoder on the CPU below 15.5 GB of VRAM; encoder started before the engine so its VRAM is set aside | live on the RTX 5060 Ti |

Live numbers on the RTX 5060 Ti 16 GB (warm, greedy, thinking off): Tiel Coder Q6 104-106 tok/s median through
this server (PolyStrata: ~98, same expert placement: 3,713 experts in VRAM); Orca's engine came up with the same
expert cache as under PolyStrata (3,985 slots, 8.27 GB).

## Not yet

Thinking budgets (`reasoning_budget_tokens`, Anthropic `budget_tokens` as a hard cap), `tool_choice` forcing a call,
the Responses API, MCP tool running, request history and GPU telemetry in the monitor, shared chat settings, image
generation, and the GLM / Gemma / DeepSeek / LFM2 tool-call formats.

For an RTX 3060 (sm_86) the engines must also be compiled for sm_86; today's PolyStrata build here is sm_120 only,
and the server says so instead of failing inside the engine.

## Use

```sh
cargo build --release
cargo test --release
# the GPU must be free: unload the PolyStrata service's model first
./target/release/strata-serve --models-dir ~/Development/PolyStrata --model orcarouter --port 8001
```

Golden files are recorded from PolyStrata with:

```sh
~/Development/PolyStrata/.venv/bin/python tools/make_golden.py
```

The engine protocol is written down in [docs/PROTOCOL.md](docs/PROTOCOL.md).
