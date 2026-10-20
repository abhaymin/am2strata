# am2Strata

am2Strata is a new edition of [PolyStrata](https://github.com/VecSzn/PolyStrata) in Rust. The plan, with phases, gates and
risks, is the "PolyStrata in Rust" document. This repository is at **Phase 0 and Phase 1 done**: the Rust server can stand in for the Python one for chat, coding agents (OpenAI, Anthropic and Responses APIs), MCP tools, pictures and documents. Image generation is the one PolyStrata feature not ported.

## What works

`am2strata` runs every model of a PolyStrata folder (each `strata-*.json`) behind one address and switches
between them, driving PolyStrata's own C++/CUDA engines (`strata` for Orca / Qwen3.8, `strata-poly` for Tiel Coder,
MiniCPM-V and the others) over their line protocol.

| Area | State | Checked against |
| --- | --- | --- |
| Tokenizer, chat template, request normalization (OpenAI and Anthropic), quoted special tokens, prompt ids | done | Golden files recorded from PolyStrata's Python: 12 template cases, 16 tokenizer cases and 30 whole requests per model, all identical |
| Output parsing: reasoning, tool calls, streamed tool arguments, quoted `</think>`, calls in code or reasoning | done | 164 recorded parser runs, all identical |
| Tool-call formats GLM, Gemma 4, DeepSeek V4 (DSML), LFM2; gemma4 and deepseek-v3 tokenizers; per-format stop tokens and prompt rules | done | recorded parser runs; synthetic tokenizers recorded from the Python tokenizer (no such model installed here) |
| OpenAI Responses `/v1/responses` (stateless, typed SSE events, namespace and custom tools, encrypted reasoning replay) | done | mock engine; live: Orca |
| Thinking budgets (with answer room), `tool_choice` forcing a call, preserve_thinking, reasoning_close_retry, reasoning_loop_recovery, repeat stop | done | mock engine; live: budget and forced call on Orca |
| MCP tools (stdio and Streamable HTTP servers from the run config, `/mcp`, tool rounds for the web app's `strata_mcp`) | done | mock MCP server |
| Monitor (`/metrics`: request history, totals, live tok/s, GPU/CPU/RAM/disk series), `/metrics/prometheus`, `/v1/status` | done | mock engine; live |
| Shared chat settings (`/settings`), run-config view (`/config`), config `sampling` block (merged key by key, as PolyStrata) | done | mock engine |
| OpenAI `/v1/chat/completions` (stream and not, tools, images), `/v1/models` | done | live: Orca, Tiel Q6, MiniCPM-V |
| Anthropic `/v1/messages` (stream and not, thinking, tool_use), `/v1/messages/count_tokens` | done | live |
| Images (image encoder process, cache, `GENI`), complex documents | done | live: invoice and purchase order fields exact |
| Video | as frames: video parts are cut into frames with ffmpeg when installed; frames sent as images always work | live (frames); ffmpeg path needs `sudo apt install ffmpeg` |
| Model switching (`/api/models/load`, `/unload`, web picker) | done | live: Orca, MiniCPM-V, Tiel Q6 |
| Web app (am2Strata's `web/`, PolyStrata's app rebranded; MIT notice in `web/LICENSE-PolyStrata.txt`) | done: chat, model picker, settings, monitor polling | live in a browser |
| Stop strings, model-card default sampling, `fit_max_tokens` | done | live |
| Hardware adaptation | GPU detection; refuses an engine not compiled for the card (with the fix); image encoder on the CPU below 15.5 GB of VRAM; encoder started before the engine so its VRAM is set aside | live on the RTX 5060 Ti |

Live numbers on the RTX 5060 Ti 16 GB (warm, greedy, thinking off): Tiel Coder Q6 104-106 tok/s median through
this server (PolyStrata: ~98, same expert placement: 3,713 experts in VRAM); Orca's engine came up with the same
expert cache as under PolyStrata (3,985 slots, 8.27 GB).

## Not yet

Image generation (`/v1/images/generations`). Parallel batch slots and replicas (PolyStrata's opt-ins) are not ported
either; this box runs one slot.

The PolyStrata engines here are built for sm_86 and sm_120 (RTX 30 and RTX 50 cards) since 2026-10-20, so the same
build runs on an RTX 3060 12 GB; warm speed on the 5060 Ti is unchanged (Orca 62 tok/s, Tiel Q6 99-110 tok/s).

## Use

```sh
python3 run.py --test                                   # the whole Rust test suite (no GPU)
python3 run.py --take-over --port 8000                  # in the foreground, in place of the PolyStrata service
python3 run.py --temp-service --port 8000               # as a temporary user service; PolyStrata's comes back when it stops
python3 run.py --stop-temp-service
# live GPU tests against a running server
STRATA_LIVE_URL=http://127.0.0.1:8000 STRATA_LIVE_KEY=... cargo test -p am2strata --release --test live -- --ignored --test-threads=1
```

Golden files are recorded from PolyStrata with:

```sh
~/Development/PolyStrata/.venv/bin/python tools/make_golden.py
```

The engine protocol is written down in [docs/PROTOCOL.md](docs/PROTOCOL.md).
