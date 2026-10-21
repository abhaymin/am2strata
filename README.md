# am2Strata

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-db61a2?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/abhaymin)

am2Strata is a new edition of [PolyStrata](https://github.com/VecSzn/PolyStrata), written in Rust: the server, the
tokenizer, chat templates, output parsing, the Hugging Face downloader, the launcher and every test. PolyStrata's
C++/CUDA engines still do the inference. am2Strata uses PolyStrata's run configs and models as they are, so either
server can run on the same machine (one at a time).

It runs local MoE models (Qwen3.8 / Orca, Tiel Coder, MiniCPM-V, LFM2 and the other architectures PolyStrata's engines
support) behind OpenAI-, Anthropic- and Responses-compatible APIs, with a web app for chat, monitoring and settings.

## Requirements

- Linux and an NVIDIA GPU, with PolyStrata's engines built (`strata`, `strata-poly`, `strata-vision`) and its run
  configs (`strata-*.json`). By default am2Strata looks in `~/Development/PolyStrata`, and for models in
  `~/Development/Strata-data`.
- Rust (stable). Optional: `ffmpeg` (video from API clients), `poppler-utils` (PDFs), `bubblewrap` (to run Python
  code from the chat in a sandbox).

## Web app

- **Chat:** several chats kept in the browser; pictures, video (frames cut in the browser) and PDFs; a thinking level.
- **Canvas** beside the chat, following what an answer makes:
  - **HTML:** a sandboxed preview with Preview / Code tabs. A Console tab appears when the page has errors, and can
    send them back to the model.
  - **Flows:** a Mermaid diagram opens by itself when an answer has one. Flow / Source / Sketch tabs (a whiteboard);
    exports to Markdown, PNG, PDF, and GIF for animated flows.
  - **Python:** runs on the server in a bubblewrap sandbox (no network, home folders hidden), only when you click Run.
  - **Links:** links in answers open in the canvas, with an address bar.
- **Monitor:** speed, GPU, VRAM, RAM, the context and the requests, live.
- **API calls:** every request's input, output, reasoning and timings.
- **Settings:**
  - models: switch, unload, download from Hugging Face with or without a token;
  - the model's run config, chat defaults, server and access, MCP tools, connect;
  - appearance: light, dark or system theme, and an accent colour.

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
| Hugging Face pull (`am2strata-run pull`: token optional, split shards, mmproj, resumable, SHA-256 checked; tokenizer extracted in Rust; run config written when an engine runs the architecture; new models listed without a restart) | done | live: LFM2-8B-A1B (lfm2moe) pulled, loaded in 9 s, 340 tok/s; extractor byte-identical to PolyStrata's on Tiel and MiniCPM-V |
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
cargo run -q --release --bin am2strata-run -- test                          # the whole test suite (no GPU)
cargo run -q --release --bin am2strata-run -- serve --take-over --port 8000 # foreground, in place of PolyStrata's service
cargo run -q --release --bin am2strata-run -- temp-service --port 8000      # temporary user service; PolyStrata's returns when it stops
cargo run -q --release --bin am2strata-run -- stop-temp-service
cargo run -q --release --bin am2strata-run -- install --port 8000           # boot service in place of strata-orca (same configs, models)
cargo run -q --release --bin am2strata-run -- uninstall                     # strata-orca back as the boot service
cargo run -q --release --bin am2strata-run -- pull LiquidAI/LFM2-8B-A1B-GGUF               # list its quantizations
cargo run -q --release --bin am2strata-run -- pull LiquidAI/LFM2-8B-A1B-GGUF --quant Q4_K_M  # download + run config
# live GPU tests against a running server
STRATA_LIVE_URL=http://127.0.0.1:8000 STRATA_LIVE_KEY=... cargo test -p am2strata --release --test live -- --ignored --test-threads=1
```

Golden files are recorded from PolyStrata with:

```sh
~/Development/PolyStrata/.venv/bin/python tools/reference-polystrata/make_golden.py
```

The engine protocol is written down in [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Support

am2Strata is free and open source. If it is useful to you, you can support its development through
[GitHub Sponsors](https://github.com/sponsors/abhaymin). Thank you.

## License

MIT, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Parts are derived from PolyStrata (MIT, Niko1221 and the Strata contributors).
