# polystrata-rs

A staged rewrite of [PolyStrata](https://github.com/VecSzn/PolyStrata) in Rust. The plan, with phases, gates and
risks, is the "PolyStrata in Rust" document. This repository is at **Phase 0 done, Phase 1 started**.

## What works

| Crate | What it does | Checked against |
| --- | --- | --- |
| `strata-tokenizer` | Byte-level BPE for PolyStrata tokenizer directories (`qwen35`, `glm4` pre-tokenizers), special-token rules, incremental UTF-8 decode | Token ids recorded from PolyStrata's Python tokenizer: 12 chat prompts and 16 edge cases each for Tiel Coder and Orca, all identical |
| `strata-template` | The model's Jinja chat template with transformers' settings, Python-style `tojson`, `raise_exception` | Prompt text recorded from PolyStrata's Python `ChatTemplate`: all 12 cases identical for both models, tools and tool round trips included |
| `strata-protocol` | GEN/STOP/QUIT lines and engine event parsing | Unit tests on the Python server's formats |
| `strata-engine` | Starts an engine from a PolyStrata run config, waits for READY, one request at a time, STOP and drain when a client leaves | Live, with the Tiel Coder Q6 engine |
| `strata-serve` | `/health`, `/v1/models`, `/v1/chat/completions` (streaming and not), API key, reasoning split | Live, with the Tiel Coder Q6 engine |

Live results on the RTX 5060 Ti box (Tiel Coder UD-Q6_K_XL, greedy, warm): 96–120 tok/s; time spent outside the
engine 1.3–1.5 ms per request (Python server: 1.6–1.8 ms); first streamed text 31 ms after the request; a client
hanging up mid-answer stops the engine and the next request answers in 0.66 s. The engine loaded more experts into
VRAM under the Rust server (4,180 against ~3,720) only because the Rust server does not start the image encoder yet.

PolyStrata's own server is not deterministic between runs, even greedy (GPU and CPU expert paths round differently
and the expert cache adapts), so comparing answer text is not a valid test. The gate is that the Rust server sends
the engine the same token ids, which the golden files prove.

## Not yet

Image input, tool-call parsing in the output, stop strings, Anthropic `/v1/messages` and the Responses API, model
switching, the web page, metrics, the model card's default sampling (requests without a temperature are greedy
here), and everything from Phase 2 on.

## Use

```sh
cargo build --release
cargo test --release
# the GPU must be free: unload the PolyStrata service's model first
./target/release/strata-serve --config ~/Development/PolyStrata/strata-tiel-ud-q6_k_xl.json --port 8001
```

Golden files are recorded from PolyStrata with:

```sh
~/Development/PolyStrata/.venv/bin/python tools/make_golden.py
```

The engine protocol is written down in [docs/PROTOCOL.md](docs/PROTOCOL.md).
