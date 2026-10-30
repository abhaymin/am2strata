# The am2Strata engine: one Rust engine for dense and MoE models

Goal: am2Strata's own engine applies Strata's logic to every model, dense or MoE, and any application can use it. The hot
part of a model sits in VRAM, the rest in pinned RAM or on the SSD, transfers overlap the compute, and the engine learns what
is hot. PolyStrata's C++ engines stay the production path for each model until the Rust engine passes that model's gate, then
the run config switches it. Nothing is removed before that.

## The rule: faster, stable, never worse

Every change has to show it, by measurement on this PC. It does not ship otherwise:

- **Correct first.** The same tokens as the reference at temperature 0. A speedup that changes the tokens fails.
- **No slower.** Decode and prompt speed against the current version on the same model and prompt, medians of repeated runs.
- **Stable under load.** A soak test with no driver errors, no engine restarts and no memory growth, before a model switches.
- **A way back.** A model moves to the Rust engine only by its run config, the C++ engine stays selectable, and an engine is
  replaced by a rename with the old one kept. A failed start falls back to the model before it.

No software can promise it never fails, so the aim is that a failure is caught by a test before it ships, and that one which
gets through degrades (one GPU, the previous engine, the previous model) and does not stop the server.

## Where it stands (2026-10-30)

| | PolyStrata C++ (`strata`, `strata-poly`) | `am2strata-hybrid` (Rust) |
|---|---|---|
| Models | Qwen3.5/3.6/3.8, Gemma 4, GLM 5, DeepSeek V4, LFM2, Clover and others | Nemotron-H, GLM-4.7-Flash / DeepSeek V2-V3, Llama, Mistral, Mixtral, Qwen3 and Qwen3-MoE |
| Dense models | no | yes |
| Correctness | reference | matches llama.cpp, per model (tools/reference-llamacpp) |
| Decode, this PC | Sage A1.5: 150-160 tok/s. Orca on two GPUs: 63 tok/s | GLM-4.7-Flash: 9.1 tok/s. Ministral 3 3B: 15.7 tok/s |
| Placement | expert cache, streamed KV, MTP drafts, layer split | packs, learned profile, residency, weights on devices |

The two rows of speed are different models, so they show the size of the gap and are not a benchmark. The cause is known:
each matrix product is a round trip between the CPU and the device (GPU-PLAN.md, the two-GPU result), while the C++ engine
runs a whole token on the GPU.

## One idea for both kinds of model

A dense model is the simple case of the same design. Its layers are always needed, so placement decides which layers sit in
VRAM. An MoE model's shared layers are the same, and its experts are the part chosen per token. So there is one placement
core with three tiers (VRAM, RAM, SSD), and experts are one more kind of block in it, ranked by the learned profile.

## Phases, in order

Each phase has a gate and is measured on this PC against the phase before. Speed work never loosens correctness: a change
that alters the tokens fails its phase.

1. **A whole token on the GPU.** A token's work becomes one GPU command stream, not a round trip per product. This is the
   largest gain. Gate: Ministral 3 3B on cuda:0 at least 3x its 15.7 tok/s, 24/24 parity steps as now.
2. **One placement core.** Dense layers and experts in one VRAM / RAM / SSD ranking; a dense model larger than VRAM runs with
   its hot layers on the GPU. Gate: Devstral at a VRAM limit below its size, faster than the CPU path, same tokens.
3. **Expert cache.** Async uploads on a second stream, swaps learned while decoding, and the CPU computing what is not
   resident, overlapped with the GPU. Gate: GLM-4.7-Flash at 3x 9.1 tok/s or better, same tokens.
4. **KV like Strata's.** An 8-bit KV cache, streamed to RAM past what VRAM holds, a conversation cache and `SAVE` / `RESTORE`
   (which KNOWLEDGE-CACHE.md phase E needs). Gate: a 128K context on a 16 GB card, reads no slower than the C++ engine at 4K.
5. **Prompt path and speculation.** Chunked prompt passes and draft tokens (MTP where the model has it, lookup otherwise).
   Gate: within 25% of the C++ engine on a model both run.
6. **Two GPUs and several requests.** A layer split as a pipeline, unequal links (OCuLink and USB4), batch slots. Gate: two
   cards faster than one on the same model, same tokens, no driver errors over a soak test.
7. **More architectures.** `qwen35moe` first (Sage, Tiel, MiniCPM-V text), then `qwen4exp`, `gemma4`, `glm4moe`, `deepseek4`,
   and vision through the same engine. Each needs the parity gate against llama.cpp plus a tokenizer test.
8. **The switch.** A run config chooses `"engine": "am2strata-hybrid"` per model once that model passes: same tokens as the
   reference, and within 10% of the C++ engine's speed. The C++ engine stays selectable.

## Other applications

- **Done:** OpenAI, Anthropic and Responses APIs; Settings > Connect has a ready setup for Claude Code, Codex, opencode, Continue,
  Cline, Open WebUI and the OpenAI SDKs; `/api/system`, `/metrics/prometheus`, media and memory APIs.
- **Planned:** embeddings and rerank through the same engine (V2-PLAN phase 6), which also serves the memory module's
  semantic search; `am2strata-run connect <app>` to write an application's config file, with a preview and a backup; MCP
  server mode, so that other agents can call media and memory as tools.

## Risks

- The C++ engine has years of tuning, so phases 3 to 5 may land short of it. The switch is per model, so a model that does not
  reach its gate stays on the C++ engine.
- CubeCL's compute model may limit phase 1. If so, phase 1 falls back to a thin CUDA layer for this card, with the Vulkan and
  HIP paths staying on CubeCL, and that is decided by measurement in phase 1.
- Upstream keeps moving (STRATA-0.2-SURVEY.md): the C++ engines keep getting faster while this is built.
