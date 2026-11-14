# Workshop test fixtures

`peft-expert-delta.safetensors` was made once by the reviewer with PyTorch, using PEFT 0.20.0's exact
`ParamWrapper.get_delta_weight` einsums (peft/tuners/lora/layer.py), seed 7, E=3, r=2, A [6, 5], B [4, 6], scaling 0.75:
- `delta_eio` = einsum("o r e, e r i -> e i o") — PEFT's default, for a 3-D parameter of shape [E, in, out] ([3, 5, 4]);
- `delta_eoi` = einsum("o r e, e r i -> e o i") — when PEFT swapped in/out, a parameter of shape [E, out, in] ([3, 4, 5]).
The tests only read it; nothing in `cargo test` runs Python.

## tiny-llama.f32.gguf

llama.cpp's own `convert_hf_to_gguf.py` (Unsloth's checkout, `~/.unsloth/llama.cpp`, the same file as
`.agent/ref/llama.cpp/convert_hf_to_gguf.py`) run once by the reviewer on `tiny-llama-hf/`, `--outtype f32`, with
`make-tiny-llama-reference.py`. One thing is patched there: `get_vocab_base_pre` returns `"gpt-2"`, because llama.cpp
recognises BPE pre-tokenizers by hashing real text through them and a synthetic 32-token tokenizer matches no known
hash. Everything else is the converter's own output (21 tensors, 27,968 bytes).

To be a real Hugging Face folder that `convert_hf_to_gguf.py` and transformers accept, `tiny-llama-hf/` needed:
- a real `tokenizer.json` (made with the `tokenizers` library from the same 32-token vocab: version, truncation,
  padding, `added_tokens` for tok_0/tok_1 as special, normalizer, ByteLevel pre_tokenizer, post_processor,
  ByteLevel decoder, and a `BPE` model with every field the library writes);
- `tokenizer_config.json` with `bos_token`/`eos_token` as plain strings (`"tok_0"`, `"tok_1"`), and
  `tokenizer_class: "PreTrainedTokenizerFast"` (`LlamaTokenizerFast` adds a default `<unk>`, making 33 tokens).

## tiny-llama.{Q8_0,Q4_K,F16}-pure.gguf

`llama-quantize --pure tiny-llama.f32.gguf tiny-llama.<T>-pure.gguf <T>` (Unsloth's build), by the reviewer. With
16-wide rows no matrix can hold a whole 32/256-value block, so llama-quantize falls back: every 2-D weight becomes F16
and the 1-D norms stay F32, for all three targets. These pin the fallback rules, not the block quantizers (those are
pinned by strata-pack's `quant-*.gguf`).

## tiny-llama-256-{Q8_0,Q4_0,Q4_K,Q5_K,Q6_K}-pure.gguf

The wide fixture `tiny-llama-256-hf/` (hidden 256, 16 heads, intermediate 512, 1 layer) converted to F32 by llama.cpp's
converter (pre forced to gpt-2, as above), then `llama-quantize --pure` per type, by the reviewer. Every 2-D weight's
rows hold whole 32/256 blocks, so all 9 matrices are really quantized (no fallback); norms stay F32.

## tiny-llama-256.f32.gguf, tiny-llama-gqa.f32.gguf

llama.cpp's converter output (pre forced to gpt-2, as above) for `tiny-llama-256-hf/` (16 heads, 16 kv heads) and
`tiny-llama-gqa-hf/` (16 heads, 4 kv heads; written by `cargo run -p strata-workshop --example write-tiny-gqa-hf`), by
the reviewer. They pin the q/k rotary permute with the real head counts: q by `num_attention_heads`, k by
`num_key_value_heads`. (Step 6 first passed with a hardcoded head count of 2, which the 2-head tiny model could not show.)

## tiny-llama-256.f16.gguf, tiny-llama-256-bf16-hf/, tiny-llama-256-bf16.f16.gguf

llama.cpp's converter with `--outtype f16` (pre forced to gpt-2), by the reviewer: on `tiny-llama-256-hf/` (F32
source), and on `tiny-llama-256-bf16-hf/` — the same model with its safetensors cast to BF16 by PyTorch, as real
Hugging Face models are stored. 2-D weights F16 (rounded as numpy does), 1-D tensors F32.

## forward-reference.json, tiny-qwen2-hf/, tiny-qwen3-hf/, tiny-llama-tied-hf/, tiny-gqa-peft-adapter/

The parity reference of the trainer's own forward pass (`src/model.rs`, `tests/model_parity.rs`). Made once, on the
CPU, with Python from the Soup virtual environment (torch 2.11.0, transformers 5.15.1, peft 0.20.0); nothing in
`cargo test` runs Python:

    cd crates/strata-workshop/tests/fixtures
    CUDA_VISIBLE_DEVICES="" ~/Development/soup-project/.venv/bin/python make-forward-reference.py

`make-forward-reference.py` writes:
- `tiny-qwen2-hf/` (q/k/v biases), `tiny-qwen3-hf/` (q_norm/k_norm, `head_dim` 16 with hidden 32) and
  `tiny-llama-tied-hf/` (embeddings tied to the head): random models of 3 blocks, hidden 32, 4 heads, 2 key/value
  heads, vocabulary 64, seed 1234, saved by transformers' `save_pretrained` (so `config.json` is transformers 5's:
  `rope_parameters`, `model_type`). They have no tokenizer: only token ids go through them;
- `forward-reference.json`: for those three and for `tiny-llama-hf`, `tiny-llama-256-hf`, `tiny-llama-gqa-hf`,
  `tiny-llama-256-bf16-hf`, the 12 token ids `(7·i + 3) mod vocab` and transformers' float32 logits `[12][vocab]`;
- `tiny-gqa-peft-adapter/`: a PEFT rsLoRA adapter (r 4, alpha 8, all seven targets, A and B random with seed 4321,
  so the adapter is not a no-op) on `tiny-llama-gqa-hf`, and PEFT's logits of base + adapter under that name in the JSON.

The files are transformers' and PEFT's own output; the models' weights are random numbers (no licence applies to them).
