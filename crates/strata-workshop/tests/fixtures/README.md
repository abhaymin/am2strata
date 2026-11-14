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
