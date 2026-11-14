# Reference: `tiny-llama-gqa.f32.gguf` (committed; gate currently runs against it)

This is llama.cpp's own `convert_hf_to_gguf.py` F32 output for the GQA checkpoint, and it is committed in
`tests/fixtures/`. It is what `gqa_matches_llama_f32_reference` gates on: a byte-for-byte F32 comparison. With 16
attention heads but only 4 kv heads, this crate's converter must permute `attn_q` with `num_attention_heads=16` and
`attn_k` with `num_key_value_heads=4` (llama.cpp collapses the k permute to `n_head_kv`, see
`conversion/llama.py::LlamaModel.permute`). A head count hardcoded for the 2-head tiny model is invisible to all
existing fixtures - this test is the first that uses a mismatched head count, so it is where such a bug fails. With the
file present the gate runs and must pass; remove it to see the `[skip]` path.

## How it was made (regeneration)

First write the GQA HF checkpoint (16 attention / 4 kv heads) - or re-run after changing sizes:

```bash
cargo run -p strata-workshop --example write-tiny-gqa-hf   # -> tiny-llama-gqa-hf/
```

Then convert it with llama.cpp's converter, the same way `tiny-llama.f32.gguf` / `tiny-llama-256.f32.gguf` were made
(patch `get_vocab_base_pre -> "gpt-2"` like `make-tiny-llama-reference.py`, and run from a checkout of
`.agent/ref/llama.cpp` on `sys.path`):

```bash
python3 .agent/ref/llama.cpp/convert_hf_to_gguf.py \
    crates/strata-workshop/tests/fixtures/tiny-llama-gqa-hf \
    --outtype f32 --outfile crates/strata-workshop/tests/fixtures/tiny-llama-gqa.f32.gguf
```

Copy the result into `tests/fixtures/tiny-llama-gqa.f32.gguf`.

## Why it is needed

It pins that `attn_k` uses the kv head count (its rows are 64 = 4*16, not 256) so the rotary permute matches llama.cpp
under GQA. The `.pure` gate for this config can be added the same way as `tiny-llama-256-*-pure.gguf` once the F32 is in
place - quantization needs the correct F32 first.
