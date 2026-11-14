# Reference: `tiny-llama-256.f32.gguf` (committed; gates currently run against it)

This is llama.cpp's own `convert_hf_to_gguf.py` F32 output for the wide checkpoint
(`tests/fixtures/tiny-llama-256-hf/`, hidden 256 / 16 heads / intermediate 512 / 1 layer / vocab 32), and it is
committed in `tests/fixtures/`. It is what `wide_matches_pure_reference`'s **loud F32 gate** compares against: that
gate converts the same checkpoint with this crate's converter and asserts byte-for-byte equality, reporting exactly
which tensor diverges (e.g. `blk.0.attn_q.weight`) instead of dumping both byte maps. With the file present the gate
runs and must pass; remove it to see the `[skip]` path.

## How it was made (regeneration)

Run llama.cpp's converter on the committed wide HF folder, the same way `tiny-llama.f32.gguf` was made:

```bash
python3 .agent/ref/llama.cpp/convert_hf_to_gguf.py \
    crates/strata-workshop/tests/fixtures/tiny-llama-256-hf \
    --outtype f32 --outfile crates/strata-workshop/tests/fixtures/tiny-llama-256.f32.gguf
```

Add the same `get_vocab_base_pre -> "gpt-2"` patch as `make-tiny-llama-reference.py` (the synthetic 32-token tokenizer
matches no known pre-tokenizer hash), and run it from a checkout of `.agent/ref/llama.cpp` on `sys.path`. Then copy the
result into `tests/fixtures/tiny-llama-256.f32.gguf`.

## Why it is needed

The step-6 bug was `mapped_tensors` calling `permute(..., heads=2)` for both q and k (the tiny model's head count), so
no test could see it - the fixture and the constant agreed. With 16 real heads llama.cpp permutes with `n_head=16`; a
wrong head count changes `attn_q`/`attn_k` bytes. This reference is what makes that mismatch fail loudly **before** the
`.pure` quantization step, which otherwise needs this right F32 anyway. Once present it also unblocks the `.pure`
comparisons (they already exist as `tiny-llama-256-{Q8_0,Q4_0,Q4_K,Q5_K,Q6_K}-pure.gguf`).
