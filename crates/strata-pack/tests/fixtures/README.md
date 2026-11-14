# Quantizer reference fixtures

`quant-input-f32.gguf` is written by the `save_quant_input_fixture` test (fixed seed, 16384 f32 values).

`quant-<TYPE>.gguf` were made once by the reviewer, outside `cargo test` (nothing here runs Python or llama.cpp):

1. The same f32 data copied into a GGUF that llama.cpp accepts (architecture `llama`, minimal hparams, tensor
   `blk.0.attn_q.weight` with ne = [256, 64]) using llama.cpp's own `gguf-py`; the copy's tensor bytes are identical
   to `quant-input-f32.gguf`'s.
2. `llama-quantize --pure <copy> quant-<TYPE>.gguf <TYPE>` for Q8_0 Q4_0 Q4_K Q5_K Q6_K (`--pure`: every tensor exactly
   that type, no Q4_K_M-style mixing), with Unsloth's llama.cpp build (`~/.unsloth/llama.cpp`), 2026-11-14.

Each file holds one tensor; the tests compare only its ggml type id and its bytes.
ggml type ids: Q4_0 = 2, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14.
