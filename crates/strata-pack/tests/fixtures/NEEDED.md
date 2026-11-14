# quant-<TYPE>.gguf fixtures
Each is produced by `llama-quantize` from `quant-input-f32.gguf`; tests compare our bytes to these byte for byte.
Run (in the crate dir) and commit every output:

    llama-quantize quant-input-f32.gguf quant-Q8_0.gguf Q8_0
    llama-quantize quant-input-f32.gguf quant-Q4_0.gguf Q4_0
    llama-quantize quant-input-f32.gguf quant-Q4_K.gguf Q4_K
    llama-quantize quant-input-f32.gguf quant-Q5_K.gguf Q5_K
    llama-quantize quant-input-f32.gguf quant-Q6_K.gguf Q6_K
