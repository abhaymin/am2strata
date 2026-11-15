# Rust engine (am2strata-hybrid) speed on the CPU

Measured with `tools/hybrid-bench.sh <model.gguf> --prompt 64 --new 16 --threads 8 --iters 1`. The script drives the engine
over its stdin protocol and reads the engine's own timings from `DONE`.

Model: Ministral 3B Instruct 2512, Q4_K_M (2.1 GB; its weights are Q4_K and Q6_K).

| Machine | Build | Prompt tok/s | Generation tok/s | Tokens |
|---|---|---|---|---|
| Linux PC, Ryzen 7 7840HS, 8 threads, beside a running server | before: rows dequantized to f32, f32 dot | 2.52 | 2.49 | |
| same | integer Q4_K/Q6_K × Q8_K kernels, AVX2 build (`cpu_q8.rs`) | **12.73** | **12.48** | the 16 generated tokens are identical to before |
| Mac mini M4, 10 CPUs (in the server, 754-token prompt) | before | 4.94 | 4.77 | |
| same | integer kernels, dot-product build | to be measured | | |

## What changed
- **`cpu_q8.rs`: the activations are quantized to Q8_K once per matrix product,** with the same rule the f32 path
  already used (`Vdt::Q8K`, llama.cpp's `quantize_row_q8_K`). Then each Q4_K or Q6_K weight row is multiplied in
  integers: exact i32 sums per group, scaled by the block's scales.
  - The values multiplied are the same as before; only the order of the float additions changes.
  - A unit test checks the quantized values equal `round_trip`'s exactly. Another checks every kernel build equals
    the f32 path on random blocks.
- **One plain kernel body is compiled three ways,** with the build chosen once at run time:
  - AVX2 + FMA on x86-64;
  - the dot-product extension on aarch64 (Apple M-series, recent Arm servers);
  - a baseline for any CPU.

  The compiler vectorizes the fixed-size loops, so no hand-written intrinsics differ between platforms.

## Next
- **Batched prompt reading:** each weight row read once per chunk of prompt tokens, not once per token. Today the
  prompt is still read one token at a time, which is why prompt and generation speeds are equal.
- **Q5_K, Q8_0 and Q4_0 the same way.**
- **The Metal (wgpu) path on the Mac:** it is bound by host-side waits, not by the GPU (FINDINGS item 6).
