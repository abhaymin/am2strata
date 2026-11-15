# Rust engine (am2strata-hybrid) speed on the CPU

Measured with `tools/hybrid-bench.sh <model.gguf> --prompt 64 --new 16 --threads 8 --iters 1`. The script drives the engine
over its stdin protocol and reads the engine's own timings from `DONE`.

Model: Ministral 3B Instruct 2512, Q4_K_M (2.1 GB; its weights are Q4_K and Q6_K).

| Machine | Build | Prompt tok/s | Generation tok/s | Tokens |
|---|---|---|---|---|
| Linux PC, Ryzen 7 7840HS, 8 threads, beside a running server | before: rows dequantized to f32, f32 dot | 2.52 | 2.49 | |
| same | integer Q4_K/Q6_K × Q8_K kernels, AVX2 build (`cpu_q8.rs`) | **12.73** | **12.48** | the 16 generated tokens are identical to before |
| Mac mini M4, 10 CPUs (in the server, 754-token prompt) | before | 4.94 | 4.77 | |
| same | integer kernels, dot-product build (#64) | **16.80** | **14.98** | answer normal; engine alone with 10 threads: 17.70 / 18.29, with 4: 15.46 / 15.30 |

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

## Batched prompt reading

The prompt goes through the model in chunks of 32 (`Model::forward_batch`, `QMat::matmul`). Each weight row is read
once per chunk, while attention still runs token by token, in order, at each token's own position. Every value is
the one the one-token path computes:
- **Test:** `matmul_equals_matvec_exactly` checks this for Q4_K, Q6_K, Q8_0, F32 and Q5_0.
- **Benchmark:** the generated tokens are identical.

`AM2_PROMPT_BATCH=1` turns batching off, for comparison. Models whose layers aren't plain attention + dense
feed-forward, and the `--graph` GPU path, still read token by token.

| Machine | How the 256-token prompt is read | Prompt tok/s | Tokens |
|---|---|---|---|
| Linux PC, 8 threads, **busy** (the live server's coding agent was running) | one token at a time | 3.78 | |
| same, same minute | chunks of 32 | **11.21** (3.0x) | identical |

### Mac M4 with batching (#65)

| How the prompt is read | Engine, 256 tokens, 10 threads | Server, 754-token chat |
|---|---|---|
| one token at a time | 17.87 tok/s | 16.80 tok/s (before #65) |
| chunks of 32 | **25.19** (1.41x) | **23.67** |

The M4's unified memory is fast, so it gains less from saving reads than the PC does: there the work is arithmetic.

### Unpacking each weight block once per chunk

`cpu_q8::row_dot_many` unpacks each block's nibbles and scales once for all the chunk's tokens. Before, they were unpacked
again for every token. Each token's sums keep the same order, so the results are bit-identical (unit test
`many_inputs_equal_one_at_a_time_exactly`; the generated tokens are identical).

| Linux PC, 8 threads, **busy** (an agent run on the live server), 256-token prompt, alternated twice | Prompt tok/s |
|---|---|
| main (#65) | 6.97, 6.98 |
| unpack once | **8.67, 11.25** |

## Next
- **Q5_K, Q8_0 and Q4_0 the same way.**
- **The Metal (wgpu) path on the Mac:** it is bound by host-side waits, not by the GPU (FINDINGS item 6).
