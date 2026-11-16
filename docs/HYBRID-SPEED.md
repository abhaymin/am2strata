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

## The GPU path (wgpu: Metal on macOS, Vulkan elsewhere)

**Correctness.** On a Mac M4, Metal gave wrong tokens: different from the first token, or one token repeated with
the graph. Per-layer checksums (`AM2_DEBUG_LAYERS=1`) showed Metal matching the CPU through layer 2, then the
feed-forward of layers 3, 4 and 6 going wrong: exactly the layers whose ffn_down is Q4_K, 3072 x 9216.
- **Cause:** the wide Q4_K kernel keeps the whole input in threadgroup memory, 42,112 bytes at 9216 columns, over
  Metal's 32 KiB limit.
- **Fix:** it now runs only where it fits the device's limit, else the plain kernel runs.
- **Result:** Metal gives the CPU's tokens exactly.

**The device graph is taken by itself.** The matrix-by-matrix path syncs with the device after every product, which is
about 6x slower on the M4. So `--devices` now builds the whole-token graph whenever the model allows it (`graph=auto`);
`--no-graph` keeps the old path.

| Ministral 3B, 64-token prompt | Prompt tok/s | Generation tok/s |
|---|---|---|
| Mac M4, Metal, matrix by matrix | 2.21 | 1.98 |
| Mac M4, Metal, graph | 12.53 | 12.46 |
| Mac M4, CPU, 4 threads | **20.82** | **15.28** |
| Linux PC, Radeon 780M (Vulkan), matrix by matrix | 6.8 | 7.0 |
| Linux PC, Radeon 780M (Vulkan), graph | **14.0** | **14.2** |

On the M4 the CPU stays faster: the graph reads the prompt one token at a time, while the CPU reads it in chunks of
32 with integer kernels. A batched prompt on the device is the next GPU step.

## Past the memory limit: GLM-4.7-Flash (17.5 GB) in less RAM

This is the case the engine is for: a mixture of experts bigger than the RAM, with the hot experts kept in memory and the
cold ones read from the SSD. GLM-4.7-Flash UD-Q4_K_XL (deepseek2: MLA attention, 64 experts, 4 used, ~3B active), a
512-token prompt and 64 generated tokens, greedy.

**Mac mini M4, 16 GB** (old build d8c65da, 4 threads):

| | Prompt tok/s | Generation tok/s | Swap written |
|---|---|---|---|
| llama.cpp CPU (`-ngl 0`, default) | 11.4 | 2.05 | ~98 GB (6.1 M swapouts) |
| llama.cpp Metal (`-ngl 99`) | 24.1 | out of GPU memory | |
| llama.cpp `-ngl 20` | 18.0 | 2.30 | ~70 GB |
| ours, `--resident-experts --expert-ram-gib 8` with a learned profile | 7.0 | **5.90** | 0 |

llama.cpp's CPU backend repacks the weights into anonymous memory by default, which on 16 GB means swapping; our engine
keeps the hot experts resident and reads only the cold ones (resident_hit 0.98).

**PC, in a 10 GiB cgroup with no swap** (`systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0`, page cache
dropped first, 8 threads):

| | Prompt tok/s | Generation tok/s |
|---|---|---|
| ours before (plain mmap) | 4.91 | 4.40 |
| ours before, `--resident-experts --expert-ram-gib 6` (advice only) | 2.77 | 2.36 |
| ours now, plain | 9.68 | 5.32 |
| ours now, `--resident-experts --expert-profile` (auto budget: 6.3 GiB experts + 1.1 GiB always-used, copied) | **11.28** | **5.88** |
| llama.cpp default (repacks into RAM) | killed: out of memory | |
| llama.cpp `--no-repack` (plain mmap) | 31.2 | 13.4 |

**Mac mini M4, 16 GB, this build** (4 threads): ours plain 9.32 / 5.67, with `--resident-experts --expert-ram-gib 8` and the
profile 10.73 / 6.02 (no swapouts); llama.cpp `llama-completion -nr` (plain mmap, no repack) 17.31 / 3.58 (no swapouts).
On macOS llama.cpp's mmap path reads the prompt 1.6x faster and we generate 1.7x faster. The time split there
(`AM2_TIMES=1`): MLA 24.1 s, MoE 30.7 s of 57 s, so the attention needs the same work as the experts.

Parity with llama.cpp (tools/reference-llamacpp/parity.sh, 32 steps of real text): 30 identical and 2 near-ties, worst
logit difference 3.41 (the build before: 31 + 1, 4.19). On a synthetic prompt of consecutive ids the first token flips
between builds; those are near-ties of a nearly flat distribution.

The same model with everything in RAM: ours before 6.3 / 6.1, ours now 15.4 / 7.9, llama.cpp 76 / 18.3. So the tiers
work (capped runs at 75-90% of in-RAM speed) and the remaining gap to llama.cpp is compute on this model, not reading.

What changed:
- **MoE and MLA prompts are batched.** `forward_batch` took only dense attention and feed-forwards, so an MoE prompt went
  token by token and each token pulled its experts again. Now a chunk's tokens are grouped by expert (each chosen expert is
  read once and multiplied for all its tokens), MLA's projections take the whole chunk, and blocks that cannot batch (Mamba,
  GDN, QSA) run token by token inside it. Each token's expert sum is added in its own routing order, so the tokens are
  identical with and without batching (checked on GLM and Ministral).
- **Integer kernels for Q5_K and Q8_0** (cpu_q8.rs). MLA's matrices are Q8_0 and some experts Q5_K; they took the float
  path, which decodes each row again for every token. One input is the many-input function, so single and batched results
  are identical.
- **No MADV_RANDOM on cold experts.** It switched off readahead: an expert is megabytes in one piece, and it was read 4 KiB
  per fault (5.8 M major faults instead of 0.2 M). macOS got only that hint for its cold experts.
- **Residency that holds.** Where mlock is refused (the usual 8 MiB memlock limit), the always-used matrices and the
  resident experts are copied into the engine's own memory (`pinned=copy`; `AM2_RESIDENT_COPY=0` for the old advice):
  file pages are evicted under pressure, so advice alone read 25 GB a request with 95% of the routing "resident". The
  automatic budget subtracts the always-used matrices and reads a cgroup's limit.
- **The chosen experts are asked for at once** (MADV_WILLNEED before multiplying, on with `--resident-experts`,
  `AM2_PREFETCH=0/1`).
- `AM2_TIMES=1` prints each request's time by block kind (`INFO times_ms=mla=...,moe=...`), and `tools/hybrid-bench.sh`
  passes engine flags after `--`.

### One parallel pass per step for the experts and MLA's heads

Generating a token was about 1,000 matrix-vector products, each its own parallel job (and the MoE layer's experts one
after another). `QMat::mul_set` multiplies several matrices in one pass over all their rows (ggml's mul_mat_id idea):
the chosen experts' up and gate with the shared expert's, then their downs; MLA's q_a with kv_a, and the 20 heads'
absorbed k_b and v_b. Same row functions on the same inputs, so the tokens are identical (`AM2_NO_FUSED=1` for the old
path). GLM-4.7-Flash in RAM, 64 tokens: 7.83 s -> 6.3-6.4 s (8.2 -> 10.0 tok/s).

Kernel throughput, one thread (`cargo test --release -- --ignored kernel_speed`), this PC (AVX2): one input 10-15 GMAC/s
(Q4_K 5.6 GB/s of weights), 8 inputs only 16-22 GMAC/s. The prompt is bound by that (chunks of 128-512 help MoE only a
little: 22.6 -> 18.5 s of a 512-token GLM prompt), so the next step is explicit SIMD kernels.

### Hand-written SIMD: AVX2 on x86, NEON sdot on Apple silicon

Q4_K (most experts) and Q8_0 (MLA) get hand-written kernels: AVX2 maddubs on x86, the dot-product instruction (sdot) on
aarch64. The integer sums are exact and the float arithmetic is the scalar kernels' in the same order, so they equal the
baseline build bit for bit (test: `this_cpus_kernels_equal_the_baseline_exactly`, passing on both machines).

| Single thread, GMAC/s (one input / eight) | PC, AVX2: before | after | Mac M4: before | after (NEON) |
|---|---|---|---|---|
| Q4_K | 10.0 / 15.9 | 17.0 / 32.3 | 15.9 / 19.7 | 33.5 / 64.9 |
| Q8_0 | 11.3 / 21.3 | 11.3 / 23.8 | 16.1 / 38.9 | 20.1 / 38.9 |

(Each machine keeps the compiler's loop where it is faster: one-input Q8_0 on x86, eight-input Q8_0 on the Mac.)

**GLM-4.7-Flash on the 16 GB Mac** (17.5 GB model; `--resident-experts --expert-ram-gib 8` with the profile; no swapouts):

| Threads | Prompt tok/s | Generation tok/s |
|---|---|---|
| 4 | 13.81 | 7.26 |
| 6 | 15.72 | 8.46 |
| 8 | 16.52 | 8.93 |
| 10 | 16.85 | 8.93 |

From 7.0 / 5.9 this morning (the build before #78). llama.cpp `--no-repack` at 4 threads: 17.3 / 3.6. So past the memory
limit the Mac now reads the prompt about as fast as llama.cpp and generates 2.5x faster. 8 threads is the sweet spot
on the M4 (4 performance + 6 efficiency cores). These Mac runs had eight-input Q8_0 on NEON (since moved back to the
loop), so MLA's prompt share there should drop on the next run.

## Gated delta net prompts, and a chunk's experts in one pass (Tiel)

Tiel (qwen35moe: 40 layers, 3 in 4 a gated delta net, 256 experts with 8 chosen) read its prompt barely faster batched,
because each chunk's experts ran as ~250 small parallel jobs one after another and the GDN layers ran token by token.
Now the GDN input and output projections take the whole chunk (the recurrence stays token by token), and a chunk's
experts run in two parallel passes (`QMat::matmul_set`: every expert's up and gate on its tokens, then every down).
128-token prompt, 8 threads, this PC: **Tiel 13.0 -> 6.8 s, GLM-4.7-Flash 11.9 -> 8.2 s**, tokens identical batched
and not. Tiel against llama.cpp (`parity.sh`, oracle3, 24 steps of real text): **24/24 identical**, worst logit
difference 1.28.

## Another machine's memory as an expert tier (remote.rs)

Whichever machine runs the model lends nothing; another paired machine lends RAM as an expert host. Tokens identical
with and without it (each expert's output comes back separately and is added in routing order), mutual TLS with the
cluster's pins, failure falls back to local computation. GLM-4.7-Flash, master 3 GiB resident + host 4 GiB:

| Pairing | Result |
|---|---|
| PC to PC (two processes) | identical tokens batched and not; prompt 11.65 -> 7.3 s (the host's threads add compute) |
| PC to PC, host killed mid-prompt | marked down after 91 calls, identical tokens |
| Unpaired master, wrong host pin | refused at the TLS handshake; the model runs on locally |
| Mac to Mac (two processes, 16 GB) | identical 64 tokens, 397 experts and 1,243 calls remote (slower: one machine, swap) |
| PC master, Mac host (4 GiB) | identical tokens batched and not (x86 and Apple silicon give the same bits); no gain, as this PC holds the model in RAM |
| **Mac master, PC host (8 GiB)** | identical tokens; see below |

**The 16 GB Mac running GLM-4.7-Flash (17.5 GB) with 8 GiB of the PC's RAM as its next tier** (512-token prompt, 64
tokens, 8 threads, gigabit LAN; tokens identical in every pair):

| Mac's own RAM for experts | Prompt tok/s, alone -> with the PC | Generation tok/s | Pages read from the SSD |
|---|---|---|---|
| 3 GiB | 15.52 -> **23.83** (+54%) | 8.83 -> 8.75 | 1.20 M -> 50 K |
| 6 GiB | 16.46 -> **25.49** (+55%) | 9.40 -> **10.40** (+11%) | 1.05 M -> 8 K |

Two bugs these runs found, both fixed: the host kept a previous master's experts and refused a different set (it now
lets go of what is no longer wanted), and `f32::exp` differs in the last bit between glibc and Apple's libm, which
flipped a near-tie once enough experts were remote (the engine now has a portable `exp`, `exp_f32`, whose bits are
checked by a hash test that passes on both machines).


## Next
- **Explicit SIMD integer kernels** (AVX2 maddubs here, NEON sdot on Apple silicon; integer sums are exact, so they can match the current results bit for bit), several rows by several inputs per step; Q4_0 the same way (MoE is now the largest share of a GLM prompt: 42 ms a token).
- **The Metal (wgpu) path on the Mac:** it is bound by host-side waits, not by the GPU (FINDINGS item 6).
