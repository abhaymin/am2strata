# Every GPU, in Rust — plan

Goal: am2strata's own engines run on NVIDIA (CUDA), AMD (ROCm/HIP, discrete and integrated) and any other GPU through
Vulkan, several at once, from Rust code. C or C++ only where it is integration (a driver's runtime, an existing kernel
library called through FFI), never as the engine itself.

## Where things stand

| Engine | Language | Runs on today |
|---|---|---|
| `am2strata-hybrid` (Nemotron-H, GLM-4.7-Flash, Mistral/Mixtral, Qwen3) | Rust | CPU only |
| `am2strata-media` (images, video, speech; candle + mold) | Rust | NVIDIA (CUDA) or CPU |
| `strata`, `strata-poly` (PolyStrata's) | C++ / CUDA (HIP for strata-poly) | NVIDIA; strata-poly also AMD |

This PC: RTX 5060 Ti 16 GB (CUDA, Vulkan) + Radeon 780M integrated (gfx1103: ROCm and Vulkan/RADV), 60 GB RAM.

## The choice: CubeCL

[CubeCL](https://github.com/tracel-ai/cubecl) (Burn's GPU language; Apache-2.0/MIT) writes a kernel once as Rust
(`#[cube]` functions) and compiles it at run time for:
- **CUDA** (NVIDIA): through NVRTC;
- **HIP** (AMD ROCm): through hiprtc;
- **wgpu**: WGSL/SPIR-V on **Vulkan**, which covers integrated AMD and Intel and NVIDIA without its toolkit;
- a **CPU** runtime.

Stable 0.10, 0.11 in pre-release. The kernels stay Rust; the C/C++ part is only the vendors' runtimes it calls,
which fits "C/C++ only for integration".

Considered and set aside:
- candle's own backends: CUDA and Metal only, no ROCm, no Vulkan;
- raw `wgpu` compute shaders: one backend (Vulkan), and the WGSL would be hand-written;
- llama.cpp's Vulkan/HIP backends through FFI: C++ engines, the opposite of the goal.

## Phases

1. **Device layer** (`crates/strata-gpu`):
   - enumerate devices across the CUDA, HIP and wgpu runtimes (name, VRAM, integrated or discrete, unified memory);
   - one `Device` trait for buffers, uploads and kernel launches;
   - shown in Monitor and `/health`.
   - Gate: lists both GPUs here.
2. **Quantized kernels for the hybrid engine**, in CubeCL:
   - matrix × vector and matrix × matrix for the GGUF types the models here use (Q4_K, Q5_K, Q6_K, Q8_0, IQ3_XXS, F16, BF16, F32);
   - RMSNorm, RoPE, SiLU/SwiGLU, softmax, top-k routing, attention with an f16 KV cache;
   - the Mamba2 scan (Nemotron-H) and MLA (GLM).
   - Gate: each kernel matches the CPU path to 1e-3 relative on random data, on CUDA, HIP and Vulkan.
3. **One GPU end to end**: the hybrid engine with `--device cuda:0|hip:0|vulkan:N`.
   - Gate: the existing parity gate (teacher-forced against llama.cpp, near-ties allowed) passes on each backend.
   - Speed measured against the CPU path and against llama.cpp's own CUDA/Vulkan builds.
4. **Several GPUs**, the strata placement across devices:
   - layers split by each device's memory;
   - hot experts (from the STRP profile) on the fastest device, warm ones on the integrated GPU, which shares system RAM;
   - cold ones on the CPU.
   - Gate: Nemotron Bolt or GLM-4.7-Flash split over the 5060 Ti and the 780M, faster than either alone, same tokens.
5. **Media on AMD/Vulkan**: candle has no ROCm. The speech models (Whisper, Parler) are small enough to move to the
   same CubeCL kernels. Diffusion (mold) stays CUDA/CPU until candle gains a backend; noted as the exception.
6. **The server**: per-device placement in the run config (`"devices": ["cuda:0", "vulkan:1"]`), the queue's
   resident model per device (a media job on the NVIDIA card while text runs on the AMD one), and the Settings UI.

## Risks

- **IQ3_XXS and the k-quants on Vulkan:** subgroup operations differ by driver (RADV, NVIDIA). Each kernel has a
  plain fallback.
- **The 780M shares system RAM:** it helps only for weights that would otherwise sit in RAM read by the CPU. Measure
  before relying on it.
- **CubeCL 0.11 is pre-release:** pin a version, and keep the CPU path as the reference.

## Status (2026-10-29)

**Phases 1–3 are built and gated. Phase 4's placement across several devices is built.**

**Pieces**
- `crates/strata-gpu`: the kernels and the device layer, on CubeCL 0.10, the last release for this rustc (1.93).
- `am2strata-hybrid --devices cuda:0=13,hip:0=8`: GiB per device, in order of preference.
- Build: `tools/build-hybrid.sh`. It is memory-capped and includes a hipconfig shim: Ubuntu's HIP is 7.1.52801,
  and cubecl-hip-sys ships 7.1.52802's bindings.

**Devices found on this PC**
- `cuda:0` (RTX 5060 Ti);
- `hip:0` (Radeon 780M, gfx1103: ROCm 7.1 runs it without an override);
- `vulkan:discrete` and `vulkan:integrated` (the same two GPUs through RADV / NVIDIA's Vulkan);
- CubeCL's CPU runtime.

**Gates**
- **Kernels:** F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K and Q6_K against the CPU dequantizer on every device. Worst
  relative error 7.6e-8; the limit is 1e-3.
- **Engine:** Ministral-3 3B Q4_K_M teacher-forced against llama.cpp. 24/24 steps identical on cuda:0, hip:0,
  vulkan:discrete and vulkan:integrated, with logit differences as on the CPU (≤ 0.21).

**Decode speed (tok/s)**

| Model | CPU engine | cuda:0 | vulkan (5060 Ti) | hip:0 (780M) | vulkan (780M) |
|---|---|---|---|---|---|
| Ministral-3 3B Q4_K_M (all on the device) | 2.2 | 15.7 | 11.9 | 8.8 | 7.2 |
| GLM-4.7-Flash UD-Q4_K_XL (17.6 GB; cuda:0 = 13 GB, the rest on the CPU) | 1.84 | 9.09 | – | – | – |

GLM-4.7-Flash split over two GPUs, nothing on the CPU (cuda:0 13 GB, the 780M 2.9 GB):

| Split | tok/s |
|---|---|
| cuda:0 + hip:0 | 9.24 |
| cuda:0 + vulkan:integrated | 8.23 |

It is barely faster than leaving the rest on the CPU: each product is still a round trip.

Parity after the fused CPU products and the batched GPU products:
- Ministral-3 on the CPU: 23/24 steps identical and 1 near-tie.
- Ministral-3 on cuda:0: 24/24.
- GLM-4.7-Flash split over cuda:0 and hip:0: 16/16 identical, but its worst logit difference is 2.44, larger than
  Ministral's (about 0.2) - to look into.

The CPU numbers are after the fused CPU products (cpu_dot.rs). Before them, GLM on the CPU took over 8 s a token.

**Next**
- Keep activations on the device between layers. Every product is still its own round trip (0.1–0.6 ms); batching
  q/k/v and gate/up halves that.
- Wider loads in the k-quant kernels: about 70 GB/s of the card's ~448 GB/s.
- Speech kernels for AMD (phase 5).
- The server's per-device placement in the run config and the UI (phase 6).
