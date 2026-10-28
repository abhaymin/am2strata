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
