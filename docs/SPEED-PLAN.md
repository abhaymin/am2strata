# MoE speed: basalt's ideas, for every GPU (owner, 2026-11-19)

basalt (github.com/jesdga95/basalt, MIT) is a Strata downstream that runs big MoE models fast with their experts
mostly in system RAM. Its kernels are CUDA for Blackwell (sm_120) only. The owner's rule: **take its ideas into
am2Strata, but every feature must work on every GPU am2Strata supports, not just RTX 50x0.**

am2Strata's engine (strata-hybrid, kernels in strata-gpu) is already written for that. Its kernels are **CubeCL**:
one Rust source compiled at run time for CUDA (NVIDIA, any generation), HIP (AMD ROCm) and wgpu (any Vulkan, Metal
or DX12 GPU, integrated ones included), with a native Metal path for Apple GPUs. So each feature below is built on
CubeCL and host-side Rust, never as backend-specific code. A backend-specific fast path, such as a CUDA graph, is
only ever an option on top of a portable path that works without it.

## Where the speed is today (measured 2026-11-19; see the basalt investigation)

| Engine | Orca (qwen4exp, IQ3_XXS) on 2x RTX 5060 Ti |
|---|---|
| the live C++ engine (PolyStrata, CUDA only) | ~72 tok/s mean (median 84); 91-98% of experts from VRAM; 84% draft acceptance, 1.24 drafts per window |
| am2Strata's Rust engine | ~0.4 tok/s, CPU only: no IQ3_XXS or IQ2_S GPU kernel, so no expert can go on a GPU |
| basalt, self-reported | conversation-like decode ~90-165 tok/s, mostly from a VRAM expert cache plus drafting |

## The features, each for every backend

| # | Feature (basalt's idea) | Portable design | Notes for each backend |
|---|---|---|---|
| 1 | **Kernels for every quant the big MoE models use**: IQ3_XXS, IQ3_S, IQ2_S, IQ2_XS, IQ4_XS and the down-projection's type | CubeCL kernels that dequantize as they read, like the k-quants already do, with the grids and sign tables in constant arrays. Checked against `strata_pack::quants` on the CPU to 1e-3 | All backends; integer tricks only where CubeCL has them on every target |
| 2 | **A VRAM expert cache** (experts placed one by one by how often they are used) | One slot buffer per device, its size from the device's free memory less what the dense layers and KV need, filled in the order of the model's profile (strata-hybrid already ranks by profile), with adaptive swaps as the routing changes | On a **unified-memory** device (Apple, AMD APUs, Intel iGPU) the "cache" is a view onto host memory, with no copy at all: read through a shared buffer |
| 3 | **A whole decode window on the device** | Activations stay on the device across the layer; the GDN, attention, router, shared expert and cached experts run as one queue of launches, with no host round trip per product. Today there are about 1,000 round trips per token | CUDA graphs and HIP graphs become an optional replay on top; wgpu and Metal batch the launches into one command buffer |
| 4 | **Several cards that each hold different experts** (basalt's peer tiers, with no GPU-to-GPU link) | Each device holds its own set of experts; partial results come back through mapped host memory and are summed on the main device. The split is set by each card's free memory and measured speed | Works across vendors and backends (an NVIDIA card with an AMD card, a dGPU with an iGPU), because it needs only host-mapped buffers |
| 5 | **Misses handled three ways at once**: cached ones on the GPU, some copied over PCIe and run on the GPU, the rest on the CPU | The PCIe share comes from a measured link probe (bytes per second for each device), not a constant | Unified-memory devices skip the copy share |
| 6 | **Drafting**: MTP, a suffix/n-gram drafter, a cost-aware window length | Host-side Rust: the MTP block is loaded but not run yet; prompt lookup exists on the CPU. The window length comes from measured window costs | Backend-independent |
| 7 | NVFP4 | Not portable (it needs Blackwell's FP4 tensor cores): it is part of the Blackwell tier below, for the cards where it is faster | Blackwell only |

## Blackwell: basalt's way, as a fast tier (owner, 2026-11-19)

On an NVIDIA Blackwell card (compute capability 12.x, sm_120: the RTX 50x0 series and RTX PRO Blackwell), the
engine takes **basalt's own approach** as a fast tier. It is picked at run time from the device's compute
capability, and every other GPU keeps the portable path above. The portable path stays the reference: the Blackwell
tier must give the same tokens (the parity gate) and be measurably faster, or it is not used.

| Blackwell feature (as basalt does it) | In am2Strata |
|---|---|
| Kernels tuned for sm_120: its tile sizes, warp layout and async copies (cp.async / TMA) for the expert products | CubeCL's CUDA target first, tuned for sm_120, with tensor-core matrix ops (CubeCL `cmma`) for prefill |
| **NVFP4 weights** on Blackwell's FP4 tensor cores | Optional. A Workshop quantize step writes NVFP4 (our own GGUF writer), and an FP4 matrix kernel runs where the card has FP4 tensor cores. On 16 GB cards basalt measured no gain (156.6 against IQ3_XXS's 229.6), so it is for the bigger cards (32 GB and up), measured first |
| basalt's tuned decode settings for sm_120: draft depth, window cost table, PCIe share | The same host logic as the portable path, with sm_120's measured costs as its starting table |

**The open choice (the owner's):** basalt's fastest kernels are hand-written CUDA C++. am2Strata is all Rust (its
GPU code is CubeCL).
- **(a) CubeCL only.** The Blackwell tier is written in CubeCL with its CUDA-only features. It stays all Rust, but
  may fall short of basalt where CubeCL cannot express a Blackwell instruction (FP4 MMA, TMA).
- **(b) CubeCL, plus a small optional CUDA crate for the instructions CubeCL lacks.** basalt's MIT kernels are
  ported with its notice, built only with `--features blackwell`, and checked against the CubeCL path.
- The recommendation is (a), going to (b) only where a measured gap remains.

**Live C++ engine, now (CUDA only, while the Rust engine catches up):** cheap A/B tests in an idle window: `--spec`
6/8 with a lower `--spec-min-p`, `--pipeline-windows 2`, and a shorter context to give VRAM to experts.

## Gates (every feature, every backend that is here)

- **Parity:** the token-for-token gate against the oracle (`parity.sh`, oracle3), on a prompt past 2,051 tokens (where
  qwen4exp's sparse attention still fails today).
- **Speed:** tokens per second on `golden/orca-requests.json` against the live C++ engine, on each backend:
  - **CUDA:** the PC's 2x RTX 5060 Ti;
  - **Vulkan (wgpu):** the PC's Radeon 780M iGPU, and the 5060 Tis through Vulkan;
  - **HIP:** the 780M through ROCm, where it builds;
  - **Metal:** the Mac mini M4, with a model that fits its 16 GB.
- **No regressions:** models that already run (Ministral on the Mac, the dense models) stay as fast or faster.

## Order

1. Kernels (feature 1), so experts can go on any GPU at all.
2. The expert cache (2), with unified memory.
3. The device window (3).
4. Several cards (4), and misses three ways (5).
5. Drafting (6).

Each lands with its parity and speed numbers in docs/HYBRID-SPEED.md. Where code follows basalt closely, basalt's MIT
notice goes in NOTICE.
