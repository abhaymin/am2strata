# Strata 0.1.45 to 0.2.8: what matters for am2Strata

Survey of Strata's release notes, 2026-10-30, for roadmap item 4 in [V2-PLAN.md](V2-PLAN.md). PolyStrata and its engines carry
Strata 0.1.45, so none of the engine work below has reached this PC. The figures are Strata's own, from its release notes,
not measured here. Target machine: two RTX 5060 Ti 16 GB (sm_120), one on OCuLink and one on USB4, 60 GB of RAM.

## Worth having here

| Release | Change | Strata's figure | Why it applies |
|---|---|---|---|
| 0.2.5 | Fewer GPU launches per layer | decode +13% on one test PC (5 of 5 pairs, same output) | NVIDIA decode path |
| 0.2.5 | The CPU starts on the experts it already has while the rest are read from the SSD | +13.5% on an RTX 5070, +11-15% on an RTX 3060 | Orca is larger than VRAM |
| 0.2.5 | The RAM store can hold every expert (it stopped at 48 GB) | not comparable (251 GB machine) | Check against 60 GB of RAM first |
| 0.2.5 | The same prompt always takes the same path, chosen by its length | reproducible runs | Helps every parity test |
| 0.2.0-beta | Linux: experts read in place from the GGUF (zero-copy page cache), I/O prefetch with a gate | RAM and SSD | Orca reads a large model from disk |
| 0.2.2 | Graphics cards can be Main, Helper or Off | not a speed figure | Fits a fast OCuLink card with a slower USB4 one |
| 0.2.3 | DFlash2 drafting (opt-in) instead of MTP | not stated here | Worth one measurement |
| 0.2.8 | Paired decode gain on an RTX A6000, Qwen IQ3_S | 105.9 to 111.4 tok/s (story), 139.8 to 145.2 (code) | The latest engine as a whole |

## Packs

Strata 0.2's Hadamard-folded packs (native_experts v5) and the trellis expert types (TQ2_T, TQK6, TQK7, the Gyro models) need
engine support. am2Strata refuses v5 packs by name until an engine reads them. Existing packs here keep working.

## Not for this PC

SYCL / Intel Arc, Metal / Apple silicon, HIP / AMD Radeon and Strix Halo kernels, four-card profiles, the Windows and Docker
packaging, the desktop app, the installer and the signed update feed. The server-side fixes of 0.2.8 are ported already.
The 0.2.7 routing speed-up for cards of 12 GB or less is off again in 0.2.8 and does not apply to 16 GB cards.

## Recommendation

**Follow PolyStrata; do not port the C++ engine to Rust.** These are CUDA kernel and scheduling changes tied to Strata's
engine, and they arrive whole when PolyStrata merges Strata 0.2, after which the rebuild here is the same as for poly.6
(`git pull --ff-only`, build, swap with a rename). `am2strata-run upstream-check` shows when that happens. Rewriting them in
Rust would duplicate work that is still moving upstream every few days.

Two things are am2Strata's own to do meanwhile:
- the Main / Helper idea for two unequal links: the order of the cards and the layer split are already settings in a run
  config, so this can be measured with `"gpu"` and `"layer_split"` before any code;
- `am2strata-hybrid` stays the long-term Rust engine (docs/GPU-PLAN.md) and can take ideas from this list one at a time, each
  behind the usual gate.
