#!/usr/bin/env bash
# Build am2strata-hybrid (the Rust engine) with its GPU kernels (crates/strata-gpu: CUDA, ROCm/HIP, Vulkan).
# Ubuntu's HIP 7.1.52801 needs the hipconfig shim first in PATH (cubecl-hip-sys has 7.1.52802's bindings).
# The build is memory-capped so it cannot push a loaded model's server out of RAM: MEM (default 2500M), JOBS (2).
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
exec systemd-run --user --scope --quiet -p MemoryMax="${MEM:-2500M}" -p MemorySwapMax=0 --setenv=PATH="$here/tools/hip-shim:$PATH" \
    nice cargo build --release -j "${JOBS:-2}" --manifest-path "$here/crates/strata-hybrid/Cargo.toml" "$@"
