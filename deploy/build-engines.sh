#!/usr/bin/env bash
# Build PolyStrata's engines (strata, strata-poly, strata-vision) from its source, for RTX 30 and RTX 50 cards.
#
#   deploy/build-engines.sh [--strata-home DIR] [--archs "86;120"] [--jobs N] [--install]
#
# Needs: PolyStrata's source in --strata-home (git clone https://github.com/VecSzn/PolyStrata, with its third_party
# folders), CUDA (nvcc, default /usr/local/cuda), CMake and Ninja. Builds into build-am2-*/; --install copies the
# binaries into engine/ and engine-poly/, keeping the old ones in engine-backup-<date>/.
# Architectures: 86 = RTX 30, 89 = RTX 40, 120 = RTX 50 (several: "86;89;120").
set -euo pipefail
R="${STRATA_HOME:-$HOME/Development/PolyStrata}"
ARCHS="86;120" J="$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))" INSTALL=0
CUDA="${CUDA_HOME:-/usr/local/cuda}"
while [ $# -gt 0 ]; do
  case "$1" in
    --strata-home) R="$2"; shift ;;
    --archs) ARCHS="$2"; shift ;;
    --jobs) J="$2"; shift ;;
    --install) INSTALL=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
[ -f "$R/CMakeLists.txt" ] || { echo "no PolyStrata source in $R (git clone https://github.com/VecSzn/PolyStrata $R)" >&2; exit 1; }
NINJA="$(command -v ninja || echo "$R/.venv/bin/ninja")"
[ -x "$NINJA" ] || { echo "Ninja is needed (sudo apt install ninja-build)" >&2; exit 1; }
[ -x "$CUDA/bin/nvcc" ] || { echo "nvcc not found in $CUDA/bin (set CUDA_HOME)" >&2; exit 1; }
VISION_LLAMA="$(ls -d "$R"/third_party/llama.cpp-* 2>/dev/null | head -1)"; VISION_LLAMA="${VISION_LLAMA:-$R/third_party/llama.cpp}"
COMMON=(-G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCHS"
        -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA")
ENG=(-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR="$R/third_party/llama.cpp")
step() { printf '\033[1;32m=== %s %s\033[0m\n' "$(date +%T)" "$*"; }
build() { nice -n 10 cmake --build "$1" --target "$2" -j "$J" || nice -n 10 cmake --build "$1" --target "$2" -j "$J"; }

step "strata (Qwen3.8 / Orca), CUDA archs $ARCHS"
cmake -S "$R" -B "$R/build-am2" "${COMMON[@]}" "${ENG[@]}" -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON >/dev/null
build "$R/build-am2" strata
step "strata-poly (Tiel, MiniCPM-V, LFM2, GLM, Gemma, DeepSeek)"
cmake -S "$R" -B "$R/build-am2-poly" "${COMMON[@]}" "${ENG[@]}" >/dev/null
build "$R/build-am2-poly" strata-poly
step "strata-vision (image encoder)"
cmake -S "$R/tools/vision" -B "$R/build-am2-vision" "${COMMON[@]}" -DLLAMA_DIR="$VISION_LLAMA" -DSTRATA_VISION_CUDA=ON -DSTRATA_PORTABLE=OFF >/dev/null
build "$R/build-am2-vision" strata-vision

if [ $INSTALL = 1 ]; then
  B="$R/engine-backup-$(date +%Y%m%d-%H%M)"; mkdir -p "$B" "$R/engine" "$R/engine-poly"
  cp -a "$R/engine" "$R/engine-poly" "$B/" 2>/dev/null || true
  cp "$R/build-am2/strata" "$R/engine/"; [ -f "$R/build-am2/libstrata_kq.so" ] && cp "$R/build-am2/libstrata_kq.so" "$R/engine/"
  cp "$R/build-am2-vision/bin/strata-vision" "$R/engine/"
  cp "$R/build-am2-poly/strata-poly" "$R/build-am2-vision/bin/strata-vision" "$R/engine-poly/"
  step "installed into engine/ and engine-poly/ (the old ones: $B). Stop a running server before you swap engines."
else
  step "built: build-am2/strata, build-am2-poly/strata-poly, build-am2-vision/bin/strata-vision (--install copies them)"
fi
