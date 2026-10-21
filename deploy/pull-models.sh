#!/usr/bin/env bash
# Download the models the shipped run configs (configs/) use, into the folders they name.
#
#   deploy/pull-models.sh tiel|minicpm|orca|all [--strata-data DIR] [--token hf_...]
#
# tiel     Tiel Coder 35B-A3B UD-Q6_K_XL        peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP       30 GB
# minicpm  MiniCPM-V 4.7 35B-A3B Uncensored     AtomicChat/MiniCPM-V-4.7-35B-A3B-Uncensored-GGUF   23 GB
# orca     Qwen3.8 Flash Next Uncensored IQ3_XXS orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF     85 GB, gated:
#          accept its terms on huggingface.co, then pass --token (or set HF_TOKEN). Then its native pack and the MTP
#          draft runtime are made (am2strata-run iq-pack, mtp-fetch, mtp-pack, mtp-rt; ~5 GB more download).
#
# Downloads resume, and every file is checked against the SHA-256 Hugging Face publishes. The tokenizer is written
# from the model file the first time the server loads it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STRATA_DATA="${STRATA_DATA:-$HOME/Development/Strata-data}"
STRATA_HOME="${STRATA_HOME:-$HOME/Development/PolyStrata}"
WHICH="${1:-}"; [ -n "$WHICH" ] && shift
TOKEN=()
while [ $# -gt 0 ]; do
  case "$1" in
    --strata-data) STRATA_DATA="$2"; shift ;;
    --token) TOKEN=(--token "$2"); shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
RUN="$ROOT/target/release/am2strata-run"
[ -x "$RUN" ] || (cd "$ROOT" && cargo build --release --bins)
pull() { "$RUN" pull "$@" --data-dir "$STRATA_DATA" --no-config "${TOKEN[@]}"; }

tiel()    { pull peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP --quant UD-Q6_K_XL --dir tiel-UD-Q6_K_XL; }
minicpm() { pull AtomicChat/MiniCPM-V-4.7-35B-A3B-Uncensored-GGUF --quant AD-Q5_K-Q4_K --dir minicpmv-uncensored-AD-Q5_K-Q4_K; }
orca() {
  pull orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF --quant IQ3_XXS --dir orca-IQ3_XXS
  local m="$STRATA_DATA/models/orca-IQ3_XXS/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf"
  # Orca's engine reads a native pack and the MTP draft runtime: made here in Rust (am2strata-run iq-pack, mtp-fetch,
  # mtp-pack, mtp-rt: ports of PolyStrata's Python tools, writing the same files byte for byte)
  if [ ! -f "$STRATA_DATA/packs/orca-iq3_xxs/native_experts.txt" ]; then
    "$RUN" iq-pack --gguf "$m" --out "$STRATA_DATA/packs/orca-iq3_xxs" --compat-bf16
  fi
  if ! "$RUN" mtp-fetch verify --out "$STRATA_DATA/mtp" 2>/dev/null || [ ! -f "$STRATA_DATA/mtp/mtp-manifest.json" ]; then
    "$RUN" mtp-fetch fetch --out "$STRATA_DATA/mtp"            # ~5 GB of the BF16 checkpoint's MTP tensors, by range reads
  fi
  [ -f "$STRATA_DATA/mtp/mtp-q2_0.gguf" ] || "$RUN" mtp-pack --src "$STRATA_DATA/mtp" --experts q2_0 --out "$STRATA_DATA/mtp/mtp-q2_0.gguf"
  [ -f "$STRATA_DATA/mtp/rt/experts.bin" ] || "$RUN" mtp-rt --gguf "$STRATA_DATA/mtp/mtp-q2_0.gguf" --out "$STRATA_DATA/mtp/rt"
  # the draft vocabulary is a data file of PolyStrata's repository
  if [ ! -f "$STRATA_DATA/mtp/rt/draft_vocab.bin" ]; then
    if [ -f "$STRATA_HOME/data/draft_vocab.bin" ]; then cp "$STRATA_HOME/data/draft_vocab.bin" "$STRATA_DATA/mtp/rt/"
    else echo "copy PolyStrata's data/draft_vocab.bin to $STRATA_DATA/mtp/rt/ (without it the engine drafts by lookup only)"; fi
  fi
}
case "$WHICH" in
  tiel) tiel ;; minicpm) minicpm ;; orca) orca ;;
  all) tiel; minicpm; orca ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
