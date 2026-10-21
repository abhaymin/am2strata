#!/usr/bin/env bash
# Download the models the shipped run configs (configs/) use, into the folders they name.
#
#   deploy/pull-models.sh tiel|minicpm|orca|all [--strata-data DIR] [--token hf_...]
#
# tiel     Tiel Coder 35B-A3B UD-Q6_K_XL        peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP       30 GB
# minicpm  MiniCPM-V 4.7 35B-A3B Uncensored     AtomicChat/MiniCPM-V-4.7-35B-A3B-Uncensored-GGUF   23 GB
# orca     Qwen3.8 Flash Next Uncensored IQ3_XXS orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF     85 GB, gated:
#          accept its terms on huggingface.co, then pass --token (or set HF_TOKEN). Orca also needs a pack and the
#          MTP runtime built with PolyStrata's tools: the script prints the commands (docs/DEPLOYMENT.md, step 4).
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
  if [ ! -f "$STRATA_DATA/packs/orca-iq3_xxs/native_experts.txt" ] || [ ! -d "$STRATA_DATA/mtp/rt" ]; then
    cat <<MSG

Orca's engine needs a pack and the MTP runtime, made once with PolyStrata's tools (in $STRATA_HOME):

  cd $STRATA_HOME
  .venv/bin/python tools/iq_pack.py --gguf $m --out $STRATA_DATA/packs/orca-iq3_xxs --compat-bf16
  .venv/bin/python tools/mtp_fetch.py fetch --out $STRATA_DATA/mtp
  .venv/bin/python tools/mtp_pack.py --src $STRATA_DATA/mtp --experts q2_0 --out $STRATA_DATA/mtp/mtp-q2_0.gguf
  .venv/bin/python tools/mtp_rt.py --gguf $STRATA_DATA/mtp/mtp-q2_0.gguf --out $STRATA_DATA/mtp/rt
  cp data/draft_vocab.bin $STRATA_DATA/mtp/rt/draft_vocab.bin

(PolyStrata's docs/ORCA.md explains each step.)
MSG
  fi
}
case "$WHICH" in
  tiel) tiel ;; minicpm) minicpm ;; orca) orca ;;
  all) tiel; minicpm; orca ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
