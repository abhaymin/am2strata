#!/usr/bin/env bash
# Install am2Strata on this machine: build it, put the run configs in place, optionally make it the boot service.
#
#   deploy/install.sh [--strata-home DIR] [--strata-data DIR] [--port N] [--host ADDR]
#                     [--api-key KEY | --new-key] [--model ID] [--service] [--force]
#
# --strata-home  PolyStrata's folder: its engines (engine/, engine-poly/) and the run configs (default ~/Development/PolyStrata)
# --strata-data  where models and packs live (default ~/Development/Strata-data)
# --host         127.0.0.1 (default: this machine only) or 0.0.0.0 (your network: set an API key too)
# --api-key KEY  clients must send "Authorization: Bearer KEY"; --new-key makes a random one and prints it
# --service      start at boot as the systemd user service am2strata-boot (needs: loginctl enable-linger $USER)
# --force        overwrite run configs that already exist
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STRATA_HOME="${STRATA_HOME:-$HOME/Development/PolyStrata}"
STRATA_DATA="${STRATA_DATA:-$HOME/Development/Strata-data}"
PORT=8000 HOST=127.0.0.1 KEY="" MODEL="" SERVICE=0 FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --strata-home) STRATA_HOME="$2"; shift ;;
    --strata-data) STRATA_DATA="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --host) HOST="$2"; shift ;;
    --api-key) KEY="$2"; shift ;;
    --new-key) KEY="am2-$(head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32)" ;;
    --model) MODEL="$2"; shift ;;
    --service) SERVICE=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
say() { printf '\033[1;32m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install]\033[0m %s\n' "$*"; }

command -v cargo >/dev/null || { echo "Rust is needed: https://rustup.rs (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh)" >&2; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/[install] GPU: /' \
  || warn "nvidia-smi not found: the engines need an NVIDIA GPU and its driver"
for t in pdftotext ffmpeg bwrap; do command -v $t >/dev/null || warn "optional: $t is missing (sudo apt install $( [ $t = pdftotext ] && echo poppler-utils || ([ $t = bwrap ] && echo bubblewrap || echo $t)))"; done
[ -x "$STRATA_HOME/engine-poly/strata-poly" ] || [ -x "$STRATA_HOME/engine/strata" ] \
  || warn "no PolyStrata engines in $STRATA_HOME (engine/strata, engine-poly/strata-poly): see deploy/build-engines.sh"

say "building am2Strata (release)"
(cd "$ROOT" && cargo build --release --bins)

mkdir -p "$STRATA_HOME" "$STRATA_DATA/models" "$STRATA_DATA/packs"
for src in "$ROOT"/configs/strata-*.json; do
  dst="$STRATA_HOME/$(basename "$src")"
  if [ -e "$dst" ] && [ $FORCE = 0 ]; then warn "kept $dst (exists; --force overwrites)"; continue; fi
  # the shared form names ${STRATA_HOME} and ${STRATA_DATA}; the installed copy has the real folders, so a service
  # needs no environment
  sed -e "s#\${STRATA_HOME}#$STRATA_HOME#g" -e "s#\${STRATA_DATA}#$STRATA_DATA#g" \
      -e "s#\"host\": \"127.0.0.1\"#\"host\": \"$HOST\"#" -e "s#\"port\": 8000#\"port\": $PORT#" "$src" > "$dst.tmp"
  if [ -n "$KEY" ]; then sed -i "s#^ \"port\": $PORT,#&\n \"api_key\": \"$KEY\",#" "$dst.tmp"; fi
  mv "$dst.tmp" "$dst"; chmod 600 "$dst"
  say "run config: $dst"
done
[ -n "$KEY" ] && say "API key: $KEY   (clients send: Authorization: Bearer $KEY; the web app asks for it under Settings › Server)"
[ "$HOST" = 0.0.0.0 ] && [ -z "$KEY" ] && warn "listening on the network without an API key: anyone on it can use the models"

RUN="$ROOT/target/release/am2strata-run"
if [ $SERVICE = 1 ]; then
  loginctl show-user "$USER" -p Linger 2>/dev/null | grep -q yes || warn "for a start at boot before you log in: loginctl enable-linger $USER"
  say "installing the boot service"
  "$RUN" install --models-dir "$STRATA_HOME" --port "$PORT" --host "$HOST" ${MODEL:+--model "$MODEL"} -- --data-dir "$STRATA_DATA"
else
  say "done. Start it with:"
  echo "  $RUN serve --models-dir $STRATA_HOME --port $PORT --host $HOST ${MODEL:+--model $MODEL} -- --data-dir $STRATA_DATA"
fi
say "models: deploy/pull-models.sh (Tiel Coder, MiniCPM-V, Orca); guide: docs/DEPLOYMENT.md"
