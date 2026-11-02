#!/usr/bin/env bash
# A download for people who should not have to build anything: the server, its tools, the web app and PolyStrata's engines.
# It leaves out the media engine (its codecs have their own terms, see NOTICE) and every run config (they hold API keys).
#   deploy/make-release.sh [--strata-home DIR]     -> dist/am2strata-<version>-linux-x86_64.tar.gz and its .sha256
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STRATA_HOME="${STRATA_HOME:-$HOME/Development/PolyStrata}"
[ "${1:-}" = "--strata-home" ] && STRATA_HOME="$2"
VER=$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)
NAME="am2strata-$VER-linux-x86_64"
OUT="$ROOT/dist/$NAME"
rm -rf "$OUT" && mkdir -p "$OUT/bin" "$OUT/engine" "$OUT/engine-poly"
systemd-run --user --scope --quiet -p MemoryMax=3G -p MemorySwapMax=0 nice cargo build --release -j 2 --manifest-path "$ROOT/Cargo.toml" -p am2strata
install -m755 "$ROOT/target/release/am2strata" "$ROOT/target/release/am2strata-run" "$OUT/bin/"
install -m755 "$STRATA_HOME/engine/strata" "$STRATA_HOME/engine/strata-vision" "$STRATA_HOME/engine/libstrata_kq.so" "$OUT/engine/"
install -m755 "$STRATA_HOME/engine-poly/strata-poly" "$STRATA_HOME/engine-poly/strata-vision" "$OUT/engine-poly/"
cp -r "$ROOT/web" "$OUT/web"
cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$ROOT/deploy/release/QUICKSTART.md" "$OUT/"
install -m755 "$ROOT/deploy/release/am2strata" "$OUT/am2strata"
strip "$OUT"/engine/strata "$OUT"/engine/strata-vision "$OUT"/engine-poly/* 2>/dev/null || true
tar -C "$ROOT/dist" -czf "$ROOT/dist/$NAME.tar.gz" "$NAME"
( cd "$ROOT/dist" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )
du -h "$ROOT/dist/$NAME.tar.gz"
