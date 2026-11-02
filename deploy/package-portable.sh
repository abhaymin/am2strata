#!/usr/bin/env bash
# The portable download for one platform: the server, its tools, the Rust engine (CPU, or any GPU through wgpu) and the web app.
#   deploy/package-portable.sh <label>      after building am2strata and strata-hybrid in release mode; -> dist/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL=$1
VER=$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)
NAME="am2strata-$VER-$LABEL"
OUT="$ROOT/dist/$NAME"
EXT=""
case "$LABEL" in windows*) EXT=.exe ;; esac
HYBRID="${HYBRID_TARGET:-$ROOT/crates/strata-hybrid/target/release}"
rm -rf "$OUT" && mkdir -p "$OUT/bin"
cp "$ROOT/target/release/am2strata$EXT" "$ROOT/target/release/am2strata-run$EXT" "$HYBRID/am2strata-hybrid$EXT" "$OUT/bin/"
cp -r "$ROOT/web" "$OUT/web"
cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$ROOT/deploy/release/QUICKSTART.md" "$OUT/"
cd "$ROOT/dist"
case "$LABEL" in
    windows*)
        cp "$ROOT/deploy/release/am2strata.cmd" "$OUT/"
        7z a -tzip -bd "$NAME.zip" "$NAME" >/dev/null
        FILE=$NAME.zip
        ;;
    *)
        install -m755 "$ROOT/deploy/release/am2strata" "$OUT/am2strata"
        tar -czf "$NAME.tar.gz" "$NAME"
        FILE=$NAME.tar.gz
        ;;
esac
rm -rf "$NAME"
if command -v sha256sum >/dev/null; then sha256sum "$FILE" > "$FILE.sha256"; else shasum -a 256 "$FILE" > "$FILE.sha256"; fi
ls -la "$FILE"
