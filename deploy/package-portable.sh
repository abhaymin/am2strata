#!/usr/bin/env bash
# The portable download for one platform: the server, its tools, the Rust engine (CPU, or any GPU through wgpu) and the web app.
#   deploy/package-portable.sh <label> [<rust-target>]     after building am2strata, strata-workshop and strata-hybrid
#                                                          in release mode; -> dist/am2strata-<version>-<label>.<ext>
# <label> names the download:  linux-x86_64  linux-arm64  macos-arm64  macos-x86_64  windows-x86_64  windows-arm64
# <rust-target> is the triple the build was given with `cargo build --target` (a cross-compile): the binaries are then
# in target/<rust-target>/release. Without it (a build on the platform itself) they are in target/release.
# CARGO_TARGET_DIR is followed for the workspace; HYBRID_TARGET names the engine's release folder when it is elsewhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL=${1:?usage: deploy/package-portable.sh <label> [<rust-target>]}
TRIPLE=${2:-${CARGO_BUILD_TARGET:-}}
# the label and the build must be the same platform: a wrong pair would ship binaries that do not start
case "$LABEL" in
    linux-x86_64)   WANT=x86_64-unknown-linux-gnu ;;
    linux-arm64)    WANT=aarch64-unknown-linux-gnu ;;
    macos-arm64)    WANT=aarch64-apple-darwin ;;
    macos-x86_64)   WANT=x86_64-apple-darwin ;;
    windows-x86_64) WANT=x86_64-pc-windows-msvc ;;
    windows-arm64)  WANT=aarch64-pc-windows-msvc ;;
    *) echo "unknown label $LABEL (linux-x86_64, linux-arm64, macos-arm64, macos-x86_64, windows-x86_64, windows-arm64)" >&2; exit 2 ;;
esac
HOST=$(rustc -vV 2>/dev/null | sed -n 's/^host: //p' || true)
BUILT=${TRIPLE:-$HOST}
if [ -n "$BUILT" ] && [ "$BUILT" != "$WANT" ]; then
    echo "$LABEL is $WANT, but this build is $BUILT: pass the target it was built for, or build on that platform" >&2
    exit 2
fi
VER=$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)
NAME="am2strata-$VER-$LABEL"
OUT="$ROOT/dist/$NAME"
EXT=""
case "$LABEL" in windows*) EXT=.exe ;; esac
REL="${CARGO_TARGET_DIR:-$ROOT/target}/${TRIPLE:+$TRIPLE/}release"
HYBRID="${HYBRID_TARGET:-$ROOT/crates/strata-hybrid/target/${TRIPLE:+$TRIPLE/}release}"
for f in "$REL/am2strata$EXT" "$REL/am2strata-run$EXT" "$REL/am2strata-workshop$EXT" "$HYBRID/am2strata-hybrid$EXT"; do
    [ -f "$f" ] || { echo "not built: $f" >&2; exit 1; }
done
rm -rf "$OUT" && mkdir -p "$OUT/bin"
cp "$REL/am2strata$EXT" "$REL/am2strata-run$EXT" "$REL/am2strata-workshop$EXT" "$HYBRID/am2strata-hybrid$EXT" "$OUT/bin/"
cp -r "$ROOT/web" "$OUT/web"
cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$ROOT/deploy/release/QUICKSTART.md" "$OUT/"
cd "$ROOT/dist"
case "$LABEL" in
    windows*)
        cp "$ROOT/deploy/release/am2strata.cmd" "$OUT/"
        # PDFs in chats: Windows has no poppler of its own. The x64 build; Windows on Arm runs it as it runs any x64 program.
        bash "$ROOT/deploy/fetch-poppler-windows.sh" "$OUT"
        rm -f "$NAME.zip"
        if command -v 7z >/dev/null; then
            7z a -tzip -bd "$NAME.zip" "$NAME" >/dev/null
        else
            powershell -NoProfile -Command "Compress-Archive -Path '$NAME' -DestinationPath '$NAME.zip' -Force"
        fi
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
# the one-line installers ride along with every platform's download (the same files; the release keeps one copy)
cp "$ROOT/deploy/release/install.sh" "$ROOT/deploy/release/install.ps1" "$ROOT/dist/"
ls -la "$FILE"
