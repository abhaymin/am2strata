#!/usr/bin/env bash
# Chrome for Testing's chrome-headless-shell (the chat's browser tool), the current Stable, into <dir>.
#   deploy/fetch-chrome.sh <dir>       <dir>: <am2strata data>/data/browser, where am2strata looks; prints the program's path
# From Google's own Chrome for Testing downloads (https://googlechromelabs.github.io/chrome-for-testing/).
set -euo pipefail
DIR=$1
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) PLAT=linux64 ;;
    Darwin-arm64) PLAT=mac-arm64 ;;
    Darwin-x86_64) PLAT=mac-x64 ;;
    MINGW*|MSYS*|CYGWIN*) PLAT=win64 ;;
    *) echo "no chrome-headless-shell for $(uname -s) $(uname -m): install Chrome or Chromium instead" >&2; exit 1 ;;
esac
JSON=https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json
URL=$(curl -fsSL --retry 3 "$JSON" | tr ',{}' '\n\n\n' | grep -o "https://[^\"]*/$PLAT/chrome-headless-shell-$PLAT.zip" | head -1)
[ -n "$URL" ] || { echo "Chrome for Testing lists no chrome-headless-shell for $PLAT" >&2; exit 1; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
curl -fsSL --retry 3 -o "$TMP/c.zip" "$URL"
mkdir -p "$DIR"
rm -rf "$DIR/chrome-headless-shell-$PLAT"
(cd "$DIR" && unzip -q "$TMP/c.zip")
EXE="$DIR/chrome-headless-shell-$PLAT/chrome-headless-shell"
[ "$PLAT" = win64 ] && EXE="$EXE.exe"
[ -f "$EXE" ] || { echo "the download has no $EXE" >&2; exit 1; }
echo "$EXE"
