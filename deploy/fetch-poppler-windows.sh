#!/usr/bin/env bash
# Poppler for Windows (pdftotext, pdftoppm: PDFs in chats), pinned and checked, into <dest>/poppler.
#   deploy/fetch-poppler-windows.sh <dest>
# The Windows release carries it beside bin/; am2strata looks there before PATH. Poppler is GPL-2.0-or-later: its
# COPYING files and where its source is go in with it.
set -euo pipefail
DEST=$1
VER=26.09.0
TAG=v$VER-0
SHA=7a6f256a0ddf7536182246a5733331bf4677cbcc34f4663774947ad34556c8d0
URL=https://github.com/oschwartz10612/poppler-windows/releases/download/$TAG/Release-$VER-0.zip
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
curl -fsSL --retry 3 -o "$TMP/p.zip" "$URL"
got=$( (sha256sum "$TMP/p.zip" 2>/dev/null || shasum -a 256 "$TMP/p.zip") | cut -d' ' -f1)
[ "$got" = "$SHA" ] || { echo "poppler: checksum $got, expected $SHA" >&2; exit 1; }
(cd "$TMP" && unzip -q p.zip)
SRC="$TMP/poppler-$VER"
mkdir -p "$DEST/poppler/share"
rm -rf "$DEST/poppler/bin"
cp -r "$SRC/Library/bin" "$DEST/poppler/bin"
cp -r "$SRC/share/poppler" "$DEST/poppler/share/poppler"
cat > "$DEST/poppler/README.txt" <<EOF
Poppler $VER for Windows (pdftotext, pdftoppm), as built by https://github.com/oschwartz10612/poppler-windows ($TAG).
Poppler is free software under the GNU General Public License, version 2 or later: share/poppler/COPYING*.
Its source: https://poppler.freedesktop.org/poppler-$VER.tar.xz, and the build recipe at the link above.
am2Strata runs these programs as they are, to read PDF files; they are not part of am2Strata's own MIT-licensed code.
EOF
echo "poppler $VER -> $DEST/poppler"
