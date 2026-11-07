#!/bin/sh
# am2Strata installer for Linux (x86_64) and macOS (Apple Silicon): the latest release, checked against its SHA-256.
#
#   curl -fsSL https://github.com/abhaymin/am2strata/releases/latest/download/install.sh | sh
#   curl -fsSL .../install.sh | sh -s -- [--version v0.1.0] [--pdf] [--no-browser] [--uninstall]
#
# Installs into ~/.local/share/am2strata (models and settings stay there too) and puts `am2strata` in ~/.local/bin.
#   --pdf         install poppler for PDFs in chats (brew on macOS; apt, dnf or pacman with sudo on Linux)
#   --no-browser  skip Chrome for Testing's headless shell (the chat's browser tool), fetched only when no Chrome or
#                 Chromium is installed
#   --uninstall   remove the program (your models and settings in ~/.local/share/am2strata stay unless you delete them)
set -eu
REPO=abhaymin/am2strata
DATA=${AM2STRATA_DATA:-$HOME/.local/share/am2strata}
APPS=$DATA/app
BIN=${AM2STRATA_BIN:-$HOME/.local/bin}
VERSION="" PDF=0 BROWSER=1 UNINSTALL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --version) VERSION=$2; shift ;;
        --pdf) PDF=1 ;;
        --no-browser) BROWSER=0 ;;
        --uninstall) UNINSTALL=1 ;;
        -h|--help) sed -n '2,14p' "$0" 2>/dev/null || echo "see https://github.com/$REPO"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
say() { printf '%s\n' "$*"; }
fail() { printf 'am2strata install: %s\n' "$*" >&2; exit 1; }

if [ "$UNINSTALL" = 1 ]; then
    rm -rf "$APPS" "$BIN/am2strata"
    say "Removed am2Strata. Your models and settings are still in $DATA."
    exit 0
fi

case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) LABEL=linux-x86_64 ;;
    Darwin-arm64) LABEL=macos-arm64 ;;
    *) fail "no download for $(uname -s) $(uname -m) yet: build from source (https://github.com/$REPO#build-from-source-the-nvidia-fast-path)" ;;
esac
for t in curl tar; do command -v $t >/dev/null || fail "needs $t"; done
if command -v sha256sum >/dev/null; then SUM="sha256sum"; else SUM="shasum -a 256"; fi

if [ -z "$VERSION" ]; then
    VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)
    [ -n "$VERSION" ] || fail "could not find the latest release (GitHub's API refused or is unreachable): try --version v0.1.0"
fi
VER=${VERSION#v}
NAME=am2strata-$VER-$LABEL
URL=https://github.com/$REPO/releases/download/$VERSION
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
say "Downloading am2Strata $VERSION for $LABEL..."
curl -fL --progress-bar -o "$TMP/$NAME.tar.gz" "$URL/$NAME.tar.gz" || fail "download failed: $URL/$NAME.tar.gz"
curl -fsSL -o "$TMP/$NAME.tar.gz.sha256" "$URL/$NAME.tar.gz.sha256" || fail "no checksum file for $NAME"
want=$(cut -d' ' -f1 < "$TMP/$NAME.tar.gz.sha256")
got=$(cd "$TMP" && $SUM "$NAME.tar.gz" | cut -d' ' -f1)
[ "$want" = "$got" ] || fail "checksum mismatch ($got, expected $want): nothing was installed"

mkdir -p "$APPS" "$BIN"
rm -rf "$APPS/$NAME"
tar -xzf "$TMP/$NAME.tar.gz" -C "$APPS"
ln -sfn "$APPS/$NAME" "$APPS/current"
# a wrapper, not a link: the launcher finds its files beside itself
cat > "$BIN/am2strata" <<EOF
#!/bin/sh
exec "$APPS/current/am2strata" "\$@"
EOF
chmod 755 "$BIN/am2strata"
# older versions go once the new one is in place
for d in "$APPS"/am2strata-*; do
    [ "$d" = "$APPS/$NAME" ] || rm -rf "$d"
done
say "Installed $VERSION in $APPS/$NAME"

# PDFs: poppler
if ! command -v pdftotext >/dev/null || ! command -v pdftoppm >/dev/null; then
    if [ "$PDF" = 1 ]; then
        if [ "$LABEL" = macos-arm64 ]; then
            command -v brew >/dev/null && brew install poppler || say "Install Homebrew (https://brew.sh), then: brew install poppler"
        elif command -v apt-get >/dev/null; then sudo apt-get install -y poppler-utils
        elif command -v dnf >/dev/null; then sudo dnf install -y poppler-utils
        elif command -v pacman >/dev/null; then sudo pacman -S --noconfirm poppler
        else say "Install poppler with your package manager to read PDFs in chats."
        fi
    else
        say "PDFs in chats need poppler: run this again with --pdf, or install it yourself ($( [ "$LABEL" = macos-arm64 ] && echo 'brew install poppler' || echo 'sudo apt install poppler-utils'))."
    fi
fi

# the browser tool: a Chrome already here is used as it is
have_chrome() {
    for n in chromium chromium-browser google-chrome google-chrome-stable; do command -v $n >/dev/null && return 0; done
    [ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ] || [ -x "/Applications/Chromium.app/Contents/MacOS/Chromium" ]
}
if [ "$BROWSER" = 1 ] && ! have_chrome; then
    command -v unzip >/dev/null || { say "Skipping the browser tool: unzip is missing."; BROWSER=0; }
fi
if [ "$BROWSER" = 1 ] && ! have_chrome; then
    case $LABEL in linux-x86_64) PLAT=linux64 ;; *) PLAT=mac-arm64 ;; esac
    JSON=https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json
    CURL=$(curl -fsSL "$JSON" | tr ',{}' '\n\n\n' | grep -o "https://[^\"]*/$PLAT/chrome-headless-shell-$PLAT.zip" | head -1 || true)
    if [ -n "$CURL" ]; then
        say "Fetching Chrome for Testing's headless shell for the browser tool (about 100 MB; --no-browser skips it)..."
        mkdir -p "$DATA/data/browser"
        curl -fL --progress-bar -o "$TMP/c.zip" "$CURL" && rm -rf "$DATA/data/browser/chrome-headless-shell-$PLAT" \
            && (cd "$DATA/data/browser" && unzip -q "$TMP/c.zip") || say "The browser download failed; the rest works without it."
    fi
fi

case ":$PATH:" in
    *":$BIN:"*) ;;
    *) say "Add $BIN to your PATH (for example in ~/.profile): export PATH=\"$BIN:\$PATH\"" ;;
esac
say ""
say "Next:"
say "  am2strata pull mistralai/Ministral-3-3B-Instruct-2512-GGUF Q4_K_M    # a small model to start with"
say "  am2strata                                                           # then open http://127.0.0.1:8000"
