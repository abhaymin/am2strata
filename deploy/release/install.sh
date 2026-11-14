#!/bin/sh
# am2Strata installer for Linux (x86_64, arm64) and macOS (Apple silicon, Intel): the latest release, checked against
# its SHA-256. No sudo: everything goes under your home folder.
#
#   curl -fsSL https://github.com/abhaymin/am2strata/releases/latest/download/install.sh | sh
#   curl -fsSL .../install.sh | sh -s -- [options]
#
# See usage() below for the options. The program goes into ~/.local/share/am2strata/app, your models and settings stay
# beside it in ~/.local/share/am2strata, and `am2strata`, `am2strata-run` and `am2strata-workshop` go into ~/.local/bin.
set -eu

REPO=abhaymin/am2strata
UNIT=am2strata-app.service                 # the systemd user service --service makes (Linux)
AGENT=io.github.abhaymin.am2strata         # the launchd agent --service makes (macOS)
MARK="# written by am2Strata's install.sh"  # in every file this script writes outside its own folder: only those are removed
MIN_GLIBC=2.35                             # the Linux downloads are built on Ubuntu 22.04

usage() {
    cat <<EOF
am2Strata installer (Linux x86_64/arm64, macOS Apple silicon/Intel)

  curl -fsSL https://github.com/$REPO/releases/latest/download/install.sh | sh
  curl -fsSL .../install.sh | sh -s -- [options]

  --version X     install release X (v0.1.8) instead of the latest
  --dir DIR       install into DIR instead of ~/.local/share/am2strata (models and settings stay there too)
  --bin-dir DIR   where the am2strata commands go (default ~/.local/bin)
  --from DIR      install from a folder that already holds the download and its .sha256 (no network)
  --service       start the server when you sign in: a systemd user service (Linux) or a launchd agent (macOS)
  --no-service    no service (the default); also removes the one an earlier --service made
  --with-deps     install what the optional features miss with apt, dnf, pacman or Homebrew: the exact command is
                  shown first and runs only after you confirm (--pdf is the older name)
  --yes           with --with-deps: run that command without asking
  --no-browser    do not fetch Chrome for Testing's headless shell for the browser tool (about 100 MB, fetched
                  only when this PC has no Chrome or Chromium)
  --uninstall     remove the program, its commands and its service; your models and settings stay
  -h, --help      this text
EOF
}

say() { printf '%s\n' "$*"; }
fail() { printf 'am2strata install: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
# a file's SHA-256 (sha256sum on Linux, shasum on macOS)
sha256() {
    if have sha256sum; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# linux-x86_64, linux-arm64, macos-arm64 or macos-x86_64 for this PC
detect_label() {
    os=$(uname -s)
    cpu=$(uname -m)
    case "$cpu" in
        arm64|aarch64) cpu=arm64 ;;
        x86_64|amd64) cpu=x86_64 ;;
        *) fail "no download for a $cpu processor: build from source (https://github.com/$REPO#build-from-source-the-nvidia-fast-path)" ;;
    esac
    case "$os" in
        Linux)
            # the downloads need glibc (Alpine's musl cannot run them) of Ubuntu 22.04's age or newer
            libc=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')
            [ -n "$libc" ] || fail "this Linux has no glibc (Alpine and other musl systems): build from source"
            if ! printf '%s\n%s\n' "$MIN_GLIBC" "$libc" | awk -F. 'NR==1 {a=$1; b=$2} NR==2 {exit !($1 > a || ($1 == a && $2 >= b))}'; then
                fail "this Linux has glibc $libc and the download needs $MIN_GLIBC or newer (Ubuntu 22.04, Debian 12, Fedora 36): build from source"
            fi
            LABEL=linux-$cpu ;;
        Darwin)
            # a terminal running under Rosetta says x86_64 on an Apple silicon Mac: the native build is the right one
            if [ "$cpu" = x86_64 ]; then
                if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || true)" = 1 ] || [ "$(sysctl -n hw.optional.arm64 2>/dev/null || true)" = 1 ]; then
                    cpu=arm64
                fi
            fi
            LABEL=macos-$cpu ;;
        *) fail "no download for $os: on Windows use install.ps1 (https://github.com/$REPO#install)" ;;
    esac
}

# a folder name that is safe inside the launchers, the systemd unit and the launchd plist this script writes
check_dir() {
    case "$1" in
        *'"'*|*'$'*|*'`'*|*'\'*|*'&'*|*'<'*|*'>'*|*'%'*) fail "the folder $1 has a character the launchers cannot carry (one of \" \$ \` \\ & < > %): choose another" ;;
    esac
    case "$1" in
        *"
"*) fail "the folder has a line break in its name: choose another" ;;
    esac
}

# is this one of the files we wrote? (never remove anything else of that name)
ours() { [ -f "$1" ] && grep -q "am2Strata's install.sh" "$1" 2>/dev/null; }

service_remove() {
    unit_file="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/$UNIT"
    if ours "$unit_file"; then
        if have systemctl; then systemctl --user disable --now "$UNIT" >/dev/null 2>&1 || true; fi
        rm -f "$unit_file"
        if have systemctl; then systemctl --user daemon-reload >/dev/null 2>&1 || true; fi
        say "Removed the service $UNIT."
    fi
    plist="$HOME/Library/LaunchAgents/$AGENT.plist"
    if ours "$plist"; then
        if have launchctl; then launchctl bootout "gui/$(id -u)/$AGENT" >/dev/null 2>&1 || true; fi
        rm -f "$plist"
        say "Removed the launchd agent $AGENT."
    fi
}

service_install() {
    case "$LABEL" in
        linux-*)
            unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
            mkdir -p "$unit_dir"
            # no model yet: the unit is skipped, not failed, until `am2strata pull` has brought one
            cat > "$unit_dir/$UNIT" <<EOF
$MARK
[Unit]
Description=am2Strata
After=network-online.target
ConditionPathExistsGlob=$DIR/strata-*.json

[Service]
Environment="AM2STRATA_DATA=$DIR"
ExecStart="$BIN/am2strata" serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
            if have systemctl && systemctl --user daemon-reload >/dev/null 2>&1; then
                systemctl --user enable "$UNIT" >/dev/null 2>&1 || true
                systemctl --user restart "$UNIT" >/dev/null 2>&1 || true
                say "Service $UNIT: starts when you sign in (now too, once a model is there). Its log: journalctl --user -u $UNIT"
                say "  To start it at boot, before you sign in: loginctl enable-linger $(id -un)"
            else
                say "Wrote $unit_dir/$UNIT, but this session has no systemd user manager to enable it. Later:"
                say "  systemctl --user daemon-reload && systemctl --user enable --now $UNIT"
            fi ;;
        macos-*)
            mkdir -p "$HOME/Library/LaunchAgents"
            plist="$HOME/Library/LaunchAgents/$AGENT.plist"
            cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!-- $MARK -->
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$AGENT</string>
    <key>ProgramArguments</key><array><string>$BIN/am2strata</string><string>serve</string></array>
    <key>EnvironmentVariables</key><dict><key>AM2STRATA_DATA</key><string>$DIR</string></dict>
    <key>RunAtLoad</key><true/>
    <key>StandardOutPath</key><string>$DIR/service.log</string>
    <key>StandardErrorPath</key><string>$DIR/service.log</string>
</dict>
</plist>
EOF
            if ls "$DIR"/strata-*.json >/dev/null 2>&1 && have launchctl; then
                launchctl bootout "gui/$(id -u)/$AGENT" >/dev/null 2>&1 || true
                launchctl bootstrap "gui/$(id -u)" "$plist" >/dev/null 2>&1 || true
                say "launchd agent $AGENT: started, and starts when you sign in. Its log: $DIR/service.log"
            else
                say "launchd agent $AGENT: starts when you sign in, once a model is there. Its log: $DIR/service.log"
            fi ;;
    esac
}

uninstall() {
    # where an install put its commands, when it was not the default
    if [ -f "$APPS/install-info" ]; then
        b=$(sed -n 's/^bin=//p' "$APPS/install-info" | head -1)
        [ -z "$b" ] || [ "$BIN_GIVEN" = 1 ] || BIN=$b
    fi
    service_remove
    for n in am2strata am2strata-run am2strata-workshop; do
        if ours "$BIN/$n"; then rm -f "$BIN/$n"; fi
    done
    # the launcher of installers before this one: a two-line wrapper naming this folder
    if [ -f "$BIN/am2strata" ] && grep -q "exec \"$APPS/current/am2strata\"" "$BIN/am2strata" 2>/dev/null; then rm -f "$BIN/am2strata"; fi
    rm -rf "$APPS"
    say "Removed am2Strata. Your models and settings are still in $DIR (delete that folder to remove them too)."
}

have_chrome() {
    for n in chromium chromium-browser google-chrome google-chrome-stable; do
        if have "$n"; then return 0; fi
    done
    [ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ] || [ -x "/Applications/Chromium.app/Contents/MacOS/Chromium" ] \
        || [ -x "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge" ]
}

# the browser tool: a Chrome already here is used as it is; otherwise Chrome for Testing's headless shell
fetch_browser() {
    case "$LABEL" in
        linux-x86_64) plat=linux64 ;;
        macos-arm64) plat=mac-arm64 ;;
        macos-x86_64) plat=mac-x64 ;;
        *) return 0 ;;      # Chrome for Testing has no Linux arm64 build: the report below asks for Chromium
    esac
    if have_chrome || [ -d "$DIR/data/browser/chrome-headless-shell-$plat" ]; then return 0; fi
    if [ -n "$FROM" ]; then say "Skipping the browser tool's Chrome (--from: no network)."; return 0; fi
    if ! have unzip; then say "Skipping the browser tool's Chrome: unzip is missing."; return 0; fi
    json=https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json
    url=$(curl -fsSL "$json" 2>/dev/null | tr ',{}' '\n\n\n' | grep -o "https://[^\"]*/$plat/chrome-headless-shell-$plat.zip" | head -1 || true)
    [ -n "$url" ] || { say "Chrome for Testing could not be reached; the rest works without the browser tool."; return 0; }
    say "Fetching Chrome for Testing's headless shell for the browser tool (about 100 MB; --no-browser skips it)..."
    mkdir -p "$DIR/data/browser"
    if curl -fL --progress-bar -o "$TMP/c.zip" "$url" && (cd "$DIR/data/browser" && unzip -q "$TMP/c.zip"); then :; else
        say "The browser download failed; the rest works without it."
    fi
}

# The report for a release from before `am2strata-run doctor`. Sets PKGS for this system's package manager.
shell_doctor() {
    PKGS=""
    need() {    # need <program> <apt> <dnf> <pacman> <brew>: adds its package when the program is missing
        if have "$1"; then return 0; fi
        case "$PM" in apt) p=$2 ;; dnf) p=$3 ;; pacman) p=$4 ;; brew) p=$5 ;; *) p="" ;; esac
        case " $PKGS " in *" $p "*) ;; *) [ -z "$p" ] || PKGS="$PKGS $p" ;; esac
        MISSING="$MISSING $1"
    }
    row() {     # row <feature>: one line from MISSING
        if [ -z "$MISSING" ]; then
            printf '  %-9s %-31s\n' ready "$1"
        else
            printf '  %-9s %-31s needs%s\n' missing "$1" "$(printf '%s' "$MISSING" | sed 's/ /, /g; s/^,//')"
        fi
        MISSING=""
    }
    MISSING=""
    say "am2Strata on $LABEL"
    say ""
    printf '  %-9s %-31s %s\n' ready "HTTPS" "built in"
    need pdftotext poppler-utils poppler-utils poppler poppler
    need pdftoppm poppler-utils poppler-utils poppler poppler
    row "PDFs in chats"
    need ffmpeg ffmpeg ffmpeg-free ffmpeg ffmpeg
    row "Video from API clients"
    case "$LABEL" in
        linux-*)
            need Xvfb xvfb xorg-x11-server-Xvfb xorg-server-xvfb ""
            need xdotool xdotool xdotool xdotool ""
            need xclip xclip xclip xclip ""
            row "Computer tab (sandbox desktop)" ;;
        *) printf '  %-9s %-31s %s\n' "not here" "Computer tab (sandbox desktop)" "Linux only" ;;
    esac
    PKGS=${PKGS# }
    say ""
    cmd=$(deps_command) && [ -n "$PKGS" ] && { say "To add what is missing:"; say "  $cmd"; }
    return 0
}

# apt, dnf, pacman or brew ("" when this system has none of them)
detect_pm() {
    PM=""
    case "$LABEL" in
        macos-*) if have brew; then PM=brew; fi ;;
        *)
            if have apt-get; then PM=apt
            elif have dnf; then PM=dnf
            elif have pacman; then PM=pacman
            fi ;;
    esac
}

# the command that installs $PKGS (fails when this system has no package manager we know)
deps_command() {
    case "$PM" in
        apt) c="apt-get install -y $PKGS" ;;
        dnf) c="dnf install -y $PKGS" ;;
        pacman) c="pacman -S --needed --noconfirm $PKGS" ;;
        brew) say "brew install $PKGS"; return 0 ;;
        *) return 1 ;;
    esac
    if [ "$(id -u)" = 0 ]; then say "$c"; else say "sudo $c"; fi
}

confirm() {
    if [ "$YES" = 1 ]; then return 0; fi
    # piped from curl, stdin is the script itself: the question goes to the terminal
    if [ -t 0 ]; then
        printf '%s [y/N] ' "$1"
        read -r answer || answer=""
    elif (exec 3</dev/tty) 2>/dev/null; then
        printf '%s [y/N] ' "$1" > /dev/tty
        read -r answer < /dev/tty || answer=""
    else
        say "There is no terminal to ask on: run the command yourself, or add --yes."
        return 1
    fi
    case "$answer" in y|Y|yes|YES|Yes) return 0 ;; *) return 1 ;; esac
}

# what the optional features need here; with --with-deps, the command that installs it, run after a yes
doctor() {
    run="$APPS/current/bin/am2strata-run"
    detect_pm
    say ""
    # the product's own report, so the installer and `am2strata doctor` say the same; older releases have none
    if "$run" doctor --data-dir "$DIR/data" 2>/dev/null; then
        PKGS=$("$run" doctor --data-dir "$DIR/data" --packages "$PM" 2>/dev/null || true)
    else
        shell_doctor
    fi
    if [ "$WITH_DEPS" != 1 ]; then
        [ -z "$PKGS" ] || say "This installer does that for you when you run it again with --with-deps (it shows the command and asks first)."
        return 0
    fi
    if [ -z "$PKGS" ]; then
        if [ -z "$PM" ]; then
            case "$LABEL" in
                macos-*) say "--with-deps needs Homebrew (https://brew.sh): install it, then run this again." ;;
                *) say "--with-deps knows apt, dnf and pacman, and this system has none of them: install what is missing with its own package manager." ;;
            esac
        else
            say "--with-deps: nothing to install."
        fi
        return 0
    fi
    cmd=$(deps_command)
    if [ "$PM" != brew ] && [ "$(id -u)" != 0 ] && ! have sudo; then
        say "This would install what is missing, but there is no sudo here. As root:"
        say "  ${cmd#sudo }"
        return 0
    fi
    say "--with-deps: this installs what is missing:"
    say "  $cmd"
    if confirm "Run it now?"; then
        if sh -c "$cmd"; then
            say ""
            "$run" doctor --data-dir "$DIR/data" 2>/dev/null || true
        else
            say "That command failed; am2Strata itself is installed and works without the optional features."
        fi
    else
        say "Not run. am2Strata works without it; the features above stay off until you install them."
    fi
}

main() {
    DIR=${AM2STRATA_DATA:-$HOME/.local/share/am2strata}
    BIN=${AM2STRATA_BIN:-$HOME/.local/bin}
    VERSION="" FROM="" SERVICE="" WITH_DEPS=0 YES=0 BROWSER=1 UNINSTALL=0 BIN_GIVEN=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --version|--dir|--bin-dir|--from)
                [ $# -ge 2 ] || fail "$1 needs a value"
                case "$1" in
                    --version) VERSION=$2 ;;
                    --dir) DIR=$2 ;;
                    --bin-dir) BIN=$2; BIN_GIVEN=1 ;;
                    --from) FROM=$2 ;;
                esac
                shift ;;
            --service) SERVICE=1 ;;
            --no-service) SERVICE=0 ;;
            --with-deps|--pdf) WITH_DEPS=1 ;;
            --yes|-y) YES=1 ;;
            --no-browser) BROWSER=0 ;;
            --uninstall) UNINSTALL=1 ;;
            -h|--help) usage; exit 0 ;;
            *) usage >&2; fail "unknown option $1" ;;
        esac
        shift
    done
    [ -n "${HOME:-}" ] || fail "HOME is not set"
    # absolute folders: the launchers and the service name them
    case "$DIR" in /*) ;; *) DIR=$(pwd)/$DIR ;; esac
    case "$BIN" in /*) ;; *) BIN=$(pwd)/$BIN ;; esac
    DIR=${DIR%/}
    BIN=${BIN%/}
    check_dir "$DIR"
    check_dir "$BIN"
    APPS=$DIR/app

    if [ "$UNINSTALL" = 1 ]; then
        uninstall
        exit 0
    fi

    detect_label
    for t in tar awk sed grep; do have "$t" || fail "needs $t"; done
    have sha256sum || have shasum || fail "needs sha256sum or shasum to check the download"
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT

    if [ -n "$FROM" ]; then
        [ -d "$FROM" ] || fail "--from $FROM is not a folder"
        if [ -n "$VERSION" ]; then
            ARCHIVE=$FROM/am2strata-${VERSION#v}-$LABEL.tar.gz
        else
            ARCHIVE=$(find "$FROM" -maxdepth 1 -name "am2strata-*-$LABEL.tar.gz" | sort | tail -1)
        fi
        if [ -z "$ARCHIVE" ] || [ ! -f "$ARCHIVE" ]; then fail "--from $FROM has no am2strata-<version>-$LABEL.tar.gz"; fi
        [ -f "$ARCHIVE.sha256" ] || fail "no checksum file beside $ARCHIVE: nothing was installed"
        NAME=$(basename "$ARCHIVE" .tar.gz)
        VER=${NAME#am2strata-}
        VER=${VER%-"$LABEL"}
        VERSION=v$VER
        SUMFILE=$ARCHIVE.sha256
    else
        have curl || fail "needs curl"
        if [ -z "$VERSION" ]; then
            # the address /releases/latest redirects to names the tag; GitHub's API (rate-limited) is the second try
            VERSION=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest" 2>/dev/null | sed -n 's#.*/releases/tag/##p')
            if [ -z "$VERSION" ]; then
                VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)
            fi
            [ -n "$VERSION" ] || fail "could not find the latest release (GitHub is unreachable or refused): try --version v0.1.8"
        fi
        case "$VERSION" in v*) ;; *) VERSION=v$VERSION ;; esac
        VER=${VERSION#v}
        NAME=am2strata-$VER-$LABEL
        url=https://github.com/$REPO/releases/download/$VERSION
        ARCHIVE=$TMP/$NAME.tar.gz
        SUMFILE=$ARCHIVE.sha256
        say "Downloading am2Strata $VERSION for $LABEL..."
        curl -fL --progress-bar -o "$ARCHIVE" "$url/$NAME.tar.gz" \
            || fail "no download $NAME.tar.gz in release $VERSION ($url): that release may be from before $LABEL was built; see https://github.com/$REPO/releases"
        curl -fsSL -o "$SUMFILE" "$url/$NAME.tar.gz.sha256" || fail "release $VERSION has no checksum file for $NAME: nothing was installed"
    fi
    want=$(awk '{print $1; exit}' "$SUMFILE")
    got=$(sha256 "$ARCHIVE")
    if [ -z "$want" ] || [ "$want" != "$got" ]; then fail "checksum mismatch ($got, expected ${want:-nothing}): nothing was installed"; fi
    say "Checked its SHA-256."

    # unpacked beside its final place, then moved: an install that stops half way leaves the old one working
    mkdir -p "$APPS" "$BIN"
    rm -rf "$APPS/.unpack"
    mkdir "$APPS/.unpack"
    tar -xzf "$ARCHIVE" -C "$APPS/.unpack"
    [ -x "$APPS/.unpack/$NAME/am2strata" ] || fail "the download has no $NAME/am2strata: nothing was installed"
    rm -rf "${APPS:?}/$NAME"
    mv "$APPS/.unpack/$NAME" "$APPS/$NAME"
    rm -rf "$APPS/.unpack"
    ln -sfn "$APPS/$NAME" "$APPS/current"
    printf 'bin=%s\nversion=%s\nlabel=%s\n' "$BIN" "$VERSION" "$LABEL" > "$APPS/install-info"
    # wrappers, not links: the launcher finds its files beside itself, and they carry the data folder for --dir
    wrapper() {
        cat > "$BIN/$1" <<EOF
#!/bin/sh
$MARK
AM2STRATA_DATA="\${AM2STRATA_DATA:-$DIR}"
export AM2STRATA_DATA
exec "$APPS/current/$2" "\$@"
EOF
        chmod 755 "$BIN/$1"
    }
    wrapper am2strata am2strata
    wrapper am2strata-run bin/am2strata-run
    wrapper am2strata-workshop bin/am2strata-workshop
    # older versions go once the new one is in place
    for d in "$APPS"/am2strata-*; do
        [ "$d" = "$APPS/$NAME" ] || rm -rf "$d"
    done
    say "Installed am2Strata $VERSION in $APPS/$NAME"
    if ls "$DIR"/strata-*.json >/dev/null 2>&1; then say "Your models and settings in $DIR were kept as they are."; fi

    if [ "$BROWSER" = 1 ]; then fetch_browser; fi
    case "$SERVICE" in
        1) service_install ;;
        0) service_remove ;;
    esac
    doctor

    say ""
    case ":$PATH:" in
        *":$BIN:"*) ;;
        *) say "Add $BIN to your PATH (for example in ~/.profile), then open a new terminal:"
           say "  export PATH=\"$BIN:\$PATH\""
           say "" ;;
    esac
    say "Next:"
    say "  am2strata pull mistralai/Ministral-3-3B-Instruct-2512-GGUF Q4_K_M    # a small model to start with"
    say "  am2strata                                                           # then open http://127.0.0.1:8000"
    say "  am2strata doctor                                                    # this report again, any time"
    say "Uninstall: run this installer with --uninstall (your models and settings stay)."
}

# everything is in functions and runs from this last line: a download cut short runs nothing
main "$@"
