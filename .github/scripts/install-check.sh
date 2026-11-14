#!/bin/sh
# What a new user does, against a download packaged from this checkout (deploy/package-portable.sh), on Linux or macOS:
# install with deploy/release/install.sh, run the commands, then uninstall and see that it is gone and the data stayed.
#   .github/scripts/install-check.sh <dist folder> <label>        (CI: install-check.yml)
# It installs under $HOME: on a PC of your own, give it a HOME of its own (HOME=$(mktemp -d)).
set -eu
DIST=$1
LABEL=$2
ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
INSTALL="$ROOT/deploy/release/install.sh"
VER=$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)
BIN=$HOME/.local/bin
DIR=$HOME/.local/share/am2strata
step() { printf '\n== %s\n' "$*"; }
die() { printf 'install-check: %s\n' "$*" >&2; exit 1; }

step "a download whose checksum does not match is refused"
BAD=$(mktemp -d)
cp "$DIST/am2strata-$VER-$LABEL.tar.gz" "$BAD/"
echo "0000000000000000000000000000000000000000000000000000000000000000  am2strata-$VER-$LABEL.tar.gz" > "$BAD/am2strata-$VER-$LABEL.tar.gz.sha256"
if sh "$INSTALL" --from "$BAD" --no-browser; then die "a bad checksum was accepted"; fi
[ ! -e "$DIR/app" ] || die "something was installed after a bad checksum"
rm -rf "$BAD"

step "install (the default folders), picking this PC's download by itself"
sh "$INSTALL" --from "$DIST" --no-browser | tee "$HOME/install.log"
grep -q "Installed am2Strata v$VER in $DIR/app/am2strata-$VER-$LABEL" "$HOME/install.log" || die "it did not install $LABEL"
grep -q "ready     HTTPS" "$HOME/install.log" || die "the installer printed no doctor report"

step "the commands"
out=$("$BIN/am2strata" --version)
echo "$out"
[ "$out" = "am2strata $VER" ] || die "am2strata --version said: $out"
"$BIN/am2strata-run" doctor | tee "$HOME/doctor.log"
grep -q "ready     HTTPS" "$HOME/doctor.log" || die "am2strata-run doctor did not report HTTPS"
grep -q "ready     Workshop" "$HOME/doctor.log" || die "am2strata-run doctor did not find the Workshop"
"$BIN/am2strata" doctor | grep -q "ready     HTTPS" || die "am2strata doctor did not report"
"$BIN/am2strata-run" doctor --packages apt >/dev/null || die "doctor --packages failed"
"$BIN/am2strata-workshop" --help >/dev/null || die "am2strata-workshop does not start"
[ -x "$DIR/app/current/bin/am2strata-hybrid" ] || die "no engine in the download"
out=$("$BIN/am2strata" 2>&1 || true)
echo "$out"
echo "$out" | grep -q "No model yet" || die "the launcher did not ask for a model"

step "an update keeps models and settings; --service and --no-service"
mkdir -p "$DIR/data"
echo kept > "$DIR/data/settings-kept.txt"
sh "$INSTALL" --from "$DIST" --no-browser --service
case "$LABEL" in
    linux-*) SVC="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/am2strata-app.service" ;;
    *) SVC="$HOME/Library/LaunchAgents/io.github.abhaymin.am2strata.plist" ;;
esac
[ -f "$SVC" ] || die "--service wrote no $SVC"
grep -q "$BIN/am2strata" "$SVC" || die "the service does not start $BIN/am2strata"
sh "$INSTALL" --from "$DIST" --no-browser --no-service
[ ! -e "$SVC" ] || die "--no-service left $SVC"
[ "$(cat "$DIR/data/settings-kept.txt")" = kept ] || die "an update lost the data"

step "uninstall: the program is gone, the data stays"
sh "$INSTALL" --uninstall
for f in "$DIR/app" "$BIN/am2strata" "$BIN/am2strata-run" "$BIN/am2strata-workshop" "$SVC"; do
    [ ! -e "$f" ] || die "uninstall left $f"
done
[ "$(cat "$DIR/data/settings-kept.txt")" = kept ] || die "uninstall removed the data"

step "--dir and --bin-dir: another place, with its own data"
OTHER=$(mktemp -d)
sh "$INSTALL" --from "$DIST" --no-browser --dir "$OTHER/a place" --bin-dir "$OTHER/bin" >/dev/null
[ "$("$OTHER/bin/am2strata" --version)" = "am2strata $VER" ] || die "--dir: am2strata does not run"
"$OTHER/bin/am2strata" 2>&1 | grep -q "No model yet" || die "--dir: the launcher did not ask for a model"
[ ! -e "$DIR/app" ] || die "--dir also installed into the default folder"
sh "$INSTALL" --uninstall --dir "$OTHER/a place"
if [ -e "$OTHER/a place/app" ] || [ -e "$OTHER/bin/am2strata" ]; then die "--uninstall --dir left files"; fi
rm -rf "$OTHER"

step "ok: $LABEL"
