#!/usr/bin/env bash
# The regression suite (docs/REGRESSION.md): the regression tests against test servers of their own, then, when a
# server's address is given, the live checklist against that RUNNING server (for after an update).
#
#   deploy/regression.sh                                   the tests only (no GPU, no model; nothing running is touched)
#   deploy/regression.sh --url http://127.0.0.1:8000       the tests, then the quick live checklist against that server
#   deploy/regression.sh --url ... --full                  ... with a real call to every API and a tool round trip
#   deploy/regression.sh --url ... --live-only             skip the tests: only the live checklist
#   deploy/regression.sh --update-golden                   write the response shapes as they are now into a TEMP folder
#                                                          (never into the repository) and say how to accept them
#
# The server's API key is read from AM2STRATA_API_KEY (or, for this PC's own server, its run configs): it is not taken
# on the command line here, so it does not end up in the shell's history or the process list. It is never printed.
# --quick (the default) changes nothing on the server and makes at most 3 short model calls.
# AM2STRATA_TEST_THREADS: how many tests run at once (default 4: each starts a server of its own).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="" MODE="--quick" LIVE_ONLY=0 UPDATE=0 JSON=""
while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="${2:?--url needs an address}"; shift 2 ;;
    --quick|--full) MODE="$1"; shift ;;
    --json) JSON="--json"; shift ;;
    --live-only) LIVE_ONLY=1; shift ;;
    --update-golden) UPDATE=1; shift ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "regression.sh: unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
done
cd "$ROOT"
THREADS="${AM2STRATA_TEST_THREADS:-4}"

if [ "$UPDATE" = 1 ]; then
  OUT="$(mktemp -d "${TMPDIR:-/tmp}/am2strata-golden.XXXXXX")"
  UPDATE_GOLDEN=1 GOLDEN_OUT="$OUT" cargo test -p am2strata --test regression_shapes -- --test-threads="$THREADS" --nocapture 2>&1 | grep '^golden ' || true
  echo
  echo "The shapes as they are now: $OUT"
  echo "Compare:  diff -ru crates/am2strata/tests/golden $OUT"
  echo "Accept:   cp $OUT/<name>.json crates/am2strata/tests/golden/   (only the ones you mean; then run this script again)"
  exit 0
fi

if [ "$LIVE_ONLY" = 0 ]; then
  echo "== regression tests (test servers of their own) =="
  cargo test -p am2strata --test regression_api --test regression_shapes --test regression_bugs --test regression_runner -- --test-threads="$THREADS"
  # the regressions that live beside the code they guard
  cargo test -p am2strata --bin am2strata regress
  cargo test -p strata-workshop --lib regress
fi

if [ -n "$URL" ]; then
  echo "== live checklist against $URL =="
  # shellcheck disable=SC2086
  cargo run -q -p am2strata --bin am2strata-run -- regression --url "$URL" $MODE $JSON
elif [ "$LIVE_ONLY" = 1 ]; then
  echo "regression.sh: --live-only needs --url <the running server>" >&2
  exit 2
fi
