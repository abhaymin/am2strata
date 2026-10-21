#!/usr/bin/env bash
# Parity of am2strata-hybrid with llama.cpp on one model: the same prompt (tokenized by llama.cpp), greedy decoding,
# and per step the top-5 logits of both. The engine is fed llama.cpp's tokens (teacher forcing), so every step is
# compared on the same history. Passes when every step's best token is the same, except a near-tie: llama.cpp's own
# top two within TIE (default 0.25) of each other and the engine's choice one of them.
#   tools/reference-llamacpp/parity.sh <model.gguf> <n_new> "<prompt text>"
# ENGINE_ARGS are passed to the engine (e.g. "--pack DIR --resident-experts").
# ORACLE (default $STRATA_DATA/oracle/oracle2) is tools/reference-llamacpp/oracle.cpp built against the patched copy.
set -euo pipefail
model=$1; n=$2; text=$3
here=$(cd "$(dirname "$0")/../.." && pwd)
oracle=${ORACLE:-${STRATA_DATA:-$HOME/Development/Strata-data}/oracle/oracle2}
engine=${ENGINE:-$here/crates/strata-media/target/release/am2strata-hybrid}
tmp=${KEEP:-$(mktemp -d)}; [ -n "${KEEP:-}" ] || trap 'rm -rf "$tmp"' EXIT; mkdir -p "$tmp"
"$oracle" "$model" "$n" "text:$text" >"$tmp/o" 2>"$tmp/oe" || { tail -5 "$tmp/oe"; exit 1; }
ids=$(grep -o 'IDS [0-9,]*' "$tmp/oe" | cut -d' ' -f2)
forced=$(python3 -I -c 'import json,sys; print(",".join(str(json.loads(l)["top"][0][0]) for l in open(sys.argv[1]) if l.startswith("{")))' "$tmp/o")
(echo "GEN $n $ids"; sleep 1) | AM2_DEBUG_TOP=1 AM2_DEBUG_FORCE=$forced "$engine" ${ENGINE_ARGS:-} --serve --model "$model" --max-context 4096 \
    --threads "${THREADS:-12}" 2>"$tmp/h" >"$tmp/hs"
grep -E '^(INFO|ERR)' "$tmp/hs" | cut -c1-200
python3 -I - "$tmp/o" "$tmp/h" <<'PY'
import json, sys
o = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
h = [json.loads(l) for l in open(sys.argv[2]) if l.startswith('{')]
tie = float(__import__('os').environ.get('TIE', '0.25'))
same = ties = 0
worst = 0.0
for a, b in zip(o, h):
    ta, tb = a['top'][0][0], b['top'][0][0]
    da = dict((x[0], x[1]) for x in a['top']); db = dict((x[0], x[1]) for x in b['top'])
    err = max(abs(da[t] - db[t]) for t in da if t in db) if set(da) & set(db) else float('inf')
    worst = max(worst, err)
    margin = a['top'][0][1] - a['top'][1][1]
    near = ta != tb and margin < tie and tb == a['top'][1][0]
    print(f"step {a['step']:3}  llama.cpp {ta:7} {a['top'][0][1]:9.4f}   rust {tb:7} {b['top'][0][1]:9.4f}  "
          f"top5 {len(set(da) & set(db))}/5  max|dlogit| {err:.4f}  {'ok' if ta == tb else ('tie (margin %.3f)' % margin if near else 'DIFFERENT')}")
    same += ta == tb
    ties += near
n = min(len(o), len(h))
print(f"{same}/{n} steps identical, {ties} near-ties, worst logit difference {worst:.4f}")
sys.exit(0 if n > 0 and same + ties == n else 1)
PY
