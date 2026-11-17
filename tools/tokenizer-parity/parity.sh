#!/usr/bin/env bash
# Tokenizer parity with llama.cpp: every case in corpus.txt (escapes \n \t \r \uXXXX decoded) through llama.cpp's
# llama-tokenize (vocab only, no BOS) and through strata-tokenizer (examples/tok_ids), compared id for id.
#   tools/tokenizer-parity/parity.sh <model.gguf> [special]
# LLAMA_TOKENIZE: the llama-tokenize binary. "special" parses control tokens (<|im_start|>) on both sides.
set -euo pipefail
model=$1; special=${2:-}
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd)
lt=${LLAMA_TOKENIZE:-llama-tokenize}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
python3 -I - "$here/corpus.txt" "$tmp" <<'PY'
import json, sys, os, re
out = sys.argv[2]
with open(sys.argv[1], encoding="utf-8") as f, open(os.path.join(out, "cases.jsonl"), "w", encoding="utf-8") as j:
    for i, line in enumerate(f.read().split("\n")):
        if not line: continue
        s = re.sub(r"\\(u[0-9a-fA-F]{4}|n|t|r|\\)", lambda m: {"n": "\n", "t": "\t", "r": "\r", "\\": "\\"}.get(m[1]) or chr(int(m[1][1:], 16)), line)
        j.write(json.dumps(s, ensure_ascii=False) + "\n")
        open(os.path.join(out, f"c{i:03}.txt"), "w", encoding="utf-8", newline="").write(s)
PY
flags=(--ids --log-disable --no-bos --no-escape); [ "$special" = special ] || flags+=(--no-parse-special)
"$root/target/release/examples/tok_ids" "$model" "$tmp/cases.jsonl" ${special:+--special} >"$tmp/ours" 2>"$tmp/ours.err"
for f in "$tmp"/c*.txt; do
    CUDA_VISIBLE_DEVICES= "$lt" -m "$model" -f "$f" "${flags[@]}" 2>/dev/null | tail -1 | tr -d '[] ' 
done >"$tmp/llama"
python3 -I - "$tmp/llama" "$tmp/ours" "$tmp/cases.jsonl" <<'PY'
import json, sys
a = open(sys.argv[1]).read().splitlines(); b = open(sys.argv[2]).read().splitlines()
cases = [json.loads(l) for l in open(sys.argv[3], encoding="utf-8")]
bad = 0
for i, (x, y, c) in enumerate(zip(a, b, cases)):
    if x != y:
        bad += 1
        print(f"DIFF case {i}: {c[:60]!r}\n  llama.cpp {x[:200]}\n  ours      {y[:200]}")
print(f"{len(cases) - bad}/{len(cases)} cases identical")
PY
cat "$tmp/ours.err"
