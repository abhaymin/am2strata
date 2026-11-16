#!/usr/bin/env bash
# Benchmark am2strata-hybrid over its stdin protocol (docs/PROTOCOL.md): a greedy prompt read, then a greedy decode.
# The engine measures both itself (DONE #3 prompt ms, #4 decode ms); this parses those and reports tok/s. Same script
# drives the Mac: nothing here is x86 specific. Keep each model's runs short (the numbers are engine-side).
#   tools/hybrid-bench.sh <model.gguf> [--prompt N] [--new N] [--threads T] [--iters I] [--engine PATH] [-- ENGINE ARGS...]
# Everything after `--` goes to the engine as is (e.g. -- --resident-experts --expert-ram-gib 8).
set -uo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
engine=${ENGINE:-$here/target-hybrid/release/am2strata-hybrid}  # where CARGO_TARGET_DIR=target-hybrid builds it
model=; prompt_len=754; new=64; threads=; iters=2; extra=()
while [ $# -gt 0 ]; do
  case $1 in
    --prompt) prompt_len=$2; shift 2;;
    --new) new=$2; shift 2;;
    --threads) threads=$2; shift 2;;
    --iters) iters=$2; shift 2;;
    --engine) engine=$2; shift 2;;
    --) shift; extra=("$@"); break;;
    *) [ -z "$model" ] && model=$1; shift;;
  esac
done
[ -n "$model" ] || { echo "usage: hybrid-bench.sh <model.gguf> [--prompt N] [--new N] [--threads T] [--iters I]" >&2; exit 2; }

# the prompt ids: $prompt_len ids in 1000..1753 (under any supported vocab), as the Mac run fed
ids=$(python3 -c "print(','.join(str(1000+i%$prompt_len) for i in range($prompt_len)))")
thread_args=(); [ -n "$threads" ] && thread_args=("--threads" "$threads")

[ -x "$engine" ] || { echo "no engine at $engine (build: cd crates/strata-hybrid && CARGO_TARGET_DIR=\$PWD/../../target-hybrid cargo build --release --no-default-features; or pass --engine)" >&2; exit 2; }
mkdir -p "$here/target-hybrid"
out=$here/target-hybrid/.hybrid-bench.out
{
  for _ in $(seq 1 "$iters"); do echo "GEN $new $ids"; done
  echo QUIT
} | timeout 900 "$engine" ${thread_args[@]} "${extra[@]}" --serve --model "$model" --max-context 4096 >"$out" 2>"$out.err"

# DONE fields (docs/PROTOCOL.md): 1 generated, 2 prompt tokens, 3 prompt ms, 4 decode ms; after the word DONE they are
# awk's $2..$5. Summed across iters.
read -r prompt_ms decode_ms n_gen n_prompt < <(grep -E '^DONE ' "$out" | awk '{ng+=$2; np+=$3; pm+=$4; dm+=$5} END {print pm+0, dm+0, ng+0, np+0}')
if [ "${n_gen:-0}" -eq 0 ]; then echo "no DONE received"; grep -E '^(INFO|ERR)' "$out" | head; exit 1; fi
pp_tok="$(awk -v p="$n_prompt" -v m="$prompt_ms" 'BEGIN{printf "%.2f", (m > 0 ? p/(m/1000.0) : 0)}') prompt tok/s"
dt_tok="$(awk -v n="$n_gen" -v d="$decode_ms" 'BEGIN{printf "%.2f", (d > 0 ? n/(d/1000.0) : 0)}') decode tok/s"
echo "prompt read: ${n_prompt} tokens in ${prompt_ms} ms  ->  $pp_tok"
echo "generation:  ${n_gen} tokens in ${decode_ms} ms  ->  $dt_tok"
