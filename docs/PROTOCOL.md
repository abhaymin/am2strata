# PolyStrata engine protocol

The server starts an engine (`strata` for Qwen3.8, `strata-poly` for the other families) as a child process with
`--serve` and the run config's `args`, and talks to it over pipes: one command per line on the engine's stdin, one
event per line on its stdout. The engine's own log goes to stderr (the run config's `log` file). Token ids are the
model's vocabulary ids; the server owns tokenization, the chat template and output parsing.

Source of truth: PolyStrata `serve/server.py` (`StrataEngine`) at commit `26e4cb5`. This document covers the
commands the Rust server uses today and lists the rest.

## Start-up

The engine prints zero or more `INFO key=value ...` lines, then a line starting with `READY`. Commands sent before
`READY` are not read. A resident-RAM load can take minutes; the Python server allows 15.

Environment the server sets: `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` from the config's `gpu`
(numbered as nvidia-smi does), `LD_LIBRARY_PATH` prefixed with the config's existing `lib_dirs`, and every pair in
the config's `env`.

## Commands

| Command | Form | Meaning |
| --- | --- | --- |
| `GEN` | `GEN <max_new>[ key=value ...] <id>,<id>,...` | Generate up to `max_new` tokens after the prompt ids. |
| `GENI` | `GENI <max_new>[ keys] <embeddings-file> <ids>` | As GEN, with image embeddings from the vision encoder. |
| `STOP` | `STOP` | Stop the running request; the engine still ends it with `DONE`. |
| `BGEN`, `BGENI` | batch forms of GEN / GENI | Admit a request into a batch slot (`--batch N`). |
| `BSTOP` | `BSTOP <slot>` | Stop one batch slot. |
| `BYIELD` | `BYIELD` | Ask a long prompt read to give way at its next chunk boundary. |
| `VRAM` | `VRAM <reserve_mib>` | Change how much VRAM the expert cache leaves free. |
| `SAVE`, `RESTORE` | slot state to / from a file | llama-server's slot save API. |
| `QUIT` | `QUIT` | Exit. |

### GEN keys

Written in this order, each only when it applies (Python `serve/server.py` `sampling_keys`; floats as Python `repr`,
so `1.0` keeps its `.0`):

| Key | Sent when | Note |
| --- | --- | --- |
| `temperature=F` | F > 0 | No temperature: greedy. |
| `top_p=F` | F < 1 | |
| `top_k=N` | N >= 0 | 0 and values above 64 are sent as 64 (the sampled path keeps 64 candidates). |
| `min_p=F` | 0 < F <= 1 | |
| `penalty_repeat=F`, `penalty_freq=F`, `penalty_present=F` | not neutral (1, 0, 0) | |
| `penalty_last_n=N` | any penalty is set | Default 64. |
| `seed=N` | N > 0 | |
| `turn_at=POS` | the prompt has a turn start after position 0 | Position of the last `<|im_start|>`. strata-poly keeps a conversation checkpoint there; strata ignores it. |

Example: `GEN 600 temperature=0.7 top_p=0.8 top_k=20 turn_at=184 248045,8678,198,...`

## Events

| Event | Form | Meaning |
| --- | --- | --- |
| `T` | `T <id>` | One generated token. CONTROL tokens such as `<|im_end|>` are not text. |
| `PP` | `PP <done> <total> [ms] [tok/s]` | Prompt-reading progress after each chunk. Doubles as a heartbeat. |
| `RESUME` | `RESUME <n>` | `n` prompt tokens were reused from the cache. |
| `DONE` | see below | The request ended. Exactly one per GEN, also after STOP. |
| `ERR` | `ERR <message>` | The request failed; no DONE follows. |
| `INFO` | `INFO key=value ...` | Engine facts (start-up, VRAM changes, `batch_slots=N`). |

### DONE fields

Positional, space-separated; later engines append fields, so read what is there:

| # | Field | | # | Field |
| --- | --- | --- | --- | --- |
| 1 | generated tokens | | 9 | reused prompt tokens |
| 2 | prompt tokens | | 10 | decode expert-cache hits |
| 3 | prompt ms | | 11 | decode expert-cache lookups |
| 4 | decode ms | | 12 | blobs served from RAM |
| 5 | finish (`eos`, `length`, `stop`, `nonfinite`) | | 13 | blobs read from files |
| 6 | drafts accepted | | 14 | file MB read |
| 7 | drafts offered | | 15 | prompt tokens actually read |
| 8 | (reserved) | | 16 | routed experts read over PCIe |

After the positional fields, `key=value` pairs may follow (`chunk=`, `experts_streamed=`).

## Rules the server must keep

- One request at a time per engine (unless batch slots are configured).
- A client that goes away mid-request: send `STOP`, then keep reading until `DONE` (or `ERR`), so the next request
  starts clean.
- Requests are never truncated: prompt + `max_new` must fit `--max-context`. With the config's `fit_max_tokens`,
  `max_new` is clamped to the room left instead of refusing.
- Output stopping at end-of-sequence is the engine's job; stop strings are the server's.
