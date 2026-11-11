# Model optimizers: am2Strata tunes each model's run settings for this machine

Owner's request (2026-11-11): am2Strata should optimise the models it runs on Strata's engines, so each model runs as
fast as this machine allows, within its memory, with no hand tuning.

**State: steps 1-5 built, and measured on this PC (2026-11-11); the cluster split (step 6) waits for CLUSTER-PLAN phase 2.** Step 4:
Settings › Models has **Tune** on each model (with a confirmation saying how long it takes and that chats wait), its
progress (trial N, the last speed), the result ("Tuned: 23.4 tok/s, was 18.9", or "no setting gained 3%"), and
**Use the run config / Use the tuned settings**; a tune started through the API is followed too.

Step 5, quantization advice:
- `POST /api/hub/quants` marks each quant `fit`: "gpu", "gpu+ram" or "too big". The fit counts the VRAM of every
  card, the RAM less 8 GB for the system, and about 2 GB for the context.
- **Recommended:** the largest quant that fits, leaving out 16- and 32-bit files.
- **Fastest:** the largest quant that fits in VRAM alone.
- Settings › Models › Add a model chooses the recommended quant and labels each one.
- On this PC, for unsloth/Qwen3-30B-A3B-GGUF: recommended UD-Q8_K_XL (36.0 GB, VRAM + RAM), fastest UD-Q6_K_XL
  (26.3 GB).
- A download is still the user's choice.
- **Step 1, the machine's facts:**
  - `GET /api/machine` gives the CPU (name, cores, threads, and AVX2 / AVX-512 / AMX support), the RAM, each GPU
    (VRAM, PCI address, PCIe generation and width now and at most, and the link's speed in GB/s), and the models'
    disk (device, model, whether it spins), with a fingerprint of the hardware.
  - `POST /api/machine/measure` measures the disk's sequential read speed: up to 1 GiB of the largest model file,
    with the page cache dropped first.
  - Both are kept in `<data>/machine.json`. A new fingerprint (a card added, RAM changed) drops the old
    measurements.
- **Step 2, the search** (`crates/am2strata/src/optimizer.rs`):
  - `POST /api/models/{id}/optimise` with `{"objective": "decode"|"prefill", "budget_s": 1200}` runs in the
    background and holds the turn (chats wait, shown in the queue). `GET /api/optimise` follows it.
  - It tunes only the flags the run config already passes: `-t`, `-ub`, `-b`, `--resident-experts`. The KV types
    are never changed.
  - Each trial starts the engine with those settings and runs the prompt set greedily: a short chat, an
    8,000-token prompt, a 256-token story. Each trial is capped at 10 minutes.
  - A candidate is dropped when it fails to start, fails a prompt, or answers with other tokens than the run config.
  - Coordinate descent: the best value is kept when it gains 3% or more, in up to 3 rounds, within the budget.
- **Step 3, profiles used at load:**
  - The result is saved as `<config>.tuned.json`, with every trial, the machine and the date. The run config is
    never changed.
  - A load uses the profile when it is on and was tuned on this hardware, and `/health` "adapted" says so.
  - `POST /api/models/{id}/tuned {"enabled": false}` turns it off.
  - `GET /api/models` shows each model's `tuned` block.
  - A profile is turned on only when it gained 3% or more.
- **Tested on the mock** (`the_optimizer_tunes_a_model_on_the_mock`): the mock engine's speed comes from a table
  (`MOCK_SPEED`). The search finds the table's best, drops the setting that fails to start, keeps the run config
  unchanged, and loads the model again with the profile.

## Measured: Tiel Coder Q8 on this PC (2026-11-11)

Hardware: Ryzen 7 7840HS (8 cores, 16 threads, AVX-512), 59.7 GB RAM, 2 × RTX 5060 Ti 16 GB (each at PCIe gen 4 x4,
7.9 GB/s), WD Blue SN580 (2,409 MB/s read). Engine `strata-poly`, layers split across both cards. Objective: decode.
Budget 20 min; the search took 696 s, 12 trials.

| Setting | Decode tok/s | Prefill tok/s | Kept? |
|---|---|---|---|
| The run config (engine defaults) | 70.4 | 393 | the starting point |
| The run config again | 70.3 | 394 | agrees with itself for 64 / 11 / 116 tokens of the three answers |
| `--threads` 6 / 8 / 12 | 69.9 / 70.7 / 71.2 | 401 / **415** / 414 | under the 3% gain |
| `--window` 1 / 4 | 70.9 / 70.8 | 397 / 395 | under the 3% gain |
| `--window 12` | the engine stopped | | dropped |
| `--prompt-batch` 512 / 2048 | 70.6 / 70.1 | 386 / 398 | under the 3% gain |
| `--expert-swaps 32` | 70.2 | 394 | dropped: its answers parted too soon (11 tokens against 64) |
| `--expert-swaps 128` | the engine stopped | | dropped |

- **Decode:** nothing gains 3% (the best, `--threads 12`, gains 1.1%). The profile is written but off: Tiel's
  defaults are already within about 1% of the best on this machine.
- **Prefill:** `--threads 8` reads prompts 5.5% faster (415 against 393 tok/s). An `"objective": "prefill"` tune
  would keep it.
- **Two settings stop the engine:** `--window 12` and `--expert-swaps 128` (probably out of VRAM). They are dropped,
  and the model in memory was started again each time.
- **The first try** (the floor that wanted exactly the same tokens) dropped every candidate. Tiel's greedy answers
  are not bit-exact from run to run, which is why the run config is now run twice.

## Why it is needed

A Strata run config has many settings that decide the speed, and the best values depend on the model and the machine:
- how many experts stay resident on the GPU;
- the KV cache's types (k8v4 and others);
- the context;
- the micro-batch (`-ub`);
- the threads (`-t`);
- the split across the GPUs;
- the expert cache's size;
- streaming the KV.

Today they are set by hand, and some choices are traps: on this PC, `-t 16` and `-ub 64` made Qwen slower, and about
20 tok/s was the wall on the two RTX 5060 Ti. A wrong value costs speed, and a value that does not fit costs a failed
load.

## What the optimizer does

1. **Know the machine:** GPUs, VRAM, RAM, CPU cores, the PCIe or USB4 link of each card, the SSD's read speed (for
   streamed experts). Measured once, and again when the hardware changes.
2. **A search over the settings that matter,** one model at a time, only when the server is idle (the queue empty for
   N minutes) or when asked (Settings › Models › Optimise):
   - **Prune first:** the starting point is the run config as it is, and the memory guard's numbers (PRODUCTION-PLAN
     step 3) leave out what cannot fit;
   - **Measure each candidate:** a fixed prompt set (short chat, a 8,000-token prompt, a long generation), recording
     prefill tok/s, decode tok/s, the time to the first token, peak VRAM and RAM;
   - **Search:** coordinate descent, one setting at a time, with a few tries each, stopping when no step gains 3% or
     more;
   - **Bounded:** each trial is capped (time and memory), a trial that fails or would not fit is dropped, and the
     whole search has a time budget (default 20 minutes).
3. **The objective:** decode speed by default, or chosen per model (prefill for agents with long prompts, context
   size, VRAM left for media). Never below a quality floor: the KV types are kept within what the model was checked
   with, and the answers are compared token for token at temperature 0 against the starting config on the prompt set.
4. **The result is a tuned profile:**
   - saved beside the run config (`strata-<model>.tuned.json`), with the measurements, the machine it was tuned on,
     and the date;
   - shown in Settings › Models as "tuned: 23.4 tok/s, was 18.9";
   - used from the next load, and easy to turn off or redo;
   - never overwriting the user's own run config.
5. **Quantization advice:** for a model with several GGUF quants on Hugging Face, which one fits this machine with
   the most quality, from the measured memory per quant (a download is always the user's choice).
6. **The cluster** (CLUSTER-PLAN.md): the same search for a split across nodes, with the network's measured speed
   in the model.

## Tests

- **Mock engine:** the mock reports speed and memory per setting from a table, and the search must find the table's
  best within its budget, drop the candidates that do not fit, keep the quality floor, write the profile, and never
  touch the run config.
- **Real:** tune Tiel Q8 on this PC and report the before and after speed with the conditions.

## Order

1. Machine facts.
2. The mock and the search.
3. Profiles used at load.
4. Settings › Models.
5. Quantization advice.
6. The cluster split.
