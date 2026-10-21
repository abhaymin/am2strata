# Deploying am2Strata

This guide sets up am2Strata on a Linux PC with an NVIDIA GPU, with three ready run configs:

| Model | Config | Engine | Download |
|---|---|---|---|
| Tiel Coder 35B-A3B UD-Q6_K_XL (coding, pictures) | `configs/strata-tiel-coder-35b-a3b-mtp-ud-q6_k_xl.json` | strata-poly | 30 GB |
| MiniCPM-V 4.7 35B-A3B Uncensored AD-Q5_K-Q4_K (pictures, video, PDFs) | `configs/strata-minicpm-v-4.7-35b-a3b-uncensored-ad-q5_k-q4_k.json` | strata-poly | 23 GB |
| Qwen3.8 Flash Next Uncensored IQ3_XXS ("Orca", 262K context, MTP drafts) | `configs/strata-qwen3.8-flash-next-uncensored-iq3_xxs.json` | strata | 85 GB, gated |

The settings in these configs are the ones measured on an RTX 5060 Ti 16 GB with 60 GB of RAM:
- Tiel Coder: ~100 tok/s warm;
- MiniCPM-V: ~90 tok/s warm;
- Orca: ~60 tok/s warm, with a 262K context streamed to RAM.

Other cards work too; the engines size their expert caches to the free VRAM.

## 1. What you need

- Linux (tested on Ubuntu 26.04) and an NVIDIA GPU with its driver: an RTX 30, 40 or 50 card, 12 GB of VRAM or more.
- RAM: about 24 GB for MiniCPM-V, 32 GB for Tiel Coder, and 60 GB for Orca (its experts live in RAM).
- Disk: the downloads above plus about 10% for packs; an NVMe SSD helps.
- CUDA toolkit 12.8 or newer (13.x tested), CMake, Ninja and a C++ compiler, to build the engines.
- Rust, stable: <https://rustup.rs>.
- Optional:
  - `poppler-utils`, for PDFs in chats;
  - `ffmpeg`, for video sent by API clients (the web app cuts video frames itself);
  - `bubblewrap`, to run Python from the chat's canvas.

```bash
sudo apt install build-essential cmake ninja-build git poppler-utils ffmpeg bubblewrap
```

## 2. Get the code

am2Strata drives PolyStrata's engines, so you need both checkouts. The default folders are the ones the configs and
scripts expect; others work, see "Folders" below.

```bash
mkdir -p ~/Development && cd ~/Development
git clone https://github.com/abhaymin/am2strata
git clone --recursive https://github.com/VecSzn/PolyStrata
```

## 3. Build the engines

```bash
cd ~/Development/am2strata
deploy/build-engines.sh --install
```

- **What it builds:** `strata` (Qwen3.8 / Orca), `strata-poly` (Tiel, MiniCPM-V, LFM2, GLM, Gemma, DeepSeek) and
  `strata-vision` (the image encoder).
- **Which cards:** RTX 30 and RTX 50 by default. `--archs "86;89;120"` adds RTX 40.
- **Time:** about 30–60 minutes.
- **Where they go:** `--install` copies them into `PolyStrata/engine/` and `engine-poly/`, keeping the old ones in a
  backup folder.

## 4. Get the models

```bash
deploy/pull-models.sh tiel        # Tiel Coder
deploy/pull-models.sh minicpm     # MiniCPM-V
deploy/pull-models.sh orca --token hf_...   # Orca (gated: accept its terms on huggingface.co first)
```

- **Downloads:** each one resumes after a break, and every file is checked against the SHA-256 Hugging Face publishes.
- **Folders:** files go to `~/Development/Strata-data/models/<folder>`, the folders the configs name.
- **Tokenizer:** the server writes it from the model file the first time it loads the model.

**Orca needs two more things**, which `pull-models.sh orca` makes once, in Rust (no Python needed):

| Command | What it makes |
|---|---|
| `am2strata-run iq-pack --gguf <shard 1> --out <pack> --compat-bf16` | the native pack (index, converted dense tensors, expert table, tokenizer) |
| `am2strata-run mtp-fetch fetch --out <mtp>` | the 31 MTP tensors of the BF16 checkpoint, ~5 GB by range requests, checked against pinned SHA-256s |
| `am2strata-run mtp-pack --src <mtp> --experts q2_0 --out <mtp>/mtp-q2_0.gguf` | the MTP draft block as a GGUF |
| `am2strata-run mtp-rt --gguf <mtp>/mtp-q2_0.gguf --out <mtp>/rt` | the draft runtime the engine loads (`--mtp`) |

These are ports of PolyStrata's `tools/iq_pack.py`, `mtp_fetch.py`, `mtp_pack.py` and `mtp_rt.py`. They write the
same files byte for byte (checked on the installed Orca pack and MTP files, and on recorded fixtures), and they are
faster: the Orca pack takes ~4 s, the MTP GGUF ~25 s. The draft vocabulary, `data/draft_vocab.bin`, is a data file of
PolyStrata's repository and is copied from there.

Any other GGUF from Hugging Face whose architecture the engines run can be added later, from the web app (Settings ›
Models) or with `target/release/am2strata-run pull <repo> --quant <quant>`. Either way the run config is written for
you.

## 5. Install

```bash
deploy/install.sh                                   # this PC only, no key
deploy/install.sh --host 0.0.0.0 --new-key          # your network, with a new API key (printed once)
deploy/install.sh --host 0.0.0.0 --new-key --service --model tiel   # ... and start at boot with Tiel Coder
```

The script does four things:
- **Builds** am2Strata.
- **Copies the configs** into PolyStrata's folder (`~/Development/PolyStrata/strata-*.json`), with your folders, port,
  address and key filled in. Each file is readable only by you, and an existing file is kept unless you pass `--force`.
- **Checks** the GPU and the optional tools.
- **With `--service`:** installs the systemd user service `am2strata-boot`. To start it before you log in, run
  `loginctl enable-linger $USER` once.

## 6. Run

```bash
target/release/am2strata-run serve --port 8000 --model tiel        # in the foreground; Ctrl+C stops it
```

Open <http://127.0.0.1:8000/>. The web app has:
- chat, with the canvas;
- the Monitor;
- Settings: switch models, download new ones, tune the run config;
- API calls.

Clients use the same address:

| API | Base URL |
|---|---|
| OpenAI | `http://<host>:8000/v1` |
| Anthropic | `http://<host>:8000` |
| Responses (Codex) | `http://<host>:8000/v1/responses` |

Model names are the configs' `model_name`.

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "tiel-coder-35b-a3b-mtp-ud-q6_k_xl", "messages": [{"role": "user", "content": "Hello"}]}'
```

**Service commands:**

| Command | What it does |
|---|---|
| `am2strata-run install` / `uninstall` | the boot service |
| `systemctl --user status am2strata-boot` | its state |
| `journalctl --user -u am2strata-boot -f` | its log |
| `am2strata-run temp-service --model X` | a temporary service, e.g. to try a model; it stops on reboot |
| `am2strata-run stop-temp-service` | stops the temporary service |

When PolyStrata's own `strata-orca` service is installed, these swap with it and give it back when am2Strata stops.

## Folders

The configs in `configs/` are written with two placeholders, so they can be shared:

| Placeholder | Meaning | Default |
|---|---|---|
| `${STRATA_HOME}` | PolyStrata's folder: engines, run configs, logs | `~/Development/PolyStrata` |
| `${STRATA_DATA}` | models, packs, tokenizers | `~/Development/Strata-data` |

Any string in a run config may also use `~/` or another `${VARIABLE}`. For other folders:
- pass `--strata-home` / `--strata-data` to the deploy scripts, or set the variables;
- `install.sh` writes the real folders into the copies it installs, so a service needs no environment.

## Tuning

Settings › Model config in the web app changes the common keys, and **Save and reload model** applies them. Things
you may want to change:

| Setting | Where | What it does |
|---|---|---|
| Context | `--max-context` in `args` | less context leaves more VRAM for experts (faster on small cards) |
| KV cache | `--kv` (`q8_0`; Orca `k8v4`) | KV precision; `--kv-resident N` streams KV beyond N positions to RAM (Orca) |
| Conversation cache | `--conversation-cache-mib` | RAM kept for parked conversations, so follow-ups are not read again |
| VRAM left free | `--vram-reserve-mib`, `vram_reserve_mib` | for the desktop or other programs |
| Idle unload | `idle_unload_s` | frees the GPU after this many seconds without requests |
| Code run time | `code_run_timeout_s`, `"code_run": false` | the canvas's Python runs |

One model is in memory at a time; switching (web app, `POST /api/models/load`) stops the loaded one first.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "no usable run config … its engine … does not exist" | build and install the engines (step 3) |
| "compiled for sm_X but this GPU is sm_Y" | rebuild with your card's architecture: `deploy/build-engines.sh --archs "<yours>" --install` |
| The model stops loading with an out-of-memory error | lower `--max-context`, raise `--vram-reserve-mib`, or close other GPU programs |
| Slow first answers | normal: the expert cache warms up over the first requests |
| 401 from clients | send `Authorization: Bearer <api_key>` (the key in the run config) |
| PDFs are refused | `sudo apt install poppler-utils` |
| Video from an API client is refused | `sudo apt install ffmpeg` (the web app does not need it) |
| Run in the canvas says it is off | `sudo apt install bubblewrap` |

Logs: each model's engine log is the config's `log` file. The server logs to the terminal, or to
`journalctl --user -u am2strata-boot` for the service.
