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

`/v1/models` lists the model a request reaches. With `"switch_on_request": true` in the config the server starts with, it lists every installed model and a request giving one's whole name loads it, the loaded one stopped first. Other names go to the loaded model.

**Unsloth:** am2Strata works as an external endpoint for Unsloth, in two ways.

- **Unsloth Studio's chat:** Settings › Connections › Add Provider › llama.cpp, with base URL `http://<host>:8000/v1` and the server's API key.
- **`unsloth start`:** the agent runs on am2Strata instead of Studio:

  ```bash
  export UNSLOTH_STUDIO_URL=http://127.0.0.1:8000
  export UNSLOTH_API_KEY=<the server's API key>
  unsloth start claude
  ```

  Unsloth remembers the key per server, so later runs need only `UNSLOTH_STUDIO_URL`. A server without an API key still needs some `UNSLOTH_API_KEY` value, because Unsloth's automatic key works only with its own server. `--model <name>` loads another installed model. Context length and the other load settings come from the run config, so `unsloth start` flags for them are refused. When the agent exits, Unsloth's message about `unsloth studio stop` doesn't apply: am2Strata keeps running.

  This relies on what the `unsloth` CLI checks in `/api/health` (`crates/am2strata/src/unsloth.rs`, read from Unsloth 2026.9.11). If an Unsloth update reports no running server, look there first.

**Service commands:**

| Command | What it does |
|---|---|
| `am2strata-run install` / `uninstall` | the boot service |
| `systemctl --user status am2strata-boot` | its state |
| `journalctl --user -u am2strata-boot -f` | its log |
| `am2strata-run temp-service --model X` | a temporary service, e.g. to try a model; it stops on reboot |
| `am2strata-run stop-temp-service` | stops the temporary service |

When PolyStrata's own `strata-orca` service is installed, these swap with it and give it back when am2Strata stops.

## Keeping up with PolyStrata and Strata

am2Strata runs PolyStrata's engines and carries ports of its server, tokenizer and pack tools in Rust;
`tools/upstream.json` records which upstream file each port is current with (by git blob). To see what changed since:

```bash
am2strata-run upstream-check
```

It reads GitHub (set `GITHUB_TOKEN` to raise the rate limit) and your PolyStrata checkout, changes nothing, and lists:
how far PolyStrata's main is past the port and how far your checkout is behind it; Strata's newest tags (Strata 0.2
is not merged into PolyStrata yet - PolyStrata, and so the engines, carry Strata 0.1.45); and per ported file whether
it changed upstream.

Updating the engines (what was done on 2026-10-26): back up `engine-poly/` and `engine/`, `git pull --ff-only` in
PolyStrata, rebuild with its own `setup.py` routine (`build_engine_poly`: build-poly, CUDA 13, sm_86 and sm_120) and,
when the shared sources changed, `cmake --build build-dual --target strata` for Orca's engine; swap the binaries with a
rename, never a copy over a running engine; restart am2Strata only when idle (`am2strata-run temp-service`).

## Security

A server that listens beyond this PC (`--host 0.0.0.0`) should have an `"api_key"` in the default model's run
config. Every route except `/health`, the web app's files and the sandboxed canvas previews needs it; it is compared in
constant time. Besides the key:

- **Names it answers to.** A request whose `Host` is not an IP address, `localhost`, this PC's name (with `.local`,
  `.lan`, `.home`, `.localdomain`, `.internal` or `.home.arpa`), its Tailscale name, or a trusted origin's host is
  refused (421): a web page that points its own domain at this PC (DNS rebinding) cannot pass for the app.
  Add other names with `"allowed_hosts": ["ai.example.lan"]`.
- **Other web pages.** Settings, the run config, MCP tools, running code, model loading and unloading, downloads and
  the media tools accept only this server's own page or a `"trusted_origins"` entry. The app may not be framed by
  another site (`X-Frame-Options: SAMEORIGIN`).
- **What a request may read.** Pictures, videos and documents given as local file paths are read only when the server
  listens on this PC alone (`"allow_local_files": true` allows them otherwise). URLs are fetched only from public
  addresses, every redirect checked again (`"allow_private_urls": true` allows the LAN). Without these an API client
  could read any file of this account, or your LAN, through the model's answer.
- **Running code** (the canvas's Python) runs in bubblewrap with no network, a read-only system and no home folder,
  and is off on a server that listens beyond this PC without a key.
- **Keys for others.** Give each app or person a generated key instead of the main one: it cannot reach the
  settings, can be limited and revoked, and its use is counted ("API keys and usage" below).
- **Limits.** 256 MB per request, at most 64 jobs waiting in the queue (a client that leaves takes its place out), 4
  audio attachments per chat request, 3 media jobs per answer.

Dependencies are checked against the OSV / RustSec advisories. On 2026-10-28, after `cargo update`, what remains is
`zip` 8.6 (its advisories are about `extract()`, which neither candle nor mold calls; the fix is zip 9, waiting on
upstream candle), `block-buffer` 0.10.4 (fixed only in 0.12; the panic path is not reached), and the unmaintained
`paste` and `number_prefix` (no known flaw).

## API keys and usage

The run config's `"api_key"` is the **main key**: it opens everything, the settings included. For everything else that
connects (an app, a person, an agent, another machine of a cluster) make a key of its own while the server runs. No
restart: a key works as soon as it is made and stops as soon as it is revoked.

**Make one.** Settings › Clients › New key in the web app, or:

```sh
am2strata-run keys create --name "phone app" --tools web,memory --expires 30d
am2strata-run keys create --name ci --requests-per-day 500 --tokens-per-day 2000000 --models qwen3.5-35b
am2strata-run keys list
am2strata-run keys revoke <id>
am2strata-run keys usage --days 30            # everyone: totals, per key, per model, per day
am2strata-run keys usage --key <id>           # one key ("owner" is the main key, "client:<name>" a run-config profile)
```

The key (`am2_` and 40 random characters) is printed **once**. The server keeps only its SHA-256 hash, in
`<data-dir>/api-keys.json` (readable by this account only), so a lost key cannot be read back: revoke it and make
another. A client sends it as any API key: `Authorization: Bearer am2_...` or `x-api-key`.

**What a key may do.** It is a client profile (see `"clients"` in the run config) made at run time:

- It opens the model APIs: chat, messages, responses, Gemini, pictures, video, speech, `/v1/models`, `/v1/memory`.
  It never opens the settings, the run config, the Workshop, the Computer tab, MCP server settings, or the keys.
- `tools`: the tool sets its chats get with no request flag: `memory`, `web`, `image`, `video`, `speech`, `music`,
  `browser`, `code`, `mcp:<server>`, `mcp:*`. None by default.
- `memory`: its memory space (default: one of its own, so keys do not read each other's facts).
- `expires`: `30d`, `12h`, or never. An expired or revoked key is refused with 401 and the reason.
- Limits per UTC day: `requests_per_day`, `tokens_per_day` (prompt + completion). Past one, the key gets 429 with the
  reason and a `Retry-After` until the next day. The tokens limit is checked when a request starts, so the request
  that crosses it still finishes.
- `models`: the only models it may name. A request for another gets 403; a request that names none is checked
  against the loaded model.

**Usage.** Every model API call is counted, streamed answers too: per UTC day, per caller, per model: requests,
prompt, completion and cached tokens, media jobs, errors, and the last use. The main key is counted as `owner`, a
run-config client profile as `client:<name>`, each key by its id. Settings › Clients shows it (per key, per model, a
bar a day); the counts are kept in `<data-dir>/usage/YYYY-MM.json`, written every 20 s while they change and at
shutdown (a server that is killed loses at most those seconds).

The API, with the main key (write calls only from the server's own page or a trusted origin):

| Call | What |
| --- | --- |
| `GET /api/keys` | the keys (no secrets), each with today's and 30 days' use |
| `POST /api/keys` | make one: `{"name", "tools", "memory", "expires_in": "30d"` or `"expires_at"`, `"limits": {"requests_per_day", "tokens_per_day"}, "models"}`; the answer's `"key"` is the key, this once |
| `POST /api/keys/{id}` | change any of those (`null` clears a limit, the expiry or the model list) |
| `DELETE /api/keys/{id}` | revoke (the record and its usage stay) |
| `GET /api/usage?days=30&key=<id>` | totals, `by_day`, `by_model`, `by_key` |

A key reads its own usage and limits with `GET /v1/usage?days=30`.

On a server with no main key every request already passes without a key, so generated keys limit nothing there: set
`"api_key"` first.

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

## Releases and CI

- **`.github/workflows/ci.yml`:** Linux tests, on code changes only. A new push cancels the run it replaces.
- **The regression suite** ([REGRESSION.md](REGRESSION.md)): a contract row for every route, golden response shapes, a test for
  every bug fixed before. After an update, check the running server: `deploy/regression.sh --url http://127.0.0.1:8000 --live-only`
  (3 short model calls, no changes).
- **`.github/workflows/release.yml`:** builds and tests Linux, Windows and macOS on a `v*` tag, then leaves a draft release to
  publish by hand. Linux builds on Ubuntu 22.04 so that its binaries run on older systems. Run it by hand from the Actions tab to
  check all three before tagging; that publishes nothing.
- **The portable downloads** hold the server and the Rust engine with wgpu, built without CUDA or ROCm, so they need no toolkit.
- **`deploy/make-release.sh`:** the NVIDIA download, built on a PC with the engines. It needs the CUDA 13 runtime and glibc 2.43.

Neither download holds the media engine, whose audio and video codecs have their own terms (NOTICE), or any run config.

## HTTPS and secure WebSockets

am2Strata can answer over HTTPS (and `wss://` for streams) on its own port, **8001** by default, next to the API's
plain-HTTP port. Turn it on in the web app (**Settings › Secure connection**), from a terminal, or in the run config.
It applies at once, without a restart, and is kept in the run config.

Three ways to get a certificate:

| | What it is | Command |
|---|---|---|
| **Self-signed** | made on this PC for every name and address it has (localhost, its hostname, LAN and Tailscale addresses); browsers warn once, so compare the SHA-256 shown in Settings with the browser's | `am2strata-run tls self-signed` |
| **Tailscale** | a real certificate for this PC's tailnet name (`<pc>.<tailnet>.ts.net`), trusted by browsers, renewed daily. Needs MagicDNS and **HTTPS Certificates** turned on in the Tailscale admin console (DNS page); if `tailscale cert` says access denied, run `sudo tailscale set --operator=$USER` once | `am2strata-run tls tailscale` |
| **Your own** | any PEM certificate (with its chain) and key; checked to belong together before it is used, reloaded when the files change | `am2strata-run tls load --cert cert.pem --key key.pem` |

`am2strata-run tls status` shows what is in force (names, issuer, expiry, fingerprint, the addresses to open);
`tls renew` gets a fresh certificate; `tls off` turns HTTPS off. `--port` and `--http` set the port and what plain
HTTP does.

**Plain HTTP once HTTPS is on** (`"http"`): `keep` (the default) changes nothing: the site and the API answer on
`http://` and `ws://` (port 8000) and on `https://` and `wss://` (port 8001) alike. `local-only` keeps answering this
PC's own tools on 127.0.0.1 and sends every other device to HTTPS (a 308 redirect, so the method and body are kept);
`redirect` sends everyone except `/health`.

**HSTS** is on only with `redirect` (or `"hsts": true`): it then sends `Strict-Transport-Security: max-age=31536000`
on HTTPS responses. A browser applies HSTS to the whole name, every port, so while plain HTTP is kept, HTTPS sends
`max-age=0` instead; that also clears a pin a browser was given before, so `http://` works in it again. Browsers
apply HSTS only to a trusted name (the Tailscale one), not to bare IP addresses or certificates they do not trust.

The run config block, if you prefer to write it:

```json
"tls": { "mode": "tailscale", "https_port": 8001, "http": "keep" }
```

`mode` is `self-signed`, `tailscale` or `files` (with `"cert"` and `"key"` paths); a mistake in this block stops
the server at start with the reason, so it never runs on plain HTTP by surprise. The certificate and key live in
`<data>/server-tls/` (the key readable by its owner only).

**Alternative:** `tailscale serve --https=443 http://127.0.0.1:8000` puts Tailscale's own HTTPS in front of
am2Strata without changing it.

## The live channel: ws:// and wss://

One continuous WebSocket between a client and the server: the server **pushes** what changes (no polling), and the
client can send its API requests over the same connection. The web app uses it for the Monitor tab and the header's
status pill (the small **Live** mark beside the pill says whether it is connected) and falls back to polling by
itself whenever the socket is not connected. Scripts, other clients and cluster nodes can use it the same way.

- **Address:** `ws://<host>:8000/v1/live` on the plain-HTTP port, `wss://<host>:8001/v1/live` on the HTTPS port.
  Both work at the same time; the plain-HTTP policy above applies to `ws://` as it does to every request.
- **Messages:** JSON, one object per text frame, each with a `"type"`. At most 16 MB per message (larger request
  bodies, and uploads, go over HTTP); an answer over 32 MB is refused with an error (fetch it over HTTP).
- **Key:** the `Authorization: Bearer <key>` header, or — for a browser, which cannot set headers on a WebSocket —
  the first message `{"type":"auth","key":"<key>"}` within 5 seconds. Never in the address. A refused key gets an
  `error` and the socket is closed with code **4401**. A web page of another origin is refused (403) before the
  socket opens, as for Settings (`trusted_origins` in the run config allows more origins).
- **Who gets what:** the server's own key gets everything. A client profile's key (`"clients"` in the run config)
  gets the topics `status` and `queue` (the queue without anyone else's jobs) and may send requests, which are
  answered exactly as over HTTP with that key — so never the settings, the Workshop, the Computer tab or another
  user's data.

After the key is accepted the server says
`{"type":"ready","role":"owner"|"client"|"node","topics":[...],"version":"..."}` (`topics`: what this key may follow).

### Pushed updates

```json
{"type":"subscribe","topics":["status","queue"]}
{"type":"unsubscribe","topics":["queue"]}
```

The server answers `{"type":"subscribed","topics":[...]}`, then one
`{"type":"update","topic":"status","data":{...},"snapshot":true}` per topic (where it stands now), and after that an
`update` (without `snapshot`) **each time that state changes** — never on a timer, never the same state twice.

| Topic | `data` | Sent when |
|---|---|---|
| `status` | the model (`model.id`, `model.name`, `loaded`), `state` (`idle`, `reading`, `generating`, `media`, `switching`, `loading`, `unloading`, `unloaded`), `phase`, `target` (the model being switched to), `media` (kind and step), `queue` (`running`, `waiting` count), `alerts` (count) | the model is loaded, switched or unloaded; a request starts, changes phase or ends; a media job steps |
| `queue` | what `GET /api/queue` answers: the running job and the waiting ones in order. A client profile's key gets `{"running":{"kind"},"waiting":<count>,"mine":[its own waiting jobs]}` | a job joins, takes its turn, leaves; the order changes (the `*_s` clocks are as of the push) |
| `jobs` | `{"jobs":[...]}`: the newest 50 media jobs, each as `GET /v1/media/jobs/{id}` | a media job is queued, runs, steps, ends |
| `workshop` | `{"jobs":[...]}`: as `GET /v1/workshop/jobs` | a Workshop job starts, reports progress, ends |
| `metrics` | what `GET /metrics` answers (the Monitor tab's figures). Another page of the request history: `{"type":"subscribe","topics":["metrics"],"params":{"metrics":{"page":2,"per":50}}}` | about once a second while its figures move (a request runs, the hardware readings change) |

`jobs`, `workshop` and `metrics` need the server's own key. Behind the topics is one task for all connections; it runs
only while someone is subscribed (no subscriber: nothing runs), wakes when the queue or a Workshop job changes, and
otherwise looks once a second.

### Requests over the socket

```json
{"type":"request","id":"abc","method":"POST","path":"/v1/chat/completions","body":{"messages":[...],"stream":true}}
```

The request runs through the same routes as HTTP, with this connection's key, origin and address: every rule, limit,
queue priority, client profile and the Monitor apply identically. `method` is GET (the default), POST, PUT, DELETE or
PATCH; `path` must be under `/v1/` or `/api/` (never `/v1/live` itself); `body` is JSON; `"headers"` may carry
`x-strata-*`, `accept`, `anthropic-version`, `anthropic-beta`, `openai-beta` and `idempotency-key` (never a key, a
Host or an Origin). `id` is yours (a string of up to 128 characters, or a number) and comes back on every message of
that request. Up to 32 requests may run at once on one connection.

| The server sends | When |
|---|---|
| `{"type":"response","id":"abc","status":200,"headers":{...},"body":{...}}` | a plain answer: `body` is the JSON (text for a text answer; `body_base64` for a file). Errors are the HTTP errors, with their status |
| `{"type":"event","id":"abc","data":{...}}` … then `{"type":"done","id":"abc","status":200}` | a streamed answer (SSE): each event's `data` (JSON, or text such as `"[DONE]"`), `"event"` when it has a name (the Anthropic and Responses streams), `"comment"` for a comment line (`queued: 2 ahead` while it waits its turn) |
| `{"type":"cancelled","id":"abc"}` | after `{"type":"cancel","id":"abc"}`: the request is dropped as when an HTTP client disconnects (the model stops generating) and nothing more of it is sent. Closing the socket cancels every request it has running |
| `{"type":"error","id":"abc","message":"..."}` | the message could not be run (a path outside `/v1/` and `/api/`, an id already in use, a stream that broke). Without `id`: a message that was not understood (not JSON, an unknown `type` or topic); the connection stays open |

`{"type":"ping"}` is answered with `{"type":"pong","t":<unix ms>}`. The server also sends a WebSocket ping every 25 s
(browsers and WebSocket libraries answer it by themselves) and closes a connection that has been silent for 60 s.

### From a terminal

```
am2strata-run live                                   # this PC's server: status and queue, one JSON per line, until Ctrl+C
am2strata-run live --topics status,queue,jobs,workshop,metrics
am2strata-run live --url https://pc.tailnet.ts.net:8001          # wss://, a certificate browsers trust
am2strata-run live --url https://192.168.1.20:8001 --cert cert.pem   # wss://, trusting that self-signed certificate
```

Each pushed `update` is one line on stdout (for `jq`, a script, another node); what the connection says (connected,
following, errors) goes to stderr. `--api-key` / `AM2STRATA_API_KEY` and `--url` / `AM2STRATA_URL` as for the other
commands. A self-signed server's certificate is `<data>/server-tls/cert.pem` on that server.

### In the web app's code

`live.on(topic, fn)` (pushed data), `live.has(topic)` (is the socket delivering it — else poll),
`live.request(method, path, body, onEvent)` (a promise of `{status, body}`; `onEvent` per streamed event;
`.cancel()`). It reconnects by itself (1 s, 2 s, 4 s … up to 30 s) and subscribes again. The chat itself still
streams over `fetch` for now.

## The Workshop: training data, convert, quantize

The **Workshop** tab (and `am2strata-run workshop …` in a terminal) makes models on this PC, all in Rust. Jobs run
one at a time, as the separate `am2strata-workshop` command, and keep going if you leave the tab. Only the server's own
API key can use the Workshop. It reads and writes only inside the models folder, `<data>/workshop`, and any folders listed
in the run config's `"workshop": {"dirs": [...]}`.

**Training data** is written by the model you have loaded, through this server's own chat API:

- **Recipes:**
  - *qa* writes questions and answers from documents;
  - *chat* writes multi-turn chats about them;
  - *instruct* writes more examples like a JSONL of your own.
- **Documents:** text, Markdown, code, HTML and PDF (PDF needs `pdftotext`, from poppler-utils). **Add files…** uploads
  them into `<data>/workshop/uploads`.
- **Quality check:** a judge pass scores each answer 1 to 5 against its source passage and drops low scores. Duplicates
  and near-duplicates (MinHash) are dropped too.
- **Output:** `<data>/workshop/datasets/<name>-<recipe>.jsonl` in the `messages` format Unsloth and the trainer read, with
  a `.card.md` beside it saying what it was made from and what was dropped.
- **Review:** page through the records under the form, fix an answer with **Edit**, and remove bad ones with **Delete selected**.

```bash
am2strata-run workshop upload ~/docs/manual.pdf
am2strata-run workshop datagen --source ~/Development/Strata-data/workshop/uploads/manual.pdf --count 200 --min-score 4 --wait
am2strata-run workshop dataset ~/Development/Strata-data/workshop/datasets/manual-qa.jsonl --limit 5
```

A small or base model may not write the JSON the recipe asks for. The job then fails, saying so, and leaves no empty
file. An instruction-tuned model of 7B or more works best. The judge roughly doubles the time.

**Models:** merge a LoRA adapter (PEFT or Unsloth) into its base, convert a Hugging Face Llama-family model to GGUF
(byte for byte as llama.cpp's converter), quantize it (as `llama-quantize --pure`), and install it so it appears in the
model list.

## The Computer tab: see and use the desktop

With computer use on in sandbox mode (Settings › Computer and browser use), the **Computer** tab shows the private
desktop am2Strata works on — a virtual X display of its own, never this PC's screen — live, and lets you use it:

- **Take control** to use it with your own mouse and keyboard (for a sign-in, a setting, a fix the job needs). While
  you have it, am2Strata's computer steps wait; **Hand back** lets the job go on. Closing the tab hands it back.
- **Paste to desktop** / **Copy from desktop** move text between your clipboard and the desktop's. The browser lets a
  page use your clipboard only over HTTPS (see above); on plain HTTP you get a box to paste into instead.
- **Open an app** starts one of the allowed apps (Settings › Computer and browser use › Allowed apps) on that desktop.
- **Screenshot** saves what you see.

It needs `xvfb`, `xdotool` and `xclip` (`sudo apt install xvfb xdotool xclip`). Apps run on the sandbox display with
your environment minus everything that would reach your real session (Wayland, the session bus), so a window can
never open on your own screen. **Use a non-snap browser there** (Mozilla's .deb Firefox, Google Chrome or Chromium
from a .deb): the snap Firefox is confined to your real display and cannot open on the sandbox. GNOME apps (Text
Editor, Files, Calculator) work as they are.

From a terminal: `am2strata-run computer status | screenshot desk.png | take | give | copy | paste "text" |
type "text" | open <app>`. The stream is a WebSocket (`wss://` on the HTTPS port); only the server's own key opens it,
and only from am2Strata's own page.

## The sandbox VM: a whole computer for computer use

Computer use can run in its own small **Debian 13 virtual machine** that am2Strata creates and runs itself (QEMU with
KVM; no root, no Docker). Inside it the agent is root: it installs and removes software, runs a shell and uses a
desktop, and nothing in it can touch this PC. Turn it on in **Settings › Computer and browser use › Mode: Sandbox VM**
(`"computer": {"use": true, "mode": "vm"}`); you can also use the VM yourself from the **Computer** tab without that.

**What this PC needs** (the tab says what is missing): `qemu-system-x86`, `qemu-utils`, `ovmf`, `openssh-client`,
access to `/dev/kvm`, and the Debian cloud image (326 MB, downloaded from cloud.debian.org by the **Download** button
and checked against Debian's SHA512SUMS). The first start sets the desktop up once (about three minutes).

**Light on this PC.**
- It runs only when used: it pauses its CPUs after 2 minutes unused, and after 15 minutes is saved to disk and quit
  (no RAM, no CPU). A start then restores it, with its open apps, in about a second.
- QEMU runs with `CPUWeight=idle`, idle scheduling and I/O class: it gets only the CPU the inference engine is not
  using. It has no GPU and never touches VRAM.
- Memory has a hard ceiling (2 GB by default) and unused memory goes back to the host. It refuses to start if that
  would leave this PC less than `host_reserve_mb` (3 GB) free, so a loaded model is never pushed out.
- Measured here: 0.9 GB of RAM idle with the desktop up, 1–4% of one core; a cold boot 11 s; a restore 0.8 s.

**Software.** The catalog installs with one click or `am2strata-run vm install <app>`: Chromium, Firefox ESR,
Bitwarden (app and browser extension), Tailscale, NetBird, Mullvad VPN, Surfshark, WireGuard and OpenVPN, VS Code,
LibreOffice, Thunderbird, KeePassXC, GIMP, VLC, Obsidian, developer tools and Docker. Anything else:
`--kind apt <packages>`, `--kind flatpak <Flathub id>`, `--kind deb <https URL>`, `--kind appimage <https URL>`.
The agent has the same through its `computer_install` and `computer_shell` tools. Sign-ins (Bitwarden, a VPN, a
browser account) are yours: open the app in the Computer tab, take control and sign in; the VM keeps it.

**Ads.** `am2strata-run vm ads on [--lists pihole,adguard] [site.to.let.through …]` / `vm ads off`, the panel's
*Ads and trackers* card, the agent's `computer_ads` tool, or the **Ads and trackers** app on the VM's own desktop. It
is a DNS filter for every program in the VM (dnsmasq, as Pi-hole uses), with Pi-hole's default list (StevenBlack),
the AdGuard DNS filter, or both; it blocks subdomains and IPv6 too, and adds uBlock Origin to the browsers. A
scraping job that must see a site as it is lists that site to let through, or turns blocking off.

**VPNs.** The VM has its own kernel, so real tunnels work: Tailscale and NetBird join it to your network as its own
device; Mullvad and Surfshark have their apps; any other provider works with its WireGuard or OpenVPN file
(`sudo wg-quick up ./provider.conf`). Only the VM's traffic goes through the VPN, never this PC's.

**Private.** The VM's network is QEMU's user-mode network: it can reach the internet, nothing can reach it. Its screen
and shell go over SSH on this PC's loopback with keys am2Strata made (the VM's host key is checked), into Unix
sockets in a private folder. Everything is in `<data>/sandbox`; **Reset** deletes the VM and the next start makes a
new one.

```bash
am2strata-run vm status
am2strata-run vm start
am2strata-run vm install chromium
am2strata-run vm exec --root -- apt list --installed
am2strata-run vm settings memory_mb=4096 cpus=4
am2strata-run vm stop
```

## The browser tool: your own Chrome

"Use a browser" (the chat's `strata_browser`) normally drives a fresh headless Chrome on the server, with none of your
logins. To have the model drive **your own browser** instead, with your sessions, so it reaches pages you are signed in to
and you clear any site check yourself, start Chrome with a debugging port and a profile folder of its own:

```bash
google-chrome --remote-debugging-port=9222 --user-data-dir="$HOME/.am2strata-chrome"
```

Then add to the run config (`strata-<model>.json`):

```json
"browser": { "connect": "http://127.0.0.1:9222" }
```

am2Strata attaches to that Chrome: it opens and drives tabs there, and when it stops it closes only the tabs it opened,
never your browser. Downloads are still denied, private and LAN addresses are still blocked unless `allow_private_urls`,
and typing a password or submitting a form still waits for your Allow. Sign in to that Chrome once and the sessions stay in
its profile folder. This is your own session, used by your choice; am2Strata does nothing to defeat a site's checks.
