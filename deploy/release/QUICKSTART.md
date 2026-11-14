# am2Strata quick start

Run AI models on your own PC, with a web app and OpenAI- and Anthropic-compatible APIs. Nothing else to install: this download
holds the server, the web app and a Rust engine.

## Install in one line

This puts am2Strata under your home folder and `am2strata` on your PATH, with no admin rights or sudo, and checks the
download's SHA-256.

**Linux (x86_64, arm64) and macOS (Apple silicon, Intel):**

```bash
curl -fsSL https://github.com/abhaymin/am2strata/releases/latest/download/install.sh | sh
```

**Windows (x64, Arm64), in PowerShell:**

```powershell
irm https://github.com/abhaymin/am2strata/releases/latest/download/install.ps1 | iex
```

Options: `--version X`, `--dir DIR`, `--service` (start at sign-in), `--with-deps` (install what the optional features
miss, after showing the command and asking), `--uninstall`; in PowerShell `-Version`, `-Dir`, `-Service`, `-WithDeps`,
`-Uninstall`. `am2strata doctor` says what each optional feature still needs on this PC.

## Or try it from this folder

**Linux and macOS:**

```bash
tar xzf am2strata-*.tar.gz
cd am2strata-*/
./am2strata pull mistralai/Ministral-3-3B-Instruct-2512-GGUF Q4_K_M
./am2strata
```

**Windows:** unzip the file, open a terminal in the folder, and run:

```bat
am2strata pull mistralai/Ministral-3-3B-Instruct-2512-GGUF Q4_K_M
am2strata
```

Then open <http://127.0.0.1:8000> and pick the model.

Models and settings stay in `~/.local/share/am2strata` (Windows: `%LOCALAPPDATA%\am2strata`). Set `AM2STRATA_DATA` to keep them
elsewhere. `pull <repo>` without a quantization lists what a repository offers. Each model has its own licence, which is yours to
read.

## What each download supports

| | Linux x86_64 | Linux arm64 | macOS Apple silicon | macOS Intel | Windows x64 | Windows Arm64 |
|---|---|---|---|---|---|---|
| Server, web app, APIs, coding agents, HTTPS | yes | yes | yes | yes | yes | yes |
| Rust engine | CPU, or any GPU (wgpu) | CPU | CPU, or any GPU (wgpu) | CPU | CPU, or any GPU (wgpu) | CPU |
| Workshop (convert, quantize, merge, training data) | yes | yes | yes | yes | yes | yes |
| PDFs in chats | needs `poppler-utils` | needs `poppler-utils` | needs `poppler` | needs `poppler` | included | included (x64 Poppler, emulated) |
| Video from API clients | needs `ffmpeg` | needs `ffmpeg` | needs `ffmpeg` | needs `ffmpeg` | needs `ffmpeg` | needs `ffmpeg` |
| Browser tool | Chrome or Chromium, else fetched | an installed Chromium | Chrome, Edge or Chromium, else fetched | Chrome, Edge or Chromium, else fetched | Edge or Chrome | Edge or Chrome |
| Python in the canvas | needs `bubblewrap` | needs `bubblewrap` | no | no | no | no |
| Computer tab (sandbox desktop) | needs `xvfb xdotool xclip` | needs `xvfb xdotool xclip` | no | no | no | no |
| Sandbox VM | needs `qemu-system-x86 qemu-utils ovmf openssh-client` and `/dev/kvm` | no | no | no | no | no |
| Start at sign-in (`--service`, `-Service`) | systemd user service | systemd user service | launchd agent | launchd agent | Startup shortcut | Startup shortcut |

"needs" names the Debian/Ubuntu packages (Homebrew's on macOS). `./am2strata doctor` checks this PC and prints the
command that installs what is missing.

## What to expect

- **Speed:** this engine runs on the CPU, so choose a small model. A 3B model answers at a readable pace. A 35B model such as
  Cyber-Tiel runs at about 4 words a second on a 16-thread desktop CPU, and bigger ones are slower still.
- **Your GPU:** add `--devices vulkan:discrete=8` to the model's run config to try weights on any GPU. This is new and less tested.
  Not in the Linux arm64, macOS Intel and Windows Arm64 downloads, which run on the CPU only.
- **Fast on NVIDIA:** the Linux NVIDIA download runs large models at 60-160 tokens a second with PolyStrata's engines.
- **PDFs in chats:** the Windows download carries Poppler. On Linux and macOS install poppler (`sudo apt install
  poppler-utils`, `brew install poppler`), or run the installer with `--with-deps`.
- **The browser tool** uses a Chrome, Edge or Chromium already on this PC, or Chrome for Testing's headless shell, which
  the Linux and macOS installer fetches when there is none (Linux arm64 has no such build: install Chromium).
- **HTTPS:** `./am2strata tls self-signed` while the server runs; plain HTTP keeps working beside it.
- **Not in this download:** image, video and voice generation, and pictures in chats. Build from source for image, video and voice.

By default the server only answers this PC. Before using `--host 0.0.0.0`, set an API key.

Problems: <https://github.com/abhaymin/am2strata/issues>
