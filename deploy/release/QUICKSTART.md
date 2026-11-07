# am2Strata quick start

Run AI models on your own PC, with a web app and OpenAI- and Anthropic-compatible APIs. Nothing else to install: this download
holds the server, the web app and a Rust engine.

## Try it

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

## What to expect

- **Speed:** this engine runs on the CPU, so choose a small model. A 3B model answers at a readable pace. A 35B model such as
  Cyber-Tiel runs at about 4 words a second on a 16-thread desktop CPU, and bigger ones are slower still.
- **Your GPU:** add `--devices vulkan:discrete=8` to the model's run config to try weights on any GPU. This is new and less tested.
- **Fast on NVIDIA:** the Linux NVIDIA download runs large models at 60-160 tokens a second with PolyStrata's engines.
- **PDFs in chats:** the Windows download carries Poppler. On Linux and macOS install poppler (`sudo apt install
  poppler-utils`, `brew install poppler`), or run the installer with `--pdf`.
- **The browser tool** uses a Chrome, Edge or Chromium already on this PC, or Chrome for Testing's headless shell, which
  the Linux and macOS installer fetches when there is none.
- **Not in this download:** image, video and voice generation, and pictures in chats. Build from source for image, video and voice.

By default the server only answers this PC. Before using `--host 0.0.0.0`, set an API key.

Problems: <https://github.com/abhaymin/am2strata/issues>
