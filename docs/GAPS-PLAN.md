# Gaps: what is missing or half done, in code, documentation and demos

Analysis of 2026-11-15 (v0.1.8 plus browser use, PRs #35-#43), at the owner's request: "the function implementation that
are missing or not complete as demos or documentation". Sources: every plan in `docs/`, the ledger, the README status
table, the routes in `main.rs`, the web app, `am2strata-run`'s commands and the published videos. Each item says what
is missing, why it matters, and the check that closes it. The order at the end is a proposal.

## A. Features that are half built

| # | Feature | What is missing | Where it is written down | Closed when |
|---|---|---|---|---|
| A1 | Training (Workshop) | Training runs only from the CLI and only on the CPU. Missing: the Train card in the Workshop tab; GPU training; QLoRA (a 4-bit base); expert adapters for MoE models. | MODEL-WORKSHOP-PLAN "Not done"; DEPLOYMENT "Not done" | A LoRA trained from the tab on the GPU; loss matches the CPU run within 1% on the fixture model |
| A2 | Computer use off Linux | macOS and Windows desktops are stubs that refuse ("not supported on this platform yet"). | computer/backend.rs; COMPUTER-USE-PLAN §6 | A click, type and screenshot test passes on each platform's CI runner |
| A3 | Productions | Video productions; MP3 output; chapters in the file's own metadata. | LONG-MEDIA-PLAN "Not yet" | A production with a video part returns one MP4 with chapters |
| A4 | Long video | Later clips degrade (detail, purple sky streaks); an anchor frame; an overnight long piece at 1280×704. | LONG-MEDIA-PLAN "Next" | 5 min at 640×352 with no visible streaks, judged on stills every 30 s |
| A5 | Native voices | Native voices for Arabic, Russian, Korean and the rest (8 languages today). | VOICES-PLAN; ledger row 4 | Each new language: the take check hears ≥ 90% of its words |
| A6 | Speech, device lanes | A CPU speech job beside a GPU chat; the native Chatterbox port; the Perth watermark. | CHATTERBOX-PLAN; ledger row 7 | A chat streams while a CPU speech job runs, with neither stopped |
| A7 | MCP servers | A per-server timeout (the Settings form has no field). | MCP-UI-PLAN; ledger row 5 | A slow server times out at its own limit |
| A8 | Server tools for clients | Settings › Clients editing (profiles exist only in the run config); media on a second card inside the turn. | SERVER-TOOLS-PLAN; ledger rows 6, 10 | A profile made and changed in Settings works with its key at once |
| A9 | Clusters | A test on a second machine; NAT hole punching; layers across machines (phase 2). | CLUSTER-PLAN | A chat answered by a worker on another PC |
| A10 | Optimizer | The cluster split (step 6); a prefill tune for agents. | OPTIMIZER-PLAN | Tuned prefill measured faster on an agent's long prompt |
| A11 | Engine | Families `gemma4`, `glm4moe`, `deepseek4`; vision through the Rust engine; embeddings and rerank. | ENGINE-PLAN | Each family token-identical to llama.cpp at temperature 0 |
| A12 | Songs | One seed in four still misses lyrics; a better judge than Whisper for Malayalam and other Indian languages. | ACESTEP-RUST-PLAN 6b | Lyrics heard ≥ 80% on 4 of 4 seeds |
| A13 | Knowledge cache | Parts A-C and E. | KNOWLEDGE-CACHE | Per the plan |
| A14 | Video checks | `check_media`'s video checks (frame jumps, frozen stretches) and video-as-frames in chat need ffmpeg, which this PC does not have. Under the all-Rust rule, the frames should come from a Rust decoder. | media_check.rs; README "Video" row | A video check runs on a PC without ffmpeg |
| A15 | Browser use, next steps | Upload support for the sandbox target (its files live in the VM); a recorder that turns a chat's browser steps into a site test; schedules for site tests; datasets shown in chat answers. | BROWSER-USE.md | Each with a live test like browser_live_tests.rs |
| A16 | Hardware | A reboot test with both GPUs; Coral TPU driver; the AMD NPU (detected, unused). | V2-PLAN | Per the plan |

## B. Features with no web app screen or no CLI command

The rule since 2026-11-14 is that every big feature has both.

| Feature | Web app | CLI | Missing |
|---|---|---|---|
| Dubbing (`/v1/media/dub`) | none | done | A Dub card in Media Generation; `am2strata-run media dub` |
| Media check (`/v1/media/check`) | none | done | A Check card; `am2strata-run media check <file>` |
| Productions (`/v1/media/productions`) | none | done | A Productions card with parts and progress; `am2strata-run media produce plan.json` |
| Images, video, speech, songs | yes | done | `am2strata-run media image/video/speak/song` |
| Clusters | Settings › Cluster | none | `am2strata-run cluster status/pair/join/forget` |
| Optimizer (Tune) | Settings › Models | none | `am2strata-run tune <model>` |
| Client profiles | read-only | none | Editing (A8); `am2strata-run clients` |
| Memory (`/v1/memory`) | in chat | none | `am2strata-run memory list/add/forget` |

## C. Missing or out-of-date documentation

- **No user documentation at all** for dubbing, the media check, productions, long video, the image/video/speech/song
  APIs (`/v1/images`, `/v1/videos`, `/v1/audio`, music), client profiles and memory. DEPLOYMENT.md mentions some only in
  passing. Proposed: `docs/MEDIA.md` (every media API with an example request and answer), `docs/CLIENTS.md` (profiles,
  keys, tools by key, limits), and a "Memory" section in DEPLOYMENT.md.
- **The README status table** stops at v0.1.8 and has no rows for browser use, dubbing, the media check, productions,
  long video, clusters or the optimizer.
- **The ledger is stale.** It still says release v0.1.4, has no rows for the Workshop trainer, the sandbox VM, HTTPS, API
  keys, the live channel or browser use, and was last changed on 11 Nov. It should be brought up to date in the same PR
  as this plan, and kept up to date from then on (its own rule).
- **The site's feature grid** has no browser use (shipped after v0.1.8; add it with the next release).
- **Release notes**: browser use and fixes #36-#43 are unreleased; v0.1.9 needs notes.

## D. Features with no demo video

Shown so far: the tour, 11 languages, songs and v0.1.4, v0.1.8 (Workshop, VM, Computer tab, HTTPS, keys, live
channel), the v0.1.7 follow-up (dubbing and media check, ready, upload pending).

Not shown anywhere: **browser use** (forms, scraping into datasets, maps, site tests, an agent over MCP), **productions**
and **long video**, **clusters**, **Tune**, **client profiles** (tools by key), **the Monitor and API calls in depth**,
the **regression suite**. Browser use is the strongest next demo, and am2Strata's own publishing job (tools/yt_publish.py)
is real footage for it.

## E. Judged against a field test: "Is this how Local AI reaches Frontier Capabilities?" (Codacus, 2026-11-15)

The video built a team of small expert models on one RTX 3060 (Qwen 4B planning; T-Max, 3D coders and others as
experts). It tested the team on real jobs with checkers written and locked before any run: a kanban app, a three.js
pagoda, fixing a broken server in a sandbox without touching production, and a research report. What it found, and what
that means for am2Strata:

| Finding | am2Strata today | Judgement |
|---|---|---|
| Specialists win one checkable lane. A 15B DNA model came within a point of the frontier, and a big model consulting it gained 11 points. | The specialists are media models (Whisper, Parler, ACE-Step, image/video) and tools (browser, search, code). No language-model expert can be consulted. | **New A17:** an `ask_expert` server tool. The main model sends a sub-question to a specialist model (OCR, math, terminal, a domain model) loaded on the CPU or the second card, and the answer comes back into the turn. The expert also gets **the user's original request**, not just the planner's brief: the video's team failed because its experts only saw a short brief. |
| The planner is the weak link. A 4B planner made its experts worse; a **35B-A3B MoE with experts in system RAM** planned well enough, on the same card. | The default model is Tiel Coder 35B-A3B Q8, with experts streamed and resident in RAM: the setup the video ends up recommending. | Keep a 35B-A3B MoE as the default planner and say so plainly (README, site, Add a model). The optimizer's advice (OPTIMIZER-PLAN) should prefer "the biggest MoE that fits with experts in RAM" over a small dense model for agent work. |
| Long jobs fail by accumulation. 20-35 steps, and one wrong command ends the run. Small models score about 27% on Terminal-Bench against about 85% for the frontier. | Seen here too: am2Strata's local model failed an upload task, reported a Save that had not stuck, and changed nothing for 16 minutes in a repeat loop (now guarded). Risky steps ask first; the sandbox VM is separate. | **New A19:** make long jobs safe by design. (1) Sandbox first: computer jobs run in the VM, and a fix for the real machine comes back as a script to approve. (2) A VM snapshot before each job and one-click rollback (today the VM only saves and restores its last state). (3) Verify after acting: a step that changes something reads its result back (as yt_publish.py now does), built into the tool loop. (4) A guard that refuses destructive commands aimed at anything outside the sandbox. |
| Locked checkers before any run are what made the results honest. | Route contracts, golden shapes, live browser tests, and site tests (JUnit). Nothing measures **agent jobs** end to end per model. | **New A18:** `am2strata-run eval`: real jobs with locked checkers (an app, a 3D scene, a broken service in the VM, a research question, a browser job), run against each local model and the planner and expert mixes, with pass rate, time and tokens. It decides the default model and the expert set by measurement, and it is the before-and-after check for training (A1). |
| Facts don't fit a small model: "give your team search". The video's research job scored 0 for every model because the search engines blocked its local search server. | `web_search` is SearXNG if configured, else DuckDuckGo's HTML page: one fallback, which is exactly what gets blocked. | **New A20:** search that does not break. Several backends in turn (SearXNG, DuckDuckGo, Brave/Bing with a key, and **the browser target as the last resort**, reading a results page like a person), rate limits backed off, and a "search is blocked" answer instead of an empty result. |
| A bad habit can be removed with a few hours of RL on a 12 GB card; a weight edit fixed one narrow bug but not a habit spread over a long job. | The Workshop trainer is CPU only and LoRA only. | Raises **A1 (GPU training)**. Training at home is the practical way to give a planner the missing skill, and the eval (A18) is how to check it. RL from checker rewards is the natural next step after LoRA. |
| Local is slow: the 35B took about 30 minutes where the cloud model took 3. | Tok/s is measured and tuned (optimizer, resident experts); clusters spread jobs. | No new item. Report time-to-done in the eval (A18), so speed is judged per job and not only in tok/s. |

## Proposed order (revised after E)

1. **Housekeeping (small):** the ledger and the README status table up to date; v0.1.9 release notes; `docs/MEDIA.md`
   and `docs/CLIENTS.md` (C); state the 35B-A3B planner recommendation (E).
2. **Agent eval with locked checkers (A18).** Measure first: it decides the rest and guards against changes that feel
   better but aren't.
3. **Safe long jobs (A19):** a VM snapshot and rollback per job; verify-after-act in the tool loop; sandbox-first with a
   script for the real machine; the destructive-command guard.
4. **Search that does not break (A20)**, with the browser as the last backend.
5. **`ask_expert` (A17)**, with the original request passed to the expert; judged by the eval (with and without experts).
6. **CLI and UI parity for media (B)**, then **video without ffmpeg (A14)**.
7. **Training from the tab, on the GPU, then RL from checker rewards (A1)**, judged by the eval.
8. **A browser-use demo video (D)**, then v0.1.9.
9. Then by the owner's priority: Settings › Clients editing (A8), video productions (A3), native voices (A5), device
   lanes (A6), the MCP timeout (A7), computer use on macOS/Windows (A2), clusters on a second PC (A9). Engine families
   (A11), songs (A12), long video quality (A4), the knowledge cache (A13) and hardware (A16) continue in their own plans.

Each step lands with tests (unit, live where a real browser or model is involved, a contract row per route), its ledger
row changed in the same commit, and documentation in the same PR.
