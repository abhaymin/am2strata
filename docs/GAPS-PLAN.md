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
| Dubbing (`/v1/media/dub`) | none | none | A Dub card in Media Generation; `am2strata-run media dub` |
| Media check (`/v1/media/check`) | none | none | A Check card; `am2strata-run media check <file>` |
| Productions (`/v1/media/productions`) | none | none | A Productions card with parts and progress; `am2strata-run media produce plan.json` |
| Images, video, speech, songs | yes | only `media-pull` | `am2strata-run media image/video/speak/song` |
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

## Proposed order

1. **Housekeeping (small, this week):** bring the ledger and the README status table up to date; v0.1.9 release
   notes (browser use, the dub and session fixes); `docs/MEDIA.md` and `docs/CLIENTS.md` (C).
2. **CLI and UI parity for media (B):** `am2strata-run media …` for dub, check, produce, image, video, speak, song; Dub,
   Check and Productions cards. Each with a contract row and an integration test, as browser use has.
3. **Video without ffmpeg (A14):** frames from a Rust decoder, so the video checks and video in chat work everywhere.
4. **Train from the tab, then on the GPU (A1).**
5. **A browser-use demo video (D)**, then v0.1.9.
6. Then by the owner's priority: Settings › Clients editing (A8), video productions (A3), native voices (A5), device
   lanes (A6), the MCP timeout (A7), computer use on macOS/Windows (A2), clusters on a second PC (A9). Engine families
   (A11), songs (A12), long video quality (A4), the knowledge cache (A13) and hardware (A16) continue in their own plans.

Each step lands with tests (unit, live where a real browser or model is involved, a contract row per route), its ledger
row changed in the same commit, and documentation in the same PR.
