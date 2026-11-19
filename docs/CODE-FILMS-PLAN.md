# Code-drawn films (owner, 2026-11-19)

A film whose every frame is drawn in code: canvas or WebGL scenes, rendered frame by frame in am2Strata's own headless
browser, with narration (Chatterbox or Parler), music (ACE-Step) and captions laid on by am2Strata's compose. It uses
no video model and no stock footage. This suits explainers, product demos, infographics, data stories and the
release videos, where diagrams, type and exact timing matter more than painted scenes. Generated pictures and clips
stay the way for songs and moods.

Lemo-Opuscar (github.com/lemomo-ai/lemo-opuscar) prompted the idea. Its code is MIT, but its style guides and prompts
are CC BY-NC 4.0, and putting them into a product needs a commercial licence. So am2Strata ships **its own** scene kit
and styles, written from scratch. Nothing of Opuscar's docs or prompts is copied.

## Why a scene kit, not free-form drawing code

Opuscar relies on a frontier model writing each film's drawing code, and it says other models may not reproduce its
styles well. am2Strata's producer runs on a local model. So the hard part goes into a tested library, and the model
writes a **scene list** (JSON) that the library draws. The model writes code only for a scene the kit cannot express.
A small model fills in a list reliably; it does not write good WebGL.

## Parts

| # | Part | What | Where |
|---|---|---|---|
| 1 | **Frame renderer** | `POST /v1/media/films` `{"page": html or files, "seconds", "fps", "size", "audio"?, "captions"?, "title_card"?, "end_card"?}` gives a media job. am2Strata opens the page in its headless browser at the exact size. The page exports `window.film = {duration, draw(t)}`. For each frame the renderer calls `draw(t)` with an exact time (no real clock, no timers: deterministic) and captures the frame (CDP `Page.captureScreenshot` into PNG, or the canvas read back), then sends the frames into compose's encoder with the audio and captions. Pages run with no network: fonts and assets are served from the job. | `crates/am2strata/src/films.rs`, `browser.rs` |
| 2 | **Stills for the judge** | `media film --stills 8` renders 8 frames as PNGs without encoding, so the producer and the judge can check a film before the minutes of a full render. | the same |
| 3 | **Scene kit** | `web/films/kit.js` (MIT, ours), drawing in canvas 2D first. Components: title, kinetic type, a bullet build, a bar/line/pie chart that animates, a flow diagram with nodes and arrows, a machine/cluster diagram (boxes linked, packets moving), a terminal typing a command and its output, a UI screenshot with a pan, zoom and highlight, a code block with lines lighting up, a number counter, a comparison table, a map pin, and a lower third. Each component: `(t, props) -> draws`, with easing and enter/exit. | `web/films/` |
| 4 | **Styles** (ours) | 5 to start: **Blueprint** (white lines on blue, grid); **Isometric Infographic** (soft 3D blocks); **Terminal Night** (dark mono, green and amber); **Paper Explainer** (warm paper, ink lines, a handwritten feel); **Clean Product** (white, one accent colour, big type). Each style is a palette, fonts (shipped with their licences), line weights, motion rules and a `STYLE.md`, plus a short demo film made with the kit. | `web/films/styles/` |
| 5 | **Scene lists** | `film.json`: `{"style", "size", "fps", "scenes": [{"kind": "flow", "from_s", "to_s", "props": {...}, "say": "narration line"}]}`. The kit turns it into the page. The renderer takes `film.json` directly. The `say` lines become the narration (TTS) and the captions, and each scene's timing follows its spoken line. | kit + `films.rs` |
| 6 | **The producer** | PRODUCER.md gets a "Code film" recipe: write the script → `film.json` → stills → judge the stills → full render → judge the film → thumbnail → upload files. The judge's brief says what each scene must show. | am2strata-promo/producer |
| 7 | **UI and CLI** | Media Generation › Film: paste or upload a `film.json`, pick a style, preview stills, render, play, save. `am2strata-run media film --stills N | --out film.mp4 -- film.json`. A chat tool `make_film` queues it as a job. | web, `bin_media.rs`, media_jobs |
| 8 | **Cluster** | A film is rendered where a browser is: the PC or the Mac. Rendering is CPU and browser work, not the GPU, so it runs beside a GPU job. | cluster.rs (kind "film") |

## The first film: v0.2.0, what's new (long form at least 5 minutes, plus a Short)

- **Facts first:** the script is written from the v0.2.0 commits and PR descriptions (git log, the PR body), not from
  memory. Every claim must trace back to a commit.
- **Scenes:**
  1. a title;
  2. "one machine from many" (the cluster diagram: a PC and a Mac linked, chats and media jobs moving between them);
  3. media routing;
  4. captions drawn natively (a Devanagari line shaping in);
  5. the judge (frames, scores);
  6. GPU training in the Workshop (the CUDA 19x number from the measured run);
  7. the Workshop pipeline;
  8. how to get it;
  9. the end card.
- **Narration and music:** Laura's voice (Chatterbox), with captions from the script, and a calm ACE-Step bed.
- **Style:** Blueprint for the architecture scenes, Terminal Night for the CLI parts.
- **The Short** (up to 3 minutes, 1080x1920): the three best moments, re-laid out for portrait.

## Gates

- **Deterministic:** the same `film.json` gives byte-identical frames on two renders (a test).
- **Exact timing:** the frame count equals seconds x fps. Narration is checked by `media check` (words_heard); the
  pictures are checked against the sound.
- **The kit:** each component has a reference still, with a pixel test against it (small tolerance) on CI.
- **The judge:** as for songs, with a brief per film.

## Order

1. Renderer and stills (part 1, part 2), with the determinism test.
2. The kit with 6 components and 1 style (Blueprint), and `film.json`.
3. CLI and producer recipe; the v0.2.0 film made by the producer.
4. The other components and styles, the UI card, `make_film`, and the cluster kind.
