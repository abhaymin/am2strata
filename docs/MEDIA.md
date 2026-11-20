# Media: images, video, speech, songs, productions, dubbing and checks

Every media feature is two things at once: an HTTP endpoint any client can call, and a tool the server gives a chat
(`"strata_media": true`, or a client profile with the kind in its `tools`). The media engine (`am2strata-media`, Rust)
does the work. A GPU job holds the GPU's turn: by default the text model stops for it and loads again with the next
chat. All endpoints take the API key (the owner's or a client key).

## From the command line

`am2strata-run media <action>` is a thin client of the endpoints above, writing or printing the answer it returns. Give `--out` where the action returns a file; only that answer goes to stdout, so the commands pipe cleanly (the `[am2strata] media <action>: <url>` line goes to stderr). One per action:

```bash
am2strata-run media image "a lighthouse at dusk, oil painting" --size 1024x1024 --model z-image-turbo:q8 --out shot.png
am2strata-run media video "waves on a quiet beach at night" --frames 49 --size 1280x704 --format mp4 --out clip.mp4
am2strata-run media speak "Welcome to the evening set." --voice laura --out intro.wav
am2strata-run media transcribe take.wav --model whisper-base            # its text on stdout (or --format srt|vtt|text)
am2strata-run media song "warm acoustic ballad, female vocal" --lyrics words.txt --seconds 180 --out song.wav
am2strata-run media check take.wav --expect "Welcome to the evening set."   # prints each finding; exit 1 when any is not ok
am2strata-run media analyze take.wav --from 300 --seconds 120   # music facts of that window: tempo, repeat, tonic+range, swaras, phrases
am2strata-run media dub dub-request.json --out soundtrack.wav
am2strata-run media produce plan.json --out set.wav                     # posts the plan, polls step N of M, then downloads it
am2strata-run media job mj-1234 --out set.wav                           # its status; with --out also /content
```

- **Where they point:** pass `--url <server>` and `--api-key <key>`, or export `AM2STRATA_URL` and `AM2STRATA_API_KEY`. With neither, the command uses this PC's local server (from its run configs) and that server's key — the one listed in Settings › Clients. For another am2Strata (a sandbox, a second PC), set `AM2STRATA_URL` to the address it is serving at (its `serve --host/--port`, or what `am2strata agent list` shows for it) and `AM2STRATA_API_KEY` to a key issued for that server.

## Images

`POST /v1/images/generations`, OpenAI's images API:

```json
{"prompt": "a lighthouse at dusk, oil painting", "size": "1024x1024", "n": 1, "model": "z-image-turbo:q8",
 "steps": 8, "guidance": 3.5, "seed": 7, "negative_prompt": "", "response_format": "b64_json"}
```

The answer has `data: [{"b64_json": …}]`. The model's own defaults fill in anything left out.

`POST /v1/images/edits`: either multipart (`image` or `image[]`, 1-3 files, `prompt`, `size`, `n`, `model`), or JSON
with `images` as base64 or data URLs. The first image is edited and the others are references; it needs an image
model that takes edits (Qwen Image 2.1).

### How much an edit keeps of the original

`POST /v1/images/edits` takes these besides the image and the prompt:

| Field | What it does | Models |
|---|---|---|
| `strength` (0.05-1.0) | how far the result may move from the image: it is the starting point (default 0.65) | those that redraw from an image: Z-Image, FLUX, SDXL, SD 1.5/3, Qwen Image 1 |
| `mask` (image, file id) | white = may change, black = keep. Those models redraw only there; on every model the original's own pixels are put back outside it, with a soft edge (`mask_feather`, px, default 8) | all |
| `keep_colors` | the result's colours matched to the original's | all |
| `blend` (0-0.95) | how much of the original shows through | all |

An instruction model (Qwen Image 2.1) refuses `strength` by name (it cannot start from the image); `mask`, `blend`
and `keep_colors` work with it. The answer's `strata.edit` says how it was drawn, the controls, and
`original_file_id` (the library id of the image edited), for a before/after. In the app: Media › Image, with a file
picked or **Edit this**: the sliders, the checkbox and **Paint what may change** (a brush, Erase, Invert, Clear);
each edit is shown over its original with a slider between them.

## Video

`POST /v1/videos/generations`:

```json
{"prompt": "waves on a quiet beach at night", "model": "wan22-ti2v-5b:q8", "size": "1280x704",
 "frames": 49, "fps": 24, "steps": 30, "seed": 1, "format": "mp4"}
```

- **Frame counts** are rounded to what the model takes: Wan 4n+1, LTX 8n+1.
- **Longer than one clip:** give `"seconds"` (or `"total_frames"`). The clips are chained, each continuing from the
  one before, and stitched into one video. That goes up to 10 minutes; see LONG-MEDIA-PLAN.md for what to expect from
  the quality.
- **`"image"`** starts a clip from a picture, on a model that takes a source image.
- **The answer** is the clip in base64: MP4 by default, `"webp"` or `"gif"` on request.

### Long videos: one request, one file

A video of minutes is never made in pieces by the caller: one request makes the whole of it inside the server, as one
background job, each clip continued from the one before (the media engine's chain), stitched into one file.

- `"seconds": 120`: one scene that long.
- `"scenes": [{"prompt", "seconds"}, ...]`: a story in scenes (up to 60), each its own prompt, as many clips as its
  length takes; up to 15 minutes in all.
- The chat tool `generate_video` and the job API (`POST /api/jobs {"tool": "generate_video", "arguments": {...}}`) take
  the same; the model is told to make a long video in one call, never several.
- **Following it:** `GET /api/jobs/{id}` (and the chat's job card) gives the clip now (`stage` of `stages`), the time so
  far, and `log`, a line for each clip finished with its scene; the web app's card appends them as they clear.
- **An API chat stays engaged:** a streaming chat (Chat Completions, Messages, Responses) from an API client that
  queued media keeps its stream open after the answer, adding a line as each clip clears and, at the end,
  `✓ video mj-N is ready: /v1/media/jobs/mj-N/content`. `"strata_media_wait": false` ends the stream with the answer.
- Short asks stay one shot: one image, one clip, one sound. Long audio is one job too: `speak` reads up to ~20 minutes,
  `produce` joins speech, songs and music into one file with chapters.

## Speech

| Endpoint | Body | Answer |
|---|---|---|
| `POST /v1/audio/transcriptions` | multipart `file` (as OpenAI's clients send it), or JSON `"file"` as base64 / a data URL; `language` (ISO 639-1) skips detection; `response_format`: json, text, verbose_json, srt, vtt | The text |
| `POST /v1/audio/translations` | The same | English text |
| `POST /v1/audio/speech` | `{"input", "voice"?, "model"?, "language"?: "auto", "response_format"?: "wav"}` | audio/wav |

OpenAI's model names (`whisper-1`, `tts-1`, `tts-1-hd`) mean the models chosen in Settings › Models. Native voices
cover 8 languages. Indic Parler speaks Malayalam and 11 more Indian languages, and Chatterbox clones a voice.
`POST /api/media/audio/device {"device": "auto" | "cpu" | "cuda"}` sets where speech runs.

## Songs and music

`POST /v1/audio/music` makes a sung song (ACE-Step 1.5, in Rust):

```json
{"style": "warm acoustic ballad, female vocal", "lyrics": "[verse]\n…\n[chorus]\n…", "language": "en",
 "duration": 120, "seed": 3, "bpm": 72, "key": "D major", "time_signature": "4/4", "planner": true}
```

The answer is audio/wav, 48 kHz stereo; `duration` is 10 to 600 seconds. With `planner` on (the default), the song's
structure is planned first, which adds a missing tempo, key and time signature. In a chat, `make_song` makes a sung
song and `compose_music` an instrumental.

## The media library

Every image, video and sound the server makes, and every one it is sent to edit or transcribe, is kept once, by its
content, in `<data dir>/media-files/`. Each has an id, `mf_` and 16 hex digits of its SHA-256, which works in place of
the file in any media request:

```bash
# an image made earlier, edited without sending it again
curl -s $URL/v1/images/edits -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen-image-2.1:q8", "prompt": "at night", "images": ["mf_3f2a9c0d1e4b5a67"]}'
```

- Generation answers carry the id: `data[i].file_id` for images and video, the `x-strata-file-id` header for speech
  and songs.
- `GET /v1/media/files?kind=image|video|audio&source=generated|uploaded&limit=&offset=` lists them, newest first, with
  what is known of each (model, prompt, seed, name, size, when).
- `POST /v1/media/files` adds one: JSON `{"data": base64 or a data: URL, "name"}`, or multipart `file`.
- `GET /v1/media/files/{id}` (its entry), `GET /v1/media/files/{id}/content` (the file), `DELETE /v1/media/files/{id}`.
- **Who sees what:** the main key sees every file; a client key lists and reads the files it sent or had made. An id
  is the file's content hash, so a request holding one may use the file.
- **Room:** `"media_files_max_gib"` in the run config (default 10); past it the least recently used files go.
- **Age:** a file is kept `"media_files_max_age_days"` (default 90) from when it was first made or sent, then it goes,
  unless it is **pinned**: `PATCH /v1/media/files/{id}` `{"pinned": true}`, the Pin button, or `media pin <id>`. A
  pinned file stays until unpinned or deleted, and is never the one dropped for room.
- **In the app:** Media › Library, with Edit this, Pin, Copy id, Download and Delete. **CLI:** `am2strata-run media files`,
  `media add <file>`, `media get <id> --out f`, `media rm <id>`, `media pin <id>`, `media unpin <id>`.

## Image search and reverse image search

Private to this machine: the searches run from here, in a throwaway browser session, and the results come back to you
only. **Nothing is filtered for what an image shows** (Google is asked for `safe=off`). It is not for identifying a
person from their face: no face-search engine is used.

- `POST /v1/search/images` `{"query", "n"}`: images for some words, each with its image, its page, where it is from
  and its licence when known. SearXNG's image search when the run config has one, else Wikimedia Commons.
- `POST /v1/search/reverse` `{"image": a library id, a data: URL, base64 or a URL, "engines"?: ["yandex", "google_lens"], "n"}`:
  where the image, or one like it, appears. **The image is sent to the engines named**; that is all that leaves this
  machine. Yandex answers with what the image seems to show and the sites it is on, read in the server's own headless
  browser. Google Lens asks a browser like that to prove it is a person; that is never solved here, so its answer is
  `needs_you` with its results link, to open in your own browser. The image is kept in the media library.
  The answer is `{"object": "reverse_search", "file_id", "engines": [{"engine", "status": "ok" | "needs_you" | "failed",
  "contains": [what the image seems to show], "sites": [{"title", "url", …}], "results_url", "similar_url"}]}`: the
  results are per engine, under `engines[].sites`, not in one list.
- **Chat tools** (with the web on, `"strata_web": true`): `image_search` and `reverse_image_search` (an `mf_` id or a URL).
- **In the app:** Media › Tools › Image search, and **Find on the web** on each library image.
  **CLI:** `am2strata-run media search -- "words"`, `am2strata-run media reverse -- <file | mf_id | url>`.

## Productions: one long piece from parts

`POST /v1/media/productions` queues one job for a whole piece:

```json
{"name": "Evening set", "crossfade_s": 2, "parts": [
  {"speak": {"text": "Welcome to the evening set.", "voice": "laura"}},
  {"song": {"lyrics": "…", "genre": "lofi", "duration": 180}},
  {"song": {"lyrics": "…", "continue": true, "duration": 120}},
  {"music": {"genre": "ambient", "duration": 60}}]}
```

- **How it's made:** parts are made in order and kept part by part, so a restart resumes rather than starting over.
  They are joined at 48 kHz stereo with even loudness and crossfades.
- **Continuing a song:** a song with `"continue": true` carries on from the song before it.
- **Following the job:** `GET /api/jobs/{id}` (or `/v1/media/jobs/{id}`) follows it part by part.
  `/v1/media/jobs/{id}/content` is the WAV, and the job's meta holds the chapters.
- **Not yet:** video parts, MP3 output, and chapters inside the file (GAPS-PLAN A3).
- **In the web app:** the Media Generation › Tools tab has a Productions card — paste a plan and it makes the parts part by part, showing the progress and then playing one saved file.

## Composing a video, and thumbnails

These need ffmpeg on the server.

- **`POST /v1/media/videos/compose`:**
  `{"audio", "shots": [{"image"} | {"clip"}, …], "captions"?: [[start_s, end_s, text] | [start_s, end_s, text, {"place", "size"}]], "title_card"?, "end_card"?, "engine"?: "native"|"libass", "size"?: "1920x1080", "crossfade_s"?: 0.8}`.
  Images are slowly zoomed, clips are fitted, joins crossfade, captions are burnt in (by am2Strata with ffmpeg's
  `overlay`, or by libass on `"captions_engine": "libass"`). The result is one MP4 (H.264 and AAC).
- **`POST /v1/media/thumbnails`:** `{"image", "lines": [[text, "white" | "gold"]], "tag"}`. The result is one
  1280×720 JPEG with big type over a dark gradient.

Both are queued jobs, and their files are at `/v1/media/jobs/{id}/content`.

### The caption's place and size

A caption is normally `[start_s, end_s, text]` — the verse line, drawn centred low and 120 px above the bottom (and it
is the only kind kept in `captions.srt`, so YouTube shows the verse). It can instead carry a fourth element giving where
the word sits and how large the type is, so the mantra's half-verse can show low while the word being sung shows large
in the centre at the same time:

```json
{"audio": "…", "shots": [{"image": "…"}], "captions": [
  ["0.0", "8.0", "नमस्ते अस्तु भगवन्"],                    // low, today's look (size 1.0)
  ["0.0", "8.0", "नमस्ते", {"place": "centre", "size": 2.5}] // the sung word, large, centred, shown with the line above
]}
```

- `"place"` is `"low"` (the default) or `"centre"` (centred vertically) or `"top"`. A low caption and a centre word in
  the same interval overlap in time and both are drawn.
- `"size"` scales the type (`1.0` = today's 58 px at a 1080 short side; `0.5`–`3.0` are allowed). A size `2.0` box is
  about twice as tall as a size `1.0` one.
- The default engine draws them with am2Strata and ffmpeg's `overlay`; on `"captions_engine": "libass"` the centre word
  rides a second ASS style (centred, sized by its `size`). Only the low captions go to `captions.srt`.

## Dubbing

`POST /v1/media/dub` (the chat tool `dub_narration`) lays a narration in another language at the original beat times,
so the picture does not move:

```json
{"language": "de",
 "lines": [{"speaker": "laura", "text": "Hallo …", "t0": 22.2, "dur": 4.1}, …],
 "takes": ["data:audio/wav;base64,…", …],
 "stems": "data:audio/wav;base64,…",
 "heard": ["the transcript of take 1", …],
 "source_lines": [{"speaker": "laura", "text": "Hello …"}, …]}
```

- **Takes and stems:** they may come at any sample rate (a 24 kHz take from `/v1/audio/speech` is fine).
- **A take longer than its window** (its slot plus 0.2 s) is sped up, keeping the pitch, by up to 2×. A take that would
  need more is cut, with a fade.
- **`source_lines`:** when given, the translation is checked against it: a line for every line, with the same speakers.
- **The answer:** the full-length soundtrack as a WAV data URL, plus a report entry for every line that was sped up,
  cut or kept silent. Each entry gives the real line number, and `words_heard` when a transcript was given.
- **In the web app:** the Tools tab has a Dub card — paste a request (or pick takes and stems), it dubs the narration in place and plays or saves the whole soundtrack.

## Checking a take or a video

`POST /v1/media/check` (the chat tool `check_media`):

```json
{"audio": "data:audio/wav;base64,…", "expected_text": "Hallo …", "language": "de",
 "heard_text": "optional: a transcript you already have", "video": "optional: a video to scan",
 "max_s": 0.25, "words_ok": 0.6}
```

| Check | What it measures |
|---|---|
| `longest_pause` | The longest pause inside the take |
| `clipped` | The share of samples at full scale |
| `ending` | Whether the end is quiet (a take cut off mid-word ends loud) |
| `words_heard` | The share of the expected words that were heard |
| `jumps` (video) | Single-frame jumps (a broken crossfade) |
| `near_static` (video) | Frozen stretches |
| `length` (video) | The clip's length |

- **Each check** answers with its number, ok or not, and the threshold it used.
- **Without `heard_text`,** the take is transcribed in its own language with the most accurate Whisper on the server.
- **Video checks need ffmpeg** (GAPS-PLAN A14).
- **In the web app:** the Tools tab has a Check card — drop a take or a video, give the expected words if you have them, and it shows each check's result as pass or fail.

## Analyzing the music of a take

`POST /v1/media/analyze`: the facts a producer needs to learn before making a matching take — tempo, the drone's key and
home note (the tonic), the notes the voice sits on (the swaras) with how long it spends on each, and every phrase with its
timing and notes. It is pure math over the audio (a CPU job in the media engine, no model, no FFT), so it answers instantly
for any size file it reads — MP3, WAV, FLAC, AAC, OGG — at most 200 MiB:

```json
{"audio": "data:audio/mp3;base64,…", "from_s": 0, "seconds": 0}
```

`from_s`/`seconds` select a window (both 0 mean the whole take); the CLI is `am2strata-run media analyze <file> [--from S] [--seconds N]`.
The endpoint forwards the raw bytes to the engine's `analyze` op and returns its analysis (like `/v1/audio/transcriptions`
forwards `transcribe`). The answer carries `tempo` (bpm, confidence, whether it is a steady pulse), `repeat` (one verse's
period and how many times it repeats), `pitch.tonic` (note + hertz) with the `range`, each `swara` as `{name, share}`, and a
`phrases` list. Send the file as a multipart `file` or as base64 / a data URL in JSON.

## Media from a chat

With `"strata_media": true`, the model gets these tools:

| Tool | What it does |
|---|---|
| `generate_image`, `generate_video` | Pictures and clips |
| `speak` | Speech |
| `make_song`, `compose_music` | A sung song, an instrumental |
| `produce` | A production |
| `compose_video`, `make_thumbnail` | A finished video, a thumbnail |
| `dub_narration`, `check_media` | Dubbing, checks |

- **A call doesn't hold up the answer:** it queues a job and returns the job's id. The job runs after the answer, so
  the text model isn't stopped mid-reply.
- **Follow a job** with `GET /v1/media/jobs/{id}`, and get the file at `/content`.
- **Jobs are kept on disk** and survive a restart. `POST /api/jobs {"tool", "arguments"}` makes one again.
- **`GET /api/media/capabilities`** says what this PC can make and how long it takes.

## Video without ffmpeg

MP4 files with H.264 video, which is what phones and most tools make, are decoded by the media engine itself (`openh264` and `mp4`, already part of its build). The media check's video checks and videos sent in a chat both work on a PC without ffmpeg. Other formats (WebM, MOV with other codecs, AVI) still need ffmpeg, and the error names the format.

**B-frames:** H.264 with B-frames, x264's default and common in files made with ffmpeg, can't be read by the built-in decoder (OpenH264). For those:
- a chat gets the video's keyframes only, and says so;
- the media check, which needs every frame, says plainly to install ffmpeg or re-encode without B-frames (`ffmpeg -bf 0`).

Videos made without B-frames (OpenH264, many phone and screen recorders) decode in full. With ffmpeg installed, it is used as before for every format.
