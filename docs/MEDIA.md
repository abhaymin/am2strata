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
  `{"audio", "shots": [{"image"} | {"clip"}, …], "captions"?: [[start_s, end_s, text]], "title_card"?, "end_card"?, "size"?: "1920x1080", "crossfade_s"?: 0.8}`.
  Images are slowly zoomed, clips are fitted, joins crossfade, captions are burnt in. The result is one MP4 (H.264 and
  AAC).
- **`POST /v1/media/thumbnails`:** `{"image", "lines": [[text, "white" | "gold"]], "tag"}`. The result is one
  1280×720 JPEG with big type over a dark gradient.

Both are queued jobs, and their files are at `/v1/media/jobs/{id}/content`.

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

MP4 files with H.264 video, which is what phones and most tools make, are decoded by the media engine itself (`openh264` and `mp4`, already part of its build). The media check's video checks and videos sent in a chat both work on a PC without ffmpeg. Other formats (WebM, MOV with other codecs, AVI) still need ffmpeg, and the error names the format. With ffmpeg installed, it is used as before for every format.
