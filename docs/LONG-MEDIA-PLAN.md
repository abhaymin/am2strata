# Long requests: 5 minutes and more of speech, songs and video

Owner's requirement (2026-11-09): am2Strata speaks for at least 5 minutes, makes songs that long, and makes video
that long, from one request.

## Where each stands

| Kind | Limit before | Now | How it gets long |
| --- | --- | --- | --- |
| Speech | 4,096 characters (OpenAI's limit, about 4.5 min) | **20,000 characters (about 20 min)** | already made part by part (sentences), each part checked by Whisper and retaken when it adds or drops words |
| Songs | not in am2Strata yet (ACE-Step in Rust: docs/ACESTEP-RUST-PLAN.md) | planned: **up to 10 min** | ACE-Step 1.5 makes 10 s to 10 min in one pass (8 min with its planner on a 16 GB card); measured 60 s in 12 s on one RTX 5060 Ti |
| Video | 257 frames (about 10 to 16 s, model-dependent) | planned: **5 min and more** | segments chained: each next segment starts from the last frame of the one before (image-to-video), then joined |

## Productions: am2Strata builds the whole piece itself (owner's requirement, 2026-11-10)

**State (2026-11-11): audio productions built.** `POST /v1/media/productions` and the chat tool `produce` take a
plan of speak, song and music parts.
- **Making the parts:** each part is made by its kind's own endpoint, in order (songs with 2 takes by default), and
  kept on disk as it is finished, so a restart goes on from the last finished part.
- **Joining** (`production.rs`): the parts are resampled to 48 kHz stereo, their loudness evened (-18 dBFS RMS,
  peak 0.98) and joined with equal-power crossfades (`crossfade_s`, default 1 s).
- **The result:** one WAV, with a chapter per part in the job's meta. The web app shows "part 3 of 9" while it is
  made, then the player and the chapters.
- **Tested:** `a_production_makes_its_parts_in_order_and_joins_them` and the join's unit test.
- **Not yet:**
  - songs continued from the one before (ACE-Step's "complete" task needs the Oobleck encoder ported): a long song
    is joined with a crossfade today;
  - video productions;
  - MP3 output;
  - chapters in the file's own metadata.

A request like "a 25-minute Diwali music set in Hindi, with a spoken welcome, and a video for it" must come back as
one finished file, made by am2Strata from start to end, with nobody stitching parts by hand. So:

- **A production is one job made of parts.** `POST /v1/media/productions` (and the chat tool `produce`) takes a plan:
  ```json
  {"output": "audio" | "video", "parts": [
     {"speak": {"text": "...", "voice": "female", "language": "hi"}},
     {"song": {"lyrics": "...", "genre": "festive", "duration": 600}},
     {"song": {"continue": true, "lyrics": "...", "duration": 600}},
     {"music": {"genre": "ambient", "duration": 120}},
     {"video": {"prompt": "...", "seconds": 600}},
     {"image": {"prompt": "...", "seconds": 30}}]}
  ```
  or, from the chat, the model writes that plan from the user's words ("3 songs of 10 minutes, then a spoken thank you").
- **Each part is made by the kind's own endpoint** (songs up to 10 minutes each, speech of any length, video segments,
  still images held as video), queued and retried as any job, with the same checks (Whisper for words, the take rule).
- **The parts are joined in the engine, in Rust:**
  - audio: resampled to one rate (48 kHz stereo), joined with short crossfades (a song continued from the one before
    uses ACE-Step's "complete" task from its last seconds, so the music flows on rather than restarting), loudness
    evened out;
  - video: segments joined (each next segment started from the frame before, as above), images shown for their
    seconds, the audio track (the joined audio, or music made for it) laid under the video, then one MP4;
  - one WAV/MP3 or one MP4 out, with chapters (a part per chapter) in its metadata and a list beside it.
- **Long runs survive:** the production's state is saved part by part (a restart goes on from the last finished part),
  the Monitor shows "part 4 of 9" with each part's own progress, and the finished file stays in the Monitor's history.
- **No length limit** but the disk: 30 minutes of 48 kHz stereo WAV is about 350 MB; MP3 and MP4 are kept as well.
- Tests: the mock engine for the plan (parts in order, a failed part retried, the join's length and crossfades, the
  restart resuming); a real 12-minute song (two 6-minute parts continued) checked for its length, a seamless join
  (no silence or click at the seam) and Whisper hearing the second part's words.

## Long jobs run in the background

A 5-minute request takes minutes to make (speech on the CPU at about real time, video far longer), longer than many
clients wait on one HTTP request. So, for every kind:
- `POST` with `"background": true` (or anything over a size: speech over 4,096 characters, songs over 60 s, any video
  over one segment) answers at once with a job id; `GET /api/jobs/{id}` gives the state and progress ("part 12 of
  48", "segment 3 of 30") and, when done, the file. The chat's media tools already use these jobs.
- Monitor shows the job's progress; the web app's media pages show it under the form, and the result when done.
- A job keeps going if the browser closes; its result stays in the Monitor's history.

## Video: segments chained

**State (2026-11-11): built, not yet measured on the GPU.**
- **How it works:** `/v1/videos/generations` with `"seconds"` or `"total_frames"` past one clip's `frames` sends the
  media engine a `chain` command. It uses mold's own chain:
  - `ChainOrchestrator` renders the clips, each continuing from the one before through a 17-frame motion tail;
  - `StitchPlan` joins them with smooth seams;
  - `encode_chain_frames` writes one MP4.
- **Where it applies:** on Wan 2.2 TI2V 5B (downloaded 2026-11-11) the clips carry the motion over. A text-to-video
  checkpoint chains as independent clips. The longest is 10 minutes at 24 fps.
- **Tested:** `a_long_video_is_made_as_a_chain_of_clips` on the mock. The GPU time per clip is to be measured.

- The model: Wan 2.2 TI2V 5B (text and image to video, 1280×704 or 704×1280 at 24 fps, about 5 s a segment) in mold.
  Wan 2.1 1.3B makes text-to-video only, so it can make the first segment but not continue one.
- Segment 1 from the prompt; segment k+1 from segment k's last frame and the prompt (or, with a storyboard, that
  segment's own prompt: `"segments": [{"prompt": ..., "seconds": ...}, ...]`).
- The seams: the shared frame is dropped once, and a short crossfade (a few frames) hides a jump in colour or motion.
- Joined into one MP4 in the engine (mold's `mp4` feature), with the frame rate kept.
- Time, to be measured before a promise is made: 5 minutes at 24 fps is about 60 segments. If one segment takes a few
  minutes on one RTX 5060 Ti, 5 minutes of video takes hours. Both cards can make segments of two different jobs at
  once, but not two segments of one chain (each waits for the frame before it).
- Tests: the mock engine for the chaining (frames passed on, segment count, the join); a real 30 s chain (6 segments)
  checked for its length, its seams (frame difference at each seam against inside a segment) and its time.

## Songs: one pass, up to 10 minutes

- `duration` 10 to 600 s (480 s with the planner on a 16 GB card, as ACE-Step's own tiers say); longer than that,
  ACE-Step's "complete" task continues a song from its end (stage after the Rust port).
- The lyric check (Whisper hears the vocals, retakes for missing words) runs per minute of song.
- The Songs tab under Media Generation takes the duration in minutes and seconds.
