# Long requests: 5 minutes and more of speech, songs and video

Owner's requirement (2026-11-09): am2Strata speaks for at least 5 minutes, makes songs that long, and makes video
that long, from one request.

## Where each stands

| Kind | Limit before | Now | How it gets long |
| --- | --- | --- | --- |
| Speech | 4,096 characters (OpenAI's limit, about 4.5 min) | **20,000 characters (about 20 min)** | already made part by part (sentences), each part checked by Whisper and retaken when it adds or drops words |
| Songs | not in am2Strata yet (ACE-Step in Rust: docs/ACESTEP-RUST-PLAN.md) | planned: **up to 10 min** | ACE-Step 1.5 makes 10 s to 10 min in one pass (8 min with its planner on a 16 GB card); measured 60 s in 12 s on one RTX 5060 Ti |
| Video | 257 frames (about 10 to 16 s, model-dependent) | planned: **5 min and more** | segments chained: each next segment starts from the last frame of the one before (image-to-video), then joined |

## Long jobs run in the background

A 5-minute request takes minutes to make (speech on the CPU at about real time, video far longer), longer than many
clients wait on one HTTP request. So, for every kind:
- `POST` with `"background": true` (or anything over a size: speech over 4,096 characters, songs over 60 s, any video
  over one segment) answers at once with a job id; `GET /api/jobs/{id}` gives the state and progress ("part 12 of
  48", "segment 3 of 30") and, when done, the file. The chat's media tools already use these jobs.
- Monitor shows the job's progress; the web app's media pages show it under the form, and the result when done.
- A job keeps going if the browser closes; its result stays in the Monitor's history.

## Video: segments chained

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
