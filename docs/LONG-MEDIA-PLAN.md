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

**State (2026-11-11): built and measured on the GPU. It runs, but the picture drifts from clip to clip, so it is
not yet good enough to promise.**
- **How it works:** `/v1/videos/generations` with `"seconds"` or `"total_frames"` past one clip's `frames` sends the
  media engine a `chain` command. It uses mold's own chain:
  - `ChainOrchestrator` renders the clips, each continuing from the one before (Wan: from the last frame alone,
    mold's `WAN_HANDOFF_DUPLICATED_FRAMES`; LTX: a 17-frame motion tail);
  - `StitchPlan` joins them with smooth seams;
  - `encode_chain_frames` writes one MP4.
- **Where it applies:** on Wan 2.2 TI2V 5B (downloaded 2026-11-11) the clips carry the motion over. A text-to-video
  checkpoint chains as independent clips. The longest is 10 minutes at 24 fps.
- **Tested:** `a_long_video_is_made_as_a_chain_of_clips` on the mock, and `a_video_is_refused_when_the_pc_lacks_its_memory`.

### Measured (2026-11-11, Wan 2.2 TI2V 5B Q8, one RTX 5060 Ti 16 GB, through the live API)

| Run | Result |
|---|---|
| One clip, 9 frames, 4 steps, 640×352 | an MP4 in 11 s |
| Two-clip chain, 17 frames, 2 steps, 640×352 | an MP4 in 55 s (about 12 s a clip; UMT5 on the GPU) |
| **30 s chain**, 640×352, 20 steps, 24 fps, seed 7 | **721 frames, 6 clips, 1,654 s (27.6 min)**: 264-268 s a clip of 121 frames, the same for each; UMT5 FP16 on the GPU every clip (15.4-16.0 GB free) |

The seams and the picture of the 30 s chain (frame-to-frame difference on a 160×88 grey copy; a lighthouse with waves,
so the median is high, 10.3):
- **Each seam is a visible jump:** 17-30 at the seam against 1-5 in the frames before it. At the first seam the
  framing jumps closer and the brightness goes from 71 to 91.
- **The picture drifts:** each clip starts from the last, already degraded, frame, so errors add up. By the fourth
  clip the sky is smeared, the colours are oversaturated and the rocks and waves break into painterly blocks.
- 640×352 is below the model's own 1280×704, which adds to the artefacts.

The first real try (10 s, 01:50) failed after 36 min with "device mismatch in add, lhs: Cpu, rhs: Cuda". mold had put
UMT5 on the CPU (F32, about 22.7 GB), the service reached 34.9 GB, and the kernel's OOM killer ran. Fixed since: a
video job needs 24 GB free or is refused plainly; mold's own messages (UMT5 on GPU or CPU, stage times) are in the
service log; the Wan carry is one frame, not LTX's 17.

### Colour hold and crossfaded joins (2026-11-11, built and measured)

`crates/strata-media/src/chain_hold.rs` wraps mold's Wan stage renderer:
- **The colour hold:** each later clip's histogram is matched to clip 1's (one lookup table a clip) before its frame
  starts the next clip. It is on when every clip has the same prompt (`"hold_colour"` turns it off or on).
- **The overlap:** the next clip starts `overlap_frames` (default 8) before the end of the one before, so both clips
  draw those frames, and the stitch crossfades them instead of cutting. mold is asked for enough more frames that the
  video keeps its length.
- **Monitor:** Wan's stage renderer does not pass on its denoise steps, so the chain takes them from the engine's own
  events, and Monitor shows "step N of M" for a chain.

The same 30 s request (Wan 2.2 5B Q8, 640×352, 20 steps, 24 fps, seed 7; frame difference on the full grey frame), before
and after:

| | Before | Colour hold + 8-frame crossfades |
|---|---|---|
| Time | 1,654 s, 6 clips, 721 frames | 1,717 s, 7 clips, 720 frames |
| The joins (largest difference around each, against just before) | 30, 25, 38, 47, 54 (11-20 before) | 15, 16, 13, 17, 21, 22 (8-11 before) |
| Colour drift from clip 1 (mean of R, G, B), clips 2 to 6 | 6.0, 8.0, 10.5, 13.3, 15.3 | 1.7, 1.6, 1.4, 1.5, 2.0 (5.6 in the short last clip) |
| Saturation change by clip 6 | -26.6 | -2.6 |

**What the numbers do not show:** the colour now holds and the jumps are halved, but **the detail still degrades from
clip to clip.** From clip 3 the sky breaks into horizontal blocks and the rocks are posterised, and the last clip is
broken. A crossfade between two clips that moved apart also shows both at once (a see-through lighthouse) for its 8
frames. The carried frame passes the artefacts on, and a histogram cannot remove them. **Long video is still not
offered as good.**

A third run (09:13, the same request) added two things:
- **A whole last clip:** the last clip is now made at full length and the video is cut to length afterwards. The last
  clip still drifts (9.0 from clip 1, the others 0.5-1.9), so the break was not only its short length.
- **A smoothed start frame** (half size and back): the joins came out at 17-21, the drift was the same, and the sky
  still broke up from clip 4. **It did not help**, so it is off by default (`"smooth_start": true` to try it).
- Time: 1,926 s for 7 whole clips (275 s a clip).

**A text-only model is no longer used for a chain:** Wan 2.1 1.3B can't continue from a frame, so it made the same
clip each time. A long video moves to a downloaded model that can continue when the request named none, and is
refused plainly when it named the text-only model.

### Next, before long video is offered

1. **A cleaner start frame by the model itself:** redraw the carried frame at low strength (image to image) before
   it starts the next clip. Shrinking it and blowing it back up was tried and did not help.
2. **The model's own size, 1280×704: measured (2026-11-11).**
   - One 121-frame clip (5 s), 20 steps, seed 7: **2,612 s (43.5 min)**, about 10× the 640×352 time. About 105 s
     a denoising step, then 7 min of tiled VAE decode.
   - **The picture is much cleaner:** a sharp lighthouse, natural water, no blocks.
   - The first try failed at the end with "device mismatch in add, lhs: Cpu, rhs: Cuda". The full VAE decode needs
     more than 16 GB at this size, and mold's tiled retry (`WanVideoVae::decode_tiled`) adds GPU tiles to its CPU
     totals. It's fixed by moving each tile and its mask to the CPU (two lines).
   - **That fix is not in the repository yet:** it was built here against a local copy of mold. Ship it by vendoring
     mold, a fork, or upstream: the owner's choice.
   - A 30 s video at this size is about 6-7 clips, so about 5 hours. That is for a finished piece rendered overnight,
     not for a quick request.
3. **An anchor:** clips k > 1 conditioned on the first clip's opening frame as well as the last frame, if mold's Wan
   inpaint takes two anchors (`WanImageAnchors`); otherwise a storyboard of independent clips with crossfades.
4. **The last clip:** find out why it drifts most even at full length.
5. Measure again on the same prompt and seed (`docs/` keeps these numbers).

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
