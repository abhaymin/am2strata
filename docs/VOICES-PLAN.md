# Native voices for every language, female and male

Owner's request (2026-11-09): speak each language in a voice native to it, not only the European English ones, and
give each a female and a male voice.

## Why

Chatterbox speaks in the voice of a reference clip, and it carries the clip's accent into other languages (its makers
say so, and suggest a clip in the target language). Today the named voices (Laura, Jon, Lea, Gary) are English clips
made with Parler, so Hindi or Spanish come out with an English accent.

## Rule: synthetic voices only

Every reference clip is made by a text-to-speech model from a description or a built-in synthetic voice, under a
licence that allows it. No recording of a real person is cloned, so nobody's voice is copied without consent.

## Phase 0: findings (2026-11-09)

| Source | Licence | Languages, with female (F) and male (M) | Notes |
| --- | --- | --- | --- |
| **Kokoro-82M** (hexgrad) | Apache-2.0 (weights) | Spanish 1F 2M, Hindi 2F 2M, Italian 1F 1M, Portuguese (Brazil) 1F 2M, French 1F only; also Japanese 4F 1M and Mandarin 4F 4M, for later | Graded C to B- by its author; ONNX exports exist |
| **Parler-TTS Multilingual** | to confirm | German, French, Spanish, Portuguese, Polish, Italian, Dutch, from a description ("a male speaker…") | The same architecture as the Parler mini already in the engine |
| **Indic Parler-TTS** (AI4Bharat) | to confirm | Hindi and about 20 Indian languages, from a description, with named speakers | Same architecture |
| Piper voices | per voice: CC0, CC BY-NC-SA, "unknown"… | many languages; no Malay; gender not stated | Licences too mixed to ship; not used |
| Mozilla Common Voice | CC0 | most languages | Real volunteers' voices: not used, by the rule above |

So **female and male native voices are within reach for:** Spanish, Hindi, Italian, Portuguese, German, French, Polish
and Dutch (once the Parler licences are confirmed).

**No synthetic native source found yet for:** Arabic, Danish, Greek, Finnish, Korean, Malay, Norwegian, Russian,
Swedish, Swahili and Turkish. For those, until a source is found:
- the voice is made by Chatterbox Multilingual itself, speaking that language from a neutral clip, and the best take
  (Whisper word check, the take's pitch in the female or male range) becomes the reference. This lowers the accent
  without any person's recording; how much it helps is measured in phase 1;
- if its ONNX graph exposes classifier-free guidance, a weight of 0 for these languages (its makers' advice against
  accent carry-over).

## Built (2026-11-09), all in Rust

- `crates/strata-media/src/parler_hf.rs`: the newer Parler checkpoints (mini multilingual v1.1, Indic) load into
  candle's Parler model unchanged. Their names are remapped as they load (fused codebook heads split; transformers'
  `DacModel` weights handed out as descript's weight-normed ones). All 301 codec tensors map; none are left over.
- Catalog: `parler-tts-mini-multilingual` (downloaded here, 3.75 GB) and `indic-parler-tts` (gated: its terms must be
  accepted on Hugging Face by the account whose token this PC uses).
- `voice` takes `female` / `male`; Laura and Lea are female, Jon and Gary male. In another language than English the
  engine clones `voices/<lang>-<gender>.wav`, made once by the multilingual Parler model from a description.
- Measured (Chatterbox Multilingual speaking, clips by Parler multilingual on the CPU; Whisper turbo listening in that
  language):

  | Language, voice | Clip | Speech | Whisper heard |
  | --- | --- | --- | --- |
  | Spanish, female | 9.4 s, 224 Hz | 209 Hz | every word, then about 3 s of babble |
  | Spanish, male | 12.5 s, 97 Hz | 109 Hz | every word |
  | German, female | made, 218 Hz speech | 218 Hz | every word, after a few garbled words |

  The voices are native and of the right gender. The babble is Chatterbox's own: some takes add words at the start or
  end (the narration needed several seeds per line for the same reason).
- Next: a retake in the engine when Whisper hears words that are not in the text (a few seeds, the best kept).

## Phases

1. **Clips:** for each language, a female and a male reference clip of 8 to 12 s:
   - made by the sources above;
   - checked by Whisper (the words, in that language) and by pitch (female about 165-255 Hz, male about 85-155 Hz);
   - kept in the media folder as `voices/<lang>-female.wav` and `voices/<lang>-male.wav`, with a JSON file naming the
     source, its licence and its settings for each clip.
2. **Server:**
   - `voice` takes `female` or `male` (plus the named voices);
   - with a non-English language, a named voice maps to the clip of its gender in that language (Laura → female,
     Jon → male);
   - `GET /api/media/voices` lists what exists per language.
3. **Web app:** the Voice picker shows Female and Male, with the language's names; Read aloud uses the gender chosen
   in Settings.
4. **Tests:** mock-engine tests for the mapping; an ignored real-model test per language: Whisper hears the words in
   that language, and the pitch is in range for the gender.

Downloads (asked before): Kokoro-82M (about 0.3 GB), Parler-TTS Multilingual and Indic Parler-TTS (about 3.5 GB each).

Sources: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M),
[Kokoro voices](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md),
[Piper voices](https://huggingface.co/rhasspy/piper-voices),
[Chatterbox Multilingual, accent advice](https://pypi.org/project/chatterbox-tts/0.1.3/).
