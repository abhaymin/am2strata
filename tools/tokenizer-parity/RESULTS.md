# Tokenizer parity with llama.cpp (2026-11-17)

`parity.sh <model.gguf> [special]` over corpus.txt (39 cases: code, whitespace runs, CRLF, numbers, Latin with
diacritics, Greek/Cyrillic, CJK, Arabic/Hebrew, Indic, Thai, emoji ZWJ sequences, combining marks, zero-width
characters, chat-template control tokens). llama-tokenize from llama.cpp's build-sm120, vocab only, no BOS, --no-escape.

| Model (pre-tokenizer) | plain | special |
|---|---|---|
| Tiel (qwen35) | 39/39 | 39/39 |
| GLM-4.7-Flash (glm4) | 39/39 | 39/39 |
| Orca, IQ3_XXS (qwen35) | 39/39 | 39/39 |
| LFM2-8B | 39/39 | 39/39 |
| MiniCPM-V | 39/39 | 39/39 |
| Mixtral 8x7B (spm) | 39/39 | 39/39 |
| Nemotron 3.5 Bolt | 39/39 | 39/39 |
| Strata Sage | 39/39 | 39/39 |
| Ministral-3 3B (tekken) | 38/39 | 38/39 |
| Devstral Small (tekken) | 38/39 | 38/39 |

The one difference (both tekken models) is Devanagari/Tamil/Bengali text: our pattern keeps combining marks (`\p{M}`)
with the letters before them; llama.cpp's own regex engine splits them off, giving more and shorter tokens (24 against
18 here). Which one Mistral's own tokenizer (mistral-common) gives is not checked yet; left as is until it is.

Speed: 1.6-2.7 MB/s for the byte-level BPE vocabularies, 8.5-10 MB/s for SPM (Mixtral), single thread.
