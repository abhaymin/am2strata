# tiny-llama.f32.gguf reference

`convert_hf_to_gguf.py`'s output for the tiny Llama folder committed in `tests/fixtures/tiny-llama-hf/`. The test
runs our converter on that same folder and, when this file is present, compares tensor names/shapes/types, every
tensor's bytes (this includes the q/k rotary permute), and the full metadata. Until it exists the test skips.

Produce and commit it (run from the repository root; this `convert_hf_to_gguf.py` needs its `conversion` module
and `gguf-py`, which live alongside it in this llama.cpp source):

    python .agent/ref/llama.cpp/convert_hf_to_gguf.py \
        tests/fixtures/tiny-llama-hf --outtype f32 --outfile tests/fixtures/tiny-llama.f32.gguf

(The tiny HF folder is written deterministically by the `save_tiny_checkpoint` test: `TinyLlama::default()`,
seed 0x5EED — vocab 32, hidden 16, 2 layers, tie_word_embeddings=false.)
