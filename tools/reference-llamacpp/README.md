# llama.cpp as the reference for am2strata-hybrid

`am2strata-hybrid` (crates/strata-hybrid) is checked against llama.cpp, used here only as a test oracle - it is never
built into or run by am2Strata. The oracle is a copy of PolyStrata's bundled llama.cpp (PolyStrata itself is not
changed), built for the CPU, with one patch.

## Build the oracle

```bash
D=${STRATA_DATA:-$HOME/Development/Strata-data}/oracle
rsync -a --exclude build --exclude .git ~/Development/PolyStrata/third_party/llama.cpp/ $D/llama.cpp/
patch -d $D/llama.cpp -p1 < tools/reference-llamacpp/per-layer-experts.patch
cmake -S $D/llama.cpp -B $D/build2 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=ON \
      -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF
cmake --build $D/build2 -j --target llama
g++ -O2 -std=c++17 tools/reference-llamacpp/oracle.cpp -I$D/llama.cpp/include -I$D/llama.cpp/ggml/include \
    -L$D/build2/bin -lllama -lggml -lggml-base -Wl,-rpath,$D/build2/bin -o $D/oracle2
```

`per-layer-experts.patch`: per-layer pruned Nemotron-H GGUFs (NVIDIA Nemotron 3.5 Bolt 74B-A9B: 240 experts in some
layers, 496 in the header) keep each layer's expert count only in the tensor shapes; the patch reads it from there.

## Run

- `oracle <model.gguf> <n> <id,id,...>` or `oracle <model.gguf> <n> text:<prompt>` - greedy decoding, the top-5
  logits per step as JSON lines; `-1` for `<n>` loads only the vocabulary and prints the prompt's ids.
- `parity.sh <model.gguf> <n> "<prompt>"` - the oracle and the engine on the same ids. The engine is fed the oracle's
  tokens (teacher forcing) and every step is compared; a step passes when both pick the same token, or on a near-tie
  (llama.cpp's own top two within `TIE`, default 0.25, and the engine's choice one of them). `ENGINE_ARGS` go to the
  engine (e.g. `--pack DIR --resident-experts`).
- `cargo test --release --manifest-path crates/strata-media/Cargo.toml -p strata-hybrid -- --ignored` with
  `STRATA_MODELS=a.gguf:b.gguf`: the tokenizer test (am2Strata's tokenizer, extracted from each GGUF, against
  llama.cpp's ids on a set of texts) and the forward test (parity.sh per model).

Why not bit-identical logits: llama.cpp's CPU backend quantizes the activations before each matrix product (Q8_K,
Q8_0, F16 or BF16, by the weight's type), keeps the KV cache in F16 and accumulates flash attention in F16. The engine
does the first two the same way; what remains is float rounding, about 0.1-1 logit (GLM-4.7-Flash's logits are
~100, its differences a near-uniform shift that softmax ignores).

## Results (2026-10-21, CPU, 16 greedy steps unless noted)

| Model | Architecture | Identical | Near-ties | Worst logit difference |
|---|---|---|---|---|
| NVIDIA Nemotron 3.5 Bolt 74B-A9B AD-Q5_K_M | nemotron_h_moe | 15 | 1 | 0.61 |
| the same, with its pack and 24 GiB of resident experts (12 steps) | | 11 | 1 | 0.57 |
| GLM-4.7-Flash UD-Q4_K_XL | deepseek2 (MLA) | 15 | 1 | 1.13 |
| the same, with its pack (12 steps) | | 12 | 0 | 1.13 |
| Devstral Small 2507 Q4_K_M (Mistral Small 3.1) | llama | 16 | 0 | 0.33 |
| Ministral 3 3B Q4_K_M (24 steps) | mistral3 (YaRN) | 23 | 1 | 0.15 |
| Qwen3-Coder 30B-A3B Q4_K_M (Mixtral's softmax MoE) | qwen3moe | 15 | 1 | 0.73 |
| Qwen3 8B Q8_0 (12 steps) | qwen3 | 12 | 0 | 0.31 |

Tokenizers (7 texts each, identical ids): tekken (Ministral 3, Devstral), glm4 (GLM-4.7-Flash), pixtral
(Nemotron 3.5 Bolt), qwen2 (Qwen3 8B, Qwen3-Coder).
