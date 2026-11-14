# Workshop test fixtures

`peft-expert-delta.safetensors` was made once by the reviewer with PyTorch, using PEFT 0.20.0's exact
`ParamWrapper.get_delta_weight` einsums (peft/tuners/lora/layer.py), seed 7, E=3, r=2, A [6, 5], B [4, 6], scaling 0.75:
- `delta_eio` = einsum("o r e, e r i -> e i o") — PEFT's default, for a 3-D parameter of shape [E, in, out] ([3, 5, 4]);
- `delta_eoi` = einsum("o r e, e r i -> e o i") — when PEFT swapped in/out, a parameter of shape [E, out, in] ([3, 4, 5]).
The tests only read it; nothing in `cargo test` runs Python.
