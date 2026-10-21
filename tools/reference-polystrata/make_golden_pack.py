"""Record what PolyStrata's packing code produces, for strata-pack's tests (golden/pack/).

- quants/<TYPE>.in / .out: random blocks (valid fp16 scales) and gguf-py's dequantized float32 values
- numbers.json: f16 and BF16 rounding of chosen float32 values, Python repr and round(x, 5) of doubles
- quantizers/: mtp_pack.py's q2_0 / q4_0 / q8_0 and mtp_rt.py's q8_0 on random rows
- tiny/: a two-shard model exercising every iq_pack.py path, and the pack tools/iq_pack.py writes from it
- tiny-mtp/: a small MTP source and the GGUFs tools/mtp_pack.py writes from it (q2_0, q4_0, q8_0)

Run with PolyStrata's virtual environment:
  ~/Development/PolyStrata/.venv/bin/python tools/reference-polystrata/make_golden_pack.py
"""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np

POLY = pathlib.Path.home() / "Development/PolyStrata"
sys.path[:0] = [str(POLY / "tools")]
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as Q, quants  # noqa: E402
import mtp_pack  # noqa: E402
import mtp_rt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "golden" / "pack"
PY = sys.executable
rng = np.random.default_rng(20261021)

# where each type's fp16 scales are (byte offsets in the block)
SCALES = {"Q4_0": [0], "Q4_1": [0, 2], "Q5_0": [0], "Q5_1": [0, 2], "Q8_0": [0], "Q2_K": [80, 82], "Q3_K": [108],
          "Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208], "IQ2_XXS": [0], "IQ2_XS": [0], "IQ3_XXS": [0], "IQ3_S": [0],
          "IQ4_NL": [0], "IQ4_XS": [0]}


def blocks(type_name, n):
    be, bb = gguf.GGML_QUANT_SIZES[Q[type_name]]
    b = rng.integers(0, 256, size=(n, bb), dtype=np.uint8)
    for at in SCALES[type_name]:
        d = (rng.standard_normal(n) * 10.0 ** rng.integers(-4, 1, size=n)).astype(np.float16)
        b[:, at:at + 2] = d.view(np.uint8).reshape(n, 2)
    return b.reshape(-1)


def record_quants():
    d = OUT / "quants"
    d.mkdir(parents=True, exist_ok=True)
    for t in SCALES:
        raw = blocks(t, 24)
        vals = quants.dequantize(raw, Q[t]).astype(np.float32).reshape(-1)
        (d / f"{t}.in").write_bytes(raw.tobytes())
        (d / f"{t}.out").write_bytes(vals.astype("<f4").tobytes())


def record_numbers():
    f = np.concatenate([
        rng.standard_normal(2000).astype(np.float32) * np.float32(10.0) ** rng.integers(-9, 6, size=2000).astype(np.float32),
        np.array([0.0, -0.0, 65504.0, 65519.0, 65520.0, 1e-8, 5.96e-8, 2.98e-8, 6.1e-5, 1.0 + 2 ** -11, 3e38], dtype=np.float32),
    ]).astype(np.float32)
    with np.errstate(over="ignore"):
        h = f.astype(np.float16)
    bf = quants.quantize(f, Q.BF16).view(np.uint16)
    doubles = list(rng.standard_normal(500) * 10.0 ** rng.integers(-12, 18, size=500)) + [0.1, 1e-5, 1e16, 2.5, 1e15, 123.456]
    json.dump({
        "f32_bits": [int(x) for x in f.view(np.uint32)],
        "f16_bits": [int(x) for x in h.view(np.uint16)],
        "bf16_bits": [int(x) for x in bf],
        "doubles": [float(x).hex() for x in doubles],
        "repr": [repr(float(x)) for x in doubles],
        "round5": [repr(round(float(x), 5)) for x in doubles],
    }, open(OUT / "numbers.json", "w"), indent=0)


def record_quantizers():
    d = OUT / "quantizers"
    d.mkdir(parents=True, exist_ok=True)
    x = (rng.standard_normal((64, 640)) * 0.02).astype(np.float32)
    x[3] = 0.0                       # an all-zero row
    x[5, :64] = 0.5                  # equal values
    x = (x.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)   # BF16 values, as the checkpoint's
    (d / "x.f32").write_bytes(x.astype("<f4").tobytes())
    (d / "q2_0.bin").write_bytes(mtp_pack.q2_0(x).tobytes())
    (d / "q4_0.bin").write_bytes(mtp_pack.q4_0(x).tobytes())
    (d / "q8_0.bin").write_bytes(mtp_pack.q8_0(x).tobytes())
    (d / "rt_q8_0.bin").write_bytes(mtp_rt.q8_0(x))


def tensor(w, name, ggml_shape, type_name, data=None):
    """Add a tensor given in ggml dimension order (ne0 first)."""
    np_shape = list(reversed(ggml_shape))
    n = int(np.prod(ggml_shape))
    if type_name == "F32":
        arr = data if data is not None else rng.standard_normal(n).astype(np.float32)
        w.add_tensor(name, arr.reshape(np_shape))
    elif type_name == "F16":
        w.add_tensor(name, (rng.standard_normal(n) * 0.5).astype(np.float16).reshape(np_shape))
    elif type_name == "BF16":
        u = (rng.standard_normal(n).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        w.add_tensor(name, u.reshape(np_shape), raw_dtype=Q.BF16)
    else:
        be, bb = gguf.GGML_QUANT_SIZES[Q[type_name]]
        raw = blocks(type_name, n // be)
        byte_shape = np_shape[:-1] + [np_shape[-1] // be * bb]
        w.add_tensor(name, raw.reshape(byte_shape), raw_shape=byte_shape, raw_dtype=Q[type_name])


def record_tiny():
    d = OUT / "tiny"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    n_exp, n_embd = 8, 256
    exact_router = (rng.standard_normal(n_embd * n_exp).astype(np.float32).view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    shards = [
        [("token_embd.weight", [n_embd, 16], "Q4_0"), ("output.weight", [n_embd, 16], "Q6_K"),
         ("output_hc_down.weight", [n_embd, 4], "Q8_0"),
         ("blk.0.ffn_gate_inp.weight", [n_embd, n_exp], "F32", exact_router),
         ("blk.0.ssm_a", [32], "F32"), ("blk.0.ple_conv1d.weight", [4, 64], "F32"), ("blk.0.ssm_alpha.weight", [n_embd, 2], "Q6_K"),
         ("blk.0.attn_norm.weight", [n_embd], "F32"), ("blk.0.ffn_gate_exps.weight", [n_embd, 16, n_exp], "Q4_0"),
         ("blk.0.ffn_up_exps.weight", [n_embd, 16, n_exp], "Q4_0"), ("blk.0.ffn_down_exps.weight", [32, n_embd, n_exp], "Q8_0")],
        [("blk.1.ffn_gate_inp.weight", [n_embd, n_exp], "F32"), ("blk.1.attn_q_norm.weight", [64], "F16"),
         ("blk.1.hc_attn_up.weight", [n_embd, 2], "IQ3_XXS"), ("blk.1.ple_key.weight", [n_embd, 2], "IQ4_XS"),
         ("blk.1.attn_v.weight", [n_embd, 8], "IQ4_NL"), ("blk.1.ssm_beta.weight", [n_embd, 2], "BF16"),
         ("blk.1.ffn_gate_exps.weight", [n_embd, 16, n_exp], "IQ2_XS"), ("blk.1.ffn_up_exps.weight", [n_embd, 16, n_exp], "IQ2_XS"),
         ("blk.1.ffn_down_exps.weight", [256, n_embd, n_exp], "Q5_K")],
    ]
    total = sum(len(s) for s in shards)
    for i, ts in enumerate(shards):
        w = gguf.GGUFWriter(str(d / f"tiny-{i + 1:05d}-of-00002.gguf"), "qwen4exp")
        w.add_uint16("split.no", i)
        w.add_uint16("split.count", 2)
        w.add_int32("split.tensors.count", total)
        for t in ts:
            tensor(w, t[0], t[1], t[2], t[3] if len(t) > 3 else None)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
    pack = d / "pack"
    (pack / "tokenizer").mkdir(parents=True)
    for f in ("vocab.json", "chat_template.jinja"):
        (pack / "tokenizer" / f).write_text("{}\n")
    subprocess.run([PY, str(POLY / "tools/iq_pack.py"), "--gguf", str(d / "tiny-00001-of-00002.gguf"), "--out", str(pack),
                    "--compat-bf16", "--experts-bin"], check=True, cwd=d)
    # the source path in compat-bf16.json is this machine's: the test puts its own in its place
    c = pack / "compat-bf16.json"
    c.write_text(c.read_text().replace(str((d / "tiny-00001-of-00002.gguf").absolute()), "@SRC@"))


def record_tiny_mtp():
    d = OUT / "tiny-mtp"
    if d.exists():
        shutil.rmtree(d)
    (d / "src" / "tensors").mkdir(parents=True)
    specs = [("mtp.layers.0.mlp.experts.gate_up_proj", [4, 16, 128]), ("mtp.layers.0.mlp.experts.down_proj", [4, 128, 64]),
             ("mtp.pre_fc_norm_hidden.weight", [64]), ("mtp.fc_hidden.weight", [32, 64]), ("mtp.layers.0.mlp.gate.weight", [4, 64])]
    manifest = []
    for name, shape in specs:
        u = ((rng.standard_normal(int(np.prod(shape))) * 0.05).astype(np.float32).view(np.uint32) >> 16).astype("<u2")
        f = d / "src" / "tensors" / (name + ".bin")
        f.write_bytes(u.tobytes())
        manifest.append({"name": name, "shape": shape, "file": "tensors/" + name + ".bin",
                         "sha256": hashlib.sha256(u.tobytes()).hexdigest()})
    (d / "src" / "mtp-manifest.json").write_text(json.dumps(manifest, indent=1))
    for kind in ("q2_0", "q4_0", "q8_0"):
        subprocess.run([PY, str(POLY / "tools/mtp_pack.py"), "--src", str(d / "src"), "--experts", kind,
                        "--out", str(d / f"mtp-{kind}.gguf"), "--check-experts", "3"], check=True)


if __name__ == "__main__":
    record_quants()
    record_numbers()
    record_quantizers()
    record_tiny()
    record_tiny_mtp()
    print("wrote", OUT)
