"""Record tokenizer goldens for the pre-tokenizers with no installed model to test against: small synthetic
vocabularies for `gemma4` (UTF-8 pieces, byte fallback) and `deepseek-v3` (a pattern sequence), encoded by
PolyStrata's tools/strata_tokenizer.py.

  ~/Development/PolyStrata/.venv/bin/python tools/make_golden_tokenizers.py
"""
import json
import pathlib
import sys

POLY = pathlib.Path.home() / "Development/PolyStrata"
sys.path[:0] = [str(POLY / "tools")]
from strata_tokenizer import BYTE_TO_UNICODE, SPACE, Tokenizer  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "crates/strata-tokenizer/testdata"
TEXTS = ["Hello world", "the theme of the thesis", "  two  spaces\n\n\nnew lines\n", "naïve café 東京 🙂",
         "<|turn>user\nhi<turn|>\n<|tool_call>call:f{a:1}<tool_call|>", "<think>x</think> 12345 abc123",
         "日本語のテキスト、カタカナ！", "don't stop: a.b,c;d?!", ""]


def write(name, tokens, merges, types, pre, special_ids):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False), encoding="utf-8")
    (d / "merges.json").write_text(json.dumps(merges, ensure_ascii=False), encoding="utf-8")
    (d / "token_type.json").write_text(json.dumps(types))
    (d / "tokenizer.json").write_text(json.dumps({"pre": pre, "special_ids": special_ids}))
    tk = Tokenizer.from_dir(d)
    cases = [{"text": t, "parse_special": ps, "ids": tk.encode(t, parse_special=ps)} for t in TEXTS for ps in (False, True)]
    (d / "golden.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    print(d, len(tokens), "tokens", len(cases), "cases")


def gemma4():
    specials = [("<|turn>", 3), ("<turn|>", 3), ("<|tool_call>", 4), ("<tool_call|>", 4), ("<bos>", 3)]
    chars = sorted(set("".join(TEXTS).replace(" ", SPACE)) - set("東🙂京日本語"))   # some characters left to bytes
    merges = [f"{SPACE} t", "h e", f"{SPACE}t he", "t h", "th e", f"{SPACE}w o", "r l", "rl d", "e s", "l l", "H e", "He ll",
              "Hell o", "\n \n"]
    tokens, types = [], []
    for t, ty in specials:
        tokens.append(t); types.append(ty)
    for b in range(256):
        tokens.append("<0x%02X>" % b); types.append(6)
    for c in chars:
        if c not in tokens:
            tokens.append(c); types.append(1)
    for m in merges:
        t = m.replace(" ", "")
        if t not in tokens:
            tokens.append(t); types.append(1)
    for nl in ("\n\n", "\n\n\n"):
        if nl not in tokens:
            tokens.append(nl); types.append(1)
    write("tokenizer-gemma4-mini", tokens, merges, types, "gemma4", {"tokenizer.ggml.bos_token_id": 4})


def deepseek():
    specials = [("<｜User｜>", 3), ("<｜Assistant｜>", 3), ("<think>", 4), ("</think>", 4)]
    tokens, types = [], []
    for t, ty in specials:
        tokens.append(t); types.append(ty)
    for b in range(256):
        tokens.append(BYTE_TO_UNICODE[b]); types.append(1)
    m = lambda s: "".join(BYTE_TO_UNICODE[b] for b in s.encode("utf-8"))  # noqa: E731
    j = m("日")
    pairs = [("t", "h"), ("th", "e"), ("Ġ", "the"), ("1", "2"), ("12", "3"), ("e", "s"), ("Ġ", "w"), (j[0], j[1]), (j[:2], j[2])]
    merges = []
    for a, b in pairs:
        merges.append(f"{a} {b}")
        if a + b not in tokens:
            tokens.append(a + b); types.append(1)
    write("tokenizer-deepseek-mini", tokens, merges, types, "deepseek-v3", {})


gemma4()
deepseek()
