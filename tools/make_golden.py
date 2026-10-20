"""Record golden prompt renderings and token ids from PolyStrata's Python front end.

For every case: the chat template's output text (serve/frontend.py ChatTemplate.render, the server's path) and the
ids of tools/strata_tokenizer.py Tokenizer.encode(text, parse_special=True). The Rust template and tokenizer must
reproduce both exactly.

Run with PolyStrata's virtual environment:
  ~/Development/PolyStrata/.venv/bin/python tools/make_golden.py
"""
import json
import pathlib
import sys

POLY = pathlib.Path.home() / "Development/PolyStrata"
PACKS = pathlib.Path.home() / "Development/Strata-data/packs"
sys.path[:0] = [str(POLY), str(POLY / "serve"), str(POLY / "tools")]

from serve.frontend import ChatTemplate  # noqa: E402
import strata_tokenizer as ST  # noqa: E402

MODELS = {"tiel": PACKS / "tiel-ud-q6_k_xl/tokenizer", "orca": PACKS / "orca-iq3_xxs/tokenizer"}

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search_code", "description": "Search the repository.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                   "required": ["query"]}}}

CASES = [
    ("plain", [{"role": "user", "content": "Hello!"}], None, {}),
    ("no_thinking", [{"role": "user", "content": "What is 17*23?"}], None, {"enable_thinking": False}),
    ("system", [{"role": "system", "content": "You are a terse assistant."},
                {"role": "user", "content": "Explain TCP in one sentence."}], None, {}),
    ("multi_turn", [{"role": "user", "content": "Name a prime."},
                    {"role": "assistant", "content": "7"},
                    {"role": "user", "content": "Another one?"}], None, {}),
    ("reasoning_history", [{"role": "user", "content": "2+2?"},
                           {"role": "assistant", "content": "4", "reasoning_content": "Simple addition."},
                           {"role": "user", "content": "And 3+3?"}], None, {}),
    ("unicode", [{"role": "user", "content": "नमस्ते दुनिया! 你好，世界。 Ça va? 🙂🚀 é naïve — “quotes”"}], None, {}),
    ("code", [{"role": "user", "content": "Fix this:\n```python\ndef f(x):\n\treturn x**2  # square\n```\n<b>&amp;</b> {'a': [1, 2]}"}], None, {}),
    ("whitespace", [{"role": "user", "content": "  leading\n\n\ntrailing   \n\t tabs\r\nwindows"}], None, {}),
    ("tools_offered", [{"role": "user", "content": "Weather in Pune?"}], [WEATHER, SEARCH], {}),
    ("tool_round_trip", [
        {"role": "user", "content": "Weather in Pune?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Pune", "unit": "c"}}}]},
        {"role": "tool", "content": "{\"temp\": 31, \"sky\": \"clear\"}"},
        {"role": "user", "content": "Thanks. And Mumbai?"}], [WEATHER], {}),
    ("history_no_gen_prompt", [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
     None, {"add_generation_prompt": False}),
    ("long_mixed", [{"role": "user", "content": ("The quick brown fox jumps over the lazy dog. 12345 67.89 "
                                                  "don't won't I'll we're they've. " * 20)}], None, {}),
]


def main() -> int:
    out_dir = pathlib.Path(__file__).resolve().parent.parent / "golden"
    out_dir.mkdir(exist_ok=True)
    for name, tdir in MODELS.items():
        tmpl = ChatTemplate(tdir / "chat_template.jinja")
        tok = ST.Tokenizer.from_dir(tdir)
        cases = []
        for cname, messages, tools, kwargs in CASES:
            kw = dict(kwargs)
            agp = kw.pop("add_generation_prompt", True)
            text = tmpl.render(messages, tools=tools, add_generation_prompt=agp, **kw)
            ids = tok.encode(text, parse_special=True)
            assert tok.decode(ids) == text, f"{name}/{cname}: decode does not round-trip"
            cases.append({"name": cname, "messages": messages, "tools": tools, "add_generation_prompt": agp,
                          "kwargs": kw, "text": text, "ids": ids})
        # tokenizer-only cases: encode with and without parse_special, exercising CONTROL vs USER_DEFINED literals
        tok_cases = []
        for s in ["<|im_start|>user\nhi<|im_end|>", "<think>\nreasoning\n</think>\n\nanswer", "<tool_call>x</tool_call>",
                  "a<|im_star b", "   \n\n  x  ", "1234567890", "Ünïcödé ⁂ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "x" * 300]:
            for ps in (True, False):
                tok_cases.append({"text": s, "parse_special": ps, "ids": tok.encode(s, parse_special=ps)})
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps({"tokenizer_dir": str(tdir), "cases": cases, "tokenizer_cases": tok_cases},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{path}: {len(cases)} template cases, {len(tok_cases)} tokenizer cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
