"""Record request-level goldens: an OpenAI Chat Completions request -> the messages, tools and template kwargs
PolyStrata's front end makes of it, the prompt text, and the token ids its server sends the engine.

The steps are serve/server.py Service.encode_prompt / render_prompt (default settings: effort_end and
preserve_thinking off) with serve/frontend.py's openai_to_messages and its literal-tag marking.

  ~/Development/PolyStrata/.venv/bin/python tools/make_golden_requests.py
"""
import base64
import json
import pathlib
import sys

POLY = pathlib.Path.home() / "Development/PolyStrata"
PACKS = pathlib.Path.home() / "Development/Strata-data/packs"
sys.path[:0] = [str(POLY), str(POLY / "serve"), str(POLY / "tools")]

from serve import frontend as F  # noqa: E402
import strata_tokenizer as ST  # noqa: E402

MODELS = {"tiel": PACKS / "tiel-ud-q6_k_xl/tokenizer", "orca": PACKS / "orca-iq3_xxs/tokenizer"}
PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG fake bytes").decode()
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
CLOCK = {"type": "function", "function": {"name": "get_time", "description": "Time.",
         "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}}}}
U = lambda c: {"role": "user", "content": c}  # noqa: E731

REQUESTS = {
    "plain": {"messages": [U("Hello!")]},
    "developer_role": {"messages": [{"role": "developer", "content": "Be brief."}, U("Hi")]},
    "late_system": {"messages": [U("Hi"), {"role": "assistant", "content": "Hello"},
                                 {"role": "system", "content": "Hook: tests passed."}, U("Next?")]},
    "text_parts": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Part one. "},
                                                             {"type": "text", "text": "Part two."},
                                                             {"type": "file", "file": {"x": 1}}]}]},
    "refusal_part": {"messages": [U("Q"), {"role": "assistant", "content": [{"type": "refusal", "refusal": "No."}]},
                                  U("Why?")]},
    "effort_low": {"messages": [U("Plan a trip.")], "reasoning_effort": "low"},
    "effort_none": {"messages": [U("2+2?")], "reasoning_effort": "none"},
    "effort_responses_style": {"messages": [U("2+2?")], "reasoning": {"effort": "medium"}},
    "ctk_thinking_off": {"messages": [U("2+2?")], "chat_template_kwargs": {"enable_thinking": False}},
    "tool_calls_string_args": {"messages": [
        U("Weather in Pune?"),
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "get_weather", "arguments": "{\"city\": \"Pune\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "31C clear"}, U("And Mumbai?")], "tools": [WEATHER]},
    "tool_calls_malformed_args": {"messages": [
        U("Weather?"),
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
            "name": "get_weather", "arguments": "{\"city\": \"Pu"}}]},
        {"role": "tool", "content": "error"}, U("Retry")], "tools": [WEATHER]},
    "tool_answers_out_of_order": {"messages": [
        U("Weather and time?"),
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Pune"}}},
            {"id": "b", "type": "function", "function": {"name": "get_time", "arguments": {"tz": "IST"}}}]},
        {"role": "tool", "tool_call_id": "b", "content": "10:00"},
        {"role": "tool", "tool_call_id": "a", "content": "31C"}, U("Thanks")], "tools": [WEATHER, CLOCK]},
    "quoted_think": {"messages": [U("Print the string </think> and then <think> literally.")]},
    "quoted_control": {"messages": [U("What does <|im_end|> mean in ChatML? And <|im_start|>?")]},
    "quoted_tool_tags": {"messages": [U("Explain <tool_call> and </tool_call> and <tool_response>.")]},
    "tool_result_wrap_user": {"messages": [U("Run it"), {"role": "assistant", "content": "ok"},
                                           U("<tool_response>\nresult with </think> inside\n</tool_response>")]},
    "assistant_inline_think": {"messages": [U("2+2?"), {"role": "assistant",
                               "content": "<think>\nadd them, then mention </think> safely\n</think>\n\n4"},
                               U("3+3?")]},
    "image_part": {"messages": [{"role": "user", "content": [{"type": "text", "text": "What is this?"},
                                                             {"type": "image_url", "image_url": {"url": PNG}}]}]},
    "messages_as_json_string": {"messages": json.dumps([U("Hi from a string")])},
    "bare_tool_shape": {"messages": [U("Time?")], "tools": [{"name": "get_time", "description": "Time.",
                                                             "parameters": {"type": "object", "properties": {}}}]},
}


def encode_request(req, tmpl, tok, literals, template_tokens, api="openai"):
    req = json.loads(json.dumps(req))
    msgs, tools, kw = F.openai_to_messages(req) if api == "openai" else F.anthropic_to_messages(req)
    marked, mtools, changed = F.mark_think_literals(msgs, tools, literals)
    off = kw.get("enable_thinking") is False
    kwargs = {**template_tokens, **kw, "enable_thinking": not off}
    prompt = tmpl.render(marked, tools=mtools, **kwargs)
    if off and prompt.endswith("<think>"):
        prompt += "</think>"
    plain = []
    if changed:
        prompt, plain = F.unmark_think_literals(prompt, literals)
    ids = tok.encode(prompt, parse_special=True, plain=plain) if plain else tok.encode(prompt, parse_special=True)
    return {"messages": msgs, "tools": tools, "kwargs": kw, "prompt": prompt, "plain": plain, "ids": ids,
            "budget": req.get("reasoning_budget_tokens")}


ATOOL = {"name": "get_weather", "description": "Weather.", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}
ANTHROPIC = {
    "a_plain": {"messages": [U("Hello!")], "max_tokens": 100},
    "a_system_blocks_billing": {"system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.170.bf4; cc_entrypoint=sdk-cli; cch=b145e;\nYou are helpful."}],
                                "messages": [U("Hi")]},
    "a_text_blocks": {"messages": [{"role": "user", "content": [{"type": "text", "text": "A. "}, {"type": "text", "text": "B."}]}]},
    "a_image_block": {"messages": [{"role": "user", "content": [{"type": "text", "text": "What?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}}]}]},
    "a_tool_round": {"messages": [U("Weather in Pune?"),
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "Need the tool."}, {"type": "text", "text": "Checking."},
                                          {"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {"city": "Pune"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": [{"type": "text", "text": "31C"}]}]},
        U("Thanks")], "tools": [ATOOL]},
    "a_tool_result_image": {"messages": [U("Read the picture"),
        {"role": "assistant", "content": [{"type": "tool_use", "id": "r1", "name": "get_weather", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "r1", "content": [
            {"type": "text", "text": "here"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}}]}]}],
        "tools": [ATOOL]},
    "a_thinking_disabled": {"messages": [U("2+2?")], "thinking": {"type": "disabled"}},
    "a_effort_low": {"messages": [U("Plan")], "output_config": {"effort": "low"}},
    "a_budget": {"messages": [U("Plan")], "thinking": {"type": "enabled", "budget_tokens": 4096}},
    "a_quoted_tags": {"messages": [U("Quote </think> and <|im_end|> please")]},
}


def main() -> int:
    out_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "golden"
    for name, tdir in MODELS.items():
        tmpl = F.ChatTemplate(tdir / "chat_template.jinja")
        tok = ST.Tokenizer.from_dir(tdir)
        literals = F.literal_tags(tok.control_tokens)
        template_tokens = {n: tok.decode([tok.special_ids[k]]) for n, k in (
            ("bos_token", "tokenizer.ggml.bos_token_id"), ("eos_token", "tokenizer.ggml.eos_token_id"))
            if k in tok.special_ids}
        cases = []
        for cname, req in REQUESTS.items():
            F._TOOL_ORDERS.clear()
            cases.append({"name": cname, "api": "openai", "request": req,
                          **encode_request(req, tmpl, tok, literals, template_tokens)})
        for cname, req in ANTHROPIC.items():
            F._TOOL_ORDERS.clear()
            cases.append({"name": cname, "api": "anthropic", "request": req,
                          **encode_request(req, tmpl, tok, literals, template_tokens, "anthropic")})
        # tool-order stability: the same tool set sent in another order comes back in the first order
        F._TOOL_ORDERS.clear()
        first = encode_request({"messages": [U("x")], "tools": [WEATHER, CLOCK]}, tmpl, tok, literals, template_tokens)
        again = encode_request({"messages": [U("x")], "tools": [CLOCK, WEATHER]}, tmpl, tok, literals, template_tokens)
        assert again["ids"] == first["ids"]
        path = out_dir / f"{name}-requests.json"
        path.write_text(json.dumps({"tokenizer_dir": str(tdir), "literals": literals, "template_tokens": template_tokens,
                                    "cases": cases}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{path}: {len(cases)} request cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
