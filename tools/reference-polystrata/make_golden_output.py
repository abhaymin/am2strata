"""Record OutputParser goldens: model output text fed to PolyStrata's serve/frontend.py OutputParser (call_format
"function", recover off - the server defaults) in given chunks -> the events it emits.

  ~/Development/PolyStrata/.venv/bin/python tools/make_golden_output.py
"""
import json
import pathlib
import sys

POLY = pathlib.Path.home() / "Development/PolyStrata"
sys.path[:0] = [str(POLY), str(POLY / "serve")]
from serve.frontend import OutputParser  # noqa: E402

TOOLS = [{"name": "get_weather", "description": "Weather.", "parameters": {"type": "object", "properties": {
             "city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"}}}},
         {"name": "write_file", "description": "Write.", "parameters": {"type": "object", "properties": {
             "path": {"type": "string"}, "content": {"type": "string"}}}}]

CALL = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nPune\n</parameter>\n<parameter=days>\n3\n</parameter>\n"
        "<parameter=metric>\ntrue\n</parameter>\n</function>\n</tool_call>")
WRITE = ("<tool_call>\n<function=write_file>\n<parameter=path>\nsrc/a.py\n</parameter>\n<parameter=content>\n"
         "def f():\n    return \"héllo </parameter> inside\"\n</parameter>\n</function>\n</tool_call>")

CASES = {
    "reasoning_then_answer": ("Let me add 2 and 2.\n</think>\n\nThe answer is 4.", True, None),
    "no_thinking": ("Just the answer, no reasoning.", False, None),
    "quoted_think_end": ("The tag `</think>` ends thinking. Done now.\n</think>\n\nOK.", True, None),
    "tool_call_after_answer": ("I'll check.\n</think>\n\nChecking the weather.\n\n" + CALL, True, TOOLS),
    "tool_call_only": ("\n\n" + CALL, False, TOOLS),
    "two_calls": ("\n\n" + CALL + "\n" + WRITE, False, TOOLS),
    "call_value_with_tag": ("Writing.\n\n" + WRITE, False, TOOLS),
    "call_in_code_fence": ("Example:\n```\n" + CALL + "\n```\nThat is the format.", False, TOOLS),
    "prose_naming_tag": ("I would use a <tool_call> block here but will not.", False, TOOLS),
    "call_inside_reasoning_rescued": ("I should call the tool.\n" + CALL + "\n</think>", True, TOOLS),
    "call_inside_reasoning_quoted": ("I should call the tool.\n" + CALL + "\nactually no, I will just answer.\n</think>\n\nIt is sunny.", True, TOOLS),
    "unicode_and_newlines": ("Résumé: naïve façade — 東京 🙂\n</think>\n\n\n\nFinal: ✓", True, None),
    "no_tools_declared_tag_is_text": ("Here: " + CALL, False, None),
    "unterminated_call": ("Calling.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nPu", False, TOOLS),
}

# the other call formats (serve/server.py picks one by the tokenizer's pre-tokenizer): (text, thinking, tools, format,
# opened)
GLM = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Pune</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"
GLM_TAG = "<tool_call>write_file<arg_key>path</arg_key><arg_value>a.md</arg_value><arg_key>content</arg_key><arg_value>see </tool_call> here</arg_value></tool_call>"
GEMMA = '<|tool_call>call:get_weather{city:<|"|>Pune<|"|>,days:3,metric:true}<tool_call|>'
GEMMA_NEST = '<|tool_call>call:write_file{path:<|"|>x <tool_call|> y<|"|>,content:{a:[1,2],b:<|"|>z<|"|>}}<tool_call|>'
DSML = "｜DSML｜"
DSML_CALL = ("<" + DSML + "tool_calls>\n<" + DSML + 'invoke name="get_weather">\n<' + DSML + 'parameter name="city" string="true">Pune</'
             + DSML + "parameter>\n<" + DSML + 'parameter name="days" string="false">3</' + DSML + "parameter>\n</" + DSML
             + "invoke>\n</" + DSML + "tool_calls>")
LFM2 = "<|tool_call_start|>[get_weather(city='Pune', days=3, metric=True), write_file(path=\"a.py\", content='x\\'y')]<|tool_call_end|>"
FORMAT_CASES = {
    "glm_answer_then_call": ("Need weather.</think>Checking.\n" + GLM, True, TOOLS, "glm", False),
    "glm_value_with_end_tag": ("\n" + GLM_TAG, False, TOOLS, "glm", False),
    "glm_end_tag_no_newline": ('say "Paris."</think>Paris', True, None, "glm", False),
    "gemma_opens_its_thought": ("<|channel>thought\nThinking here.<channel|>Answer.\n" + GEMMA, False, TOOLS, "gemma", False),
    "gemma_opened_by_prompt": ("Still thinking.<channel|>" + GEMMA_NEST, True, TOOLS, "gemma", True),
    "gemma_opened_not_thinking": ("Still thinking.<channel|>" + GEMMA_NEST, False, TOOLS, "gemma", True),
    "gemma_no_thought": ("Plain answer.", True, None, "gemma", False),
    "dsml_block": ("Let me look.\n</think>\n\nOne moment.\n\n" + DSML_CALL, True, TOOLS, "dsml", False),
    "lfm2_two_calls": ("Plan.</think>\n" + LFM2, True, TOOLS, "lfm2", False),
    "lfm2_unterminated": ("<|tool_call_start|>[get_weather(city='Pu", False, TOOLS, "lfm2", False),
}


def run(text, thinking, tools, chunks, stream_tools, call_format="function", opened=False):
    p = OutputParser(thinking=thinking, tools=tools, stream_tools=stream_tools, call_format=call_format, opened=opened)
    events = []
    pieces = [text] if chunks == "whole" else [text[i:i + chunks] for i in range(0, len(text), chunks)]
    try:
        for piece in pieces:
            events += p.feed(piece)
        events += p.finish()
    except ValueError as e:
        return {"error": str(e)}
    out = []
    for e in events:
        d = {"kind": e.kind, "text": e.text}
        if e.call is not None:
            d["name"] = e.call.name
            d["arguments"] = e.call.arguments
        out.append(d)
    return out


def main() -> int:
    cases = []
    for name, (text, thinking, tools) in CASES.items():
        for chunks in ("whole", 1, 3, 7):
            for stream_tools in (False, True):
                if stream_tools and not tools:
                    continue
                cases.append({"name": name, "text": text, "thinking": thinking, "tools": tools, "chunks": chunks,
                              "stream_tools": stream_tools, "events": run(text, thinking, tools, chunks, stream_tools)})
    for name, (text, thinking, tools, fmt, opened) in FORMAT_CASES.items():
        for chunks in ("whole", 1, 3, 7):
            for stream_tools in (False, True):
                if stream_tools and not tools:
                    continue
                cases.append({"name": name, "text": text, "thinking": thinking, "tools": tools, "chunks": chunks,
                              "stream_tools": stream_tools, "call_format": fmt, "opened": opened,
                              "events": run(text, thinking, tools, chunks, stream_tools, fmt, opened)})
    path = pathlib.Path(__file__).resolve().parent.parent.parent / "golden" / "output.json"
    path.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{path}: {len(cases)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
