import sys
sys.path.insert(0, "~/.unsloth/llama.cpp"); sys.path.insert(0, "~/.unsloth/llama.cpp/gguf-py")
import conversion.base as cb
import convert_hf_to_gguf as c
# the tiny synthetic tokenizer matches no known pre-tokenizer hash; name one llama.cpp supports (ByteLevel BPE)
for cls in [getattr(m, n) for m in (c, cb) for n in dir(m)]:
    if isinstance(cls, type) and hasattr(cls, "get_vocab_base_pre"):
        cls.get_vocab_base_pre = lambda self, tokenizer: "gpt-2"
sys.argv = ["convert_hf_to_gguf.py", "tiny-llama-hf", "--outtype", "f32", "--outfile", "tiny-llama.f32.gguf"]
c.main()
