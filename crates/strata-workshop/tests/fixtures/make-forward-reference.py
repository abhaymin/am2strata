"""Reference logits for the Workshop's own Llama-family forward pass (crates/strata-workshop/src/model.rs).

Run ONCE, by hand, on the CPU, from this folder (nothing in `cargo test` runs Python):

    CUDA_VISIBLE_DEVICES="" ~/Development/soup-project/.venv/bin/python make-forward-reference.py

Made with torch 2.11.0, transformers 5.15.1, peft 0.20.0. It writes:
  * tiny-qwen2-hf/, tiny-qwen3-hf/, tiny-llama-tied-hf/  - random tiny models (seeded) saved by transformers itself:
    Qwen2 has q/k/v biases, Qwen3 has q_norm/k_norm and an explicit head_dim, the tied one shares embed and lm_head;
  * tiny-gqa-peft-adapter/ - a PEFT rsLoRA adapter with random (non-zero) A and B on tiny-llama-gqa-hf, all 7 targets;
  * forward-reference.json - for each model: the token ids and transformers' float32 logits [seq][vocab]; for the
    adapter: PEFT's logits of base+adapter.
"""
import json, os, torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM, LlamaConfig, LlamaForCausalLM

torch.manual_seed(1234)
here = os.path.dirname(os.path.abspath(__file__))
out = {}

def tokens(vocab, n=12, mul=7, add=3):
    return [(i * mul + add) % vocab for i in range(n)]

def logits_of(model, ids):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0].float().tolist()

def randomize(model, std):
    # transformers' default init (std 0.02) gives near-uniform logits; larger weights make a parity check mean something
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() >= 2:
                p.normal_(0.0, std if "embed" not in n else 1.0)
            elif "bias" in n:
                p.normal_(0.0, 0.1)
            else:
                p.uniform_(0.8, 1.2)   # norm gains

def load(name):
    d = os.path.join(here, name)
    if os.path.exists(os.path.join(d, "model.safetensors")):
        return AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32)
    # the older fixtures: no model_type in config.json and a one-shard file without an index (llama.cpp's converter
    # needs neither), so build the class from the config and load the tensors by name, strictly, as float32
    from safetensors.torch import load_file
    model = LlamaForCausalLM(LlamaConfig(**json.load(open(os.path.join(d, "config.json")))))
    state = {k: v.float() for k, v in load_file(os.path.join(d, "model-00001-of-00001.safetensors")).items()}
    model.load_state_dict(state, strict=True)
    return model.float()

common = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
              num_key_value_heads=2, max_position_embeddings=128, rms_norm_eps=1e-6, bos_token_id=0, eos_token_id=1,
              pad_token_id=None, attention_dropout=0.0)
new = {
    "tiny-qwen2-hf": Qwen2ForCausalLM(Qwen2Config(**common, rope_theta=1000000.0, tie_word_embeddings=False, use_sliding_window=False)),
    "tiny-qwen3-hf": Qwen3ForCausalLM(Qwen3Config(**common, head_dim=16, rope_theta=1000000.0, tie_word_embeddings=False, use_sliding_window=False)),
    "tiny-llama-tied-hf": LlamaForCausalLM(LlamaConfig(**common, rope_theta=10000.0, tie_word_embeddings=True)),
}
for name, model in new.items():
    randomize(model, 0.15)
    if name == "tiny-llama-tied-hf":
        model.tie_weights()
    model.save_pretrained(os.path.join(here, name), safe_serialization=True)
    for junk in ("generation_config.json",):
        p = os.path.join(here, name, junk)
        if os.path.exists(p):
            os.remove(p)

for name in ["tiny-llama-hf", "tiny-llama-256-hf", "tiny-llama-gqa-hf", "tiny-llama-256-bf16-hf", *new]:
    model = load(name)
    ids = tokens(model.config.vocab_size)
    out[name] = {"tokens": ids, "logits": logits_of(model, ids)}

# a PEFT adapter with non-zero B, rsLoRA scaling, on the grouped-attention model
from peft import LoraConfig, get_peft_model
base = load("tiny-llama-gqa-hf")
cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, use_rslora=True, task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
pm = get_peft_model(base, cfg)
torch.manual_seed(4321)
with torch.no_grad():
    for n, p in pm.named_parameters():
        if "lora_" in n:
            p.normal_(0.0, 0.05)
ids = tokens(32)
out["tiny-gqa-peft-adapter"] = {"base": "tiny-llama-gqa-hf", "tokens": ids, "logits": logits_of(pm, ids)}
pm.save_pretrained(os.path.join(here, "tiny-gqa-peft-adapter"))
p = os.path.join(here, "tiny-gqa-peft-adapter", "README.md")
if os.path.exists(p):
    os.remove(p)

with open(os.path.join(here, "forward-reference.json"), "w") as f:
    json.dump(out, f)
print({k: (len(v["logits"]), len(v["logits"][0])) for k, v in out.items()})
