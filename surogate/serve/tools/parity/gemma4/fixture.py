"""A small dense Gemma 4, for comparing the engine against the transformers reference.

The 12B's *head geometry* verbatim -- 16 query heads of 256 windowed, 16 of 512 global over one
key/value head, attention_k_eq_v -- over a narrow residual and six layers. The geometry is what
the target's per-layer machinery turns on, so it is the part that must not be scaled down.

The QK-norm weights are drawn near the values the real 12B carries (q 1.0234, k 0.1221) rather
than uniformly. That is not cosmetic: Gemma 4 attends with softmax scale 1.0 and relies on
those norms to keep `q . k` in range. Drawn uniformly in [0.5, 2.0] the logits reach +-36 over
a 256-wide head, the softmax becomes effectively an argmax, and a 0.1% difference in the
operands -- which is what int8 weights cost -- flips it. The fixture then measures its own
conditioning rather than the engine.
"""
import json, pathlib, shutil, sys, torch
from safetensors.torch import save_file

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models/gemma-4-tiny")
src = pathlib.Path("models/gemma-4-12B")
out.mkdir(parents=True, exist_ok=True)

cfg = json.loads((src / "config.json").read_text())
text = cfg["text_config"]
text["hidden_size"] = 512
text["num_hidden_layers"] = 6
text["intermediate_size"] = 1024
text["layer_types"] = ["sliding_attention"] * 5 + ["full_attention"]
cfg["text_config"] = text
for key in ("vision_config", "audio_config"):
    cfg.pop(key, None)
(out / "config.json").write_text(json.dumps(cfg, indent=2))

H, L, F = text["hidden_size"], text["num_hidden_layers"], text["intermediate_size"]
V = text["vocab_size"]
HEADS, KV, HD = text["num_attention_heads"], text["num_key_value_heads"], text["head_dim"]
GKV, GHD = text["num_global_key_value_heads"], text["global_head_dim"]
windowed = [t == "sliding_attention" for t in text["layer_types"]]

torch.manual_seed(20260907)
def rnd(*shape, scale=0.02):
    return (torch.randn(*shape) * scale).to(torch.bfloat16)
def near(value, width, *shape):
    return (torch.full(shape, value) + (torch.rand(*shape) - 0.5) * width).to(torch.bfloat16)

sd = {"model.language_model.embed_tokens.weight": rnd(V, H),
      "model.language_model.norm.weight": near(1.0, 0.2, H)}
for i in range(L):
    p = f"model.language_model.layers.{i}."
    hd, kv = (HD, KV) if windowed[i] else (GHD, GKV)
    sd[p + "self_attn.q_proj.weight"] = rnd(HEADS * hd, H)
    sd[p + "self_attn.k_proj.weight"] = rnd(kv * hd, H)
    # A global layer under attention_k_eq_v ships no value projection: its value is the key
    # projection's raw output, normalised. This is what the converter must not look for.
    if windowed[i]:
        sd[p + "self_attn.v_proj.weight"] = rnd(kv * hd, H)
    sd[p + "self_attn.o_proj.weight"] = rnd(H, HEADS * hd)
    sd[p + "self_attn.q_norm.weight"] = near(1.02, 0.05, hd)
    sd[p + "self_attn.k_norm.weight"] = near(0.12, 0.02, hd)
    for n in ("input_layernorm", "post_attention_layernorm",
              "pre_feedforward_layernorm", "post_feedforward_layernorm"):
        sd[p + n + ".weight"] = near(1.0, 0.4, H)
    sd[p + "mlp.gate_proj.weight"] = rnd(F, H)
    sd[p + "mlp.up_proj.weight"] = rnd(F, H)
    sd[p + "mlp.down_proj.weight"] = rnd(H, F)
    # Spread across the range the real 12B shows (0.0053 .. 0.918), never 1.
    sd[p + "layer_scalar"] = torch.tensor([0.05 + 0.15 * i], dtype=torch.bfloat16)

save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
    shutil.copy(src / name, out / name)
print("tensors:", len(sd), " params:", round(sum(v.numel() for v in sd.values()) / 1e6, 1), "M")
