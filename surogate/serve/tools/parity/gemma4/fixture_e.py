"""A small E-series Gemma 4, for comparing the engine against the transformers reference.

Keeps everything the target's machinery turns on and shrinks only what it does not: the E2B's
head geometry (8 query heads of 256 through the window, 8 of 512 over the whole context, one
key/value head either way), its per-layer input width, its double-wide feed-forward on the
shared layers, and a schedule with shared layers of *both* kinds -- a windowed one and a global
one -- because which source a sharing layer reads depends on that.

The QK-norm weights are drawn near the real 12B's (q 1.0234, k 0.1221) for the reason
`fixture.py` gives: Gemma 4 attends with softmax scale 1.0 and a uniform draw makes the fixture
measure its own conditioning.
"""
import json, pathlib, shutil, sys, torch
from safetensors.torch import save_file

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models/gemma-4-e-tiny")
src = pathlib.Path("models/gemma-4-E2B-it")
out.mkdir(parents=True, exist_ok=True)

cfg = json.loads((src / "config.json").read_text())
text = cfg["text_config"]
text["hidden_size"] = 512
text["num_hidden_layers"] = 12
text["intermediate_size"] = 1024
# Every fifth layer global, as the E-series schedules them, and the last one global.
text["layer_types"] = ["sliding_attention" if (i + 1) % 5 else "full_attention"
                       for i in range(12)]
text["layer_types"][-1] = "full_attention"
# Four shared layers: 8..11, which covers a windowed sharer and a global one.
text["num_kv_shared_layers"] = 4
text["use_double_wide_mlp"] = True
cfg["text_config"] = text
for key in ("vision_config", "audio_config"):
    cfg.pop(key, None)
(out / "config.json").write_text(json.dumps(cfg, indent=2))

H, L, F = text["hidden_size"], text["num_hidden_layers"], text["intermediate_size"]
V, PLI = text["vocab_size"], text["hidden_size_per_layer_input"]
HEADS, KV, HD = text["num_attention_heads"], text["num_key_value_heads"], text["head_dim"]
GKV = text["num_global_key_value_heads"] or KV
GHD = text["global_head_dim"]
windowed = [t == "sliding_attention" for t in text["layer_types"]]
first_shared = L - text["num_kv_shared_layers"]

torch.manual_seed(20260907)
def rnd(*shape, scale=0.02):
    return (torch.randn(*shape) * scale).to(torch.bfloat16)
def near(value, width, *shape):
    return (torch.full(shape, value) + (torch.rand(*shape) - 0.5) * width).to(torch.bfloat16)

sd = {
    "model.language_model.embed_tokens.weight": rnd(V, H),
    "model.language_model.embed_tokens_per_layer.weight": rnd(V, L * PLI),
    "model.language_model.per_layer_model_projection.weight": rnd(L * PLI, H),
    "model.language_model.per_layer_projection_norm.weight": near(1.0, 0.2, PLI),
    "model.language_model.norm.weight": near(1.0, 0.2, H),
}
for i in range(L):
    p = f"model.language_model.layers.{i}."
    hd, kv = (HD, KV) if windowed[i] else (GHD, GKV)
    shared = i >= first_shared
    ffn = 2 * F if (shared and text["use_double_wide_mlp"]) else F
    sd[p + "self_attn.q_proj.weight"] = rnd(HEADS * hd, H)
    sd[p + "self_attn.o_proj.weight"] = rnd(H, HEADS * hd)
    sd[p + "self_attn.q_norm.weight"] = near(1.02, 0.05, hd)
    # A sharing layer holds no key or value projection and no key norm: it reads an earlier
    # layer's planes. The published E2B does ship those three tensors on its shared layers and
    # `transformers` builds no module for them; this fixture leaves them out entirely, so the
    # converter has nothing to skip and the parity run cannot pass by accident.
    if not shared:
        sd[p + "self_attn.k_proj.weight"] = rnd(kv * hd, H)
        sd[p + "self_attn.v_proj.weight"] = rnd(kv * hd, H)
        sd[p + "self_attn.k_norm.weight"] = near(0.12, 0.02, hd)
    for n in ("input_layernorm", "post_attention_layernorm",
              "pre_feedforward_layernorm", "post_feedforward_layernorm"):
        sd[p + n + ".weight"] = near(1.0, 0.4, H)
    sd[p + "mlp.gate_proj.weight"] = rnd(ffn, H)
    sd[p + "mlp.up_proj.weight"] = rnd(ffn, H)
    sd[p + "mlp.down_proj.weight"] = rnd(H, ffn)
    sd[p + "per_layer_input_gate.weight"] = rnd(PLI, H)
    sd[p + "per_layer_projection.weight"] = rnd(H, PLI)
    sd[p + "post_per_layer_input_norm.weight"] = near(1.0, 0.3, H)
    sd[p + "layer_scalar"] = torch.tensor([0.05 + 0.07 * i], dtype=torch.bfloat16)

save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
    shutil.copy(src / name, out / name)
print("tensors:", len(sd), " params:", round(sum(v.numel() for v in sd.values()) / 1e6, 1), "M")
print("layer_types:", text["layer_types"], " first shared:", first_shared)
