"""A small Gemma 4 mixture, for comparing the engine against the transformers reference.

**Only the layer count is scaled down.** The 26B-A4B's width, both head geometries and its
whole mixture -- 2,816 hidden, 128 experts of 704, top-8, a 2,112-wide dense feed-forward
beside them -- are verbatim, because `ops::SparseMoeGeometry` is a *closed* registry: the
kernels are compiled per registered mixture and a fixture with 32 experts of 128 would be
refused by the op rather than served approximately. Two layers, one windowed and one global,
is what makes that affordable: the expert bank alone is 761 M parameters per layer.

The QK-norm weights are drawn near the values the real checkpoints carry (q 1.0234, k 0.1221)
rather than uniformly, for the reason `fixture.py` gives: Gemma 4 attends with softmax scale
1.0 and relies on those norms to keep `q . k` in range.

The router's two scales are drawn near one. Drawn wide they would make the top-8 selection
effectively deterministic on a handful of experts, and the fixture would then measure its own
conditioning rather than whether the engine routes as the reference does.
"""
import json, pathlib, shutil, sys, torch
from safetensors.torch import save_file

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models/gemma-4-moe-tiny")
src = pathlib.Path("models/gemma-4-12B")
config_source = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else src / "config.json"
out.mkdir(parents=True, exist_ok=True)

cfg = json.loads(config_source.read_text())
text = cfg["text_config"]
# The mixture, verbatim from the published 26B-A4B. Restated here rather than inherited so a
# fixture built from the *dense* 12B's config still describes the registered geometry.
text["hidden_size"] = 2816
text["intermediate_size"] = 2112
text["moe_intermediate_size"] = 704
text["num_experts"] = 128
text["top_k_experts"] = 8
text["enable_moe_block"] = True
text["num_attention_heads"] = 16
text["num_key_value_heads"] = 8
text["head_dim"] = 256
text["global_head_dim"] = 512
text["num_global_key_value_heads"] = 2
text["attention_k_eq_v"] = True
text["num_kv_shared_layers"] = 0
text["hidden_size_per_layer_input"] = 0
text["use_double_wide_mlp"] = False
text["num_hidden_layers"] = 2
text["layer_types"] = ["sliding_attention", "full_attention"]
cfg["text_config"] = text
for key in ("vision_config", "audio_config"):
    cfg.pop(key, None)
(out / "config.json").write_text(json.dumps(cfg, indent=2))

H, L, F = text["hidden_size"], text["num_hidden_layers"], text["intermediate_size"]
M, E, K = text["moe_intermediate_size"], text["num_experts"], text["top_k_experts"]
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
    # projection's raw output, normalised.
    if windowed[i]:
        sd[p + "self_attn.v_proj.weight"] = rnd(kv * hd, H)
    sd[p + "self_attn.o_proj.weight"] = rnd(H, HEADS * hd)
    sd[p + "self_attn.q_norm.weight"] = near(1.02, 0.05, hd)
    sd[p + "self_attn.k_norm.weight"] = near(0.12, 0.02, hd)
    # Seven norms on a mixture layer where a dense one has four: the three extra are what the
    # dense branch and the routed branch meet under.
    for n in ("input_layernorm", "post_attention_layernorm",
              "pre_feedforward_layernorm", "post_feedforward_layernorm",
              "post_feedforward_layernorm_1", "pre_feedforward_layernorm_2",
              "post_feedforward_layernorm_2"):
        sd[p + n + ".weight"] = near(1.0, 0.4, H)
    sd[p + "mlp.gate_proj.weight"] = rnd(F, H)
    sd[p + "mlp.up_proj.weight"] = rnd(F, H)
    sd[p + "mlp.down_proj.weight"] = rnd(H, F)
    # The router. `proj` is [experts, hidden]; `scale` a per-channel vector applied to the
    # weightlessly normalised input; `per_expert_scale` multiplies the renormalised top-k
    # weights. Both are BF16 here because a checkpoint saved in bf16 stores every parameter
    # that way -- the 26B-A4B's config says `"dtype": "bfloat16"` -- and the artifact widens
    # `per_expert_scale` to FP32 on its own, because the object is declared `fp32` and
    # `derive_recipes` casts every such object.
    # **Wide, and that is the point.** Drawn at the 0.02 the other matrices use, 128 iid expert
    # rows give logits with a standard deviation of about 0.02, so the softmax over them is
    # essentially uniform at 1/128 and the gap between the 8th and 9th probabilities lands
    # around 1e-6 on a value of 8e-3 -- a relative margin of 2e-4, where BF16 carries 4e-3. The
    # artifact's router is BF16, so which experts a token takes is then a coin flip, and the
    # comparison measures the fixture's conditioning rather than the engine. Measured: one token
    # of thirteen flipped its 8th expert and scored 0.87 where the other twelve scored 0.9998.
    #
    # A trained router separates its experts; this one has to as well. At 0.5 the logits spread
    # about 25x wider and the 8th/9th margin clears BF16 by an order of magnitude.
    sd[p + "router.proj.weight"] = rnd(E, H, scale=0.5)
    sd[p + "router.scale"] = near(1.0, 0.2, H)
    sd[p + "router.per_expert_scale"] = near(1.0, 0.2, E)
    # Expert-major, exactly as the checkpoint stores them.
    sd[p + "experts.gate_up_proj"] = rnd(E, 2 * M, H)
    sd[p + "experts.down_proj"] = rnd(E, H, M)
    sd[p + "layer_scalar"] = torch.tensor([0.05 + 0.15 * i], dtype=torch.bfloat16)

save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
    shutil.copy(src / name, out / name)
print("tensors:", len(sd), " params:", round(sum(v.numel() for v in sd.values()) / 1e6, 1), "M")
