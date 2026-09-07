import os
# CPU reference for the layer-1 n-gram PLE of token 0 (llama.cpp build_ple algebra) on top of the layer-0 chain.
import sys, numpy as np
from gguf.quants import dequantize
_mlp = sys.argv[1]; sys.argv = [sys.argv[0], sys.argv[2]]
exec(open(_mlp).read().replace('print(', 'None and print('))   # mlp_reference.py -> x2 (residual after layer 0), src, hc, H, eps, f
f = lambda v: np.array2string(np.asarray(v), precision=4, suppress_small=True)
fields = src.fields
def scalar(key):
    fl = fields[key]; return int(fl.parts[fl.data[0]][0])
mult = src.array_field("qwen4exp.ple.layer_multipliers"); offs = src.array_field("qwen4exp.ple.head_offsets"); vocs = src.array_field("qwen4exp.ple.head_vocab_sizes")
eos = scalar("qwen4exp.ple.eos_token_id"); ngram = scalar("qwen4exp.ple.ngram_size"); per_gram = scalar("qwen4exp.ple.heads_per_ngram")
print("ple meta: mult", mult, "eos", eos, "ngram", ngram, "per_gram", per_gram, "heads", len(offs))
t0 = 248045
ctx = [t0] + [eos] * (ngram - 1)
M = (1 << 64) - 1
rows = []
for n in range(2, ngram + 1):
    mixed = (ctx[0] * mult[0]) & M
    for j in range(1, n):
        mixed ^= (ctx[j] * mult[j]) & M
    for g in range(per_gram):
        h = (n - 2) * per_gram + g
        rows.append(mixed % vocs[h] + offs[h])
print("rows", rows)
name = "per_layer_token_embd.weight"
t = src.tensor(name); raw = src.raw(name); print("table raw", raw.shape, raw.dtype, t.type_name)
emb = np.concatenate([np.asarray(dequantize(raw[r:r + 1], t.tensor_type), dtype=np.float32).reshape(-1) for r in rows])
print("ple_embd", f(emb[:3]), f(emb[-3:]), "| llama [0.0126 -0.0064 0.0024 ... -0.0054 0.0038 -0.0075]")
b = "blk.1."
key = src.float32(b + "ple_key.weight") @ emb          # (10240,)
value = src.float32(b + "ple_value.weight") @ emb      # (2560,)
def grouped_norm(v, w):
    s = v.reshape(hc, H); return ((s / np.sqrt((s * s).mean(axis=1, keepdims=True) + eps)).reshape(-1) * w)
keyn = grouped_norm(key, src.float32(b + "ple_norm_key.weight").reshape(-1))
query = grouped_norm(x2, src.float32(b + "ple_norm_query.weight").reshape(-1))
s_ = (keyn.reshape(hc, H) * query.reshape(hc, H)).sum(axis=1) / np.sqrt(H)
gate = 1 / (1 + np.exp(-(np.sign(s_) * np.sqrt(np.maximum(np.abs(s_), 1e-6)))))
print("ple_gate per stream", f(gate), "| llama stream0 0.7734")
gated = (np.tile(value, hc).reshape(hc, H) * gate[:, None]).reshape(-1)
print("ple_gated_value", f(gated[:3]), "| llama [-0.0017 -0.0099 0.0047]")
normalized = grouped_norm(gated, src.float32(b + "ple_norm_conv.weight").reshape(-1))
convw = src.float32(b + "ple_conv1d.weight"); print("ple conv", convw.shape)
wk = convw[:, -1] if convw.shape[0] == hc * H else convw[-1, :]
co = wk * normalized; co = co / (1 + np.exp(-co))
print("ple_conv_out", f(co[:3]), "| llama [0.0001 0.0008 -0.0007]")
x_ple = x2 + gated + co
print("layer1 input after PLE stream0", f(x_ple[:3]))
# layer-1 attention-side mix for the next check
b1 = "blk.1."
n1 = src.float32(b1 + "hc_attn_norm.weight").reshape(-1); d1 = src.float32(b1 + "hc_attn_down.weight"); u1 = src.float32(b1 + "hc_attn_up.weight")
xn2 = grouped_norm(x_ple, n1); lo2 = d1 @ xn2 / hc; lo2 = lo2 / (1 + np.exp(-lo2)); g2 = 1 / (1 + np.exp(-(u1 @ lo2)))
mixed1 = (xn2 * g2).reshape(hc, H).mean(axis=0)
print("hc_mixed-1 (attn side)", f(mixed1[:3]), f(mixed1[-3:]), "| llama [0.0021 -0.1524 0.1073 ... -0.0008 -0.1470 -0.0343]")
SC = os.environ.get("SUROGATE_PARITY_DIR", ".")
np.save(f"{SC}/ref_layer1_in.npy", x2.astype(np.float32)); np.save(f"{SC}/ref_ple_out.npy", x_ple.astype(np.float32)); np.save(f"{SC}/ref_L1_mixer_mixed.npy", mixed1.astype(np.float32))
np.save(f"{SC}/ref_layer0_ffn_mixed.npy", cur.astype(np.float32)); np.save(f"{SC}/ref_layer0_ffn_out.npy", ffn.astype(np.float32))
print("saved layer-1 reference npy files")
