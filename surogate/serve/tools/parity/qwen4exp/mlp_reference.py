# CPU reference for layer-0 MLP side (hc mix, router, top-10 experts, shared expert, combine) of token 0.
import sys, numpy as np
from gguf.quants import dequantize
from surogate.serve.convert.qwen4exp import convert as cv
exec(open(sys.argv[1]).read().split("# ---- engine-comparable")[0].replace('print(', 'None and print('))  # reuse gdn_reference up to `out`
f = lambda v: np.array2string(np.asarray(v), precision=4, suppress_small=True)
# GDN output with llama.cpp's tiled pairing (the exec'd loop leaves the interleave variant in `out`)
o_t = np.stack([beta[h] * float(kn[h % 16] @ qn[h % 16]) * scale * v[h] for h in range(48)])
gn = src.float32(b + "ssm_norm.weight").reshape(128)
on_t = o_t / np.sqrt((o_t * o_t).mean(axis=-1, keepdims=True) + eps) * gn
fin_t = on_t * (1 / (1 + np.exp(-z.reshape(48, 128))))
out_t = src.float32(b + "ssm_out.weight") @ fin_t.reshape(-1)
print("linear_attn_out (tiled)", f(out_t[:3]), "| llama [-0.0390 0.0326 -0.1039]")
# residual after the attention combine (tiled algebra == llama.cpp)
inj_attn = src.float32(b + "hc_attn_inject.weight") @ xn
x1 = x + np.repeat(2 / (1 + np.exp(-inj_attn / hc)), H) * np.tile(out_t, hc)
print("hc_combine stream0", f(x1[:3]), "| llama [-0.0062 0.0043 -0.0023]")
norm_f = src.float32(b + "hc_ffn_norm.weight").reshape(hc * H)
down_f = src.float32(b + "hc_ffn_down.weight"); up_f = src.float32(b + "hc_ffn_up.weight"); inj_f = src.float32(b + "hc_ffn_inject.weight")
s1 = x1.reshape(hc, H); xn1 = (s1 / np.sqrt((s1 * s1).mean(axis=1, keepdims=True) + eps)).reshape(hc * H) * norm_f
lo1 = down_f @ xn1 / hc; lo1 = lo1 / (1 + np.exp(-lo1)); gate1 = 1 / (1 + np.exp(-(up_f @ lo1)))
cur = (xn1 * gate1).reshape(hc, H).mean(axis=0)
inject1 = inj_f @ xn1
print("ffn hc_mixed", f(cur[:3]), f(cur[-3:]), "| llama [-0.2214 0.3401 -0.6444 ... -0.0265 -0.8906 0.7217]")
print("ffn hc_inject logits", f(inject1), "| llama [-41.5651 -19.8327 -6.7227 -22.4999]")
router = src.float32(b + "ffn_gate_inp.weight")  # (512, 2560)
logits = router @ cur
probs = np.exp(logits - logits.max()); probs /= probs.sum()
top = np.argsort(-probs)[:10]
w = probs[top] / probs[top].sum()
print("top10 experts", top.tolist(), "weights", f(w))
def expert(name, e):
    t = src.tensor(name); raw = src.raw(name)
    if raw.ndim == 3:
        return np.asarray(dequantize(raw[e], t.reader_tensor.tensor_type), dtype=np.float32).reshape(t.shape[1:])
    return src.float32(name)[e]
moe = np.zeros(H, np.float32)
for e, we in zip(top, w):
    gt = expert(b + "ffn_gate_exps.weight", e) @ cur
    ut = expert(b + "ffn_up_exps.weight", e) @ cur
    hmid = gt / (1 + np.exp(-gt)) * ut
    moe += we * (expert(b + "ffn_down_exps.weight", e) @ hmid)
print("ffn_moe_out", f(moe[:3]), f(moe[-3:]), "| llama [0.0633 -0.0325 0.0310 ... -0.0510 0.0425 0.0199]")
sg = src.float32(b + "ffn_gate_shexp.weight") @ cur; su = src.float32(b + "ffn_up_shexp.weight") @ cur
sh = src.float32(b + "ffn_down_shexp.weight") @ (sg / (1 + np.exp(-sg)) * su)
sgate = 1 / (1 + np.exp(-(src.float32(b + "ffn_gate_inp_shexp.weight").reshape(-1) @ cur)))
shg = sh * sgate
print("shared gate", f(sgate), "ffn_shexp_gated", f(shg[:3]), f(shg[-3:]), "| llama [-0.0107 0.0086 0.0187 ... 0.0172 0.0112 0.0055]")
ffn = moe + shg
print("ffn_out", f(ffn[:3]), f(ffn[-3:]), "| llama [0.0526 -0.0239 0.0497 ... -0.0338 0.0537 0.0255]")
x2 = x1 + np.repeat(2 / (1 + np.exp(-inject1 / hc)), H) * np.tile(ffn, hc)
print("l_last-0 stream0", f(x2[:3]), "| llama [-0.0062 0.0043 -0.0023]?? (that was hc_combine) ; engine f1_layer1 col0 first3 should equal this")
print("engine-order ffn inject gates", f(2 / (1 + np.exp(-inject1 / hc))))
