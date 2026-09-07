import os
# CPU reference for layer-0 GDN block of token 0 (llama.cpp algebra, tiled V pairing v-head h <-> k-head h % 16).
import sys, numpy as np, subprocess
SC = os.environ.get("SUROGATE_PARITY_DIR", ".")
from surogate.serve.convert.qwen4exp import convert as cv
gguf = "models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
src = cv.GgufSource(gguf)
tok0 = 248045
emb = src.float32("token_embd.weight")[tok0].astype(np.float32)
hc, H, eps = 4, 2560, 1e-6
b = "blk.0."
f = lambda v: np.array2string(np.asarray(v), precision=4, suppress_small=True)
# --- hyper-connection mix (validated against llama.cpp) ---
norm = src.float32(b + "hc_attn_norm.weight").reshape(hc * H)
down = src.float32(b + "hc_attn_down.weight"); up = src.float32(b + "hc_attn_up.weight")
x = np.tile(emb, hc)
s = x.reshape(hc, H); xn = (s / np.sqrt((s * s).mean(axis=1, keepdims=True) + eps)).reshape(hc * H) * norm
lo = down @ xn / hc; lo = lo / (1 + np.exp(-lo)); gate = 1 / (1 + np.exp(-(up @ lo)))
mixed = (xn * gate).reshape(hc, H).mean(axis=0)
print("mixed", f(mixed[:3]), f(mixed[-3:]))
# --- GDN ---
Wqkv = src.float32(b + "attn_qkv.weight"); Wz = src.float32(b + "attn_gate.weight")
print("Wqkv", Wqkv.shape, "Wz", Wz.shape)
qkv = Wqkv @ mixed; z = Wz @ mixed
print("qkv_mixed", f(qkv[:3]), f(qkv[-3:]), "| llama [-1.2015 1.4350 0.6142 ... -0.3388 -0.1232 -1.1096]")
print("z", f(z[:3]), f(z[-3:]), "| llama [2.1988 0.8234 0.1450 ... -1.9228 -0.2116 -1.4014]")
conv = src.float32(b + "ssm_conv1d.weight"); print("conv", conv.shape)
try:
    cbias = src.float32(b + "ssm_conv1d.bias"); print("conv bias present", cbias.shape)
except Exception as e:
    cbias = None; print("no conv bias")
K = conv.shape[-1] if conv.shape[0] == 10240 else conv.shape[0]
w_last = conv[:, -1] if conv.shape[0] == 10240 else conv[-1, :]
c = w_last * qkv + (cbias if cbias is not None else 0.0)
c = c / (1 + np.exp(-c))  # silu
print("conv_output_silu", f(c[:3]), f(c[-3:]))
q = c[:2048].reshape(16, 128); k = c[2048:4096].reshape(16, 128); v = c[4096:].reshape(48, 128)
l2 = lambda a: a / np.sqrt((a * a).sum(axis=-1, keepdims=True) + eps)
qn, kn = l2(q), l2(k)
print("q_conv (raw) head0", f(q[0, :3]), "| llama q_conv-0 [0.0089 0.0296 -0.0140 ...]")
beta = 1 / (1 + np.exp(-(src.float32(b + "ssm_beta.weight") @ mixed)))
alpha = src.float32(b + "ssm_alpha.weight") @ mixed
dt = src.float32(b + "ssm_dt.bias").reshape(48); A = src.float32(b + "ssm_a").reshape(48)
sp = np.log1p(np.exp(alpha + dt)); g = sp * A
print("alpha", f(alpha[:3]), "| llama [1.0564 -0.0000 2.0229 ...]")
print("beta_sigmoid[0]", f(beta[:2]), "| llama 0.6470 ; gate", f(g[:3]), "| llama [-1.7626 -49.2914 -4.7524]")
scale = 1 / np.sqrt(128)
for name, pair in (("tiled h%16", lambda h: h % 16), ("interleave h//3", lambda h: h // 3)):
    o = np.stack([beta[h] * float(kn[pair(h)] @ qn[pair(h)]) * scale * v[h] for h in range(48)])
    print(name, "attn_output head0", f(o[0, :4]), "| llama [0.0002 0.0001 -0.0001 ...]")
    gn = src.float32(b + "ssm_norm.weight").reshape(128)
    on = o / np.sqrt((o * o).mean(axis=-1, keepdims=True) + eps) * gn
    fin = (on.reshape(-1) * (1 / (1 + np.exp(-z))))
    print(name, "final_output", f(fin[:3]), f(fin[-3:]), "| llama [0.1579 0.0693 -0.0300 ... 0.0002 0.0006 0.0078]")
    out = src.float32(b + "ssm_out.weight") @ fin
    print(name, "linear_attn_out", f(out[:3]), f(out[-3:]), "| llama [-0.0390 0.0326 -0.1039 ... -0.0458 0.0117 0.2767]")

# ---- engine-comparable values (HF grouped head order: HF head h' = kh*3+g <-> tiled head g*16+kh) ----
print("=== engine order (compare_dumps col0 first3 .. last3) ===")
tiled_of = lambda hp: (hp % 3) * 16 + hp // 3
perm = [tiled_of(h) for h in range(48)]              # engine head h' -> tiled head
z_e = z.reshape(48, 128)[perm].reshape(-1)
v_e = v[perm]
print("gdn_fused  first3", f(qkv[:3]), "last3", f(z_e[-3:]))
c_e = np.concatenate([c[:4096], v_e.reshape(-1)])
print("gdn_conv   first3", f(c_e[:3]), "last3", f(c_e[-3:]))
print("gdn_ab     first3 (alpha HF heads 0,1,2)", f(alpha[perm][:3]), "last3 (beta logits HF 45..47)", f((src.float32(b + "ssm_beta.weight") @ mixed)[perm][-3:]))
print("gdn_g      first3", f(g[perm][:3]), "last3", f(g[perm][-3:]))
print("gdn_beta   first3", f(beta[perm][:3]), "last3", f(beta[perm][-3:]))
o_t = np.stack([beta[h] * float(kn[h % 16] @ qn[h % 16]) * scale * v[h] for h in range(48)])
gn = src.float32(b + "ssm_norm.weight").reshape(128)
on_t = o_t / np.sqrt((o_t * o_t).mean(axis=-1, keepdims=True) + eps) * gn
fin_t = on_t * (1 / (1 + np.exp(-z.reshape(48, 128))))
fin_e = fin_t[perm].reshape(-1)
print("gdn_final  first3", f(fin_e[:3]), "last3", f(fin_e[-3:]))
out_t = src.float32(b + "ssm_out.weight") @ fin_t.reshape(-1)
print("gdn_out    first3", f(out_t[:3]), "last3", f(out_t[-3:]))
print("k.q per k-head", f(np.array([float(kn[i] @ qn[i]) for i in range(16)])))

# ---- save engine-order reference vectors for full comparisons ----
np.save(f"{SC}/ref_gdn_fused.npy", np.concatenate([qkv[:4096], qkv[4096:].reshape(48, 128)[perm].reshape(-1), z_e]).astype(np.float32))  # projection rows (v un-tiled), not the conv output
np.save(f"{SC}/ref_gdn_conv.npy", c_e.astype(np.float32))
np.save(f"{SC}/ref_gdn_o.npy", o_t[perm].reshape(-1).astype(np.float32))
np.save(f"{SC}/ref_gdn_final.npy", fin_e.astype(np.float32))
np.save(f"{SC}/ref_gdn_out.npy", out_t.astype(np.float32))
np.save(f"{SC}/ref_mixer_mixed.npy", mixed.astype(np.float32))
print("saved reference npy files")
