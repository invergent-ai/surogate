# CPU reference for layer-0 attention-side hyper-connection mix of token 0 (llama.cpp algebra).
import sys, numpy as np, subprocess
from surogate.serve.convert.qwen4exp import convert as cv
gguf = "models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
prompt = open(sys.argv[1]).read()
ids = subprocess.run(["study/llama.cpp-master/build/bin/llama-tokenize", "-m", gguf, "-p", prompt, "--ids", "--log-disable"],
                     capture_output=True, text=True).stdout.strip().splitlines()[-1]
ids = [int(t) for t in ids.strip("[]").split(",")]
print("ids", ids[:6], "...", len(ids))
src = cv.GgufSource(gguf)
emb = src.float32("token_embd.weight")
print("token_embd", emb.shape)
x0 = emb[ids[0]].astype(np.float32)
print("emb token0 first3", x0[:3], "last3", x0[-3:], "sum4", 4 * x0.sum())
hc, H, eps = 4, 2560, 1e-6
b = "blk.0."
norm = src.float32(b + "hc_attn_norm.weight").reshape(hc * H)
down = src.float32(b + "hc_attn_down.weight")   # (320, 10240)
up = src.float32(b + "hc_attn_up.weight")       # (10240, 320)
inj = src.float32(b + "hc_attn_inject.weight")  # (4, 10240)
print("shapes", norm.shape, down.shape, up.shape, inj.shape)
x = np.tile(x0, hc)                              # hc_init: streams = embedding
def stream_rms(v):
    s = v.reshape(hc, H)
    return (s / np.sqrt((s * s).mean(axis=1, keepdims=True) + eps)).reshape(hc * H)
xn = stream_rms(x) * norm
lo = down @ xn / hc
lo = lo / (1 + np.exp(-lo))                     # silu
gate = 1 / (1 + np.exp(-(up @ lo)))
mixed = (xn * gate).reshape(hc, H).mean(axis=0)
inject = inj @ xn
f = lambda v: np.array2string(v, precision=4, suppress_small=True)
print("V0 llama   mixed first3", f(mixed[:3]), "last3", f(mixed[-3:]), "sum %.1f" % mixed.sum())
print("V0 inject logits", f(inject), " gates 2sig(l/4)", f(2 / (1 + np.exp(-inject / hc))))
# variants that a wrong kernel could produce
lo1 = down @ xn; lo1 = lo1 / (1 + np.exp(-lo1)); g1 = 1 / (1 + np.exp(-(up @ lo1)))
print("V1 no-1/hc mixed first3", f(((xn * g1).reshape(hc, H).mean(axis=0))[:3]))
print("V2 gate*x  mixed first3", f(((x * gate).reshape(hc, H).mean(axis=0))[:3]))
print("V3 sum     mixed first3", f(((xn * gate).reshape(hc, H).sum(axis=0))[:3]))
print("V4 no-gamma mixed first3", f(((stream_rms(x) * gate).reshape(hc, H).mean(axis=0))[:3]))
xn5 = x / np.sqrt((x * x).mean() + eps) * norm
print("V5 rms-over-all mixed first3", f(((xn5 * gate).reshape(hc, H).mean(axis=0))[:3]))
print("xn first3", f(xn[:3]), "gate first3", f(gate[:3]), "gate stream means", f(gate.reshape(hc, H).mean(axis=1)))
