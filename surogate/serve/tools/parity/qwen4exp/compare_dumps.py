import os
# Per-layer statistics of the engine's residual dumps (SUROGATE_SERVE_DUMP_RESIDUAL) for
# comparison with llama.cpp's eval-callback sums (l_last-<il>, hc_init, result_norm).
import glob, os, struct, sys
import numpy as np
d = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SUROGATE_PARITY_DIR", ".") + "/dumps"
fwd = sys.argv[2] if len(sys.argv) > 2 else "f1"
def load(path):
    with open(path, "rb") as f:
        magic, rows, cols, forward = struct.unpack("<iiii", f.read(16))
        raw = np.frombuffer(f.read(), dtype=np.uint16)
    if raw.size == rows * cols:
        x = (raw.astype(np.uint32) << 16).view(np.float32).reshape(cols, rows)  # [T, rows]
    else:  # FP32 payload
        x = raw.view(np.float32).reshape(cols, rows)
    return x
files = sorted(glob.glob(f"{d}/{fwd}_layer*.bin"), key=lambda p: int(p.split("layer")[-1].split(".")[0])) + sorted(glob.glob(f"{d}/{fwd}_L*_*.bin"))
for p in files + sorted(glob.glob(f"{d}/{fwd}_final.bin")):
    x = load(p); name = os.path.basename(p)[:-4]
    T, R = x.shape
    if R % 2560 == 0 and R > 2560:
        s = x.reshape(T, R // 2560, 2560)
        per_stream = " ".join(f"{np.abs(s[:, i]).mean():.4f}" for i in range(R // 2560))
    else:
        per_stream = "-"
    print(f"{name:14s} T={T:3d} R={R:5d} sum={x.sum():+.4e} mean|x|={np.abs(x).mean():.4f} max|x|={np.abs(x).max():.2f} stream mean|x| [{per_stream}] col0={np.round(x[0, :3], 4).tolist()}..{np.round(x[0, -3:], 4).tolist()} col1={np.round(x[1, :3], 4).tolist()}")
