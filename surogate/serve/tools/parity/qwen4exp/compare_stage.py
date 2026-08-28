# compare_stage.py <dump.bin> <ref.npy> [block]  — token-0 column of an engine dump vs a CPU reference vector,
# with per-block (head) statistics: block = elements per group (e.g. 128 for GDN heads).
import sys, struct, numpy as np
def load(path):
    with open(path, "rb") as fh:
        magic, rows, cols, fwd = struct.unpack("<IiiI", fh.read(16)); raw = fh.read()
    if len(raw) == rows * cols * 2:
        a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        a = np.frombuffer(raw, dtype=np.float32)
    return a.reshape(cols, rows)   # column-major [rows fastest]: token t -> a[t]
eng = load(sys.argv[1])[0]; ref = np.load(sys.argv[2]); blk = int(sys.argv[3]) if len(sys.argv) > 3 else 128
assert eng.shape == ref.shape, (eng.shape, ref.shape)
err = np.abs(eng - ref); print(f"n={len(ref)} max|ref|={np.abs(ref).max():.4f} max|err|={err.max():.4f} rel-L2={np.linalg.norm(eng-ref)/max(np.linalg.norm(ref),1e-12):.4f}")
nb = len(ref) // blk
for i in range(nb):
    e, r = eng[i*blk:(i+1)*blk], ref[i*blk:(i+1)*blk]
    rl2 = np.linalg.norm(e - r) / max(np.linalg.norm(r), 1e-12)
    cos = float(e @ r) / max(np.linalg.norm(e) * np.linalg.norm(r), 1e-12)
    scale = float(np.linalg.norm(e)) / max(np.linalg.norm(r), 1e-12)
    flag = "" if rl2 < 0.05 else "  <-- MISMATCH"
    if nb <= 48 or flag:
        print(f"block {i:3d}: rel-L2 {rl2:.3f} cos {cos:+.3f} |eng|/|ref| {scale:.3f}{flag}")
