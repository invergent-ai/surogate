# The family's conv op reads gdn/convolution channel-major ([c][tap]); the converter had written
# tap-major bytes. Rewrite the 36 objects in place from the GGUF (value channels un-tiled).
import numpy as np, sys
from surogate.serve.tools.convert.qwen4exp import inventory as inv, convert as cv
from surogate.serve.tools.artifact.container import Artifact
from surogate.serve.tools.artifact.layouts import decode_direct
PATH = "/home/densemax2/work/models/ninfer/qwen3_8_flash_next.ninfer"
src = cv.GgufSource("models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
art = Artifact.open(PATH)
shape = (inv.GDN_CONV_KERNEL, inv.GDN_CONV_DIM)
def expected(layer):
    conv = src.float32(f"blk.{layer}.ssm_conv1d.weight")  # (10240, 4)
    return np.ascontiguousarray(np.concatenate([conv[:4096], cv._untile_v(conv[4096:], 0)], axis=0).T)  # (4, 10240) tap-major
edits, skipped = [], 0
for layer in range(inv.LAYERS):
    if layer in inv.FULL_ATTENTION_LAYERS: continue
    name = f"text/layers/{layer}/gdn/convolution"
    obj = art.find(name)
    want = cv.bf16_bytes(expected(layer), shape)
    assert len(want) == obj.bytes == 4 * 10240 * 2, (len(want), obj.bytes)
    if bytes(art.payload(obj)) == want:
        skipped += 1; continue
    edits.append((art.payload_offset + obj.offset, want, name))
art.close()
with open(PATH, "r+b") as f:
    for off, raw, name in edits:
        f.seek(off); f.write(raw)
print(f"patched {len(edits)} objects, {skipped} already tap-major")
art = Artifact.open(PATH); worst = 0.0
for layer in range(inv.LAYERS):
    if layer in inv.FULL_ATTENTION_LAYERS: continue
    obj = art.find(f"text/layers/{layer}/gdn/convolution")
    got = decode_direct(bytes(art.payload(obj)), inv.BF16, shape).float().numpy().reshape(4, 10240)
    worst = max(worst, float(np.abs(got - expected(layer)).max()))
print(f"re-verified 36 layers: max |bf16(conv) - conv| = {worst:.3e}")
