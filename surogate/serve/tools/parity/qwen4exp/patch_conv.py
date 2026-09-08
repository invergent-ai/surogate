# The family's conv op reads gdn/convolution channel-major ([c][tap]); the converter had written
# tap-major bytes. Rewrite the checkpoint's inline convolution objects in place from the GGUF.
import numpy as np
import argparse
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("artifact", type=Path)
parser.add_argument("gguf", type=Path, help="first source GGUF shard")
args = parser.parse_args()

from surogate.serve.convert.qwen4exp import inventory as inv, convert as cv
from surogate.serve.artifact.container import Artifact
from surogate.serve.artifact.layouts import decode_direct
PATH = args.artifact
src = cv.GgufSource(args.gguf)
art = Artifact.open(PATH)
g = inv.geometry_from_gguf(src, token_domain=art.geometry["token_domain"])
shape = (g.gdn_conv_kernel, g.convolution_dim)
def expected(layer):
    conv = src.float32(f"blk.{layer}.ssm_conv1d.weight")  # (10240, 4)
    return np.ascontiguousarray(np.concatenate([conv[:2 * g.key_dim], cv._untile_v(g, conv[2 * g.key_dim:], 0)], axis=0).T)  # (4, 10240) tap-major
edits, skipped = [], 0
for layer in g.gdn_layers:
    name = f"text/layers/{layer}/gdn/convolution"
    obj = art.find(name)
    if obj.runs or tuple(obj.shape) != shape:
        raise ValueError(f"{name}: expected an inline convolution with shape {shape}")
    want = cv.bf16_bytes(expected(layer), shape)
    assert len(want) == obj.bytes == g.gdn_conv_kernel * g.convolution_dim * 2, (len(want), obj.bytes)
    if bytes(art.payload(obj)) == want:
        skipped += 1; continue
    edits.append((art.payload_offset + obj.offset, want, name))
art.close()
with open(PATH, "r+b") as f:
    for off, raw, name in edits:
        f.seek(off); f.write(raw)
print(f"patched {len(edits)} objects, {skipped} already tap-major")
art = Artifact.open(PATH); worst = 0.0
for layer in g.gdn_layers:
    obj = art.find(f"text/layers/{layer}/gdn/convolution")
    got = decode_direct(bytes(art.payload(obj)), inv.BF16, shape).float().numpy().reshape(shape)
    worst = max(worst, float(np.abs(got - expected(layer)).max()))
print(f"re-verified {len(g.gdn_layers)} layers: max |bf16(conv) - conv| = {worst:.3e}")

art.close()
src.close()
