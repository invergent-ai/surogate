# Decode chosen row spans of a W8 artifact tensor and compare with the un-tiled GGUF reference.
import numpy as np, torch
import argparse
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("artifact", type=Path)
parser.add_argument("gguf", type=Path, help="first source GGUF shard")
args = parser.parse_args()

from surogate.serve.convert.qwen4exp import inventory as inv, convert as cv
from surogate.serve.artifact.container import Artifact
from surogate.serve.artifact.layouts import split_row_planes, dequantize_row_split, row_split_geometry
from surogate.serve.artifact.numeric import get_format
art = Artifact.open(args.artifact)
src = cv.GgufSource(args.gguf)
g = inv.geometry_from_gguf(src, token_domain=art.geometry["token_domain"])
spec_by = {s.name: s for s in inv.build_text_core_specs(g)}
if not g.gdn_layers:
    raise SystemExit("checkpoint has no GDN layers")
layer = g.gdn_layers[0]
name = f"text/layers/{layer}/gdn/query_key_value_z"
qkv = src.float32(f"blk.{layer}.attn_qkv.weight"); z = src.float32(f"blk.{layer}.attn_gate.weight")
print("gguf types:", src.tensor(f"blk.{layer}.attn_qkv.weight").type_name, src.tensor(f"blk.{layer}.attn_gate.weight").type_name)
ref = np.concatenate([qkv[:2 * g.key_dim], cv._untile_v(g, qkv[2 * g.key_dim:], 0), cv._untile_v(g, z, 0)])
spec = spec_by[name]; obj = art.find(name)
if obj.runs or obj.format != inv.W8:
    raise SystemExit("check_rows requires an inline W8 parent; use verify_flash_artifact for native formats")
payload = bytes(art.payload(obj))
geometry = row_split_geometry(get_format(inv.W8), spec.shape)
count = min(128, g.value_dim, 2 * g.key_dim)
for begin in sorted({0, max(0, 2 * g.key_dim - count), 2 * g.key_dim,
                     max(0, g.convolution_dim - count), g.convolution_dim, spec.shape[0] - count}):
    planes = split_row_planes(payload, geometry, begin, count)
    got = dequantize_row_split(planes, inv.W8, (count, spec.shape[1]), dtype=torch.float32).cpu().numpy()
    r = ref[begin:begin + count]
    rel = np.linalg.norm(got - r) / np.linalg.norm(r)
    print(f"rows {begin:5d}-{begin+count-1:5d}: rel-L2 {rel:.4f}  |got|/|ref| {np.linalg.norm(got)/np.linalg.norm(r):.3f}  got[0,:3]={got[0,:3]}  ref[0,:3]={r[0,:3]}")
# also: raw tiled order (no untile) for the v rows — does the artifact hold tiled rows?
ref_tiled = np.concatenate([qkv, z])
for begin in (2 * g.key_dim, g.convolution_dim - count):
    planes = split_row_planes(payload, geometry, begin, count)
    got = dequantize_row_split(planes, inv.W8, (count, spec.shape[1]), dtype=torch.float32).cpu().numpy()
    r = ref_tiled[begin:begin + count]
    print(f"vs TILED rows {begin}: rel-L2 {np.linalg.norm(got - r)/np.linalg.norm(r):.4f}")

art.close()
src.close()
