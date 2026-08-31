# Decode chosen row spans of a W8 artifact tensor and compare with the un-tiled GGUF reference.
import numpy as np, torch, sys
from surogate.serve.tools.convert.qwen4exp import inventory as inv, convert as cv
from surogate.serve.tools.artifact.container import Artifact
from surogate.serve.tools.artifact.layouts import split_row_planes, dequantize_row_split, row_split_geometry
from surogate.serve.tools.artifact.numeric import get_format
art = Artifact.open("/home/densemax2/work/models/sinfer/qwen3_8_flash_next.sinfer")
src = cv.GgufSource("models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
spec_by = {s.name: s for s in inv.TENSOR_SPECS}
name = "text/layers/0/gdn/query_key_value_z"
qkv = src.float32("blk.0.attn_qkv.weight"); z = src.float32("blk.0.attn_gate.weight")
print("gguf types:", src.tensor("blk.0.attn_qkv.weight").type_name, src.tensor("blk.0.attn_gate.weight").type_name)
ref = np.concatenate([qkv[:4096], cv._untile_v(qkv[4096:], 0), cv._untile_v(z, 0)])
spec = spec_by[name]; obj = art.find(name); payload = bytes(art.payload(obj))
geometry = row_split_geometry(get_format(inv.W8), spec.shape)
for begin, count in [(0, 128), (4000, 128), (4096, 128), (4224, 128), (7000, 128), (10112, 128), (10240, 128), (16256, 128)]:
    planes = split_row_planes(payload, geometry, begin, count)
    got = dequantize_row_split(planes, inv.W8, (count, spec.shape[1]), dtype=torch.float32).cpu().numpy()
    r = ref[begin:begin + count]
    rel = np.linalg.norm(got - r) / np.linalg.norm(r)
    print(f"rows {begin:5d}-{begin+count-1:5d}: rel-L2 {rel:.4f}  |got|/|ref| {np.linalg.norm(got)/np.linalg.norm(r):.3f}  got[0,:3]={got[0,:3]}  ref[0,:3]={r[0,:3]}")
# also: raw tiled order (no untile) for the v rows — does the artifact hold tiled rows?
ref_tiled = np.concatenate([qkv, z])
for begin, count in [(4096, 128), (7000, 128)]:
    planes = split_row_planes(payload, geometry, begin, count)
    got = dequantize_row_split(planes, inv.W8, (count, spec.shape[1]), dtype=torch.float32).cpu().numpy()
    r = ref_tiled[begin:begin + count]
    print(f"vs TILED rows {begin}: rel-L2 {np.linalg.norm(got - r)/np.linalg.norm(r):.4f}")
