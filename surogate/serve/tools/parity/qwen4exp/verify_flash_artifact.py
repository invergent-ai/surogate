# Decode objects of the finished qwen3_8_flash_next.sinfer and compare against the GGUF
# (gguf-py dequantisation with the converter's own algebra: V-head un-tiling, a_log, row order).
#
# Most of this artifact is an index rather than a copy, so most objects are checked by reading
# the stretches of the GGUF their runs name and decoding those -- which is the same thing the
# engine does at load, one layer below the kernels.
import sys, numpy as np, torch
from gguf import GGMLQuantizationType
from gguf.quants import dequantize
from surogate.serve.convert.qwen4exp import inventory as inv, convert as cv, recipe as rcp
from surogate.serve.artifact.container import Artifact
from surogate.serve.artifact.layouts import encoded_size, split_row_planes, dequantize_row_split, decode_direct, row_split_geometry
from surogate.serve.artifact.numeric import get_format
from surogate.serve.convert.common.gguf_repack import REPACKABLE_TYPES

art = Artifact.open(sys.argv[1] if len(sys.argv) > 1
                    else "/home/densemax2/work/models/sinfer/qwen3_8_flash_next.sinfer")
src = cv.GgufSource("models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
spec_by = {s.name: s for s in inv.TEXT_CORE_TENSOR_SPECS}
external = [np.memmap(path, dtype=np.uint8, mode="r") for path, _ in art.external]
bad = 0


def run_rows(obj, row_bytes):
    """The object's stored bytes, gathered from the GGUF where the runs point."""
    data = b"".join(bytes(external[s - 1][o:o + b]) for s, o, b in obj.runs)
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, row_bytes)


def decode(obj, spec):
    """The object as float32, however the artifact holds it."""
    if obj.transform == "q8_0-to-w8g32":
        # Q8_0 blocks the loader rearranges into W8 planes on the device; a column permutation
        # rides along as the group map.
        rows = run_rows(obj, obj.shape[1] // 32 * 34)
        codes, scales = REPACKABLE_TYPES["Q8_0"][1](rows.reshape(obj.shape[0], -1, 34))
        values = (codes.astype(np.float32)
                  * scales.astype(np.float32)[:, :, None]).reshape(obj.shape)
        if obj.group_map:
            values = np.concatenate([values[:, g * 32:(g + 1) * 32] for g in obj.group_map], 1)
        return values
    if obj.layout == "ggml-blocks-v1":
        block = 256 if obj.format.endswith("_K") else 32
        width = {"Q4_K": 144, "Q5_K": 176, "Q6_K": 210, "Q8_0": 34, "Q5_1": 24, "IQ4_NL": 18}
        rows = run_rows(obj, obj.shape[1] // block * width[obj.format])
        return np.asarray(dequantize(rows, GGMLQuantizationType[obj.format]),
                          dtype=np.float32).reshape(obj.shape)
    payload = bytes(art.payload(obj))
    if spec.format == inv.W8:
        geometry = row_split_geometry(get_format(inv.W8), spec.shape)
        planes = split_row_planes(payload, geometry, 0, spec.shape[0])
        return dequantize_row_split(planes, inv.W8, spec.shape, dtype=torch.float32).cpu().numpy()
    return decode_direct(payload, spec.format, spec.shape).float().cpu().numpy()


def check(name, reference=None, rows=None, tol=None):
    global bad
    spec = spec_by[name]
    obj = art.find(name)
    where = f"gguf x{len(obj.runs)}" if obj.runs else "artifact"
    ok = obj.bytes == encoded_size(spec.layout if not obj.runs else obj.layout,
                                   obj.format, tuple(obj.shape))
    line = f"  {name:48s} {obj.bytes / 1e6:9.1f} MB {where:>10s} {'OK ' if ok else 'BAD'}"
    if reference is not None:
        got = decode(obj, spec)
        spans = rows if rows is not None else [(0, spec.shape[0])]
        got = np.concatenate([got[b:b + c] for b, c in spans]) if got.ndim > 1 else got
        ref = np.concatenate([reference[b:b + c] for b, c in spans]) if got.ndim > 1 else reference
        rel = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30))
        line += f"  rel_l2={rel:.2e}"
        if tol is not None and rel > tol:
            ok = False
            line += "  << OVER TOLERANCE"
    if not ok:
        bad += 1
    print(line, flush=True)


for layer in (3, 23, 47):
    b = f"blk.{layer}."
    q = src.float32(b + "attn_q.weight"); k = src.float32(b + "attn_k.weight"); v = src.float32(b + "attn_v.weight")
    qr, _, gr, _ = cv.attention_rows_q_k_gate_v()
    check(f"text/layers/{layer}/attention/query_key_gate_value", np.concatenate([q[qr], k, q[gr], v]), tol=1e-6)
    check(f"text/layers/{layer}/attention/output", src.float32(b + "attn_output.weight"), tol=1e-6)
    check(f"text/layers/{layer}/attention/query_norm", src.float32(b + "attn_q_norm.weight") - 1.0, tol=1e-2)
    check(f"text/layers/{layer}/attention/key_norm", src.float32(b + "attn_k_norm.weight") - 1.0, tol=1e-2)
    check(f"text/layers/{layer}/attention/indexer/query", src.float32(b + "indexer.q_proj.weight"), tol=1e-6)
for layer in (0, 1, 22, 46):
    b = f"blk.{layer}."
    qkv = src.float32(b + "attn_qkv.weight"); z = src.float32(b + "attn_gate.weight")
    ref = np.concatenate([qkv[:4096], cv._untile_v(qkv[4096:], 0), cv._untile_v(z, 0)])
    check(f"text/layers/{layer}/gdn/query_key_value_z", ref, tol=1e-6)
    check(f"text/layers/{layer}/gdn/output", cv._untile_v(src.float32(b + "ssm_out.weight"), 1), tol=1e-6)
    check(f"text/layers/{layer}/gdn/a_log", np.log(-cv._untile_v_heads(src.float32(b + "ssm_a").reshape(48), 0)), tol=1e-6)
    check(f"text/layers/{layer}/gdn/dt_bias", cv._untile_v_heads(src.float32(b + "ssm_dt.bias").reshape(48), 0), tol=1e-6)
    check(f"text/layers/{layer}/gdn/norm", src.float32(b + "ssm_norm.weight"), tol=1e-2)
    alpha = cv._untile_v_heads(src.float32(b + "ssm_alpha.weight"), 0)
    beta = cv._untile_v_heads(src.float32(b + "ssm_beta.weight"), 0)
    check(f"text/layers/{layer}/gdn/a_b_projection", np.concatenate([alpha, beta]), tol=1e-2)
    conv = src.float32(b + "ssm_conv1d.weight")  # (10240, 4) channel-major, as ggml holds it
    check(f"text/layers/{layer}/gdn/convolution", np.ascontiguousarray(np.concatenate([conv[:4096], cv._untile_v(conv[4096:], 0)], axis=0).T), tol=1e-2)
    check(f"text/layers/{layer}/hc_attn/norm", src.float32(b + "hc_attn_norm.weight"), tol=1e-6)
    check(f"text/layers/{layer}/hc_attn/down", src.float32(b + "hc_attn_down.weight"), tol=1e-2)
    check(f"text/layers/{layer}/hc_attn/up", src.float32(b + "hc_attn_up.weight"), tol=1e-2)
    check(f"text/layers/{layer}/hc_attn/inject", src.float32(b + "hc_attn_inject.weight"), tol=1e-2)
    check(f"text/layers/{layer}/hc_ffn/norm", src.float32(b + "hc_ffn_norm.weight"), tol=1e-6)
    check(f"text/layers/{layer}/hc_ffn/up", src.float32(b + "hc_ffn_up.weight"), tol=1e-2)
    check(f"text/layers/{layer}/mlp/router_shared_gate", np.concatenate([src.float32(b + "ffn_gate_inp.weight"), src.float32(b + "ffn_gate_inp_shexp.weight").reshape(1, -1)]), tol=1e-2)
    check(f"text/layers/{layer}/mlp/shared_gate_up", np.concatenate([src.float32(b + "ffn_gate_shexp.weight"), src.float32(b + "ffn_up_shexp.weight")]), tol=1e-6)
    check(f"text/layers/{layer}/mlp/shared_down", src.float32(b + "ffn_down_shexp.weight"), tol=1e-6)
    gate = src.float32(b + "ffn_gate_exps.weight"); up = src.float32(b + "ffn_up_exps.weight")
    fused = np.concatenate([gate, up], axis=1).reshape(-1, 2560); del gate, up
    check(f"text/layers/{layer}/mlp/routed_gate_up", fused, rows=[(0, 64), (1270, 20), (100 * 1280, 64), (511 * 1280 + 1200, 80)], tol=1e-6)
    del fused
    down = src.float32(b + "ffn_down_exps.weight").reshape(-1, 640)
    check(f"text/layers/{layer}/mlp/routed_down", down, rows=[(0, 64), (2550, 20), (300 * 2560, 64), (511 * 2560 + 2496, 64)], tol=1e-6)
    del down
b = "blk.1."
check("text/layers/1/ple/key", src.float32(b + "ple_key.weight"), tol=1e-2)
check("text/layers/1/ple/value", src.float32(b + "ple_value.weight"), tol=1e-2)
check("text/layers/1/ple/norm_key", src.float32(b + "ple_norm_key.weight"), tol=1e-6)
check("text/layers/1/ple/norm_conv", src.float32(b + "ple_norm_conv.weight"), tol=1e-6)
check("text/layers/1/ple/convolution", src.float32(b + "ple_conv1d.weight").T, tol=1e-2)
check("text/output_hc/norm", src.float32("output_hc_norm.weight"), tol=1e-6)
check("text/output_hc/down", src.float32("output_hc_down.weight"), tol=1e-2)
check("text/output_head", src.float32("output.weight"), rows=[(0, 256), (248044, 64), (248320 - 256, 256)], tol=1e-6)
check("text/token_embedding", src.float32("token_embd.weight"), rows=[(0, 256), (248320 - 256, 256)], tol=1e-6)
check("text/ple/multipliers", None); check("text/ple/head_offsets", None); check("text/ple/head_vocab_sizes", None)
# PLE table: one run over the GGUF, so the check is that the run names exactly the tensor.
table = art.find(inv.PLE_TABLE_RESOURCE)
tensor = src.tensor(rcp.PLE_TABLE_SOURCE)
same = table.runs == ((tensor.shard, tensor.offset, tensor.nbytes),)
bad += 0 if same else 1
print(f"  {inv.PLE_TABLE_RESOURCE:48s} {table.bytes / 1e9:9.1f} GB  gguf shard {tensor.shard} "
      f"{'exact run' if same else 'MISMATCH'}", flush=True)
print("VERIFY_DONE bad=%d" % bad, flush=True)
