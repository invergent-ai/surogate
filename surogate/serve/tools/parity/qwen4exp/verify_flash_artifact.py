# Decode objects of the finished qwen3_8_flash_next.ninfer and compare against the GGUF
# (gguf-py dequantisation with the converter's own algebra: V-head un-tiling, a_log, row order).
import sys, time, numpy as np, torch
from surogate.serve.tools.convert.qwen4exp import inventory as inv, convert as cv
from surogate.serve.tools.artifact.container import Artifact
from surogate.serve.tools.artifact.layouts import encoded_size, split_row_planes, dequantize_row_split, decode_direct, row_split_geometry
from surogate.serve.tools.artifact.numeric import get_format

art = Artifact.open("/home/densemax2/work/models/ninfer/qwen3_8_flash_next.ninfer")
src = cv.GgufSource("models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
spec_by = {s.name: s for s in inv.TENSOR_SPECS}
bad = 0

def check(name, reference=None, rows=None, tol=None):
    global bad
    spec = spec_by[name]
    obj = art.find(name)
    payload = art.payload(obj)
    expect = encoded_size(spec.layout, spec.format, spec.shape)
    ok = len(payload) == expect
    line = f"  {name:48s} {len(payload)/1e6:9.1f} MB {'OK ' if ok else 'BAD'}"
    if reference is not None:
        if spec.format == inv.W8:
            geometry = row_split_geometry(get_format(inv.W8), spec.shape)
            spans = rows if rows is not None else [(0, min(512, spec.shape[0])), (spec.shape[0] - 512, 512)]
            gots, refs = [], []
            for begin, count in spans:
                planes = split_row_planes(bytes(payload), geometry, begin, count)
                gots.append(dequantize_row_split(planes, inv.W8, (count, spec.shape[1]), dtype=torch.float32).cpu().numpy())
                refs.append(reference[begin:begin + count])
            got = np.concatenate(gots); ref = np.concatenate(refs)
        else:
            got = decode_direct(bytes(payload), spec.format, spec.shape).float().cpu().numpy(); ref = reference
        rel = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30))
        line += f"  rel_l2={rel:.2e}"
        if tol is not None and rel > tol:
            ok = False; line += "  << OVER TOLERANCE"
    if not ok: bad += 1
    print(line, flush=True)

for layer in (3, 23, 47):
    b = f"blk.{layer}."
    q = src.float32(b + "attn_q.weight"); k = src.float32(b + "attn_k.weight"); v = src.float32(b + "attn_v.weight")
    qr, _, gr, _ = cv.attention_rows_q_k_gate_v()
    check(f"text/layers/{layer}/attention/query_key_gate_value", np.concatenate([q[qr], k, q[gr], v]), tol=1e-2)
    check(f"text/layers/{layer}/attention/output", src.float32(b + "attn_output.weight"), tol=1e-2)
    check(f"text/layers/{layer}/attention/query_norm", src.float32(b + "attn_q_norm.weight"), tol=1e-2)
    check(f"text/layers/{layer}/attention/key_norm", src.float32(b + "attn_k_norm.weight"), tol=1e-2)
    check(f"text/layers/{layer}/attention/indexer/query", src.float32(b + "indexer.q_proj.weight"), tol=1e-2)
for layer in (0, 1, 22, 46):
    b = f"blk.{layer}."
    qkv = src.float32(b + "attn_qkv.weight"); z = src.float32(b + "attn_gate.weight")
    ref = np.concatenate([qkv[:4096], cv._untile_v(qkv[4096:], 0), cv._untile_v(z, 0)])
    check(f"text/layers/{layer}/gdn/query_key_value_z", ref, tol=1e-2)
    check(f"text/layers/{layer}/gdn/output", cv._untile_v(src.float32(b + "ssm_out.weight"), 1), tol=1e-2)
    check(f"text/layers/{layer}/gdn/a_log", np.log(-cv._untile_v_heads(src.float32(b + "ssm_a").reshape(48), 0)), tol=1e-3)
    check(f"text/layers/{layer}/gdn/dt_bias", cv._untile_v_heads(src.float32(b + "ssm_dt.bias").reshape(48), 0), tol=1e-2)
    check(f"text/layers/{layer}/gdn/norm", src.float32(b + "ssm_norm.weight"), tol=1e-2)
    conv = src.float32(b + "ssm_conv1d.weight")  # (10240, 4) channel-major, as the conv op reads it
    check(f"text/layers/{layer}/gdn/convolution", np.ascontiguousarray(np.concatenate([conv[:4096], cv._untile_v(conv[4096:], 0)], axis=0).T), tol=1e-2)
    check(f"text/layers/{layer}/hc_attn/norm", src.float32(b + "hc_attn_norm.weight"), tol=1e-6)
    check(f"text/layers/{layer}/hc_attn/down", src.float32(b + "hc_attn_down.weight"), tol=1e-2)
    check(f"text/layers/{layer}/hc_attn/up", src.float32(b + "hc_attn_up.weight"), tol=1e-2)
    check(f"text/layers/{layer}/hc_attn/inject", src.float32(b + "hc_attn_inject.weight"), tol=1e-2)
    check(f"text/layers/{layer}/hc_ffn/norm", src.float32(b + "hc_ffn_norm.weight"), tol=1e-6)
    check(f"text/layers/{layer}/hc_ffn/up", src.float32(b + "hc_ffn_up.weight"), tol=1e-2)
    check(f"text/layers/{layer}/mlp/router_shared_gate", np.concatenate([src.float32(b + "ffn_gate_inp.weight"), src.float32(b + "ffn_gate_inp_shexp.weight").reshape(1, -1)]), tol=1e-2)
    check(f"text/layers/{layer}/mlp/shared_gate_up", np.concatenate([src.float32(b + "ffn_gate_shexp.weight"), src.float32(b + "ffn_up_shexp.weight")]), tol=1e-2)
    check(f"text/layers/{layer}/mlp/shared_down", src.float32(b + "ffn_down_shexp.weight"), tol=1e-2)
    gate = src.float32(b + "ffn_gate_exps.weight"); up = src.float32(b + "ffn_up_exps.weight")
    fused = np.concatenate([gate, up], axis=1).reshape(-1, 2560); del gate, up
    check(f"text/layers/{layer}/mlp/routed_gate_up", fused, rows=[(0, 64), (1270, 20), (100 * 1280, 64), (511 * 1280 + 1200, 80)], tol=1.5e-2)
    del fused
    down = src.float32(b + "ffn_down_exps.weight").reshape(-1, 640)
    check(f"text/layers/{layer}/mlp/routed_down", down, rows=[(0, 64), (2550, 20), (300 * 2560, 64), (511 * 2560 + 2496, 64)], tol=1.5e-2)
    del down
b = "blk.1."
check("text/layers/1/ple/key", src.float32(b + "ple_key.weight"), tol=1e-2)
check("text/layers/1/ple/value", src.float32(b + "ple_value.weight"), tol=1e-2)
check("text/layers/1/ple/norm_key", src.float32(b + "ple_norm_key.weight"), tol=1e-6)
check("text/layers/1/ple/norm_conv", src.float32(b + "ple_norm_conv.weight"), tol=1e-6)
check("text/layers/1/ple/convolution", src.float32(b + "ple_conv1d.weight").T, tol=1e-2)
check("text/output_hc/norm", src.float32("output_hc_norm.weight"), tol=1e-6)
check("text/output_hc/down", src.float32("output_hc_down.weight"), tol=1e-2)
check("text/output_head", src.float32("output.weight"), rows=[(0, 256), (248044, 64), (248320 - 256, 256)], tol=1e-2)
check("text/token_embedding", src.float32("token_embd.weight"), rows=[(0, 256), (248320 - 256, 256)], tol=1e-2)
check("text/ple/multipliers", None); check("text/ple/head_offsets", None); check("text/ple/head_vocab_sizes", None)
# PLE table: exact repack, compare raw bytes of sampled row ranges against the GGUF tensor bytes.
tab = art.find(inv.PLE_TABLE_RESOURCE); tpay = art.payload(tab)
raw = src.raw("per_layer_token_embd.weight")
print(f"  {inv.PLE_TABLE_RESOURCE:48s} {len(tpay)/1e9:9.1f} GB  gguf raw {raw.nbytes/1e9:.1f} GB", flush=True)
row = inv.PLE_TABLE_ROW_BYTES
for r0 in (0, 12345, 320001536 // 2, 320001536 - 1000):
    a = bytes(tpay[r0 * row:(r0 + 1000) * row]); g = raw.tobytes()[r0 * row:(r0 + 1000) * row] if hasattr(raw, "tobytes") else bytes(raw[r0 * row:(r0 + 1000) * row])
    same = a == g
    if not same: bad += 1
    print(f"    ple rows {r0}..+1000: {'bit-exact' if same else 'MISMATCH'}", flush=True)
print("VERIFY_DONE bad=%d" % bad, flush=True)
