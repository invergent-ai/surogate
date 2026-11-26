"""The engine-op trainer's CUDA backward kernels (csrc/src/kernels/inkling, library libsinfer_inkling_bwd.so), loaded with
ctypes. Each function here replaces one torch recompute + autograd section of the visit backward and is checked against it
(tests/lever_check.py arms, tests/test_inkling2_moe_bwd_gpu.py). Library path: env IFT_BWD_LIB.

moe_groups: the routed experts of one visit, group-major (ift_long.MoEGM's row loop): the cotangents of the MLP input, the
routing weights and the layer's LoRA tensors.
"""
from __future__ import annotations

import ctypes
import os

import torch

F32, BF16 = torch.float32, torch.bfloat16
EG = 16  # experts decoded and handled per call (ift_backward.EG)

_LIB = None
_WS = {}


def lib():
    global _LIB
    if _LIB is None:
        path = os.environ.get("IFT_BWD_LIB")
        if not path:
            raise RuntimeError("IFT_BWD_LIB is not set (path of libsinfer_inkling_bwd.so)")
        _LIB = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        _LIB.inkling_capi_last_error.restype = ctypes.c_char_p
        _LIB.inkling_capi_moe_group_bwd_workspace_bytes.restype = ctypes.c_long
    return _LIB


def _workspace(dev, nbytes):
    t = _WS.get(dev)
    if t is None or t.numel() < nbytes:
        _WS[dev] = t = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    return t


def _ptr(t):
    return None if t is None else t.data_ptr()


def moe_groups(w, x16, g, tok, gam_r, starts, cnt, lora, scale, dx, dgam_r, lgr, max_rows):
    """w: ift_backward.EngineExperts of the layer. x16 bf16 [T, D] (the MLP input), g fp32 [T, D] (cotangent of the MoE
    output), tok int64 [R] / gam_r fp32 [R] (rows sorted by expert), starts / cnt: host lists per expert. lora: the layer's
    (A13, B13, A2, B2) fp32 tensors or None. Adds into dx fp32 [T, D]; writes dgam_r [R]; adds into lgr[j] (None = that
    tensor's gradient is not needed)."""
    L = lib()
    dev = x16.device
    T, D = x16.shape
    n_exp = len(cnt)
    I = D // 2
    rank = lora[0].shape[1] if lora is not None else 0
    x16 = x16.contiguous()
    g = g.contiguous()
    tok32 = tok.to(torch.int32).contiguous()
    gam_r = gam_r.contiguous()
    cap = max_rows if max_rows > 0 else 8192
    wsb = L.inkling_capi_moe_group_bwd_workspace_bytes(cap, D, I, rank)
    ws = _workspace(dev, wsb)
    if lora is not None and any(not t.is_contiguous() for t in lora):
        raise ValueError("moe_groups: the LoRA tensors must be contiguous")
    torch.cuda.current_stream(dev).synchronize()  # the kernels run on the default stream
    for g0 in range(0, n_exp, EG):
        g1 = min(n_exp, g0 + EG)
        r0, r1 = starts[g0], starts[g1 - 1] + cnt[g1 - 1]
        if r1 == r0:
            continue
        ng = g1 - g0
        u13, f13 = w.eng.decode_experts(w.layer, "w13", g0, ng)
        u2, f2 = w.eng.decode_experts(w.layer, "w2", g0, ng)
        if w.fmt == "nvfp4":
            f13h, f2h = f13.tolist(), f2.tolist()
        else:
            f13h, f2h = [1.0] * ng, [1.0] * ng
        torch.cuda.current_stream(dev).synchronize()
        seg = (ctypes.c_int * (ng + 1))(*([starts[e] for e in range(g0, g1)] + [r1]))
        lo = [None] * 4 if lora is None else [t[g0:g1] for t in lora]
        gr = [None] * 4 if lora is None else [None if t is None else t[g0:g1] for t in lgr]
        ints = (ctypes.c_int * 6)(T, D, I, ng, rank, cap)
        ptrs = (ctypes.c_void_p * 17)(_ptr(x16), _ptr(g), _ptr(tok32), _ptr(gam_r), _ptr(u13), _ptr(u2), *[_ptr(t) for t in lo],
                                       _ptr(dx), _ptr(dgam_r), *[_ptr(t) for t in gr], _ptr(ws))
        rc = L.inkling_capi_moe_group_bwd(ints, ptrs, seg, (ctypes.c_float * ng)(*f13h), (ctypes.c_float * ng)(*f2h),
                                          ctypes.c_float(float(scale) if scale is not None else 0.0), ctypes.c_long(wsb), None)
        if rc != 0:
            raise RuntimeError("inkling_capi_moe_group_bwd: " + L.inkling_capi_last_error().decode())
        torch.cuda.synchronize(dev)  # the call has finished before the decoded units go back to the allocator
        del u13, u2
