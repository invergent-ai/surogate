"""GPU check of the routed experts' CUDA backward (csrc/src/kernels/inkling/moe_backward.cu) against torch autograd.

The reference is the trainer's own arithmetic for one (token, expert) row (surogate/train/inkling_fastpath/ift_long.py
_moe_rows): unit GEMMs with bf16 operands and fp32 accumulation, the expert's global scale, the LoRA paths in fp32, one
bf16 rounding per GEMM output (straight-through), and a unit-GEMM backward whose cotangent is rounded to bf16.

Lines (per tensor): cosine >= 0.9999 and relative error <= 1 %. Production widths (D 2304, I 1152, rank 16); a few
experts and a few hundred tokens. Cases: NVFP4-like (global scales != 1) and MXFP8-like (scales 1), with and without
LoRA, one call and row-chunked calls.

Run: CUDA_VISIBLE_DEVICES=0 INKLING_LIB=/path/libsinfer_inkling_bwd.so python tests/test_inkling2_moe_bwd_gpu.py
Exit code 0 = all PASS.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys

import torch

LIB = os.environ.get("INKLING_LIB", "libsinfer_inkling_bwd.so")
F32, BF16 = torch.float32, torch.bfloat16
D, I, RANK = 2304, 1152, 16
DEV = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def rb(x):
    """bf16 rounding, straight-through."""
    return x + (x.detach().to(BF16).float() - x.detach())


class UnitMM(torch.autograd.Function):
    """x [n, K] @ unit [M, K]^T in fp32 over bf16 values; backward: the cotangent rounded to bf16, then @ unit."""

    @staticmethod
    def forward(ctx, x, unit):
        ctx.unit = unit
        return x.to(BF16).float() @ unit.float().T

    @staticmethod
    def backward(ctx, g):
        return g.to(BF16).float() @ ctx.unit.float(), None


def swiglu(gu, gam):
    g, u = gu[:, 0::2], gu[:, 1::2]
    return rb(((4.0 * torch.tanh(g * 0.25) * torch.sigmoid(g)) * (25.0 * torch.tanh(u / 25.0))) * gam[:, None])


def reference(x, g, tok, gam, seg, u13, u2, f13, f2, lora, scale):
    T = x.shape[0]
    xf = x.float()
    dx = torch.zeros(T, D, dtype=F32, device=DEV)
    dgam = torch.zeros_like(gam)
    grads = None if lora is None else [torch.zeros_like(t) for t in lora]
    for e in range(len(seg) - 1):
        r0, r1 = seg[e], seg[e + 1]
        if r1 == r0:
            continue
        t = tok[r0:r1].long()
        xr = xf[t].detach().requires_grad_()
        gr = gam[r0:r1].detach().requires_grad_()
        leaves = [] if lora is None else [p[e].detach().requires_grad_() for p in lora]
        gu = UnitMM.apply(xr, u13[e]) * f13[e]
        if lora is not None:
            A13, B13, A2, B2 = leaves
            gu = gu + scale * ((xr @ A13.T) @ B13.T)
        y = swiglu(rb(gu), gr)
        dn = UnitMM.apply(y, u2[e]) * f2[e]
        if lora is not None:
            dn = dn + scale * ((y @ A2.T) @ B2.T)
        dn = rb(dn)
        got = torch.autograd.grad(dn, [xr, gr] + leaves, g[t])
        dx.index_add_(0, t, got[0])
        dgam[r0:r1] = got[1]
        for j, gj in enumerate(got[2:]):
            grads[j][e] = gj
    return dx, dgam, grads


def make_case(seed, T, ng, topk, nvfp4):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(T, D, generator=gen) * 0.7).to(BF16)
    g = torch.randn(T, D, generator=gen) * 1e-3
    u13 = (torch.randn(ng, 2 * I, D, generator=gen) * 0.02).to(BF16)
    u2 = (torch.randn(ng, D, I, generator=gen) * 0.02).to(BF16)
    f13 = (torch.rand(ng, generator=gen) * 1.5 + 0.5) if nvfp4 else torch.ones(ng)
    f2 = (torch.rand(ng, generator=gen) * 1.5 + 0.5) if nvfp4 else torch.ones(ng)
    # routing: topk distinct experts per token, uneven on purpose (expert 0 gets every token)
    ids = torch.stack([torch.cat([torch.zeros(1, dtype=torch.long), 1 + torch.randperm(ng - 1, generator=gen)[:topk - 1]])
                       for _ in range(T)])
    gam_tk = torch.rand(T, topk, generator=gen) * 2.0 + 0.1
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    tok = (order // topk).to(torch.int32)
    gam = gam_tk.reshape(-1)[order].contiguous()
    counts = torch.bincount(flat, minlength=ng)
    seg = [0] + torch.cumsum(counts, 0).tolist()
    A13 = (torch.rand(ng, RANK, D, generator=gen) * 2 - 1) / D ** 0.5
    A2 = (torch.rand(ng, RANK, I, generator=gen) * 2 - 1) / I ** 0.5
    B13 = torch.randn(ng, 2 * I, RANK, generator=gen) * 1e-2
    B2 = torch.randn(ng, D, RANK, generator=gen) * 1e-2
    dev = lambda t: t.to(DEV).contiguous()  # noqa: E731
    return dict(x=dev(x), g=dev(g), tok=dev(tok), gam=dev(gam), seg=seg, u13=dev(u13), u2=dev(u2), f13=f13.tolist(),
                f2=f2.tolist(), lora=[dev(A13), dev(B13), dev(A2), dev(B2)])


def run_kernel(lib, c, use_lora, max_rows, scale):
    T, ng = c["x"].shape[0], len(c["seg"]) - 1
    dx = torch.zeros(T, D, dtype=F32, device=DEV)
    dgam = torch.zeros_like(c["gam"])
    lora = c["lora"] if use_lora else [None] * 4
    grads = [torch.zeros_like(t) for t in c["lora"]] if use_lora else [None] * 4
    cap = max_rows if max_rows > 0 else 8192
    wsb = lib.inkling_capi_moe_group_bwd_workspace_bytes(cap, D, I, RANK)
    ws = torch.empty(wsb, dtype=torch.uint8, device=DEV)
    ints = (ctypes.c_int * 6)(T, D, I, ng, RANK if use_lora else 0, max_rows)
    plist = [c["x"], c["g"], c["tok"], c["gam"], c["u13"], c["u2"], *lora, dx, dgam, *grads, ws]
    ptrs = (ctypes.c_void_p * 17)(*[None if t is None else t.data_ptr() for t in plist])
    seg = (ctypes.c_int * (ng + 1))(*c["seg"])
    f13 = (ctypes.c_float * ng)(*c["f13"])
    f2 = (ctypes.c_float * ng)(*c["f2"])
    rc = lib.inkling_capi_moe_group_bwd(ints, ptrs, seg, f13, f2, ctypes.c_float(scale), ctypes.c_long(wsb), None)
    torch.cuda.synchronize()
    if rc != 0:
        raise RuntimeError(lib.inkling_capi_last_error().decode())
    return dx, dgam, (grads if use_lora else None)


def compare(name, got, ref, report):
    got, ref = got.double().flatten(), ref.double().flatten()
    cos = float(torch.dot(got, ref) / (got.norm() * ref.norm()).clamp_min(1e-300))
    rel = float((got - ref).norm() / ref.norm().clamp_min(1e-300))
    ok = cos >= 0.9999 and rel <= 0.01 and bool(torch.isfinite(got).all())
    report.append({"check": name, "pass": ok, "cos": cos, "rel_err": rel, "ref_norm": float(ref.norm())})
    print(f"  {'PASS' if ok else 'FAIL'} {name} cos {cos:.8f} rel_err {rel:.3e}")
    return ok


def main():
    lib = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
    lib.inkling_capi_last_error.restype = ctypes.c_char_p
    lib.inkling_capi_moe_group_bwd_workspace_bytes.restype = ctypes.c_long
    report, ok = [], True
    scale = 2.0
    for nvfp4 in (True, False):
        c = make_case(11 if nvfp4 else 12, T=384, ng=4, topk=3, nvfp4=nvfp4)
        fmt = "nvfp4" if nvfp4 else "mxfp8"
        f13 = torch.tensor(c["f13"], device=DEV)
        f2 = torch.tensor(c["f2"], device=DEV)
        for use_lora in (False, True):
            ref = reference(c["x"], c["g"], c["tok"], c["gam"], c["seg"], c["u13"], c["u2"], f13, f2,
                            c["lora"] if use_lora else None, scale)
            for max_rows in (0, 100):
                tag = f"{fmt}/{'lora' if use_lora else 'base'}/{'chunk100' if max_rows else 'whole'}"
                print(f"[{tag}]")
                got = run_kernel(lib, c, use_lora, max_rows, scale)
                ok &= compare(f"{tag}/dx", got[0], ref[0], report)
                ok &= compare(f"{tag}/dgam", got[1], ref[1], report)
                if use_lora:
                    for k, gk, rk in zip(("dA13", "dB13", "dA2", "dB2"), got[2], ref[2]):
                        ok &= compare(f"{tag}/{k}", gk, rk, report)
    # determinism: two runs of the same call are bit-identical
    c = make_case(13, T=384, ng=4, topk=3, nvfp4=True)
    a = run_kernel(lib, c, True, 0, scale)
    b = run_kernel(lib, c, True, 0, scale)
    same = all(bool(torch.equal(p, q)) for p, q in zip([a[0], a[1], *a[2]], [b[0], b[1], *b[2]]))
    report.append({"check": "rerun_bit_identical", "pass": same})
    print(f"  {'PASS' if same else 'FAIL'} rerun_bit_identical")
    ok &= same
    n_pass = sum(1 for r in report if r["pass"])
    print(f"SUMMARY {n_pass}/{len(report)} checks PASS")
    with open(os.environ.get("INKLING_REPORT", "inkling2_moe_bwd_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
