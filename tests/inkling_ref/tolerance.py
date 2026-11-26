"""Tolerance rules of the GPU op tests of the Inkling-2 trainer kernels, as functions the tests call.

ref64 = the fp64 reference with the engine's storage roundings; pt32 = the same reference in fp32 with the kernels'
operand roundings (attention: p_bf16 / ds_bf16, dout bf16). Each function returns (ok: bool, report: dict)."""

from __future__ import annotations

import torch


def bf16_ulps(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """|x - ref| in bf16 ulps of ref."""
    r = ref.double()
    _, e = torch.frexp(r.abs().clamp(min=2.0 ** -126))
    return (x.double() - r).abs() / torch.ldexp(torch.ones_like(r), e - 8)


def rule_bf16_output(x, ref64, max_flip_frac=0.005, max_ulp=1.0, floor_rel_rms=1e-6):
    """C.1: every element within 1 bf16 ulp of ref64 (cancellation residues |diff| <= 1e-6 rms exempt); fraction of
    1-ulp flips <= max_flip_frac (0.5 % default; loop blend 2 %; SiTU 15 %)."""
    u = bf16_ulps(x, ref64)
    floor = floor_rel_rms * float(ref64.double().pow(2).mean().sqrt())
    u = torch.where((x.double() - ref64.double()).abs() <= floor, torch.zeros_like(u), u)
    rep = dict(max_ulp=float(u.max()), flip_frac=float((u > 0).double().mean()))
    return rep["max_ulp"] <= max_ulp and rep["flip_frac"] <= max_flip_frac, rep


def rule_e4m3_output(x, ref64, max_flip_frac=0.005):
    """C.1 (KV): codes identical except <= 0.5 % at one e4m3 ulp (e4m3 has 3 mantissa bits: ulp = 2^(e-3))."""
    r = ref64.double()
    _, e = torch.frexp(r.abs().clamp(min=2.0 ** -6))  # subnormals: ulp 2^-9
    u = (x.double() - r).abs() / torch.ldexp(torch.ones_like(r), e - 4)
    rep = dict(max_ulp=float(u.max()), flip_frac=float((u > 0).double().mean()))
    return rep["max_ulp"] <= 1.0 and rep["flip_frac"] <= max_flip_frac, rep


def rule_vs_pt(x, pt32, ref64, factor, atol_rel=1e-6, mean_factor=1.5):
    """C.2: max|x - ref64| <= factor * max|pt32 - ref64| + atol_rel * max|ref64| (factor 2 forward, 3 backward) and
    mean|x - ref64| <= mean_factor * mean|pt32 - ref64| + the same floor (no systematic bias)."""
    r = ref64.double()
    ek, ep = (x.double() - r).abs(), (pt32.double() - r).abs()
    floor = atol_rel * float(r.abs().max())
    rep = dict(err=float(ek.max()), err_pt=float(ep.max()), mean=float(ek.mean()), mean_pt=float(ep.mean()))
    ok = rep["err"] <= factor * rep["err_pt"] + floor and rep["mean"] <= mean_factor * rep["mean_pt"] + floor
    return ok, rep


def rule_fp32_path(x, ref64, rel=4e-6):
    """C.3: fp32 paths without bf16 operand rounding: max|x - ref64| <= 4e-6 max|ref64|."""
    r = ref64.double()
    err = float((x.double() - r).abs().max())
    return err <= rel * float(r.abs().max()), dict(err=err, bound=rel * float(r.abs().max()))


def rule_bit_exact(a, b):
    """C.4 / C.5 / C.6: bitwise equality (chunked vs unchunked row ops, determinism, forward equality)."""
    eq = bool(torch.equal(a.view(torch.uint8) if a.dtype != torch.bool else a, b.view(torch.uint8) if b.dtype != torch.bool else b))
    return eq, dict(differing=int((a != b).sum()))
