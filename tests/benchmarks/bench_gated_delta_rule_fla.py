"""Benchmark FLA gated-delta-rule chunk kernels (forward/backward/total)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule_fwd,
)


def _signal(
    shape: tuple[int, ...],
    *,
    device: str,
    dtype: torch.dtype,
    a: float,
    b: float,
    c: float,
    d: float,
) -> torch.Tensor:
    n = 1
    for s in shape:
        n *= s
    idx = torch.arange(n, device=device, dtype=torch.float32)
    x = 0.7 * torch.sin(idx * a + b) + 0.3 * torch.cos(idx * c + d)
    return x.reshape(shape).to(dtype)


def _bench_cuda_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / float(iters)


def _run_case(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    *,
    warmup: int,
    iters: int,
    chunk_size: int = 64,
    use_qk_l2norm_in_kernel: bool = True,
) -> dict[str, float | int | bool | str]:
    device = "cuda"
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(K)

    q = _signal((B, T, H, K), device=device, dtype=dtype, a=0.173, b=0.31, c=0.097, d=-0.22)
    k = _signal((B, T, H, K), device=device, dtype=dtype, a=0.137, b=-0.41, c=0.083, d=0.57)
    v = _signal((B, T, H, V), device=device, dtype=dtype, a=0.191, b=0.73, c=0.121, d=-0.35)

    g_raw = _signal((B, T, H), device=device, dtype=torch.float32, a=0.157, b=-0.27, c=0.109, d=0.44)
    g = (-0.7 + 0.2 * torch.sin(g_raw)).to(dtype)
    beta_raw = _signal((B, T, H), device=device, dtype=torch.float32, a=0.113, b=0.19, c=0.071, d=-0.63)
    beta = torch.sigmoid(beta_raw).to(dtype)

    initial_state = _signal((B, H, K, V), device=device, dtype=torch.float32, a=0.167, b=-0.52, c=0.061, d=0.28)
    do = _signal((B, T, H, V), device=device, dtype=dtype, a=0.149, b=0.66, c=0.101, d=-0.14)
    dht = _signal((B, H, K, V), device=device, dtype=torch.float32, a=0.089, b=-0.33, c=0.055, d=0.47)

    if use_qk_l2norm_in_kernel:
        q_norm, q_rstd = l2norm_fwd(q)
        k_norm, k_rstd = l2norm_fwd(k)
    else:
        q_norm, q_rstd = q, None
        k_norm, k_rstd = k, None

    g_cum, _o_tmp, A, _final_tmp = chunk_gated_delta_rule_fwd(
        q=q_norm,
        k=k_norm,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
    )

    def run_fwd() -> None:
        if use_qk_l2norm_in_kernel:
            qx, _ = l2norm_fwd(q)
            kx, _ = l2norm_fwd(k)
        else:
            qx, kx = q, k
        chunk_gated_delta_rule_fwd(
            q=qx,
            k=kx,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=None,
        )

    def run_bwd() -> None:
        dq, dk, _dv, _db, dg, _dh0 = chunk_gated_delta_rule_bwd(
            q=q_norm,
            k=k_norm,
            v=v,
            g=g_cum,
            beta=beta,
            A=A,
            scale=scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=None,
        )
        if use_qk_l2norm_in_kernel:
            _ = l2norm_bwd(q_norm, q_rstd, dq)
            _ = l2norm_bwd(k_norm, k_rstd, dk)
        _ = dg

    def run_total() -> None:
        if use_qk_l2norm_in_kernel:
            qx, qx_rstd = l2norm_fwd(q)
            kx, kx_rstd = l2norm_fwd(k)
        else:
            qx, qx_rstd = q, None
            kx, kx_rstd = k, None
        gx, _ox, Ax, _fx = chunk_gated_delta_rule_fwd(
            q=qx,
            k=kx,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=None,
        )
        dq, dk, _dv, _db, dg, _dh0 = chunk_gated_delta_rule_bwd(
            q=qx,
            k=kx,
            v=v,
            g=gx,
            beta=beta,
            A=Ax,
            scale=scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=None,
        )
        if use_qk_l2norm_in_kernel:
            _ = l2norm_bwd(qx, qx_rstd, dq)
            _ = l2norm_bwd(kx, kx_rstd, dk)
        _ = dg

    fwd_ms = _bench_cuda_ms(run_fwd, warmup=warmup, iters=iters)
    bwd_ms = _bench_cuda_ms(run_bwd, warmup=warmup, iters=iters)
    total_ms = _bench_cuda_ms(run_total, warmup=warmup, iters=iters)

    return {
        "B": B,
        "T": T,
        "H": H,
        "K": K,
        "V": V,
        "dtype": str(dtype),
        "chunk_size": chunk_size,
        "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
        "forward_ms": fwd_ms,
        "backward_ms": bwd_ms,
        "total_ms": total_ms,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    warmup_iters = 5
    bench_iters = 20
    configs = [
        (1, 512, 8, 128, 128),
        (2, 2048, 16, 128, 128),
    ]

    report: dict[str, object] = {
        "meta": {
            "device": torch.cuda.get_device_name(0),
            "warmup_iters": warmup_iters,
            "bench_iters": bench_iters,
        },
        "cases": [],
    }
    for cfg in configs:
        case = _run_case(*cfg, warmup=warmup_iters, iters=bench_iters)
        report["cases"].append(case)
        print(
            f"[fla gdr benchmark] B={case['B']} T={case['T']} H={case['H']} K={case['K']} V={case['V']} "
            f"| fwd={case['forward_ms']:.6f} ms bwd={case['backward_ms']:.6f} ms total={case['total_ms']:.6f} ms"
        )

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "tests" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gated_delta_rule_fla_perf.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[fla gdr benchmark] wrote {out_path}")


if __name__ == "__main__":
    main()
