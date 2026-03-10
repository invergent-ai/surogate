"""Benchmark surogate gated-delta-rule Triton kernels (forward/backward/total)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import triton

triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.uint8, device="cuda").data_ptr())

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd

from surogate.kernels.triton.gated_delta_rule import (
    chunk_local_cumsum_kernel,
    chunk_scaled_dot_kkt_fwd_kernel,
    solve_tril_64x64_kernel,
    recompute_w_u_fwd_kernel,
    chunk_fwd_h_kernel,
    chunk_fwd_o_kernel,
    chunk_bwd_dv_local_kernel,
    chunk_bwd_dhu_kernel,
    chunk_bwd_dqkwg_kernel,
    prepare_wy_repr_bwd_kernel,
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


# ---------------------------------------------------------------------------
# Forward / backward pipelines using our kernels
# ---------------------------------------------------------------------------

def _surogate_fwd(q, k, v, g_input, beta, scale, h0):
    """Run surogate forward pipeline, return (g_cum, o, Ai, w, u, h, v_new, ht)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = 64
    NT = triton.cdiv(T, BT)

    g = torch.empty(B, T, H, dtype=torch.float32, device=q.device)
    chunk_local_cumsum_kernel[(NT, B * H)](g_input, g, T, H=H, BT=BT, REVERSE=0)

    BK_kkt = min(max(triton.next_power_of_2(K), 16), 64)
    A = torch.empty(B, T, H, BT, dtype=torch.float32, device=q.device)
    chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](k, g, beta, A, T, H=H, K=K, BT=BT, BK=BK_kkt)

    sm = torch.cuda.get_device_capability()
    use_tma = sm[0] >= 9
    Ai = torch.zeros(B, T, H, BT, dtype=torch.bfloat16, device=q.device)
    solve_tril_64x64_kernel[(NT, B * H)](A, Ai, T, H=H, BT=BT, USE_TMA=int(use_tma), DOT_PRECISION="tf32")

    w = torch.empty(B, T, H, K, dtype=torch.bfloat16, device=q.device)
    u = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=q.device)
    recompute_w_u_fwd_kernel[(NT, B * H)](k, v, beta, w, u, Ai, g, T, H=H, K=K, V=V, BT=BT, BK=64, BV=64)

    BV_h = 32 if K > 64 else min(max(triton.next_power_of_2(V), 16), 64)
    h = torch.empty(B, NT, H, K, V, dtype=torch.bfloat16, device=q.device)
    ht = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)
    v_new = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=q.device)
    chunk_fwd_h_kernel[(triton.cdiv(V, BV_h), B * H)](
        k, u, w, v_new, g, h, h0, ht, T, H=H, K=K, V=V, BT=BT, BV=BV_h, num_stages=2)

    BK_o = min(max(triton.next_power_of_2(K), 16), 64)
    BV_o = min(max(triton.next_power_of_2(V), 16), 64)
    o = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=q.device)
    chunk_fwd_o_kernel[(triton.cdiv(V, BV_o), NT, B * H)](
        q, k, v_new, h, g, o, scale, T, H=H, K=K, V=V, BT=BT, BK=BK_o, BV=BV_o)

    return g, o, Ai, w, u, h, v_new, ht


def _surogate_bwd(q, k, v, g, beta, Ai, scale, h0, do, dht):
    """Run surogate backward pipeline, return (dq, dk, dv, db, dg, dh0)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = 64
    NT = triton.cdiv(T, BT)

    # Recompute w, u
    w = torch.empty(B, T, H, K, dtype=torch.bfloat16, device=q.device)
    u = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=q.device)
    recompute_w_u_fwd_kernel[(NT, B * H)](k, v, beta, w, u, Ai, g, T, H=H, K=K, V=V, BT=BT, BK=64, BV=64)

    # Recompute h, v_new
    BV_h = 32 if K > 64 else min(max(triton.next_power_of_2(V), 16), 64)
    h = torch.empty(B, NT, H, K, V, dtype=torch.bfloat16, device=q.device)
    ht_dummy = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)
    v_new = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=q.device)
    chunk_fwd_h_kernel[(triton.cdiv(V, BV_h), B * H)](
        k, u, w, v_new, g, h, h0, ht_dummy, T, H=H, K=K, V=V, BT=BT, BV=BV_h, num_stages=2)

    # dv_local
    BK_bwd = min(max(triton.next_power_of_2(K), 16), 64)
    BV_bwd = min(max(triton.next_power_of_2(V), 16), 64)
    dv = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=q.device)
    chunk_bwd_dv_local_kernel[(NT, B * H)](q, k, g, do, dv, scale, T, H=H, K=K, V=V, BT=BT, BK=BK_bwd, BV=BV_bwd)

    # bwd_dhu
    dh = torch.empty(B, NT, H, K, V, dtype=torch.bfloat16, device=q.device)
    dh0 = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)
    dv2 = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=q.device)
    chunk_bwd_dhu_kernel[(triton.cdiv(V, BV_h), B * H)](
        q, k, w, g, dht, dh0, do, dh, dv, dv2, scale, T, H=H, K=K, V=V, BT=BT, BV=BV_h, num_stages=2)

    # bwd_dqkwg
    NK = triton.cdiv(K, BK_bwd)
    dq = torch.empty(B, T, H, K, dtype=torch.bfloat16, device=q.device)
    dk = torch.empty(B, T, H, K, dtype=torch.bfloat16, device=q.device)
    dw = torch.empty(B, T, H, K, dtype=torch.bfloat16, device=q.device)
    dg_nk = torch.empty(NK, B, T, H, dtype=torch.float32, device=q.device)
    chunk_bwd_dqkwg_kernel[(NK, NT, B * H)](
        q, k, v_new, g, h, do, dh, dq, dk, dw, dv2, dg_nk, scale, B, T,
        H=H, K=K, V=V, BT=BT, BK=BK_bwd, BV=BV_bwd)
    dg = dg_nk.sum(0)

    # bwd_wy
    db = torch.empty(B, T, H, dtype=torch.bfloat16, device=q.device)
    dg_wy = torch.empty(B, T, H, dtype=torch.float32, device=q.device)
    prepare_wy_repr_bwd_kernel[(NT, B * H)](
        k, v, beta, g, Ai, dw, dv2, dk, dv, db, dg_wy, T,
        H=H, K=K, V=V, BT=BT, BK=BK_bwd, BV=BV_bwd)
    dg.add_(dg_wy)

    # reverse cumsum
    dg_out = torch.empty_like(dg)
    chunk_local_cumsum_kernel[(NT, B * H)](dg, dg_out, T, H=H, BT=BT, REVERSE=1)

    return dq, dk, dv, db, dg_out, dh0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

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
    do_t = _signal((B, T, H, V), device=device, dtype=dtype, a=0.149, b=0.66, c=0.101, d=-0.14)
    dht = _signal((B, H, K, V), device=device, dtype=torch.float32, a=0.089, b=-0.33, c=0.055, d=0.47)

    # Pre-run forward to get intermediates for backward benchmark
    if use_qk_l2norm_in_kernel:
        q_norm, q_rstd = l2norm_fwd(q)
        k_norm, k_rstd = l2norm_fwd(k)
    else:
        q_norm, q_rstd = q, None
        k_norm, k_rstd = k, None

    g_cum, _o, Ai, _w, _u, _h, _vn, _ht = _surogate_fwd(q_norm, k_norm, v, g, beta, scale, initial_state)

    def run_fwd() -> None:
        if use_qk_l2norm_in_kernel:
            qx, _ = l2norm_fwd(q)
            kx, _ = l2norm_fwd(k)
        else:
            qx, kx = q, k
        _surogate_fwd(qx, kx, v, g, beta, scale, initial_state)

    def run_bwd() -> None:
        dq, dk, _dv, _db, dg, _dh0 = _surogate_bwd(
            q_norm, k_norm, v, g_cum, beta, Ai, scale, initial_state, do_t, dht)
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
        gx, _ox, Ax, _w, _u, _h, _vn, _ht = _surogate_fwd(qx, kx, v, g, beta, scale, initial_state)
        dq, dk, _dv, _db, dg, _dh0 = _surogate_bwd(
            qx, kx, v, gx, beta, Ax, scale, initial_state, do_t, dht)
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


def _jit_warmup(B: int, T: int, H: int, K: int, V: int) -> None:
    """Trigger Triton JIT compilation for all kernels with progress output."""
    import sys
    import time

    device = "cuda"
    dtype = torch.bfloat16
    BT = 64
    NT = triton.cdiv(T, BT)
    scale = 1.0 / math.sqrt(K)
    sm = torch.cuda.get_device_capability()
    use_tma = sm[0] >= 9
    BK_kkt = min(max(triton.next_power_of_2(K), 16), 64)
    BV_h = 32 if K > 64 else min(max(triton.next_power_of_2(V), 16), 64)
    BK_o = min(max(triton.next_power_of_2(K), 16), 64)
    BV_o = min(max(triton.next_power_of_2(V), 16), 64)
    BK_bwd = min(max(triton.next_power_of_2(K), 16), 64)
    BV_bwd = min(max(triton.next_power_of_2(V), 16), 64)
    NK = triton.cdiv(K, BK_bwd)

    # Allocate tiny tensors just for compilation
    q = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.1
    g_in = -torch.rand(B, T, H, dtype=dtype, device=device)
    beta = torch.rand(B, T, H, dtype=dtype, device=device)
    h0 = torch.zeros(B, H, K, V, dtype=dtype, device=device)
    g = torch.empty(B, T, H, dtype=torch.float32, device=device)
    A = torch.empty(B, T, H, BT, dtype=torch.float32, device=device)
    Ai = torch.zeros(B, T, H, BT, dtype=dtype, device=device)
    w = torch.empty(B, T, H, K, dtype=dtype, device=device)
    u = torch.empty(B, T, H, V, dtype=dtype, device=device)
    h = torch.empty(B, NT, H, K, V, dtype=dtype, device=device)
    ht = torch.empty(B, H, K, V, dtype=torch.float32, device=device)
    v_new = torch.empty(B, T, H, V, dtype=dtype, device=device)
    o = torch.empty(B, T, H, V, dtype=dtype, device=device)
    do_t = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.1
    dht = torch.zeros(B, H, K, V, dtype=torch.float32, device=device)
    dv = torch.empty(B, T, H, V, dtype=dtype, device=device)
    dh = torch.empty(B, NT, H, K, V, dtype=dtype, device=device)
    dh0 = torch.empty(B, H, K, V, dtype=torch.float32, device=device)
    dv2 = torch.empty(B, T, H, V, dtype=dtype, device=device)
    dq = torch.empty(B, T, H, K, dtype=dtype, device=device)
    dk = torch.empty(B, T, H, K, dtype=dtype, device=device)
    dw = torch.empty(B, T, H, K, dtype=dtype, device=device)
    dg_nk = torch.empty(NK, B, T, H, dtype=torch.float32, device=device)
    db = torch.empty(B, T, H, dtype=dtype, device=device)
    dg_wy = torch.empty(B, T, H, dtype=torch.float32, device=device)

    kernels = [
        ("cumsum_fwd", lambda: chunk_local_cumsum_kernel[(NT, B*H)](g_in, g, T, H=H, BT=BT, REVERSE=0)),
        ("cumsum_rev", lambda: chunk_local_cumsum_kernel[(NT, B*H)](g, torch.empty_like(g), T, H=H, BT=BT, REVERSE=1)),
        ("kkt_fwd", lambda: chunk_scaled_dot_kkt_fwd_kernel[(NT, B*H)](k, g, beta, A, T, H=H, K=K, BT=BT, BK=BK_kkt)),
        ("solve_tril", lambda: solve_tril_64x64_kernel[(NT, B*H)](A, Ai, T, H=H, BT=BT, USE_TMA=int(use_tma), DOT_PRECISION="tf32")),
        ("wy_fwd", lambda: recompute_w_u_fwd_kernel[(NT, B*H)](k, v, beta, w, u, Ai, g, T, H=H, K=K, V=V, BT=BT, BK=64, BV=64)),
        ("fwd_h", lambda: chunk_fwd_h_kernel[(triton.cdiv(V, BV_h), B*H)](k, u, w, v_new, g, h, h0, ht, T, H=H, K=K, V=V, BT=BT, BV=BV_h, num_stages=2)),
        ("fwd_o", lambda: chunk_fwd_o_kernel[(triton.cdiv(V, BV_o), NT, B*H)](q, k, v_new, h, g, o, scale, T, H=H, K=K, V=V, BT=BT, BK=BK_o, BV=BV_o)),
        ("bwd_dv_local", lambda: chunk_bwd_dv_local_kernel[(NT, B*H)](q, k, g, do_t, dv, scale, T, H=H, K=K, V=V, BT=BT, BK=BK_bwd, BV=BV_bwd)),
        ("bwd_dhu", lambda: chunk_bwd_dhu_kernel[(triton.cdiv(V, BV_h), B*H)](q, k, w, g, dht, dh0, do_t, dh, dv, dv2, scale, T, H=H, K=K, V=V, BT=BT, BV=BV_h, num_stages=2)),
        ("bwd_dqkwg", lambda: chunk_bwd_dqkwg_kernel[(NK, NT, B*H)](q, k, v_new, g, h, do_t, dh, dq, dk, dw, dv2, dg_nk, scale, B, T, H=H, K=K, V=V, BT=BT, BK=BK_bwd, BV=BV_bwd)),
        ("bwd_wy", lambda: prepare_wy_repr_bwd_kernel[(NT, B*H)](k, v, beta, g, Ai, dw, dv2, dk, dv, db, dg_wy, T, H=H, K=K, V=V, BT=BT, BK=BK_bwd, BV=BV_bwd)),
    ]

    total = len(kernels)
    for i, (name, fn) in enumerate(kernels, 1):
        sys.stdout.write(f"  JIT compiling [{i}/{total}] {name}...")
        sys.stdout.flush()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f" {dt:.1f}s")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    warmup_iters = 5
    bench_iters = 20
    configs = [
        (1, 512, 8, 128, 128),
        (2, 2048, 16, 128, 128),
    ]

    # JIT-compile all kernel variants upfront with progress output
    unique_shapes = set((H, K, V) for _, _, H, K, V in configs)
    for H, K, V in unique_shapes:
        B0, T0 = 1, 128  # minimal size for compilation
        print(f"[surogate gdr] JIT warmup for H={H} K={K} V={V}...")
        _jit_warmup(B0, T0, H, K, V)
        print(f"[surogate gdr] JIT warmup done.")

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
            f"[surogate gdr benchmark] B={case['B']} T={case['T']} H={case['H']} K={case['K']} V={case['V']} "
            f"| fwd={case['forward_ms']:.6f} ms bwd={case['backward_ms']:.6f} ms total={case['total_ms']:.6f} ms"
        )

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "tests" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gated_delta_rule_surogate_perf.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[surogate gdr benchmark] wrote {out_path}")


if __name__ == "__main__":
    main()
