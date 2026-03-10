#!/usr/bin/env python3
"""Test our gated delta rule Triton kernels against FLA reference.

Runs both forward and backward passes, comparing intermediate tensors
and final outputs for numerical agreement.
"""

import torch
import triton

# Blackwell (SM120+) requires an explicit allocator for scratch memory
def _torch_alloc(size, align, stream):
    return torch.empty(size, dtype=torch.uint8, device="cuda").data_ptr()

triton.set_allocator(_torch_alloc)

from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_fwd as fla_fwd,
    chunk_gated_delta_rule_bwd as fla_bwd,
)
from fla.ops.utils.cumsum import chunk_local_cumsum

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


def check(name, ours, ref, atol=1e-2, rtol=1e-2):
    """Compare two tensors and print result."""
    if ours.dtype != ref.dtype:
        ours = ours.to(ref.dtype)
    if ours.shape != ref.shape:
        # Trim to common shape
        slices = tuple(slice(0, min(a, b)) for a, b in zip(ours.shape, ref.shape))
        ours = ours[slices]
        ref = ref[slices]
    max_diff = (ours.float() - ref.float()).abs().max().item()
    mean_diff = (ours.float() - ref.float()).abs().mean().item()
    ok = torch.allclose(ours.float(), ref.float(), atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    return ok


# ---------------------------------------------------------------------------
# Our forward pipeline (mirrors FLA step by step)
# ---------------------------------------------------------------------------

def our_forward(q, k, v, g_raw, beta, scale, h0):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = 64
    NT = triton.cdiv(T, BT)

    # 1. Local cumsum of g
    g = torch.empty(B, T, H, dtype=torch.float32, device=q.device)
    chunk_local_cumsum_kernel[(NT, B * H)](
        g_raw, g, T, H=H, BT=BT, REVERSE=0,
    )

    # 2. Scaled dot kkt
    A = torch.empty(B, T, H, BT, dtype=torch.float32, device=q.device)
    BK_kkt = min(max(triton.next_power_of_2(K), 16), 64)
    chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](
        k, g, beta, A, T, H=H, K=K, BT=BT, BK=BK_kkt,
    )

    # 3. Solve tril
    sm = torch.cuda.get_device_capability()
    use_tma = sm[0] >= 9
    Ai = torch.zeros(B, T, H, BT, dtype=torch.bfloat16, device=q.device)
    solve_tril_64x64_kernel[(NT, B * H)](
        A, Ai, T, H=H, BT=BT, USE_TMA=int(use_tma), DOT_PRECISION="ieee",
    )

    # 4. WY forward (w, u)
    w = torch.empty(B, T, H, K, dtype=torch.bfloat16, device=q.device)
    u = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=q.device)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k, v, beta, w, u, Ai, g, T, H=H, K=K, V=V, BT=BT, BK=64, BV=64,
    )

    # 5. Forward h (state recurrence)
    h = torch.empty(B, NT, H, K, V, dtype=torch.bfloat16, device=q.device)
    ht = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)
    v_new = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=q.device)
    # Use BV=32 for large K to avoid shared memory OOM
    BV_h = 32 if K > 64 else min(max(triton.next_power_of_2(V), 16), 64)
    chunk_fwd_h_kernel[(triton.cdiv(V, BV_h), B * H)](
        k, u, w, v_new, g, h, h0, ht, T, H=H, K=K, V=V, BT=BT, BV=BV_h,
        num_stages=1,
    )

    # 6. Forward o (output)
    o = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=q.device)
    BK_o = min(max(triton.next_power_of_2(K), 16), 64)
    BV_o = min(max(triton.next_power_of_2(V), 16), 64)
    chunk_fwd_o_kernel[(triton.cdiv(V, BV_o), NT, B * H)](
        q, k, v_new, h, g, o, scale, T, H=H, K=K, V=V, BT=BT, BK=BK_o, BV=BV_o,
    )

    return g, o, Ai, w, u, h, v_new, ht


def our_backward(q, k, v, g, beta, Ai, scale, h0, do, dht):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = 64
    NT = triton.cdiv(T, BT)

    # Recompute w, u
    w = torch.empty(B, T, H, K, dtype=torch.bfloat16, device=q.device)
    u = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=q.device)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k, v, beta, w, u, Ai, g, T, H=H, K=K, V=V, BT=BT, BK=64, BV=64,
    )

    # Recompute h, v_new
    h = torch.empty(B, NT, H, K, V, dtype=torch.bfloat16, device=q.device)
    ht_dummy = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)
    v_new = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=q.device)
    BV_h = 32 if K > 64 else min(max(triton.next_power_of_2(V), 16), 64)
    chunk_fwd_h_kernel[(triton.cdiv(V, BV_h), B * H)](
        k, u, w, v_new, g, h, h0, ht_dummy, T, H=H, K=K, V=V, BT=BT, BV=BV_h,
        num_stages=1,
    )

    # 1. Local dv
    BK_bwd = min(max(triton.next_power_of_2(K), 16), 64)
    BV_bwd = min(max(triton.next_power_of_2(V), 16), 64)
    dv = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=q.device)
    chunk_bwd_dv_local_kernel[(NT, B * H)](
        q, k, g, do, dv, scale, T, H=H, K=K, V=V, BT=BT, BK=BK_bwd, BV=BV_bwd,
    )

    # 2. Backward dhu
    dh = torch.empty(B, NT, H, K, V, dtype=torch.bfloat16, device=q.device)
    dh0 = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)
    dv2 = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=q.device)
    chunk_bwd_dhu_kernel[(triton.cdiv(V, BV_h), B * H)](
        q, k, w, g, dht, dh0, do, dh, dv, dv2, scale, T,
        H=H, K=K, V=V, BT=BT, BV=BV_h,
        num_stages=1,
    )

    # 3. Backward dqkwg
    NK = triton.cdiv(K, BK_bwd)
    dq = torch.empty(B, T, H, K, dtype=torch.bfloat16, device=q.device)
    dk = torch.empty(B, T, H, K, dtype=torch.bfloat16, device=q.device)
    dw = torch.empty(B, T, H, K, dtype=torch.bfloat16, device=q.device)
    dg_nk = torch.empty(NK, B, T, H, dtype=torch.float32, device=q.device)
    chunk_bwd_dqkwg_kernel[(NK, NT, B * H)](
        q, k, v_new, g, h, do, dh, dq, dk, dw, dv2, dg_nk, scale, B, T,
        H=H, K=K, V=V, BT=BT, BK=BK_bwd, BV=BV_bwd,
    )
    dg = dg_nk.sum(0)  # [B, T, H]

    # 4. Backward WY repr
    db = torch.empty(B, T, H, dtype=torch.bfloat16, device=q.device)
    dg_wy = torch.empty(B, T, H, dtype=torch.float32, device=q.device)
    # du is dv2 (the updated dv from bwd_dhu)
    prepare_wy_repr_bwd_kernel[(NT, B * H)](
        k, v, beta, g, Ai, dw, dv2, dk, dv, db, dg_wy, T,
        H=H, K=K, V=V, BT=BT, BK=BK_bwd, BV=BV_bwd,
    )

    # Note: dk already has dk2 added in-place by prepare_wy_repr_bwd_kernel
    dg.add_(dg_wy)

    # 5. Reverse cumsum of dg
    dg_out = torch.empty_like(dg)
    chunk_local_cumsum_kernel[(NT, B * H)](
        dg, dg_out, T, H=H, BT=BT, REVERSE=1,
    )

    return dq, dk, dv, db, dg_out, dh0


# ---------------------------------------------------------------------------
# FLA reference wrappers
# ---------------------------------------------------------------------------

def fla_forward(q, k, v, g_raw, beta, scale, h0):
    g, o, A, final_state = fla_fwd(
        q=q, k=k, v=v, g=g_raw, beta=beta, scale=scale,
        initial_state=h0, output_final_state=True,
    )
    return g, o, A, final_state


def fla_backward(q, k, v, g, beta, A, scale, h0, do, dht):
    return fla_bwd(
        q=q, k=k, v=v, g=g, beta=beta, A=A, scale=scale,
        initial_state=h0, do=do, dht=dht,
    )


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def make_inputs(B, T, H, K, V, seed=42):
    """Create realistic test inputs matching FLA's expected ranges."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda") * 0.1
    # g must be negative (log-space gate, e.g. logsigmoid)
    g_raw = -torch.rand(B, T, H, dtype=torch.bfloat16, device="cuda") * 5
    # beta in [0, 1] range (learning rate / update strength)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.bfloat16, device="cuda"))
    h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device="cuda") * 0.01
    scale = K ** -0.5
    return q, k, v, g_raw, beta, h0, scale


def test_forward(B=2, T=256, H=4, K=64, V=64, seed=42):
    print(f"\n=== Forward Test (B={B}, T={T}, H={H}, K={K}, V={V}) ===")
    q, k, v, g_raw, beta, h0, scale = make_inputs(B, T, H, K, V, seed)

    # FLA reference
    g_ref, o_ref, A_ref, ht_ref = fla_forward(q, k, v, g_raw, beta, scale, h0)

    # Ours
    g_ours, o_ours, Ai_ours, w_ours, u_ours, h_ours, vn_ours, ht_ours = \
        our_forward(q, k, v, g_raw, beta, scale, h0)

    all_pass = True
    all_pass &= check("g_cumsum", g_ours, g_ref)
    all_pass &= check("A_inv", Ai_ours, A_ref)
    all_pass &= check("output o", o_ours, o_ref, atol=2e-2, rtol=2e-2)
    all_pass &= check("final_state ht", ht_ours, ht_ref, atol=5e-2, rtol=5e-2)

    return all_pass, (g_ours, Ai_ours)


def test_backward(B=2, T=256, H=4, K=64, V=64, seed=42):
    print(f"\n=== Backward Test (B={B}, T={T}, H={H}, K={K}, V={V}) ===")
    q, k, v, g_raw, beta, h0, scale = make_inputs(B, T, H, K, V, seed)

    # Run forward to get intermediates
    g_ref, o_ref, A_ref, ht_ref = fla_forward(q, k, v, g_raw, beta, scale, h0)
    g_ours, o_ours, Ai_ours, w_ours, u_ours, h_ours, vn_ours, ht_ours = \
        our_forward(q, k, v, g_raw, beta, scale, h0)

    do = torch.randn_like(o_ref)
    dht = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda")

    # FLA backward
    dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref = fla_backward(
        q, k, v, g_ref, beta, A_ref, scale, h0, do, dht,
    )

    # Our backward (use our g_ours and Ai_ours for consistency)
    dq_ours, dk_ours, dv_ours, db_ours, dg_ours, dh0_ours = our_backward(
        q, k, v, g_ours, beta, Ai_ours, scale, h0, do, dht,
    )

    all_pass = True
    all_pass &= check("dq", dq_ours, dq_ref, atol=5e-2, rtol=5e-2)
    all_pass &= check("dk", dk_ours, dk_ref, atol=5e-2, rtol=5e-2)
    all_pass &= check("dv", dv_ours, dv_ref, atol=5e-2, rtol=5e-2)
    all_pass &= check("db", db_ours, db_ref, atol=5e-2, rtol=5e-2)
    all_pass &= check("dg", dg_ours, dg_ref, atol=5e-2, rtol=5e-2)
    all_pass &= check("dh0", dh0_ours, dh0_ref, atol=5e-2, rtol=5e-2)

    return all_pass


def main():
    print("GPU:", torch.cuda.get_device_name())
    print("SM:", torch.cuda.get_device_capability())

    configs = [
        (2, 256, 4, 64, 64),    # small
        (2, 512, 4, 128, 128),   # medium
        (1, 1024, 2, 64, 64),    # longer seq
    ]

    fwd_ok = True
    bwd_ok = True
    for B, T, H, K, V in configs:
        ok, _ = test_forward(B, T, H, K, V)
        fwd_ok &= ok
        ok = test_backward(B, T, H, K, V)
        bwd_ok &= ok

    print("\n" + "=" * 50)
    print(f"Forward:  {'ALL PASS' if fwd_ok else 'SOME FAILED'}")
    print(f"Backward: {'ALL PASS' if bwd_ok else 'SOME FAILED'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
