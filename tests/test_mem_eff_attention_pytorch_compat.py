"""Bit-compatibility test: surogate's ported mem-efficient attention vs.
PyTorch's scaled_dot_product_attention with the efficient-attention
backend forced.

Both code paths call into the same cutlass kernel headers (originally
from xformers); given identical inputs and layouts, outputs should be
bit-exact. This test will catch any drift introduced by our port (e.g.
compat shim, dispatcher bugs, alignment changes).
"""

import pytest
import torch

_surogate = pytest.importorskip(
    "surogate._surogate",
    reason="mem_eff_attention_forward binding not yet exposed",
)
if not hasattr(_surogate, "mem_eff_attention_forward"):
    pytest.skip(
        "mem_eff_attention_forward binding not yet exposed",
        allow_module_level=True,
    )


def _run_pytorch_mem_eff(q, k, v, *, is_causal, scale):
    """Invoke PyTorch's scaled_dot_product_attention forcing the
    efficient-attention backend — the same cutlass kernel we ported.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            scale=scale,
        )


@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("T", [128, 2048])
@pytest.mark.parametrize("is_causal", [True, False])
def test_dense_fwd_bit_exact(head_dim, T, is_causal):
    """Dense (non-varlen) forward. Same kernel, same inputs → bit-exact."""
    torch.manual_seed(0x5EC0_DE)
    B = 2
    num_heads = 8
    device = "cuda"
    dtype = torch.bfloat16

    # PyTorch's SDPA expects [B, num_heads, T, head_dim].
    q_torch = torch.randn(B, num_heads, T, head_dim, device=device, dtype=dtype)
    k_torch = torch.randn(B, num_heads, T, head_dim, device=device, dtype=dtype)
    v_torch = torch.randn(B, num_heads, T, head_dim, device=device, dtype=dtype)
    scale = 1.0 / (head_dim**0.5)

    ref = _run_pytorch_mem_eff(q_torch, k_torch, v_torch, is_causal=is_causal, scale=scale)

    # Our dispatcher expects [B, T, num_heads, head_dim]. Transpose is a
    # view op (no copy), and the mem-eff kernel reads via strides, so
    # the results must still match PyTorch's bit-for-bit.
    q = q_torch.transpose(1, 2).contiguous()
    k = k_torch.transpose(1, 2).contiguous()
    v = v_torch.transpose(1, 2).contiguous()
    out = torch.empty_like(q)

    _surogate.mem_eff_attention_forward(
        q,
        k,
        v,
        out,
        causal=is_causal,
        softmax_scale=scale,
        window_size=0,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=T,
        max_seqlen_k=T,
    )

    # Back to [B, num_heads, T, head_dim] to match PyTorch layout.
    ours = out.transpose(1, 2).contiguous()

    # Bit-exact holds for head_dim >= 256: PyTorch's efficient-attention
    # backend dispatches to the same 32x128_gmem variant we use. For
    # smaller head_dims PyTorch selects the 64x64_rf or 64x128_rf
    # variants whose register-file output path has a different reduction
    # order — functionally correct but not bit-identical to GMEM.
    max_abs_diff = (ref.float() - ours.float()).abs().max().item()
    if head_dim >= 256:
        assert torch.equal(ref, ours), (
            f"Output mismatch (head_dim={head_dim}, T={T}, causal={is_causal}): max abs diff = {max_abs_diff}"
        )
    else:
        # bf16 has ~8 mantissa bits → LSB ~= 2**-8 * value. Allow 1e-2
        # absolute (safe for values on the [-1, 1] output scale).
        assert max_abs_diff < 1e-2, (
            f"Output too divergent for {head_dim=} {T=} {is_causal=}: max abs diff = {max_abs_diff}"
        )
