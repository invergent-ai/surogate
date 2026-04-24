"""Bit-compatibility test: surogate's ported mem-efficient attention vs.
PyTorch's scaled_dot_product_attention with the efficient-attention
backend forced.

Both code paths call into the same cutlass kernel headers (originally
from xformers); given identical inputs and layouts, outputs should be
bit-exact. This test will catch any drift introduced by our port (e.g.
compat shim, dispatcher bugs, alignment changes).

Status: SKELETON. The Python binding for
``surogate._surogate.mem_eff_attention_forward`` is not yet exposed.
Once it is, remove the pytest.importorskip guard below and the tests
will run end-to-end. See design/capture-safe-runtime-plan.md for the
integration plan.
"""

import pytest
import torch

# The binding doesn't exist yet; the test skips cleanly until it does.
_surogate = pytest.importorskip(
    "surogate._surogate",
    reason="mem_eff_attention_forward binding not yet exposed",
)
if not hasattr(_surogate, "mem_eff_attention_forward"):
    pytest.skip(
        "mem_eff_attention_forward binding not yet exposed",
        allow_module_level=True,
    )


def _run_pytorch_mem_eff(q, k, v, *, is_causal, scale, window_size):
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


def _run_surogate_mem_eff(q, k, v, *, is_causal, scale, window_size, lse=None):
    """Invoke our ported dispatcher via the (to-be-added) binding."""
    return _surogate.mem_eff_attention_forward(
        q,
        k,
        v,
        causal=is_causal,
        softmax_scale=scale,
        window_size=window_size,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
    )


@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("T", [128, 2048])
@pytest.mark.parametrize("is_causal", [True, False])
def test_dense_fwd_bit_exact(head_dim, num_heads, T, is_causal):
    """Dense (non-varlen) forward. Same kernel, same inputs → bit-exact."""
    torch.manual_seed(0x5EC0_DE)
    B = 2
    device = "cuda"
    dtype = torch.bfloat16

    # [B, num_heads, T, head_dim] — PyTorch's SDPA expected layout
    q = torch.randn(B, num_heads, T, head_dim, device=device, dtype=dtype)
    k = torch.randn(B, num_heads, T, head_dim, device=device, dtype=dtype)
    v = torch.randn(B, num_heads, T, head_dim, device=device, dtype=dtype)
    scale = 1.0 / (head_dim**0.5)

    ref = _run_pytorch_mem_eff(q, k, v, is_causal=is_causal, scale=scale, window_size=0)
    ours = _run_surogate_mem_eff(q, k, v, is_causal=is_causal, scale=scale, window_size=0)

    # Same kernel, same inputs: outputs must match bit-for-bit.
    # torch.equal compares by value but treats NaN as not equal; we
    # expect no NaNs in a clean forward.
    assert torch.equal(ref, ours), (
        f"Output mismatch (head_dim={head_dim}, T={T}, causal={is_causal}): "
        f"max abs diff = {(ref - ours).abs().max().item()}"
    )


@pytest.mark.parametrize("head_dim", [128, 256, 512])
def test_varlen_fwd_bit_exact(head_dim):
    """Packed (varlen) forward with synthetic cu_seqlens. Both paths
    drop to the same cutlass kernel with seqstart_q/k set; outputs
    should be bit-exact.

    Varlen bit-compat requires PyTorch's nested-tensor path since
    torch.nn.functional.scaled_dot_product_attention only accepts
    dense inputs. Use the nested-tensor variant where PyTorch
    internally builds cu_seqlens from NestedTensor offsets.
    """
    torch.manual_seed(0xBADC_0DE)
    device = "cuda"
    dtype = torch.bfloat16
    num_heads = 8
    doc_lens = [256, 1024, 768]  # 3 docs, uneven lengths

    # Build nested tensors ([doc_i_len, num_heads, head_dim]) for PyTorch.
    # PyTorch's nested-tensor SDPA will call its varlen kernel.
    nested_q = torch.nested.nested_tensor(
        [torch.randn(L, num_heads, head_dim, device=device, dtype=dtype) for L in doc_lens],
        device=device,
        dtype=dtype,
    )
    nested_k = torch.nested.nested_tensor(
        [torch.randn(L, num_heads, head_dim, device=device, dtype=dtype) for L in doc_lens],
        device=device,
        dtype=dtype,
    )
    nested_v = torch.nested.nested_tensor(
        [torch.randn(L, num_heads, head_dim, device=device, dtype=dtype) for L in doc_lens],
        device=device,
        dtype=dtype,
    )
    scale = 1.0 / (head_dim**0.5)

    ref_nested = _run_pytorch_mem_eff(
        nested_q.transpose(1, 2),
        nested_k.transpose(1, 2),
        nested_v.transpose(1, 2),
        is_causal=True,
        scale=scale,
        window_size=0,
    )
    ref = ref_nested.transpose(1, 2).values()  # [sum(doc_lens), num_heads, head_dim]

    # Build the equivalent packed layout for our dispatcher.
    packed_q = nested_q.values()  # [sum(doc_lens), num_heads, head_dim]
    packed_k = nested_k.values()
    packed_v = nested_v.values()
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(doc_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )

    ours = _surogate.mem_eff_attention_forward(
        packed_q,
        packed_k,
        packed_v,
        causal=True,
        softmax_scale=scale,
        window_size=0,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
    )

    assert torch.equal(ref, ours), (
        f"Varlen output mismatch (head_dim={head_dim}): max abs diff = {(ref - ours).abs().max().item()}"
    )
