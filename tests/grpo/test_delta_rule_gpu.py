"""Packed recurrent forward and gradients against an independent autograd reference."""

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("key_width,value_width", [(16, 32), (64, 16), (128, 128)])
def test_rollout_delta_rule_matches_autograd_across_document_boundaries(key_width, value_width):
    from surogate import _surogate as ext

    torch.manual_seed(15)
    batch, length, heads = 2, 67, 2
    q = torch.randn(batch, length, heads, key_width, device="cuda", dtype=torch.bfloat16) * 0.15
    k = torch.randn_like(q) * 0.15
    v = torch.randn(batch, length, heads, value_width, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(batch, length, heads, device="cuda")
    beta = torch.rand(batch, length, heads, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(v) * 0.01
    inputs = [q, k, v, g, beta]
    result = [torch.empty_like(v), *[torch.empty_like(x) for x in inputs]]
    boundaries = [0, 1, 35, 67, 104, 134]
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    ext._rollout_delta_rule(inputs + [upstream], result, cu, torch.cuda.current_stream().cuda_stream)

    # Double precision on the CPU avoids sharing kernels or accumulation order
    # with the CUDA implementation. Each document starts with an empty state.
    reference_inputs = [x.detach().cpu().double().requires_grad_(True) for x in inputs]
    qr, kr, vr, gr, br = [x.reshape(-1, *x.shape[2:]) for x in reference_inputs]
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        state = torch.zeros(heads, key_width, value_width, dtype=torch.float64)
        for i in range(start, end):
            state = state * gr[i].exp()[:, None, None]
            residual = (vr[i] - (state * kr[i, :, :, None]).sum(1)) * br[i, :, None]
            state = state + kr[i, :, :, None] * residual[:, None, :]
            outputs.append((state * qr[i, :, :, None]).sum(1) / np.sqrt(key_width))
    reference = torch.stack(outputs).reshape_as(v)
    (reference * upstream.cpu().double()).sum().backward()
    torch.testing.assert_close(result[0].float().cpu(), reference.detach().float(), atol=0.001, rtol=0.008)
    for actual, source in zip(result[1:], reference_inputs, strict=True):
        torch.testing.assert_close(actual.float().cpu(), source.grad.float(), atol=2e-5, rtol=0.01)
