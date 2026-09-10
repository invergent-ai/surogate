"""The native page-table kernel matches dense attention, including local windows."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("dim", [64, 80, 256, 512])
@pytest.mark.parametrize("length,window", [(1, 0), (129, 0), (263, 32)])
def test_paged_attention_matches_dense_math(dim, length, window):
    from surogate import _surogate as ext

    torch.manual_seed(79)
    H, HK = 4, 2
    position = max(0, length - 3)
    qkv = torch.randn(1, length, H + 2 * HK, dim, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(1, length - position, H, dim, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(1, H, length - position, device="cuda", dtype=torch.float32)
    ext._decode_attention(qkv, out, lse, position, H, HK, window, torch.cuda.current_stream().cuda_stream)
    q = qkv[0, position:, :H].float().transpose(0, 1)
    k = qkv[0, :, H : H + HK].float().repeat_interleave(H // HK, dim=1).transpose(0, 1)
    v = qkv[0, :, H + HK :].float().repeat_interleave(H // HK, dim=1).transpose(0, 1)
    scores = (q @ k.transpose(1, 2)) / dim**0.5
    rows = torch.arange(position, length, device="cuda")[:, None]
    keys = torch.arange(length, device="cuda")[None, :]
    mask = keys <= rows
    if window:
        mask &= keys > rows - window
    scores.masked_fill_(~mask, -torch.inf)
    expected = (scores.softmax(-1) @ v).transpose(0, 1).bfloat16()
    torch.testing.assert_close(out[0], expected, atol=0.004, rtol=0.005)
    torch.testing.assert_close(lse[0], scores.logsumexp(-1), atol=2e-5, rtol=0)
