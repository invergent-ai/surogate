"""The vectorised IQ4_NL quantiser makes llama.cpp's choices block for block."""

import numpy as np
import torch

from surogate.serve.convert.common import iq4nl


def _best_index(values, x):
    if x <= values[0]:
        return 0
    if x >= values[-1]:
        return len(values) - 1
    lo, hi = 0, len(values) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x < values[mid]:
            hi = mid
        else:
            lo = mid
    return hi - 1 if x - values[hi - 1] < values[hi] - x else hi


def _reference_block(xb):
    """`quantize_row_iq4_nl_impl` for one block, no importance matrix, ntry = 7 (float32)."""
    values = iq4nl.KVALUES
    xb = [np.float32(v) for v in xb]
    weight = [v * v for v in xb]
    amax, mx = np.float32(0), np.float32(0)
    for v in xb:
        if abs(v) > amax:
            amax, mx = abs(v), v
    if amax < 1e-15:
        return np.float32(0), [_best_index(values, np.float32(0)) for _ in xb]
    d = np.float32(-mx / np.float32(values[0]))
    inverse = np.float32(1) / d
    sumqx = sumq2 = np.float32(0)
    for v, w in zip(xb, weight):
        q = np.float32(values[_best_index(values, inverse * v)])
        sumqx += w * q * v
        sumq2 += w * q * q
    d = sumqx / sumq2
    best = d * sumqx
    for attempt in range(-7, 8):
        inverse = np.float32(attempt + values[0]) / mx
        sumqx = sumq2 = np.float32(0)
        for v, w in zip(xb, weight):
            q = np.float32(values[_best_index(values, inverse * v)])
            sumqx += w * q * v
            sumq2 += w * q * q
        if sumq2 > 0 and sumqx * sumqx > best * sumq2:
            d = sumqx / sumq2
            best = d * sumqx
    inverse = np.float32(1) / d if d else np.float32(0)
    return d, [_best_index(values, inverse * v) for v in xb]


def test_matches_llama_cpp_search():
    generator = torch.Generator().manual_seed(7)
    blocks = torch.randn(64, 32, generator=generator) * torch.rand(64, 1, generator=generator)
    blocks[3] = 0.0
    blocks[5, 9] = 40.0  # one outlier dominating its block
    out = iq4nl.quantize_blocks(blocks)
    assert out.shape == (64, 18) and out.dtype == torch.uint8
    agree = 0
    for i in range(blocks.shape[0]):
        d, codes = _reference_block(blocks[i].tolist())
        scale = out[i, :2].contiguous().view(torch.float16).item()
        packed = out[i, 2:]
        ours = (packed & 15).tolist() + (packed >> 4).tolist()
        # Summation order differs (float32 sums), so a near-tie can flip one choice; allow it
        # in at most a couple of blocks, never in the scale's magnitude.
        assert abs(scale - float(np.float16(d))) <= 2e-3 * max(abs(float(d)), 1e-6)
        agree += ours == codes
    assert agree >= 62


def test_round_trip_error_is_small():
    generator = torch.Generator().manual_seed(1)
    rows = torch.randn(16, 160, generator=generator)
    data = iq4nl.quantize_rows(rows, chunk_blocks=7)
    assert data.shape == (16, 5 * 18)
    back = iq4nl.dequantize_rows(data, 160)
    rel = (back - rows).norm() / rows.norm()
    assert rel < 0.12
    cos = torch.nn.functional.cosine_similarity(back, rows, dim=1)
    assert float(cos.min()) > 0.99
