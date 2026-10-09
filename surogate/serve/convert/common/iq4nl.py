"""IQ4_NL blocks from float rows: llama.cpp's search, vectorised over blocks.

`quantize_row_iq4_nl_impl` with no importance matrix: per block of 32 the weight is x², the
first guess maps the largest |x| onto the code grid's end, and fourteen nearby inverse scales
(mirrored too, since the grid is asymmetric) are tried, keeping the one with the best weighted
least-squares fit. Each block is the fp16 scale followed by sixteen bytes, element j in the low
nibble of byte j and element j + 16 in its high nibble.
"""

from __future__ import annotations

import torch

BLOCK = 32
BLOCK_BYTES = 2 + BLOCK // 2
KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
_TRIES = 7
_GROUP_MAX_EPS = 1e-15


def _grid(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor(KVALUES, dtype=torch.float32, device=device)
    # A value exactly between two codes takes the upper one, as `best_index_int8` does.
    return values, (values[1:] + values[:-1]) * 0.5


def _codes(scaled: torch.Tensor, midpoints: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(scaled, midpoints, out_int32=True, right=True)


def quantize_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """[n, 32] float -> [n, 18] uint8 IQ4_NL blocks, on the input's device."""
    if blocks.dim() != 2 or blocks.shape[1] != BLOCK:
        raise ValueError(f"IQ4_NL quantises [n, {BLOCK}] blocks, not {tuple(blocks.shape)}")
    x = blocks.to(torch.float32)
    values, midpoints = _grid(x.device)
    weight = x * x
    first = x.abs().argmax(dim=1, keepdim=True)
    amax = x.gather(1, first).abs().squeeze(1)
    signed_max = x.gather(1, first).squeeze(1)
    empty = amax < _GROUP_MAX_EPS
    signed_max = torch.where(empty, torch.ones_like(signed_max), signed_max)

    def fit(inverse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = values[_codes(x * inverse[:, None], midpoints)]
        wq = weight * q
        return (wq * x).sum(1), (wq * q).sum(1)

    sumqx, sumq2 = fit(-float(KVALUES[0]) / signed_max)
    d = sumqx / torch.where(sumq2 > 0, sumq2, torch.ones_like(sumq2))
    best = d * sumqx
    for attempt in range(-_TRIES, _TRIES + 1):
        sumqx, sumq2 = fit((attempt + float(KVALUES[0])) / signed_max)
        better = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        candidate = sumqx / torch.where(sumq2 > 0, sumq2, torch.ones_like(sumq2))
        d = torch.where(better, candidate, d)
        best = torch.where(better, candidate * sumqx, best)
    d = torch.where(empty, torch.zeros_like(d), d)
    inverse = torch.where(d != 0, 1.0 / torch.where(d != 0, d, torch.ones_like(d)), torch.zeros_like(d))
    codes = _codes(x * inverse[:, None], midpoints).to(torch.uint8)
    packed = codes[:, : BLOCK // 2] | (codes[:, BLOCK // 2 :] << 4)
    scale = d.to(torch.float16).view(torch.uint8).reshape(-1, 2)
    return torch.cat((scale, packed), dim=1)


def quantize_rows(rows: torch.Tensor, *, chunk_blocks: int = 1 << 21) -> torch.Tensor:
    """[n, k] float -> [n, k / 32 * 18] uint8, k a whole number of blocks."""
    if rows.dim() != 2 or rows.shape[1] % BLOCK:
        raise ValueError(f"IQ4_NL rows need a width divisible by {BLOCK}, not {tuple(rows.shape)}")
    blocks = rows.reshape(-1, BLOCK)
    out = torch.empty((blocks.shape[0], BLOCK_BYTES), dtype=torch.uint8, device=rows.device)
    for begin in range(0, blocks.shape[0], chunk_blocks):
        end = min(begin + chunk_blocks, blocks.shape[0])
        out[begin:end] = quantize_blocks(blocks[begin:end])
    return out.reshape(rows.shape[0], rows.shape[1] // BLOCK * BLOCK_BYTES)


def dequantize_rows(data: torch.Tensor, k: int) -> torch.Tensor:
    """The inverse, for checks: [n, k / 32 * 18] uint8 -> [n, k] float32."""
    blocks = data.reshape(-1, BLOCK_BYTES)
    scale = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)
    packed = blocks[:, 2:]
    codes = torch.cat((packed & 15, packed >> 4), dim=1).to(torch.int64)
    values, _ = _grid(data.device)
    return (values[codes] * scale).reshape(-1, k)
