"""GLM GEMMs with a fixed K reduction, independent of token batch size."""

from itertools import product

import triton
import triton.language as tl

from surogate.kernels.compiler import compile_triton_kernel


@triton.jit
def matmul(a, b, out, M, N, K, AM, AK, BN, BK, OC, alpha, beta, TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr):
    # C[M,N] is column-major, as in the native cuBLAS interface. Computing
    # its transpose makes tokens the short tile dimension for forward GEMMs.
    rows = tl.program_id(0) * TN + tl.arange(0, TN)
    cols = tl.program_id(1) * TM + tl.arange(0, TM)
    kk = tl.arange(0, TK)
    acc = tl.zeros((TN, TM), tl.float32)
    for start in range(tl.cdiv(K, TK)):
        k = start * TK + kk
        left = tl.load(b + rows[:, None] * BN + k[None, :] * BK, (rows[:, None] < N) & (k[None, :] < K), 0)
        right = tl.load(a + k[:, None] * AK + cols[None, :] * AM, (k[:, None] < K) & (cols[None, :] < M), 0)
        if left.dtype != right.dtype:
            left = left.to(tl.float32)
            right = right.to(tl.float32)
        acc = tl.dot(left, right, acc, input_precision="tf32x3")
    ptr = out + rows[:, None] * OC + cols[None, :]
    mask = (rows[:, None] < N) & (cols[None, :] < M)
    value = acc * alpha
    if beta != 0:
        value += beta * tl.load(ptr, mask, 0).to(tl.float32)
    tl.store(ptr, value, mask)


@triton.jit
def grouped_matmul(
    x,
    weights,
    out,
    offsets,
    M,
    K,
    E,
    alpha,
    beta,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
    NE: tl.constexpr,
):
    # Bound the grid by ceil(total_tokens/TN) + E, independent of routing.
    # Prefix sums map each tile to its expert without a host read or scratch.
    e = tl.arange(0, NE)
    begin = tl.load(offsets + e, e < E, 0)
    end = tl.load(offsets + e + 1, e < E, 0)
    counts = tl.cdiv(end - begin, TN)
    ends = tl.cumsum(counts)
    tile = tl.program_id(0)
    expert = tl.sum(((tile >= ends) & (e < E)).to(tl.int32))
    if expert >= E:
        return
    first_tile = tl.sum(tl.where(e == expert, ends - counts, 0))
    bos = tl.load(offsets + expert)
    eos = tl.load(offsets + expert + 1)
    rows = bos + (tile - first_tile) * TN + tl.arange(0, TN)
    cols = tl.program_id(1) * TM + tl.arange(0, TM)
    kk = tl.arange(0, TK)
    acc = tl.zeros((TN, TM), tl.float32)
    for start in range(tl.cdiv(K, TK)):
        k = start * TK + kk
        left = tl.load(x + rows[:, None] * K + k[None, :], (rows[:, None] < eos) & (k[None, :] < K), 0)
        right = tl.load(
            weights + expert * M * K + cols[None, :] * K + k[:, None], (cols[None, :] < M) & (k[:, None] < K), 0
        )
        acc = tl.dot(left, right, acc, input_precision="tf32x3")
    ptr = out + rows[:, None] * M + cols[None, :]
    mask = (rows[:, None] < eos) & (cols[None, :] < M)
    value = acc * alpha
    if beta != 0:
        value += beta * tl.load(ptr, mask, 0).to(tl.float32)
    tl.store(ptr, value, mask)


def compile_glm_matmul(experts, output_dir, sm):
    manifests = {}
    tiles = dict(TM=64, TN=16, TK=32)
    for a, b, c in product(("bf16", "fp32"), repeat=3):
        name = f"glm_matmul_{a}_{b}_{c}"
        signature = dict(a=f"*{a}", b=f"*{b}", out=f"*{c}")
        signature.update({p: "i32" for p in ("M", "N", "K", "AM", "AK", "BN", "BK", "OC")})
        signature.update(alpha="fp32", beta="fp32")
        manifests[name] = compile_triton_kernel(matmul, signature, tiles, output_dir, name, sm=sm)
    for dtype in ("bf16", "fp32"):
        name = f"glm_grouped_matmul_{dtype}"
        signature = dict(
            x=f"*{dtype}",
            weights=f"*{dtype}",
            out=f"*{dtype}",
            offsets="*i32",
            M="i32",
            K="i32",
            E="i32",
            alpha="fp32",
            beta="fp32",
        )
        manifests[name] = compile_triton_kernel(
            grouped_matmul,
            signature,
            tiles | dict(NE=triton.next_power_of_2(max(experts, 1))),
            output_dir,
            name,
            sm=sm,
        )
    return manifests
