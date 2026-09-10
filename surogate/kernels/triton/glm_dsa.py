"""Pooled GLM DSA selection and attention over gathered keys, without a dense mask."""

from pathlib import Path

import triton
import triton.language as tl

from surogate.kernels.compiler import compile_triton_kernel


@triton.jit
def normalize_keys(X, W, Bias, Y, T, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    x = tl.load(X + row * D + d, d < D, 0).to(tl.float32)
    mean = tl.sum(x, 0) / D
    centered = tl.where(d < D, x - mean, 0)
    rstd = tl.rsqrt(tl.sum(centered * centered, 0) / D + 1.0e-6)
    w = tl.load(W + d, d < D, 0).to(tl.float32)
    bias = tl.load(Bias + d, d < D, 0).to(tl.float32)
    tl.store(Y + row * D + d, centered * rstd * w + bias, d < D)


@triton.jit
def page_pointer(table, row, col, width: tl.constexpr, dtype: tl.constexpr, valid):
    row = row + 0 * col
    address = tl.load(table + tl.maximum(row, 0) // 128, valid, 0)
    return address.to(tl.pointer_type(dtype)) + (row % 128) * width + col


@triton.jit
def pool_keys(K, Gate, Ape, Pos, Pooled, Ends, T, CAP, NP, START, D: tl.constexpr, P: tl.constexpr, BD: tl.constexpr, PAGED: tl.constexpr):
    slot, batch = tl.program_id(0) + START, tl.program_id(1)
    p, d = tl.arange(0, P), tl.arange(0, BD)
    # Complete pools are at least P tokens apart, even across packed documents.
    # Their final tokens therefore occupy distinct P-wide slots in the row.
    candidate = slot * P + p
    pos_ptr = Pos + batch * CAP + candidate
    if PAGED:
        pos_ptr = page_pointer(Pos, candidate, 0, 1, tl.int32, candidate < T)
    position = tl.load(pos_ptr, candidate < T, -1)
    end = tl.max(tl.where((candidate < T) & (position >= 0) & ((position + 1) % P == 0), candidate, -1), 0)
    token = end - P + 1 + p
    valid = (end >= 0) & (d[None, :] < D)
    gate_ptr = Gate + ((batch * CAP + token[:, None]) * D + d[None, :])
    key_ptr = K + ((batch * CAP + token[:, None]) * D + d[None, :])
    if PAGED:
        gate_ptr = page_pointer(Gate, token[:, None], d[None, :], D, tl.bfloat16, valid)
        key_ptr = page_pointer(K, token[:, None], d[None, :], D, tl.bfloat16, valid)
    gate = tl.load(gate_ptr, valid, 0).to(tl.float32)
    ape = tl.load(Ape + p[:, None] * D + d[None, :], d[None, :] < D, 0).to(tl.float32)
    logits = gate + ape
    prob = tl.exp(logits - tl.max(logits, 0)[None, :])
    prob = (prob / tl.sum(prob, 0)[None, :]).to(tl.bfloat16)
    key = tl.load(key_ptr, valid, 0)
    # Match the checkpoint's BF16 probability/product rounding before reduction.
    value = tl.sum((prob.to(tl.float32) * key.to(tl.float32)).to(tl.bfloat16).to(tl.float32), 0)
    pooled_ptr = Pooled + (batch * NP + slot) * D + d
    ends_ptr = Ends + batch * NP + slot
    if PAGED:
        pooled_ptr = page_pointer(Pooled, slot, d, D, tl.bfloat16, d < D)
        ends_ptr = page_pointer(Ends, slot, 0, 1, tl.int32, True)
    tl.store(pooled_ptr, value, d < D)
    tl.store(ends_ptr, end)


@triton.jit
def score_pools(
    Q,
    Weights,
    Pooled,
    Ends,
    Pos,
    Scores,
    TQ,
    TK,
    NP,
    PCAP,
    OFFSET,
    QSTART,
    QCOUNT,
    H: tl.constexpr,
    D: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BP: tl.constexpr,
    PAGED: tl.constexpr,
):
    local_row, tile, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = QSTART + local_row
    h, d, p = tl.arange(0, BH), tl.arange(0, BD), tile * BP + tl.arange(0, BP)
    q = tl.load(Q + ((batch * TQ + row) * H + h[:, None]) * D + d[None, :], (h[:, None] < H) & (d[None, :] < D), 0).to(
        tl.float32
    )
    mask = (p[None, :] < NP) & (d[:, None] < D)
    pooled_ptr = Pooled + (batch * PCAP + p[None, :]) * D + d[:, None]
    ends_ptr = Ends + batch * PCAP + p
    if PAGED:
        pooled_ptr = page_pointer(Pooled, p[None, :], d[:, None], D, tl.bfloat16, mask)
        ends_ptr = page_pointer(Ends, p, 0, 1, tl.int32, p < NP)
    k = tl.load(pooled_ptr, mask, 0).to(tl.float32)
    scores = tl.maximum(tl.dot(q, k) * (D**-0.5), 0)
    weight = tl.load(Weights + (batch * TQ + row) * H + h, h < H, 0).to(tl.float32) * (H**-0.5)
    score = tl.sum(scores * weight[:, None], 0)
    end = tl.load(ends_ptr, p < NP, -1)
    position = tl.load(Pos + batch * TQ + row)
    absolute = OFFSET + row
    visible = (end >= absolute - position) & (end <= absolute) & (end >= 0)
    tl.store(Scores + (batch * QCOUNT + local_row) * NP + p, tl.where(visible, score, -float("inf")), p < NP)


@triton.jit
def select_pools(Scores, Ends, Selected, TQ, NP, PCAP, SELECT: tl.constexpr, BLOCK: tl.constexpr, PAGED: tl.constexpr):
    row, batch = tl.program_id(0), tl.program_id(1)
    p = tl.arange(0, BLOCK)
    scores = tl.load(Scores + (batch * TQ + row) * NP + p, p < NP, -float("inf"))
    # Sort an order-preserving float key together with its pool index. Keep
    # all score bits and break exact ties by the earlier pool deterministically.
    bits = scores.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    key = (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - p.to(tl.uint32)).to(tl.uint64)
    sorted_key = tl.sort(key, descending=True)
    selected = (0xFFFFFFFF - (sorted_key & 0xFFFFFFFF)).to(tl.int32)
    valid = (selected < NP) & ((sorted_key >> 32) > 0x007FFFFF)
    ends_ptr = Ends + batch * PCAP + selected
    if PAGED:
        ends_ptr = page_pointer(Ends, selected, 0, 1, tl.int32, valid)
    sorted_end = tl.load(ends_ptr, valid, -1)
    tl.store(Selected + (batch * TQ + row) * SELECT + p, sorted_end, p < SELECT)


@triton.jit
def expand_indices(
    Selected,
    Pos,
    Indices,
    TQ,
    OFFSET,
    KSEL,
    QSTART,
    QCOUNT,
    SELECT: tl.constexpr,
    P: tl.constexpr,
    TAIL: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local_row, batch = tl.program_id(0), tl.program_id(1)
    row = QSTART + local_row
    i = tl.arange(0, BLOCK)
    end = tl.load(Selected + (batch * QCOUNT + local_row) * SELECT + i // P, i < KSEL * P, -1)
    index = tl.where(end >= 0, end - P + 1 + i % P, -1)
    position = tl.load(Pos + batch * TQ + row)
    tail_count = (position + 1) % P
    tail_offset = i - KSEL * P
    tail = OFFSET + row + 1 - tail_count + tail_offset
    index = tl.where(TAIL & (tail_offset >= 0) & (tail_offset < tail_count), tail, index)
    tl.store(Indices + (batch * TQ + row) * STRIDE + i, index, i < STRIDE)


@triton.jit
def gather_latents(Latent, Indices, Out, Slots, R, S, BD: tl.constexpr):
    slot = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    index = tl.load(Indices + slot)
    value = tl.load(Latent + index * R + d, (index >= 0) & (d < R), 0)
    tl.store(Out + slot * R + d, value, d < R)
    if tl.program_id(1) == 0:
        tl.store(Slots + slot, tl.where(index >= 0, slot, -1))


@triton.jit
def repack_kv(Input, Out, T, H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, col = i // (2 * H * D), i % (2 * H * D)
    kv, head, d = col // (H * D), (col // D) % H, col % D
    value = tl.load(Input + ((token * H + head) * 2 + kv) * D + d, token < T, 0)
    tl.store(Out + i, value, token < T)


@triton.jit
def sparse_attention(
    QKV,
    KV,
    Indices,
    Out,
    LSE,
    TQ,
    TCAP,
    S,
    H: tl.constexpr,
    D: tl.constexpr,
    STRIDE: tl.constexpr,
    BD: tl.constexpr,
    BK: tl.constexpr,
    CACHED: tl.constexpr,
):
    query, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d, ks = tl.arange(0, BD), tl.arange(0, BK)
    q = tl.load(QKV + ((batch * TQ + query) * 3 * H + head) * D + d, d < D, 0).to(tl.float32)
    acc = tl.full((BD,), 0, tl.float32)
    maximum, total = -float("inf"), 0.0
    for start in range(tl.cdiv(S, BK)):
        slot = start * BK + ks
        index = tl.load(Indices + (batch * TQ + query) * STRIDE + slot, slot < S, -1)
        valid = index >= 0
        if tl.sum(valid.to(tl.int32), 0) > 0:
            if CACHED:
                base = KV + ((batch * TCAP + index[:, None]) * 2 * H + head) * D + d[None, :]
            else:
                base = QKV + ((batch * TQ + index[:, None]) * 3 * H + H + head) * D + d[None, :]
            k = tl.load(base, valid[:, None] & (d[None, :] < D), 0).to(tl.float32)
            v = tl.load(base + H * D, valid[:, None] & (d[None, :] < D), 0).to(tl.float32)
            score = tl.sum(q[None, :] * k, 1) * (D**-0.5)
            score = tl.where(valid, score, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(score, 0))
            correction = tl.exp(maximum - new_max)
            p = tl.exp(score - new_max)
            acc = acc * correction + tl.sum(p[:, None] * v, 0)
            total = total * correction + tl.sum(p, 0)
            maximum = new_max
    tl.store(Out + ((batch * TQ + query) * H + head) * D + d, acc / tl.where(total > 0, total, 1), d < D)
    tl.store(LSE + (batch * TQ + query) * H + head, tl.where(total > 0, maximum + tl.log(total), -float("inf")))


@triton.jit
def sparse_attention_backward(
    DO,
    QKV,
    Indices,
    Out,
    LSE,
    DQKV,
    T,
    S,
    H: tl.constexpr,
    D: tl.constexpr,
    STRIDE: tl.constexpr,
    BD: tl.constexpr,
    BK: tl.constexpr,
):
    query, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d, ks = tl.arange(0, BD), tl.arange(0, BK)
    offset = ((batch * T + query) * H + head) * D + d
    qoffset = ((batch * T + query) * 3 * H + head) * D + d
    q = tl.load(QKV + qoffset, d < D, 0).to(tl.float32)
    out = tl.load(Out + offset, d < D, 0).to(tl.float32)
    dout = tl.load(DO + offset, d < D, 0).to(tl.float32)
    delta = tl.sum(out * dout, 0)
    lse = tl.load(LSE + (batch * T + query) * H + head)
    dq = tl.full((BD,), 0, tl.float32)
    for start in range(tl.cdiv(S, BK)):
        slot = start * BK + ks
        index = tl.load(Indices + (batch * T + query) * STRIDE + slot, slot < S, -1)
        valid = index >= 0
        if tl.sum(valid.to(tl.int32), 0) > 0:
            koff = ((batch * T + index[:, None]) * 3 * H + H + head) * D + d[None, :]
            mask = valid[:, None] & (d[None, :] < D)
            k = tl.load(QKV + koff, mask, 0).to(tl.float32)
            v = tl.load(QKV + koff + H * D, mask, 0).to(tl.float32)
            score = tl.sum(q[None, :] * k, 1) * (D**-0.5)
            p = tl.where(valid, tl.exp(score - lse), 0)
            dp = tl.sum(dout[None, :] * v, 1)
            ds = p * (dp - delta) * (D**-0.5)
            dq += tl.sum(ds[:, None] * k, 0)
            tl.atomic_add(DQKV + koff, ds[:, None] * q[None, :], mask, sem="relaxed")
            tl.atomic_add(DQKV + koff + H * D, p[:, None] * dout[None, :], mask, sem="relaxed")
    tl.store(DQKV + qoffset, dq, d < D)


def compile_glm_dsa(
    H: int, D: int, IH: int, ID: int, P: int, topk: int, tail: bool, max_seq: int, output_dir: str | Path, sm: int
) -> dict[str, str]:
    if min(H, D, IH, ID, P, topk, max_seq) <= 0 or P & (P - 1) or topk % P:
        raise ValueError("Invalid GLM DSA geometry")
    stride = topk + (P - 1 if tail else 0)
    common = dict(
        H=IH,
        D=ID,
        BH=max(16, triton.next_power_of_2(IH)),
        BD=max(16, triton.next_power_of_2(ID)),
        P=P,
        BP=32,
        SELECT=topk // P,
        STRIDE=stride,
        TAIL=tail,
        PAGED=False,
    )
    manifests = {}

    def add(name, fn, constants, fp32=(), integers=(), warps=4, tables=()):
        constants = {key: constants[key] for i, key in enumerate(fn.arg_names) if i in fn.constexprs}
        signature = {
            key: (
                "i32"
                if key
                in (
                    "T",
                    "TQ",
                    "TK",
                    "CAP",
                    "NP",
                    "PCAP",
                    "START",
                    "OFFSET",
                    "KSEL",
                    "TCAP",
                    "S",
                    "QSTART",
                    "QCOUNT",
                    "R",
                )
                else "*i64"
                if key in tables
                else "*i32"
                if key in integers
                else "*fp32"
                if key in fp32
                else "*bf16"
            )
            for key in fn.arg_names
            if key not in constants
        }
        manifests[name] = compile_triton_kernel(
            fn, signature, constants, output_dir, name, sm=sm, num_warps=warps, dot_input_precision="tf32x3"
        )

    add("dsa_norm", normalize_keys, common)
    add("dsa_pool", pool_keys, common, integers=("Pos", "Ends"))
    add("dsa_score", score_pools, common, fp32=("Scores",), integers=("Ends", "Pos"))
    add(
        "dsa_select",
        select_pools,
        common | dict(BLOCK=triton.next_power_of_2(max(topk // P, triton.cdiv(max_seq, P)))),
        fp32=("Scores",),
        integers=("Ends", "Selected"),
        warps=8,
    )
    add(
        "dsa_indices",
        expand_indices,
        common | dict(BLOCK=triton.next_power_of_2(stride)),
        integers=("Selected", "Pos", "Indices"),
    )
    attention = common | dict(H=H, D=D, BD=triton.next_power_of_2(D), BK=32)
    add("dsa_attn_fwd", sparse_attention, attention | dict(CACHED=False), fp32=("LSE",), integers=("Indices",))
    add("dsa_attn_decode", sparse_attention, attention | dict(CACHED=True), fp32=("LSE",), integers=("Indices",))
    add("dsa_repack_kv", repack_kv, dict(H=H, D=D, BLOCK=256))
    add("dsa_gather_latents", gather_latents, dict(BD=256), integers=("Indices", "Slots"))
    add("dsa_attn_bwd", sparse_attention_backward, attention, fp32=("DO", "LSE", "DQKV"), integers=("Indices",))
    add("dsa_pool_paged", pool_keys, common | dict(PAGED=True), tables=("K", "Gate", "Pos", "Pooled", "Ends"))
    add("dsa_score_paged", score_pools, common | dict(PAGED=True), fp32=("Scores",), integers=("Pos",), tables=("Pooled", "Ends"))
    add("dsa_select_paged", select_pools,
        common | dict(PAGED=True, BLOCK=triton.next_power_of_2(max(topk // P, triton.cdiv(max_seq, P)))),
        fp32=("Scores",), integers=("Selected",), tables=("Ends",), warps=8)
    return manifests
