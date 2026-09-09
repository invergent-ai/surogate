"""AOT compilation of vendored FLA KDA kernels for the native training runtime.

Only device code is imported from ``fla_kda``. Python FLA, its autograd wrappers,
and its runtime autotuner are not dependencies. Launch tilings are recorded in
the manifests and consumed by the C++ pipeline.
"""

from pathlib import Path

import triton
import triton.language as tl

from surogate.kernels.compiler import compile_triton_kernel

from .fla_kda import backward, cumsum, intra, norm, output, state, wy


@triton.jit
def prepare_metadata(cu_in, cu, offsets, N, T, PACKED, BLOCK: tl.constexpr):
    # One CTA computes a prefix sum in bounded blocks, without a host readback.
    carry = 0
    for start in range(tl.cdiv(N, BLOCK)):
        i = start * BLOCK + tl.arange(0, BLOCK)
        if PACKED:
            bos = tl.load(cu_in + i, i < N, 0)
            eos = tl.load(cu_in + i + 1, i < N, 0)
        else:
            bos = i * T
            eos = bos + T
        chunks = tl.where(i < N, tl.cdiv(eos - bos, 64), 0)
        prefix = tl.cumsum(chunks)
        tl.store(cu + i, bos, i < N)
        tl.store(offsets + i, carry + prefix - chunks, i < N)
        carry += tl.sum(chunks)
    end = tl.load(cu_in + N) if PACKED else N * T
    tl.store(cu + N, end)
    tl.store(offsets + N, carry)


@triton.jit
def prepare_indices(offsets, indices, N, NT, BLOCK: tl.constexpr):
    # The last CTA fills unused capacity with sentinels. All grids stay valid
    # when graph replay changes document lengths while keeping N and T fixed.
    doc = tl.program_id(0)
    start = tl.load(offsets + doc)
    end = tl.load(offsets + doc + 1) if doc < N else NT
    for tile in range(tl.cdiv(end - start, BLOCK)):
        i = tile * BLOCK + tl.arange(0, BLOCK)
        tl.store(indices + (start + i) * 2, tl.where(doc < N, doc, -1), start + i < end)
        tl.store(indices + (start + i) * 2 + 1, i, start + i < end)


@triton.jit
def reduce_beta(partials, base, out, T, H: tl.constexpr, NK: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(base + i, i < T * H, 0)
    for k in range(NK):
        value += tl.load(partials + k * T * H + i, i < T * H, 0)
    tl.store(out + i, value, i < T * H)


def compile_kimi_delta_rule(H: int, D: int, output_dir: str | Path, sm: int) -> dict[str, str]:
    """Compile BF16 KDA, with FP32 decay and gradients and 64-token chunks."""
    if H <= 0 or D not in (8, 16, 32, 64, 128):
        raise ValueError(f"Unsupported KDA geometry: H={H}, D={D}")
    if sm < 80:
        raise ValueError("KDA BF16 Triton kernels require SM80 or newer")
    common = dict(
        B=1,
        H=H,
        HV=H,
        K=D,
        V=D,
        S=D,
        D=D,
        BT=64,
        BC=16,
        NC=4,
        BK=32,
        BV=32,
        BS=32,
        BD=triton.next_power_of_2(D),
        NB=1,
        IS_VARLEN=True,
        HEAD_FIRST=False,
        STATE_V_FIRST=False,
        USE_G=False,
        USE_GK=True,
        USE_INITIAL_STATE=False,
        STORE_FINAL_STATE=False,
        SAVE_NEW_VALUE=True,
        USE_FINAL_STATE_GRADIENT=False,
        USE_SAFE_GATE=True,
        SAFE_GATE=True,
        USE_GATHER=True,
        STORE_QG=True,
        STORE_KG=True,
    )
    manifests = {}

    def add(name, fn, fp32="", integers="cu_seqlens chunk_indices chunk_offsets", *, tiles=None, warps=4):
        constants = common | (tiles or {})
        constants = {key: constants[key] for i, key in enumerate(fn.arg_names) if i in fn.constexprs}
        signature = {}
        for arg in fn.arg_names:
            if arg in constants:
                continue
            if arg in ("T", "B", "N", "NT", "PACKED"):
                dtype = "i32"
            elif arg in ("eps", "scale"):
                dtype = "fp32"
            elif arg in integers.split():
                dtype = "*i32"
            elif arg in fp32.split():
                dtype = "*fp32"
            else:
                dtype = "*bf16"
            signature[arg] = dtype
        manifests[name] = compile_triton_kernel(
            fn,
            signature,
            constants,
            output_dir,
            name,
            num_warps=warps,
            num_stages=2,
            sm=sm,
            # FP32 intra-chunk products enter differences of nearly equal
            # terms in dg. TF32 rounding there can dominate small gate grads.
            dot_input_precision="tf32x3",
        )

    add("kda_metadata", prepare_metadata, integers="cu_in cu offsets", tiles={"BLOCK": 256})
    add("kda_indices", prepare_indices, integers="offsets indices", tiles={"BLOCK": 128})
    add("kda_norm_fwd", norm.l2norm_fwd_kernel, "rstd", tiles={"BT": 32})
    add("kda_norm_bwd", norm.l2norm_bwd_kernel, "rstd dy dx", tiles={"BT": 32})
    add("kda_cumsum_fwd", cumsum.chunk_local_cumsum_vector_kernel, "s o", tiles={"REVERSE": False, "HAS_SCALE": True})
    add("kda_cumsum_rev", cumsum.chunk_local_cumsum_vector_kernel, "s o", tiles={"REVERSE": True, "HAS_SCALE": False})
    add("kda_intra_fwd", intra.chunk_kda_fwd_kernel_intra_sub_chunk, "g Akk", tiles={"BK": max(16, D)})
    add("kda_solve", intra.chunk_kda_fwd_kernel_inter_solve_fused, "g Akkd")
    add("kda_wy_fwd", wy.recompute_w_u_fwd_kda_kernel, "gk", tiles={"BK": 64, "BV": 64})
    # Upstream restricts this kernel to two warps on Blackwell for correctness.
    add("kda_state_fwd", state.chunk_gated_delta_rule_fwd_kernel_h_blockdim64, "g gk h0 ht", warps=2)
    add("kda_output", output.chunk_gla_fwd_kernel_o, "g")
    add("kda_dav", backward.chunk_kda_bwd_kernel_dAv, "dA")
    add("kda_state_bwd", state.chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64, "g gk dht dh0", warps=2)
    add("kda_wy_bwd", backward.chunk_kda_bwd_kernel_wy_dqkg_fused, "g dq dk dg db dA dv2", tiles={"BV": 64})
    add("kda_intra_bwd", intra.chunk_kda_bwd_kernel_intra, "g dAqk dAkk dq dq2 dk dk2 dg dg2 db")
    add("kda_beta_bwd", reduce_beta, "partials base out", integers="", tiles={"NK": triton.cdiv(D, 32), "BLOCK": 256})
    return manifests
