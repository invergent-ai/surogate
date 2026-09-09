"""AOT compilation of vendored FLA KDA kernels for the native training runtime.

Only device code is imported from ``fla_kda``. Python FLA, its autograd wrappers,
and its runtime autotuner are not dependencies. Launch tilings are recorded in
the manifests and consumed by the C++ pipeline.
"""

from pathlib import Path

import triton
import triton.language as tl

from surogate.kernels.compiler import compile_triton_kernel

from .fla_kda import backward, cumsum, intra, norm, output, recurrent, state, wy


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
    """Compile BF16 KDA with FP32 intermediates and 64-token chunks.

    Normalized q/k, triangular solves, WY values and state stay in FP32 so
    chunked training agrees with recurrent decoding near hard routing ties.
    TF32x3 dot products retain tensor-core execution for these intermediates.
    """
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

    def add(name, fn, fp32="", integers="cu_seqlens chunk_indices chunk_offsets", *, tiles=None, warps=4, stages=2):
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
            num_stages=stages,
            sm=sm,
            # FP32 intra-chunk products enter differences of nearly equal
            # terms in dg. TF32 rounding there can dominate small gate grads.
            dot_input_precision="tf32x3",
        )

    add("kda_metadata", prepare_metadata, integers="cu_in cu offsets", tiles={"BLOCK": 256})
    add("kda_indices", prepare_indices, integers="offsets indices", tiles={"BLOCK": 128})
    add("kda_norm_fwd_fp32", norm.l2norm_fwd_kernel, "y rstd", tiles={"BT": 32})
    add("kda_norm_bwd", norm.l2norm_bwd_kernel, "y rstd dy dx", tiles={"BT": 32})
    add("kda_cumsum_fwd", cumsum.chunk_local_cumsum_vector_kernel, "s o", tiles={"REVERSE": False, "HAS_SCALE": True})
    add("kda_cumsum_rev", cumsum.chunk_local_cumsum_vector_kernel, "s o", tiles={"REVERSE": True, "HAS_SCALE": False})
    add("kda_intra_fwd", intra.chunk_kda_fwd_kernel_intra_sub_chunk, "q k g Aqk Akk", tiles={"BK": max(16, D)})
    add("kda_solve", intra.chunk_kda_fwd_kernel_inter_solve_fused, "q k g Aqk Akk Akkd")
    add("kda_wy_fwd", wy.recompute_w_u_fwd_kda_kernel, "q k qg kg w u A gk", tiles={"BK": 64, "BV": 64})
    # Upstream restricts this kernel to two warps on Blackwell for correctness.
    add("kda_state_fwd", state.chunk_gated_delta_rule_fwd_kernel_h_blockdim64, "k v w v_new g gk h h0 ht", warps=2)
    add("kda_output", output.chunk_gla_fwd_kernel_o, "q v g h A")
    add("kda_dav", backward.chunk_kda_bwd_kernel_dAv, "q k v A dv dA")
    # At D=128 the two-warp FP32 lowering exceeds the 99-KiB shared-memory
    # budget on consumer GPUs. Four warps with one stage fit without reducing
    # precision. The upstream two-warp restriction applies to forward only.
    add(
        "kda_state_bwd",
        state.chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
        "q k w g gk dh dv dv2 dht dh0",
        warps=4 if D == 128 else 2,
        stages=1 if D == 128 else 2,
    )
    add(
        "kda_wy_bwd",
        backward.chunk_kda_bwd_kernel_wy_dqkg_fused,
        "q k v_new g A h dh dq dk dv dg db dA dv2",
        tiles={"BV": 64},
    )
    add("kda_intra_bwd", intra.chunk_kda_bwd_kernel_intra, "q k g dAqk dAkk dq dq2 dk dk2 dg dg2 db")
    add("kda_beta_bwd", reduce_beta, "partials base out", integers="", tiles={"NK": triton.cdiv(D, 32), "BLOCK": 256})
    # Persistent inference carries one FP32 [H,K,V] state per sequence. It
    # accepts a complete prefill chunk or one subsequent token, in-place.
    for mode in ("init", "continue", "train"):
        fn = recurrent.fused_recurrent_kda_fwd_kernel
        options = common | dict(
            BK=D,
            BV=32,
            IS_VARLEN=mode == "train",
            USE_INITIAL_STATE=mode == "continue",
            STORE_FINAL_STATE=mode != "train",
            INPLACE_FINAL_STATE=True,
            IS_BETA_HEADWISE=False,
            USE_QK_L2NORM_IN_KERNEL=True,
            IS_CONTINUOUS_BATCHING=False,
            IS_SPEC_DECODING=False,
            HAS_DT_BIAS=False,
            USE_GATE_IN_KERNEL=False,
            USE_LOWER_BOUND=False,
            APPLY_BETA_SIGMOID=False,
            ALLOW_NEG_EIGVAL=False,
            stride_init_state_token=H * D * D,
            stride_final_state_token=H * D * D,
            stride_indices_seq=1,
            stride_indices_tok=1,
            scale=D**-0.5,
            num_stages=1,
        )
        constants = {key: options[key] for i, key in enumerate(fn.arg_names) if i in fn.constexprs}
        signature = {
            key: (
                "i64"
                if key in ("N", "T")
                else "fp32"
                if key == "lower_bound"
                else "*i32"
                if key in ("cu_seqlens", "ssm_state_indices", "num_accepted_tokens")
                else "*fp32"
                if key in ("g", "h0", "ht", "A_log", "dt_bias")
                else "*bf16"
            )
            for key in fn.arg_names
            if key not in constants
        }
        name = "kda_recurrent" if mode == "continue" else f"kda_recurrent_{mode}"
        manifests[name] = compile_triton_kernel(
            fn, signature, constants, output_dir, name, num_warps=4, num_stages=1, sm=sm
        )
    return manifests
