"""AOT precision must reach Triton's language frontend, not just its metadata."""

import os

import pytest

from surogate.kernels.compiler import compile_triton_kernel

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")


@triton.jit
def _dot(x, y, out, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    offsets = i[:, None] * BLOCK + i[None, :]
    a = tl.load(x + offsets)
    b = tl.load(y + offsets)
    tl.store(out + offsets, tl.dot(a, b))


def test_explicit_fp32_precision_reaches_generated_ir_and_restores_default(tmp_path, monkeypatch):
    import triton.compiler

    compile_original = triton.compiler.compile
    ir = []

    def record(*args, **kwargs):
        result = compile_original(*args, **kwargs)
        ir.append(result.asm["ttir"])
        return result

    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    monkeypatch.setattr(triton.compiler, "compile", record)
    compile_triton_kernel(
        _dot,
        {"x": "*fp32", "y": "*fp32", "out": "*fp32"},
        {"BLOCK": 32},
        tmp_path,
        sm=80,
        dot_input_precision="tf32x3",
    )
    assert "inputPrecision = tf32x3" in ir[0]
    assert os.environ["TRITON_F32_DEFAULT"] == "ieee"
    if hasattr(triton, "knobs"):
        assert triton.knobs.language.fp32_default == "ieee"


def test_failed_compilation_restores_precision_default(tmp_path, monkeypatch):
    import triton.compiler

    def fail(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    monkeypatch.setattr(triton.compiler, "compile", fail)
    with pytest.raises(RuntimeError, match="compile failed"):
        compile_triton_kernel(
            _dot,
            {"x": "*fp32", "y": "*fp32", "out": "*fp32"},
            {"BLOCK": 32},
            tmp_path,
            sm=80,
            dot_input_precision="tf32x3",
        )
    assert os.environ["TRITON_F32_DEFAULT"] == "ieee"
