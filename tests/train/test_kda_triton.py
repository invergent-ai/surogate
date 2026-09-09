"""Exercise the native FLA launcher against an independent FP32 recurrence."""

import functools
import inspect
import sys

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@functools.lru_cache
def kernels(heads, dim):
    from surogate import _surogate as ext
    from surogate.kernels.jit_compile import _compile_kimi_delta_rule

    result = ext._KdaKernels()
    result.load(_compile_kimi_delta_rule(heads, dim))
    return result


def reference(inputs, lengths):
    from transformers.models.glm5_next.modeling_glm5_next import recurrent_kimi_delta_attention

    recurrence = inspect.unwrap(recurrent_kimi_delta_attention)
    values = [x.detach().float().requires_grad_() for x in inputs]
    flat = [x.flatten(0, 1)[None] for x in values]
    parts, start = [], 0
    for length in lengths:
        if length:
            part = recurrence(
                *(x[:, start : start + length] for x in flat),
                None,
                False,
                use_qk_l2norm_in_kernel=True,
            )[0]
            parts.append(part)
        start += length
    return torch.cat(parts, dim=1).reshape_as(values[0]), values


def make_case(batch, length, dim, gate_scale=5.0):
    torch.manual_seed(320 + dim + length)
    shape = (batch, length, 2, dim)
    q, k, v = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    g = -gate_scale * torch.rand(shape, device="cuda")
    beta = torch.rand(shape[:-1], dtype=torch.bfloat16, device="cuda")
    return [q, k, v, g, beta], torch.randn_like(q)


def check(inputs, dy, out, grads, lengths):
    expected, leaves = reference(inputs, lengths)
    expected_grads = torch.autograd.grad(expected, leaves, dy.float())
    torch.testing.assert_close(out.float(), expected, atol=0.002, rtol=0.04)
    for name, actual, expected in zip(("q", "k", "v", "g", "beta"), grads, expected_grads):
        error = (actual - expected).square().mean().sqrt()
        assert error < 0.025 * expected.square().mean().sqrt() + 2e-6, (name, error.item())


@pytest.mark.parametrize(
    "batch,length,dim,gate_scale",
    [(2, 1, 8, 5), (2, 17, 16, 5), (2, 65, 32, 5), (1, 129, 64, 0.01), (1, 257, 128, 0.01)],
)
def test_native_fla_forward_backward(batch, length, dim, gate_scale):
    inputs, dy = make_case(batch, length, dim, gate_scale)
    runner = kernels(2, dim)
    work = torch.empty(runner.workspace_bytes(batch, length, 2, dim, 0, True), dtype=torch.uint8, device="cuda")
    out = torch.empty_like(inputs[0])
    grads = [torch.empty_like(x, dtype=torch.float32) for x in inputs]
    stream = torch.cuda.current_stream().cuda_stream
    runner.run(False, inputs, [out], None, work, stream)
    runner.run(True, [dy, *inputs], grads, None, work, stream)
    check(inputs, dy, out, grads, [length] * batch)


@pytest.mark.parametrize(
    "layouts",
    [
        [[1, 65, 127], [64, 64, 65], [0, 129, 64], [128, 1, 64]],
        # More than one metadata block and more than one block of sentinels:
        # moving all tokens into one document leaves 256 unused chunk slots.
        [[1] * 256 + [65], [65] + [1] * 256, [0] * 256 + [321]],
    ],
    ids=["chunk_boundaries", "many_documents"],
)
def test_packed_cuda_graph_replay_changes_chunk_count(layouts):
    # Same N and total tokens, but different numbers of real 64-token chunks,
    # an empty document, and resets on both sides of chunk boundaries.
    length, num_docs = sum(layouts[0]), len(layouts[0])
    inputs, dy = make_case(1, length, 32, 0.01)
    runner = kernels(2, 32)
    cu = torch.tensor([0, *torch.tensor(layouts[0]).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    work = torch.empty(runner.workspace_bytes(1, length, 2, 32, num_docs, True), dtype=torch.uint8, device="cuda")
    out = torch.empty_like(inputs[0])
    grads = [torch.empty_like(x, dtype=torch.float32) for x in inputs]

    def step():
        stream = torch.cuda.current_stream().cuda_stream
        runner.run(False, inputs, [out], cu, work, stream)
        runner.run(True, [dy, *inputs], grads, cu, work, stream)

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for lengths in layouts:
        cu.copy_(torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32))
        graph.replay()
        check(inputs, dy, out, grads, lengths)


def test_compile_without_external_fla(monkeypatch, tmp_path):
    from surogate.kernels.triton.kimi_delta_rule import compile_kimi_delta_rule

    monkeypatch.setitem(sys.modules, "fla", None)
    sm = torch.cuda.get_device_capability()
    manifests = compile_kimi_delta_rule(2, 32, tmp_path, sm[0] * 10 + sm[1])
    assert len(manifests) == 16
    from surogate import _surogate as ext

    runner = ext._KdaKernels()
    runner.load(manifests)
    inputs, dy = make_case(1, 17, 32)
    work = torch.empty(runner.workspace_bytes(1, 17, 2, 32, 0, True), dtype=torch.uint8, device="cuda")
    out = torch.empty_like(inputs[0])
    grads = [torch.empty_like(x, dtype=torch.float32) for x in inputs]
    runner.run(False, inputs, [out], None, work, torch.cuda.current_stream().cuda_stream)
    runner.run(True, [dy, *inputs], grads, None, work, torch.cuda.current_stream().cuda_stream)
    check(inputs, dy, out, grads, [17])
