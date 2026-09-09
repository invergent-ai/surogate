"""GLM projections retain the same reduction order across token batch sizes."""

import pytest
import torch
import triton

from surogate.kernels.triton.glm_matmul import grouped_matmul, matmul

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]
TILES = dict(TM=64, TN=16, TK=32)


@pytest.mark.parametrize(
    "weight_dtype,input_dtype,out_dtype",
    [
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.float32, torch.bfloat16),
        (torch.float32, torch.bfloat16, torch.float32),
        (torch.float32, torch.float32, torch.float32),
    ],
)
def test_projection_is_invariant_to_batch_size_and_row_position(weight_dtype, input_dtype, out_dtype):
    torch.manual_seed(503)
    rows, features, width = 73, 83, 70
    x = torch.randn(rows, width, device="cuda", dtype=input_dtype)
    w = torch.randn(features, width, device="cuda", dtype=weight_dtype)
    original = torch.randn(rows, features + 5, device="cuda", dtype=out_dtype)
    out = original.clone()
    matmul[(triton.cdiv(rows, 16), triton.cdiv(features, 64))](
        w,
        x,
        out,
        features,
        rows,
        width,
        width,
        1,
        width,
        1,
        features + 5,
        0.7,
        0.3,
        **TILES,
    )
    expected = 0.7 * (x.double() @ w.double().T) + 0.3 * original[:, :features].double()
    torch.testing.assert_close(
        out[:, :features].double(), expected, atol=2e-5, rtol=0.004 if out_dtype == torch.bfloat16 else 2e-5
    )
    assert torch.equal(out[:, features:], original[:, features:])
    for row in (0, 1, 17, 36, 72):
        single = original[row].clone()
        matmul[(1, triton.cdiv(features, 64))](
            w,
            x[row],
            single,
            features,
            1,
            width,
            width,
            1,
            width,
            1,
            features + 5,
            0.7,
            0.3,
            **TILES,
        )
        assert torch.equal(out[row], single)


def test_grouped_projection_replays_with_empty_experts_and_changed_routing():
    torch.manual_seed(507)
    rows, features, width, experts = 73, 83, 70, 4
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(experts, features, width, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(rows, features, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, 1, 1, 18, rows], dtype=torch.int32, device="cuda")

    def run():
        grouped_matmul[(triton.cdiv(rows, 16) + experts, triton.cdiv(features, 64))](
            x,
            w,
            out,
            offsets,
            features,
            width,
            experts,
            1.0,
            0.0,
            NE=4,
            **TILES,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for bounds in ([0, 1, 1, 18, rows], [0, 0, 0, rows, rows], [0, 17, 33, 34, rows]):
        offsets.copy_(torch.tensor(bounds, dtype=torch.int32))
        graph.replay()
        for e, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            if start == end:
                continue
            expected = x[start:end].double() @ w[e].double().T
            torch.testing.assert_close(out[start:end].double(), expected, atol=1e-5, rtol=0.004)
            single = torch.empty_like(out[0])
            matmul[(1, triton.cdiv(features, 64))](
                w[e],
                x[end - 1],
                single,
                features,
                1,
                width,
                width,
                1,
                width,
                1,
                features,
                1.0,
                0.0,
                **TILES,
            )
            assert torch.equal(out[end - 1], single)
