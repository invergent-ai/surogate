"""Compare the native CUDA reference and registered, vendored FLA KDA pipeline.

CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_glm_kda.py --length 256 --dim 128
No external flash-linear-attention package is needed. Timings include native
metadata preparation and backward recomputation, and exclude compilation.
"""

import argparse
import json

import torch


def main():
    from triton.testing import do_bench_cudagraph

    from surogate import _surogate as ext
    from surogate.kernels.jit_compile import _compile_kimi_delta_rule

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128, choices=[32, 64, 128])
    args = parser.parse_args()
    torch.manual_seed(41)
    shape = (1, args.length, args.heads, args.dim)
    q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    g = -5 * torch.rand(shape, device="cuda")
    beta = torch.rand(shape[:-1], device="cuda", dtype=torch.bfloat16)
    inputs = [q, k, v, g, beta]
    dy = torch.randn_like(q)
    out = torch.empty_like(q)
    grads = [torch.empty_like(x, dtype=torch.float32) for x in inputs]
    checkpoints = torch.empty((1, (args.length + 15) // 16, args.heads, args.dim, args.dim), device="cuda")
    runner = ext._KdaKernels()
    runner.load(_compile_kimi_delta_rule(args.heads, args.dim))
    work = torch.empty(runner.workspace_bytes(*shape, 0, True), dtype=torch.uint8, device="cuda")
    chunk_out = torch.empty_like(out)
    chunk_grads = [torch.empty_like(x) for x in grads]

    def native_forward():
        ext._glm5_kernel(3, False, inputs, [out], {}, torch.cuda.current_stream().cuda_stream)

    def native_backward():
        ext._glm5_kernel(3, True, [dy] + inputs, grads, {}, torch.cuda.current_stream().cuda_stream, checkpoints)

    def chunk_forward():
        runner.run(False, inputs, [chunk_out], None, work, torch.cuda.current_stream().cuda_stream)

    def chunk_backward():
        runner.run(True, [dy, *inputs], chunk_grads, None, work, torch.cuda.current_stream().cuda_stream)

    native_forward()
    native_backward()
    chunk_forward()
    chunk_backward()
    errors = [float((a - b).square().mean().sqrt() / b.square().mean().sqrt()) for a, b in zip(chunk_grads, grads)]
    torch.testing.assert_close(chunk_out, out, atol=0.002, rtol=0.03)
    assert max(errors) < 0.02, errors
    max_output_error = float((out - chunk_out).abs().max())

    def native_step():
        native_forward()
        native_backward()

    def chunk_step():
        chunk_forward()
        chunk_backward()

    result = dict(
        gpu=torch.cuda.get_device_name(),
        shape=shape,
        fla_revision="9c8e42e762fce087c27b673af4922795d9edb85e",
        workspace_bytes=work.numel(),
        max_output_error=max_output_error,
        gradient_relative_rms=errors,
        forward_ms={
            "cuda_reference": do_bench_cudagraph(native_forward, rep=50),
            "vendored_fla": do_bench_cudagraph(chunk_forward, rep=50),
        },
        forward_backward_ms={
            "cuda_reference": do_bench_cudagraph(native_step, rep=50),
            "vendored_fla": do_bench_cudagraph(chunk_step, rep=50),
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
