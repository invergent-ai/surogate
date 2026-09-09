"""Native sparse selection, gathered attention and persistent index/KV history."""

import functools

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@functools.lru_cache
def runner(dim, ih, idim, pool, topk, tail=True):
    from surogate import _surogate as ext
    from surogate.kernels.jit_compile import _compile_glm_dsa

    result = ext._DsaKernels()
    result.load(
        _compile_glm_dsa(
            dict(
                num_attention_heads=2,
                qk_nope_head_dim=dim,
                index_n_heads=ih,
                index_head_dim=idim,
                index_kpool=pool,
                index_topk=topk,
                index_kpool_always_select_tail=tail,
                max_seq=128,
            )
        )
    )
    return result


def inputs(batch, length, dim, ih, idim, pool):
    torch.manual_seed(57 + dim + pool)

    def rand(*shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    # Positive head weights keep random score ties away from the top-k boundary.
    weights = torch.rand((batch, length, ih), dtype=torch.bfloat16, device="cuda") + 0.5
    x = [
        rand(batch, length, ih, idim),
        rand(batch, length, idim),
        rand(batch, length, idim),
        weights,
        rand(idim),
        rand(idim),
        rand(pool, idim),
        torch.arange(length, dtype=torch.int32, device="cuda").repeat(batch, 1),
    ]
    return x, rand(batch, length, 6, dim)


def reference_mask(x, lengths, topk, tail):
    q, raw_k, gate, weights, nw, nb, ape, _ = x
    B, T, IH, D = q.shape
    P = ape.shape[0]
    key = F.layer_norm(raw_k, (D,), nw, nb, 1e-6)
    mask = torch.zeros((B, T, T), dtype=torch.bool, device="cuda")
    for b, docs in enumerate(lengths):
        begin = 0
        for length in docs:
            count = length // P
            if count:
                k = key[b, begin : begin + count * P].view(count, P, D)
                g = gate[b, begin : begin + count * P].view(count, P, D)
                prob = (g.float() + ape.float()).softmax(1).to(k.dtype)
                pooled = (prob * k).sum(1)
                scores = torch.einsum("qhd,pd->qhp", q[b, begin : begin + length].float(), pooled.float())
                scores = (scores * D**-0.5).relu()
                scores = (scores * (weights[b, begin : begin + length].float() * IH**-0.5)[..., None]).sum(1)
            for t in range(length):
                visible = (t + 1) // P
                if visible:
                    chosen = scores[t, :visible].topk(min(topk // P, visible)).indices
                    indices = begin + chosen[:, None] * P + torch.arange(P, device="cuda")
                    mask[b, begin + t, indices.flatten()] = True
                if tail:
                    mask[b, begin + t, begin + visible * P : begin + t + 1] = True
            begin += length
    return mask


def run_indexer(kernel, x, topk, tail=True, cache=None):
    B, T = x[0].shape[:2]
    pool = x[6].shape[0]
    indices = torch.empty((B, T, topk + (pool - 1 if tail else 0)), device="cuda", dtype=torch.int32)
    work = torch.empty(kernel.workspace_bytes(B, T, cache.length if cache else 0), dtype=torch.uint8, device="cuda")
    kernel.run(0, x, [indices], work, torch.cuda.current_stream().cuda_stream, cache)
    return indices, work


def run_attention(kernel, qkv, indices, work, cache=None):
    B, T, _, D = qkv.shape
    out = torch.empty((B, T, 2 * D), device="cuda", dtype=qkv.dtype)
    lse = torch.empty((B, T, 2), device="cuda")
    kernel.run(1, [qkv, indices], [out, lse], work, torch.cuda.current_stream().cuda_stream, cache)
    return out, lse


@pytest.mark.parametrize(
    "dim,ih,idim,pool,topk,tail",
    [(64, 4, 32, 1, 8, True), (64, 4, 32, 4, 8, True), (64, 4, 32, 4, 8, False), (256, 32, 128, 16, 32, True)],
)
def test_sparse_selection_and_attention_gradients(dim, ih, idim, pool, topk, tail):
    kernel = runner(dim, ih, idim, pool, topk, tail)
    x, qkv = inputs(2, 65, dim, ih, idim, pool)
    lengths = [[65], [3, 33, 29]]
    x[-1][1] = torch.cat([torch.arange(n, device="cuda") for n in lengths[1]])
    selected, work = run_indexer(kernel, x, topk, tail)
    mask = reference_mask(x, lengths, topk, tail)
    counts = torch.zeros_like(mask, dtype=torch.int32)
    counts.scatter_add_(2, selected.clamp_min(0).long(), (selected >= 0).int())
    torch.testing.assert_close(counts.bool(), mask)
    assert not counts.gt(1).any()
    assert not mask[1, 36:, :36].any()
    assert mask.sum(-1).max() <= topk + (pool - 1 if tail else 0)
    output, lse = run_attention(kernel, qkv, selected, work)
    leaf = qkv.float().requires_grad_()
    q, k, v = (part.transpose(1, 2) for part in leaf.split(2, dim=2))
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None]).transpose(1, 2).flatten(2)
    torch.testing.assert_close(output.float(), expected, atol=0.005, rtol=0.03)
    dy = torch.randn_like(output, dtype=torch.float32)
    grad = torch.empty_like(qkv, dtype=torch.float32)
    kernel.run(2, [dy, qkv, selected, output, lse], [grad], work, torch.cuda.current_stream().cuda_stream)
    target = torch.autograd.grad(expected, leaf, dy)[0]
    error = (grad - target).square().mean().sqrt()
    assert error < 0.01 * target.square().mean().sqrt() + 1e-5


def test_cached_indexer_and_attention_match_full_prefix():
    from surogate import _surogate as ext

    kernel = runner(64, 4, 32, 4, 8)
    x, qkv = inputs(1, 73, 64, 4, 32, 4)
    selected, work = run_indexer(kernel, x, 8)
    expected, _ = run_attention(kernel, qkv, selected, work)
    cache = ext._GlmDecodeState()
    cache.capacity = 128
    start, allocated = 0, None
    for length in (17, 1, 3, 1, 43, 8):
        chunk = [v[:, start : start + length].contiguous() if i in (0, 1, 2, 3, 7) else v for i, v in enumerate(x)]
        ids, scratch = run_indexer(kernel, chunk, 8, cache=cache)
        out, _ = run_attention(kernel, qkv[:, start : start + length].contiguous(), ids, scratch, cache)
        torch.testing.assert_close(out, expected[:, start : start + length], atol=0.005, rtol=0.03)
        allocated = cache.bytes if allocated is None else allocated
        assert cache.bytes == allocated
        start += length
        cache.length = start
    # Reusing allocations for a new prompt must not expose the previous prompt.
    cache.length = 0
    chunk = [v[:, :3].contiguous() if i in (0, 1, 2, 3, 7) else v for i, v in enumerate(x)]
    ids, scratch = run_indexer(kernel, chunk, 8, cache=cache)
    out, _ = run_attention(kernel, qkv[:, :3].contiguous(), ids, scratch, cache)
    torch.testing.assert_close(out, expected[:, :3], atol=0.005, rtol=0.03)
